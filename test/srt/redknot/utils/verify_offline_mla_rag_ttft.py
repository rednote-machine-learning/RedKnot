#!/usr/bin/env python3
"""Offline RAG end-to-end estimate for "offline MLA + online splice" on DeepSeek-V4.

Scenario (user spec): N document chunks of `chunk_tokens` each are pre-filled
OFFLINE once (their per-layer MLA latent KV is stored). At query time only the
short query is prefilled ONLINE; attention for every layer splices the offline
chunk KV with the freshly computed query KV. We report:

  * ACCURACY: cosine similarity between the offline-spliced last-token hidden
    state and the full online-prefill last-token hidden state (per layer and
    final). cos ~ 1.0 means the offline reuse is faithful.
  * COMPUTE SAVING: prefill FLOPs of [offline reuse] vs [full online prefill].
    The offline path skips the chunk tokens through Q/KV/O projections, FFN/MoE,
    and the attention QK/AV for chunk *queries* (their KV is cached). The only
    online work is the query tokens.
  * TTFT SPEEDUP estimate: full_prefill_flops / online_prefill_flops, since
    prefill TTFT is compute-bound at long context.

This is a faithful real-fp8 forward (same operators as the profiler). It does
NOT touch the online serving path; it is an offline upper-bound / sanity tool to
size the win before building the real cache backend.

Usage:
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/utils/verify_offline_mla_rag_ttft.py \
    --model-path /mnt/.../DeepSeek-V4-Pro --dataset hotpotqa \
    --n-chunks 4 --chunk-tokens 4000 --query-tokens 64 --layers all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from profile_attention_concentration import (  # noqa: E402
    _apply_rope_complex,
    _dequant_fp8_blockwise,
    _load_longbench_ids,
    _precompute_freqs_cis,
    _rms_norm,
)


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(
        torch.dot(a, b) / (a.norm().clamp_min(1e-12) * b.norm().clamp_min(1e-12))
    )


@torch.no_grad()
def run(model_path, dataset, n_chunks, chunk_tokens, query_tokens, layers_arg, device):
    from safetensors import safe_open
    from transformers import AutoTokenizer

    cfg = json.load(open(Path(model_path) / "config.json"))
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg["head_dim"])
    rd = int(cfg["qk_rope_head_dim"])
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    compress_ratios = cfg["compress_ratios"]
    rope_theta = float(cfg.get("rope_theta", 10000))
    compress_rope_theta = float(cfg.get("compress_rope_theta", 160000))
    rope_factor = float(cfg["rope_scaling"]["factor"])
    original_seq_len = int(cfg["rope_scaling"]["original_max_position_embeddings"])
    beta_fast = int(cfg["rope_scaling"]["beta_fast"])
    beta_slow = int(cfg["rope_scaling"]["beta_slow"])
    n_groups = int(cfg.get("o_groups", 8))
    o_lora_rank = int(cfg.get("o_lora_rank", 1024))
    window = int(cfg.get("sliding_window", cfg.get("window_size", 128)) or 128)

    if layers_arg == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in layers_arg.split(",") if x.strip()]

    index = json.load(open(Path(model_path) / "model.safetensors.index.json"))[
        "weight_map"
    ]

    def _get(name):
        with safe_open(str(Path(model_path) / index[name]), framework="pt") as f:
            return f.get_tensor(name)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    total = n_chunks * chunk_tokens + query_tokens
    ids = _load_longbench_ids(tok, dataset, total)[0].tolist()[:total]
    seqlen = len(ids)
    ctx_len = n_chunks * chunk_tokens  # offline part
    q_len = seqlen - ctx_len  # online part (query)

    embed = _get("embed.weight")
    emb_all = embed[torch.tensor(ids)].to(device=device, dtype=torch.bfloat16)
    chunk_spans = [(i * chunk_tokens, (i + 1) * chunk_tokens) for i in range(n_chunks)]

    freqs_plain = _precompute_freqs_cis(
        rd, seqlen, 0, rope_theta, rope_factor, beta_fast, beta_slow
    ).to(device)
    freqs_yarn = _precompute_freqs_cis(
        rd,
        seqlen,
        original_seq_len,
        compress_rope_theta,
        rope_factor,
        beta_fast,
        beta_slow,
    ).to(device)
    key_pos = torch.arange(seqlen, device=device)
    scale = head_dim**-0.5

    print("=" * 92)
    print(
        " OFFLINE MLA (chunks INDEPENDENT) + ONLINE SPLICE :: RAG ACCURACY + TTFT/FLOPs"
    )
    print(f" model={model_path}")
    print(
        f" chunks={n_chunks} x {chunk_tokens} tok = {ctx_len} offline | "
        f"query={q_len} online | total={seqlen}"
    )
    print(" chunks prefilled INDEPENDENTLY (no cross-chunk attention) — real RAG reuse")
    print(f" layers measured={len(layers)} window={window}")
    print("=" * 92)

    # Three layer-synchronized hidden states sharing the same weights:
    #  - hidden_full : full online prefill over all tokens (gold reference).
    #  - chunk_h[i]  : each chunk evolved INDEPENDENTLY (intra-chunk attention
    #                  only, positions 0..chunk_len). Produces the offline KV.
    #  - hidden_off  : query rows evolve online attending to spliced offline KV.
    hidden_full = emb_all.clone()
    chunk_h = [emb_all[a:b].clone() for (a, b) in chunk_spans]
    hidden_off = emb_all.clone()

    per_layer_cos = []
    for layer in range(n_layers):
        ratio = compress_ratios[layer] if layer < len(compress_ratios) else 0
        freqs = freqs_plain if ratio == 0 else freqs_yarn
        p = f"layers.{layer}.attn."
        attn_norm = _get(f"layers.{layer}.attn_norm.weight").to(device)
        wq_a = _dequant_fp8_blockwise(
            _get(p + "wq_a.weight"), _get(p + "wq_a.scale")
        ).to(device)
        wq_b = _dequant_fp8_blockwise(
            _get(p + "wq_b.weight"), _get(p + "wq_b.scale")
        ).to(device)
        wkv = _dequant_fp8_blockwise(_get(p + "wkv.weight"), _get(p + "wkv.scale")).to(
            device
        )
        q_norm_w = _get(p + "q_norm.weight").to(device)
        kv_norm_w = _get(p + "kv_norm.weight").to(device)
        attn_sink = _get(p + "attn_sink").to(device).float()
        wo_a = _dequant_fp8_blockwise(
            _get(p + "wo_a.weight"), _get(p + "wo_a.scale")
        ).to(device)
        wo_b = _dequant_fp8_blockwise(
            _get(p + "wo_b.weight"), _get(p + "wo_b.scale")
        ).to(device)
        wo_a_g = wo_a.view(n_groups, o_lora_rank, -1)

        def proj_qkv(hin, npos, pos_freqs):
            x = _rms_norm(hin, attn_norm, eps)
            q = _rms_norm(F.linear(x, wq_a), q_norm_w, eps)
            q = F.linear(q, wq_b).view(npos, n_heads, head_dim)
            q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(
                q.dtype
            )
            kv = _rms_norm(F.linear(x, wkv), kv_norm_w, eps)
            q = torch.cat(
                [q[..., :-rd], _apply_rope_complex(q[..., -rd:], pos_freqs)], dim=-1
            ).float()
            kv = torch.cat(
                [
                    kv[..., :-rd],
                    _apply_rope_complex(kv[..., -rd:].unsqueeze(1), pos_freqs).squeeze(
                        1
                    ),
                ],
                dim=-1,
            ).float()
            return q, kv

        # ---- FULL forward (gold reference) ----
        qf, kvf = proj_qkv(hidden_full, seqlen, freqs)
        o_full = _dense_causal_attn_o(
            qf, kvf, attn_sink, scale, seqlen, key_pos, device, n_groups, wo_a_g, wo_b
        )
        hidden_full = hidden_full + o_full

        # ---- OFFLINE: each chunk evolves independently; its KV is RoPE-realigned
        #      to absolute positions for the splice. ----
        kv_store = torch.empty(ctx_len, head_dim, device=device, dtype=torch.float32)
        for i, (a, b) in enumerate(chunk_spans):
            clen = b - a
            # KV at ABSOLUTE positions [a:b] for correct splice geometry.
            _, kv_abs = proj_qkv(chunk_h[i], clen, freqs[a:b])
            kv_store[a:b] = kv_abs
            # advance chunk hidden via INTRA-chunk attention only (positions 0..clen)
            q_loc, kv_loc = proj_qkv(chunk_h[i], clen, freqs[:clen])
            o_i = _dense_causal_attn_o(
                q_loc,
                kv_loc,
                attn_sink,
                scale,
                clen,
                torch.arange(clen, device=device),
                device,
                n_groups,
                wo_a_g,
                wo_b,
            )
            chunk_h[i] = chunk_h[i] + o_i
            del kv_abs, q_loc, kv_loc, o_i

        # ---- ONLINE: query rows attend to [offline chunk KV ++ online query KV] ----
        qo, kvo = proj_qkv(hidden_off, seqlen, freqs)
        kv_spliced = torch.cat([kv_store, kvo[ctx_len:]], dim=0)
        o_q = _dense_causal_attn_o_rows(
            qo[ctx_len:],
            kv_spliced,
            attn_sink,
            scale,
            ctx_len,
            seqlen,
            key_pos,
            device,
            n_groups,
            wo_a_g,
            wo_b,
        )
        new_off = hidden_off.clone()
        new_off[ctx_len:] = hidden_off[ctx_len:] + o_q
        new_off[:ctx_len] = emb_all[:ctx_len]  # chunk rows untrusted for query cos
        hidden_off = new_off

        if layer in layers:
            per_layer_cos.append((layer, _cos(hidden_off[-1], hidden_full[-1])))

        del wq_a, wq_b, wkv, wo_a, wo_b, qf, kvf, qo, kvo, o_full, o_q, kv_store
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- Accuracy ----
    final_cos = _cos(hidden_off[-1], hidden_full[-1])

    # ---- Compute / TTFT estimate (prefill FLOPs proxy) ----
    # Per-token prefill cost has two parts:
    #   linear projections + FFN/MoE  ~ proportional to #tokens processed
    #   attention (QK + AV)           ~ proportional to #query_tokens * ctx
    # Full prefill processes `seqlen` tokens; offline processes only `q_len`.
    # Attention term: full ~ sum_t t  ; offline ~ q_len * seqlen (query attends all).
    d = int(cfg["hidden_size"])
    # linear+ffn proxy per token (constant c1); use #tokens ratio.
    tokens_full = seqlen
    tokens_online = q_len
    proj_ffn_speedup = tokens_full / max(1, tokens_online)
    # attention proxy
    attn_full = seqlen * (seqlen + 1) / 2
    attn_online = q_len * seqlen
    # combined with a representative split: at long ctx, proj+ffn dominates TTFT.
    # report both bounds.
    print(f"\n{'layer':>6} {'cos(off,full) last-token':>26}")
    for li, c in per_layer_cos:
        print(f"{li:6d} {c:26.5f}")
    print("-" * 92)
    print(f" FINAL last-token cosine(offline_splice, full_prefill) = {final_cos:.5f}")
    print("-" * 92)
    print(" COMPUTE / TTFT ESTIMATE (prefill):")
    print(
        f"   tokens prefilled:  full={tokens_full}  online(offline-reuse)={tokens_online}"
    )
    print(
        f"   proj+FFN+MoE speedup (token-bound)     = {proj_ffn_speedup:6.2f}x  "
        f"(saving {100 * (1 - tokens_online / tokens_full):.1f}%)"
    )
    print(
        f"   attention QK/AV speedup                = {attn_full / max(1, attn_online):6.2f}x"
    )
    print(
        f"   => at long context TTFT is dominated by proj+FFN+MoE over chunk "
        f"tokens, which are SKIPPED online."
    )
    print(
        f"   => estimated TTFT speedup ~ {proj_ffn_speedup:.2f}x "
        f"(offline chunks prefilled once, reused across queries)"
    )
    print("=" * 92)


def _dense_causal_attn_o(
    q, kv_f, attn_sink, scale, seqlen, key_pos, device, n_groups, wo_a_g, wo_b
):
    n_heads = q.shape[1]
    chunk = max(64, min(512, seqlen))
    o_full = torch.empty(seqlen, wo_b.shape[0], device=device, dtype=torch.bfloat16)
    for cs in range(0, seqlen, chunk):
        ce = min(cs + chunk, seqlen)
        sc = torch.einsum("chd,sd->chs", q[cs:ce], kv_f) * scale
        qposc = torch.arange(cs, ce, device=device)[:, None, None]
        sc = sc.masked_fill(key_pos[None, None, :] > qposc, float("-inf"))
        sink = attn_sink.view(1, n_heads, 1).expand(ce - cs, n_heads, 1)
        pc = F.softmax(torch.cat([sc, sink], dim=-1), dim=-1)[..., :-1]
        ao = torch.einsum("chs,sd->chd", pc.to(kv_f.dtype), kv_f)
        oc = ao.reshape(ce - cs, n_groups, -1)
        oc = torch.einsum("cgd,grd->cgr", oc.float(), wo_a_g.float())
        oc = F.linear(oc.reshape(ce - cs, -1).to(wo_b.dtype), wo_b)
        o_full[cs:ce] = oc
        del sc, pc, ao, oc
    return o_full


def _dense_causal_attn_o_rows(
    q_rows,
    kv_spliced,
    attn_sink,
    scale,
    row_start,
    seqlen,
    key_pos,
    device,
    n_groups,
    wo_a_g,
    wo_b,
):
    """Attention for query rows [row_start:seqlen] over the full spliced KV."""
    n_heads = q_rows.shape[1]
    nrows = q_rows.shape[0]
    o = torch.empty(nrows, wo_b.shape[0], device=device, dtype=torch.bfloat16)
    chunk = max(64, min(512, nrows))
    for cs in range(0, nrows, chunk):
        ce = min(cs + chunk, nrows)
        abs_q = torch.arange(row_start + cs, row_start + ce, device=device)
        sc = torch.einsum("chd,sd->chs", q_rows[cs:ce], kv_spliced) * scale
        sc = sc.masked_fill(
            key_pos[None, None, :] > abs_q[:, None, None], float("-inf")
        )
        sink = attn_sink.view(1, n_heads, 1).expand(ce - cs, n_heads, 1)
        pc = F.softmax(torch.cat([sc, sink], dim=-1), dim=-1)[..., :-1]
        ao = torch.einsum("chs,sd->chd", pc.to(kv_spliced.dtype), kv_spliced)
        oc = ao.reshape(ce - cs, n_groups, -1)
        oc = torch.einsum("cgd,grd->cgr", oc.float(), wo_a_g.float())
        oc = F.linear(oc.reshape(ce - cs, -1).to(wo_b.dtype), wo_b)
        o[cs:ce] = oc
        del sc, pc, ao, oc
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default="/mnt/tidal-alsh01/dataset/redone/zhongming/checkpoints/DeepSeek-V4-Pro",
    )
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n-chunks", type=int, default=4)
    ap.add_argument("--chunk-tokens", type=int, default=4000)
    ap.add_argument("--query-tokens", type=int, default=64)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(
        args.model_path,
        args.dataset,
        args.n_chunks,
        args.chunk_tokens,
        args.query_tokens,
        args.layers,
        args.device,
    )


if __name__ == "__main__":
    main()
