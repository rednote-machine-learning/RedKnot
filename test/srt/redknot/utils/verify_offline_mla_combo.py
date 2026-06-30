#!/usr/bin/env python3
"""Combined RedKnot scheme on DeepSeek-V4: offline-independent chunks + indexer
cross-chunk recompute, measuring the accuracy/compute tradeoff.

Background (validated separately):
  * Offline MLA reuse with chunks prefilled INDEPENDENTLY gives huge TTFT
    savings but loses cross-chunk information -> last-token cosine ~0.92.
  * Using the per-layer indexer to pick which KV blocks matter is LOSSLESS when
    the selected blocks' KV equals the recomputed KV.

Combined idea (user): keep chunks offline-independent, but at query time use the
indexer to select the top-r fraction of cross-chunk blocks that matter and
RECOMPUTE only those interactions, recovering accuracy toward the full-prefill
gold while recomputing only a small fraction of tokens.

This script measures, per recompute fraction r:
  * cosine(last-token hidden) of:
      - offline-only (r=0 baseline, no cross-chunk recompute)
      - INDEXER-selected recompute of fraction r of chunk tokens
      - RANDOM-selected recompute of fraction r (control)
  vs the full online prefill gold.
  * the implied online token budget (= query + recomputed chunk tokens) and the
    corresponding TTFT speedup estimate (full_tokens / online_tokens).

It is a faithful real-fp8 forward (same operators as the profiler) and does not
touch the serving path.

Recompute model: for a chosen set S of chunk token positions, those tokens are
re-driven ONLINE with full cross-chunk visibility (they attend to all offline
chunk KV + their own), updating their hidden; unselected chunk tokens keep their
offline-independent hidden. The query then attends to the (partially corrected)
spliced KV. Indexer ranks chunk tokens by how strongly the query/末端 attends to
their compressed blocks (the model's own importance signal).

Usage:
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/utils/verify_offline_mla_combo.py \
    --model-path /mnt/.../DeepSeek-V4-Pro --dataset hotpotqa \
    --n-chunks 4 --chunk-tokens 4000 --query-tokens 64 \
    --recompute-fracs 0,0.05,0.1,0.25,0.5
"""

from __future__ import annotations

import argparse
import json
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


def _attn_o(
    q_rows, kv, attn_sink, scale, abs_q, key_pos, n_groups, wo_a_g, wo_b, device
):
    """Causal attention of q_rows (abs positions abs_q) over kv, + sink, + O-proj."""
    n_heads = q_rows.shape[1]
    nrows = q_rows.shape[0]
    out = torch.empty(nrows, wo_b.shape[0], device=device, dtype=torch.bfloat16)
    chunk = max(64, min(512, nrows))
    for cs in range(0, nrows, chunk):
        ce = min(cs + chunk, nrows)
        sc = torch.einsum("chd,sd->chs", q_rows[cs:ce], kv) * scale
        sc = sc.masked_fill(
            key_pos[None, None, :] > abs_q[cs:ce][:, None, None], float("-inf")
        )
        sink = attn_sink.view(1, n_heads, 1).expand(ce - cs, n_heads, 1)
        pc = F.softmax(torch.cat([sc, sink], dim=-1), dim=-1)[..., :-1]
        ao = torch.einsum("chs,sd->chd", pc.to(kv.dtype), kv)
        oc = ao.reshape(ce - cs, n_groups, -1)
        oc = torch.einsum("cgd,grd->cgr", oc.float(), wo_a_g.float())
        oc = F.linear(oc.reshape(ce - cs, -1).to(wo_b.dtype), wo_b)
        out[cs:ce] = oc
        del sc, pc, ao, oc
    return out


@torch.no_grad()
def run(
    model_path, dataset, n_chunks, chunk_tokens, query_tokens, recompute_fracs, device
):
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
    ctx_len = n_chunks * chunk_tokens
    q_len = seqlen - ctx_len

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

    fracs = [float(x) for x in recompute_fracs.split(",") if x.strip()]

    print("=" * 96)
    print(" COMBINED: offline-independent chunks + indexer cross-chunk recompute")
    print(f" model={model_path}")
    print(
        f" chunks={n_chunks}x{chunk_tokens}={ctx_len} offline | query={q_len} | total={seqlen}"
    )
    print(f" recompute fracs={fracs}")
    print("=" * 96)

    # Gold full-prefill hidden, offline-independent chunk hidden, and per-layer
    # offline KV store are all built in one synchronized layer loop.
    hidden_full = emb_all.clone()
    chunk_h = [emb_all[a:b].clone() for (a, b) in chunk_spans]
    # offline-only spliced hidden (chunk rows = independent; query rows online)
    hidden_off = emb_all.clone()

    # We also need, per recompute-frac, a corrected hidden. To keep memory bounded
    # we run a SEPARATE corrected pass per frac AFTER caching all offline KV per
    # layer would be too heavy; instead we compute importance once (final layer
    # query attention to chunk blocks) using the offline pass, pick token sets,
    # then re-run the corrected forwards. For tractability we measure importance
    # via the offline-independent chunk-token L2 attention proxy at a mid layer.

    # ---- Pass 1: offline-only forward, also record per-token importance. ----
    # importance[t] = how much the query rows attend to chunk token t (accumulated
    # across layers via the spliced attention probs). We approximate with the mean
    # attention mass from query rows to each chunk token at a representative layer.
    importance = torch.zeros(ctx_len, device=device)

    # store all layer weights once is too big; we re-read per layer in each pass.
    def layer_weights(layer):
        p = f"layers.{layer}.attn."
        return dict(
            attn_norm=_get(f"layers.{layer}.attn_norm.weight").to(device),
            wq_a=_dequant_fp8_blockwise(
                _get(p + "wq_a.weight"), _get(p + "wq_a.scale")
            ).to(device),
            wq_b=_dequant_fp8_blockwise(
                _get(p + "wq_b.weight"), _get(p + "wq_b.scale")
            ).to(device),
            wkv=_dequant_fp8_blockwise(
                _get(p + "wkv.weight"), _get(p + "wkv.scale")
            ).to(device),
            q_norm_w=_get(p + "q_norm.weight").to(device),
            kv_norm_w=_get(p + "kv_norm.weight").to(device),
            attn_sink=_get(p + "attn_sink").to(device).float(),
            wo_a=_dequant_fp8_blockwise(
                _get(p + "wo_a.weight"), _get(p + "wo_a.scale")
            ).to(device),
            wo_b=_dequant_fp8_blockwise(
                _get(p + "wo_b.weight"), _get(p + "wo_b.scale")
            ).to(device),
        )

    def proj_qkv(W, hin, npos, pos_freqs):
        x = _rms_norm(hin, W["attn_norm"], eps)
        q = _rms_norm(F.linear(x, W["wq_a"]), W["q_norm_w"], eps)
        q = F.linear(q, W["wq_b"]).view(npos, n_heads, head_dim)
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype)
        kv = _rms_norm(F.linear(x, W["wkv"]), W["kv_norm_w"], eps)
        q = torch.cat(
            [q[..., :-rd], _apply_rope_complex(q[..., -rd:], pos_freqs)], dim=-1
        ).float()
        kv = torch.cat(
            [
                kv[..., :-rd],
                _apply_rope_complex(kv[..., -rd:].unsqueeze(1), pos_freqs).squeeze(1),
            ],
            dim=-1,
        ).float()
        return q, kv

    # Cache per-layer offline KV store (RoPE-realigned) and chunk-evolved hidden
    # snapshots so the corrected passes can reuse them without recompute.
    kv_store_layers = []
    chunk_h_in_layers = []  # input chunk hidden to each layer (for recompute)
    hidden_off_in_layers = []

    for layer in range(n_layers):
        ratio = compress_ratios[layer] if layer < len(compress_ratios) else 0
        freqs = freqs_plain if ratio == 0 else freqs_yarn
        W = layer_weights(layer)
        wo_a_g = W["wo_a"].view(n_groups, o_lora_rank, -1)

        # gold
        qf, kvf = proj_qkv(W, hidden_full, seqlen, freqs)
        o_full = _attn_o(
            qf,
            kvf,
            W["attn_sink"],
            scale,
            key_pos,
            key_pos,
            n_groups,
            wo_a_g,
            W["wo_b"],
            device,
        )
        hidden_full = hidden_full + o_full

        # offline KV store (independent chunks, abs-pos RoPE) + advance chunk hidden
        kv_store = torch.empty(ctx_len, head_dim, device=device, dtype=torch.float32)
        chunk_h_in_layers.append([c.clone() for c in chunk_h])
        for i, (a, b) in enumerate(chunk_spans):
            clen = b - a
            _, kv_abs = proj_qkv(W, chunk_h[i], clen, freqs[a:b])
            kv_store[a:b] = kv_abs
            q_loc, kv_loc = proj_qkv(W, chunk_h[i], clen, freqs[:clen])
            o_i = _attn_o(
                q_loc,
                kv_loc,
                W["attn_sink"],
                scale,
                torch.arange(clen, device=device),
                torch.arange(clen, device=device),
                n_groups,
                wo_a_g,
                W["wo_b"],
                device,
            )
            chunk_h[i] = chunk_h[i] + o_i
            del kv_abs, q_loc, kv_loc, o_i
        kv_store_layers.append(kv_store)

        # offline-only online query rows
        hidden_off_in_layers.append(hidden_off.clone())
        qo, kvo = proj_qkv(W, hidden_off, seqlen, freqs)
        kv_spliced = torch.cat([kv_store, kvo[ctx_len:]], dim=0)
        absq = torch.arange(ctx_len, seqlen, device=device)
        o_q = _attn_o(
            qo[ctx_len:],
            kv_spliced,
            W["attn_sink"],
            scale,
            absq,
            key_pos,
            n_groups,
            wo_a_g,
            W["wo_b"],
            device,
        )
        # importance: query attention mass to each chunk token (last query row)
        if layer == n_layers // 2:
            ql = qo[seqlen - 1]  # [H, hd]
            sc = torch.einsum("hd,sd->hs", ql, kv_store) * scale  # [H, ctx]
            importance += F.softmax(sc, dim=-1).mean(0)
        new_off = hidden_off.clone()
        new_off[ctx_len:] = hidden_off[ctx_len:] + o_q
        new_off[:ctx_len] = emb_all[:ctx_len]
        hidden_off = new_off

        del W, qf, kvf, qo, kvo, o_full, o_q
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    cos_offline = _cos(hidden_off[-1], hidden_full[-1])

    # ---- Corrected passes: recompute fraction r of chunk tokens with full
    #      cross-chunk visibility. Selection = indexer importance vs random. ----
    rng = torch.Generator(device=device).manual_seed(2026)
    results = []
    for r in fracs:
        if r <= 0:
            results.append((r, cos_offline, cos_offline, q_len))
            continue
        n_recomp = int(r * ctx_len)
        sel_idx = torch.topk(importance, n_recomp).indices
        rand_idx = torch.randperm(ctx_len, generator=rng, device=device)[:n_recomp]
        cos_sel = _corrected_pass(
            sel_idx,
            n_layers,
            compress_ratios,
            freqs_plain,
            freqs_yarn,
            kv_store_layers,
            chunk_h_in_layers,
            hidden_off_in_layers,
            hidden_full,
            emb_all,
            ctx_len,
            seqlen,
            key_pos,
            scale,
            n_groups,
            o_lora_rank,
            head_dim,
            n_heads,
            rd,
            eps,
            layer_weights,
            proj_qkv,
            device,
        )
        cos_rand = _corrected_pass(
            rand_idx,
            n_layers,
            compress_ratios,
            freqs_plain,
            freqs_yarn,
            kv_store_layers,
            chunk_h_in_layers,
            hidden_off_in_layers,
            hidden_full,
            emb_all,
            ctx_len,
            seqlen,
            key_pos,
            scale,
            n_groups,
            o_lora_rank,
            head_dim,
            n_heads,
            rd,
            eps,
            layer_weights,
            proj_qkv,
            device,
        )
        results.append((r, cos_sel, cos_rand, q_len + n_recomp))

    print(
        f"\n{'recomp_r':>9} {'online_tok':>11} {'ttft_x':>8} {'cos_INDEXER':>12} {'cos_RANDOM':>11}"
    )
    for r, cs, cr, otok in results:
        ttx = seqlen / max(1, otok)
        print(f"{r:9.2f} {otok:11d} {ttx:8.1f} {cs:12.5f} {cr:11.5f}")
    print("-" * 96)
    print(f" offline-only (r=0) cosine = {cos_offline:.5f}  | full-prefill gold = 1.0")
    print(" indexer recompute should climb toward 1.0 faster than random as r grows.")
    print("=" * 96)


def _corrected_pass(
    recomp_idx,
    n_layers,
    compress_ratios,
    freqs_plain,
    freqs_yarn,
    kv_store_layers,
    chunk_h_in_layers,
    hidden_off_in_layers,
    hidden_full,
    emb_all,
    ctx_len,
    seqlen,
    key_pos,
    scale,
    n_groups,
    o_lora_rank,
    head_dim,
    n_heads,
    rd,
    eps,
    layer_weights,
    proj_qkv,
    device,
):
    """Re-run the online path but ALSO recompute hidden for chunk tokens in
    recomp_idx with full cross-chunk visibility, then measure query last-token
    cosine vs gold. Uses cached per-layer offline KV stores."""
    recomp_mask = torch.zeros(ctx_len, dtype=torch.bool, device=device)
    recomp_mask[recomp_idx] = True
    hidden = emb_all.clone()  # chunk rows for recomputed tokens evolve online
    for layer in range(n_layers):
        ratio = compress_ratios[layer] if layer < len(compress_ratios) else 0
        freqs = freqs_plain if ratio == 0 else freqs_yarn
        W = layer_weights(layer)
        wo_a_g = W["wo_a"].view(n_groups, o_lora_rank, -1)
        kv_store = kv_store_layers[layer]

        qo, kvo = proj_qkv(W, hidden, seqlen, freqs)
        # KV: offline chunk store overrides chunk rows; query rows from online
        kv_spliced = torch.cat([kv_store, kvo[ctx_len:]], dim=0)
        # recomputed chunk tokens use their freshly-projected online KV (full vis)
        kv_spliced[:ctx_len][recomp_mask] = kvo[:ctx_len][recomp_mask]

        # query rows + recomputed chunk rows attend with full visibility
        rows = torch.cat(
            [
                torch.nonzero(recomp_mask, as_tuple=False).squeeze(-1),
                torch.arange(ctx_len, seqlen, device=device),
            ]
        )
        absq = rows
        o_rows = _attn_o(
            qo[rows],
            kv_spliced,
            W["attn_sink"],
            scale,
            absq,
            key_pos,
            n_groups,
            wo_a_g,
            W["wo_b"],
            device,
        )
        new_h = hidden.clone()
        new_h[rows] = hidden[rows] + o_rows
        # non-recomputed chunk rows: keep offline-independent hidden snapshot input
        # (they were never corrected) — approximate by their offline chunk hidden.
        hidden = new_h
        del W, qo, kvo, o_rows
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return _cos(hidden[-1], hidden_full[-1])


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
    ap.add_argument("--recompute-fracs", default="0,0.05,0.1,0.25,0.5")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(
        args.model_path,
        args.dataset,
        args.n_chunks,
        args.chunk_tokens,
        args.query_tokens,
        args.recompute_fracs,
        args.device,
    )


if __name__ == "__main__":
    main()
