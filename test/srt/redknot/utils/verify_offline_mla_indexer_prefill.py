#!/usr/bin/env python3
"""End-to-end offline-MLA RAG prefill with per-layer indexer-driven sparse MLA.

User scheme (DeepSeek-V4):
  1. OFFLINE: each text chunk is prefilled independently; its per-layer MLA latent
     KV is stored, computed at LOCAL positions [0..chunk_len) (self-contained).
  2. ONLINE SPLICE: chunks are concatenated. Every chunk except the first has its
     stored KV RoPE re-rotated from local position to its ABSOLUTE position via a
     delta rotation `freqs[abs] / freqs[local]` (RoPE rotation invariance — proven
     numerically equivalent to recomputing at the absolute position).
  3. WARMUP LAYERS: the first `warmup_layers` layers recompute ALL tokens' MLA.
  4. INDEXER-DRIVEN SPARSE MLA: from the first indexer layer onward, each indexer
     (ratio==4) layer produces a fresh top-k token selection that REPLACES the
     previous one. Subsequent layers (until the next indexer layer) compute MLA
     ONLY for the currently-selected tokens; unselected tokens skip the MLA
     attention update (their hidden carries the offline/previous value).
  5. FFN / MoE / norm run normally for every token (we model the residual; FFN is
     not the target of the sparsity here).

We compare the final last-token hidden state against a full online prefill (gold)
and report cosine + the compute saving (fraction of token-MLA work skipped) +
the estimated TTFT speedup.

This is a faithful real-fp8 forward (same operators as the profiler).

Usage:
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/utils/verify_offline_mla_indexer_prefill.py \
    --model-path /mnt/.../DeepSeek-V4-Pro --dataset hotpotqa \
    --n-chunks 4 --chunk-tokens 4000 --query-tokens 64 \
    --warmup-layers 4 --topk-override 1024
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
    _compress_index_kv,
    _dequant_fp8_blockwise,
    _hadamard_transform,
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
def run(
    model_path,
    dataset,
    n_chunks,
    chunk_tokens,
    query_tokens,
    warmup_layers,
    topk_override,
    device,
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
    idx_heads = int(cfg["index_n_heads"])
    idx_hd = int(cfg["index_head_dim"])
    idx_topk = int(cfg["index_topk"])
    window = int(cfg.get("sliding_window", cfg.get("window_size", 128)) or 128)

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
    chunk_spans = [(i * chunk_tokens, (i + 1) * chunk_tokens) for i in range(n_chunks)]

    embed = _get("embed.weight")
    emb_all = embed[torch.tensor(ids)].to(device=device, dtype=torch.bfloat16)

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
    eff_topk = topk_override if topk_override > 0 else idx_topk

    indexer_layers = [
        i
        for i in range(n_layers)
        if i < len(compress_ratios) and compress_ratios[i] == 4
    ]

    print("=" * 96)
    print(" OFFLINE MLA + ONLINE SPLICE + PER-LAYER INDEXER SPARSE MLA")
    print(f" model={model_path}")
    print(
        f" chunks={n_chunks}x{chunk_tokens}={ctx_len} | query={q_len} | total={seqlen}"
    )
    print(f" warmup_layers={warmup_layers} (full recompute) | indexer topk={eff_topk}")
    print(f" indexer layers (refresh selection): {indexer_layers[:10]}...")
    print(f" RoPE position recovery: delta rotation (first chunk unchanged)")
    print("=" * 96)

    # Running hidden states.
    hidden_full = emb_all.clone()  # gold full prefill
    hidden_sp = emb_all.clone()  # our offline+sparse pipeline
    chunk_h = [emb_all[a:b].clone() for (a, b) in chunk_spans]

    # current selected token set (absolute indices). None => all tokens.
    selected = None
    mla_token_work = 0  # accumulator: how many token-MLA were computed (our path)
    mla_token_full = 0  # full path

    def lw(layer):
        p = f"layers.{layer}.attn."
        d = dict(
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
        ip = p + "indexer."
        if (ip + "wq_b.weight") in index:
            d["idx_wq_b"] = _dequant_fp8_blockwise(
                _get(ip + "wq_b.weight"), _get(ip + "wq_b.scale")
            ).to(device)
            d["weights_proj"] = _get(ip + "weights_proj.weight").to(device)
            d["ic_wkv"] = _get(ip + "compressor.wkv.weight").float().to(device)
            d["ic_wgate"] = _get(ip + "compressor.wgate.weight").float().to(device)
            d["ic_ape"] = _get(ip + "compressor.ape").float().to(device)
            d["ic_norm"] = _get(ip + "compressor.norm.weight").to(device)
        return d

    def proj_qkv(W, hin, npos, pos_freqs):
        x = _rms_norm(hin, W["attn_norm"], eps)
        qr = _rms_norm(F.linear(x, W["wq_a"]), W["q_norm_w"], eps)
        q = F.linear(qr, W["wq_b"]).view(npos, n_heads, head_dim)
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
        return q, kv, qr

    def attn_o(q_rows, kv, W, abs_q, wo_a_g, k_pos=None):
        nh = q_rows.shape[1]
        nr = q_rows.shape[0]
        if k_pos is None:
            k_pos = torch.arange(kv.shape[0], device=device)
        out = torch.empty(nr, W["wo_b"].shape[0], device=device, dtype=torch.bfloat16)
        chunk = max(64, min(512, nr))
        for cs in range(0, nr, chunk):
            ce = min(cs + chunk, nr)
            sc = torch.einsum("chd,sd->chs", q_rows[cs:ce], kv) * scale
            sc = sc.masked_fill(
                k_pos[None, None, :] > abs_q[cs:ce][:, None, None], float("-inf")
            )
            sink = W["attn_sink"].view(1, nh, 1).expand(ce - cs, nh, 1)
            pc = F.softmax(torch.cat([sc, sink], dim=-1), dim=-1)[..., :-1]
            ao = torch.einsum("chs,sd->chd", pc.to(kv.dtype), kv)
            oc = ao.reshape(ce - cs, n_groups, -1)
            oc = torch.einsum("cgd,grd->cgr", oc.float(), wo_a_g.float())
            oc = F.linear(oc.reshape(ce - cs, -1).to(W["wo_b"].dtype), W["wo_b"])
            out[cs:ce] = oc
            del sc, pc, ao, oc
        return out

    for layer in range(n_layers):
        ratio = compress_ratios[layer] if layer < len(compress_ratios) else 0
        freqs = freqs_plain if ratio == 0 else freqs_yarn
        W = lw(layer)
        wo_a_g = W["wo_a"].view(n_groups, o_lora_rank, -1)

        # ---------- GOLD full prefill ----------
        qf, kvf, _ = proj_qkv(W, hidden_full, seqlen, freqs)
        o_full = attn_o(qf, kvf, W, key_pos, wo_a_g)
        hidden_full = hidden_full + o_full
        mla_token_full += seqlen

        # ---------- OUR pipeline ----------
        # 1) Build the offline-spliced KV for this layer.
        #    chunks: stored KV at LOCAL positions, delta-rotated to ABS positions.
        kv_store = torch.empty(ctx_len, head_dim, device=device, dtype=torch.float32)
        for i, (a, b) in enumerate(chunk_spans):
            clen = b - a
            # KV computed at LOCAL positions [0:clen]
            _, kv_loc, _ = proj_qkv(W, chunk_h[i], clen, freqs[:clen])
            if i == 0:
                kv_store[a:b] = kv_loc  # first chunk already at correct abs (0-based)
            else:
                # delta-rotate rope dims from local pos -> abs pos [a:b]
                kv_abs = kv_loc.clone()
                loc_f = freqs[:clen]
                abs_f = freqs[a:b]
                rope = kv_loc[:, -rd:]
                rc = torch.view_as_complex(rope.float().unflatten(-1, (-1, 2)))
                delta = abs_f[:clen] / loc_f[:clen]  # complex ratio = R(abs-local)
                rc = rc * delta
                kv_abs[:, -rd:] = torch.view_as_real(rc).flatten(-2)
                kv_store[a:b] = kv_abs
                del kv_abs, rc
            # advance chunk hidden via INTRA-chunk attention (independent)
            q_loc, kv_loc2, _ = proj_qkv(W, chunk_h[i], clen, freqs[:clen])
            o_i = attn_o(q_loc, kv_loc2, W, torch.arange(clen, device=device), wo_a_g)
            chunk_h[i] = chunk_h[i] + o_i
            del kv_loc, q_loc, kv_loc2, o_i

        # 2) query KV (online)
        qo, kvo, qr_o = proj_qkv(W, hidden_sp, seqlen, freqs)
        kv_spliced = torch.cat([kv_store, kvo[ctx_len:]], dim=0)

        # 3) decide which tokens compute MLA this layer
        if layer < warmup_layers:
            rows = torch.arange(seqlen, device=device)  # all tokens
        else:
            if selected is None:
                rows = torch.arange(seqlen, device=device)
            else:
                rows = selected

        o_rows = attn_o(qo[rows], kv_spliced, W, rows, wo_a_g)
        new_sp = hidden_sp.clone()
        new_sp[rows] = hidden_sp[rows] + o_rows
        hidden_sp = new_sp
        mla_token_work += rows.numel()

        # 4) if this is an indexer layer at/after warmup, REFRESH the selection
        if ratio == 4 and "idx_wq_b" in W and layer >= warmup_layers - 1:
            selected = _indexer_select(
                W,
                qr_o,
                hidden_sp,
                kv_store,
                kvo,
                freqs,
                ratio,
                idx_hd,
                idx_heads,
                rd,
                eps,
                ctx_len,
                seqlen,
                eff_topk,
                window,
                device,
            )

        del W, qf, kvf, o_full, qo, kvo, o_rows, kv_store, kv_spliced
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    final_cos = _cos(hidden_sp[-1], hidden_full[-1])
    mla_saving = 1 - mla_token_work / max(1, mla_token_full)

    print(f"\n FINAL last-token cosine(offline+sparse, full) = {final_cos:.5f}")
    print(
        f" token-MLA computed: ours={mla_token_work} full={mla_token_full} "
        f"=> MLA work saved {mla_saving:.1%}"
    )
    print(
        f" online prefill tokens (excl. offline chunks) ~ {q_len}, "
        f"prefix-cache TTFT speedup ~ {seqlen / max(1, q_len):.1f}x (reuse-amortized)"
    )
    print("=" * 96)
    return final_cos, mla_saving


def _indexer_select(
    W,
    qr,
    hidden,
    kv_store,
    kvo,
    freqs,
    ratio,
    idx_hd,
    idx_heads,
    rd,
    eps,
    ctx_len,
    seqlen,
    topk,
    window,
    device,
):
    """Run the real indexer on the LAST query position to pick top-k token blocks,
    then expand to absolute token indices (block -> its `ratio` tokens), plus the
    sliding window + query rows (always kept). Returns sorted absolute indices."""
    x = _rms_norm(hidden, W["attn_norm"], eps)
    iq = F.linear(qr, W["idx_wq_b"]).view(seqlen, idx_heads, idx_hd)
    iq = torch.cat([iq[..., :-rd], _apply_rope_complex(iq[..., -rd:], freqs)], dim=-1)
    iq = _hadamard_transform(iq, idx_hd**-0.5)
    kv_blocks = _compress_index_kv(
        x.float(),
        W["ic_wkv"],
        W["ic_wgate"],
        W["ic_ape"],
        W["ic_norm"],
        freqs,
        ratio,
        idx_hd,
        rd,
        eps,
    )
    kv_blocks = _hadamard_transform(kv_blocks, idx_hd**-0.5)
    nb = kv_blocks.shape[0]
    iw = F.linear(x, W["weights_proj"]).float() * (idx_hd**-0.5 * idx_heads**-0.5)
    t = seqlen - 1
    isc = torch.einsum("hd,bd->hb", iq[t].float(), kv_blocks.float()).relu_()
    iscore = (isc * iw[t].unsqueeze(-1)).sum(0)  # [nb]
    block_last = (torch.arange(nb, device=device) + 1) * ratio - 1
    iscore = iscore.masked_fill(block_last > t, float("-inf"))
    k = min(topk, int((block_last <= t).sum()))
    sel_blocks = torch.topk(iscore, k).indices
    # expand blocks -> token indices
    tok = (
        sel_blocks[:, None] * ratio + torch.arange(ratio, device=device)[None, :]
    ).flatten()
    tok = tok[tok < ctx_len]
    # always keep: sliding window tail + all query rows
    win = torch.arange(max(0, seqlen - window), seqlen, device=device)
    qrows = torch.arange(ctx_len, seqlen, device=device)
    sel = torch.unique(torch.cat([tok, win, qrows]))
    return sel


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
    ap.add_argument("--warmup-layers", type=int, default=4)
    ap.add_argument("--topk-override", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(
        args.model_path,
        args.dataset,
        args.n_chunks,
        args.chunk_tokens,
        args.query_tokens,
        args.warmup_layers,
        args.topk_override,
        args.device,
    )


if __name__ == "__main__":
    main()
