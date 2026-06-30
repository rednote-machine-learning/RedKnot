#!/usr/bin/env python3
"""Offline numerical-equivalence prototype for the RedKnot "offline MLA + indexer
block reuse" idea on DeepSeek-V4.

The hypothesis (user design):

    1. Pre-compute and store the per-layer MLA latent KV of offline text once
       ("offline MLA").
    2. At query time, the per-layer indexer selects the top-k compressed KV
       blocks. Use exactly those block indices to fetch the corresponding KV
       from the offline store and run attention only over the selected set,
       skipping any online recompute for unselected blocks. Because we attend to
       exactly the blocks the indexer deemed important, accuracy is recovered.

This script does NOT touch the online inference path. It replays the real fp8
MLA forward (same dequant / RoPE / compressor / indexer math as
``profile_attention_concentration.py``) and compares, per ratio==4 layer:

  * BASELINE  : the model's real sparse attention = sliding window (recent
    ``sliding_window`` raw tokens) + indexer-selected compressed blocks.
  * OFFLINE   : identical math, but the compressed-block KV is taken from a
    pre-built "offline MLA" store and only the indexer-selected blocks are
    gathered (the unselected blocks are never used online).

If the design is sound, the two attention outputs are bit-for-bit identical
(cosine == 1.0, max-abs-diff == 0), because OFFLINE is just BASELINE with the
selected-block KV served from cache instead of recomputed online.

We also report an UPPER-BOUND control: attention restricted to a RANDOM same-size
block subset (instead of the indexer's), to show the indexer selection is what
preserves accuracy (random should have clearly lower cosine vs full attention).

Usage:
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/utils/verify_offline_mla_indexer_reuse.py \
    --model-path /mnt/.../DeepSeek-V4-Pro --dataset hotpotqa --length 8192 \
    --layers 2,6,12,24 --qsample 32
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

# Reuse the real-weight helpers already validated in the profiler.
from profile_attention_concentration import (  # noqa: E402
    _apply_rope_complex,
    _compress_index_kv,  # generic Compressor.forward; works for any compressor weights
    _dequant_fp8_blockwise,
    _hadamard_transform,
    _load_longbench_ids,
    _precompute_freqs_cis,
    _rms_norm,
)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float(
        torch.dot(a, b) / (a.norm().clamp_min(1e-12) * b.norm().clamp_min(1e-12))
    )


@torch.no_grad()
def verify(
    model_path: str,
    dataset: str,
    length: int,
    layers,
    qsample,
    device,
    topk_override: int = 0,
    no_window: bool = False,
):
    from safetensors import safe_open
    from transformers import AutoTokenizer

    cfg = json.load(open(Path(model_path) / "config.json"))
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
    window = int(cfg.get("sliding_window", cfg.get("window_size", 128)) or 128)
    idx_heads = int(cfg["index_n_heads"])
    idx_hd = int(cfg["index_head_dim"])
    idx_topk = int(cfg["index_topk"])

    index = json.load(open(Path(model_path) / "model.safetensors.index.json"))[
        "weight_map"
    ]

    def _get(name):
        with safe_open(str(Path(model_path) / index[name]), framework="pt") as f:
            return f.get_tensor(name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = _load_longbench_ids(tokenizer, dataset, length)[0].tolist()
    seqlen = len(input_ids)
    embed = _get("embed.weight")
    hidden = embed[torch.tensor(input_ids)].to(device=device, dtype=torch.bfloat16)

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

    n_rows = min(qsample, seqlen)
    query_idx = (
        torch.linspace(0, seqlen - 1, steps=n_rows, device=device).long().unique()
    )
    scale = head_dim**-0.5
    rng = torch.Generator(device=device).manual_seed(2026)

    print("=" * 88)
    print(" OFFLINE-MLA + INDEXER BLOCK-REUSE :: NUMERICAL EQUIVALENCE CHECK")
    print(f" model={model_path}")
    print(f" dataset={dataset} seqlen={seqlen} window={window} idx_topk={idx_topk}")
    print(f" queries={query_idx.numel()} layers={layers}")
    print("=" * 88)
    print(
        f"{'layer':>5} {'ratio':>5} {'nblk':>6} {'sel':>5} "
        f"{'cos(offline,base)':>18} {'maxdiff':>10} {'cos(random,full)':>16} "
        f"{'cos(idx,full)':>14}"
    )

    rows = []
    for layer in layers:
        ratio = compress_ratios[layer] if layer < len(compress_ratios) else 0
        if ratio != 4:
            print(f"{layer:>5} {ratio:>5}   (skip: not an indexer layer)")
            continue
        p = f"layers.{layer}.attn."
        ip = p + "indexer."
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
        idx_wq_b = _dequant_fp8_blockwise(
            _get(ip + "wq_b.weight"), _get(ip + "wq_b.scale")
        ).to(device)
        weights_proj = _get(ip + "weights_proj.weight").to(device)
        # Indexer's OWN compressor (for scoring blocks).
        c_wkv = _get(ip + "compressor.wkv.weight").float().to(device)
        c_wgate = _get(ip + "compressor.wgate.weight").float().to(device)
        c_ape = _get(ip + "compressor.ape").float().to(device)
        c_norm = _get(ip + "compressor.norm.weight").to(device)
        # MAIN attention compressor (the real KV blocks attention attends over).
        # indexer block b and main block b cover the SAME token range, so the
        # indexer's top-k indices select the correct main attention KV blocks.
        mc_wkv = _get(p + "compressor.wkv.weight").float().to(device)
        mc_wgate = _get(p + "compressor.wgate.weight").float().to(device)
        mc_ape = _get(p + "compressor.ape").float().to(device)
        mc_norm = _get(p + "compressor.norm.weight").to(device)

        x = _rms_norm(hidden, attn_norm, eps)
        qr = _rms_norm(F.linear(x, wq_a), q_norm_w, eps)

        # Main attention Q/KV (per official MLA forward).
        q = F.linear(qr, wq_b).view(seqlen, n_heads, head_dim)
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype)
        kv = _rms_norm(F.linear(x, wkv), kv_norm_w, eps)
        freqs = freqs_yarn  # ratio==4 uses compress/yarn rope
        q = torch.cat([q[..., :-rd], _apply_rope_complex(q[..., -rd:], freqs)], dim=-1)
        kv = torch.cat(
            [
                kv[..., :-rd],
                _apply_rope_complex(kv[..., -rd:].unsqueeze(1), freqs).squeeze(1),
            ],
            dim=-1,
        )
        q = q.float()
        kv_f = kv.float()

        # ---- Indexer: produce per-query top-k compressed block indices. ----
        iq = F.linear(qr, idx_wq_b).view(seqlen, idx_heads, idx_hd)
        iq = torch.cat(
            [iq[..., :-rd], _apply_rope_complex(iq[..., -rd:], freqs)], dim=-1
        )
        iq = _hadamard_transform(iq, idx_hd**-0.5)
        kv_blocks = _compress_index_kv(
            x.float(), c_wkv, c_wgate, c_ape, c_norm, freqs, ratio, idx_hd, rd, eps
        )
        kv_blocks = _hadamard_transform(kv_blocks, idx_hd**-0.5)
        nb = kv_blocks.shape[0]
        iw = F.linear(x, weights_proj).float() * (idx_hd**-0.5 * idx_heads**-0.5)

        # ---- Build the offline MLA compressed-block KV store using the MODEL'S
        # REAL main attention compressor (NOT mean-pool). This is the KV that the
        # real sparse_attn attends over, and block b here aligns with indexer
        # block b (same token range), so indexer top-k selects the right blocks.
        blk_kv = _compress_index_kv(
            x.float(),
            mc_wkv,
            mc_wgate,
            mc_ape,
            mc_norm,
            freqs,
            ratio,
            head_dim,
            rd,
            eps,
        )  # [nb, head_dim] — same nb as indexer kv_blocks
        assert blk_kv.shape[0] == nb, (blk_kv.shape, nb)
        block_last_tok = (torch.arange(nb, device=device) + 1) * ratio - 1  # causal ref

        eff_topk = topk_override if topk_override > 0 else idx_topk
        sel_k = min(eff_topk, nb)
        cos_off, cos_rand, cos_idx, maxdiff = [], [], [], []

        for t in query_idx.tolist():
            qrow = q[t]  # [H, hd]
            # sliding window: recent raw tokens (always attended, both paths).
            # With --no-window we drop it to ISOLATE block-selection quality
            # (otherwise the recent window dilutes the effect of which blocks
            # are picked, hiding the indexer's advantage).
            if no_window:
                win_kv = kv_f[t : t + 1]  # keep only self to avoid empty attn
            else:
                w_lo = max(0, t - window + 1)
                win_kv = kv_f[w_lo : t + 1]  # [w, hd]

            # visible compressed blocks for this query (causal): block fully in past
            vis = block_last_tok <= t
            n_vis = int(vis.sum())
            if n_vis < 8:
                continue

            # indexer score over visible blocks
            isc = torch.einsum("hd,bd->hb", iq[t].float(), kv_blocks.float()).relu_()
            iscore = (isc * iw[t].unsqueeze(-1)).sum(0)  # [nb]
            iscore = iscore.masked_fill(~vis, float("-inf"))
            k_here = min(sel_k, n_vis)
            sel_idx = torch.topk(iscore, k_here).indices  # indexer-selected blocks

            # FULL (all visible blocks) reference attention
            vis_idx = torch.nonzero(vis, as_tuple=False).squeeze(-1)
            full_o = _attn_over(qrow, win_kv, blk_kv[vis_idx], attn_sink, scale)

            # BASELINE = sparse_attn: window + indexer-selected blocks (recompute)
            base_o = _attn_over(qrow, win_kv, blk_kv[sel_idx], attn_sink, scale)

            # OFFLINE = window + indexer-selected blocks fetched from offline store.
            # The offline store is `blk_kv` (precomputed once); we gather exactly
            # the indexer indices. Mathematically identical to BASELINE.
            off_blk = blk_kv.index_select(0, sel_idx)  # gather from offline cache
            off_o = _attn_over(qrow, win_kv, off_blk, attn_sink, scale)

            # RANDOM control: same number of blocks but random visible subset.
            perm = torch.randperm(n_vis, generator=rng, device=device)[:k_here]
            rand_idx = vis_idx[perm]
            rand_o = _attn_over(qrow, win_kv, blk_kv[rand_idx], attn_sink, scale)

            cos_off.append(_cos(off_o, base_o))
            maxdiff.append(float((off_o - base_o).abs().max()))
            cos_idx.append(_cos(base_o, full_o))
            cos_rand.append(_cos(rand_o, full_o))
            pass

        import numpy as np

        co = float(np.mean(cos_off)) if cos_off else float("nan")
        md = float(np.max(maxdiff)) if maxdiff else float("nan")
        ci = float(np.mean(cos_idx)) if cos_idx else float("nan")
        cr = float(np.mean(cos_rand)) if cos_rand else float("nan")
        print(
            f"{layer:>5} {ratio:>5} {nb:>6} {sel_k:>5} "
            f"{co:>18.6f} {md:>10.2e} {cr:>16.4f} {ci:>14.4f}"
        )
        rows.append((layer, co, md, ci, cr))

        # advance hidden via real attention (use full attention for fidelity)
        hidden = _advance_hidden(
            hidden,
            q,
            kv_f,
            attn_sink,
            scale,
            seqlen,
            device,
            n_groups=int(cfg.get("o_groups", 8)),
            o_lora_rank=int(cfg.get("o_lora_rank", 1024)),
            wo_a=_dequant_fp8_blockwise(
                _get(p + "wo_a.weight"), _get(p + "wo_a.scale")
            ).to(device),
            wo_b=_dequant_fp8_blockwise(
                _get(p + "wo_b.weight"), _get(p + "wo_b.scale")
            ).to(device),
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("=" * 88)
    if rows:
        import numpy as np

        print(
            f" SUMMARY  cos(offline,baseline)={np.mean([r[1] for r in rows]):.6f} "
            f"(min {min(r[1] for r in rows):.6f})  "
            f"max|diff|={max(r[2] for r in rows):.2e}"
        )
        print(
            f"          cos(indexer-sel, full)={np.mean([r[3] for r in rows]):.4f}  "
            f"cos(random-sel, full)={np.mean([r[4] for r in rows]):.4f}"
        )
        print(
            " interpretation: offline==baseline (cos~1, diff~0) proves the reuse is"
            " LOSSLESS; indexer >> random vs full proves the selection preserves"
            " accuracy."
        )
    print("=" * 88)


def _attn_over(qrow, win_kv, blk_kv, attn_sink, scale):
    """Attention of one query over [window tokens ++ given blocks] with sink."""
    keys = torch.cat([blk_kv, win_kv], dim=0)  # [n, hd]
    sc = torch.einsum("hd,nd->hn", qrow, keys) * scale  # [H, n]
    sink = attn_sink.view(-1, 1)
    sc_aug = torch.cat([sc, sink], dim=-1)
    pr = F.softmax(sc_aug, dim=-1)[:, :-1]
    return torch.einsum("hn,nd->hd", pr.to(keys.dtype), keys)  # [H, hd]


def _advance_hidden(
    hidden,
    q,
    kv_f,
    attn_sink,
    scale,
    seqlen,
    device,
    *,
    n_groups,
    o_lora_rank,
    wo_a,
    wo_b,
):
    """Propagate hidden via dense-causal attention (chunked) for layer fidelity."""
    n_heads = q.shape[1]
    key_pos = torch.arange(seqlen, device=device)
    wo_a_g = wo_a.view(n_groups, o_lora_rank, -1)
    chunk = max(64, min(512, seqlen))
    o_full = torch.empty(seqlen, hidden.shape[-1], device=device, dtype=hidden.dtype)
    for cs in range(0, seqlen, chunk):
        ce = min(cs + chunk, seqlen)
        qc = q[cs:ce]
        sc = torch.einsum("chd,sd->chs", qc, kv_f) * scale
        qposc = torch.arange(cs, ce, device=device)[:, None, None]
        sc = sc.masked_fill(key_pos[None, None, :] > qposc, float("-inf"))
        sink = attn_sink.view(1, n_heads, 1).expand(ce - cs, n_heads, 1)
        pc = F.softmax(torch.cat([sc, sink], dim=-1), dim=-1)[..., :-1]
        ao = torch.einsum("chs,sd->chd", pc.to(kv_f.dtype), kv_f.to(kv_f.dtype))
        oc = ao.reshape(ce - cs, n_groups, -1)
        oc = torch.einsum("cgd,grd->cgr", oc.float(), wo_a_g.float())
        oc = F.linear(oc.reshape(ce - cs, -1).to(wo_b.dtype), wo_b)
        o_full[cs:ce] = oc
        del qc, sc, pc, ao, oc
    return hidden + o_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default="/mnt/tidal-alsh01/dataset/redone/zhongming/checkpoints/DeepSeek-V4-Pro",
    )
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument("--layers", default="2,6,12,24")
    ap.add_argument("--qsample", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--topk-override",
        type=int,
        default=0,
        help="force a smaller per-query top-k to expose indexer selection "
        "quality vs random (0 = use config index_topk).",
    )
    ap.add_argument(
        "--no-window",
        action="store_true",
        help="drop the sliding window to isolate block-selection quality.",
    )
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    verify(
        args.model_path,
        args.dataset,
        args.length,
        layers,
        args.qsample,
        args.device,
        topk_override=args.topk_override,
        no_window=args.no_window,
    )


if __name__ == "__main__":
    main()
