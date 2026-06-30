#!/usr/bin/env python3
"""Heterogeneous per-head window: SegPaged vs Paged KV bandwidth (Qwen3-32B).

Two stages:

STAGE 1 — PROBE (real model): load Qwen3-32B, run a small calibration corpus
  with output_attentions, and for every (layer, kv-head) measure the REAL
  coverage window = the smallest distance w such that the cumulative attention
  mass within the last w tokens reaches `coverage` (e.g. 0.95). Heads vary a
  lot: some concentrate within ~256, others spread to thousands -> a real
  HETEROGENEOUS per-head window distribution.

STAGE 2 — BANDWIDTH: with the real per-head windows, compare decode KV-read
  bandwidth of two backend storages:
    Paged (token-major, unified block): the layer's block must cover the MAX
      window over its heads; reading any token row drags ALL Hkv heads ->
      heads with small windows are READ-AMPLIFIED to the layer max.
    SegPaged (head-major, per-head pages): each head reads exactly its own
      window -> no amplification.

The point: a UNIFORM window hides SegPaged's advantage (Paged block == window);
the REAL heterogeneous windows expose Paged's read amplification.

Usage:
  REDKNOT_MODEL_PATH=.../Qwen3-32B CUDA_VISIBLE_DEVICES=0 \
    python heterogeneous_window_segpaged_vs_paged.py --probe-samples 4 --coverage 0.95
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL_PATH = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3-32B",
)
HOTPOT = os.environ.get(
    "REDKNOT_HOTPOT_PARQUET",
    str(
        Path(__file__).resolve().parent
        / "datasets/HotpotQA/distractor/validation-00000-of-00001.parquet"
    ),
)


LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)


def load_calib_prompts(n, target_tokens=5000):
    """Use REAL long LongBench contexts so attention mass spreads over distance
    (short HotpotQA contexts ~1K tokens cannot reveal per-head window
    heterogeneity)."""
    import json as _json

    prompts = []
    for ds in ["narrativeqa", "musique", "hotpotqa", "2wikimqa"]:
        p = os.path.join(LONGBENCH_DIR, f"{ds}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                r = _json.loads(line)
                ctx = r.get("context", "")
                if ctx and len(ctx) // 4 >= target_tokens:
                    q = r.get("input", "").strip()
                    prompts.append(f"{ctx}\n\nQuestion: {q}\nAnswer:")
                if len(prompts) >= n:
                    return prompts
    return prompts


@torch.no_grad()
def probe_coverage_windows(model, tok, prompts, coverage, device):
    """Return cov_win[L][Hkv] = real coverage window per (layer, kv-head)."""
    cfg = model.config
    L = cfg.num_hidden_layers
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    group = Hq // Hkv
    # distance bins (log-ish) up to max ctx
    edges = [
        0,
        32,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        8192,
        16384,
        1 << 20,
    ]
    # accumulate mass-by-distance per (layer, q-head)
    acc = torch.zeros(L, Hq, len(edges), dtype=torch.float64)
    nsamp = 0
    PROBE_LEN = int(os.environ.get("REDKNOT_PROBE_LEN", "6144"))
    for p in prompts:
        ids = tok(
            p, return_tensors="pt", truncation=True, max_length=PROBE_LEN
        ).input_ids.to(device)
        out = model(ids, output_attentions=True, use_cache=False)
        atts = out.attentions  # tuple[L] of [1, Hq, S, S]
        S = ids.shape[1]
        qi = torch.arange(S, device=device)
        d = qi[:, None] - qi[None, :]  # [S,S] distance, causal d>=0
        for li in range(L):
            a = atts[li][0]  # [Hq, S, S]
            for bi in range(len(edges) - 1):
                lo, hi = edges[bi], edges[bi + 1]
                m = (d > lo) & (d <= hi)
                mass = (a * m[None]).sum(dim=(1, 2))  # [Hq]
                acc[li, :, bi + 1] += mass.double().cpu()
        nsamp += 1
        del out, atts, d
        torch.cuda.empty_cache()
    # cumulative mass by distance, normalize, find coverage window per q-head
    cum = torch.cumsum(acc, dim=2)
    total = cum[:, :, -1:].clamp(min=1e-9)
    frac = cum / total
    cov_win_q = torch.full((L, Hq), float(edges[-2]))
    for li in range(L):
        for h in range(Hq):
            reached = (frac[li, h] >= coverage).nonzero()
            if len(reached):
                cov_win_q[li, h] = edges[int(reached[0])]
    # reduce q-heads -> kv-heads (max over group = the kv-head must serve all its q-heads)
    cov_win = torch.zeros(L, Hkv)
    for li in range(L):
        for kv in range(Hkv):
            cov_win[li, kv] = cov_win_q[li, kv * group : (kv + 1) * group].max()
    return cov_win.tolist(), {"L": L, "Hkv": Hkv, "Hq": Hq, "samples": nsamp}


def bandwidth(cov_win, L, Hkv, D, DT, Lctx, sink, global_thresh):
    """Paged vs SegPaged decode KV-read bytes, with heterogeneous per-head window.
    A head whose coverage window >= global_thresh*Lctx is treated as 'full'."""
    pg = sg = 0
    for li in range(L):
        wins = []
        for kv in range(Hkv):
            w = cov_win[li][kv]
            if w >= global_thresh * Lctx:
                wins.append(Lctx)  # global/full
            else:
                wins.append(min(int(w) + sink, Lctx))
        for w in wins:
            sg += w * D * 2 * DT
        cover = max(wins)  # paged unified block covers the layer max
        pg += cover * Hkv * D * 2 * DT
    return pg, sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-samples", type=int, default=4)
    ap.add_argument("--coverage", type=float, default=0.95)
    ap.add_argument(
        "--global-thresh",
        type=float,
        default=0.5,
        help="head is 'full' if coverage window >= thresh*Lctx",
    )
    ap.add_argument("--lengths", default="16000,32000,64000,128000")
    ap.add_argument("--kv-dtype-bytes", type=int, default=2)
    ap.add_argument("--cache-windows", default="/tmp/qwen3_cov_windows.json")
    args = ap.parse_args()

    # STAGE 1: probe (or reuse cached)
    if os.path.exists(args.cache_windows):
        print(f"Reusing cached coverage windows: {args.cache_windows}")
        data = json.load(open(args.cache_windows))
        cov_win, meta = data["cov_win"], data["meta"]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {MODEL_PATH} (NF4) for attention probing...")
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb,
            device_map="cuda:0",
            attn_implementation="eager",
            trust_remote_code=True,
        )
        model.eval()
        prompts = load_calib_prompts(args.probe_samples)
        print(
            f"Probing {len(prompts)} calibration prompts for per-head coverage windows..."
        )
        cov_win, meta = probe_coverage_windows(
            model, tok, prompts, args.coverage, "cuda:0"
        )
        json.dump({"cov_win": cov_win, "meta": meta}, open(args.cache_windows, "w"))
        print(f"Saved coverage windows -> {args.cache_windows}")
        del model
        torch.cuda.empty_cache()

    L, Hkv, Hq = meta["L"], meta["Hkv"], meta["Hq"]
    D = 128
    DT = args.kv_dtype_bytes
    lengths = [int(x) for x in args.lengths.split(",")]

    # window distribution summary
    flat = [w for row in cov_win for w in row]
    import statistics

    print(f"\nReal per-(layer,kv-head) coverage window (coverage={args.coverage}):")
    print(
        f"  min={min(flat):.0f} median={statistics.median(flat):.0f} "
        f"max={max(flat):.0f} mean={statistics.mean(flat):.0f}"
    )
    from collections import Counter

    print(f"  distribution: {dict(sorted(Counter(flat).items()))}")

    # STAGE 2: bandwidth
    print("\n" + "=" * 78)
    print("DECODE KV-read bandwidth: Paged vs SegPaged (REAL heterogeneous windows)")
    print("=" * 78)
    print(f"{'L_ctx':>7} | {'Paged_GB':>9} {'SegP_GB':>9} {'ratio':>6} {'BW saved':>9}")
    out = {"meta": meta, "coverage": args.coverage, "results": []}
    for Lctx in lengths:
        pg, sg = bandwidth(
            cov_win, L, Hkv, D, DT, Lctx, sink=4, global_thresh=args.global_thresh
        )
        print(
            f"{Lctx:>7} | {pg / 1e9:>9.3f} {sg / 1e9:>9.3f} {pg / sg:>5.2f}x {100 * (1 - sg / pg):>7.1f}%"
        )
        out["results"].append(
            {
                "Lctx": Lctx,
                "paged_GB": pg / 1e9,
                "segpaged_GB": sg / 1e9,
                "ratio": pg / sg,
                "saved_pct": 100 * (1 - sg / pg),
            }
        )
    json.dump(out, open("/tmp/redknot_hetero_window_bandwidth.json", "w"), indent=2)
    print("\nSaved /tmp/redknot_hetero_window_bandwidth.json")
    print("\nKey: with REAL heterogeneous per-head windows, Paged's unified block is")
    print("forced to the layer-max window -> small-window heads are read-amplified;")
    print("SegPaged reads each head's true window -> the advantage is much larger")
    print("than under the artificial uniform-window setting.")


if __name__ == "__main__":
    main()
