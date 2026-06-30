#!/usr/bin/env python3
"""Iso-throughput memory comparison: LRU vs store-everything.

Question: to reach the same throughput (= same cache hit rate = same prefills
avoided), how much KV memory does LRU need vs storing all offline KV?

Method
------
1. STORE-ALL baseline: cache every unique chunk forever. Its memory is the sum
   of all unique chunk KV; its hit rate is the theoretical maximum (every
   non-first access hits).
2. LRU: sweep many byte budgets, record (budget -> hit rate).
3. For target throughput levels (fractions of the store-all hit rate), find the
   minimum LRU budget that reaches that hit rate (interpolated).
4. Report memory-saving factor = store_all_mem / lru_mem at iso-throughput.
"""

from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LB = "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data"
ROOT = "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets"

DATASETS = {
    "musique_ans": (f"{ROOT}/musique_ans_v1.0_dev.jsonl", "musique_ans"),
    "synth_zipf0.8": (f"{HERE}/figures/synth_zipf0.8.jsonl", "musique_ans"),
    "synth_zipf1.1": (f"{HERE}/figures/synth_zipf1.1.jsonl", "musique_ans"),
    "synth_zipf1.5": (f"{HERE}/figures/synth_zipf1.5.jsonl", "musique_ans"),
}


def _load(modfile):
    spec = importlib.util.spec_from_file_location(modfile.stem, str(modfile))
    m = importlib.util.module_from_spec(spec)
    sys.modules[modfile.stem] = m
    spec.loader.exec_module(m)
    return m


def interp_budget_for_hit(curve, target_hit):
    """curve = sorted list of (budget_gb, hit). Return min budget reaching target."""
    for i in range(len(curve)):
        b, h = curve[i]
        if h >= target_hit:
            if i == 0:
                return b
            b0, h0 = curve[i - 1]
            if h == h0:
                return b
            # linear interpolation in budget vs hit
            frac = (target_hit - h0) / (h - h0)
            return b0 + frac * (b - b0)
    return None  # never reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-bytes-per-token", type=float, default=49536)
    ap.add_argument("--chunk-tokens", type=int, default=600)
    ap.add_argument(
        "--targets",
        default="0.5,0.8,0.9,0.95,0.99",
        help="fractions of store-all hit rate to match",
    )
    ap.add_argument("--out", default=str(HERE / "figures/mem_iso_throughput.json"))
    args = ap.parse_args()

    cl = _load(HERE / "chunk_lifecycle.py")
    mgr = _load(HERE / "kv_cache_lifecycle.py")
    kv_per_chunk = int(args.chunk_tokens * args.kv_bytes_per_token)
    targets = [float(x) for x in args.targets.split(",")]

    # budget grid (GB) up to near store-all so high targets resolve
    grid = [
        0.02,
        0.05,
        0.1,
        0.15,
        0.25,
        0.4,
        0.6,
        0.8,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        24.0,
        32.0,
        48.0,
        64.0,
        96.0,
        128.0,
        192.0,
        256.0,
        384.0,
        512.0,
    ]

    out = {}
    for name, (path, fmt) in DATASETS.items():
        if not Path(path).exists():
            print(f"skip {name}")
            continue
        stream = cl.load_stream(fmt, path)
        accesses = []
        for rid, cids, texts in stream:
            for cid in cids:
                accesses.append((cid, args.chunk_tokens, kv_per_chunk))

        # store-all: memory = unique chunks * kv_per_chunk; hit = max
        ca = mgr.CacheAll()
        for cid, n, kvb in accesses:
            ca.access(cid, n, kvb)
        ca_s = ca.stats()
        store_all_gb = ca_s["peak_kv_gb"]
        store_all_hit = ca_s["hit_rate"]

        # LRU curve
        curve = []
        for b in grid:
            lru = mgr.LRU(capacity_bytes=int(b * 1e9))
            for cid, n, kvb in accesses:
                lru.access(cid, n, kvb)
            curve.append((b, lru.stats()["hit_rate"]))

        rows = []
        for frac in targets:
            tgt_hit = store_all_hit * frac
            lru_gb = interp_budget_for_hit(curve, tgt_hit)
            if lru_gb is None:
                rows.append(
                    dict(
                        target_frac=frac,
                        target_hit=round(tgt_hit, 4),
                        lru_gb=None,
                        saving="n/a",
                    )
                )
            else:
                rows.append(
                    dict(
                        target_frac=frac,
                        target_hit=round(tgt_hit, 4),
                        lru_gb=round(lru_gb, 3),
                        saving=round(store_all_gb / lru_gb, 1),
                    )
                )
        out[name] = dict(
            store_all_gb=store_all_gb,
            store_all_hit=store_all_hit,
            iso_throughput=rows,
            lru_curve=curve,
        )

        print(f"\n=== {name} ===")
        print(f"  store-all: {store_all_gb:.2f} GB  (max hit {store_all_hit:.3f})")
        print(f"  {'target':>8} {'hit':>7} {'LRU_GB':>8} {'mem_saving':>11}")
        for r in rows:
            sv = f"{r['saving']}x" if r["lru_gb"] else "n/a"
            lg = f"{r['lru_gb']:.2f}" if r["lru_gb"] else "n/a"
            print(
                f"  {int(r['target_frac'] * 100):>6}% {r['target_hit']:>7.3f} {lg:>8} {sv:>11}"
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
