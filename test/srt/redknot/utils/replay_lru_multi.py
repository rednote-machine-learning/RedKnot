#!/usr/bin/env python3
"""Multi-dataset LRU KV-cache replay.

Drives the LRU policy over several real multi-document reuse streams and
reports, per dataset and per memory budget:
  * non-prefix reuse ratio (why prefix-cache fails)
  * LRU hit rate
  * peak KV memory
  * GPU-seconds saved (DeepSeek V4 Flash MLA cost model)
"""

from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LB = "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data"
ROOT = "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets"

DATASETS = {
    "musique_ans": f"{ROOT}/musique_ans_v1.0_dev.jsonl",
    "2wikimqa": f"{LB}/2wikimqa.jsonl",
    "synth_zipf0.8": f"{HERE}/figures/synth_zipf0.8.jsonl",
    "synth_zipf1.1": f"{HERE}/figures/synth_zipf1.1.jsonl",
    "synth_zipf1.5": f"{HERE}/figures/synth_zipf1.5.jsonl",
}


def _load(modfile):
    spec = importlib.util.spec_from_file_location(modfile.stem, str(modfile))
    m = importlib.util.module_from_spec(spec)
    sys.modules[modfile.stem] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="0.05,0.1,0.25,0.5,1.0,2.0")
    ap.add_argument("--kv-bytes-per-token", type=float, default=49536)
    ap.add_argument("--chunk-tokens", type=int, default=600)
    ap.add_argument("--prefill-ms-per-token", type=float, default=0.066)
    ap.add_argument("--out", default=str(HERE / "figures/replay_lru_multi.json"))
    args = ap.parse_args()

    cl = _load(HERE / "chunk_lifecycle.py")
    mgr = _load(HERE / "kv_cache_lifecycle.py")
    budgets = [float(x) for x in args.budgets.split(",")]
    kv_per_chunk = int(args.chunk_tokens * args.kv_bytes_per_token)

    all_out = {}
    for name, path in DATASETS.items():
        if not Path(path).exists():
            print(f"skip {name}: not found")
            continue
        # synth files use musique paragraphs[] format
        if name == "musique_ans" or name.startswith("synth_"):
            ds_key = "musique_ans"
        else:
            ds_key = name
        stream = cl.load_stream(ds_key, path)
        if not stream:
            print(f"skip {name}: empty stream")
            continue
        # reuse stats
        stats = cl.analyze(stream, None, args.kv_bytes_per_token)
        npf = sum(s["non_prefix_ratio"] for s in stats.values()) / len(stats)
        reused2 = sum(1 for s in stats.values() if s["reuse_count"] >= 2)
        max_reuse = max(s["reuse_count"] for s in stats.values())

        accesses = []
        for rid, cids, texts in stream:
            for cid in cids:
                accesses.append((cid, args.chunk_tokens, kv_per_chunk))

        rows = []
        for b in budgets:
            lru = mgr.LRU(capacity_bytes=int(b * 1e9))
            for cid, n, kvb in accesses:
                lru.access(cid, n, kvb, carry_tokens=0)
            s = lru.stats()
            s["budget_gb"] = b
            s["gpu_seconds_saved"] = round(
                s["reused_tokens"] * args.prefill_ms_per_token / 1000, 1
            )
            rows.append(s)

        all_out[name] = dict(
            n_requests=len(stream),
            n_chunks=len(stats),
            n_accesses=len(accesses),
            reused_ge2=reused2,
            max_reuse=max_reuse,
            mean_non_prefix=round(npf, 3),
            lru_by_budget=rows,
        )
        print(f"\n=== {name} ===")
        print(
            f"  requests={len(stream)} chunks={len(stats)} accesses={len(accesses)} "
            f"reused>=2={reused2} max_reuse={max_reuse} non_prefix={npf:.2f}"
        )
        print(f"  {'budget':>8} {'hit_rate':>8} {'peak_GB':>8} {'GPU_s_saved':>11}")
        for r in rows:
            print(
                f"  {r['budget_gb']:>8} {r['hit_rate']:>8.3f} {r['peak_kv_gb']:>8.2f} {r['gpu_seconds_saved']:>11.1f}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_out, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
