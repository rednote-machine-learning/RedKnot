#!/usr/bin/env python3
"""Sweep KV-cache budget and plot hit-rate / efficiency for all policies.

Shows the key finding: the 3-layer admission+score+TTL policy wins under
memory scarcity (realistic for 100B+ models) and converges to LRU under
abundance.
"""

from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(modfile):
    spec = importlib.util.spec_from_file_location(modfile.stem, str(modfile))
    m = importlib.util.module_from_spec(spec)
    sys.modules[modfile.stem] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default="/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/musique_ans_v1.0_dev.jsonl",
    )
    ap.add_argument("--kv-bytes-per-token", type=float, default=49536)
    ap.add_argument("--chunk-tokens", type=int, default=600)
    ap.add_argument("--budgets", default="0.05,0.1,0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--out", default=str(HERE / "figures/kv_lifecycle_sweep.png"))
    args = ap.parse_args()

    cl = _load(HERE / "chunk_lifecycle.py")
    mgr = _load(HERE / "kv_cache_lifecycle.py")
    stream = cl.load_musique_stream(args.path, None)

    accesses = []
    for rid, chunk_ids, texts in stream:
        for cid in chunk_ids:
            accesses.append(
                (
                    cid,
                    args.chunk_tokens,
                    int(args.chunk_tokens * args.kv_bytes_per_token),
                )
            )

    budgets = [float(x) for x in args.budgets.split(",")]
    series = {"cache_all": [], "lru": [], "lfu": [], "redknot_3layer": []}
    peak_all = None
    for b in budgets:
        budget = int(b * 1e9)
        pols = {
            "lru": mgr.LRU(capacity_bytes=budget),
            "lfu": mgr.LFU(capacity_bytes=budget),
            "redknot_3layer": mgr.KVCacheLifecycleManager(
                capacity_bytes=budget, r_admit=2
            ),
        }
        for name, pol in pols.items():
            for cid, n, kvb in accesses:
                pol.access(cid, n, kvb, carry_tokens=0)
            series[name].append(pol.stats()["hit_rate"])
        # cache_all is budget-independent
        if peak_all is None:
            ca = mgr.CacheAll()
            for cid, n, kvb in accesses:
                ca.access(cid, n, kvb, carry_tokens=0)
            st = ca.stats()
            peak_all = st["peak_kv_gb"]
            ca_hit = st["hit_rate"]
        series["cache_all"].append(ca_hit)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    colors = {
        "cache_all": "#7f7f7f",
        "lru": "#2c7fb8",
        "lfu": "#d99900",
        "redknot_3layer": "#d7191c",
    }
    labels = {
        "cache_all": f"cache-all ({peak_all:.0f}GB!)",
        "lru": "LRU",
        "lfu": "LFU",
        "redknot_3layer": "RedKnot 3-layer",
    }
    for name in ["cache_all", "lru", "lfu", "redknot_3layer"]:
        ls = "--" if name == "cache_all" else "-"
        lw = 2.0 if name == "redknot_3layer" else 1.3
        ax[0].plot(
            budgets,
            series[name],
            "o" + ls,
            color=colors[name],
            lw=lw,
            ms=4,
            label=labels[name],
        )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("KV cache budget (GB)")
    ax[0].set_ylabel("hit rate")
    ax[0].set_title("Hit rate vs memory budget")
    ax[0].grid(True, alpha=0.3, which="both")
    ax[0].legend(fontsize=8)

    # relative improvement of redknot over LRU
    impr = [
        100 * (r - l) / max(l, 1e-9)
        for r, l in zip(series["redknot_3layer"], series["lru"])
    ]
    ax[1].axhline(0, color="#999", lw=0.8)
    ax[1].plot(budgets, impr, "s-", color="#d7191c", lw=1.8, ms=5)
    ax[1].fill_between(
        budgets, impr, 0, where=[x > 0 for x in impr], color="#d7191c", alpha=0.15
    )
    ax[1].set_xscale("log")
    ax[1].set_xlabel("KV cache budget (GB)")
    ax[1].set_ylabel("RedKnot hit-rate gain over LRU (%)")
    ax[1].set_title("RedKnot 3-layer advantage\n(wins when memory is scarce)")
    ax[1].grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "KV-cache lifecycle policy on real non-prefix reuse (musique, DeepSeek V4 MLA)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot] saved {out}")
    for b, l, r, i in zip(budgets, series["lru"], series["redknot_3layer"], impr):
        print(f"  budget={b:>5}GB  LRU={l:.3f}  RedKnot={r:.3f}  gain={i:+.1f}%")


if __name__ == "__main__":
    main()
