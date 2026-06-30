#!/usr/bin/env python3
"""Plot multi-dataset LRU KV-cache results."""

from __future__ import annotations
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", default=str(HERE / "figures/replay_lru_multi.json")
    )
    ap.add_argument("--out", default=str(HERE / "figures/lru_multi.png"))
    args = ap.parse_args()

    data = json.load(open(args.inp))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # keep datasets with non-trivial reuse
    names = [n for n in data if data[n]["n_accesses"] >= 1000]
    colors = plt.cm.viridis([i / max(len(names) - 1, 1) for i in range(len(names))])

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))

    # (1) hit rate vs budget
    for n, c in zip(names, colors):
        rows = data[n]["lru_by_budget"]
        b = [r["budget_gb"] for r in rows]
        h = [r["hit_rate"] for r in rows]
        lw = 2.2 if n == "musique_ans" else 1.4
        ax[0].plot(b, h, "o-", color=c, lw=lw, ms=4, label=n)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("KV budget (GB)")
    ax[0].set_ylabel("LRU hit rate")
    ax[0].set_title("LRU hit rate vs budget")
    ax[0].grid(True, alpha=0.3, which="both")
    ax[0].legend(fontsize=7)

    # (2) non-prefix ratio bar (why prefix-cache fails)
    npf = [data[n]["mean_non_prefix"] for n in names]
    ax[1].bar(range(len(names)), npf, color=colors, edgecolor="white")
    ax[1].axhline(0.9, color="#d7191c", ls="--", lw=0.8)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax[1].set_ylabel("mean non-prefix ratio")
    ax[1].set_title("Non-prefix reuse\n(prefix-cache cannot hit)")
    ax[1].set_ylim(0, 1)

    # (3) GPU-seconds saved vs budget
    for n, c in zip(names, colors):
        rows = data[n]["lru_by_budget"]
        b = [r["budget_gb"] for r in rows]
        g = [r["gpu_seconds_saved"] for r in rows]
        ax[2].plot(b, g, "s-", color=c, lw=1.4, ms=4, label=n)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("KV budget (GB)")
    ax[2].set_ylabel("GPU-seconds saved")
    ax[2].set_title("Recompute saved (DeepSeek V4 MLA)")
    ax[2].grid(True, alpha=0.3, which="both")
    ax[2].legend(fontsize=7)

    fig.suptitle(
        "LRU KV-cache on non-prefix RAG reuse across datasets / skew", fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
