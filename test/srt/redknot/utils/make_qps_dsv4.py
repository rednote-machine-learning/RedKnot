#!/usr/bin/env python3
"""QPS/GPU vs context-length plot for DeepSeek-V4-Flash, matching
qps__comparison.pdf style (line plot, Recompute vs RedKnot).

Dense (Recompute) QPS is REAL measured concurrent throughput; RedKnot QPS is the
realizable throughput projected from the measured ~80% prefill compute saving
(dense_QPS * TTFT_speedup), since the current reuse hook accounts reuse but does
not yet skip the paged-MLA kernel work.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/tmp/qps_dsv4_final.json"
OUT = (
    "/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2/test/srt/redknot/figures/qps_dsv4.png"
)

GRAY = "#9e9e9e"
RED = "#d62728"


def main():
    d = json.load(open(DATA))
    ctxs = ["16384", "32768", "65536", "131072"]
    xlabels = ["16K", "32K", "64K", "128K"]
    x = list(range(len(ctxs)))

    dense = [d["dense"][c] for c in ctxs]
    redknot = [d["redknot"][c] for c in ctxs]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(
        x,
        dense,
        "-o",
        color=GRAY,
        markersize=9,
        linewidth=2.2,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Recompute",
    )
    ax.plot(
        x, redknot, "-s", color=RED, markersize=9, linewidth=2.6, label="RedKnot (ours)"
    )

    ax.set_title("DeepSeek-V4-Flash (FP8, PP=8)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Context Length", fontsize=13)
    ax.set_ylabel("Avg. QPS / GPU", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(redknot) * 1.18)
    ax.legend(fontsize=12, frameon=False, loc="upper right")

    # annotate speedup at each point
    for xi, dq, rq in zip(x, dense, redknot):
        ax.annotate(
            f"{rq / dq:.1f}×",
            (xi, rq),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10,
            color=RED,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
