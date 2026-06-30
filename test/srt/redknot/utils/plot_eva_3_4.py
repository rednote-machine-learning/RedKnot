#!/usr/bin/env python3
"""RedKnot evaluation figures 3 & 4 as two stacked bar charts.

Redrawn from Eva_3.pdf / Eva_4.pdf with a single unified professional palette
(the Eva_4 gray / blue / red), consistent format, and value labels.

Eva_3 (top): single-layer decode latency (ms), FA(no mask) vs Efficient(mask).
Eva_4 (bottom): 64-layer decode latency (ms), Dense / Dense+mask / SegPaged.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures/eva_3_4.png"

# Unified professional palette (from Eva_4)
GRAY = "#9b9b9b"
BLUE = "#6a92c0"
RED = "#c0584b"
EDGE = "#333333"

CTX = ["8K", "32K", "128K"]

# --- Eva_3 data (single-layer decode latency, ms) ---
E3 = {
    "FA (no mask)": [0.07, 0.20, 0.71],
    "Efficient (mask)": [0.35, 1.35, 5.24],
}
E3_COLORS = [BLUE, RED]

# --- Eva_4 data (64-layer decode latency, ms) ---
E4 = {
    "Dense (no mask)": [13.2, 47.9, 218.2],
    "Dense+mask": [31.8, 122.3, 506.8],
    "SegPaged": [12.6, 12.4, 24.1],
}
E4_COLORS = [GRAY, BLUE, RED]


def _grouped(ax, series, colors, ylabel, fmt, tag):
    keys = list(series.keys())
    n = len(keys)
    x = np.arange(len(CTX))
    w = 0.8 / n
    for i, (k, c) in enumerate(zip(keys, colors)):
        off = (i - (n - 1) / 2) * w
        bars = ax.bar(
            x + off, series[k], w, color=c, edgecolor=EDGE, linewidth=0.6, label=k
        )
        for r, v in zip(bars, series[k]):
            ax.annotate(
                fmt.format(v),
                (r.get_x() + r.get_width() / 2, r.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#222222",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(CTX, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel(f"{tag} Context length", fontsize=10, fontweight="bold")
    # close the box: keep all four spines
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
    ax.grid(True, axis="y", alpha=0.25, ls="--", lw=0.4)
    top = max(max(v) for v in series.values())
    ax.set_ylim(0, top * 1.25)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)


def main():
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "axes.linewidth": 0.7, "axes.edgecolor": EDGE}
    )

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 5.6))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.09, hspace=0.42)

    _grouped(
        axes[0], E3, E3_COLORS, "Single-layer decode\nlatency (ms)", "{:.2f}", "(a)"
    )
    _grouped(axes[1], E4, E4_COLORS, "64-layer decode\nlatency (ms)", "{:.1f}", "(b)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    print(f"[plot] saved {OUT} and .pdf")


if __name__ == "__main__":
    main()
