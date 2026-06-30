#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RedKnot QPS comparison line figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    }
)

INK = "#111111"
GRID = "#8E8E8E"
RECOMP = "#8A8A8A"
REDKNOT = "#D62728"
CACHE = "#4D4D4D"
PROPHET = "#E6614C"

SERIES_STYLE = {
    "CacheBlend (r=15%)": dict(
        color=CACHE, marker="x", linestyle="-", mfc="none", mew=1.3
    ),
    "Recompute": dict(color=RECOMP, marker="^", linestyle="-", mfc="none", mew=1.3),
    "RedKnot (ours)": dict(
        color=REDKNOT, marker="o", linestyle="-", mfc="none", mew=1.3
    ),
    "ProphetKV (r=20%)": dict(
        color=PROPHET, marker="s", linestyle="-", mfc="none", mew=1.3
    ),
}

LEGEND_ORDER = [
    "CacheBlend (r=15%)",
    "Recompute",
    "RedKnot (ours)",
    "ProphetKV (r=20%)",
]

PANELS = [
    {
        "title": "Qwen3-32B (TP=2)",
        "x": [8, 16, 32],
        "xticks": [8, 16, 32],
        "ylim": (0.15, 4.0),
        "series": {
            "Recompute": [0.95, 0.55, 0.22],
            "CacheBlend (r=15%)": [2.85, 1.35, 0.62],
            "ProphetKV (r=20%)": [2.25, 1.10, 0.50],
            "RedKnot (ours)": [1.25, 0.78, 0.42],
        },
    },
    {
        "title": "Llama-3.3-70B-Instruct (TP=4)",
        "x": [16, 64, 128],
        "xticks": [16, 64, 128],
        "ylim": (0.018, 0.40),
        "series": {
            "Recompute": [0.13, 0.055, 0.023],
            "CacheBlend (r=15%)": [0.30, 0.085, 0.055],
            "ProphetKV (r=20%)": [0.24, 0.075, 0.045],
            "RedKnot (ours)": [0.16, 0.065, 0.035],
        },
    },
    {
        "title": "Qwen3.5-397B-A17B (TP=8)",
        "x": [16, 32, 64],
        "xticks": [16, 32, 64],
        "ylim": (0.16, 3.6),
        "series": {
            "Recompute": [0.78, 0.38, 0.22],
            "CacheBlend (r=15%)": [2.60, 1.00, 0.48],
            "ProphetKV (r=20%)": [2.00, 0.78, 0.42],
            "RedKnot (ours)": [1.55, 0.68, 0.30],
        },
    },
    {
        "title": "DeepSeek-V4-Flash (PP=8)",
        "x": [16, 32, 64, 128],
        "xticks": [16, 32, 64, 128],
        "ylim": (0.012, 0.45),
        "series": {
            "Recompute": [0.20, 0.10, 0.040, 0.016],
            "RedKnot (ours)": [0.32, 0.18, 0.080, 0.042],
        },
    },
]


def style_axis(ax, show_xlabel=True):
    ax.set_facecolor("white")
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.3)
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle=":",
        linewidth=0.70,
        color=GRID,
        alpha=0.90,
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle=":",
        linewidth=0.45,
        color=GRID,
        alpha=0.45,
    )
    ax.tick_params(
        axis="both", which="major", labelsize=11.5, width=1.1, length=4.0, color=INK
    )
    ax.tick_params(axis="y", which="minor", width=0.8, length=2.5, color=INK)
    if show_xlabel:
        ax.set_xlabel("Context Length", fontsize=13.0, fontweight="bold")


fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.6), sharey=False)
axes = axes.ravel()

for idx, (ax, panel) in enumerate(zip(axes, PANELS)):
    for name, y in panel["series"].items():
        st = SERIES_STYLE[name]
        ax.plot(
            panel["x"],
            y,
            label=name,
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.7,
            marker=st["marker"],
            markersize=6.8,
            markerfacecolor=st["mfc"],
            markeredgecolor=st["color"],
            markeredgewidth=st["mew"],
            zorder=3,
        )
    ax.set_title(panel["title"], fontsize=13.2, pad=8)
    ax.set_yscale("log")
    ax.set_ylim(*panel["ylim"])
    ax.set_xticks(panel["xticks"])
    ax.set_xticklabels([f"{x}K" for x in panel["xticks"]])
    style_axis(ax, show_xlabel=True)

fig.text(
    0.018,
    0.53,
    "Avg. QPS / GPU",
    rotation=90,
    ha="center",
    va="center",
    fontsize=14.0,
    fontweight="bold",
)
handles, labels = axes[0].get_legend_handles_labels()
handle_map = dict(zip(labels, handles))
handles = [handle_map[name] for name in LEGEND_ORDER]
labels = LEGEND_ORDER
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.50, 1.040),
    ncol=4,
    fontsize=12.0,
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    edgecolor=INK,
    facecolor="white",
    handlelength=2.6,
    columnspacing=1.5,
    borderpad=0.38,
)
fig.subplots_adjust(left=0.055, right=0.995, top=0.760, bottom=0.185, wspace=0.30)
fig.savefig("qps_comparison.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("qps_comparison.png", dpi=260, bbox_inches="tight", pad_inches=0.03)
fig.savefig("qps__comparison.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("qps__comparison.png", dpi=260, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved qps_comparison.pdf/png and qps__comparison.pdf/png")
