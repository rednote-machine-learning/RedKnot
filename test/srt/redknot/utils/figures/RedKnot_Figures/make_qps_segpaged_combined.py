#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combined QPS and SegPagedAttention line figures."""

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
        "axes.labelweight": "bold",
    }
)

INK = "#111111"
GRID = "#8E8E8E"
GRAY = "#8A8A8A"
BLACK = "#4D4D4D"
RED = "#D62728"
ORANGE = "#E6614C"
GOLD = "#D7A42A"

QPS_STYLE = {
    "CacheBlend (r=15%)": dict(color=BLACK, marker="x", mfc="none", mew=1.3),
    "Recompute": dict(color=GRAY, marker="^", mfc="none", mew=1.3),
    "RedKnot (ours)": dict(color=RED, marker="o", mfc="none", mew=1.3),
    "ProphetKV (r=20%)": dict(color=GOLD, marker="s", mfc="none", mew=1.3),
}

SEG_STYLE = {
    "FA (no mask)": dict(color=BLACK, marker="x", mfc="none", mew=1.3),
    "Efficient (mask)": dict(color=RED, marker="o", mfc="none", mew=1.3),
    "Dense (no mask)": dict(color=GRAY, marker="^", mfc="none", mew=1.3),
    "Dense+mask": dict(color=GOLD, marker="s", mfc="none", mew=1.3),
    "SegPaged": dict(color=RED, marker="D", mfc="none", mew=1.3),
    "Manual": dict(color=BLACK, marker="P", mfc="none", mew=1.3),
    "SDPA+mask": dict(color=ORANGE, marker="v", mfc="none", mew=1.3),
    "Fused": dict(color=ORANGE, marker="s", mfc="none", mew=1.3),
}

QPS_ORDER = ["CacheBlend (r=15%)", "Recompute", "RedKnot (ours)", "ProphetKV (r=20%)"]
SEG_ORDER = [
    "FA (no mask)",
    "Efficient (mask)",
    "Dense (no mask)",
    "Dense+mask",
    "SegPaged",
    "Manual",
    "SDPA+mask",
    "Fused",
]

QPS_PANELS = [
    (
        "(a) Qwen3-32B (TP=2)",
        [8, 16, 32],
        (0.15, 4.0),
        {
            "Recompute": [0.95, 0.55, 0.22],
            "CacheBlend (r=15%)": [2.85, 1.35, 0.62],
            "ProphetKV (r=20%)": [2.25, 1.10, 0.50],
            "RedKnot (ours)": [1.25, 0.78, 0.42],
        },
    ),
    (
        "(b) Llama-3.3-70B (TP=4)",
        [16, 64, 128],
        (0.018, 0.40),
        {
            "Recompute": [0.13, 0.055, 0.023],
            "CacheBlend (r=15%)": [0.30, 0.085, 0.055],
            "ProphetKV (r=20%)": [0.24, 0.075, 0.045],
            "RedKnot (ours)": [0.16, 0.065, 0.035],
        },
    ),
    (
        "(c) Qwen3.5-397B (TP=8)",
        [16, 32, 64],
        (0.16, 3.6),
        {
            "Recompute": [0.78, 0.38, 0.22],
            "CacheBlend (r=15%)": [2.60, 1.00, 0.48],
            "ProphetKV (r=20%)": [2.00, 0.78, 0.42],
            "RedKnot (ours)": [1.55, 0.68, 0.30],
        },
    ),
    (
        "(d) DeepSeek-V4-Flash (PP=8)",
        [16, 32, 64, 128],
        (0.012, 0.45),
        {
            "Recompute": [0.20, 0.10, 0.040, 0.016],
            "RedKnot (ours)": [0.32, 0.18, 0.080, 0.042],
        },
    ),
]

SEG_PANELS = [
    (
        "(e) Single-layer Decode",
        [8, 32, 64, 128],
        (0, 5.8),
        False,
        {
            "FA (no mask)": [0.07, 0.20, 0.38, 0.71],
            "Efficient (mask)": [0.35, 1.35, 2.75, 5.24],
        },
    ),
    (
        "(f) 64-layer Decode",
        [8, 32, 64, 128],
        (8, 650),
        True,
        {
            "Dense (no mask)": [13.2, 47.9, 101.0, 218.2],
            "Dense+mask": [31.8, 122.3, 250.0, 506.8],
            "SegPaged": [12.6, 12.4, 16.3, 24.1],
        },
    ),
    (
        "(g) Decode Latency",
        [8, 32, 64, 128],
        (6, 700),
        True,
        {
            "Manual": [45, 88, 190, 520],
            "SDPA+mask": [22, 42, 90, 230],
            "Fused": [8, 18, 45, 156],
        },
    ),
    (
        "(h) Prefill Latency",
        [8, 32, 64, 128],
        (0.08, 7.5),
        True,
        {
            "Manual": [0.66, 2.7, 3.7, 5.4],
            "SDPA+mask": [0.30, 1.10, 0.72, 0.40],
            "Fused": [0.12, 0.28, 0.25, 0.23],
        },
    ),
]


def style_axis(ax, ylabel=None):
    ax.set_facecolor("white")
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.25)
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
        axis="both", which="major", labelsize=10.5, width=1.0, length=3.8, color=INK
    )
    ax.tick_params(axis="y", which="minor", width=0.8, length=2.4, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12.6, fontweight="bold")
    ax.set_xlabel("Context Length", fontsize=12.0, fontweight="bold", labelpad=5)


def caption_below(ax, text):
    ax.text(
        0.5,
        -0.37,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.2,
        fontweight="bold",
    )


def plot_lines(ax, x, series, styles):
    for name, y in series.items():
        st = styles[name]
        ax.plot(
            x,
            y,
            label=name,
            color=st["color"],
            linestyle="-",
            linewidth=1.7,
            marker=st["marker"],
            markersize=6.6,
            markerfacecolor=st["mfc"],
            markeredgecolor=st["color"],
            markeredgewidth=st["mew"],
            zorder=3,
        )


def add_legend(
    fig, axes, order, y, ncol, fontsize=10.2, handlelength=2.35, columnspacing=1.15
):
    seen = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        seen.update(dict(zip(labels, handles)))
    handles = [seen[name] for name in order if name in seen]
    labels = [name for name in order if name in seen]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor=INK,
        facecolor="white",
        handlelength=handlelength,
        columnspacing=columnspacing,
        borderpad=0.34,
    )


fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.4), sharey=False)

for i, (caption, x, ylim, series) in enumerate(QPS_PANELS):
    ax = axes[0, i]
    plot_lines(ax, x, series, QPS_STYLE)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}K" for v in x])
    style_axis(ax, ylabel="Avg. QPS / GPU" if i == 0 else None)
    caption_below(ax, caption)

for i, (caption, x, ylim, log, series) in enumerate(SEG_PANELS):
    ax = axes[1, i]
    plot_lines(ax, x, series, SEG_STYLE)
    if log:
        ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}K" for v in x])
    ylabel = "Latency (ms)" if i == 0 else ("Latency (s)" if i == 3 else None)
    style_axis(ax, ylabel=ylabel)
    caption_below(ax, caption)

add_legend(fig, axes[0, :], QPS_ORDER, y=0.965, ncol=4)
add_legend(
    fig,
    axes[1, :],
    SEG_ORDER,
    y=0.455,
    ncol=8,
    fontsize=9.8,
    handlelength=2.0,
    columnspacing=0.95,
)

fig.subplots_adjust(
    left=0.055, right=0.995, top=0.88, bottom=0.095, wspace=0.30, hspace=0.90
)
fig.savefig("qps_segpaged_combined.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("qps_segpaged_combined.png", dpi=260, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved qps_segpaged_combined.pdf/png")
