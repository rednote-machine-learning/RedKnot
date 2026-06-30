#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RedKnot evaluation Figure Eva_1."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

RED = "#D95A48"
BLUE = "#B8B8B8"
ORANGE = "#F0A17F"
WARM_GRAY = "#DED2C8"
GRAY = "#DADADA"
LIGHT = "#FFFFFF"
INK = "#111111"
EDGE = "#111111"
GRID = "#9B9B9B"

METHOD_COLORS = {
    "recompute": GRAY,
    "RedKnot": RED,
    "CacheBlend": BLUE,
    "ProphetKV": ORANGE,
}


def style_ax(ax, ylim=None, ylabel=None):
    ax.set_facecolor(LIGHT)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(EDGE)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=9.8, colors=INK, width=1.0, length=3.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.65, linestyle=":")
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.0, color=INK, fontweight="bold")


def grouped_bars(
    ax, labels, series, colors, ylim, ylabel=None, rotate=22, value_fmt="{:.2g}"
):
    n = len(labels)
    m = len(series)
    x = np.arange(n)
    width = 0.24 if m == 2 else min(0.80 / m, 0.21)
    offset_step = 0.32 if m == 2 else width
    offsets = (np.arange(m) - (m - 1) / 2) * offset_step
    for i, (name, vals) in enumerate(series.items()):
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width,
            color=colors[name],
            edgecolor=EDGE,
            linewidth=0.65,
            label=name,
            zorder=3,
        )
        for b, v in zip(bars, vals):
            if v <= 0:
                continue
            label_y = b.get_height() + (ylim[1] - ylim[0]) * (
                0.040 + (0.030 if m == 2 and i % 2 else 0)
            )
            label_x = b.get_x() + b.get_width() / 2
            if m == 2:
                label_x += -0.025 if i == 0 else 0.025
            elif m > 2:
                label_x += (i - (m - 1) / 2) * 0.012
            ax.text(
                label_x,
                label_y,
                value_fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=5.9 if m == 2 else 5.0,
                color=INK,
                zorder=5,
            )
    style_ax(ax, ylim=ylim, ylabel=ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, rotation=rotate, ha="right" if rotate else "center", fontsize=9.6
    )
    return ax


def section_label(fig, y, text):
    fig.text(
        0.36,
        y,
        text,
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color=INK,
    )


fig, axes = plt.subplots(3, 4, figsize=(13.8, 8.3))
fig.patch.set_facecolor("white")

# Row 1: multi-model comparison.
cats = ["M-TQA-16K", "Q-MFQA-24K", "L70-HQA-32K", "L70-HQA-64K"]
row1_methods = ["recompute", "RedKnot", "CacheBlend", "ProphetKV"]
row1_colors = {m: METHOD_COLORS[m] for m in row1_methods}

grouped_bars(
    axes[0, 0],
    cats,
    {
        "recompute": [1.0, 1.0, 1.0, 1.0],
        "RedKnot": [1.60, 2.91, 2.93, 3.52],
        "CacheBlend": [1.80, 2.10, 2.00, 2.40],
        "ProphetKV": [1.40, 1.70, 1.70, 2.00],
    },
    row1_colors,
    ylim=(0, 4.2),
    ylabel="Speedup (x)",
    value_fmt="{:.2g}",
)
axes[0, 0].set_title("(a) TTFT Speedup", fontsize=11.2)

grouped_bars(
    axes[0, 1],
    cats,
    {
        "recompute": [1.0, 0.0, 0.6, 0.6],
        "RedKnot": [1.0, 0.4, 0.8, 0.8],
        "CacheBlend": [1.0, 0.0, 0.6, 0.4],
        "ProphetKV": [1.0, 0.0, 0.6, 0.6],
    },
    row1_colors,
    ylim=(0, 1.25),
    ylabel="EM",
    value_fmt="{:.1f}",
)
axes[0, 1].set_title("(b) Exact-Match Accuracy", fontsize=11.2)

grouped_bars(
    axes[0, 2],
    cats,
    {
        "recompute": [0.8, 0.6, 0.3, 0.3],
        "RedKnot": [1.0, 0.52, 0.3, 0.2],
        "CacheBlend": [0.8, 0.2, 0.3, 0.3],
        "ProphetKV": [0.8, 0.1, 0.3, 0.3],
    },
    row1_colors,
    ylim=(0, 1.25),
    ylabel="F1",
    value_fmt="{:.2g}",
)
axes[0, 2].set_title("(c) Token-Level F1", fontsize=11.2)

grouped_bars(
    axes[0, 3],
    ["top-1", "top-10"],
    {
        "recompute": [1.0, 1.0],
        "RedKnot": [0.93, 0.87],
        "CacheBlend": [0.40, 0.40],
        "ProphetKV": [0.50, 0.40],
    },
    row1_colors,
    ylim=(0, 1.25),
    ylabel="Match",
    rotate=0,
    value_fmt="{:.2g}",
)
axes[0, 3].set_title("(d) Top-K Match", fontsize=11.2)

# Row 2: Qwen3.5-397B-A17B.
two_colors = {"recomputed F1": GRAY, "RedKnot F1": RED}
row3_colors = {"recomputed": WARM_GRAY, "RedKnot": ORANGE}
grouped_bars(
    axes[1, 0],
    ["MFQA", "HQA", "2Wiki", "MuSiQue", "TriviaQA"],
    {
        "recomputed F1": [0.41, 0.25, 0.62, 0.28, 0.45],
        "RedKnot F1": [0.38, 0.20, 0.53, 0.31, 0.30],
    },
    two_colors,
    ylim=(0, 0.95),
    ylabel="F1",
    value_fmt="{:.2f}",
)
axes[1, 0].set_title("(e) 16K", fontsize=11.2)

grouped_bars(
    axes[1, 1],
    ["MFQA", "HQA", "2Wiki", "MuSiQue", "NarrQA"],
    {
        "recomputed F1": [0.52, 0.45, 0.62, 0.31, 0.17],
        "RedKnot F1": [0.53, 0.30, 0.68, 0.26, 0.17],
    },
    two_colors,
    ylim=(0, 0.95),
    ylabel="F1",
    value_fmt="{:.2f}",
)
axes[1, 1].set_title("(f) 32K", fontsize=11.2)

grouped_bars(
    axes[1, 2],
    ["NarrQA", "HQA", "2Wiki", "MuSiQue", "MFQA"],
    {
        "recomputed F1": [0.14, 0.25, 0.60, 0.26, 0.32],
        "RedKnot F1": [0.12, 0.25, 0.62, 0.17, 0.36],
    },
    two_colors,
    ylim=(0, 0.95),
    ylabel="F1",
    value_fmt="{:.2f}",
)
axes[1, 2].set_title("(g) 64K", fontsize=11.2)

axes[1, 3].bar(
    [0, 1, 2],
    [2.05, 2.17, 2.30],
    color=RED,
    edgecolor=EDGE,
    linewidth=0.8,
    width=0.55,
    zorder=3,
)
for x, v in enumerate([2.05, 2.17, 2.30]):
    axes[1, 3].text(x, v + 0.08, f"{v:.2f}x", ha="center", fontsize=8.0, color=INK)
style_ax(axes[1, 3], ylim=(0, 2.8), ylabel="TTFT speedup (x)")
axes[1, 3].set_xticks([0, 1, 2])
axes[1, 3].set_xticklabels(["16K", "32K", "64K"])
axes[1, 3].set_title("(h) RedKnot TTFT vs. Length", fontsize=11.2)


# Row 3: DeepSeek-V4-Flash.
def dsv4_panel(ax, title, ttft, rec, red):
    grouped_bars(
        ax,
        ["HQA\n(F1)", "2Wiki\n(EM)", "MuSiQue\n(EM)", "TriviaQA\n(EM)"],
        {"recomputed": rec, "RedKnot": red},
        row3_colors,
        ylim=(0, 1.05),
        ylabel="Accuracy",
        rotate=0,
        value_fmt="{:.2f}",
    )
    ax.set_title(title, fontsize=11.2)
    ax.text(
        0.03,
        0.94,
        f"TTFT {ttft:.2f}x",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
        bbox=dict(boxstyle="round,pad=0.20", fc="#FFF8E8", ec=EDGE, lw=0.8),
        zorder=8,
    )


dsv4_panel(
    axes[2, 0], "(i) 16K", 3.51, [0.67, 0.68, 0.44, 0.76], [0.67, 0.53, 0.38, 0.76]
)
dsv4_panel(
    axes[2, 1], "(j) 32K", 5.01, [0.64, 0.56, 0.44, 0.88], [0.64, 0.43, 0.38, 0.88]
)
dsv4_panel(
    axes[2, 2], "(k) 64K", 5.05, [0.66, 0.56, 0.44, 0.88], [0.66, 0.43, 0.38, 0.88]
)
dsv4_panel(
    axes[2, 3], "(l) 128K", 5.16, [0.63, 0.52, 0.44, 0.80], [0.59, 0.43, 0.29, 0.80]
)

# Row-level titles and legends.
section_label(fig, 0.985, "Mistral, Qwen3, Llama-3.3")
section_label(fig, 0.642, "Qwen3.5-397B-A17B")
section_label(fig, 0.322, "DeepSeek-V4-Flash (FP8)")

fig.legend(
    handles=[
        Patch(facecolor=METHOD_COLORS[n], edgecolor=EDGE, label=n) for n in row1_methods
    ],
    loc="center left",
    bbox_to_anchor=(0.50, 0.985),
    ncol=4,
    frameon=True,
    edgecolor=EDGE,
    facecolor="white",
    framealpha=1.0,
    fontsize=9.8,
    handlelength=1.0,
    columnspacing=1.2,
)
fig.legend(
    handles=[
        Patch(facecolor=GRAY, edgecolor=EDGE, label="recomputed"),
        Patch(facecolor=RED, edgecolor=EDGE, label="RedKnot"),
    ],
    loc="center left",
    bbox_to_anchor=(0.50, 0.642),
    ncol=2,
    frameon=True,
    edgecolor=EDGE,
    facecolor="white",
    framealpha=1.0,
    fontsize=9.8,
    handlelength=1.3,
    columnspacing=1.0,
)
fig.legend(
    handles=[
        Patch(facecolor=WARM_GRAY, edgecolor=EDGE, label="recomputed"),
        Patch(facecolor=ORANGE, edgecolor=EDGE, label="RedKnot"),
    ],
    loc="center left",
    bbox_to_anchor=(0.50, 0.322),
    ncol=2,
    frameon=True,
    edgecolor=EDGE,
    facecolor="white",
    framealpha=1.0,
    fontsize=9.8,
    handlelength=1.3,
    columnspacing=1.0,
)

fig.subplots_adjust(
    left=0.055, right=0.995, top=0.925, bottom=0.065, wspace=0.30, hspace=0.82
)
fig.savefig("Eva_1.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("Eva_1.png", dpi=260, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved Eva_1.pdf + Eva_1.png")
