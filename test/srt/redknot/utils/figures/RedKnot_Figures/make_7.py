#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RedKnot Figure 7 bar charts."""

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
        "axes.labelweight": "bold",
    }
)

INK = "#111111"
GRID = "#9B9B9B"
GRAY = "#DADADA"
MID_GRAY = "#B8B8B8"
RED = "#D95A48"
ORANGE = "#F0A17F"


def style_ax(ax, ylabel=None, ylim=None):
    ax.set_facecolor("white")
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.25)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.70, color=GRID, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9.2, colors=INK, width=1.0, length=3.6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.6, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)


def caption_below(ax, text):
    ax.text(
        0.5,
        -0.48,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )


def label_bars(ax, bars, fmt="{:.1f}", dy=0.02, size=8.0):
    y0, y1 = ax.get_ylim()
    off = (y1 - y0) * dy
    for b in bars:
        v = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + off,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=size,
            color=INK,
        )


fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.45))

# (a) KV-cache transfer saving.
ax = axes[0]
labels = ["Llama\n8K", "Llama\n16K", "Qwen3\n8K", "Qwen3\n16K", "Qwen3\n24K"]
x = np.arange(len(labels))
w = 0.34
kv = [4.3, 5.7, 5.6, 6.1, 6.3]
transfer = [4.1, 4.1, 1.5, 1.4, 1.0]
b1 = ax.bar(
    x - w / 2,
    kv,
    w,
    color=MID_GRAY,
    edgecolor=INK,
    linewidth=0.75,
    label="KV bytes",
    zorder=3,
)
b2 = ax.bar(
    x + w / 2,
    transfer,
    w,
    color=ORANGE,
    edgecolor=INK,
    linewidth=0.75,
    label="Transfer time",
    zorder=3,
)
style_ax(ax, ylabel="Saving over dense (x)", ylim=(0, 7.3))
ax.set_xticks(x)
ax.set_xticklabels(labels)
label_bars(ax, b1, size=7.8)
label_bars(ax, b2, size=7.8)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.52, 1.30),
    ncol=2,
    frameon=True,
    fancybox=False,
    edgecolor=INK,
    facecolor="white",
    fontsize=8.8,
    handlelength=0.9,
    columnspacing=0.9,
)
caption_below(ax, "(a) KV-cache Transfer Saving")

# (b) Burst-mode throughput.
ax = axes[1]
labels = [
    "Llama\n8K\nN=2",
    "Llama\n16K\nN=2",
    "Qwen3\n8K\nN=4",
    "Qwen3\n16K\nN=4",
    "Qwen3\n24K\nN=2",
]
x = np.arange(len(labels))
dense = [0.14, 0.08, 0.20, 0.11, 0.07]
redknot = [0.20, 0.10, 0.26, 0.13, 0.08]
gain = ["+43%", "+27%", "+28%", "+19%", "+15%"]
b1 = ax.bar(
    x - w / 2,
    dense,
    w,
    color=GRAY,
    edgecolor=INK,
    linewidth=0.75,
    label="Dense",
    zorder=3,
)
b2 = ax.bar(
    x + w / 2,
    redknot,
    w,
    color=RED,
    edgecolor=INK,
    linewidth=0.75,
    label="RedKnot",
    zorder=3,
)
style_ax(ax, ylabel="Throughput (req/s)", ylim=(0, 0.35))
ax.set_xticks(x)
ax.set_xticklabels(labels)
for i, (b, txt) in enumerate(zip(b2, gain)):
    ax.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 0.025,
        txt,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#9F2D24",
    )
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.52, 1.30),
    ncol=2,
    frameon=True,
    fancybox=False,
    edgecolor=INK,
    facecolor="white",
    fontsize=8.8,
    handlelength=0.9,
    columnspacing=0.9,
)
caption_below(ax, "(b) Burst-mode Throughput")

# (c) Concurrent capacity.
ax = axes[2]
labels = ["32K\ncontext", "64K\ncontext"]
x = np.arange(len(labels))
dense = [4, 3]
seg = [31, 14]
b1 = ax.bar(
    x - w / 2,
    dense,
    w,
    color=GRAY,
    edgecolor=INK,
    linewidth=0.75,
    label="Dense/vLLM",
    zorder=3,
)
b2 = ax.bar(
    x + w / 2,
    seg,
    w,
    color=RED,
    edgecolor=INK,
    linewidth=0.75,
    label="SegPagedAttention",
    zorder=3,
)
style_ax(ax, ylabel="Sessions / GPU", ylim=(0, 36))
ax.set_xticks(x)
ax.set_xticklabels(labels)
label_bars(ax, b1, fmt="{:.0f}", dy=0.018, size=8.0)
label_bars(ax, b2, fmt="{:.0f}", dy=0.018, size=8.0)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.54, 1.30),
    ncol=2,
    frameon=True,
    fancybox=False,
    edgecolor=INK,
    facecolor="white",
    fontsize=8.4,
    handlelength=0.9,
    columnspacing=0.7,
)
caption_below(ax, "(c) Concurrent Capacity per GPU")

fig.subplots_adjust(left=0.10, right=0.995, top=0.70, bottom=0.40, wspace=0.36)
fig.savefig("7.pdf", bbox_inches="tight", pad_inches=0.08)
fig.savefig("7.png", dpi=260, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print("saved 7.pdf + 7.png")
