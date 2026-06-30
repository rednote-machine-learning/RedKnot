#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prefix compression evaluation figure."""

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
GRAY = "#9B9B9B"
LIGHT_GRAY = "#DADADA"
RED = "#D95A48"
ORANGE = "#F0A17F"
BLACK = "#4D4D4D"


def style_ax(ax, ylabel=None, ylim=None):
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
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10.5, colors=INK, width=1.0, length=3.8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.2, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)


def caption_below(ax, text):
    ax.text(
        0.5,
        -0.32,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color=INK,
    )


def line(ax, x, y, label, color, marker, mfc="none", lw=1.8):
    return ax.plot(
        x,
        y,
        label=label,
        color=color,
        linestyle="-",
        linewidth=lw,
        marker=marker,
        markersize=6.8,
        markerfacecolor=mfc,
        markeredgecolor=color,
        markeredgewidth=1.35,
        zorder=3,
    )[0]


fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.45))

# (a) Accuracy and KV saving.
ax = axes[0]
x = np.array([8, 12, 16, 24, 32])
cos = np.array([0.9910, 0.9960, 0.9988, 0.9978, 0.9986])
saved = np.array([24, 33, 37, 41, 44])

l1 = line(ax, x, cos, "cos (decode step-1)", BLACK, "x")
ax.axhline(0.990, color=GRAY, linestyle="--", linewidth=1.1, zorder=1)
ax.text(8.15, 0.9905, "pass threshold 0.99", fontsize=8.7, color="#666666", va="bottom")
style_ax(ax, ylabel="Logits cosine", ylim=(0.985, 1.001))
ax.set_xlabel("Prefix length (K tokens)", fontsize=11.8, fontweight="bold")
ax.set_xticks(x)

ax2 = ax.twinx()
l2 = line(ax2, x, saved, "KV transfer saved", RED, "s", mfc="none", lw=1.8)
ax2.set_ylim(0, 60)
ax2.set_ylabel("KV saved (%)", fontsize=11.0, fontweight="bold", color=INK, labelpad=6)
ax2.tick_params(axis="y", labelsize=10.5, colors=INK, width=1.0, length=3.8)
for side in ["top", "right", "left", "bottom"]:
    ax2.spines[side].set_color(INK)
    ax2.spines[side].set_linewidth(1.25)
for i, (xx, yy) in enumerate(zip(x, saved)):
    ax2.text(
        xx,
        yy + 3.0 if i == 0 else yy - 5.0,
        f"{yy:.0f}%",
        ha="center",
        va="bottom" if i == 0 else "top",
        fontsize=8.2,
        color="#9F2D24",
    )
ax.legend(
    [l1, l2],
    ["cos (decode step-1)", "KV transfer saved"],
    loc="lower right",
    fontsize=8.6,
    frameon=True,
    fancybox=False,
    edgecolor=INK,
    facecolor="white",
    framealpha=1.0,
    handlelength=1.8,
)
caption_below(ax, "(a) Accuracy and KV Saving")

# (b) Concurrency throughput.
ax = axes[1]
x = np.array([8, 16, 32, 36])
baseline = np.array([0.24, 0.12, 0.058, 0.055])
trim = np.array([0.32, 0.193, 0.110, 0.098])
speedups = [1.33, 1.61, 1.90, 1.79]

line(ax, x, baseline, "baseline (full KV)", GRAY, "^")
line(ax, x, trim, "trim<32", RED, "o")
ax.fill_between(x, baseline, trim, color=RED, alpha=0.08, zorder=1)
style_ax(ax, ylabel="QPS / GPU", ylim=(0.04, 0.34))
ax.set_xlabel("Prefix length (K tokens)", fontsize=11.8, fontweight="bold")
ax.set_xticks(x)
ax.set_xlim(6.5, 40)
for i, (xx, yy, sp) in enumerate(zip(x, trim, speedups)):
    dx = [1.15, 0.30, -0.60, 0.0][i]
    dy = [-0.004, 0.028, 0.022, 0.018][i]
    ha = "left" if i == 0 else "center"
    ax.text(
        xx + dx,
        yy + dy,
        f"{sp:.2f}x",
        ha=ha,
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#9F2D24",
    )
ax.legend(
    loc="upper right",
    fontsize=8.8,
    frameon=True,
    fancybox=False,
    edgecolor=INK,
    facecolor="white",
    framealpha=1.0,
    handlelength=1.8,
)
caption_below(ax, "(b) Concurrency Throughput")

# (c) Per-dataset agreement.
ax = axes[2]
labels = ["hotpotqa", "gov_report", "lcc", "wikitext"]
cos_bar = np.array([0.999, 0.998, 0.974, 1.000])
top_match = np.array([0.90, 0.97, 0.98, 0.93])
idx = np.arange(len(labels))
w = 0.34
b1 = ax.bar(
    idx - w / 2,
    cos_bar,
    w,
    color=LIGHT_GRAY,
    edgecolor=INK,
    linewidth=0.75,
    label="cos (step-1)",
    zorder=3,
)
b2 = ax.bar(
    idx + w / 2,
    top_match,
    w,
    color=ORANGE,
    edgecolor=INK,
    linewidth=0.75,
    label="top-match",
    zorder=3,
)
ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.1, zorder=1)
style_ax(ax, ylabel="Score", ylim=(0, 1.16))
ax.set_xticks(idx)
ax.set_xticklabels(labels, rotation=18, ha="right")
for bars, fmt, dy, color in [
    (b1, "{:.2f}", 0.022, INK),
    (b2, "{:.2f}", 0.022, "#9F6C18"),
]:
    for b in bars:
        v = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + dy,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=color,
        )
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.20),
    ncol=2,
    fontsize=8.6,
    frameon=True,
    fancybox=False,
    edgecolor=INK,
    facecolor="white",
    framealpha=1.0,
    handlelength=0.9,
    columnspacing=0.9,
)
caption_below(ax, "(c) Dataset Agreement")

fig.subplots_adjust(left=0.065, right=0.995, top=0.78, bottom=0.34, wspace=0.58)
fig.savefig("prefix_compression.pdf", bbox_inches="tight", pad_inches=0.06)
fig.savefig("prefix_compression.png", dpi=260, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print("saved prefix_compression.pdf + prefix_compression.png")
