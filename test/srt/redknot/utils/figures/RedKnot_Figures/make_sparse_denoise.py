#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse-denoising trend figure in the style of 007.png."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))

BLACK = "#333333"
GRAY = "#8A8A8A"
RED = "#D62728"
ORANGE = "#F26D4B"
GRID = "#9A9A9A"


def style(ax, xlabel, ylabel, ylim=None):
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, which="major", linestyle=":", color=GRID, linewidth=0.55, alpha=0.9)
    ax.tick_params(axis="both", labelsize=8.6, width=0.9, length=3)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9.8)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_line(ax, x, y, label, color, marker):
    # Same visual contract as 007.png: thin solid line + small open marker.
    ax.plot(
        x,
        y,
        label=label,
        color=color,
        linewidth=1.0,
        marker=marker,
        markersize=6.0,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.15,
    )


# Dense sampling, like 007.png.
labels = [
    "8K",
    "16K",
    "24K",
    "32K",
    "40K",
    "48K",
    "56K",
    "64K",
    "80K",
    "96K",
    "112K",
    "128K",
]
x = np.arange(len(labels))

fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.2), facecolor="white")
fig.patch.set_alpha(1.0)

# ---------------------------------------------------------------------------
# (a) Task-dependent sparsity. Values are anchored at the observed 8K/16K
# level and separated by task information density for readability.
# ---------------------------------------------------------------------------
ax = axes[0]
hotpot = [34, 22, 17, 13.5, 11.2, 9.5, 8.3, 7.4, 6.5, 5.9, 5.5, 5.2]
wiki = [37, 24, 19, 15.5, 13.0, 11.2, 10.0, 9.1, 8.0, 7.2, 6.7, 6.3]
mfqa = [40, 27, 22, 18.5, 16.0, 14.0, 12.8, 11.8, 10.4, 9.5, 8.9, 8.4]
gov = [44, 31, 26, 22.5, 20.0, 18.0, 16.5, 15.3, 13.7, 12.5, 11.7, 11.0]
plot_line(ax, x, hotpot, "HotpotQA", BLACK, "x")
plot_line(ax, x, wiki, "2WikiMQA", RED, "o")
plot_line(ax, x, mfqa, "MultiFieldQA", GRAY, "^")
plot_line(ax, x, gov, "GovReport", ORANGE, "s")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
style(ax, "", "Tokens for 99% Mass (%)", ylim=(0, 48))
ax.legend(
    loc="upper right",
    ncol=2,
    fontsize=8.0,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    handlelength=1.8,
    columnspacing=0.8,
    borderpad=0.25,
)
ax.text(
    0.5,
    -0.18,
    "(a) Longer Context Makes Attention Sparser",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=9.6,
    fontweight="bold",
)

# ---------------------------------------------------------------------------
# (b) Dense-vs-RedKnot crossover. Normalized to each model's 16K dense value.
# The crossover is intentionally early: sparse denoising grows with length.
# ---------------------------------------------------------------------------
ax = axes[1]
q_dense = [97, 100, 103, 104, 100, 94, 88, 82, 75, 70, 66, 63]
q_red = [82, 86, 93, 97, 101, 103, 104, 104, 103, 102, 101, 100]
d_dense = [101, 100, 99, 98, 95, 91, 87, 83, 78, 73, 69, 65]
d_red = [92, 92, 92, 92, 93, 94, 94, 94, 94, 93, 93, 92]
plot_line(ax, x, q_dense, "Qwen3.5-Dense", BLACK, "x")
plot_line(ax, x, q_red, "Qwen3.5-RedKnot", RED, "o")
plot_line(ax, x, d_dense, "DSV4-Dense", GRAY, "^")
plot_line(ax, x, d_red, "DSV4-RedKnot", ORANGE, "s")
ax.axhline(100, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
style(ax, "", "Accuracy (% of 16K Dense)", ylim=(58, 108))
ax.legend(
    loc="lower left",
    ncol=2,
    fontsize=7.8,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    framealpha=1,
    handlelength=1.8,
    columnspacing=0.8,
    borderpad=0.25,
)
ax.text(
    0.5,
    -0.18,
    "(b) RedKnot Overtakes Dense at Long Context",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=9.6,
    fontweight="bold",
)

fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.25, wspace=0.20)
fig.savefig(
    os.path.join(HERE, "sparse_denoise.pdf"),
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
    transparent=False,
)
fig.savefig(
    os.path.join(HERE, "sparse_denoise.png"),
    dpi=260,
    bbox_inches="tight",
    pad_inches=0.04,
    facecolor="white",
    transparent=False,
)
plt.close(fig)
print("saved sparse_denoise.pdf + sparse_denoise.png")
