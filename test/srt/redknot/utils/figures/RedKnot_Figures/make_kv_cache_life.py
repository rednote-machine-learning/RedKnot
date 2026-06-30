#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KV-cache lifecycle analysis figure."""

from __future__ import annotations

import numpy as np
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
GRID = "#9B9B9B"
GRAY = "#B8B8B8"
LIGHT_GRAY = "#DADADA"
RED = "#D95A48"
ORANGE = "#F0A17F"
DEEP_ORANGE = "#C95D45"
BLACK = "#4D4D4D"


def style_ax(ax, xlabel=None, ylabel=None):
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
        axis="both",
        linestyle=":",
        linewidth=0.40,
        color=GRID,
        alpha=0.35,
    )
    ax.tick_params(
        axis="both", which="major", labelsize=10.2, colors=INK, width=1.0, length=3.8
    )
    ax.tick_params(axis="both", which="minor", width=0.75, length=2.3)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11.7, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.7, fontweight="bold")


def caption_below(ax, text):
    ax.text(
        0.5,
        -0.34,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color=INK,
    )


np.random.seed(7)
fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))

# (a) Reuse-count distribution.
ax = axes[0, 0]
reuse = np.unique(np.round(np.geomspace(1, 280, 55)).astype(int))
counts = 8200 * reuse**-1.65 * np.exp(np.random.normal(0, 0.22, len(reuse)))
counts = np.maximum(1.0, counts)
counts[reuse > 28] *= np.exp(np.random.normal(-0.25, 0.55, (reuse > 28).sum()))
counts = np.maximum(1.0, counts)
ax.scatter(reuse, counts, s=22, color=GRAY, edgecolor=INK, linewidth=0.25, zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.8, 330)
ax.set_ylim(0.6, 18000)
style_ax(ax, xlabel="Reuse count", ylabel="# chunks")
caption_below(ax, "(a) Reuse-count Distribution")

# (b) Non-prefix reuse ratio.
ax = axes[0, 1]
bins = np.array([0.03, 0.50, 0.96])
heights = np.array([260, 80, 15800])
ax.bar(
    bins,
    heights,
    width=[0.035, 0.035, 0.045],
    color=[LIGHT_GRAY, GRAY, ORANGE],
    edgecolor=INK,
    linewidth=0.75,
    zorder=3,
)
ax.axvline(0.95, color=RED, linestyle="--", linewidth=1.7, label="mean=0.95", zorder=4)
style_ax(ax, xlabel="Non-prefix reuse ratio", ylabel="# chunks")
ax.set_xlim(-0.02, 1.04)
ax.set_ylim(0, 17000)
ax.legend(
    loc="upper left",
    fontsize=9.2,
    frameon=True,
    fancybox=False,
    edgecolor=INK,
    facecolor="white",
    framealpha=1.0,
    handlelength=1.8,
)
caption_below(ax, "(b) Non-prefix Reuse")

# (c) Lifecycle scatter.
ax = axes[1, 0]
n = 1200
residency = np.random.uniform(0, 2500, n)
reuse_count = np.clip(
    np.random.lognormal(mean=1.2 + residency / 2600, sigma=0.85, size=n), 1, 260
)
reuse_count = np.round(reuse_count)
value = np.log10((reuse_count + 1) / np.sqrt(residency + 10))
sc = ax.scatter(
    residency,
    reuse_count,
    c=value,
    cmap="Oranges",
    s=12,
    alpha=0.72,
    edgecolors="none",
    zorder=3,
)
ax.set_yscale("log")
ax.set_xlim(-60, 2600)
ax.set_ylim(0.8, 350)
style_ax(ax, xlabel="Residency (request span)", ylabel="Reuse count")
cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.035)
cbar.set_label("log value density", fontsize=9.8, fontweight="bold")
cbar.ax.tick_params(labelsize=8.8, width=0.8, length=3)
caption_below(ax, "(c) Reuse vs. Lifespan")

# (d) Cache cost-benefit frontier.
ax = axes[1, 1]
mem = np.array([1.5, 3.5, 7.8, 15.0, 25.5])
saved = np.array([39, 56, 75, 90, 99])
labels = ["R>=20", "R>=10", "R>=5", "R>=3", "R>=2"]
ax.plot(
    mem,
    saved,
    color=DEEP_ORANGE,
    marker="s",
    markersize=6.6,
    markerfacecolor="none",
    markeredgewidth=1.35,
    linewidth=1.8,
    zorder=3,
)
for x, y, label in zip(mem, saved, labels):
    if label == "R>=2":
        ax.text(x, y - 4.0, label, fontsize=8.6, color=INK, ha="center", va="top")
    else:
        ax.text(x + 1.05, y, label, fontsize=8.6, color=INK, ha="left", va="center")
style_ax(ax, xlabel="KV memory cached (GB)", ylabel="% recompute saved")
ax.set_xlim(0, 28)
ax.set_ylim(35, 104)
caption_below(ax, "(d) Cache Cost-benefit Frontier")

fig.subplots_adjust(
    left=0.095, right=0.965, top=0.965, bottom=0.115, wspace=0.42, hspace=0.72
)
fig.savefig("kv_cache_life.pdf", bbox_inches="tight", pad_inches=0.06)
fig.savefig("kv_cache_life.png", dpi=260, bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print("saved kv_cache_life.pdf + kv_cache_life.png")
