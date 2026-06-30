#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Line-style SegPagedAttention evaluation figures."""

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
GRAY = "#8A8A8A"
BLACK = "#4D4D4D"
RED = "#D62728"
ORANGE = "#E6614C"

STYLE = {
    "FA (no mask)": dict(color=BLACK, marker="x", mfc="none", mew=1.35),
    "Efficient (mask)": dict(color=RED, marker="o", mfc="none", mew=1.35),
    "Dense (no mask)": dict(color=GRAY, marker="^", mfc="none", mew=1.35),
    "Dense+mask": dict(color=BLACK, marker="x", mfc="none", mew=1.35),
    "SegPaged": dict(color=RED, marker="o", mfc="none", mew=1.35),
    "Manual": dict(color=BLACK, marker="x", mfc="none", mew=1.35),
    "SDPA+mask": dict(color=RED, marker="o", mfc="none", mew=1.35),
    "Fused": dict(color=ORANGE, marker="s", mfc="none", mew=1.35),
}

X = [8, 32, 128]
X_SHORT = [8, 32]


def style_axis(ax, ylabel=None, xlabel=True):
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
        alpha=0.9,
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
        axis="both", which="major", labelsize=10.8, width=1.0, length=3.8, color=INK
    )
    ax.tick_params(axis="y", which="minor", width=0.8, length=2.4, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12.6, fontweight="bold")
    if xlabel:
        ax.set_xlabel("Context Length", fontsize=12.3, fontweight="bold")


def plot_panel(ax, title, x, series, ylabel=None, ylim=None, log=False, annotate=None):
    for name, y in series.items():
        st = STYLE[name]
        ax.plot(
            x,
            y,
            label=name,
            color=st["color"],
            linestyle="-",
            linewidth=1.75,
            marker=st["marker"],
            markersize=6.6,
            markerfacecolor=st["mfc"],
            markeredgecolor=st["color"],
            markeredgewidth=st["mew"],
            zorder=3,
        )
    ax.set_title(title, fontsize=12.2, pad=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}K" for v in x])
    if log:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    style_axis(ax, ylabel=ylabel)
    if annotate:
        for xx, yy, txt in annotate:
            ax.text(
                xx,
                yy,
                txt,
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color=RED,
            )


def shared_legend(fig, axes, order, y=1.02, ncol=4):
    handles, labels = [], []
    seen = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        seen.update(dict(zip(l, h)))
    for name in order:
        if name in seen:
            handles.append(seen[name])
            labels.append(name)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fontsize=10.5,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor=INK,
        facecolor="white",
        handlelength=2.4,
        columnspacing=1.35,
        borderpad=0.35,
    )


def save(fig, stem):
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{stem}.png", dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# Figure 1: latency comparison from SegPagedattention_1.
fig1, axes1 = plt.subplots(1, 2, figsize=(7.8, 3.3))
plot_panel(
    axes1[0],
    "(a) Single-layer Decode",
    X,
    {"FA (no mask)": [0.07, 0.20, 0.71], "Efficient (mask)": [0.35, 1.35, 5.24]},
    ylabel="Latency (ms)",
    ylim=(0, 5.8),
)
plot_panel(
    axes1[1],
    "(b) 64-layer Decode",
    X,
    {
        "Dense (no mask)": [13.2, 47.9, 218.2],
        "Dense+mask": [31.8, 122.3, 506.8],
        "SegPaged": [12.6, 12.4, 24.1],
    },
    ylabel="Latency (ms)",
    ylim=(8, 650),
    log=True,
)
shared_legend(
    fig1,
    axes1,
    ["FA (no mask)", "Efficient (mask)", "Dense (no mask)", "Dense+mask", "SegPaged"],
    y=1.08,
    ncol=3,
)
fig1.subplots_adjust(left=0.105, right=0.99, top=0.74, bottom=0.20, wspace=0.28)
save(fig1, "SegPagedattention_1")


# Figure 2: implementation comparison from SegPagedAttention_2.
fig2, axes2 = plt.subplots(1, 3, figsize=(10.8, 3.35))
plot_panel(
    axes2[0],
    "(a) Decode Latency",
    X,
    {"Manual": [45, 88, 520], "SDPA+mask": [22, 42, 230], "Fused": [8, 18, 156]},
    ylabel="Latency (ms)",
    ylim=(6, 700),
    log=True,
    annotate=[(8, 25, "2.85x"), (32, 48, "3.03x"), (128, 270, "3.39x")],
)
plot_panel(
    axes2[1],
    "(b) Prefill Latency",
    X,
    {
        "Manual": [0.66, 2.7, 5.4],
        "SDPA+mask": [0.30, 1.10, 0.40],
        "Fused": [0.12, 0.28, 0.23],
    },
    ylabel="Latency (s)",
    ylim=(0.08, 7.5),
    log=True,
    annotate=[(8, 0.36, "6.3x"), (32, 1.25, "11.3x"), (128, 0.48, "23.3x")],
)
plot_panel(
    axes2[2],
    "(c) Throughput",
    X_SHORT,
    {"SDPA+mask": [5.9, 1.5], "SegPaged": [37.4, 17.1]},
    ylabel="tok/s (x10^3)",
    ylim=(0, 42),
)
shared_legend(fig2, axes2, ["Manual", "SDPA+mask", "Fused", "SegPaged"], y=1.08, ncol=4)
fig2.subplots_adjust(left=0.075, right=0.99, top=0.75, bottom=0.20, wspace=0.30)
save(fig2, "SegPagedAttention_2")


# Optional compact paper-ready version: 4 panels in one row.
fig3, axes3 = plt.subplots(1, 4, figsize=(14.8, 3.45))
plot_panel(
    axes3[0],
    "(a) Single-layer Decode",
    X,
    {"FA (no mask)": [0.07, 0.20, 0.71], "Efficient (mask)": [0.35, 1.35, 5.24]},
    ylabel="Latency (ms)",
    ylim=(0, 5.8),
)
plot_panel(
    axes3[1],
    "(b) 64-layer Decode",
    X,
    {
        "Dense (no mask)": [13.2, 47.9, 218.2],
        "Dense+mask": [31.8, 122.3, 506.8],
        "SegPaged": [12.6, 12.4, 24.1],
    },
    ylim=(8, 650),
    log=True,
)
plot_panel(
    axes3[2],
    "(c) Decode Latency",
    X,
    {"Manual": [45, 88, 520], "SDPA+mask": [22, 42, 230], "Fused": [8, 18, 156]},
    ylim=(6, 700),
    log=True,
)
plot_panel(
    axes3[3],
    "(d) Prefill Latency",
    X,
    {
        "Manual": [0.66, 2.7, 5.4],
        "SDPA+mask": [0.30, 1.10, 0.40],
        "Fused": [0.12, 0.28, 0.23],
    },
    ylabel="Latency (s)",
    ylim=(0.08, 7.5),
    log=True,
)
shared_legend(
    fig3,
    axes3,
    [
        "FA (no mask)",
        "Efficient (mask)",
        "Dense (no mask)",
        "Dense+mask",
        "SegPaged",
        "Manual",
        "SDPA+mask",
        "Fused",
    ],
    y=1.08,
    ncol=4,
)
fig3.subplots_adjust(left=0.055, right=0.995, top=0.70, bottom=0.20, wspace=0.30)
save(fig3, "SegPagedAttention_lines")

print("saved SegPagedattention_1, SegPagedAttention_2, SegPagedAttention_lines")
