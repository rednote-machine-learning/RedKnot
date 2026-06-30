#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate ffn_share_combined and global/local head distribution figures.

Style is aligned with the rest of RedKnot_Figures:
  * line chart: 007-style thin lines + open markers + dotted grid.
  * bar chart: white background, black frame, muted gray/red palette,
    compact value labels.
"""

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
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))

INK = "#111111"
GRID = "#A8A8A8"
GRAY = "#B8B8B8"
DARK_GRAY = "#4D4D4D"
RED = "#D95A48"
ORANGE = "#F0A17F"
ATTN_GRAY = "#8A8A8A"


def style_line_ax(ax, xlabel, ylabel, ylim=None):
    ax.set_facecolor("white")
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.05)
    ax.grid(True, which="major", linestyle=":", linewidth=0.65, color=GRID, alpha=0.9)
    ax.tick_params(axis="both", labelsize=8.8, width=0.9, length=3.2, colors=INK)
    ax.set_xlabel(xlabel, fontsize=10.8)
    ax.set_ylabel(ylabel, fontsize=10.8)
    if ylim is not None:
        ax.set_ylim(*ylim)


def open_line(ax, x, y, label, color, marker):
    ax.plot(
        x,
        y,
        label=label,
        color=color,
        linewidth=1.15,
        marker=marker,
        markersize=5.8,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.05,
        zorder=3,
    )


def save(fig, name):
    pdf_path = os.path.join(HERE, f"{name}.pdf")
    png_path = os.path.join(HERE, f"{name}.png")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        transparent=False,
    )
    fig.savefig(
        png_path,
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        transparent=False,
    )
    try:
        from PIL import Image

        im = Image.open(png_path).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        bg.convert("RGB").save(png_path)
    except Exception:
        pass
    plt.close(fig)


def make_ffn_share():
    lengths = ["2K", "4K", "8K", "16K", "32K"]
    x = np.arange(len(lengths))

    # Values reconstructed from the existing figure/measurements.
    qwen_ffn = [61.3, 60.6, 57.7, 52.1, 44.4]
    qwen_attn = [25.4, 26.8, 31.3, 38.1, 47.6]
    llama_ffn = [59.5, 59.7, 60.3, 56.7, 53.4]
    llama_attn = [33.7, 35.4, 36.5, 40.7, 44.7]

    # Two near-square panels; easier to crop into single-column subfigures.
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.25), facecolor="white", sharey=True)
    panels = [
        (axes[0], "Qwen3-32B (TP=2)", qwen_ffn, qwen_attn),
        (axes[1], "Llama-3.3-70B (TP=4)", llama_ffn, llama_attn),
    ]
    for ax, title, ffn, attn in panels:
        open_line(ax, x, ffn, "FFN", ORANGE, "o")
        open_line(ax, x, attn, "Attention", ATTN_GRAY, "s")
        ax.set_xticks(x)
        ax.set_xticklabels(lengths, fontsize=8.7)
        style_line_ax(ax, "Context length", "Share of prefill TTFT (%)", ylim=(18, 66))
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
        # Compact endpoint annotations only; keeps the line chart clean.
        ax.text(
            x[0] + 0.05,
            ffn[0] + 1.4,
            f"{ffn[0]:.1f}%",
            fontsize=7.9,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            x[0] + 0.05,
            attn[0] - 4.4,
            f"{attn[0]:.1f}%",
            fontsize=7.9,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            x[-1] - 0.32,
            ffn[-1] - 4.0,
            f"{ffn[-1]:.1f}%",
            fontsize=7.9,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            x[-1] - 0.32,
            attn[-1] + 2.0,
            f"{attn[-1]:.1f}%",
            fontsize=7.9,
            color=INK,
            fontweight="bold",
        )
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor=INK,
        facecolor="white",
        framealpha=1.0,
        fontsize=9.2,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.8,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.80, bottom=0.18, wspace=0.18)
    save(fig, "ffn_share_combined")


def make_global_local_heads():
    models = [
        "Llama-3.3\n70B",
        "Qwen3\n32B",
        "Mistral\n7B",
        "Qwen3.5\n397B",
        "DeepSeek\nV4 Flash",
    ]
    local = np.array([87.5, 85.0, 84.4, 96.8, 83.4])
    global_dense = np.array([12.5, 15.0, 15.6, 3.2, 16.6])
    local_cnt = [560, 435, 216, 484, 2296]
    global_cnt = [80, 77, 40, 16, 456]
    x = np.arange(len(models))
    w = 0.58

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.85), facecolor="white")
    ax.set_facecolor("white")
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.05)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.65, color=GRID, alpha=0.9)
    ax.set_axisbelow(True)
    b1 = ax.bar(
        x,
        local,
        width=w,
        color=GRAY,
        edgecolor=INK,
        linewidth=0.75,
        label="Local heads",
        zorder=3,
    )
    b2 = ax.bar(
        x,
        global_dense,
        bottom=local,
        width=w,
        color=RED,
        edgecolor=INK,
        linewidth=0.75,
        label="Global / dense heads",
        zorder=3,
    )
    for i, (l, g) in enumerate(zip(local, global_dense)):
        ax.text(
            i,
            l / 2,
            f"{l:.1f}%\n({local_cnt[i]})",
            ha="center",
            va="center",
            fontsize=7.5,
            color=INK,
            fontweight="bold",
        )
        if g < 5:
            ax.text(
                i,
                l - 2.2,
                f"{g:.1f}%\n({global_cnt[i]})",
                ha="center",
                va="center",
                fontsize=6.7,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                i,
                l + g / 2,
                f"{g:.1f}%\n({global_cnt[i]})",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white",
                fontweight="bold",
            )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Share of KV heads (%)", fontsize=10.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.4)
    ax.tick_params(axis="y", labelsize=8.8, width=0.9, length=3.2)
    ax.legend(
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor=INK,
        facecolor="white",
        framealpha=1.0,
        fontsize=8.8,
        bbox_to_anchor=(0.5, 1.13),
        handlelength=1.1,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.22)
    save(fig, "global_and_local_heads_in_models")


if __name__ == "__main__":
    make_ffn_share()
    make_global_local_heads()
    print("saved ffn_share_combined + global_and_local_heads_in_models")
