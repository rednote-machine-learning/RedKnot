#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate back_1.pdf and back_2.pdf in the same visual language as
Introduce.pdf.

Design goals:
  * serious systems-paper style: serif fonts, black axes/spines, muted palette
  * colors match Introduce.pdf semantics (global=red, local=blue, FFN=orange)
  * compact annotations, no oversized presentation-style typography
"""

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
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
    }
)

# Palette aligned with Introduce.pdf
RED = "#EFA59C"  # global / recompute
BLUE = "#A7B5E0"  # local / reuse
ORANGE = "#F4B43C"  # FFN / selected tokens
GRAY = "#C9CCD2"
INK = "#111111"
GRID = "#D7D9DE"


LENGTHS = ["2K", "4K", "8K", "16K", "32K"]
BACK2 = {
    "Qwen3-32B (TP=2)": {
        "ffn": [61.3, 60.6, 57.8, 52.1, 44.4],
        "attn": [25.4, 26.5, 31.4, 38.1, 47.6],
    },
    "Llama-3.3-70B (TP=4)": {
        "ffn": [59.5, 59.6, 60.1, 56.7, 53.4],
        "attn": [33.7, 35.6, 36.7, 40.5, 44.7],
    },
}

BACK1 = [
    ("Llama-3.3\n70B", 560, 80),
    ("Qwen3\n32B", 435, 77),
    ("Mistral\n7B", 216, 40),
    ("Qwen3.5\n397B", 484, 16),
    ("DeepSeek\nV4 Flash", 2296, 456),
]


def style_axes(ax, grid_axis="y"):
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color(INK)
    ax.tick_params(axis="both", labelsize=8.5, width=1.0, length=3)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.55, ls="--", alpha=0.85)
    ax.set_axisbelow(True)


def make_back_1() -> None:
    models = [m[0] for m in BACK1]
    local = np.array([m[1] for m in BACK1], dtype=float)
    glob = np.array([m[2] for m in BACK1], dtype=float)
    total = local + glob
    local_pct = local / total * 100.0
    glob_pct = glob / total * 100.0

    x = np.arange(len(models))
    w = 0.34

    fig, ax = plt.subplots(figsize=(5.1, 3.15))
    ax.bar(
        x - w / 2,
        local_pct,
        w,
        color=BLUE,
        edgecolor=INK,
        linewidth=0.85,
        label="Local heads",
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        glob_pct,
        w,
        color=RED,
        edgecolor=INK,
        linewidth=0.85,
        label="Global / dense heads",
        zorder=3,
    )

    for xi, pct, n in zip(x - w / 2, local_pct, local.astype(int)):
        ax.text(
            xi,
            pct + 2.2,
            f"{pct:.1f}%\n({n})",
            ha="center",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
            color=INK,
            linespacing=0.9,
        )
    for xi, pct, n in zip(x + w / 2, glob_pct, glob.astype(int)):
        ax.text(
            xi + 0.08,
            pct + 2.0,
            f"{pct:.1f}%\n({n})",
            ha="center",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
            color=INK,
            linespacing=0.9,
        )

    ax.set_ylabel("Share of KV heads (%)", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.0)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    style_axes(ax, grid_axis="y")
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        fontsize=8.2,
        handlelength=1.7,
        columnspacing=1.6,
    )
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout(pad=0.35)
    fig.savefig("back_1.pdf")
    fig.savefig("back_1.png", dpi=240)
    plt.close(fig)


def make_back_2() -> None:
    x = np.arange(len(LENGTHS))
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 3.15), sharey=True)
    for ax, (title, d) in zip(axes, BACK2.items()):
        ffn = np.array(d["ffn"])
        attn = np.array(d["attn"])
        ax.plot(
            x,
            ffn,
            "-o",
            color=ORANGE,
            lw=2.2,
            ms=5.8,
            markeredgecolor=INK,
            markeredgewidth=0.7,
            label="FFN",
            zorder=4,
        )
        ax.plot(
            x,
            attn,
            "--s",
            color=BLUE,
            lw=2.2,
            ms=5.5,
            markeredgecolor=INK,
            markeredgewidth=0.7,
            label="Attention",
            zorder=4,
        )
        ax.set_title(title, fontsize=9.8, fontweight="bold", pad=5)
        ax.set_xlim(-0.35, len(LENGTHS) - 0.45)
        ax.set_ylim(18, 72)
        ax.set_xticks(x)
        ax.set_xticklabels(LENGTHS, fontsize=8.5)
        ax.set_yticks([20, 35, 50, 65])
        ax.set_xlabel("Context length", fontsize=9.5, fontweight="bold")
        style_axes(ax, grid_axis="both")

        # Compact endpoint annotations; all text black to avoid presentation look.
        ax.annotate(
            f"{ffn[0]:.1f}%",
            (x[0], ffn[0]),
            xytext=(3, 8),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=INK,
        )
        ax.annotate(
            f"{attn[0]:.1f}%",
            (x[0], attn[0]),
            xytext=(3, -15),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color=INK,
        )
        # Avoid overlap at the right endpoint by placing labels on opposite sides.
        ffn_lower = ffn[-1] < attn[-1]
        ax.annotate(
            f"{ffn[-1]:.1f}%",
            (x[-1], ffn[-1]),
            xytext=(0, -16 if ffn_lower else 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.0,
            fontweight="bold",
            color=INK,
        )
        ax.annotate(
            f"{attn[-1]:.1f}%",
            (x[-1], attn[-1]),
            xytext=(0, 8 if ffn_lower else -16),
            textcoords="offset points",
            ha="center",
            fontsize=8.0,
            fontweight="bold",
            color=INK,
        )

    axes[0].set_ylabel("Share of prefill TTFT (%)", fontsize=9.5, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=8.4,
        handlelength=1.8,
        columnspacing=1.6,
    )
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout(pad=0.35, w_pad=0.75, rect=[0, 0, 1, 0.94])
    fig.savefig("back_2.pdf")
    fig.savefig("back_2.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    make_back_1()
    make_back_2()
    print("saved back_1.pdf/back_2.pdf")
