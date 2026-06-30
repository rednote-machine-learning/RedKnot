#!/usr/bin/env python3
"""Regenerate fix_1 (FFN vs Attention TTFT share) and fix_2 (local/global KV
head distribution) as clean single-column PNGs.

fix_1: two stacked panels (one per model) so each line plot is wide and legible
in a single column.
fix_2: one row of grouped bars across models, plotting the local vs global KV
head SHARE (%) so models of very different head counts stay comparable, with the
absolute head counts annotated on top of each bar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

REDKNOT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = REDKNOT_DIR / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "figure.dpi": 300,
    }
)

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GOLD = "#f6ad1e"
GRID = "#d9dee5"

# ──────────────────────────────────────────────────────────────────────────
# fix_1 data: measured share of prefill TTFT (%) for FFN and Attention.
# ──────────────────────────────────────────────────────────────────────────
LENGTHS = ["2K", "4K", "8K", "16K", "32K"]
FIX1 = {
    "Qwen3-32B (TP=2)": {
        "ffn": [61.3, 60.6, 57.8, 52.1, 44.4],
        "attn": [25.4, 26.5, 31.4, 38.1, 47.6],
    },
    "Llama-3.3-70B (TP=4)": {
        "ffn": [59.5, 59.6, 60.1, 56.7, 53.4],
        "attn": [33.7, 35.6, 36.7, 40.5, 44.7],
    },
}

# ──────────────────────────────────────────────────────────────────────────
# fix_2 data: KV-head distribution (local vs global/dense). Counts use a
# consistent per-model head accounting; shares make the models comparable.
#   Llama-3.3-70B : 80 layers x 8 KV heads = 640 (1/8 global)
#   Qwen3-32B     : 64 layers x 8 KV heads = 512 (measured head-class JSON)
#   Mistral-7B    : 32 layers x 8 KV heads = 256
#   Qwen3.5-35B   : 40 layers x 2 KV heads = 80 (hybrid: full+linear)
#   DeepSeek-V4   : 43 layers x 64 logical MLA heads = 2752
# ──────────────────────────────────────────────────────────────────────────
FIX2 = [
    ("Llama-3.3-70B", 560, 80),
    ("Qwen3-32B", 435, 77),
    ("Mistral-7B", 216, 40),
    ("Qwen3.5-35B", 484, 16),
    ("DeepSeek-V4", 2296, 456),
]


def make_fix1() -> Path:
    x = np.arange(len(LENGTHS))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), sharey=True)
    for ax, (title, d) in zip(axes, FIX1.items()):
        # Force each subplot's plotting area to be a 1:1 square.
        ax.set_box_aspect(1)
        ax.plot(
            x,
            d["ffn"],
            "-o",
            color=BLUE,
            lw=3.2,
            ms=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="FFN",
        )
        ax.plot(
            x,
            d["attn"],
            "--s",
            color=ORANGE,
            lw=3.2,
            ms=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="Attention",
        )
        ax.set_title(title, fontsize=17, fontweight="bold", pad=8)
        ax.grid(True, color=GRID, lw=0.6, ls=":")
        ax.tick_params(labelsize=15, length=3, width=1.0)
        ax.set_ylim(18, 74)
        ax.set_yticks([20, 35, 50, 65])
        # Pad the x-axis on the right so the 32K labels stay inside the frame.
        ax.set_xlim(x[0] - 0.35, x[-1] + 0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(LENGTHS, fontsize=15)
        ax.set_xlabel("Context length", fontsize=16)
        # FFN endpoints: place above the line.
        ax.annotate(
            f"{d['ffn'][0]:.1f}%",
            (x[0], d["ffn"][0]),
            textcoords="offset points",
            xytext=(4, 11),
            fontsize=15,
            color=BLUE,
            fontweight="bold",
        )
        # Last point (32K): label each line directly above/below its own point
        # depending on which line is higher, so the two labels never overlap.
        ffn_below = d["ffn"][-1] < d["attn"][-1]
        ax.annotate(
            f"{d['ffn'][-1]:.1f}%",
            (x[-1], d["ffn"][-1]),
            textcoords="offset points",
            xytext=(0, -22 if ffn_below else 13),
            ha="center",
            fontsize=15,
            color=BLUE,
            fontweight="bold",
        )
        # Attention endpoints: first point below the line.
        ax.annotate(
            f"{d['attn'][0]:.1f}%",
            (x[0], d["attn"][0]),
            textcoords="offset points",
            xytext=(4, -20),
            fontsize=15,
            color=ORANGE,
            fontweight="bold",
        )
        ax.annotate(
            f"{d['attn'][-1]:.1f}%",
            (x[-1], d["attn"][-1]),
            textcoords="offset points",
            xytext=(0, 13 if ffn_below else -22),
            ha="center",
            fontsize=15,
            color=ORANGE,
            fontweight="bold",
        )
    axes[0].set_ylabel("Share of prefill TTFT (%)", fontsize=16)
    axes[0].legend(
        fontsize=15,
        loc="center left",
        frameon=True,
        framealpha=0.9,
        handlelength=1.8,
        borderpad=0.5,
    )
    fig.tight_layout(pad=0.5, w_pad=1.0)
    out = FIG_DIR / "fix_1.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_fix2() -> Path:
    models = [m[0] for m in FIX2]
    local = np.array([m[1] for m in FIX2], dtype=float)
    glob = np.array([m[2] for m in FIX2], dtype=float)
    total = local + glob
    local_pct = local / total * 100
    glob_pct = glob / total * 100

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    b1 = ax.bar(
        x - w / 2,
        local_pct,
        w,
        color=BLUE,
        label="Local heads",
        edgecolor="white",
        lw=0.4,
    )
    b2 = ax.bar(
        x + w / 2,
        glob_pct,
        w,
        color=GOLD,
        label="Global / dense heads",
        edgecolor="white",
        lw=0.4,
    )

    # Bar height is the percentage; annotate the percentage (matching the axis)
    # on the first line and the absolute head count on the second line.
    for rect, pct, n in zip(b1, local_pct, local.astype(int)):
        ax.annotate(
            f"{pct:.1f}%\n({n})",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="bottom",
            fontsize=6,
            color=BLUE,
            fontweight="bold",
            linespacing=0.95,
        )
    for rect, pct, n in zip(b2, glob_pct, glob.astype(int)):
        ax.annotate(
            f"{pct:.1f}%\n({n})",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            textcoords="offset points",
            xytext=(4, 2),
            ha="center",
            va="bottom",
            fontsize=6,
            color="#9c6a00",
            fontweight="bold",
            linespacing=0.95,
        )

    ax.set_ylabel("Share of KV heads (%)", fontsize=8)
    ax.set_title(
        "Local vs global KV-head distribution", fontsize=9, fontweight="bold", pad=16
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7, rotation=18, ha="right")
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y", color=GRID, lw=0.5, ls=":")
    ax.set_axisbelow(True)
    ax.legend(
        fontsize=7, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0)
    )
    fig.tight_layout(pad=0.4)
    out = FIG_DIR / "fix_2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(make_fix1())
    print(make_fix2())
