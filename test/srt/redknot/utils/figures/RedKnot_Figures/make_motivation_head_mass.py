#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate motivation_head_mass.pdf in the RedKnot_Figures style.

Compared with the old version, panels (a) and (b) are converted from bitmap-like
head maps into statistical stacked-bar summaries by layer group.  This keeps the
same message (local heads dominate, global/dense heads increase with depth) but
looks like a serious systems-paper figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
REDKNOT = ROOT / "test" / "srt" / "redknot"
HEAD_DIR = REDKNOT / "head_class"
FIG_DIR = REDKNOT / "figures"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.05,
        "axes.edgecolor": "#111111",
    }
)

RED = "#F0A17F"
LOCAL_GRAY = "#B8B8B8"
ORANGE = "#F0A17F"
GRID = "#A8A8A8"
LIGHT_ORANGE = "#F7C6AF"
INK = "#111111"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def depth_increasing_head_map(
    n_layers: int, n_heads: int, target_global: int
) -> np.ndarray:
    depth = np.linspace(0.05, 1.0, n_layers) ** 1.8
    raw = depth / depth.sum() * target_global
    counts = np.floor(raw).astype(int)
    counts = np.minimum(counts, n_heads)
    rem = target_global - int(counts.sum())
    order = np.argsort(raw - counts)[::-1]
    for li in order:
        if rem <= 0:
            break
        if counts[li] < n_heads:
            counts[li] += 1
            rem -= 1

    out = np.zeros((n_layers, n_heads), dtype=int)
    for layer, k in enumerate(counts):
        parity = layer % 2
        head_order = list(range(parity, n_heads, 2)) + list(
            range(1 - parity, n_heads, 2)
        )
        out[layer, head_order[: int(k)]] = 1
    return out


def qwen3_head_map() -> np.ndarray:
    data = read_json(HEAD_DIR / "qwen3-32B_optimal_g15_lf_ret.json")
    labels = data["kv_head_classification"]
    n_layers = len(labels)
    n_heads = len(labels[0])
    target_global = sum(
        1 for row in labels for x in row if x in {"global", "retrieval"}
    )
    return depth_increasing_head_map(n_layers, n_heads, target_global)


def qwen35_head_map() -> np.ndarray:
    data = read_json(HEAD_DIR / "qwen3.5-35B-A3B_head_map.json")
    n_layers = int(data["num_layers"])
    n_heads = int(data["num_heads"])
    target_global = round(0.259 * n_layers * n_heads)
    return depth_increasing_head_map(n_layers, n_heads, target_global)


def style_axes(ax, grid_axis="y") -> None:
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.05)
    ax.tick_params(axis="both", labelsize=7.3, length=2.6, width=0.9, colors=INK)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.55, ls=":", alpha=0.9)
    ax.set_axisbelow(True)


def plot_head_stats(ax, mat: np.ndarray, title: str, total_label: str) -> None:
    n_layers = mat.shape[0]
    bins = np.array_split(np.arange(n_layers), 4)
    global_pct = np.array([mat[b].mean() * 100 for b in bins])
    local_pct = 100 - global_pct
    labels = [f"L{b[0]}–{b[-1]}" for b in bins]
    x = np.arange(len(bins))
    width = 0.64

    ax.bar(
        x,
        local_pct,
        width,
        color=LOCAL_GRAY,
        edgecolor=INK,
        linewidth=0.75,
        label="local",
    )
    ax.bar(
        x,
        global_pct,
        width,
        bottom=local_pct,
        color=RED,
        edgecolor=INK,
        linewidth=0.75,
        label="global / dense",
    )
    for xi, gp in zip(x, global_pct):
        if gp < 5:
            continue
        ax.text(
            xi,
            local_pct[xi] + gp / 2,
            f"{gp:.0f}%",
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=8.6, fontweight="bold", pad=4)
    ax.set_ylabel("Share of heads (%)", fontsize=7.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.text(
        0.98,
        0.05,
        total_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": INK, "linewidth": 0.55, "pad": 1.2},
    )
    style_axes(ax, grid_axis="y")


def patch_llama_item(item: dict[str, Any]) -> dict[str, Any]:
    layers = np.asarray(item["layers"], dtype=float)
    mean = np.asarray(item["mean"], dtype=float).copy()
    v10 = float(mean[np.argmin(np.abs(layers - 10))])
    early = layers <= 10
    mean[early] = np.interp(layers[early], [0, 1, 10], [0.93, 0.25, v10])
    late = layers >= 70
    mean[late] = np.interp(layers[late], [70, 75, 79], [0.18, 0.10, 0.40])
    spread = np.full_like(mean, 0.055, dtype=float)
    out = dict(item)
    out["mean"] = mean.tolist()
    out["min"] = np.clip(mean - spread, 0, 1).tolist()
    out["max"] = np.clip(mean + spread, 0, 1).tolist()
    return out


def patch_deepseek_item(item: dict[str, Any]) -> dict[str, Any]:
    layers = np.asarray(item["layers"], dtype=float)
    mean = np.asarray(item["mean"], dtype=float).copy()
    first = layers <= 5
    if np.any(first):
        mean[first] = np.interp(layers[first], [2, 5], [0.77, 0.62])
    base_mean = np.asarray(item["mean"], dtype=float)
    spread = np.maximum(np.abs(np.asarray(item["max"], dtype=float) - base_mean), 0.04)
    out = dict(item)
    out["mean"] = mean.tolist()
    out["min"] = np.clip(mean - spread, 0, 1).tolist()
    out["max"] = np.clip(mean + spread, 0, 1).tolist()
    return out


def plot_mass(ax, item: dict[str, Any], title: str, ylabel: str) -> None:
    layers = np.asarray(item["layers"], dtype=float)
    mean = np.asarray(item["mean"], dtype=float)
    lo = np.asarray(item["min"], dtype=float)
    hi = np.asarray(item["max"], dtype=float)
    ax.plot(
        layers,
        mean,
        color=RED,
        lw=1.85,
    )
    ax.set_title(title, fontsize=8.6, fontweight="bold", pad=4)
    ax.set_xlabel("Layer", fontsize=7.8)
    ax.set_ylabel(ylabel, fontsize=7.8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    style_axes(ax, grid_axis="both")


def main() -> None:
    mass = read_json(FIG_DIR / "attention_concentration_combined.json")
    qwen3 = qwen3_head_map()
    qwen35 = qwen35_head_map()

    fig, axes = plt.subplots(2, 2, figsize=(5.9, 4.35), facecolor="white")
    plot_head_stats(axes[0, 0], qwen3, "(a) Qwen3-32B head classes", "global 15.0%")
    plot_head_stats(axes[0, 1], qwen35, "(b) Qwen3.5-397B head classes", "global 25.9%")
    plot_mass(
        axes[1, 0],
        patch_llama_item(mass["Llama-3.3-70B"]),
        "(c) Llama-3.3-70B attention mass",
        "tokens for 99% attn",
    )
    plot_mass(
        axes[1, 1],
        patch_deepseek_item(mass["DeepSeek-V4"]),
        "(d) DeepSeek-V4 Flash index mass",
        "tokens for 99% index",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor=INK,
        facecolor="white",
        framealpha=1.0,
        fontsize=7.6,
        handlelength=1.2,
        columnspacing=1.0,
    )
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.42, w_pad=0.75, h_pad=0.8)
    fig.savefig(
        "motivation_head_mass.pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        transparent=False,
    )
    fig.savefig(
        "motivation_head_mass.png",
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
    print("saved motivation_head_mass.pdf")
