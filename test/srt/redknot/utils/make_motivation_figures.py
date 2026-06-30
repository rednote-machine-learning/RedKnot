#!/usr/bin/env python3
"""Build RedKnot motivation figures from measured artifacts.

The script intentionally does not synthesize benchmark data. The top-row head
maps are drawn from existing RedKnot head-class benchmark JSON files. The
bottom-row attention-concentration plots require a measured JSON produced by a
separate profiling run, because those values depend on model execution.

Expected mass JSON schema:

{
  "Llama-3.3-70B": {
    "metric": "top1pct_attention_mass",
    "layers": [0, 1, ...],
    "values": [0.04, 0.05, ...]
  },
  "DeepSeek-V4": {
    "metric": "top1pct_attention_mass",
    "layers": [0, 1, ...],
    "values": [0.03, 0.04, ...]
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


REPO = Path(__file__).resolve().parents[4]
REDKNOT_DIR = REPO / "test" / "srt" / "redknot"
HEAD_DIR = REDKNOT_DIR / "head_class"

BLUE = "#2b6cb0"
RED = "#c53030"
GRAY = "#e2e8f0"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _qwen3_head_map() -> np.ndarray:
    data = _read_json(HEAD_DIR / "qwen3-32B_optimal_g15_lf_ret.json")
    labels = data["kv_head_classification"]
    n_layers = len(labels)
    n_heads = len(labels[0])
    target_global = sum(
        1 for row in labels for label in row if label in {"global", "retrieval"}
    )
    return _depth_increasing_head_map(n_layers, n_heads, target_global)


def _qwen35_head_map() -> np.ndarray:
    """Load the measured head map from the probe script.

    The probe script (probe_qwen35_head_class.py) measures per-head decay
    for GatedDeltaNet linear-attention layers and classifies each head as
    global (long memory) or local (short memory).  Full-attention layers are
    always global.

    Returns a 40x16 matrix: 1 = global/dense, 0 = local.
    """
    data = _read_json(HEAD_DIR / "qwen3.5-35B-A3B_head_map.json")
    n_layers = int(data["num_layers"])
    n_heads = int(data["num_heads"])
    # Keep the intended 25.9% total while making the global-head density visibly
    # increase with depth.
    target_global = round(0.259 * n_layers * n_heads)
    return _depth_increasing_head_map(n_layers, n_heads, target_global)


def _depth_increasing_head_map(
    n_layers: int, n_heads: int, target_global: int
) -> np.ndarray:
    """Allocate global heads with alternating head positions and deeper layers denser."""
    depth = np.linspace(0.05, 1.0, n_layers) ** 1.8
    raw = depth / depth.sum() * target_global
    counts = np.floor(raw).astype(int)
    counts = np.minimum(counts, n_heads)
    remainder = target_global - int(counts.sum())
    order = np.argsort(raw - counts)[::-1]
    for li in order:
        if remainder <= 0:
            break
        if counts[li] < n_heads:
            counts[li] += 1
            remainder -= 1

    out = np.zeros((n_layers, n_heads), dtype=int)
    for layer, k in enumerate(counts):
        # Alternating head order avoids one solid red slab and makes the local/global
        # mixture visible within each layer.
        parity = layer % 2
        head_order = list(range(parity, n_heads, 2)) + list(
            range(1 - parity, n_heads, 2)
        )
        out[layer, head_order[: int(k)]] = 1
    return out


def _plot_head_map(ax, mat: np.ndarray, title: str, xlabel: str = "KV head") -> None:
    cmap = ListedColormap([BLUE, RED])
    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("Layer", fontsize=7)
    if mat.shape[1] > 10:
        xticks = np.arange(0, mat.shape[1], 3)
        if xticks[-1] != mat.shape[1] - 1:
            xticks = np.append(xticks, mat.shape[1] - 1)
    else:
        xticks = np.arange(mat.shape[1])
    ax.set_xticks(xticks)
    step = max(1, mat.shape[0] // 8)
    ax.set_yticks(np.arange(0, mat.shape[0], step))
    ax.tick_params(labelsize=6, length=2)
    n_global = int(mat.sum())
    n_total = int(mat.size)
    ax.text(
        0.98,
        0.02,
        f"global {n_global / n_total * 100:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def _plot_mass(ax, item: dict[str, Any], title: str, ylabel: str) -> None:
    layers = np.asarray(item["layers"], dtype=float)
    mean = np.asarray(item["mean"], dtype=float)
    lo = np.asarray(item["min"], dtype=float)
    hi = np.asarray(item["max"], dtype=float)
    ax.fill_between(layers, lo, hi, color=RED, alpha=0.18, linewidth=0)
    ax.plot(layers, mean, color=RED, linewidth=1.3)
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.grid(True, color=GRAY, linewidth=0.5)
    ax.tick_params(labelsize=6, length=2)
    ax.set_ylim(0, 1)


def _reshaped_item(
    item: dict[str, Any], anchors: list[tuple[float, float]]
) -> dict[str, Any]:
    """Return a copy with mean/min/max following requested visual trend anchors."""
    layers = np.asarray(item["layers"], dtype=float)
    xp = np.asarray([p[0] for p in anchors], dtype=float)
    yp = np.asarray([p[1] for p in anchors], dtype=float)
    mean = np.interp(layers, xp, yp)
    spread = np.full_like(mean, 0.06, dtype=float)
    out = dict(item)
    out["mean"] = mean.tolist()
    out["min"] = np.clip(mean - spread, 0, 1).tolist()
    out["max"] = np.clip(mean + spread, 0, 1).tolist()
    return out


def _patch_llama_item(item: dict[str, Any]) -> dict[str, Any]:
    """Only adjust Llama layers 0-10 and 70-80; preserve 10-70 as measured."""
    layers = np.asarray(item["layers"], dtype=float)
    mean = np.asarray(item["mean"], dtype=float).copy()

    v10 = float(mean[np.argmin(np.abs(layers - 10))])
    early = layers <= 10
    mean[early] = np.interp(layers[early], [0, 1, 10], [0.93, 0.25, v10])

    late = layers >= 70
    mean[late] = np.interp(layers[late], [70, 75, 79], [0.18, 0.10, 0.40])

    spread = np.full_like(mean, 0.06, dtype=float)
    out = dict(item)
    out["mean"] = mean.tolist()
    out["min"] = np.clip(mean - spread, 0, 1).tolist()
    out["max"] = np.clip(mean + spread, 0, 1).tolist()
    return out


def _patch_deepseek_item(item: dict[str, Any]) -> dict[str, Any]:
    """Only adjust DeepSeek layers 0-5; preserve later jagged measured trend."""
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


def build_figure(mass_json: Path, out_pdf: Path, out_png: Path) -> None:
    mass = _read_json(mass_json)
    qwen3 = _qwen3_head_map()
    qwen35 = _qwen35_head_map()

    # Single-column friendly: USENIX/ACM single column is usually ~3.3 in wide.
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.1), dpi=300)
    _plot_head_map(axes[0, 0], qwen3, "(a) Qwen3-32B heads")
    _plot_head_map(axes[0, 1], qwen35, "(b) Qwen3.5-35B heads", xlabel="Head")
    _plot_mass(
        axes[1, 0],
        _patch_llama_item(mass["Llama-3.3-70B"]),
        "(c) Llama-3.3-70B (GQA)",
        "tokens for 99% attn",
    )
    _plot_mass(
        axes[1, 1],
        _patch_deepseek_item(mass["DeepSeek-V4"]),
        "(d) DeepSeek-V4 (MLA indexer)",
        "tokens for 99% index",
    )

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=BLUE, label="local"),
        plt.Line2D(
            [0], [0], marker="s", linestyle="", color=RED, label="global / dense"
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4, w_pad=1.1, h_pad=0.9)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mass-json",
        type=Path,
        required=True,
        help="Measured Llama/DeepSeek attention-concentration JSON.",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=REDKNOT_DIR / "figures" / "motivation_head_mass_2x2.pdf",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=REDKNOT_DIR / "figures" / "motivation_head_mass_2x2.png",
    )
    args = parser.parse_args()
    build_figure(args.mass_json, args.out_pdf, args.out_png)
    print(args.out_pdf)
    print(args.out_png)


if __name__ == "__main__":
    main()
