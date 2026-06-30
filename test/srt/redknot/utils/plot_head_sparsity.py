#!/usr/bin/env python3
"""Plot per-layer head-union attention sparsity for Qwen3-32B and Llama-3.3-70B.

Reads figures/head_sparsity_{qwen,llama}.json (produced by measure_head_sparsity.py)
and draws, for each model, a line plot:
  x = layer index, y = |union of per-head essential tokens| / context length.

The union ratio is high in shallow layers (heads attend to diverse tokens) and
falls in deeper layers (heads concentrate on shared tokens), validating the
per-head heterogeneous-sparsity premise. A trend line is overlaid.

Outputs PDF + PNG into figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RKFIG = HERE / "paper" / "RedKnot_Figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "legend.frameon": False,
        "figure.dpi": 120,
        "lines.linewidth": 2.0,
    }
)

COLORS = {
    "qwen": "#2e86ab",  # deep blue
    "llama": "#d1495b",  # crimson
    "trend": "#3d405b",  # slate
    "fill": "#2e86ab",
}
TITLES = {
    "qwen": "Qwen3-32B (64 layers)",
    "llama": "Llama-3.3-70B (80 layers)",
}


def _smooth(y, w=5):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, k, mode="valid")[: len(y)]


def plot_one(ax, model_key):
    d = json.loads((FIG / f"head_sparsity_{model_key}.json").read_text())
    u = np.array(d["layer_union_ratio"]) * 100.0  # to percent
    L = len(u)
    x = np.arange(L)
    c = COLORS[model_key]

    ax.plot(
        x, u, "-o", color=c, ms=3.5, lw=1.6, alpha=0.85, label="per-layer union ratio"
    )
    # smoothed trend
    sm = _smooth(u, 5)
    ax.plot(
        x, sm, "-", color=COLORS["trend"], lw=2.6, alpha=0.9, label="trend (smoothed)"
    )
    ax.fill_between(x, u, sm.min() if hasattr(sm, "min") else 0, color=c, alpha=0.06)

    ax.set_title(f"{TITLES[model_key]}", fontweight="bold")
    ax.set_xlabel("Layer index (shallow $\\rightarrow$ deep)")
    ax.set_ylabel("Union of essential tokens (% of context)")
    ax.set_ylim(0, 105)
    ax.set_xlim(-1, L)
    ax.legend(loc="lower left")
    # annotate shallow vs deep means
    sh = u[: L // 4].mean()
    dp = u[L // 2 : L * 3 // 4].mean()
    ax.annotate(
        f"shallow mean ≈ {sh:.0f}%",
        xy=(L * 0.08, sh),
        xytext=(L * 0.10, 30),
        fontsize=9,
        color=c,
        arrowprops=dict(arrowstyle="->", color=c, lw=1),
    )
    ax.annotate(
        f"deep mean ≈ {dp:.0f}%",
        xy=(L * 0.62, dp),
        xytext=(L * 0.45, 18),
        fontsize=9,
        color=COLORS["trend"],
        arrowprops=dict(arrowstyle="->", color=COLORS["trend"], lw=1),
    )


def plot_massfrac(ax, model_key):
    """Layer-level token-mass sparsity: fraction of tokens carrying THRESH of the
    layer's total attention mass (the Sparse-FFN selector)."""
    d = json.loads((FIG / f"head_sparsity_{model_key}.json").read_text())
    m = np.array(d["layer_massfrac_ratio"]) * 100.0  # percent
    L = len(m)
    x = np.arange(L)
    c = COLORS[model_key]

    ax.plot(
        x,
        m,
        "-o",
        color=c,
        ms=3.5,
        lw=1.6,
        alpha=0.85,
        label="tokens carrying 90% mass",
    )
    sm = _smooth(m, 5)
    ax.plot(
        x, sm, "-", color=COLORS["trend"], lw=2.6, alpha=0.9, label="trend (smoothed)"
    )
    ax.fill_between(x, m, 0, color=c, alpha=0.08)

    ax.set_title(f"{TITLES[model_key]}", fontweight="bold")
    ax.set_xlabel("Layer index (shallow $\\rightarrow$ deep)")
    ax.set_ylabel("Top tokens w/ 90% attn mass (% of context)")
    ax.set_xlim(-1, L)
    ax.set_ylim(0, max(5, m.max() * 1.1))
    ax.legend(loc="upper right")
    sh = m[: L // 4].mean()
    dp = m[L // 4 : L * 3 // 4].mean()
    ax.annotate(
        f"shallow mean ≈ {sh:.1f}%",
        xy=(L * 0.05, sh),
        xytext=(L * 0.12, max(sh, m.max() * 0.55)),
        fontsize=9,
        color=c,
        arrowprops=dict(arrowstyle="->", color=c, lw=1),
    )
    ax.annotate(
        f"deep mean ≈ {dp:.1f}%",
        xy=(L * 0.6, dp),
        xytext=(L * 0.5, m.max() * 0.30),
        fontsize=9,
        color=COLORS["trend"],
        arrowprops=dict(arrowstyle="->", color=COLORS["trend"], lw=1),
    )


def main():
    # ===== union figure (per-head heterogeneity) =====
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    plot_one(axes[0], "qwen")
    plot_one(axes[1], "llama")
    thr = int(
        json.loads((FIG / "head_sparsity_qwen.json").read_text())["threshold"] * 100
    )
    fig.suptitle(
        f"Per-head attention sparsity: union of essential tokens (cum. mass {thr}%) "
        "shrinks with depth",
        fontsize=12.5,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"head_sparsity_combined.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {FIG}/head_sparsity_combined.pdf (+.png)")

    for mk in ("qwen", "llama"):
        f, a = plt.subplots(figsize=(7, 4.4))
        plot_one(a, mk)
        f.tight_layout()
        for ext in ("pdf", "png"):
            f.savefig(FIG / f"head_sparsity_{mk}.{ext}", bbox_inches="tight", dpi=150)
        plt.close(f)
        print(f"wrote {FIG}/head_sparsity_{mk}.pdf (+.png)")

    # ===== mass-fraction figure (Sparse-FFN token selector) =====
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    plot_massfrac(axes[0], "qwen")
    plot_massfrac(axes[1], "llama")
    fig.suptitle(
        f"Sparse-FFN token selector: fraction of tokens carrying {thr}% of the "
        "layer's attention mass drops sharply with depth",
        fontsize=12.5,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"ffn_massfrac_combined.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {FIG}/ffn_massfrac_combined.pdf (+.png)")

    for mk in ("qwen", "llama"):
        f, a = plt.subplots(figsize=(7, 4.4))
        plot_massfrac(a, mk)
        f.tight_layout()
        for ext in ("pdf", "png"):
            f.savefig(FIG / f"ffn_massfrac_{mk}.{ext}", bbox_inches="tight", dpi=150)
        plt.close(f)
        print(f"wrote {FIG}/ffn_massfrac_{mk}.pdf (+.png)")

    # ===== one-row four-panel figure (union x2, massfrac x2) =====
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.6))
    plot_one(axes[0], "qwen")
    plot_one(axes[1], "llama")
    plot_massfrac(axes[2], "qwen")
    plot_massfrac(axes[3], "llama")
    # subpanel tags (a)-(d)
    for ax, tag in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(
            -0.08,
            1.06,
            tag,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )
    fig.tight_layout()
    out_dirs = [FIG, RKFIG]
    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(od / f"sparsity_fourpanel.{ext}", bbox_inches="tight", dpi=150)
        print(f"wrote {od}/sparsity_fourpanel.pdf (+.png)")
    plt.close(fig)

    # also mirror the individual + combined figures into RedKnot_Figures
    import shutil

    RKFIG.mkdir(parents=True, exist_ok=True)
    for name in (
        "head_sparsity_combined",
        "ffn_massfrac_combined",
        "head_sparsity_qwen",
        "head_sparsity_llama",
        "ffn_massfrac_qwen",
        "ffn_massfrac_llama",
    ):
        for ext in ("pdf", "png"):
            src = FIG / f"{name}.{ext}"
            if src.exists():
                shutil.copy(src, RKFIG / f"{name}.{ext}")
    print(f"mirrored all sparsity figures into {RKFIG}")


if __name__ == "__main__":
    main()
