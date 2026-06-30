#!/usr/bin/env python3
"""Plot unified SegPaged-vs-Paged KV ratio across models (log y-axis).

Reads figures/segpaged_ratio_multimodel.json. Line chart styled after
折线图示例.png (thin lines, small open markers). Log y so that the GQA models
(1.2-1.5x) and the MLA+indexer model (10-15x) are both legible.

Story: SegPaged's benefit over uniform-block Paged grows with how SCATTERED the
known sparse pattern is. Head-window sparsity (GQA) gives ~1.2-1.5x; token-level
indexer sparsity (MLA) is highly scattered -> ~10-15x.

Output: figures/RedKnot_Figures/segpaged_ratio_multimodel.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = FIG / "RedKnot_Figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.labelsize": 15.5,
        "legend.fontsize": 11.5,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.linewidth": 1.1,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "figure.dpi": 120,
    }
)

SERIES = [
    ("DeepSeek-V4-Flash", "DeepSeek-V4-Flash (MLA + indexer)", "#c1121f", "o"),
    ("Qwen3-32B", "Qwen3-32B (GQA-8, measured)", "#3d405b", "x"),
    ("Llama-3.3-70B", "Llama-3.3-70B (GQA-8, measured)", "#e8833a", "^"),
]


def main():
    d = json.loads((FIG / "segpaged_ratio_multimodel.json").read_text())
    lengths = d["Qwen3-32B"]["lengths"]
    xpos = list(range(len(lengths)))
    xlab = [f"{L // 1000}K" for L in lengths]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for key, label, color, marker in SERIES:
        y = d[key]["segpaged_ratio"]
        filled = marker == "x"
        ax.plot(
            xpos,
            y,
            color=color,
            lw=1.6,
            marker=marker,
            ms=8,
            mfc=(color if filled else "white"),
            mec=color,
            mew=1.6,
            label=label,
            zorder=3,
        )

    ax.axhline(1.0, color="grey", lw=1.0, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("SegPaged vs. Paged KV ratio (log)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlab)
    ax.set_ylim(0.95, 25)
    ax.set_yticks([1, 2, 3, 5, 7, 10, 20])
    ax.set_yticklabels(["1", "2", "3", "5", "7", "10", "20"])
    ax.legend(loc="lower right", borderpad=0.55)

    # clean group labels (no arrows) placed in clear regions
    ax.text(
        1.5,
        17.5,
        "token-level (scattered) sparsity",
        fontsize=10.5,
        color="#7a1212",
        ha="center",
        style="italic",
    )
    ax.text(
        1.5,
        3.4,
        "head-window sparsity (GQA)",
        fontsize=10.5,
        color="#3d405b",
        ha="center",
        style="italic",
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            OUT / f"segpaged_ratio_multimodel.{ext}", bbox_inches="tight", dpi=200
        )
    plt.close(fig)
    print(f"wrote {OUT}/segpaged_ratio_multimodel.pdf (+.png)")


if __name__ == "__main__":
    main()
