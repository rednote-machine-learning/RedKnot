#!/usr/bin/env python3
"""Plot heterogeneous-window SegPaged-vs-Paged KV ratio across models.

Reads figures/hetero_ratio_multimodel.json and draws one line chart (styled
after 折线图示例.png): thin lines, small open markers.
  x = context length, y = SegPaged-vs-Paged KV ratio under heterogeneous windows.

Qwen3-32B / Qwen3.5-397B are GQA -> per-head heterogeneity yields 1.2-2.0x.
DeepSeek-V4 is MLA (shared latent) -> ratio == 1.0 (mechanism not applicable);
shown as a flat baseline for contrast.

Output: figures/RedKnot_Figures/hetero_ratio_multimodel.{pdf,png}
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
OUT = FIG / "RedKnot_Figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
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

# (json key, label, color, marker)
SERIES = [
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B (GQA-2)", "#c1121f", "o"),
    ("Qwen3-32B", "Qwen3-32B (GQA-8)", "#3d405b", "x"),
    ("DeepSeek-V4-Flash", "DeepSeek-V4-Flash (MLA)", "#9a9a9a", "s"),
]


def main():
    d = json.loads((FIG / "hetero_ratio_multimodel.json").read_text())
    lengths = d["Qwen3-32B"]["lengths"]
    x = [L // 1000 for L in lengths]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for key, label, color, marker in SERIES:
        y = d[key]["hetero_ratio"]
        filled = marker == "x"  # x is naturally "filled"
        ax.plot(
            x,
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
    ax.set_xlabel("Context length (K tokens)")
    ax.set_ylabel("Hetero-window KV ratio\n(SegPaged vs. Paged)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}K" for v in x])
    ax.set_ylim(0.9, 2.28)
    ax.legend(loc="upper left", borderpad=0.55)
    # annotate the MLA flat line
    ax.annotate(
        "MLA: shared latent (no per-head gain)",
        xy=(x[2], 1.0),
        xytext=(x[1] + 4, 1.15),
        fontsize=10.5,
        color="#6a6a6a",
        arrowprops=dict(arrowstyle="->", color="#9a9a9a", lw=1),
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            OUT / f"hetero_ratio_multimodel.{ext}", bbox_inches="tight", dpi=200
        )
    plt.close(fig)
    print(f"wrote {OUT}/hetero_ratio_multimodel.pdf (+.png)")


if __name__ == "__main__":
    main()
