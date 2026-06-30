#!/usr/bin/env python3
"""Bar chart: per-100-token state storage (MB), models ~2B to ~397B.

Styled after RedKnot_Figures/7.png: thick black axes frame, bold labels/ticks,
grey + red bars with black edges, value labels on top, framed horizontal legend,
dashed y-grid.

Each bar is split (stacked) into:
  * growing KV cache (full-attention K/V)  — grey
  * fixed recurrent state (linear-attn)    — red  (0 for GQA/MLA)

x = model, y = storage (MB, bf16).
Output: figures/RedKnot_Figures/kv_storage_100tok.{pdf,png}
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
        "font.weight": "bold",
        "font.size": 14,
        "axes.labelsize": 17,
        "axes.labelweight": "bold",
        "legend.fontsize": 13,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 14,
        "axes.axisbelow": True,
        "axes.linewidth": 2.0,  # thick frame like 7.png
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "figure.dpi": 120,
    }
)

C_KV = "#c7c7c7"  # growing KV cache — light grey (7.png style)
C_LIN = "#c0392b"  # fixed recurrent state — brick red (7.png style)


def main():
    d = json.loads((FIG / "kv_storage_100tok.json").read_text())
    rows = sorted(d["rows"], key=lambda r: r["params_b"])
    names = [f"{r['model'].replace('-', '-')}\n{r['params_b']}B" for r in rows]
    kv = [r["kv_only_mb_100tok"] for r in rows]
    lin = [r.get("lin_mb", 0.0) for r in rows]
    total = [r["kv_mb_100tok"] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(15, 5.8))
    ax.grid(axis="y", alpha=0.5, linestyle=":", linewidth=1.0, color="grey")

    ax.bar(
        x,
        kv,
        width=0.62,
        color=C_KV,
        edgecolor="black",
        linewidth=1.6,
        label="KV cache (full-attn)",
        zorder=3,
    )
    ax.bar(
        x,
        lin,
        width=0.62,
        bottom=kv,
        color=C_LIN,
        edgecolor="black",
        linewidth=1.6,
        label="Recurrent state (linear-attn)",
        zorder=3,
    )

    for xi, t in zip(x, total):
        ax.text(
            xi,
            t + max(total) * 0.015,
            f"{t:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Storage / 100 tokens (MB)", fontweight="bold")
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontweight="bold")
    ax.set_ylim(0, max(total) * 1.18)

    # thick black frame on all four sides (like 7.png)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(2.0)
        s.set_color("black")

    ax.legend(
        loc="upper left", ncol=2, borderpad=0.5, columnspacing=1.3, handletextpad=0.5
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"kv_storage_100tok.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}/kv_storage_100tok.pdf (+.png)")


if __name__ == "__main__":
    main()
