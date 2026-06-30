#!/usr/bin/env python3
"""RedKnot quality / TTFT Pareto figure (Qwen3-32B & Llama-3.3-70B).

Layout: 3 rows (datasets) x 2 columns (models).
  - Color encodes the METHOD (Dense / CacheBlend / ProphetKV / RedKnot).
  - Marker SHAPE encodes the TEST LENGTH (16K / 32K).
  - Two legends: one maps color->method, one maps shape->length.
  - No per-point text labels.

Axes:
  - x = normalized TTFT = TTFT / TTFT_dense (lower = better). RedKnot uses
    measured wall-clock TTFT normalized by the dense run; token-level baselines
    use the paper's recompute-ratio analytic normalization.
  - y = token-level F1 (higher = better). Values that measured ~0.0 are floored
    to 0.1 for plotting visibility (raw value kept in comments).

All points are single-sample local smoke runs on this workspace
(Qwen3-32B INT4 single-GPU; Llama-3.3-70B INT4 single-GPU for RedKnot, bf16
multi-GPU for token-level baselines). Numbers are recorded in the comments next
to each entry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REDKNOT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = REDKNOT_DIR / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

MODELS = ["Qwen3-32B", "Llama-3.3-70B", "Qwen3.5-35B-A3B"]
DATASETS = ["MultiFieldQA", "HotpotQA", "MuSiQue"]

# Models whose normalized TTFT comes from the analytic theoretical speedup
# (multi-GPU device_map, comm-bound wall clock); flagged in the title.
THEORETICAL_TTFT_MODELS = {"Qwen3.5-35B-A3B"}

# Color encodes method.
METHOD_COLOR = {
    "Dense": "#9ca3af",
    "CacheBlend": "#3b82f6",
    "ProphetKV": "#d8a21b",
    "RedKnot": "#dc2626",
}
METHOD_SIZE = {"Dense": 70, "CacheBlend": 70, "ProphetKV": 78, "RedKnot": 150}
METHOD_Z = {"Dense": 2, "CacheBlend": 3, "ProphetKV": 3, "RedKnot": 5}

# Shape encodes test length.
LENGTH_MARKER = {"16K": "o", "32K": "*"}

FLOOR = 0.10  # plot floor for ~0.0 F1


# Each entry: (model, dataset) -> { length: {method: (nttft, f1)} }
DATA = {
    # ───────────────── Qwen3-32B (INT4, single-GPU) ─────────────────
    ("Qwen3-32B", "MultiFieldQA"): {
        # 16K: dense 1.0 -> rk 0.75, TTFT 3.42->1.95 (1.75x); CB/PK F1=0.25.
        "16K": {
            "Dense": (1.00, 1.00),
            "RedKnot": (1.95 / 3.42, 0.75),
            "CacheBlend": (0.50, 0.25),
            "ProphetKV": (0.53, 0.25),
        },
        # 32K: dense 0.75 -> rk 0.75, TTFT 7.78->4.04 (1.93x); CB/PK 0.0->floor.
        "32K": {
            "Dense": (1.00, 0.75),
            "RedKnot": (4.04 / 7.78, 0.75),
            "CacheBlend": (0.44, FLOOR),
            "ProphetKV": (0.48, FLOOR),
        },
    },
    ("Qwen3-32B", "HotpotQA"): {
        # 16K: dense 0.33 -> rk 0.33, TTFT 3.42->1.97 (1.74x); CB/PK 0.33.
        "16K": {
            "Dense": (1.00, 0.33),
            "RedKnot": (1.97 / 3.42, 0.33),
            "CacheBlend": (0.50, 0.33),
            "ProphetKV": (0.53, 0.33),
        },
        # 32K: dense 1.0 -> rk 0.33, TTFT 7.77->4.04 (1.92x); CB 0.33 / PK 1.0.
        "32K": {
            "Dense": (1.00, 1.00),
            "RedKnot": (4.04 / 7.77, 0.33),
            "CacheBlend": (0.44, 0.33),
            "ProphetKV": (0.48, 1.00),
        },
    },
    ("Qwen3-32B", "MuSiQue"): {
        # 16K: dense 0.0 -> rk 0.80, TTFT 3.43->1.96 (1.75x); CB/PK 0.0->floor.
        "16K": {
            "Dense": (1.00, FLOOR),
            "RedKnot": (1.96 / 3.43, 0.80),
            "CacheBlend": (0.50, FLOOR),
            "ProphetKV": (0.53, FLOOR),
        },
        # 32K: dense 0.0 -> rk 0.0, TTFT 7.78->4.14 (1.88x); all floor.
        "32K": {
            "Dense": (1.00, FLOOR),
            "RedKnot": (4.14 / 7.78, FLOOR),
            "CacheBlend": (0.44, FLOOR),
            "ProphetKV": (0.48, FLOOR),
        },
    },
    # ───────────── Llama-3.3-70B (RedKnot INT4 1-GPU; baselines bf16) ─────────
    ("Llama-3.3-70B", "MultiFieldQA"): {
        # 16K: dense 0.33 -> rk 0.33, TTFT 3.73->2.46 (1.51x); CB/PK 0.0->floor.
        "16K": {
            "Dense": (1.00, 0.33),
            "RedKnot": (2.46 / 3.73, 0.33),
            "CacheBlend": (0.50, FLOOR),
            "ProphetKV": (0.53, FLOOR),
        },
        # 32K: dense 1.0 -> rk 0.75, TTFT 12.54->6.90 (1.82x); CB/PK 0.0->floor.
        "32K": {
            "Dense": (1.00, 1.00),
            "RedKnot": (6.90 / 12.54, 0.75),
            "CacheBlend": (0.44, FLOOR),
            "ProphetKV": (0.48, FLOOR),
        },
    },
    ("Llama-3.3-70B", "HotpotQA"): {
        # 16K: dense 0.33 -> rk 1.0, TTFT 4.03->2.50 (1.61x); CB/PK 0.33.
        "16K": {
            "Dense": (1.00, 0.33),
            "RedKnot": (2.50 / 4.03, 1.00),
            "CacheBlend": (0.50, 0.33),
            "ProphetKV": (0.53, 0.33),
        },
        # 32K: dense 0.33 -> rk 0.75, TTFT 12.88->6.86 (1.88x); CB/PK 0.33.
        "32K": {
            "Dense": (1.00, 0.33),
            "RedKnot": (6.86 / 12.88, 0.75),
            "CacheBlend": (0.44, 0.33),
            "ProphetKV": (0.48, 0.33),
        },
    },
    ("Llama-3.3-70B", "MuSiQue"): {
        # 16K: dense 0.0 -> rk 0.0, TTFT 5.54->3.42 (1.62x); all floor.
        "16K": {
            "Dense": (1.00, FLOOR),
            "RedKnot": (3.42 / 5.54, FLOOR),
            "CacheBlend": (0.50, FLOOR),
            "ProphetKV": (0.53, FLOOR),
        },
        # 32K: dense 0.0 -> rk 0.0, TTFT 12.55->6.93 (1.81x); all floor.
        "32K": {
            "Dense": (1.00, FLOOR),
            "RedKnot": (6.93 / 12.55, FLOOR),
            "CacheBlend": (0.44, FLOOR),
            "ProphetKV": (0.48, FLOOR),
        },
    },
    # ───── Qwen3.5-35B-A3B (hybrid MoE, bf16 multi-GPU), 10-sample mean ─────
    # RedKnot nTTFT = 1 / theoretical TTFT speedup (compute-bound; comm excluded).
    # Dense = standard run (stdF1), fixed at nTTFT=1.0. F1 are 10-sample averages.
    ("Qwen3.5-35B-A3B", "MultiFieldQA"): {
        # 16K: std 0.629 -> rk 0.613, theo speedup 1.896x; CB 0.076 / PK 0.043.
        "16K": {
            "Dense": (1.00, 0.629),
            "RedKnot": (1.0 / 1.896, 0.613),
            "CacheBlend": (0.50, FLOOR),  # raw 0.076
            "ProphetKV": (0.53, FLOOR),  # raw 0.043
        },
        # 32K: std 0.598 -> rk 0.558, theo speedup 2.030x; CB 0.012 / PK 0.051.
        "32K": {
            "Dense": (1.00, 0.598),
            "RedKnot": (1.0 / 2.030, 0.558),
            "CacheBlend": (0.44, FLOOR),  # raw 0.012
            "ProphetKV": (0.48, FLOOR),  # raw 0.051
        },
    },
    ("Qwen3.5-35B-A3B", "HotpotQA"): {
        # 16K: std 0.413 -> rk 0.513, theo speedup 1.840x; CB 0.180 / PK 0.190.
        "16K": {
            "Dense": (1.00, 0.413),
            "RedKnot": (1.0 / 1.840, 0.513),
            "CacheBlend": (0.50, 0.180),
            "ProphetKV": (0.53, 0.190),
        },
        # 32K: std 0.447 -> rk 0.347, theo speedup 1.977x; CB 0.290 / PK 0.230.
        "32K": {
            "Dense": (1.00, 0.447),
            "RedKnot": (1.0 / 1.977, 0.347),
            "CacheBlend": (0.44, 0.290),
            "ProphetKV": (0.48, 0.230),
        },
    },
    ("Qwen3.5-35B-A3B", "MuSiQue"): {
        # 16K: std 0.330 -> rk 0.330, theo speedup 1.848x; CB 0.200 / PK 0.200.
        "16K": {
            "Dense": (1.00, 0.330),
            "RedKnot": (1.0 / 1.848, 0.330),
            "CacheBlend": (0.50, 0.200),
            "ProphetKV": (0.53, 0.200),
        },
        # 32K: std 0.300 -> rk 0.100, theo speedup 1.989x; CB 0.200 / PK 0.0->floor.
        "32K": {
            "Dense": (1.00, 0.300),
            "RedKnot": (1.0 / 1.989, 0.100),
            "CacheBlend": (0.44, 0.200),
            "ProphetKV": (0.48, FLOOR),  # raw 0.0
        },
    },
}


def draw_panel(ax: plt.Axes, model: str, dataset: str) -> None:
    cell = DATA.get((model, dataset), {})
    ax.set_xlim(0.30, 1.08)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, color="#d9dee5", lw=0.55, ls=":", zorder=0)
    ax.axvline(1.0, color="#c7cbd1", lw=0.8, ls="--", zorder=1)
    ax.tick_params(labelsize=8, length=2.5, width=0.8)
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    if not cell:
        ax.text(
            0.5,
            0.5,
            "pending\nrun",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="#9ca3af",
            fontweight="bold",
        )
        return

    for length, methods in cell.items():
        marker = LENGTH_MARKER.get(length, "o")
        for method, (x, y) in methods.items():
            ax.scatter(
                x,
                y,
                s=METHOD_SIZE[method] * (1.35 if marker == "*" else 1.0),
                marker=marker,
                color=METHOD_COLOR[method],
                edgecolor="white",
                linewidth=0.6,
                zorder=METHOD_Z[method],
            )


def main() -> None:
    fig, axes = plt.subplots(
        len(DATASETS),
        len(MODELS),
        figsize=(10.6, 8.6),
        sharex=True,
        sharey=True,
    )

    for r, dataset in enumerate(DATASETS):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            draw_panel(ax, model, dataset)
            if r == 0:
                title = model
                if model in THEORETICAL_TTFT_MODELS:
                    title += "\n(theoretical TTFT)"
                ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(dataset + "\nF1", fontsize=12, fontweight="bold")
            if r == len(DATASETS) - 1:
                ax.set_xlabel("Normalized TTFT", fontsize=11)

    # Legend 1: color -> method.
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=METHOD_COLOR[m],
            markeredgecolor="white",
            markersize=10,
            label=m,
        )
        for m in METHOD_COLOR
    ]
    # Legend 2: shape -> length (neutral color).
    length_handles = [
        Line2D(
            [0],
            [0],
            marker=LENGTH_MARKER[l],
            linestyle="none",
            markerfacecolor="#444444",
            markeredgecolor="white",
            markersize=11 if LENGTH_MARKER[l] == "*" else 9,
            label=l,
        )
        for l in LENGTH_MARKER
    ]

    leg1 = fig.legend(
        handles=method_handles,
        title="Method (color)",
        loc="upper center",
        bbox_to_anchor=(0.30, 0.965),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        title_fontsize=10,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=length_handles,
        title="Test length (shape)",
        loc="upper center",
        bbox_to_anchor=(0.70, 0.965),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        title_fontsize=10,
        handletextpad=0.4,
        columnspacing=1.1,
    )

    fig.suptitle(
        "Quality-Latency Pareto: RedKnot vs Token-level PIC Baselines",
        y=1.07,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Lower normalized TTFT is better; Dense fixed at 1.0. "
        "F1\u22480.0 floored to 0.1 for visibility. "
        "Qwen3.5 = 10-sample mean (theoretical, compute-bound TTFT); "
        "Qwen3-32B / Llama-3.3-70B = single-sample smoke runs.",
        ha="center",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, top=0.85, bottom=0.075, wspace=0.08, hspace=0.16
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "quality_ttft_pareto_3x4.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
