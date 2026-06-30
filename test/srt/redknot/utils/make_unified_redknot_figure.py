#!/usr/bin/env python3
"""Unified RedKnot results figure for the paper (two-column, figure* width).

Combines into a single, style-consistent figure:
  Row 1  (Eva_1.pdf)  : (a) TTFT speedup, (b) Exact-Match accuracy,
                        (c) Token-level F1, (d) Top-K match
                        -- Dense vs RedKnot vs CacheBlend vs ProphetKV.
  Row 2  (Qwen3.5-397B): (e) 16K, (f) 32K, (g) 64K  std-F1 vs RedKnot-F1,
                        (h) RedKnot TTFT speedup across lengths.

Outputs PDF (vector, for LaTeX) and PNG (preview) into ../figures/.
Run:  python utils/make_unified_redknot_figure.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Global style — LaTeX-like serif (STIX), tuned for a two-column figure*.
# ----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "figure.dpi": 200,
    }
)

# Unified palette (matches the original Eva_1.pdf scheme).
C_DENSE = "#BFBFBF"  # grey  — Dense / std baseline
C_RED = "#C0392B"  # red   — RedKnot (ours)
C_BLUE = "#3B6FA0"  # blue  — CacheBlend
C_GOLD = "#D4A12A"  # gold  — ProphetKV
EDGE = "#3a3a3a"

# ----------------------------------------------------------------------------
# Row 1 data — multi-method comparison (read from Eva_1.pdf).
# ----------------------------------------------------------------------------
EVA_X = [
    "M-TQA-16K",
    "Q-MFQA-24K",
    "L70-HQA-32K",
    "L70-HQA-64K",
]
# Which model each row-1 column belongs to: (model name, [x-indices]).
EVA_MODEL_GROUPS = [
    ("Mistral", [0]),
    ("Qwen3", [1]),
    ("Llama-3.3", [2, 3]),
]
METHODS = ["Dense", "RedKnot", "CacheBlend", "ProphetKV"]
M_COLORS = [C_DENSE, C_RED, C_BLUE, C_GOLD]

# (a) TTFT speedup (x)
ttft = {
    "Dense": [1.0, 1.0, 1.0, 1.0],
    "RedKnot": [1.60, 2.91, 2.93, 3.52],
    "CacheBlend": [1.78, 2.14, 1.98, 2.40],
    "ProphetKV": [1.38, 1.66, 1.74, 2.04],
}
# (b) Exact-Match accuracy
em = {
    "Dense": [1.0, 0.4, 0.6, 0.6],
    "RedKnot": [1.0, 0.4, 0.8, 0.8],
    "CacheBlend": [1.0, 0.0, 0.6, 0.4],
    "ProphetKV": [1.0, 0.0, 0.6, 0.6],
}
# (c) Token-level F1
f1 = {
    "Dense": [0.82, 0.60, 0.30, 0.30],
    "RedKnot": [1.0, 0.52, 0.30, 0.30],
    "CacheBlend": [0.81, 0.20, 0.35, 0.24],
    "ProphetKV": [0.82, 0.05, 0.35, 0.30],
}
# (d) Top-K match
topk_x = ["top-1", "top-10"]
topk = {
    "Dense": [1.0, 1.0],
    "RedKnot": [0.93, 0.87],
    "CacheBlend": [0.42, 0.38],
    "ProphetKV": [0.51, 0.41],
}

# ----------------------------------------------------------------------------
# Row 2 data — Qwen3.5-397B std-F1 vs RedKnot-F1 + TTFT, per length.
# ----------------------------------------------------------------------------
qwen = {
    "16K": {
        "ds": ["MFQA", "HQA", "2Wiki", "MuSiQue", "TriviaQA"],
        "std": [0.412, 0.250, 0.625, 0.284, 0.450],
        "rk": [0.378, 0.200, 0.525, 0.306, 0.300],
        "ttft": [2.05, 2.04, 2.03, 2.05, 2.06],
    },
    "32K": {
        "ds": ["MFQA", "HQA", "2Wiki", "MuSiQue", "NarrQA"],
        "std": [0.520, 0.450, 0.625, 0.306, 0.167],
        "rk": [0.533, 0.300, 0.675, 0.256, 0.167],
        "ttft": [2.16, 2.16, 2.15, 2.17, 2.19],
    },
    "64K": {
        "ds": ["NarrQA", "HQA", "2Wiki", "MuSiQue", "MFQA"],
        "std": [0.144, 0.250, 0.600, 0.256, 0.320],
        "rk": [0.122, 0.250, 0.625, 0.167, 0.358],
        "ttft": [2.31, 2.29, 2.29, 2.30, 2.29],
    },
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _fmt_2dp_trim(v):
    """2 decimals, but drop the 2nd decimal when it is 0 (e.g. 0.80 -> 0.8)."""
    s = f"{v:.2f}"
    if s.endswith("0"):
        s = f"{v:.1f}"
    return s


def _bar_label(ax, rects, vals, fmt="{:.2f}", fs=5.6, rot=0, stagger=None):
    """Annotate bar tops with horizontal numbers.

    ``stagger`` (a per-bar y-offset list, in points) lifts alternating labels
    onto a second tier so neighbouring numbers in a dense group never overlap.
    """
    for i, (r, v) in enumerate(zip(rects, vals)):
        dy = 1.6 if stagger is None else stagger[i]
        text = fmt(v) if callable(fmt) else fmt.format(v)
        ax.annotate(
            text,
            xy=(r.get_x() + r.get_width() / 2, r.get_height()),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fs,
            rotation=rot,
        )


def grouped_bars(
    ax,
    xlabels,
    series,
    colors,
    names,
    ylabel,
    title,
    fmt="{:.2f}",
    ann_fs=5.6,
    ymax=None,
    rot_x=22,
    rk_smart=False,
    model_groups=None,
):
    n = len(names)
    x = np.arange(len(xlabels))
    w = 0.8 / n
    # Equal label-to-bar distance for every bar (uniform offset, no staggering).
    label_dy = 1.8
    for i, name in enumerate(names):
        off = (i - (n - 1) / 2) * w
        rects = ax.bar(
            x + off,
            series[name],
            w,
            color=colors[i],
            edgecolor=EDGE,
            linewidth=0.4,
            zorder=3,
        )
        # RedKnot keeps the detailed format; all other methods round to 1 dp.
        if name == "RedKnot":
            lbl_fmt = _fmt_2dp_trim if rk_smart else fmt
        else:
            lbl_fmt = "{:.1f}"
        _bar_label(
            ax,
            rects,
            series[name],
            fmt=lbl_fmt,
            fs=ann_fs,
            stagger=[label_dy] * len(series[name]),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=rot_x, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.margins(x=0.02)
    if model_groups is not None:
        _draw_model_band(ax, x, model_groups)


def _draw_model_band(ax, x, model_groups):
    """Draw a model-grouping band (bracket + name) beneath the x-tick labels."""
    import matplotlib.transforms as mtransforms

    # Blended transform: x in data coords, y in axes-fraction (below the axis).
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    y_line = -0.30  # bracket height below axis (in axes fraction)
    y_text = -0.37
    for name, idxs in model_groups:
        x0 = x[idxs[0]] - 0.42
        x1 = x[idxs[-1]] + 0.42
        xc = (x0 + x1) / 2
        ax.plot(
            [x0, x1],
            [y_line, y_line],
            transform=tr,
            color=EDGE,
            lw=0.8,
            clip_on=False,
            zorder=5,
        )
        for xe in (x0, x1):
            ax.plot(
                [xe, xe],
                [y_line, y_line + 0.045],
                transform=tr,
                color=EDGE,
                lw=0.8,
                clip_on=False,
                zorder=5,
            )
        ax.text(
            xc,
            y_text,
            name,
            transform=tr,
            ha="center",
            va="top",
            fontsize=8.0,
            clip_on=False,
        )


def qwen_panel(ax, key, title):
    d = qwen[key]
    x = np.arange(len(d["ds"]))
    w = 0.38
    r1 = ax.bar(
        x - w / 2,
        d["std"],
        w,
        color=C_DENSE,
        edgecolor=EDGE,
        linewidth=0.4,
        label="std F1",
        zorder=3,
    )
    r2 = ax.bar(
        x + w / 2,
        d["rk"],
        w,
        color=C_RED,
        edgecolor=EDGE,
        linewidth=0.4,
        label="RedKnot F1",
        zorder=3,
    )
    # Equal label-to-bar distance for both series (same height as std).
    _bar_label(ax, r1, d["std"], fmt="{:.2f}", fs=6.6, stagger=[1.8] * len(d["std"]))
    _bar_label(ax, r2, d["rk"], fmt="{:.2f}", fs=6.6, stagger=[1.8] * len(d["rk"]))
    ax.set_xticks(x)
    ax.set_xticklabels(d["ds"], rotation=22, ha="right")
    ax.set_ylabel("F1")
    ax.set_title(title, pad=4)
    ax.set_ylim(0, 0.98)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.02)


# ----------------------------------------------------------------------------
# Figure layout: 2 rows x 4 cols (figure*-width, ~7.16in).
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(12.5, 4.6))

# ---- Row 1: Eva multi-method comparison ----
grouped_bars(
    axes[0, 0],
    EVA_X,
    ttft,
    M_COLORS,
    METHODS,
    r"Speedup ($\times$)",
    r"(a) TTFT speedup ($\times$)",
    fmt="{:.2f}",
    ann_fs=6.6,
    ymax=4.3,
)
grouped_bars(
    axes[0, 1],
    EVA_X,
    em,
    M_COLORS,
    METHODS,
    "EM",
    "(b) Exact-Match accuracy",
    fmt="{:.2f}",
    ann_fs=6.6,
    ymax=1.22,
    rk_smart=True,
)
grouped_bars(
    axes[0, 2],
    EVA_X,
    f1,
    M_COLORS,
    METHODS,
    "F1",
    "(c) Token-level F1",
    fmt="{:.2f}",
    ann_fs=6.6,
    ymax=1.22,
    rk_smart=True,
)
grouped_bars(
    axes[0, 3],
    topk_x,
    topk,
    M_COLORS,
    METHODS,
    "Match",
    r"(d) Top-$K$ match",
    fmt="{:.2f}",
    ann_fs=7.0,
    ymax=1.22,
    rot_x=0,
    rk_smart=True,
)

# ---- Row 2: Qwen3.5-397B ----
qwen_panel(axes[1, 0], "16K", "(e) 16K")
qwen_panel(axes[1, 1], "32K", "(f) 32K")
qwen_panel(axes[1, 2], "64K", "(g) 64K")

# (h) RedKnot TTFT speedup vs context length (mean over datasets).
axh = axes[1, 3]
lens = ["16K", "32K", "64K"]
mean_ttft = [float(np.mean(qwen[k]["ttft"])) for k in lens]
xpos = np.arange(len(lens))
bars = axh.bar(
    xpos, mean_ttft, 0.55, color=C_RED, edgecolor=EDGE, linewidth=0.4, zorder=3
)
_bar_label(axh, bars, mean_ttft, fmt="{:.2f}x", fs=6.5)
axh.axhline(1.0, color="#888", lw=0.7, ls=":")
axh.set_xticks(xpos)
axh.set_xticklabels(lens)
axh.set_ylabel(r"TTFT speedup ($\times$)")
axh.set_title("(h) RedKnot TTFT vs length", pad=4)
axh.set_ylim(0, 2.85)
axh.set_axisbelow(True)
axh.grid(axis="x", visible=False)

# ----------------------------------------------------------------------------
# Shared legends (one per row, centered above the row).
# ----------------------------------------------------------------------------
row1_handles = [
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markersize=8,
        markerfacecolor=c,
        markeredgecolor=EDGE,
        label={
            "Dense": "recompute",
            "RedKnot": "RedKnot (ours)",
            "CacheBlend": r"CacheBlend ($r{=}0.15$)",
            "ProphetKV": r"ProphetKV ($r{=}0.20$)",
        }[m],
    )
    for m, c in zip(METHODS, M_COLORS)
]
# Leading text-only entry naming the three row-1 models, then the method keys.
row1_handles = [
    Line2D(
        [0], [0], marker="none", linestyle="none", label="(Mistral, Qwen3, Llama-3.3)"
    )
] + row1_handles
leg1 = fig.legend(
    handles=row1_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.945),
    ncol=5,
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.3,
)

row2_handles = [
    # Leading text-only entry keeps the model name on the SAME line as the keys.
    Line2D([0], [0], marker="none", linestyle="none", label="Qwen3.5-397B-A17B:"),
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markersize=8,
        markerfacecolor=C_DENSE,
        markeredgecolor=EDGE,
        label="recomputed F1",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markersize=8,
        markerfacecolor=C_RED,
        markeredgecolor=EDGE,
        label="RedKnot F1",
    ),
]
fig.legend(
    handles=row2_handles,
    loc="center",
    bbox_to_anchor=(0.5, 0.44),
    ncol=3,
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.6,
)

fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.6, w_pad=1.1)
fig.subplots_adjust(top=0.905, hspace=0.95, wspace=0.30)

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.normpath(os.path.join(here, "..", "figures"))
pdf_path = os.path.join(outdir, "redknot_unified_results.pdf")
png_path = os.path.join(outdir, "redknot_unified_results.png")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight", dpi=240)
print("saved:", pdf_path)
print("saved:", png_path)
