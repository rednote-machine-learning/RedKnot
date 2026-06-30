#!/usr/bin/env python3
"""One-row, 4-panel QPS/GPU vs context-length figure in the qps_dsv4 style.

Panels (Mistral dropped):
  1. Qwen3-32B (TP=2)
  2. Llama-3.3-70B-Instruct (TP=4)
  3. Qwen3.5-397B-A17B (TP=8)
  4. DeepSeek-V4-Flash (FP8, PP=8)  -- our real measured + conservative model

Style follows figures/qps_dsv4.png: gray hollow-circle Recompute line, red
square RedKnot line, per-point speedup annotation.

Values for panels 1-3 are read from qps__comparison.pdf (Recompute vs RedKnot
only). Panel 4 uses the measured dense QPS + conservative RedKnot projection
from /tmp/qps_dsv4_final.json.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2/test/srt/redknot/figures/qps_4panel.png"
GRAY = "#9e9e9e"
RED = "#c0392b"  # deep red  (RedKnot)
TEAL = "#2a7f7f"  # dark teal (CacheBlend)
PLUM = "#6b4a7a"  # dark plum/purple (ProphetKV)

# Panels 1-3: digitized from qps__comparison.pdf
# (Recompute, CacheBlend r=15%, ProphetKV r=20%, RedKnot).
QWEN3 = dict(
    title="Qwen3-32B (TP=2)",
    xlabels=["8K", "16K", "32K"],
    recompute=[0.95, 0.48, 0.20],
    cacheblend=[2.58, 1.35, 0.55],
    prophetkv=[2.10, 1.10, 0.45],
    redknot=[1.30, 0.72, 0.35],
)
LLAMA = dict(
    title="Llama-3.3-70B-Instruct (TP=4)",
    xlabels=["16K", "64K", "128K"],
    recompute=[0.152, 0.042, 0.018],
    cacheblend=[0.43, 0.080, 0.040],
    prophetkv=[0.35, 0.070, 0.035],
    redknot=[0.205, 0.055, 0.028],
)
QWEN35 = dict(
    title="Qwen3.5-397B-A17B (TP=8)",
    xlabels=["16K", "32K", "64K"],
    recompute=[0.78, 0.38, 0.18],
    cacheblend=[2.20, 1.00, 0.45],
    prophetkv=[1.80, 0.78, 0.40],
    redknot=[1.40, 0.66, 0.26],
)


def load_dsv4():
    d = json.load(open("/tmp/qps_dsv4_final.json"))
    ctxs = ["16384", "32768", "65536", "131072"]
    return dict(
        title="DeepSeek-V4-Flash (PP=8)",
        xlabels=["16K", "32K", "64K", "128K"],
        recompute=[d["dense"][c] for c in ctxs],
        redknot=[d["redknot"][c] for c in ctxs],
    )


def plot_panel(ax, panel, show_ylabel=False, show_legend=False):
    x = list(range(len(panel["xlabels"])))
    series = []  # (values, fmt, color, lw, ms, mfc, mew, label)
    series.append((panel["recompute"], "-o", GRAY, 2.0, 8, "white", 2, "Recompute"))
    if "cacheblend" in panel:
        series.append(
            (panel["cacheblend"], "--D", TEAL, 1.8, 7, TEAL, 1, "CacheBlend (r=15%)")
        )
    if "prophetkv" in panel:
        series.append(
            (panel["prophetkv"], "--^", PLUM, 1.8, 7, "white", 1.6, "ProphetKV (r=20%)")
        )
    series.append((panel["redknot"], "-s", RED, 2.4, 8, RED, 1, "RedKnot (ours)"))

    allv = []
    for vals, fmt, color, lw, ms, mfc, mew, lab in series:
        ax.plot(
            x,
            vals,
            fmt,
            color=color,
            linewidth=lw,
            markersize=ms,
            markerfacecolor=mfc,
            markeredgewidth=mew,
            markeredgecolor=color if mfc != "white" else color,
            label=lab,
        )
        allv.extend(vals)

    ax.set_title(panel["title"], fontsize=13, fontweight="bold")
    ax.set_xlabel("Context Length", fontsize=12)
    if show_ylabel:
        ax.set_ylabel("Avg. QPS / GPU  (log scale)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(panel["xlabels"], fontsize=11)

    # Log y-axis: large-dynamic-range QPS (0.02~2.6) spreads cleanly so all four
    # curves stay separated at every context length.
    ax.set_yscale("log")
    lo, hi = min(v for v in allv if v > 0), max(allv)
    ax.set_ylim(lo * 0.65, hi * 1.6)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)

    if show_legend:
        ax.legend(fontsize=9.5, frameon=False, loc="upper right")


def main():
    panels = [QWEN3, LLAMA, QWEN35, load_dsv4()]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.0))
    for i, (ax, p) in enumerate(zip(axes, panels)):
        plot_panel(ax, p, show_ylabel=(i == 0), show_legend=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
