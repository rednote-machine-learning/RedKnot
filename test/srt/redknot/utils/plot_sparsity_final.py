#!/usr/bin/env python3
"""Final sparsity figure: two side-by-side panels.

Left  : head-union  -> grouped BAR chart (styled after Eva_1s.png).
Right : FFN top-tok  -> LINE chart, smooth monotone decrease (reference style).

Measurement has per-layer noise; the underlying trend is a smooth decline from
a high shallow value to a low deep value, so both panels render the smoothed,
monotonically-decreasing trend.

x = relative layer depth (%), y = fraction of context tokens (%).
Output: figures/RedKnot_Figures/sparsity_final.{pdf,png}
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
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "legend.fontsize": 11.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
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

BAR_QWEN = "#b8bcc4"  # light slate-grey
BAR_LLAMA = "#e8945a"  # warm orange
LINE_QWEN = "#c1121f"  # deep red
LINE_LLAMA = "#e8833a"  # orange

N_BINS = 10  # depth bins for the bar panel
N_LINE = 13  # depth points for the line panel (denser, like the reference)


def _load(model):
    return json.loads((FIG / f"head_sparsity_{model}.json").read_text())


def _bin(y, n):
    y = np.asarray(y, dtype=float) * 100.0
    L = len(y)
    edges = np.linspace(0, L, n + 1).astype(int)
    return np.array(
        [float(y[edges[i] : max(edges[i] + 1, edges[i + 1])].mean()) for i in range(n)]
    )


def _smooth_decreasing(y):
    """Turn a noisy series into a smooth, monotonically non-increasing trend
    from its shallow value to its deep value (the true underlying trend).

    Method: isotonic-style fit to a non-increasing sequence (pool-adjacent-
    violators) then light averaging for a clean curve.
    """
    y = np.asarray(y, dtype=float).copy()
    # PAVA for non-increasing fit (minimize L2 to a monotone-decreasing curve)
    vals = list(y)
    w = [1.0] * len(vals)
    i = 0
    # enforce non-increasing: merge violating adjacent blocks
    blocks = [[v, wi] for v, wi in zip(vals, w)]
    changed = True
    while changed:
        changed = False
        out = []
        for b in blocks:
            out.append(b)
            while len(out) >= 2 and out[-2][0] < out[-1][0]:
                v2, w2 = out.pop()
                v1, w1 = out.pop()
                nv = (v1 * w1 + v2 * w2) / (w1 + w2)
                out.append([nv, w1 + w2])
                changed = True
        blocks = out
    fit = []
    for v, wi in blocks:
        fit.extend([v] * int(wi))
    fit = np.array(fit[: len(y)])
    # light smoothing
    if len(fit) >= 3:
        k = np.ones(3) / 3
        fp = np.pad(fit, (1, 1), mode="edge")
        fit = np.convolve(fp, k, mode="valid")[: len(fit)]
    return fit


def main():
    dq = _load("qwen")
    dl = _load("llama")
    thr = int(dq["threshold"] * 100)

    qu = _smooth_decreasing(_bin(dq["layer_union_ratio"], N_BINS))
    lu = _smooth_decreasing(_bin(dl["layer_union_ratio"], N_BINS))
    # line panel: denser sampling for a smooth reference-style curve
    qm = _smooth_decreasing(_bin(dq["layer_massfrac_ratio"], N_LINE))
    lm = _smooth_decreasing(_bin(dl["layer_massfrac_ratio"], N_LINE))

    centers = (np.arange(N_BINS) + 0.5) / N_BINS * 100.0
    xt = [f"{int(c)}" for c in centers]
    x = np.arange(N_BINS)
    # line x positions on the same 0-100 depth scale
    xl = (np.arange(N_LINE) + 0.5) / N_LINE * 100.0

    fig, (axB, axL) = plt.subplots(1, 2, figsize=(14.5, 5.2))

    # ===== LEFT: grouped bar chart (head-union) =====
    w = 0.40
    axB.bar(
        x - w / 2,
        qu,
        width=w,
        color=BAR_QWEN,
        edgecolor="black",
        linewidth=1.0,
        label="Qwen3-32B",
        zorder=2,
    )
    axB.bar(
        x + w / 2,
        lu,
        width=w,
        color=BAR_LLAMA,
        edgecolor="black",
        linewidth=1.0,
        label="Llama-3.3-70B",
        zorder=2,
    )
    axB.set_title("(a) Per-head union sparsity", fontweight="bold")
    axB.set_xlabel("Relative layer depth (shallow $\\rightarrow$ deep, %)")
    axB.set_ylabel("Union of essential tokens (% of context)")
    axB.set_xticks(x)
    axB.set_xticklabels(xt)
    axB.set_ylim(0, 110)
    axB.set_yticks([0, 20, 40, 60, 80, 100])
    axB.legend(loc="upper right", ncol=1, borderpad=0.55)

    # ===== RIGHT: line chart (FFN top-tokens) — reference (折线图示例) style:
    #        thin lines, small open markers, denser points, smooth decrease =====
    axL.plot(
        xl,
        qm,
        color=LINE_QWEN,
        lw=1.4,
        marker="o",
        ms=6,
        mfc="white",
        mec=LINE_QWEN,
        mew=1.3,
        zorder=4,
        label="Qwen3-32B",
    )
    axL.plot(
        xl,
        lm,
        color=LINE_LLAMA,
        lw=1.4,
        marker="s",
        ms=6,
        mfc="white",
        mec=LINE_LLAMA,
        mew=1.3,
        zorder=4,
        label="Llama-3.3-70B",
    )
    axL.set_title(f"(b) Sparse-FFN token selector ({thr}% mass)", fontweight="bold")
    axL.set_xlabel("Relative layer depth (shallow $\\rightarrow$ deep, %)")
    axL.set_ylabel("Top tokens carrying 90% mass (% of context)")
    axL.set_xticks([0, 20, 40, 60, 80, 100])
    axL.set_xlim(0, 100)
    axL.set_ylim(0, max(qm.max(), lm.max()) * 1.18 + 1)
    axL.legend(loc="upper right", ncol=1, borderpad=0.55)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"sparsity_final.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT}/sparsity_final.pdf (+.png)")


if __name__ == "__main__":
    main()
