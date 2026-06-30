#!/usr/bin/env python3
"""Generate academic figures (bar / line) for the RedKnot eval from result JSONs.

Outputs PDF (vector, for papers) + PNG (preview) into test/srt/redknot/figures/.

Figures:
  fig1_indexer_topk_f1        : line  — F1 vs indexer top-k (prefix compression)
  fig2_segpaged_bandwidth     : grouped bars — Paged vs SegPaged (prefill+decode)
  fig3_hetero_window          : grouped bars + line — uniform vs heterogeneous win
  fig4_offload_latency        : line (log) — decode latency vs link (HBM/PCIe/SSD)
  fig5_extreme_local_f1       : grouped bars — dense vs extreme vs safe (3 datasets)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ---- academic style ----
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "figure.dpi": 120,
    }
)
C = {
    "paged": "#d1495b",
    "segp": "#2e86ab",
    "dense": "#8d99ae",
    "a": "#2e86ab",
    "b": "#e07a5f",
    "c": "#3d405b",
    "d": "#81b29a",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def J(p):
    return json.load(open(p))


# ── Fig 1: indexer top-k F1 (line) ──
def fig1():
    d = J("/tmp/redknot_indexer_topk_sweep.json")
    ks = [512, 256, 128, 64]
    ds = ["hotpotqa", "2wikimqa", "musique"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, col in zip(ds, [C["a"], C["b"], C["d"]]):
        ys = [d[str(k)]["f1_per_ds"].get(name) for k in ks]
        ax.plot([str(k) for k in ks], ys, "-o", color=col, label=name, lw=2, ms=6)
    avg = [d[str(k)]["f1_avg"] for k in ks]
    ax.plot(
        [str(k) for k in ks], avg, "-s", color=C["c"], label="average", lw=2.5, ms=7
    )
    ax.set_xlabel("indexer top-k (prefix compression; 512 = native)")
    ax.set_ylabel("F1")
    ax.set_title("Prefix compression via indexer top-k (DeepSeek-V4)")
    ax.invert_xaxis()
    ax.legend(frameon=False)
    save(fig, "fig1_indexer_topk_f1")


# ── Fig 2: SegPaged vs Paged bandwidth, prefill + decode (grouped bars) ──
def fig2():
    d = J("/tmp/redknot_bandwidth_paged_vs_segpaged.json")
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for ax, phase, title in [
        (axes[0], "decode", "Decode (per step)"),
        (axes[1], "prefill", "Prefill (whole prompt)"),
    ]:
        rows = d[phase]
        labels = [r["len"] for r in rows]
        paged = [r["paged_GB"] for r in rows]
        segp = [r["segpaged_GB"] for r in rows]
        x = np.arange(len(labels))
        w = 0.38
        ax.bar(x - w / 2, paged, w, label="Paged", color=C["paged"])
        ax.bar(x + w / 2, segp, w, label="SegPaged", color=C["segp"])
        for i, r in enumerate(rows):
            ax.text(
                x[i],
                max(paged[i], segp[i]),
                f"-{r['saved_pct']:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=C["segp"],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("prefix length")
        ax.set_ylabel("KV-read (GB)")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.suptitle(
        "SegPaged vs Paged KV-read bandwidth (Qwen3-32B, uniform window)",
        fontsize=12,
        y=1.02,
    )
    save(fig, "fig2_segpaged_bandwidth")


# ── Fig 3: uniform vs heterogeneous window (grouped bars + ratio line) ──
def fig3():
    d = J("/tmp/redknot_hetero_window_bandwidth.json")
    import numpy as np

    rows = d["results"]
    labels = [f"{r['Lctx'] // 1000}K" for r in rows]
    uni_saved = [
        100 * (1 - r["uniform"]["segpaged_GB"] / r["uniform"]["paged_GB"]) for r in rows
    ]
    het_saved = [
        100 * (1 - r["hetero"]["segpaged_GB"] / r["hetero"]["paged_GB"]) for r in rows
    ]
    uni_ratio = [r["uniform"]["ratio"] for r in rows]
    het_ratio = [r["hetero"]["ratio"] for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    # serious amber + dark blue palette
    COL_UNI = "#E1A100"  # serious gold/amber
    COL_HET = "#1F4E79"  # serious dark blue
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    b1 = ax.bar(
        x - w / 2,
        uni_saved,
        w,
        label="uniform window",
        color=COL_UNI,
        edgecolor="black",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x + w / 2,
        het_saved,
        w,
        label="heterogeneous window",
        color=COL_HET,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("prefix length")
    ax.set_ylabel("bandwidth saved by SegPaged (%)")
    ax.set_title(
        "Per-head heterogeneous window exposes SegPaged's advantage\n(Qwen3-32B)"
    )
    ax.set_ylim(0, 58)
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{b.get_height():.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    # legend side-by-side (horizontal), centered on top
    ax.legend(
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=2.0,
        handletextpad=0.6,
    )
    save(fig, "fig3_hetero_window")


# ── Fig 4: offload decode latency vs link bandwidth (log line) ──
def fig4():
    # Use the REALISTIC heterogeneous-window IO (Paged read-amplification is much
    # larger than under the uniform-window setting -> the latency gap is more
    # representative of real deployments).
    d = J("/tmp/redknot_hetero_window_bandwidth.json")
    import numpy as np

    rows = sorted(d["results"], key=lambda r: r["Lctx"])
    lengths = [r["Lctx"] for r in rows]
    paged_GB = {r["Lctx"]: r["hetero"]["paged_GB"] for r in rows}
    segp_GB = {r["Lctx"]: r["hetero"]["segpaged_GB"] for r in rows}
    n_gen = 256
    links = [("HBM", 3000e9), ("PCIe4", 25e9), ("NVMe SSD", 5e9)]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    markers = {"HBM": "o", "PCIe4": "s", "NVMe SSD": "^"}
    for name, bw in links:
        lat_p = [paged_GB[L] * 1e9 * n_gen / bw for L in lengths]
        lat_s = [segp_GB[L] * 1e9 * n_gen / bw for L in lengths]
        ax.plot(
            [L // 1000 for L in lengths],
            lat_p,
            markers[name] + "--",
            color=C["paged"],
            lw=1.8,
            ms=6,
            alpha=0.95,
            label=f"{name} · Paged",
        )
        ax.plot(
            [L // 1000 for L in lengths],
            lat_s,
            markers[name] + "-",
            color=C["segp"],
            lw=2.2,
            ms=6,
            label=f"{name} · SegPaged",
        )
    ax.set_yscale("log")
    ax.set_xlabel("prefix length (K tokens)")
    ax.set_ylabel(f"decode latency for {n_gen} steps (s, log)")
    ax.set_title(
        "Offloaded KV: SegPaged cuts read latency on slow links\n"
        "(Qwen3-32B, heterogeneous per-head window)"
    )
    ax.legend(frameon=False, ncol=1, fontsize=8.5)
    save(fig, "fig4_offload_latency")


# ── Fig 5: extreme-local vs dense vs safe F1 (grouped bars) ──
def fig5():
    d = J("/tmp/redknot_3way.json")["results"]
    import numpy as np

    ds = ["hotpotqa", "2wikimqa", "musique"]
    modes = [
        ("dense", "Dense (full prefix)", C["dense"]),
        ("extreme", "Extreme local (store-128)", C["paged"]),
        ("safe", "Safe (stride-8)", C["segp"]),
    ]
    x = np.arange(len(ds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (m, lab, col) in enumerate(modes):
        ys = [d[m]["datasets"].get(x_, {}).get("f1", 0) for x_ in ds]
        ax.bar(x + (i - 1) * w, ys, w, label=lab, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.set_ylabel("F1")
    ax.set_title(
        "Storing only last-128 prefix collapses multi-hop QA\n(DeepSeek-V4, 4K prefix)"
    )
    ax.legend(frameon=False)
    save(fig, "fig5_extreme_local_f1")


if __name__ == "__main__":
    print(f"Generating figures into {OUT} ...")
    for fn in [fig1, fig2, fig3, fig4, fig5]:
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {e}")
    print("Done.")
