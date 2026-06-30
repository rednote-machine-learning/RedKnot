#!/usr/bin/env python3
"""Plot iso-throughput memory: LRU vs store-everything.

Two panels, single-column paper layout, muted professional palette
(grays + a single red accent).
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Muted professional palette: grays + one red accent for the real dataset.
STYLE = {
    "musique_ans": dict(
        color="#c0392b", marker="o", lw=1.4, ms=3.6, z=5, label="MuSiQue (real)"
    ),
    "synth_zipf1.5": dict(
        color="#404040", marker="s", lw=1.1, ms=3.2, z=4, label="Zipf 1.5 (high skew)"
    ),
    "synth_zipf1.1": dict(
        color="#7f7f7f", marker="^", lw=1.1, ms=3.2, z=3, label="Zipf 1.1 (med skew)"
    ),
    "synth_zipf0.8": dict(
        color="#b0b0b0", marker="D", lw=1.1, ms=2.8, z=2, label="Zipf 0.8 (low skew)"
    ),
}
ORDER = ["musique_ans", "synth_zipf1.5", "synth_zipf1.1", "synth_zipf0.8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", default=str(HERE / "figures/mem_iso_throughput.json")
    )
    ap.add_argument("--out", default=str(HERE / "figures/mem_iso_throughput.png"))
    args = ap.parse_args()
    data = json.load(open(args.inp))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#333333",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    names = [n for n in ORDER if n in data]

    # Single-column width ~3.4 in; two panels side by side.
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.78))
    fig.subplots_adjust(wspace=0.55, left=0.13, right=0.985, bottom=0.24, top=0.84)

    # (1) KV memory needed vs target throughput (dotted = store-all)
    for n in names:
        st = STYLE[n]
        rows = [r for r in data[n]["iso_throughput"] if r["lru_gb"]]
        x = [int(r["target_frac"] * 100) for r in rows]
        y = [r["lru_gb"] for r in rows]
        ax[0].plot(
            x,
            y,
            linestyle="-",
            marker=st["marker"],
            color=st["color"],
            lw=st["lw"],
            ms=st["ms"],
            markeredgewidth=0.5,
            zorder=st["z"],
        )
        ax[0].axhline(
            data[n]["store_all_gb"], color=st["color"], ls=":", lw=0.7, alpha=0.55
        )
    ax[0].set_yscale("log")
    ax[0].set_xlabel("(a) target throughput (%)", fontsize=6.5, labelpad=2)
    ax[0].set_ylabel("KV memory (GB)", fontsize=6.5, labelpad=2)
    ax[0].tick_params(labelsize=6)
    ax[0].grid(True, axis="y", alpha=0.25, ls="--", lw=0.35)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    # (2) memory saving factor vs target throughput
    for n in names:
        st = STYLE[n]
        rows = [r for r in data[n]["iso_throughput"] if r["lru_gb"]]
        x = [int(r["target_frac"] * 100) for r in rows]
        y = [r["saving"] for r in rows]
        ax[1].plot(
            x,
            y,
            linestyle="-",
            marker=st["marker"],
            color=st["color"],
            lw=st["lw"],
            ms=st["ms"],
            markeredgewidth=0.5,
            zorder=st["z"],
            label=st["label"],
        )
    ax[1].set_yscale("log")
    ax[1].axhline(1, color="#999999", lw=0.6, ls="-")
    ax[1].set_xlabel("(b) target throughput (%)", fontsize=6.5, labelpad=2)
    ax[1].set_ylabel("memory saving (x)", fontsize=6.5, labelpad=2)
    ax[1].tick_params(labelsize=6)
    ax[1].grid(True, axis="y", alpha=0.25, ls="--", lw=0.35)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)

    # Shared compact legend on top, in one row.
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=4.8,
        frameon=False,
        bbox_to_anchor=(0.55, 1.005),
        handlelength=1.3,
        columnspacing=0.7,
        handletextpad=0.3,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot] saved {out} and .pdf  size={fig.get_size_inches()}")


if __name__ == "__main__":
    main()
