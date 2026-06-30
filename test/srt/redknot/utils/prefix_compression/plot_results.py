#!/usr/bin/env python3
"""Plot the 3 most representative figures for the prefix-compression study.

Reads the *.results.json in this folder and writes 3 PNGs:
  fig1_accuracy_vs_prefix.png   accuracy (cos d1) + KV transfer saving vs prefix
  fig2_concurrency_qps.png      single-stream vs memory-bound concurrency QPS
  fig3_cross_dataset.png        cross-dataset cos / greedy-match / dPPL
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: accuracy + KV saving vs prefix length (trim<32)
# ---------------------------------------------------------------------------
def fig1():
    b = load("test_swa_pd_bench.results.json")
    rows = [r for r in b["rows"] if r["config"] == "trim<32"]
    rows.sort(key=lambda r: r["prefix_len"])
    x = [r["prefix_len"] // 1024 for r in rows]
    cos = [r["cos_d1"] for r in rows]
    saved = [r["transfer_saved_pct"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    c1 = "#1f77b4"
    ax1.plot(x, cos, "o-", color=c1, lw=2, ms=7, label="cos(decode step-1)")
    ax1.axhline(0.99, ls="--", color="gray", lw=1, label="pass threshold 0.99")
    ax1.set_xlabel("Prefix length (K tokens)")
    ax1.set_ylabel("Logits cosine vs full-KV baseline", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(0.985, 1.001)
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    c2 = "#d62728"
    ax2.plot(x, saved, "s--", color=c2, lw=2, ms=7, label="KV transfer saved")
    ax2.set_ylabel("KV transfer saved (%)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.set_ylim(0, 60)
    for xi, s in zip(x, saved):
        ax2.annotate(
            f"{s:.0f}%",
            (xi, s),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color=c2,
            fontsize=8,
        )

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower left", fontsize=8)
    plt.title("Qwen3-32B: accuracy preserved while KV transfer shrinks (trim<32)")
    fig.tight_layout()
    out = os.path.join(HERE, "fig1_accuracy_vs_prefix.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 2: single-stream vs concurrency QPS (the real win)
# ---------------------------------------------------------------------------
def fig2():
    c = load("test_swa_pd_concurrency.results.json")
    b = load("test_swa_pd_bench.results.json")
    crows = sorted(c["rows"], key=lambda r: r["prefix_len"])
    x = [r["prefix_len"] // 1024 for r in crows]
    qps_base = [r["qps_base"] for r in crows]
    qps_trim = [r["qps_trim"] for r in crows]
    gain = [r["qps_gain"] for r in crows]

    # single-stream qps_est for trim<32 at matching prefixes
    ss = {
        r["prefix_len"] // 1024: r["qps_est"]
        for r in b["rows"]
        if r["config"] == "trim<32"
    }
    ss_base = {
        r["prefix_len"] // 1024: r["qps_est"]
        for r in b["rows"]
        if r["config"] == "baseline"
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        x,
        qps_base,
        "o-",
        color="#7f7f7f",
        lw=2,
        ms=7,
        label="concurrency QPS (baseline, full KV)",
    )
    ax.plot(
        x,
        qps_trim,
        "o-",
        color="#2ca02c",
        lw=2.5,
        ms=8,
        label="concurrency QPS (trim<32)",
    )
    # single-stream reference (much lower gain)
    xs = sorted(ss.keys())
    ax.plot(
        xs,
        [ss_base[k] for k in xs],
        "^--",
        color="#bbbbbb",
        lw=1.5,
        ms=6,
        label="single-stream QPS (baseline)",
    )
    ax.plot(
        xs,
        [ss[k] for k in xs],
        "^--",
        color="#98df8a",
        lw=1.5,
        ms=6,
        label="single-stream QPS (trim<32)",
    )

    ax.set_xlabel("Prefix length (K tokens)")
    ax.set_ylabel("QPS (single decode GPU)")
    ax.set_xticks(x)
    # annotate concurrency speedup
    for xi, yb, yt, g in zip(x, qps_base, qps_trim, gain):
        ax.annotate(
            f"{g:.2f}x",
            (xi, yt),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color="#2ca02c",
            fontsize=9,
            fontweight="bold",
        )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        "Memory-bound concurrency drives the QPS gain (1.3-1.9x), not single-stream"
    )
    fig.tight_layout()
    out = os.path.join(HERE, "fig2_concurrency_qps.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 3: cross-dataset accuracy
# ---------------------------------------------------------------------------
def fig3():
    d = load("test_swa_pd_datasets.results.json")
    s = d["summary"]
    names = [r["dataset"] for r in s]
    cos = [r["cos_d1"] for r in s]
    greedy = [r["greedy_match"] for r in s]
    dppl = [r["dppl"] for r in s]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    xpos = np.arange(len(names))
    wd = 0.38
    # left: cos + greedy match (both [0,1])
    axL.bar(xpos - wd / 2, cos, wd, color="#1f77b4", label="cos(d1)")
    axL.bar(xpos + wd / 2, greedy, wd, color="#ff7f0e", label="greedy match")
    axL.axhline(0.99, ls="--", color="gray", lw=1)
    axL.set_xticks(xpos)
    axL.set_xticklabels(names, rotation=15, fontsize=9)
    axL.set_ylim(0, 1.05)
    axL.set_ylabel("score")
    axL.set_title("Logits cosine & greedy-token agreement")
    axL.legend(fontsize=8, loc="lower right")
    for xi, v in zip(xpos, cos):
        axL.annotate(
            f"{v:.3f}",
            (xi - wd / 2, v),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7,
        )

    # right: dPPL
    bars = axR.bar(xpos, dppl, 0.5, color="#d62728")
    axR.set_xticks(xpos)
    axR.set_xticklabels(names, rotation=15, fontsize=9)
    axR.set_ylabel("ΔPPL (trim - baseline)")
    axR.set_title("Perplexity penalty (lower is better)")
    for xi, v in zip(xpos, dppl):
        axR.annotate(
            f"+{v:.2f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    axR.set_ylim(0, max(dppl) * 1.25)

    fig.suptitle(
        "Cross-dataset accuracy of trim<32 (QA / summary / code / LM)", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "fig3_cross_dataset.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("done")
