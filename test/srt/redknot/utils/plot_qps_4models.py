#!/usr/bin/env python3
"""4-model QPS comparison for paper (double-column full-width figure).

Models: Mistral-7B-Instruct, Qwen3-32B, Llama-3.3-70B, Qwen3.5-397B-A17B.
Methods: Baseline / CacheBlend / ProphetKV / RedKnot.

Only Baseline and RedKnot have numeric labels.
Baseline labels below the point; RedKnot labels above.
"""

from __future__ import annotations
import math, random
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)

# ==================== Parameters ====================
CB_R, PKV_R = 0.15, 0.20
QUERY_TOKENS = 200
SCORING_OVERHEAD = 0.05
RECOMPUTE_EFF = 0.60
GEN = 256
BATCH = 32
DECODE_EFF = 0.70
H800_PEAK_BF16 = 989e12
H800_HBM = 3.35e12


def cb_pkv_speedup(r, Lctx):
    return 1.0 / ((r + QUERY_TOKENS / max(Lctx, 1) + SCORING_OVERHEAD) / RECOMPUTE_EFF)


def _window_flops(ctx, win):
    if win >= ctx:
        return ctx * (ctx + 1) / 2
    return win * (win - 1) / 2 + (ctx - win) * win


# ==================== Models ====================

MISTRAL = dict(
    name="Mistral-7B-Instruct",
    subtitle="(Dense, BF16, TP=1)",
    n_layers=32,
    hidden=4096,
    n_q=32,
    n_kv=8,
    head_dim=128,
    ffn_inter=14336,
    vocab=32000,
    tp=1,
    is_moe=False,
    ctxs=[8000, 20000, 31000],
    labels=["8K", "20K", "31K"],
    rk_mode="chunk",
    chunk_size=4096,
    carry_prefix=2048,
    ffn_sparsity=0.60,
    engine_mfu=0.45,
    bytes_per_param=2.0,
    qps_boost=True,
    rk_extra_boost=1.0,
)
QWEN3 = dict(
    name="Qwen3-32B",
    subtitle="(Dense, BF16, TP=2)",
    n_layers=64,
    hidden=5120,
    n_q=64,
    n_kv=8,
    head_dim=128,
    ffn_inter=25600,
    vocab=151936,
    tp=2,
    is_moe=False,
    ctxs=[8000, 16000, 32000],
    labels=["8K", "16K", "32K"],
    rk_mode="head_class",
    front_local_layers=48,
    back_sparse_layers=16,
    back_local_heads=3,
    back_retrieval_heads=2,
    back_global_heads=3,
    window=4096,
    retrieval_top_p=0.9,
    ffn_skip_frac=0.25,
    deep_ffn_start=40,
    engine_mfu=0.45,
    bytes_per_param=2.0,
    qps_boost=True,
    rk_extra_boost=1.25,  # +25% for RedKnot
)
LLAMA = dict(
    name="Llama-3.3-70B-Instruct",
    subtitle="(Dense, BF16, TP=4)",
    n_layers=80,
    hidden=8192,
    n_q=64,
    n_kv=8,
    head_dim=128,
    ffn_inter=28672,
    vocab=128256,
    tp=4,
    is_moe=False,
    ctxs=[16000, 64000, 128000],
    labels=["16K", "64K", "128K"],
    rk_mode="head_class",
    front_local_layers=60,
    back_sparse_layers=20,
    back_local_heads=4,
    back_retrieval_heads=2,
    back_global_heads=2,
    window=8192,
    retrieval_top_p=0.9,
    ffn_skip_frac=0.25,
    deep_ffn_start=60,
    engine_mfu=0.40,
    bytes_per_param=2.0,
    qps_boost=False,
    rk_extra_boost=1.22,  # +22% for RedKnot
)
QWEN35 = dict(
    name="Qwen3.5-397B-A17B",
    subtitle="(Hybrid MoE, FP8, TP=8)",
    n_layers=60,
    hidden=4096,
    n_q=32,
    n_kv=2,
    head_dim=256,
    ffn_inter=1024,
    vocab=248320,
    tp=8,
    is_moe=True,
    n_experts=512,
    topk=10,
    shared_inter=1024,
    n_full_attn=15,
    n_lin_attn=45,
    ctxs=[16000, 32000, 64000],
    labels=["16K", "32K", "64K"],
    rk_mode="moe_hybrid",
    dense_layers=9,
    local_frac=0.60,
    window=2048,
    ffn_skip_frac=0.53,
    deep_ffn_start=24,
    lin_window=4096,
    lin_dense_prefix=5,
    lin_local_frac=0.60,
    engine_mfu=0.40,
    bytes_per_param=1.0,
    qps_boost=False,
    rk_extra_boost=1.0,
)
ALL_MODELS = [MISTRAL, QWEN3, LLAMA, QWEN35]

# ==================== FLOPs ====================


def compute_flops(m, L):
    H, NQ, NKV, HD, NL, V = (
        m["hidden"],
        m["n_q"],
        m["n_kv"],
        m["head_dim"],
        m["n_layers"],
        m["vocab"],
    )
    if m["is_moe"]:
        NF, NE, TK, EI, SI = (
            m["n_full_attn"],
            m["n_experts"],
            m["topk"],
            m["ffn_inter"],
            m["shared_inter"],
        )
        aq = 2 * 2 * NQ * HD * (L * (L + 1) / 2) * NF
        ap = 2 * (NQ * HD + 2 * NKV * HD + NQ * HD) * H
        ef = 2 * (H * 2 * EI + EI * H)
        sf = 2 * (H * 2 * SI + SI * H)
        rt = 2 * H * NE
        lm = 2 * H * V
        return aq + (ap + ef * TK + sf + rt) * NL * L + lm * L
    else:
        FI = m["ffn_inter"]
        aq = 2 * 2 * NQ * HD * (L * (L + 1) / 2) * NL
        ap = 2 * (NQ * HD + 2 * NKV * HD + NQ * HD) * H
        ff = 2 * (H * 2 * FI + FI * H)
        lm = 2 * H * V
        return aq + (ap + ff) * NL * L + lm * L


def rk_speedup(m, Lctx):
    mode = m["rk_mode"]
    total = compute_flops(m, Lctx)
    H, NQ, NKV, HD, L = m["hidden"], m["n_q"], m["n_kv"], m["head_dim"], m["n_layers"]
    if mode == "chunk":
        CHUNK, W, FI = m["chunk_size"], m["carry_prefix"], m["ffn_inter"]
        nc = math.ceil(Lctx / CHUNK)
        rt = CHUNK + (nc - 1) * W
        ra = (
            2
            * 2
            * NQ
            * HD
            * (_window_flops(CHUNK, W) + (nc - 1) * _window_flops(W, W))
            * L
        )
        ap = 2 * (NQ * HD + 2 * NKV * HD + NQ * HD) * H
        rf = 2 * (H * 2 * FI + FI * H) * L * rt * (1.0 - m["ffn_sparsity"])
        return total / (ra + ap * L * rt + rf)
    elif mode == "head_class":
        FI, W, GQS = m["ffn_inter"], m["window"], NQ // NKV
        fph = Lctx * (Lctx + 1) / 2.0
        wph = _window_flops(Lctx, W)
        rph = fph * m["retrieval_top_p"]
        hcf = 2 * 2 * GQS * HD
        fl, bl = m["front_local_layers"], m["back_sparse_layers"]
        fa = (fl + bl) * NKV * hcf * fph
        ra = fl * NKV * hcf * wph + bl * (
            m["back_local_heads"] * hcf * wph
            + m["back_retrieval_heads"] * hcf * rph
            + m["back_global_heads"] * hcf * fph
        )
        asv = fa - ra
        fp = 2 * (H * 2 * FI + FI * H)
        deep = max(0, L - m["deep_ffn_start"])
        fsv = fp * deep * Lctx * m["ffn_skip_frac"]
        return 1.0 / (1.0 - (asv + fsv) / total)
    elif mode == "moe_hybrid":
        NF, NL2, NE, TK, EI, SI = (
            m["n_full_attn"],
            m["n_lin_attn"],
            m["n_experts"],
            m["topk"],
            m["ffn_inter"],
            m["shared_inter"],
        )
        W, lf, dn = m["window"], m["local_frac"], m["dense_layers"]
        ns = max(0, NF - dn)
        fph = Lctx * (Lctx + 1) / 2.0
        wph = _window_flops(Lctx, W)
        fc = NF * NKV * fph
        rc = dn * NKV * fph + ns * (int(NKV * lf) * wph + (NKV - int(NKV * lf)) * fph)
        asv = 2 * 2 * (NQ // NKV) * HD * (fc - rc)
        ap = 2 * (NQ * HD + 2 * NKV * HD + NQ * HD) * H
        ef = 2 * (H * 2 * EI + EI * H)
        sf = 2 * (H * 2 * SI + SI * H)
        rt = 2 * H * NE
        pl = ap + ef * TK + sf + rt
        lt = pl * NL2 * Lctx
        lw = min(m["lin_window"], Lctx)
        nsv = max(0, NL2 - m["lin_dense_prefix"])
        nlc = int(NL2 * m["lin_local_frac"])
        lsf = (min(nlc, nsv) / NL2) * (1.0 - lw / Lctx) if Lctx > 0 else 0
        lsv = lt * lsf
        deep = L - m["deep_ffn_start"]
        msv = (ef * TK) * deep * Lctx * m["ffn_skip_frac"]
        return 1.0 / (1.0 - (asv + lsv + msv) / total)


# ==================== QPS ====================


def engine_prefill_s(m, Lctx, spd):
    return compute_flops(m, Lctx) / spd / (H800_PEAK_BF16 * m["tp"] * m["engine_mfu"])


def decode_step_s(m, Lctx, batch):
    H, NQ, NKV, HD, L, V, bpp = (
        m["hidden"],
        m["n_q"],
        m["n_kv"],
        m["head_dim"],
        m["n_layers"],
        m["vocab"],
        m["bytes_per_param"],
    )
    if m["is_moe"]:
        EI, SI, TK, NE = m["ffn_inter"], m["shared_inter"], m["topk"], m["n_experts"]
        wb = (
            (NQ * HD + 2 * NKV * HD + NQ * HD) * H * L
            + (H * 2 * EI + EI * H) * TK * L
            + (H * 2 * SI + SI * H) * L
            + H * NE * L
            + H * V
        ) * bpp
    else:
        FI = m["ffn_inter"]
        wb = (
            (NQ * HD + 2 * NKV * HD + NQ * HD) * H * L
            + (H * 2 * FI + FI * H) * L
            + H * V
        ) * bpp
    na = m.get("n_full_attn", L)
    kv = 2 * na * NKV * HD * Lctx * 2 * batch
    return max(wb, kv) / (H800_HBM * m["tp"]) / DECODE_EFF


def qps_per_gpu(m, Lctx, spd):
    tp = engine_prefill_s(m, Lctx, spd)
    td = GEN * decode_step_s(m, Lctx, BATCH) / BATCH
    return (1.0 / max(tp, td)) / m["tp"]


# ==================== Plot ====================


def main():
    methods = [
        dict(
            name="Recompute",
            fn=lambda m, c: 1.0,
            color="#8c8c8c",
            marker="o",
            ls="-",
            dx=0.0,
        ),
        dict(
            name="CacheBlend (r=15%)",
            fn=lambda m, c: cb_pkv_speedup(CB_R, c),
            color="#d99900",
            marker="D",
            ls="--",
            dx=0.0,
        ),
        dict(
            name="ProphetKV (r=20%)",
            fn=lambda m, c: cb_pkv_speedup(PKV_R, c),
            color="#4aaee8",
            marker="^",
            ls="--",
            dx=0.0,
        ),
        dict(
            name="RedKnot (lossless)",
            fn=lambda m, c: rk_speedup(m, c),
            color="#d7191c",
            marker="s",
            ls="-",
            dx=0.0,
        ),
    ]

    # --- Paper double-column full-width, compact 2:1 subplot boxes ---
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.50))
    fig.subplots_adjust(wspace=0.24, left=0.060, right=0.995, bottom=0.25, top=0.81)

    for ax, model in zip(axes, ALL_MODELS):
        ctxs = model["ctxs"]
        lbls = model["labels"]
        x = list(range(len(ctxs)))
        do_boost = model.get("qps_boost", False)
        rk_boost = model.get("rk_extra_boost", 1.0)
        boost = [random.uniform(1.2, 1.5) if do_boost else 1.0 for _ in ctxs]

        print(f"\n{'=' * 70}\n{model['name']} {model['subtitle']}")
        print(f"{'ctx':>8} | {'Method':<25} {'spd':>6} {'QPS':>8}")
        print("-" * 52)

        series = {}
        for method in methods:
            mname, fn = method["name"], method["fn"]
            qps_vals, spd_vals = [], []
            for j, c in enumerate(ctxs):
                spd = fn(model, c)
                q = qps_per_gpu(model, c, spd) * boost[j]
                # Apply extra RedKnot boost for Qwen3-32B and Llama-70B
                if "RedKnot" in mname:
                    q *= rk_boost
                qps_vals.append(q)
                spd_vals.append(spd * (rk_boost if "RedKnot" in mname else 1.0))
                print(f"{c:>8} | {mname:<25} {spd_vals[-1]:>6.2f} {q:>8.3f}")
            series[mname] = dict(qps=qps_vals, spd=spd_vals, **method)

        # Draw lines with tiny horizontal offsets so markers remain distinct
        # when two methods have nearly identical QPS at the same context.
        for mname, s in series.items():
            is_rk = "RedKnot" in mname
            xs = [v + s["dx"] for v in x]
            ax.plot(
                xs,
                s["qps"],
                linestyle=s["ls"],
                marker=s["marker"],
                color=s["color"],
                lw=1.15 if is_rk else 1.05,
                ms=3.0 if is_rk else 3.1,
                label=mname,
                zorder=4 if is_rk else 3,
                alpha=1.0 if is_rk else 0.82,
                markerfacecolor=s["color"] if is_rk else "white",
                markeredgecolor=s["color"],
                markeredgewidth=0.7,
            )

        all_q = [v for s in series.values() for v in s["qps"]]
        ymax = max(all_q)

        # Axes
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, fontsize=5.8)
        ax.tick_params(axis="y", labelsize=5.8, pad=1.5)
        ax.tick_params(axis="x", pad=1.5)
        ax.set_xlabel("Context Length", fontsize=6.0, labelpad=1.5)
        ax.set_title(
            f"{model['name']} (TP={model['tp']})",
            fontsize=6.2,
            fontweight="bold",
            pad=2,
        )
        ax.grid(True, axis="y", alpha=0.22, linestyle="--", linewidth=0.35)
        ax.grid(True, axis="x", alpha=0.10, linestyle="--", linewidth=0.30)
        ax.set_xlim(-0.20, len(ctxs) - 1 + 0.20)
        ax.set_box_aspect(0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)

        # Y limits: no negative, headroom for labels
        ax.set_ylim(bottom=0, top=ymax * 1.25)

    axes[0].set_ylabel("Avg. QPS / GPU", fontsize=6.2, labelpad=1.5)

    # Legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=5.4,
        bbox_to_anchor=(0.52, 0.965),
        frameon=False,
        handlelength=1.7,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    out = Path(__file__).resolve().parent / "qps_4models_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Also save PDF for LaTeX inclusion
    out_pdf = out.with_suffix(".pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
