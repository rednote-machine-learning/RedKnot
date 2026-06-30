#!/usr/bin/env python3
"""Plot chunk reuse-lifecycle economics for RedKnot offline-KV caching."""

from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_stats(dataset_path, kv_bytes):
    spec = importlib.util.spec_from_file_location(
        "cl", str(HERE / "chunk_lifecycle.py")
    )
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    stream = cl.load_musique_stream(dataset_path, None)
    return cl.analyze(stream, None, kv_bytes), len(stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default="/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/musique_ans_v1.0_dev.jsonl",
    )
    ap.add_argument(
        "--kv-bytes", type=float, default=49536, help="KV bytes/token (DeepSeek V4 MLA)"
    )
    ap.add_argument("--out", default=str(HERE / "figures/chunk_lifecycle.png"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stats, n_req = load_stats(args.path, args.kv_bytes)
    rc = np.array([s["reuse_count"] for s in stats.values()])
    res = np.array([s["residency"] for s in stats.values()])
    npf = np.array([s["non_prefix_ratio"] for s in stats.values()])

    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.0))

    # (1) reuse-count distribution (log-log: long tail)
    vals, counts = np.unique(rc, return_counts=True)
    ax[0].loglog(vals, counts, "o", ms=3, color="#d7191c")
    ax[0].set_xlabel("reuse count")
    ax[0].set_ylabel("# chunks")
    ax[0].set_title("Reuse-count distribution\n(long tail)")
    ax[0].grid(True, alpha=0.3, which="both")

    # (2) non-prefix ratio histogram
    ax[1].hist(npf, bins=20, color="#2c7fb8", edgecolor="white")
    ax[1].axvline(npf.mean(), color="#d7191c", ls="--", label=f"mean={npf.mean():.2f}")
    ax[1].set_xlabel("non-prefix reuse ratio")
    ax[1].set_ylabel("# chunks")
    ax[1].set_title("Non-prefix reuse\n(prefix-cache cannot hit)")
    ax[1].legend(fontsize=8)

    # (3) lifecycle: reuse vs residency scatter (value = color)
    sc = ax[2].scatter(
        res,
        rc,
        c=np.log10(rc / np.maximum(res, 1) + 1e-3),
        s=6,
        cmap="viridis",
        alpha=0.5,
    )
    ax[2].set_xlabel("residency (request span)")
    ax[2].set_ylabel("reuse count")
    ax[2].set_yscale("log")
    ax[2].set_title("Lifecycle: reuse vs lifespan\n(bright = cache-worthy)")
    plt.colorbar(sc, ax=ax[2], label="log value density")

    # (4) policy frontier: %cached vs prefills saved, for R_min sweep
    rmins = [2, 3, 5, 10, 20]
    pct, saved, kvgb = [], [], []
    for r in rmins:
        kept = {c: s for c, s in stats.items() if s["reuse_count"] >= r}
        pct.append(100 * len(kept) / len(stats))
        saved.append(sum(s["reuse_count"] - 1 for s in kept.values()))
        kvgb.append(sum(s["kv_bytes"] for s in kept.values()) / 1e9)
    total_saved = sum(s["reuse_count"] - 1 for s in stats.values())
    ax[3].plot(kvgb, [100 * x / total_saved for x in saved], "s-", color="#d7191c")
    for i, r in enumerate(rmins):
        ax[3].annotate(
            f"R>={r}",
            (kvgb[i], 100 * saved[i] / total_saved),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax[3].set_xlabel("KV memory cached (GB)")
    ax[3].set_ylabel("% recompute saved")
    ax[3].set_title("Cache cost-benefit frontier\n(DeepSeek V4 MLA KV)")
    ax[3].grid(True, alpha=0.3)

    fig.suptitle(
        f"Non-prefix chunk reuse lifecycle (musique, {n_req} requests, "
        f"{len(stats)} chunks, mean non-prefix={npf.mean():.0%})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot] saved {out} and .pdf")


if __name__ == "__main__":
    main()
