#!/usr/bin/env python3
"""Plot RedKnot latency-breakdown results.

Input is the JSON emitted by latency_multi_ctx_397b.py via REDKNOT_LATENCY_OUT.
It draws:
  * prefill latency vs context
  * decode latency vs context
  * total latency vs context
  * RedKnot prefill speedup over recompute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    with open(path) as f:
        raw = json.load(f)
    rows = []
    for ctx_s, v in raw.items():
        ctx = int(ctx_s)
        base = v.get("base", {})
        rk = v.get("rk", {})
        if not base.get("t_prefill"):
            continue
        row = {
            "ctx": ctx,
            "base_prefill_ms": 1000.0 * float(base.get("t_prefill", 0.0)),
            "base_decode_ms": 1000.0 * float(base.get("t_dec", 0.0)),
            "base_total_ms": 1000.0
            * float(base.get("t_prefill", 0.0) + base.get("t_dec", 0.0)),
            "rk_ok": bool(rk and rk.get("t_prefill")),
        }
        if rk and rk.get("t_prefill"):
            row.update(
                {
                    "rk_prefill_ms": 1000.0 * float(rk.get("t_prefill", 0.0)),
                    "rk_decode_ms": 1000.0 * float(rk.get("t_dec", 0.0)),
                    "rk_total_ms": 1000.0
                    * float(rk.get("t_prefill", 0.0) + rk.get("t_dec", 0.0)),
                    "prefill_speedup": float(base.get("t_prefill", 0.0))
                    / max(float(rk.get("t_prefill", 0.0)), 1e-9),
                }
            )
        rows.append(row)
    rows.sort(key=lambda r: r["ctx"])
    return rows


def _print_table(rows):
    print("\nLatency breakdown summary")
    print("=" * 92)
    print(
        f"{'ctx':>8} | {'base pf ms':>10} {'base dec ms':>11} {'base total':>11} | "
        f"{'rk pf ms':>9} {'rk dec ms':>10} {'rk total':>9} | {'pf spd':>6}"
    )
    print("-" * 92)
    for r in rows:
        if r["rk_ok"]:
            print(
                f"{r['ctx']:>8} | {r['base_prefill_ms']:>10.1f} {r['base_decode_ms']:>11.1f} "
                f"{r['base_total_ms']:>11.1f} | {r['rk_prefill_ms']:>9.1f} "
                f"{r['rk_decode_ms']:>10.1f} {r['rk_total_ms']:>9.1f} | "
                f"{r['prefill_speedup']:>5.2f}x"
            )
        else:
            print(
                f"{r['ctx']:>8} | {r['base_prefill_ms']:>10.1f} {r['base_decode_ms']:>11.1f} "
                f"{r['base_total_ms']:>11.1f} | {'RedKnot failed':>31} | {'-':>6}"
            )


def _plot(rows, out: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx = [r["ctx"] // 1000 for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.6))
    metrics = [
        ("prefill_ms", "Prefill latency (ms)", "Prefill"),
        ("decode_ms", "Decode latency (ms)", "Decode"),
        ("total_ms", "Total latency (ms)", "End-to-end"),
    ]
    for ax, (suffix, ylabel, subtitle) in zip(axes[:3], metrics):
        ax.plot(
            ctx,
            [r[f"base_{suffix}"] for r in rows],
            "o-",
            color="#8c8c8c",
            label="Recompute",
        )
        rk_x = [r["ctx"] // 1000 for r in rows if r["rk_ok"]]
        rk_y = [r[f"rk_{suffix}"] for r in rows if r["rk_ok"]]
        if rk_y:
            ax.plot(rk_x, rk_y, "s-", color="#d7191c", label="RedKnot")
        ax.set_title(subtitle)
        ax.set_xlabel("Context length (K)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)

    spd_rows = [r for r in rows if r["rk_ok"]]
    axes[3].plot(
        [r["ctx"] // 1000 for r in spd_rows],
        [r["prefill_speedup"] for r in spd_rows],
        "s-",
        color="#d7191c",
    )
    axes[3].axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[3].set_title("Prefill speedup")
    axes[3].set_xlabel("Context length (K)")
    axes[3].set_ylabel("Speedup vs recompute")
    axes[3].grid(True, alpha=0.25, linestyle="--")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[plot] saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="latency JSON file")
    ap.add_argument("--out", default=None, help="output PNG, default <in>.png")
    ap.add_argument("--title", default="RedKnot latency breakdown")
    args = ap.parse_args()

    inp = Path(args.inp)
    rows = _load(inp)
    if not rows:
        raise SystemExit(f"no valid rows found in {inp}")
    _print_table(rows)
    out = Path(args.out) if args.out else inp.with_suffix(".png")
    _plot(rows, out, args.title)


if __name__ == "__main__":
    main()
