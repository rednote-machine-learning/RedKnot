#!/usr/bin/env python3
"""plot_qps_sweep.py — plot concurrency-vs-QPS curves from the sweep JSONL.

Reads the JSONL produced by benchmark_RedKnot_QPS_sweep.py (one row per
(backend, concurrency) run, as appended by sglang.bench_serving) and draws:

  * QPS (request_throughput)        vs concurrency
  * Output throughput (tok/s)       vs concurrency
  * P99 end-to-end latency (ms)     vs concurrency

One line per backend (tag), so you get RedKnot vs baseline overlaid.

Usage:
  python test/srt/redknot/plot_qps_sweep.py \
      --in test/srt/redknot/figures/qps_sweep_qwen35.jsonl \
      --out test/srt/redknot/figures/qps_sweep_qwen35.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # bench_serving may also append a detail line; skip non-metric rows
                continue
    return rows


def series_by_backend(rows):
    """tag -> sorted list of dicts with concurrency + metrics."""
    by = defaultdict(list)
    for r in rows:
        # Only keep rows that carry the throughput metric (the metric line).
        if "request_throughput" not in r or r.get("max_concurrency") is None:
            continue
        tag = r.get("tag") or r.get("backend") or "unknown"
        if tag == "baseline":
            tag = "Recompute"
        elif tag == "redknot":
            tag = "RedKnot"
        by[tag].append(
            {
                "concurrency": int(r["max_concurrency"]),
                "qps": float(r["request_throughput"]),
                "out_tok_s": float(r.get("output_throughput") or 0.0),
                "p99_ms": float(r.get("p99_e2e_latency_ms") or 0.0),
                "median_ms": float(r.get("median_e2e_latency_ms") or 0.0),
            }
        )
    for tag in by:
        by[tag].sort(key=lambda d: d["concurrency"])
    return by


def print_table(by):
    print("\n" + "=" * 78)
    print("Concurrency sweep summary")
    print("=" * 78)
    for tag, pts in by.items():
        print(f"\n[{tag}]")
        print(
            f"  {'conc':>5} {'QPS':>9} {'out tok/s':>11} {'p99 ms':>10} {'median ms':>11}"
        )
        for p in pts:
            print(
                f"  {p['concurrency']:>5} {p['qps']:>9.3f} {p['out_tok_s']:>11.1f} "
                f"{p['p99_ms']:>10.1f} {p['median_ms']:>11.1f}"
            )


def make_plot(by, out_path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("qps", "QPS (req/s)", "Throughput: QPS vs concurrency", False),
        (
            "out_tok_s",
            "Output throughput (tok/s)",
            "Output tok/s vs concurrency",
            False,
        ),
        ("p99_ms", "P99 e2e latency (ms)", "P99 latency vs concurrency", True),
    ]
    markers = ["o", "s", "^", "D", "v", "*"]
    for ax, (key, ylabel, subtitle, log_y) in zip(axes, metrics):
        for i, (tag, pts) in enumerate(by.items()):
            xs = [p["concurrency"] for p in pts]
            ys = [p[key] for p in pts]
            ax.plot(xs, ys, marker=markers[i % len(markers)], label=tag, linewidth=2)
        ax.set_xlabel("concurrency (in-flight requests)")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\n[plot] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="sweep JSONL.")
    ap.add_argument("--out", default=None, help="output PNG (default: <in>.png).")
    ap.add_argument("--title", default="RedKnot vs baseline — concurrency throughput")
    args = ap.parse_args()

    inp = Path(args.inp)
    rows = load_rows(inp)
    by = series_by_backend(rows)
    if not by:
        raise SystemExit(f"no metric rows found in {inp}")

    print_table(by)

    out = Path(args.out) if args.out else inp.with_suffix(".png")
    try:
        make_plot(by, out, args.title)
    except ImportError:
        print("[plot] matplotlib not available; printed table only.")


if __name__ == "__main__":
    main()
