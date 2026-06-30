#!/usr/bin/env python3
"""benchmark_RedKnot_QPS_sweep.py — REAL concurrency/QPS throughput sweep.

Unlike the old ``benchmark_RedKnot_QPS.py`` (which ran requests serially through
HF ``model(...)`` and reported ``n_requests / total_time`` — i.e. just the
inverse of mean latency, NOT real throughput), this script measures TRUE
serving throughput:

  * Launches a real SGLang HTTP server (continuous batching) with the chosen
    attention backend.
  * For each concurrency level C in --concurrency, runs the standard
    ``sglang.bench_serving`` client against the SAME fixed request pool with
    ``--max-concurrency C`` and ``--request-rate inf`` (so the number of
    in-flight requests — the real knob — is exactly C).
  * bench_serving appends one JSONL row per run with request_throughput (QPS),
    output_throughput (tok/s), p99/median latency, etc.

X-axis = concurrency, Y-axis = QPS (and output tok/s, p99 latency).

It sweeps TWO backends so you get two curves on the same plot:
  * baseline : default attention backend (e.g. fa3 / flashinfer)
  * redknot  : --attention-backend redknot (+ head config)

Each backend gets its own server (started, swept, then shut down) and writes to
``--out`` (a single JSONL), tagged by backend so the plot script can split them.

Typical run (Qwen3.5-397B-A17B, HYBRID MoE, 8x80GB, FP8):

  The 397B BF16 checkpoint is ~752GB and will NOT fit in 8x80GB=640GB, so it
  MUST be quantized (--quantization fp8 -> ~376GB). It is also a HYBRID model
  (45 linear-attention + 15 full-attention layers): SGLang auto-wraps the
  chosen --attention-backend as the FULL-attention sub-backend of
  HybridLinearAttnBackend, and RedKnot now maps global full-attn layer ids to
  the full-attn-indexed head config (use the *_server.json variant).

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=python \
    python test/srt/redknot/benchmark_RedKnot_QPS_sweep.py \
      --model-path /mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-397B-A17B \
      --tp-size 8 --quantization fp8 \
      --redknot-head-config test/srt/redknot/head_class/qwen3.5-397B-A17B_redknot_server.json \
      --context-length 40000 --mem-fraction-static 0.85 \
      --concurrency 1,2,4,8,16,32,64 \
      --num-prompts 256 --random-input-len 8000 --random-output-len 128 \
      --out test/srt/redknot/figures/qps_sweep_qwen35_397b.jsonl

Typical run (Qwen3.5-35B, 2 GPUs):

  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python \
    python test/srt/redknot/benchmark_RedKnot_QPS_sweep.py \
      --model-path /mnt/.../Qwen3.5-35B-A3B \
      --tp-size 2 \
      --redknot-head-config test/srt/redknot/head_class/qwen3.5-35B-A3B_redknot_server.json \
      --concurrency 1,2,4,8,16,32,64 \
      --num-prompts 256 \
      --random-input-len 8000 --random-output-len 128 \
      --out test/srt/redknot/figures/qps_sweep_qwen35.jsonl

Then plot:

  python test/srt/redknot/plot_qps_sweep.py \
      --in test/srt/redknot/figures/qps_sweep_qwen35_397b.jsonl \
      --out test/srt/redknot/figures/qps_sweep_qwen35_397b.png
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from sglang.test.test_utils import kill_process_tree, popen_launch_server


def parse_args():
    ap = argparse.ArgumentParser(
        description="Real concurrency-vs-QPS sweep for RedKnot vs baseline."
    )
    ap.add_argument("--model-path", required=True, help="HF model path / id.")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument(
        "--backends",
        default="baseline,redknot",
        help="comma list of backends to sweep. Use 'baseline' for the default "
        "attention backend, 'redknot' for the RedKnot backend.",
    )
    ap.add_argument(
        "--baseline-attention-backend",
        default=None,
        help="explicit attention backend for the 'baseline' curve (default: let "
        "SGLang pick its default, e.g. fa3/flashinfer).",
    )
    ap.add_argument(
        "--redknot-head-config",
        default=None,
        help="path to RedKnot head-class JSON (passed as "
        "--redknot-head-config-path). Optional but recommended.",
    )
    ap.add_argument(
        "--redknot-extra-args",
        default="",
        help="extra raw server args for the redknot backend, space-separated, "
        "e.g. '--redknot-kernel fa3 --redknot-segpaged-decode'.",
    )
    ap.add_argument(
        "--concurrency",
        default="1,2,4,8,16,32,64",
        help="comma-separated concurrency levels (the x-axis).",
    )
    # --- workload (fixed across all concurrency points so curves are comparable) ---
    ap.add_argument("--dataset-name", default="random", help="bench_serving dataset.")
    ap.add_argument(
        "--num-prompts",
        type=int,
        default=256,
        help="size of the request pool sent at EACH concurrency point.",
    )
    ap.add_argument("--random-input-len", type=int, default=8000)
    ap.add_argument("--random-output-len", type=int, default=128)
    ap.add_argument("--random-range-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    # --- server / infra ---
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=31900)
    ap.add_argument("--mem-fraction-static", type=float, default=None)
    ap.add_argument("--max-total-tokens", type=int, default=None)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument(
        "--quantization",
        default=None,
        help="server --quantization, e.g. 'fp8' (needed to fit the 397B BF16 "
        "checkpoint into 8x80GB).",
    )
    ap.add_argument("--kv-cache-dtype", default=None, help="server --kv-cache-dtype.")
    ap.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="cap on concurrent running requests in the engine.",
    )
    ap.add_argument("--chunked-prefill-size", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument(
        "--server-timeout",
        type=float,
        default=1800.0,
        help="seconds to wait for each server to become healthy.",
    )
    ap.add_argument(
        "--extra-server-args",
        default="",
        help="extra raw server args applied to ALL backends, space-separated.",
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "figures" / "qps_sweep.jsonl"),
        help="output JSONL (bench_serving appends one row per run).",
    )
    return ap.parse_args()


def _common_server_args(args) -> list[str]:
    extra = []
    if args.tp_size > 1:
        extra += ["--tp-size", str(args.tp_size)]
    if args.mem_fraction_static is not None:
        extra += ["--mem-fraction-static", str(args.mem_fraction_static)]
    if args.max_total_tokens is not None:
        extra += ["--max-total-tokens", str(args.max_total_tokens)]
    if args.context_length is not None:
        extra += ["--context-length", str(args.context_length)]
    if args.quantization is not None:
        extra += ["--quantization", str(args.quantization)]
    if args.kv_cache_dtype is not None:
        extra += ["--kv-cache-dtype", str(args.kv_cache_dtype)]
    if args.max_running_requests is not None:
        extra += ["--max-running-requests", str(args.max_running_requests)]
    if args.chunked_prefill_size is not None:
        extra += ["--chunked-prefill-size", str(args.chunked_prefill_size)]
    if args.trust_remote_code:
        extra += ["--trust-remote-code"]
    if args.extra_server_args.strip():
        extra += args.extra_server_args.split()
    return extra


def _server_args_for_backend(args, backend: str) -> list[str]:
    """Build the --attention-backend (+ config) args for one curve."""
    sa = _common_server_args(args)
    if backend == "baseline":
        if args.baseline_attention_backend:
            sa += ["--attention-backend", args.baseline_attention_backend]
        # else: let SGLang use its default backend.
    elif backend == "redknot":
        sa += ["--attention-backend", "redknot"]
        if args.redknot_head_config:
            sa += ["--redknot-head-config-path", args.redknot_head_config]
        if args.redknot_extra_args.strip():
            sa += args.redknot_extra_args.split()
    else:
        raise SystemExit(f"unknown backend: {backend!r}")
    return sa


def _run_bench(args, base_url: str, backend: str, concurrency: int) -> bool:
    """Invoke sglang.bench_serving once for one concurrency point."""
    host = base_url.split(":")[1][2:]
    port = base_url.rsplit(":", 1)[1]
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        host,
        "--port",
        port,
        "--model",
        args.model_path,
        "--dataset-name",
        args.dataset_name,
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        "inf",  # concurrency is bounded by --max-concurrency, not arrival rate
        "--max-concurrency",
        str(concurrency),
        "--seed",
        str(args.seed),
        # tag lets the plot script tell the two curves apart in one JSONL.
        "--tag",
        backend,
        "--output-file",
        args.out,
    ]
    if args.dataset_name.startswith("random"):
        cmd += [
            "--random-input-len",
            str(args.random_input_len),
            "--random-output-len",
            str(args.random_output_len),
            "--random-range-ratio",
            str(args.random_range_ratio),
        ]
    print("\n" + "-" * 84)
    print(f"[bench] backend={backend} concurrency={concurrency}")
    print("[bench] " + " ".join(cmd))
    print("-" * 84)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[bench] WARNING: bench_serving exited with code {rc}")
    return rc == 0


def sweep_one_backend(args, backend: str, concurrencies: list[int]):
    base_url = f"http://{args.host}:{args.port}"
    server_args = _server_args_for_backend(args, backend)
    print("\n" + "=" * 84)
    print(f"Launching server for backend={backend}")
    print(f"  base_url   : {base_url}")
    print(f"  extra args : {' '.join(server_args)}")
    print("=" * 84)

    proc = popen_launch_server(
        model=args.model_path,
        base_url=base_url,
        timeout=args.server_timeout,
        other_args=server_args,
    )
    try:
        # small settle delay after health passes
        time.sleep(3)
        for c in concurrencies:
            _run_bench(args, base_url, backend, c)
    finally:
        print(f"\n[server] shutting down backend={backend} ...")
        try:
            kill_process_tree(proc.pid)
        except Exception as e:
            print(f"[server] kill failed (non-fatal): {e}")
        # give the GPU a moment to free memory before the next server starts
        time.sleep(8)


def main():
    args = parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    concurrencies = [int(x) for x in args.concurrency.split(",")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Start from a clean JSONL so re-runs don't mix with stale rows.
    if out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + f".bak.{int(time.time())}")
        out_path.rename(backup)
        print(f"[out] existing results moved to {backup}")

    print("=" * 84)
    print("RedKnot REAL concurrency-vs-QPS sweep")
    print("=" * 84)
    print(f"  model        : {args.model_path}")
    print(f"  backends     : {backends}")
    print(f"  concurrency  : {concurrencies}")
    print(
        f"  workload     : {args.dataset_name} num_prompts={args.num_prompts} "
        f"in={args.random_input_len} out={args.random_output_len}"
    )
    print(f"  out (JSONL)  : {out_path}")

    for backend in backends:
        sweep_one_backend(args, backend, concurrencies)

    print("\n" + "=" * 84)
    print(f"Done. Results JSONL: {out_path}")
    print("Plot with:")
    print(
        f"  python {Path(__file__).resolve().parent / 'plot_qps_sweep.py'} "
        f"--in {out_path} --out {out_path.with_suffix('.png')}"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
