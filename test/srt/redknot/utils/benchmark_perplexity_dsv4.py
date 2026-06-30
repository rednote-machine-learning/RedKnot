#!/usr/bin/env python3
"""Perplexity-based evaluation of RedKnot sparse strategies on DeepSeek-V4.

Uses SGLang native /generate API with logprob_start_len=0 to get prompt token
log-probabilities.  Computes perplexity as exp(-mean(log_probs)).

Works perfectly with Base models (no QA needed).

Modes:
  dense        – Full prefill, dsv4 backend, no sparsity
  moe_sparse   – dsv4 + sparse-FFN (indexer-driven MoE sparsity)
  prefix_reuse – dsv4 + prefix caching to simulate offline KV reuse

Usage::

    # With server already running (--no-server):
    python benchmark_perplexity_dsv4.py --no-server --port 31995 --mode dense

    # Auto-launch server:
    python benchmark_perplexity_dsv4.py --mode dense --port 31995 --n-samples 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[3]
MODEL_PATH = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4-Flash-Base",
)
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
TP_SIZE = 1
PP_SIZE = 8

# Datasets with long context (narrative/document, no short-answer QA bias)
TEXT_DATASETS = ["narrativeqa", "gov_report", "multi_news", "qasper", "hotpotqa"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_texts(
    ds_name: str, n_samples: int, target_tokens: int = 4096, seed: int = 2026
) -> list[str]:
    """Load pure text contexts from LongBench, truncated to ~target_tokens."""
    path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ctx = row.get("context", "")
            if len(ctx) >= target_tokens * 2:  # at least 2 chars/token
                rows.append(ctx)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < n_samples:
        print(
            f"  WARNING: {ds_name} only has {len(rows)} texts >= {target_tokens * 2} chars"
        )

    # Truncate to ~target_tokens (4 chars/token estimate)
    target_chars = target_tokens * 4
    texts = []
    for ctx in rows[:n_samples]:
        texts.append(ctx[:target_chars])
    return texts


# ---------------------------------------------------------------------------
# Server lifecycle (reuse from benchmark_redknot_dsv4_longbench.py)
# ---------------------------------------------------------------------------
def _server_env(rank_log_dir: str):
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(REPO / "python"),
        SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="1",
        SGLANG_BARE_SUBPROCESS_LAUNCH="1",
        SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="1",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        SGLANG_OPT_USE_TILELANG_MHC_PRE="0",
        SGLANG_OPT_USE_TILELANG_MHC_POST="0",
        SGLANG_OPT_DEEPGEMM_HC_PRENORM="0",
        SGLANG_OPT_FP8_WO_A_GEMM="0",
        SGLANG_JIT_DEEPGEMM_PRECOMPILE="0",
        SGLANG_JIT_DEEPGEMM_FAST_WARMUP="1",
        SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS="16",
        SGLANG_DSV4_FP4_EXPERTS="0",
        SGLANG_REDKNOT_FFN_DEBUG="1",
        SGLANG_RANK_LOG_DIR=rank_log_dir,
    )
    return env


def start_server(
    mode: str,
    port: int,
    rank_log_dir: str,
    server_log: str,
    mass_thresh: float = 0.6,
    enable_prefix_cache: bool = False,
):
    """Start SGLang server.

    mode: 'dense', 'moe_sparse', 'prefix_reuse'
    """
    os.makedirs(rank_log_dir, exist_ok=True)
    env = _server_env(rank_log_dir)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--attention-backend",
        "dsv4",  # always dsv4 for perplexity (no head sparsity)
        "--tp-size",
        str(TP_SIZE),
        "--pp-size",
        str(PP_SIZE),
        "--mem-fraction-static",
        "0.85",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        "--chunked-prefill-size",
        "16384",
        "--max-prefill-tokens",
        "32768",
        "--max-total-tokens",
        "65536",
        "--trust-remote-code",
        "--port",
        str(port),
    ]
    if not enable_prefix_cache:
        cmd.append("--disable-radix-cache")

    if mode == "moe_sparse":
        cmd += [
            "--redknot-sparse-ffn-enable",
            "--redknot-sparse-ffn-importance",
            "indexer",
            "--redknot-sparse-ffn-dense-until",
            "4",
            "--redknot-sparse-ffn-mass-thresh",
            str(mass_thresh),
            "--redknot-sparse-ffn-recent-n",
            "256",
        ]

    lf = open(server_log, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(REPO),
    )
    return proc, lf


def wait_ready(port: int, server_log: str, timeout: int = 600) -> bool:
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        if os.path.exists(server_log):
            txt = Path(server_log).read_text(errors="ignore")
            if "fired up and ready" in txt:
                time.sleep(3)
                return True
            if "EXIT=" in txt or "SIGQUIT received" in txt:
                return False
        time.sleep(3)
    return False


def stop_server(proc, lf):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        lf.close()
    except Exception:
        pass
    subprocess.run("pkill -9 -f sglang::scheduler", shell=True)
    subprocess.run("pkill -9 -f sglang.launch_server", shell=True)
    time.sleep(8)


# ---------------------------------------------------------------------------
# Perplexity computation via /generate API
# ---------------------------------------------------------------------------
def get_prompt_logprobs(port: int, text: str, timeout: int = 900) -> dict:
    """Send text to SGLang /generate and get prompt token log-probabilities.

    SGLang native API returns meta_info["input_token_logprobs"] as a list of
    tuples: [(logprob, token_id, token_text_or_None), ...].
    The first element has logprob=None (no conditioning context for token 0).

    Returns dict with:
        token_logprobs: list of float (log-prob of each token given prefix)
        n_tokens: int
        perplexity: float
    """
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": 1,  # Need at least 1 to get response
            "temperature": 0,
        },
        "return_logprob": True,
        "logprob_start_len": 0,  # Get logprobs from position 0 (prompt tokens)
        "top_logprobs_num": 0,  # Don't need top-k, just the actual token logprob
        "return_text_in_logprobs": False,
    }
    r = requests.post(
        url, json=payload, timeout=timeout, headers={"Connection": "close"}
    )
    r.raise_for_status()
    obj = r.json()

    meta = obj.get("meta_info", {})
    prompt_tokens = meta.get("prompt_tokens", 0)

    # input_token_logprobs is a list of (logprob, token_id, text_or_None)
    # First element has logprob=None
    raw_logprobs = meta.get("input_token_logprobs", [])

    # Extract valid log-probabilities (skip None entries)
    valid_logprobs = []
    for entry in raw_logprobs:
        if entry is None:
            continue
        # entry is (logprob, token_id, text_or_None) or just a float
        if isinstance(entry, (list, tuple)):
            lp = entry[0]
        else:
            lp = entry
        if lp is not None and not math.isinf(lp):
            valid_logprobs.append(float(lp))

    n_tokens = len(valid_logprobs)
    if n_tokens == 0:
        return {
            "token_logprobs": [],
            "n_tokens": 0,
            "perplexity": float("inf"),
            "avg_logprob": float("-inf"),
            "prompt_tokens": prompt_tokens,
            "e2e_latency": meta.get("e2e_latency", 0),
        }

    avg_logprob = sum(valid_logprobs) / n_tokens
    # perplexity = exp(-avg_logprob)
    # Since logprobs are negative, -avg_logprob is positive
    perplexity = math.exp(-avg_logprob)

    return {
        "token_logprobs": valid_logprobs,
        "n_tokens": n_tokens,
        "perplexity": perplexity,
        "avg_logprob": avg_logprob,
        "prompt_tokens": prompt_tokens,
        "e2e_latency": meta.get("e2e_latency", 0),
    }
    r = requests.post(
        url, json=payload, timeout=timeout, headers={"Connection": "close"}
    )
    r.raise_for_status()
    obj = r.json()

    meta = obj.get("meta_info", {})
    input_token_logprobs = meta.get("input_token_logprobs_val", [])
    prompt_tokens = meta.get("prompt_tokens", 0)

    if not input_token_logprobs:
        # Fallback: try alternative field names
        input_token_logprobs = meta.get("input_token_logprobs", [])

    # Filter out None / first token (has no conditioning)
    valid_logprobs = []
    for lp in input_token_logprobs:
        if lp is not None and not math.isinf(lp):
            valid_logprobs.append(lp)

    n_tokens = len(valid_logprobs)
    if n_tokens == 0:
        return {
            "token_logprobs": [],
            "n_tokens": 0,
            "perplexity": float("inf"),
            "avg_logprob": float("-inf"),
            "prompt_tokens": prompt_tokens,
            "e2e_latency": meta.get("e2e_latency", 0),
        }

    avg_logprob = sum(valid_logprobs) / n_tokens
    # perplexity = exp(-avg_logprob)
    # Since logprobs are negative, -avg_logprob is positive
    perplexity = math.exp(-avg_logprob)

    return {
        "token_logprobs": valid_logprobs,
        "n_tokens": n_tokens,
        "perplexity": perplexity,
        "avg_logprob": avg_logprob,
        "prompt_tokens": prompt_tokens,
        "e2e_latency": meta.get("e2e_latency", 0),
    }


def get_prefix_reuse_logprobs(
    port: int,
    text: str,
    block_size_tokens: int = 1024,
    timeout: int = 900,
) -> dict:
    """Simulate offline KV reuse via prefix caching.

    1. Split text into blocks of ~block_size_tokens
    2. Send block_1 first (populates prefix cache)
    3. Send block_1 + block_2 (prefix cache hit for block_1)
    4. ... continue until full text
    5. Final request: full text with logprob_start_len=0

    This simulates the scenario where:
    - Each block was "offline prefilled" (cached)
    - The final request reuses all cached KV

    Returns same format as get_prompt_logprobs.
    """
    # Estimate chars per block
    chars_per_token = 4
    block_size_chars = block_size_tokens * chars_per_token

    # Split into blocks
    blocks = []
    remaining = text
    while len(remaining) > block_size_chars // 2:
        blocks.append(remaining[:block_size_chars])
        remaining = remaining[block_size_chars:]
    if remaining:
        blocks[-1] += remaining  # append remainder to last block

    # Warm up prefix cache by sending incremental prefixes
    url = f"http://127.0.0.1:{port}/generate"
    for i in range(len(blocks) - 1):
        prefix = "".join(blocks[: i + 1])
        payload = {
            "text": prefix,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }
        try:
            r = requests.post(
                url, json=payload, timeout=timeout, headers={"Connection": "close"}
            )
            r.raise_for_status()
        except Exception as e:
            print(f"    Prefix warmup {i + 1}/{len(blocks)} failed: {e}")

    # Now send full text with logprobs
    return get_prompt_logprobs(port, text, timeout=timeout)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate_perplexity(
    port: int,
    mode: str,
    datasets: list[str],
    n_samples: int,
    target_tokens: int,
    block_size_tokens: int = 1024,
) -> dict:
    """Run perplexity evaluation across datasets.

    Returns aggregated results.
    """
    all_results = {}
    all_ppls = []
    all_avg_lps = []

    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name} | Mode: {mode} | Target: {target_tokens} tokens")
        print(f"{'=' * 60}")

        texts = load_texts(ds_name, n_samples, target_tokens=target_tokens)
        if not texts:
            print(f"  No texts available, skipping")
            continue

        ds_ppls = []
        ds_avg_lps = []
        ds_n_tokens = []
        ds_latencies = []
        ds_prompt_tokens = []

        for i, text in enumerate(texts):
            try:
                if mode == "prefix_reuse":
                    result = get_prefix_reuse_logprobs(
                        port, text, block_size_tokens=block_size_tokens
                    )
                else:
                    result = get_prompt_logprobs(port, text)

                if result["n_tokens"] == 0:
                    print(f"  [{i + 1}/{len(texts)}] No logprobs returned, skipping")
                    continue

                ds_ppls.append(result["perplexity"])
                ds_avg_lps.append(result["avg_logprob"])
                ds_n_tokens.append(result["n_tokens"])
                ds_latencies.append(result["e2e_latency"])
                ds_prompt_tokens.append(result["prompt_tokens"])

                print(
                    f"  [{i + 1}/{len(texts)}] "
                    f"PPL={result['perplexity']:.4f}  "
                    f"avg_lp={result['avg_logprob']:.4f}  "
                    f"tokens={result['n_tokens']}  "
                    f"latency={result['e2e_latency']:.1f}s"
                )
            except Exception as e:
                print(f"  [{i + 1}/{len(texts)}] ERROR: {e}")
                continue

        if not ds_ppls:
            print(f"  No successful evaluations for {ds_name}")
            continue

        ds_result = {
            "dataset": ds_name,
            "n_samples": len(ds_ppls),
            "avg_perplexity": float(np.mean(ds_ppls)),
            "std_perplexity": float(np.std(ds_ppls)),
            "median_perplexity": float(np.median(ds_ppls)),
            "avg_logprob": float(np.mean(ds_avg_lps)),
            "avg_n_tokens": float(np.mean(ds_n_tokens)),
            "avg_prompt_tokens": float(np.mean(ds_prompt_tokens)),
            "avg_latency": float(np.mean(ds_latencies)),
            "perplexities": ds_ppls,
        }
        all_results[ds_name] = ds_result
        all_ppls.extend(ds_ppls)
        all_avg_lps.extend(ds_avg_lps)

        print(f"\n  {ds_name} summary:")
        print(
            f"    Avg PPL:  {ds_result['avg_perplexity']:.4f} ± {ds_result['std_perplexity']:.4f}"
        )
        print(f"    Avg LogP: {ds_result['avg_logprob']:.4f}")
        print(f"    Tokens:   {ds_result['avg_n_tokens']:.0f}")

    # Overall summary
    overall = {}
    if all_ppls:
        overall = {
            "n_datasets": len(all_results),
            "n_total_samples": len(all_ppls),
            "avg_perplexity": float(np.mean(all_ppls)),
            "std_perplexity": float(np.std(all_ppls)),
            "median_perplexity": float(np.median(all_ppls)),
            "avg_logprob": float(np.mean(all_avg_lps)),
        }
        print(f"\n{'=' * 60}")
        print(f"OVERALL ({mode}):")
        print(
            f"  Avg PPL:    {overall['avg_perplexity']:.4f} ± {overall['std_perplexity']:.4f}"
        )
        print(f"  Median PPL: {overall['median_perplexity']:.4f}")
        print(f"  Avg LogP:   {overall['avg_logprob']:.4f}")
        print(f"{'=' * 60}")

    return {
        "mode": mode,
        "target_tokens": target_tokens,
        "per_dataset": all_results,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation for RedKnot DeepSeek-V4"
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "moe_sparse", "prefix_reuse", "all"],
        default="dense",
        help="Evaluation mode",
    )
    parser.add_argument("--port", type=int, default=31995)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=4096,
        help="Target context length in tokens",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Block size in tokens for prefix_reuse mode",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Override dataset list",
    )
    parser.add_argument(
        "--mass-thresh",
        type=float,
        default=0.6,
        help="Mass threshold for MoE sparse mode",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Don't launch server (assume already running)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    datasets = args.datasets or TEXT_DATASETS

    modes = (
        [args.mode] if args.mode != "all" else ["dense", "moe_sparse", "prefix_reuse"]
    )

    all_mode_results = {}

    for mode in modes:
        print(f"\n{'#' * 70}")
        print(f"# MODE: {mode}")
        print(f"{'#' * 70}")

        proc = lf = None
        if not args.no_server:
            rank_log_dir = f"/tmp/ppl_rank_logs_{mode}"
            server_log = f"/tmp/ppl_server_{mode}.log"

            # Stop any existing server
            subprocess.run("pkill -9 -f sglang.launch_server", shell=True)
            subprocess.run("pkill -9 -f 'sglang::scheduler'", shell=True)
            time.sleep(5)

            enable_prefix_cache = mode == "prefix_reuse"
            proc, lf = start_server(
                mode=mode if mode != "prefix_reuse" else "dense",
                port=args.port,
                rank_log_dir=rank_log_dir,
                server_log=server_log,
                mass_thresh=args.mass_thresh,
                enable_prefix_cache=enable_prefix_cache,
            )
            print(f"Waiting for server (mode={mode})...")
            if not wait_ready(args.port, server_log, timeout=600):
                print("ERROR: Server failed to start!")
                if proc:
                    stop_server(proc, lf)
                continue
            print("Server ready!")

        try:
            result = evaluate_perplexity(
                port=args.port,
                mode=mode,
                datasets=datasets,
                n_samples=args.n_samples,
                target_tokens=args.target_tokens,
                block_size_tokens=args.block_size,
            )
            all_mode_results[mode] = result
        finally:
            if proc:
                stop_server(proc, lf)

    # Save results
    output_path = (
        args.output or f"/tmp/redknot_ppl_{args.mode}_{args.target_tokens}.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_mode_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Print comparison table if multiple modes
    if len(all_mode_results) > 1:
        print(f"\n{'=' * 70}")
        print("COMPARISON TABLE")
        print(f"{'=' * 70}")
        print(f"{'Mode':<20} {'Avg PPL':>12} {'Median PPL':>12} {'Avg LogP':>12}")
        print(f"{'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12}")
        baseline_ppl = None
        for mode, res in all_mode_results.items():
            ov = res.get("overall", {})
            if ov:
                avg_ppl = ov["avg_perplexity"]
                if baseline_ppl is None:
                    baseline_ppl = avg_ppl
                ratio = avg_ppl / baseline_ppl if baseline_ppl else 0
                print(
                    f"{mode:<20} {avg_ppl:>12.4f} {ov['median_perplexity']:>12.4f} "
                    f"{ov['avg_logprob']:>12.4f}  (ratio: {ratio:.4f})"
                )


if __name__ == "__main__":
    main()
