#!/usr/bin/env python3
"""Simulate offline KV reuse for DeepSeek-V4 MLA and measure accuracy impact.

This script validates the core hypothesis: since V4's MLA uses SWA (window=128)
for all heads, blocks that are >128 tokens apart have NO cross-block attention
dependency through SWA. The only cross-block information flows through the
compressed extra cache (c4/c128). If we can show that independent-block prefill
produces similar compressed KV to joint prefill, then offline KV reuse is safe.

Approach:
  1. Dense baseline: send full context + query as one request.
  2. Prefix-cached (cumulative): send block1, then block1+block2, ...,
     relying on SGLang's radix prefix cache to reuse prior blocks' KV.
     This simulates the *best case* of offline KV reuse (cumulative, not independent).
  3. Block-shuffled: send query with blocks in DIFFERENT ORDER to verify that
     V4's compressed cache is position-dependent (blocks can't be freely reordered).
  4. Single-block: send only the RELEVANT block + query, to measure how much
     the compressed extra cache from other blocks contributes to accuracy.

If approach 2 matches approach 1, prefix caching works for V4 (expected).
The gap between 4 and 1 shows how much long-range compressed cache matters.

Usage::

    # Start dsv4 server WITH prefix caching enabled (no --disable-radix-cache):
    python simulate_offline_reuse.py --port 31995 --n-samples 20

    # Or start the server yourself and use --no-server:
    python simulate_offline_reuse.py --port 31995 --no-server --n-samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import string
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

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

DATASETS = ["hotpotqa", "2wikimqa", "musique", "narrativeqa", "triviaqa"]


# ---------------------------------------------------------------------------
# Accuracy helpers (same as benchmark script)
# ---------------------------------------------------------------------------
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def _clean_answer(text: str) -> str:
    text = re.split(r"(?i)(?:\n\s*question\s*:|\n\s*q\s*:|<\|)", text)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(ds_name: str, n_samples: int, min_context_tokens: int = 0, seed=2026):
    path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("context") and "answers" in row and row["answers"]:
                ctx_tok_est = len(row["context"]) // 4
                if ctx_tok_est >= min_context_tokens:
                    rows.append(row)
    import random

    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < n_samples:
        print(
            f"  WARNING: {ds_name} only has {len(rows)} rows >= {min_context_tokens} tokens"
        )
    return rows[:n_samples]


def build_blocks(row: dict, n_blocks: int = 4, max_tokens_per_block: int = 2000):
    """Split context into blocks and return (blocks_list, question, gold)."""
    context = row["context"]
    question = row.get("input", "").strip()
    target_chars = max_tokens_per_block * 4
    blocks = []
    remaining = context
    for i in range(n_blocks):
        if i == n_blocks - 1:
            blocks.append(remaining[:target_chars])
        else:
            blocks.append(remaining[:target_chars])
            remaining = remaining[target_chars:]
    golds = row["answers"]
    if isinstance(golds, list):
        gold = str(golds[0]) if golds else ""
    else:
        gold = str(golds)
    return blocks, question, gold


def blocks_to_prompt(blocks: list, question: str) -> str:
    """Concatenate blocks and question into a prompt."""
    return (
        "\n\n".join(f"Block {i + 1}:\n{b}" for i, b in enumerate(blocks))
        + "\n\n"
        + f"Question: {question}\nAnswer:"
    )


# ---------------------------------------------------------------------------
# Server lifecycle
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
    port: int, rank_log_dir: str, server_log: str, enable_sff: bool = True
):
    """Start dsv4 server WITH prefix caching enabled (radix cache ON)."""
    os.makedirs(rank_log_dir, exist_ok=True)
    env = _server_env(rank_log_dir)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--attention-backend",
        "dsv4",  # Always dsv4 (no head sparsity)
        "--tp-size",
        str(TP_SIZE),
        "--pp-size",
        str(PP_SIZE),
        "--mem-fraction-static",
        "0.85",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        # NOTE: we do NOT disable radix cache — prefix caching is ON
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
    if enable_sff:
        cmd += [
            "--redknot-sparse-ffn-enable",
            "--redknot-sparse-ffn-importance",
            "indexer",
            "--redknot-sparse-ffn-dense-until",
            "4",
            "--redknot-sparse-ffn-mass-thresh",
            "0.6",
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
# Inference
# ---------------------------------------------------------------------------
def generate(port: int, prompt: str, max_new: int) -> tuple:
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=900, headers={"Connection": "close"})
    r.raise_for_status()
    obj = r.json()
    return obj.get("text", ""), float(obj["meta_info"].get("e2e_latency", 0.0))


def flush_cache(port: int):
    """Flush the radix prefix cache to ensure clean state."""
    try:
        requests.post(
            f"http://127.0.0.1:{port}/flush_cache",
            json={},
            timeout=30,
        )
    except Exception:
        pass
    time.sleep(1)


# ---------------------------------------------------------------------------
# Test modes
# ---------------------------------------------------------------------------
def test_dense(port: int, blocks: list, question: str, max_new: int):
    """Mode 1: Full context, single request (dense baseline)."""
    prompt = blocks_to_prompt(blocks, question)
    text, latency = generate(port, prompt, max_new)
    return _clean_answer(text), latency


def test_prefix_cached(port: int, blocks: list, question: str, max_new: int):
    """Mode 2: Cumulative prefix caching.

    Send block1+query first (populates prefix cache for block1),
    then block1+block2+query (reuses block1 KV, computes block2),
    etc. The final full request is the actual test.

    This simulates CUMULATIVE offline KV reuse (best case).
    """
    # Warm up prefix cache incrementally
    for i in range(1, len(blocks)):
        partial_blocks = blocks[:i]
        partial_prompt = blocks_to_prompt(partial_blocks, question)
        try:
            generate(port, partial_prompt, 1)  # just 1 token to populate cache
        except Exception:
            pass

    # Now send the full request — should hit prefix cache for all prior blocks
    full_prompt = blocks_to_prompt(blocks, question)
    text, latency = generate(port, full_prompt, max_new)
    return _clean_answer(text), latency


def test_reordered(port: int, blocks: list, question: str, max_new: int):
    """Mode 3: Send blocks in REVERSED order + query.

    If V4's compressed cache is position-sensitive (it has RoPE), this should
    produce different (likely worse) results than the original order.
    This validates that block order matters.
    """
    rev_blocks = list(reversed(blocks))
    prompt = blocks_to_prompt(rev_blocks, question)
    text, latency = generate(port, prompt, max_new)
    return _clean_answer(text), latency


def test_single_block_best(
    port: int, blocks: list, question: str, gold: str, max_new: int
):
    """Mode 4: Send EACH block individually + query, pick the best answer.

    This measures how much accuracy is retained when only one block is available
    (no cross-block information). The gap vs dense shows the contribution of
    the compressed extra cache from other blocks.
    """
    best_f1 = -1.0
    best_ans = ""
    for i, block in enumerate(blocks):
        prompt = f"Context:\n{block}\n\nQuestion: {question}\nAnswer:"
        try:
            text, _ = generate(port, prompt, max_new)
            ans = _clean_answer(text)
            f1 = f1_score(ans, gold)
            if f1 > best_f1:
                best_f1 = f1
                best_ans = ans
        except Exception:
            pass
    return best_ans, 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--port", type=int, default=31995)
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--max-tokens-per-block", type=int, default=1000)
    ap.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names (default: all 5).",
    )
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument(
        "--modes",
        type=str,
        default="dense,prefix_cached",
        help="Comma-separated modes to test: dense, prefix_cached, reordered, single_block",
    )
    ap.add_argument("--no-sff", action="store_true", help="Disable sparse-FFN")
    ap.add_argument("--out", type=str, default="/tmp/redknot_offline_reuse_sim.json")
    args = ap.parse_args()

    datasets = DATASETS
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]

    modes = [m.strip() for m in args.modes.split(",")]
    seq_len = args.n_blocks * args.max_tokens_per_block

    print(f"Offline KV reuse simulation")
    print(f"  seq_len={seq_len} ({args.n_blocks} x {args.max_tokens_per_block})")
    print(f"  modes={modes}")
    print(f"  datasets={datasets}")
    print(f"  n_samples={args.n_samples}")
    print()

    # Load data
    min_ctx = int(seq_len * 0.9)
    all_data = {}
    for ds_name in datasets:
        rows = load_dataset(ds_name, args.n_samples, min_context_tokens=min_ctx)
        if not rows:
            print(f"  SKIP {ds_name}: no rows with >= {min_ctx} tokens")
            continue
        data = []
        for row in rows:
            blocks, question, gold = build_blocks(
                row, args.n_blocks, args.max_tokens_per_block
            )
            data.append({"blocks": blocks, "question": question, "gold": gold})
        all_data[ds_name] = data
        print(f"  Loaded {ds_name}: {len(data)} samples")

    # Server management
    proc, lf = None, None
    rank_log_dir = "/tmp/offline_reuse_sim_ranklogs"
    server_log = "/tmp/offline_reuse_sim_server.log"

    if not args.no_server:
        subprocess.run(f"rm -rf {rank_log_dir} {server_log}", shell=True)
        print("\nStarting dsv4 server (prefix caching ON)...")
        proc, lf = start_server(
            args.port, rank_log_dir, server_log, enable_sff=not args.no_sff
        )
        if not wait_ready(args.port, server_log, timeout=args.load_timeout):
            print("Server failed to start!")
            if proc:
                stop_server(proc, lf)
            return
        print("Server ready.\n")
    else:
        try:
            requests.get(
                f"http://127.0.0.1:{args.port}/health", timeout=5
            ).raise_for_status()
        except Exception:
            print(f"Server not reachable on port {args.port}")
            return

    # Warmup
    try:
        generate(args.port, "Hello, world! Answer:", 3)
    except Exception as e:
        print(f"Warmup error: {e}")

    mode_fns = {
        "dense": test_dense,
        "prefix_cached": test_prefix_cached,
        "reordered": test_reordered,
        "single_block": test_single_block_best,
    }

    results = {mode: {} for mode in modes}

    try:
        for ds_name, data in all_data.items():
            print(f"\n{'=' * 60}")
            print(f"  Dataset: {ds_name} ({len(data)} samples)")
            print(f"{'=' * 60}")

            for mode in modes:
                print(f"\n  [{mode}]")
                ds_rows = []
                for i, sample in enumerate(data):
                    # Flush cache between MODES (not between samples within same mode)
                    # to get clean prefix cache behavior
                    if i == 0:
                        flush_cache(args.port)

                    try:
                        if mode == "single_block":
                            ans, lat = test_single_block_best(
                                args.port,
                                sample["blocks"],
                                sample["question"],
                                sample["gold"],
                                args.max_new,
                            )
                        else:
                            fn = mode_fns[mode]
                            ans, lat = fn(
                                args.port,
                                sample["blocks"],
                                sample["question"],
                                args.max_new,
                            )
                    except Exception as e:
                        print(f"    #{i + 1} error: {e}")
                        continue

                    f1 = f1_score(ans, sample["gold"])
                    em = em_score(ans, sample["gold"])
                    ds_rows.append(
                        {
                            "f1": f1,
                            "em": em,
                            "ans": ans[:60],
                            "gold": sample["gold"][:40],
                        }
                    )
                    print(
                        f"    #{i + 1:2d} F1={f1:.2f} EM={em:.0f}  "
                        f"ans={ans[:40]!r}  gold={sample['gold'][:30]!r}",
                        flush=True,
                    )

                if ds_rows:
                    n = len(ds_rows)
                    results[mode][ds_name] = {
                        "f1": sum(r["f1"] for r in ds_rows) / n,
                        "em": sum(r["em"] for r in ds_rows) / n,
                        "n": n,
                    }

        # Report
        print("\n" + "=" * 72)
        print("OFFLINE KV REUSE SIMULATION RESULTS")
        print(f"seq_len={seq_len}  datasets={datasets}  n_samples={args.n_samples}")
        print("=" * 72)

        for ds_name in datasets:
            if ds_name not in all_data:
                continue
            print(f"\n  --- {ds_name} ---")
            for mode in modes:
                if ds_name in results[mode]:
                    d = results[mode][ds_name]
                    print(
                        f"    {mode:<20s} F1={d['f1']:.3f}  EM={d['em']:.3f}  n={d['n']}"
                    )
            # Compute delta vs dense
            if "dense" in results and ds_name in results["dense"]:
                baseline_f1 = results["dense"][ds_name]["f1"]
                for mode in modes:
                    if mode != "dense" and ds_name in results[mode]:
                        df1 = results[mode][ds_name]["f1"] - baseline_f1
                        print(f"    {'Δ ' + mode:<20s} F1={df1:+.3f}")

        # Average across datasets
        print(f"\n  --- Average across datasets ---")
        for mode in modes:
            f1s = [results[mode][ds]["f1"] for ds in all_data if ds in results[mode]]
            ems = [results[mode][ds]["em"] for ds in all_data if ds in results[mode]]
            if f1s:
                print(
                    f"    {mode:<20s} F1={sum(f1s) / len(f1s):.3f}  "
                    f"EM={sum(ems) / len(ems):.3f}  ({len(f1s)} datasets)"
                )

        if "dense" in results:
            dense_avg = sum(
                results["dense"][ds]["f1"] for ds in all_data if ds in results["dense"]
            ) / max(len([ds for ds in all_data if ds in results["dense"]]), 1)
            for mode in modes:
                if mode == "dense":
                    continue
                mode_f1s = [
                    results[mode][ds]["f1"] for ds in all_data if ds in results[mode]
                ]
                if mode_f1s:
                    mode_avg = sum(mode_f1s) / len(mode_f1s)
                    retention = mode_avg / dense_avg * 100 if dense_avg > 0 else 0
                    print(f"    {mode:<20s} retention: {retention:.1f}% of dense F1")

        # Save
        output = {
            "config": {
                "seq_len": seq_len,
                "n_blocks": args.n_blocks,
                "max_tokens_per_block": args.max_tokens_per_block,
                "n_samples": args.n_samples,
                "modes": modes,
                "datasets": datasets,
                "sparse_ffn": not args.no_sff,
            },
            "results": results,
        }
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {args.out}")

    finally:
        if proc is not None:
            stop_server(proc, lf)


if __name__ == "__main__":
    main()
