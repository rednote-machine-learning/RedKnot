#!/usr/bin/env python3
"""RedKnot DeepSeek-V4 dense vs sparse (head+indexer-MoE) comparison on LongBench.

For each dataset, context is split into 4 blocks, concatenated::

    block_1 + block_2 + block_3 + block_4 + Question + Answer_prefix

Baseline (dense): full recompute of the concatenated text (dsv4, no sparse).
RedKnot sparse:  redknot_mla (head sparse) + sparse-FFN (indexer-driven MoE).

Each uses PP=8 to avoid the TP broadcast stall; the same server config is
used for both runs except the sparse flags.

Metrics: F1, EM, keep_ratio → MoE FLOPs fraction (hardware-independent).
TTFT/prefill throughput are omitted: L20Y fp8 kernels are ~100x slow.

Usage::

    python benchmark_redknot_dsv4_longbench.py --n-samples 10 --port 31995
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

# Datasets to evaluate (QA with short answers, suitable for F1/EM).
DATASETS = ["hotpotqa", "2wikimqa", "musique", "narrativeqa", "triviaqa"]


# ---------------------------------------------------------------------------
# Accuracy
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
# Data
# ---------------------------------------------------------------------------
def load_dataset(
    ds_name: str, n_samples: int, min_context_tokens: int = 0, seed: int = 2026
):
    """Load samples from a LongBench dataset.

    Args:
        min_context_tokens: Only keep rows whose context has at least this many
            tokens (estimated at 4 chars/token).  Useful for filtering samples
            that are too short for the requested total sequence length.
    """
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
            f"  WARNING: {ds_name} only has {len(rows)} rows >= {min_context_tokens} tokens (requested {n_samples})"
        )
    return rows[:n_samples]


def build_prompt(row: dict, n_blocks: int = 4, max_tokens_per_block: int = 2000):
    """Split context into ``n_blocks`` roughly-equal blocks, each ~ max_tokens_per_block tokens."""
    context = row["context"]
    question = row.get("input", "").strip()
    # rough tokenisation: ~4 chars/token
    target_chars = max_tokens_per_block * 4
    blocks = []
    remaining = context
    for i in range(n_blocks):
        if i == n_blocks - 1:
            blocks.append(remaining[:target_chars])
        else:
            blocks.append(remaining[:target_chars])
            remaining = remaining[target_chars:]
    prompt = (
        "\n\n".join(f"Block {i + 1}:\n{b}" for i, b in enumerate(blocks))
        + "\n\n"
        + f"Question: {question}\nAnswer:"
    )
    golds = row["answers"]
    if isinstance(golds, list):
        gold = str(golds[0]) if golds else ""
    else:
        gold = str(golds)
    return prompt, gold


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


def start_server(mode: str, port: int, rank_log_dir: str, server_log: str):
    """mode: 'dense' or 'sparse'"""
    os.makedirs(rank_log_dir, exist_ok=True)
    env = _server_env(rank_log_dir)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--attention-backend",
        "redknot_mla" if mode == "sparse" else "dsv4",
        "--tp-size",
        str(TP_SIZE),
        "--pp-size",
        str(PP_SIZE),
        "--mem-fraction-static",
        "0.85",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        "--disable-radix-cache",
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
    if mode == "sparse":
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
# Requests
# ---------------------------------------------------------------------------
def generate(port: int, prompt: str, max_new: int):
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=900, headers={"Connection": "close"})
    r.raise_for_status()
    obj = r.json()
    return obj.get("text", ""), float(obj["meta_info"].get("e2e_latency", 0.0))


def parse_keep_ratio(rank_log_dir: str) -> float | None:
    f = os.path.join(rank_log_dir, "rank0.log")
    if not os.path.exists(f):
        return None
    last = None
    for line in Path(f).read_text(errors="ignore").splitlines():
        m = re.search(r"keep_ratio=([0-9.]+)", line)
        if m:
            last = float(m.group(1))
    return last


def moe_flops_fraction(keep_ratio: float, n_layers: int, dense_until: int) -> float:
    dense_layers = min(dense_until, n_layers)
    sparse_layers = n_layers - dense_layers
    total_dense = n_layers
    total_sparse = dense_layers + sparse_layers * keep_ratio
    return total_sparse / total_dense


# ---------------------------------------------------------------------------
# Compute + TTFT estimation (grounded in measured sparsity)
# ---------------------------------------------------------------------------
# DeepSeek-V4-Flash-Base dims
_HID = 4096
_HEAD_DIM = 512
_N_HEADS = 64
_MOE_INTER = 2048
_N_ACT_EXPERTS = 6  # experts per token
_N_SHARED = 1
_KV_LORA = 512  # latent kv rank (approx for MLA proj)


def parse_head_policy(server_log: str):
    """Parse 'RedKnot MLA policy loaded: {...}' to get local/global/dense head counts."""
    if not os.path.exists(server_log):
        return None
    for line in Path(server_log).read_text(errors="ignore").splitlines():
        m = re.search(r"RedKnot MLA policy loaded: (\{[^}]+\})", line)
        if m:
            try:
                import ast as _ast

                return _ast.literal_eval(m.group(1))
            except Exception:
                return None
    return None


def estimate_ttft_speedup(
    seq_len: int,
    keep_ratio: float,
    head_policy: dict | None,
    n_layers: int,
    dense_until: int,
    local_window: int = 128,
):
    """Estimate prefill TTFT speedup from MoE + attention FLOPs savings.

    Prefill FLOPs per layer ~= attention + FFN(MoE). We estimate the dense
    baseline cost and the RedKnot cost, then TTFT_speedup = dense / sparse
    (compute-bound approximation; the L20Y kernels are compute-bound here).
    """
    T = seq_len
    # Per-token-pair attention cost scales O(T) per query (each query attends ~T
    # keys at full; ~window for local heads). Total attention ~ sum over queries.
    # Dense attention FLOPs (proportional): heads * T * T * head_dim
    attn_dense = _N_HEADS * T * T * _HEAD_DIM
    # FFN/MoE FLOPs per token: (active_experts + shared) * 2 * hid * moe_inter * 2
    ffn_per_tok = (_N_ACT_EXPERTS + _N_SHARED) * 2 * _HID * _MOE_INTER * 2
    ffn_dense = T * ffn_per_tok * n_layers

    # --- RedKnot sparse ---
    # MoE: dense_until layers full, rest keep_ratio fraction of tokens.
    dense_layers = min(dense_until, n_layers)
    sparse_layers = n_layers - dense_layers
    ffn_sparse = (
        dense_layers * T * ffn_per_tok + sparse_layers * (keep_ratio * T) * ffn_per_tok
    )

    # Attention: local heads attend ~local_window keys instead of T.
    if head_policy:
        total_heads = head_policy.get("total", _N_HEADS * n_layers)
        n_local = head_policy.get("local", 0)
        n_global = head_policy.get("global", 0) + head_policy.get("dense", 0)
        # average per-(layer,head): local heads cost ~ T*window, global ~ T*T
        attn_sparse = (
            n_global * T * T * _HEAD_DIM
            + n_local * T * min(local_window, T) * _HEAD_DIM
        )
        # normalize: dense is all heads doing T*T over n_layers
        attn_dense_total = total_heads * T * T * _HEAD_DIM
    else:
        attn_dense_total = attn_dense * n_layers
        attn_sparse = attn_dense_total  # no head sparsity

    dense_total = attn_dense_total + ffn_dense
    sparse_total = attn_sparse + ffn_sparse
    speedup = dense_total / max(sparse_total, 1.0)
    return {
        "attn_dense": attn_dense_total,
        "attn_sparse": attn_sparse,
        "attn_saved_pct": (1 - attn_sparse / attn_dense_total) * 100,
        "ffn_dense": ffn_dense,
        "ffn_sparse": ffn_sparse,
        "ffn_saved_pct": (1 - ffn_sparse / ffn_dense) * 100,
        "total_saved_pct": (1 - sparse_total / dense_total) * 100,
        "ttft_speedup": speedup,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_inference(mode: str, all_samples, args, rank_log_dir: str):
    """Run inference on all samples using an already-running server."""
    port = args.port
    results = {}
    for ds_name, samples in all_samples.items():
        print(f"\n  [{mode}] {ds_name} ({len(samples)} samples)", flush=True)
        ds_rows = []
        for i, s in enumerate(samples):
            try:
                text, e2e = generate(port, s["prompt"], args.max_new)
            except Exception as e:
                print(f"    [{mode}] {ds_name} #{i} error: {e}")
                continue
            ans = _clean_answer(text)
            f1 = f1_score(ans, s["gold"])
            em = em_score(ans, s["gold"])
            ds_rows.append(
                {"f1": f1, "em": em, "ans": ans[:60], "gold": s["gold"][:40]}
            )
            print(
                f"    #{i + 1:2d} F1={f1:.2f} EM={em:.0f}  ans={ans[:40]!r}  gold={s['gold'][:30]!r}",
                flush=True,
            )
        if ds_rows:
            n = len(ds_rows)
            results[ds_name] = {
                "f1": sum(r["f1"] for r in ds_rows) / n,
                "em": sum(r["em"] for r in ds_rows) / n,
                "n": n,
            }

    keep_ratio = parse_keep_ratio(rank_log_dir)
    return {"mode": mode, "datasets": results, "keep_ratio": keep_ratio}


def run_mode(mode: str, all_samples, args):
    """Start server, run inference, stop server."""
    port = args.port
    tag = f"longbench_{mode}"
    rank_log_dir = f"/tmp/{tag}_ranklogs"
    server_log = f"/tmp/{tag}_server.log"
    subprocess.run(f"rm -rf {rank_log_dir} {server_log}", shell=True)

    manage_server = not getattr(args, "no_server", False)

    print(f"\n{'=' * 60}")
    print(
        f"  MODE: {mode}  (attention={'redknot_mla' if mode == 'sparse' else 'dsv4'})"
    )
    print(f"{'=' * 60}", flush=True)

    if manage_server:
        proc, lf = start_server(mode, port, rank_log_dir, server_log)
    else:
        proc, lf = None, None

    try:
        if manage_server:
            if not wait_ready(port, server_log, timeout=args.load_timeout):
                print(f"[{mode}] server failed to become ready")
                return None
        else:
            # verify server is reachable
            try:
                requests.get(
                    f"http://127.0.0.1:{port}/health", timeout=5
                ).raise_for_status()
            except Exception:
                print(f"[{mode}] --no-server but server not reachable on port {port}")
                return None

        # warmup
        print(f"[{mode}] warmup...", flush=True)
        try:
            generate(port, "Hello, what is the answer? Answer:", 3)
        except Exception as e:
            print(f"[{mode}] warmup error: {e}")

        return _run_inference(mode, all_samples, args, rank_log_dir)
    finally:
        if manage_server and proc is not None:
            stop_server(proc, lf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--port", type=int, default=31995)
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument(
        "--max-tokens-per-block",
        type=int,
        default=2000,
        help="Tokens per block (total seq_len = n_blocks * this). "
        "Use --seq-lengths to sweep multiple total lengths instead.",
    )
    ap.add_argument(
        "--seq-lengths",
        type=str,
        default=None,
        help="Comma-separated list of total context lengths (tokens) to sweep, "
        "e.g. '4000,8000'. Overrides --max-tokens-per-block. "
        "Each length is divided by n_blocks to set per-block size.",
    )
    ap.add_argument("--n-layers", type=int, default=44)
    ap.add_argument("--dense-until", type=int, default=4)
    ap.add_argument("--local-window", type=int, default=128)
    ap.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names to override DATASETS default.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "dense", "sparse"],
        help="Run only dense, only sparse, or both (default).",
    )
    ap.add_argument(
        "--no-server",
        action="store_true",
        help="Assume server is already running on --port. Skip server start/stop.",
    )
    ap.add_argument(
        "--out", type=str, default="/tmp/redknot_longbench_dense_vs_sparse.json"
    )
    args = ap.parse_args()

    datasets = DATASETS
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]

    # Determine seq_lengths to sweep
    if args.seq_lengths:
        seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(",")]
    else:
        seq_lengths = [args.n_blocks * args.max_tokens_per_block]

    all_results = {}  # {seq_len: {"dense": ..., "sparse": ...}}

    for seq_len in seq_lengths:
        max_tpb = seq_len // args.n_blocks
        print(f"\n{'#' * 72}")
        print(f"#  SEQ_LEN = {seq_len}  ({args.n_blocks} blocks x {max_tpb} tokens)")
        print(f"{'#' * 72}\n")

        # Load datasets — require context >= seq_len tokens (with 20% margin for prompt)
        min_ctx = int(seq_len * 0.9)  # slight margin: context can be a bit shorter
        all_samples = {}
        for ds_name in datasets:
            rows = load_dataset(ds_name, args.n_samples, min_context_tokens=min_ctx)
            if not rows:
                print(
                    f"  SKIP {ds_name} at {seq_len}: no rows with >= {min_ctx} tokens"
                )
                continue
            samples = [
                {"prompt": p, "gold": g}
                for row in rows
                for p, g in [build_prompt(row, args.n_blocks, max_tpb)]
            ]
            all_samples[ds_name] = samples
            print(f"  Loaded {ds_name}: {len(samples)} samples")

        if not all_samples:
            print(f"  No datasets available for seq_len={seq_len}, skipping.")
            all_results[seq_len] = {"dense": None, "sparse": None}
            continue

        # Temporarily override max_tokens_per_block for run_mode
        orig_tpb = args.max_tokens_per_block
        args.max_tokens_per_block = max_tpb

        dense = None
        sparse = None
        if args.mode in ("both", "dense"):
            dense = run_mode("dense", all_samples, args)
        if args.mode in ("both", "sparse"):
            sparse = run_mode("sparse", all_samples, args)

        args.max_tokens_per_block = orig_tpb
        all_results[seq_len] = {"dense": dense, "sparse": sparse}

    # ---- Report ----
    print("\n" + "=" * 72)
    print("RedKnot DeepSeek-V4 LongBench: dense vs sparse (head+indexer-MoE)")
    print(f"PP={PP_SIZE}  datasets={datasets}")
    print("=" * 72)

    for seq_len in seq_lengths:
        dense = all_results[seq_len]["dense"]
        sparse = all_results[seq_len]["sparse"]
        max_tpb = seq_len // args.n_blocks

        print(f"\n{'=' * 60}")
        print(f"  SEQ_LEN = {seq_len} ({args.n_blocks} x {max_tpb})")
        print(f"{'=' * 60}")

        for ds_name in datasets:
            print(f"\n  --- {ds_name} ---")
            for label, r in [("dense", dense), ("sparse", sparse)]:
                if r and ds_name in r["datasets"]:
                    d = r["datasets"][ds_name]
                    print(
                        f"    {label:<8} F1={d['f1']:.3f}  EM={d['em']:.3f}  n={d['n']}"
                    )
            if (
                dense
                and sparse
                and ds_name in dense["datasets"]
                and ds_name in sparse["datasets"]
            ):
                dd = dense["datasets"][ds_name]
                sd = sparse["datasets"][ds_name]
                df1 = sd["f1"] - dd["f1"]
                dem = sd["em"] - dd["em"]
                print(f"    {'delta':>8} F1={df1:+.3f}  EM={dem:+.3f}")

        # ---- Compute + TTFT estimation ----
        kr_sparse = sparse["keep_ratio"] if sparse else None
        head_policy = parse_head_policy(f"/tmp/longbench_sparse_ranklogs/rank0.log")
        if head_policy is None:
            head_policy = parse_head_policy("/tmp/longbench_sparse_server.log")
        print(
            f"\n  --- Compute savings & estimated TTFT speedup (seq_len~{seq_len}) ---"
        )
        print(f"    measured MoE keep_ratio = {kr_sparse}")
        print(f"    head policy = {head_policy}")
        if kr_sparse is not None:
            est = estimate_ttft_speedup(
                seq_len=seq_len,
                keep_ratio=kr_sparse,
                head_policy=head_policy,
                n_layers=args.n_layers,
                dense_until=args.dense_until,
                local_window=args.local_window,
            )
            print(f"    Attention FLOPs saved : {est['attn_saved_pct']:.1f}%")
            print(f"    FFN/MoE  FLOPs saved : {est['ffn_saved_pct']:.1f}%")
            print(f"    TOTAL    FLOPs saved : {est['total_saved_pct']:.1f}%")
            print(
                f"    => estimated TTFT speedup (compute-bound): {est['ttft_speedup']:.2f}x"
            )
            if sparse:
                sparse["estimate"] = est

        # Also provide extrapolated estimates for longer lengths
        # (using measured keep_ratio from actual run)
        if kr_sparse is not None:
            extrap_lengths = [l for l in [4000, 8000, 16000, 32000] if l != seq_len]
            if extrap_lengths:
                print(
                    f"\n  --- Extrapolated TTFT estimates (using keep_ratio={kr_sparse:.4f}) ---"
                )
                for el in extrap_lengths:
                    ext_est = estimate_ttft_speedup(
                        seq_len=el,
                        keep_ratio=kr_sparse,
                        head_policy=head_policy,
                        n_layers=args.n_layers,
                        dense_until=args.dense_until,
                        local_window=args.local_window,
                    )
                    print(
                        f"    seq_len={el:>6d}: total_saved={ext_est['total_saved_pct']:.1f}%  TTFT_speedup={ext_est['ttft_speedup']:.2f}x"
                    )

    # Save combined results
    output = {
        "config": {
            "n_samples": args.n_samples,
            "n_blocks": args.n_blocks,
            "seq_lengths": seq_lengths,
            "datasets": datasets,
            "pp_size": PP_SIZE,
            "tp_size": TP_SIZE,
            "dense_until": args.dense_until,
            "local_window": args.local_window,
        },
        "results": {str(sl): all_results[sl] for sl in seq_lengths},
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
