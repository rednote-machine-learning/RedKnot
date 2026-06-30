#!/usr/bin/env python3
"""TRUE prefix compression for DeepSeek-V4: sweep the indexer top-k depth.

V4 attention = SWA(128 raw) + extra_k_cache(compressed full prefix) +
indexer(ranks prefix by relevance, keeps top-512). The real compression knob
is HOW MANY of the indexer-ranked compressed-prefix tokens each query attends
to. We clamp it via env REDKNOT_C4_TOPK_CLAMP (kept the full prefix range and
relevance ranking; only attend the most-relevant first k).

  k=512 : baseline (native, full indexer budget)
  k=256/128/64 : progressively more aggressive prefix compression

Same dsv4 backend for all; only the env knob changes -> a clean accuracy curve.

Uses REAL long LongBench samples (no artificial tiling) up to ~16K natural.

Usage:
  python benchmark_indexer_topk_sweep.py --ks 512,256,128,64 --n-samples 5 --port 32050
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
PP_SIZE = 8
DATASETS = ["hotpotqa", "2wikimqa", "musique"]


def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def best_f1(pred, golds):
    return max((f1_score(pred, g) for g in golds), default=0.0)


def clean(t):
    return re.split(r"(?i)(?:\n\s*question\s*:|\n\s*document\s*:|<\|)", t)[0].strip()


def load_rows(ds, n, min_ctx_tokens, seed=2026):
    """Load REAL rows with at least min_ctx_tokens of natural context."""
    rows = []
    with open(os.path.join(LONGBENCH_DIR, f"{ds}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("context") and r.get("answers"):
                if len(r["context"]) // 4 >= min_ctx_tokens:
                    rows.append(r)
    import random

    random.Random(seed).shuffle(rows)
    return rows[:n]


def build_prompt(row):
    ctx = row["context"]
    q = row.get("input", "").strip()
    prompt = (
        f"Read the documents and answer the question concisely.\n\n"
        f"{ctx}\n\nQuestion: {q}\nAnswer:"
    )
    golds = (
        row["answers"] if isinstance(row["answers"], list) else [str(row["answers"])]
    )
    return prompt, [str(x) for x in golds]


def _server_env(rank_log_dir, c4_clamp):
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
        SGLANG_RANK_LOG_DIR=rank_log_dir,
        REDKNOT_C4_TOPK_CLAMP=str(c4_clamp) if c4_clamp else "",
    )
    return env


def start_server(port, rank_log_dir, server_log, c4_clamp):
    os.makedirs(rank_log_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--attention-backend",
        "dsv4",
        "--tp-size",
        "1",
        "--pp-size",
        str(PP_SIZE),
        "--mem-fraction-static",
        "0.8",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "8192",
        "--trust-remote-code",
        "--port",
        str(port),
    ]
    lf = open(server_log, "w")
    proc = subprocess.Popen(
        cmd,
        env=_server_env(rank_log_dir, c4_clamp),
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(REPO),
    )
    return proc, lf


def wait_ready(port, server_log, timeout):
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
            if (
                "Received sigquit" in txt
                or "Not enough memory" in txt
                or "Traceback (most recent" in txt
            ):
                return False
        time.sleep(3)
    return False


def stop_server(proc, lf):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        lf.close()
    except Exception:
        pass
    subprocess.run("pkill -9 -f sglang::scheduler", shell=True)
    subprocess.run("pkill -9 -f sglang.launch_server", shell=True)
    time.sleep(8)


def gen(port, prompt, max_new, timeout=600):
    r = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
        },
        timeout=timeout,
        headers={"Connection": "close"},
    )
    r.raise_for_status()
    o = r.json()
    return o.get("text", ""), o["meta_info"]


def warmup(port):
    for _ in range(8):
        try:
            gen(port, "Hello. Answer:", 3, timeout=90)
            return True
        except Exception as e:
            print("  warmup retry:", e, flush=True)
            time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ks", default="512,256,128,64", help="indexer topk clamp values; 512=baseline"
    )
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument(
        "--min-ctx",
        type=int,
        default=6000,
        help="min natural context tokens (real long samples)",
    )
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--port", type=int, default=32050)
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--out", default="/tmp/redknot_indexer_topk_sweep.json")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    # fixed real samples shared across all k (fair comparison)
    samples = {
        ds: [build_prompt(r) for r in load_rows(ds, args.n_samples, args.min_ctx)]
        for ds in datasets
    }
    for ds in datasets:
        print(
            f"  {ds}: {len(samples[ds])} real samples >= {args.min_ctx} tok", flush=True
        )

    results = {}
    for k in ks:
        clamp = 0 if k >= 512 else k  # 512 = native (no clamp)
        tag = f"topk_{k}"
        rank_log_dir = f"/tmp/{tag}_ranklogs"
        server_log = f"/tmp/{tag}_server.log"
        subprocess.run(f"rm -rf {rank_log_dir} {server_log}", shell=True)
        print(
            f"\n{'=' * 64}\n  INDEXER TOP-K = {k}  (clamp={clamp or 'none/512'})\n{'=' * 64}",
            flush=True,
        )
        proc, lf = start_server(args.port, rank_log_dir, server_log, clamp)
        entry = {"k": k}
        try:
            if not wait_ready(args.port, server_log, args.load_timeout):
                print(f"  [k={k}] server FAILED", flush=True)
                results[str(k)] = {"error": "server_failed"}
                continue
            warmup(args.port)
            ds_f1 = {}
            for ds, plist in samples.items():
                f1s = []
                for i, (prompt, golds) in enumerate(plist):
                    try:
                        text, mi = gen(args.port, prompt, args.max_new)
                    except Exception as e:
                        print(f"    [k={k}] {ds} #{i} err: {e}", flush=True)
                        continue
                    v = best_f1(clean(text), golds)
                    f1s.append(v)
                    print(
                        f"    [k={k}] {ds} #{i + 1} ptok={mi.get('prompt_tokens')} "
                        f"F1={v:.2f}",
                        flush=True,
                    )
                if f1s:
                    ds_f1[ds] = sum(f1s) / len(f1s)
            entry["f1_per_ds"] = ds_f1
            entry["f1_avg"] = sum(ds_f1.values()) / len(ds_f1) if ds_f1 else None
            print(f"  [k={k}] avg F1 = {entry['f1_avg']}", flush=True)
        finally:
            stop_server(proc, lf)
        results[str(k)] = entry
        json.dump(results, open(args.out, "w"), indent=2)

    # report
    print("\n" + "=" * 60)
    print("INDEXER TOP-K SWEEP (true prefix compression)")
    print("=" * 60)
    base = results.get("512", {}).get("f1_avg")
    print(f"{'top-k':>6} | {'avg F1':>8} | {'retention vs 512':>16}")
    for k in ks:
        r = results.get(str(k), {})
        fa = r.get("f1_avg")
        ret = f"{100 * fa / base:.0f}%" if (fa and base) else "-"
        print(f"{k:>6} | {str(round(fa, 3) if fa else '-'):>8} | {ret:>16}")
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
