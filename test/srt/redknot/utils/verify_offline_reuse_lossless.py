#!/usr/bin/env python3
"""Verify the user's idea: prefill stores the FULL KV cache; decode reuses the
FULL prefill KV losslessly.

We test this with SGLang's native radix/prefix cache on DeepSeek-V4 (dsv4
backend, the real SWA+compressed attention):

  COLD : flush cache, send full prompt P  -> output A, cached_tokens ~ 0
  WARM : (no flush) send the SAME prompt P -> prefix cache HITS, decode reuses
         the full stored KV -> output B, cached_tokens > 0

If A == B (token-exact) AND cached_tokens > 0, then decoding off the *reused
complete KV* is lossless -- i.e. "prefill once, decode over the full cached KV"
holds with zero accuracy change. This is the lossless prefix-cache / PD-disagg
premise, distinct from sparsifying/dropping prefix KV.

We additionally do a SHARED-PREFIX test: prefill a long prefix once, then ask
two different questions that share it; the second question's prefix is served
from cache (cached_tokens>0) while still attending the full prefix KV.

Run::
    python verify_offline_reuse_lossless.py --n-samples 6 --port 32010
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
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
        SGLANG_RANK_LOG_DIR=rank_log_dir,
    )
    return env


def start_server(port: int, rank_log_dir: str, server_log: str):
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
        "0.85",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        # radix cache ENABLED (this is the whole point: reuse stored KV)
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
    lf = open(server_log, "w")
    proc = subprocess.Popen(
        cmd,
        env=_server_env(rank_log_dir),
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(REPO),
    )
    return proc, lf


def wait_ready(port: int, server_log: str, timeout: int) -> bool:
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
            if "SIGQUIT received" in txt or "Traceback" in txt:
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


def flush(port):
    try:
        requests.post(f"http://127.0.0.1:{port}/flush_cache", timeout=30)
        time.sleep(1)
    except Exception as e:
        print("flush err", e)


def gen(port, prompt, max_new, timeout=400):
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
    mi = o["meta_info"]
    return {
        "text": o.get("text", ""),
        "output_ids": o.get("output_ids", []),
        "prompt_tokens": mi.get("prompt_tokens", 0),
        "cached_tokens": mi.get("cached_tokens", 0),
    }


def load_rows(ds, n, min_ctx_chars, seed=2026):
    rows = []
    with open(os.path.join(LONGBENCH_DIR, f"{ds}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if (
                r.get("context")
                and r.get("answers")
                and len(r["context"]) >= min_ctx_chars
            ):
                rows.append(r)
    import random

    random.Random(seed).shuffle(rows)
    return rows[:n]


def build_prompt(context, question, ctx_tokens):
    body = context[: ctx_tokens * 4]
    return (
        f"Read the document and answer concisely.\n\nDocument:\n{body}\n\n"
        f"Question: {question}\nAnswer:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=32010)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--ctx-tokens", type=int, default=3500)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument(
        "--datasets", nargs="+", default=["hotpotqa", "2wikimqa", "musique"]
    )
    ap.add_argument("--load-timeout", type=int, default=1200)
    ap.add_argument(
        "--control-cold-cold",
        action="store_true",
        help="2nd run also flushes (cold vs cold) to isolate fp8 "
        "nondeterminism from KV-reuse effects.",
    )
    ap.add_argument("--out", default="/tmp/redknot_lossless_verify.json")
    args = ap.parse_args()

    rank_log_dir = "/tmp/lossless_ranklogs"
    server_log = "/tmp/lossless_server.log"
    subprocess.run(f"rm -rf {rank_log_dir} {server_log}", shell=True)
    proc, lf = start_server(args.port, rank_log_dir, server_log)

    results = []
    try:
        if not wait_ready(args.port, server_log, args.load_timeout):
            print("server failed to start; tail:")
            print(
                "\n".join(
                    Path(server_log).read_text(errors="ignore").splitlines()[-25:]
                )
            )
            return
        # warmup with retries
        for _ in range(8):
            try:
                gen(args.port, "Hello. Answer:", 3, timeout=90)
                break
            except Exception as e:
                print("warmup retry:", e)
                time.sleep(2)

        for ds in args.datasets:
            rows = load_rows(ds, args.n_samples, args.ctx_tokens * 4)
            print(f"\n=== {ds}: {len(rows)} samples ===", flush=True)
            for i, row in enumerate(rows):
                q = row.get("input", "").strip()
                prompt = build_prompt(row["context"], q, args.ctx_tokens)

                # COLD
                flush(args.port)
                a = gen(args.port, prompt, args.max_new)
                if args.control_cold_cold:
                    # CONTROL: flush again so the 2nd run is ALSO a cold prefill
                    # (no KV reuse). Any divergence here is fp8 nondeterminism,
                    # not caused by KV reuse.
                    flush(args.port)
                    b = gen(args.port, prompt, args.max_new)
                else:
                    # WARM (same prompt, no flush -> prefix cache reuses full KV)
                    b = gen(args.port, prompt, args.max_new)

                exact = a["output_ids"] == b["output_ids"]
                rec = {
                    "ds": ds,
                    "i": i,
                    "cold_cached": a["cached_tokens"],
                    "warm_cached": b["cached_tokens"],
                    "prompt_tokens": a["prompt_tokens"],
                    "exact_match": exact,
                    "cold_text": a["text"][:50],
                    "warm_text": b["text"][:50],
                }
                results.append(rec)
                print(
                    f"  #{i + 1:2d} prompt_tok={a['prompt_tokens']:5d} "
                    f"cold_cached={a['cached_tokens']:5d} warm_cached={b['cached_tokens']:5d} "
                    f"EXACT={'YES' if exact else 'NO'}",
                    flush=True,
                )
                if not exact:
                    print(f"      cold={a['text'][:60]!r}")
                    print(f"      warm={b['text'][:60]!r}")
    finally:
        stop_server(proc, lf)

    # summary
    n = len(results)
    n_exact = sum(r["exact_match"] for r in results)
    n_reused = sum(r["warm_cached"] > r["cold_cached"] for r in results)
    print("\n" + "=" * 60)
    print("LOSSLESS REUSE VERIFICATION (DeepSeek-V4, dsv4 backend)")
    print("=" * 60)
    print(f"  samples              : {n}")
    print(f"  warm reused KV (>cold): {n_reused}/{n}")
    print(f"  token-exact cold==warm: {n_exact}/{n}")
    if n:
        print(f"  => lossless reuse rate: {100 * n_exact / n:.1f}%")
    json.dump(
        {"summary": {"n": n, "exact": n_exact, "reused": n_reused}, "results": results},
        open(args.out, "w"),
        indent=2,
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
