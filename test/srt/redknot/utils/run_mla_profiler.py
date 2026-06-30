#!/usr/bin/env python3
"""Drive the RedKnot MLA head-locality profiler on DeepSeek-V4.

Starts a profiler-enabled dsv4 server (TP=1 so head classification is complete),
feeds N real long (>=8K) single-sequence prefills, and the model exports per
prefill:
  <out>.json                 : per-(layer,head) global/local classification
  <out>_concentration.json   : per-layer "tokens needed for coverage% mass" curve

We then read the concentration JSON to identify SWA-dominant layers (layers whose
heads concentrate attention nearby -> dropping their compressed-extra prefix KV
should not hurt accuracy).

Usage:
  python run_mla_profiler.py --n-prefills 8 --min-ctx 8000 --port 32070
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


def _server_env(rank_log_dir):
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


def start_server(port, rank_log_dir, server_log, profile_out, coverage):
    os.makedirs(rank_log_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--attention-backend",
        "redknot_mla",
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
        "--redknot-mla-profile-enable",
        "--redknot-mla-profile-out",
        profile_out,
        "--redknot-mla-profile-coverage",
        str(coverage),
        "--redknot-mla-profile-sample-queries",
        "256",
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


def gen(port, prompt, max_new=4, timeout=600):
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
    return r.json()["meta_info"]


def load_long_prompts(n, min_ctx, seed=2026):
    import random

    prompts = []
    for ds in ["hotpotqa", "musique", "narrativeqa", "2wikimqa"]:
        p = os.path.join(LONGBENCH_DIR, f"{ds}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("context") and len(r["context"]) // 4 >= min_ctx:
                    q = r.get("input", "").strip()
                    prompts.append(
                        f"Read and answer.\n\n{r['context']}\n\nQuestion: {q}\nAnswer:"
                    )
    random.Random(seed).shuffle(prompts)
    return prompts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prefills", type=int, default=8)
    ap.add_argument("--min-ctx", type=int, default=8000)
    ap.add_argument("--coverage", type=float, default=0.95)
    ap.add_argument("--port", type=int, default=32070)
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--profile-out", default="/tmp/redknot_mla_profile.json")
    args = ap.parse_args()

    rank_log_dir = "/tmp/profiler_ranklogs"
    server_log = "/tmp/profiler_server.log"
    subprocess.run(
        f"rm -rf {rank_log_dir} {server_log} {args.profile_out} "
        f"{args.profile_out.replace('.json', '')}_concentration.json",
        shell=True,
    )

    prompts = load_long_prompts(args.n_prefills, args.min_ctx)
    print(f"Loaded {len(prompts)} real prefills >= {args.min_ctx} tok", flush=True)

    proc, lf = start_server(
        args.port, rank_log_dir, server_log, args.profile_out, args.coverage
    )
    try:
        if not wait_ready(args.port, server_log, args.load_timeout):
            print("server FAILED; tail:")
            print(
                "\n".join(
                    Path(server_log).read_text(errors="ignore").splitlines()[-20:]
                )
            )
            return
        # warmup (NOT single-seq counted; profiler only samples single-seq extend)
        for _ in range(8):
            try:
                gen(args.port, "Hello.", 2, timeout=90)
                break
            except Exception as e:
                print("warmup retry:", e, flush=True)
                time.sleep(2)
        # feed single-sequence prefills (profiler samples each)
        for i, p in enumerate(prompts):
            try:
                mi = gen(args.port, p, max_new=2)
                print(f"  prefill #{i + 1} ptok={mi.get('prompt_tokens')}", flush=True)
            except Exception as e:
                print(f"  prefill #{i + 1} err: {e}", flush=True)
            time.sleep(1)
        time.sleep(3)  # let last incremental export flush
    finally:
        stop_server(proc, lf)

    conc = args.profile_out.replace(".json", "") + "_concentration.json"
    print(
        f"\nProfile head-class : {args.profile_out} "
        f"(exists={os.path.exists(args.profile_out)})"
    )
    print(f"Concentration curve: {conc} (exists={os.path.exists(conc)})")


if __name__ == "__main__":
    main()
