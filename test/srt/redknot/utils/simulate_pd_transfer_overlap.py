#!/usr/bin/env python3
"""Simulate PD-disaggregated c4-KV transfer overlap for DeepSeek-V4-Flash.

In PD disaggregation the prefill node must ship the prefix KV to the decode node.
For DSV4 the big chunk is the c4 compressed KV. KEY INSIGHT: during prefill the
node attends via SWA(128) + indexer-selected top-512, so the UNSELECTED c4 KV is
NOT needed by prefill itself -> it can be shipped to the decode node *while
prefill is still running* (overlap). The selected (hot) pages are shipped last.

All c4 KV ultimately reaches decode (LOSSLESS). The optimization is transfer
ORDER vs prefill compute overlap -> hide transfer behind prefill -> lower TTFT
-> higher QPS.

Three strategies (TTFT = time until decode node can start):
  A. serial     : prefill, THEN transfer all c4 KV.  TTFT = T_prefill + T_xfer_all
  B. naive ovl  : transfer all c4 KV overlapped with prefill.
                  TTFT = max(T_prefill, T_xfer_all) + tail
  C. indexer-aware (YOUR idea): ship UNSELECTED c4 KV during prefill (overlap),
                  ship SELECTED (small, 6.3MB) after prefill.
                  TTFT = max(T_prefill, T_xfer_unselected) + T_xfer_selected

We MEASURE real T_prefill per length (run the server), and compute transfer
times from real c4 byte counts and an assumed PD link bandwidth.

Usage:
  python simulate_pd_transfer_overlap.py --lengths 8000,16000,32000,64000 \
      --bw-gbps 100 --port 32080
"""

from __future__ import annotations

import argparse
import json
import os
import signal
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
MODEL_CFG = os.path.join(MODEL_PATH, "config.json")
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
PP_SIZE = 8
LATENT_B = 584  # bytes/token/layer


def c4_bytes(T, cfg):
    cr = cfg["compress_ratios"]
    n4 = Counter(cr)[4]
    topk = cfg.get("index_topk", 512)
    c4_tok = T // 4
    selected = min(topk, c4_tok)
    unselected = max(0, c4_tok - selected)
    return {
        "total": n4 * c4_tok * LATENT_B,
        "selected": n4 * selected * LATENT_B,
        "unselected": n4 * unselected * LATENT_B,
        "unselected_frac": unselected / c4_tok if c4_tok else 0,
    }


# ── server (to measure real prefill latency) ───────────────────────────────
def _env(rank_log_dir):
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


def start_server(port, rank_log_dir, server_log):
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
        env=_env(rank_log_dir),
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


def measure_prefill(port, prompt, repeats=3, timeout=600):
    """Measure prefill (TTFT-ish): use max_new_tokens=1 so e2e ~ prefill time."""
    lats = []
    for _ in range(repeats):
        try:
            r = requests.post(
                f"http://127.0.0.1:{port}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                },
                timeout=timeout,
                headers={"Connection": "close"},
            )
            r.raise_for_status()
            mi = r.json()["meta_info"]
            lats.append(mi.get("e2e_latency", 0.0))
        except Exception as e:
            print("   measure err:", e, flush=True)
    return (sum(lats) / len(lats)) if lats else None


def load_long_prompt(min_ctx, seed=2026):
    import random

    cands = []
    for ds in ["narrativeqa", "musique", "hotpotqa"]:
        p = os.path.join(LONGBENCH_DIR, f"{ds}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("context") and len(r["context"]) // 4 >= min_ctx:
                    cands.append(r["context"])
    random.Random(seed).shuffle(cands)
    return cands[0] if cands else None


def tile(ctx, T):
    tc = T * 4
    b = ctx
    while len(b) < tc:
        b += "\n\n" + ctx
    return b[:tc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="8000,16000,32000,64000")
    ap.add_argument(
        "--bw-gbps",
        type=float,
        default=100.0,
        help="PD link bandwidth (GB/s effective). 100=NVLink-ish, "
        "25=400GbE RDMA, 12=PCIe4 x16",
    )
    ap.add_argument("--port", type=int, default=32080)
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="/tmp/redknot_pd_overlap_sim.json")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    cfg = json.load(open(MODEL_CFG))
    bw = args.bw_gbps * 1e9  # bytes/s

    rank_log_dir = "/tmp/pdsim_ranklogs"
    server_log = "/tmp/pdsim_server.log"
    subprocess.run(f"rm -rf {rank_log_dir} {server_log}", shell=True)
    proc, lf = start_server(args.port, rank_log_dir, server_log)

    measured = {}
    try:
        if not wait_ready(args.port, server_log, args.load_timeout):
            print("server FAILED")
            print(
                "\n".join(
                    Path(server_log).read_text(errors="ignore").splitlines()[-15:]
                )
            )
            return
        # warmup
        for _ in range(8):
            try:
                measure_prefill(args.port, "Hello.", repeats=1, timeout=90)
                break
            except Exception:
                time.sleep(2)
        base_ctx = load_long_prompt(min_ctx=8000)
        if base_ctx is None:
            print("no long ctx")
            return
        for T in lengths:
            prompt = (
                "Read and answer.\n\n" + tile(base_ctx, T) + "\n\nQ: summarize.\nA:"
            )
            tp = measure_prefill(args.port, prompt, repeats=args.repeats)
            measured[T] = tp
            print(f"  T={T:>6}: measured prefill ~ {tp:.2f}s", flush=True)
    finally:
        stop_server(proc, lf)

    # ── analytic transfer model ──
    print("\n" + "=" * 90)
    print(f"PD c4-KV transfer overlap simulation  (link BW = {args.bw_gbps} GB/s)")
    print("=" * 90)
    hdr = (
        f"{'len':>6} {'prefill_s':>9} {'c4_tot_MB':>9} {'unsel%':>6} "
        f"{'xfer_all_s':>10} | {'A:serial':>9} {'B:naive':>9} {'C:yours':>9} "
        f"{'C speedup':>9}"
    )
    print(hdr)
    results = {}
    for T in lengths:
        tp = measured.get(T)
        if tp is None:
            continue
        cb = c4_bytes(T, cfg)
        t_all = cb["total"] / bw
        t_unsel = cb["unselected"] / bw
        t_sel = cb["selected"] / bw
        # TTFT for each strategy
        ttft_A = tp + t_all  # serial
        ttft_B = max(tp, t_all)  # naive overlap (all during prefill)
        ttft_C = max(tp, t_unsel) + t_sel  # yours: unsel overlapped, sel after
        speedup = ttft_A / ttft_C if ttft_C else 0
        print(
            f"{T:>6} {tp:>9.2f} {cb['total'] / 1e6:>9.1f} "
            f"{100 * cb['unselected_frac']:>5.0f}% {t_all:>10.4f} | "
            f"{ttft_A:>9.3f} {ttft_B:>9.3f} {ttft_C:>9.3f} {speedup:>8.2f}x"
        )
        results[str(T)] = {
            "prefill_s": tp,
            "c4_total_MB": cb["total"] / 1e6,
            "unselected_frac": cb["unselected_frac"],
            "xfer_all_s": t_all,
            "xfer_unsel_s": t_unsel,
            "xfer_sel_s": t_sel,
            "ttft_serial": ttft_A,
            "ttft_naive_overlap": ttft_B,
            "ttft_yours": ttft_C,
            "speedup_vs_serial": speedup,
        }
    # QPS proxy: 1/TTFT-bound throughput ratio
    print("\nQPS proxy (TTFT-bound, C vs A):")
    for T in lengths:
        r = results.get(str(T))
        if r:
            print(
                f"  T={T:>6}: QPS gain ~ {r['speedup_vs_serial']:.2f}x  "
                f"(TTFT {r['ttft_serial']:.2f}s -> {r['ttft_yours']:.2f}s)"
            )
    json.dump(
        {"bw_gbps": args.bw_gbps, "measured_prefill": measured, "results": results},
        open(args.out, "w"),
        indent=2,
    )
    print(f"\nSaved {args.out}")
    print(
        "\nNOTE: all c4 KV reaches decode in every strategy -> LOSSLESS. "
        "Only transfer ORDER differs."
    )


if __name__ == "__main__":
    main()
