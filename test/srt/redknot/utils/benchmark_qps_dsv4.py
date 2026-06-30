#!/usr/bin/env python3
"""QPS / throughput benchmark for DeepSeek-V4-Flash-FP8 on the live SGLang server.

For each context length we submit `concurrency` requests CONCURRENTLY (thread
pool) and measure total wall-clock time, then report:
    QPS       = n_requests / total_wall_time
    QPS/GPU   = QPS / n_gpus   (PP=8 -> 8 GPUs serve one model)
    avg_latency

Two modes share the SAME server process; which one is active depends on the
server launch flags:
    dense    : server started WITHOUT reuse hook / sparse-FFN
    redknot  : server started WITH  reuse hook + sparse-FFN (offline MLA reuse)

So run this script twice (once against each server), or pass --label to tag the
output, then plot both with plot_qps_dsv4.py.

Usage:
    python benchmark_qps_dsv4.py --port 31995 --label redknot \
        --output /tmp/qps_dsv4_redknot.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
BOS, USER, ASST = "<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>"
N_GPUS = int(os.environ.get("REDKNOT_QPS_NGPU", "8"))

# context_len -> concurrency (number of requests submitted at once)
PLAN = {
    16384: 8,
    32768: 8,
    65536: 6,
    131072: 4,
}


def load_pool(ds, seed=2026):
    rows = []
    with open(os.path.join(LB, f"{ds}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("context"):
                rows.append(r)
    import random

    random.Random(seed).shuffle(rows)
    return rows


def build(ctx_tokens, row, pool):
    target_chars = max(ctx_tokens - 200, 256) * 4
    docs = [row["context"]]
    cur = len(docs[0])
    i = 0
    while cur < target_chars and i < len(pool):
        d = pool[i]
        i += 1
        if d is row:
            continue
        docs.append(d.get("context", ""))
        cur += len(docs[-1])
    full = "\n\n".join(docs)[:target_chars]
    q = row.get("input", "").strip()
    user = (
        f"Read the documents and answer the question concisely.\n\n"
        f"Documents:\n{full}\n\nQuestion: {q}\n\nAnswer with only the answer."
    )
    return f"{BOS}{USER}{user}{ASST}"


def one_request(port, prompt, max_new, timeout=2400):
    t0 = time.perf_counter()
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
    return time.perf_counter() - t0, int(o["meta_info"].get("prompt_tokens", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31995)
    ap.add_argument("--label", default="redknot")
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument(
        "--ctx-lens", nargs="+", type=int, default=[16384, 32768, 65536, 131072]
    )
    ap.add_argument("--output", default="/tmp/qps_dsv4.json")
    args = ap.parse_args()

    pool = load_pool(args.dataset)
    results = {}
    for ctx in args.ctx_lens:
        conc = PLAN.get(ctx, 4)
        # build `conc` distinct prompts
        prompts = [build(ctx, pool[i], pool) for i in range(conc)]

        # warm up once (avoid cold-start skew) with a tiny request
        try:
            one_request(args.port, prompts[0], 1)
        except Exception:
            pass

        # concurrent submission
        t0 = time.perf_counter()
        lats, ptoks = [], []
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(one_request, args.port, p, args.max_new) for p in prompts]
            for f in futs:
                try:
                    lat, pt = f.result()
                    lats.append(lat)
                    ptoks.append(pt)
                except Exception as e:
                    print(f"  req err: {e}")
        total = time.perf_counter() - t0

        n = len(lats)
        qps = n / max(total, 1e-6)
        qps_gpu = qps / N_GPUS
        avg_lat = sum(lats) / max(n, 1)
        avg_pt = sum(ptoks) / max(n, 1)
        results[str(ctx)] = {
            "concurrency": conc,
            "n_ok": n,
            "total_s": total,
            "qps": qps,
            "qps_per_gpu": qps_gpu,
            "avg_latency": avg_lat,
            "avg_prompt_tokens": avg_pt,
        }
        print(
            f"[{args.label}] ctx={ctx // 1024}K conc={conc} n={n} "
            f"total={total:.1f}s QPS={qps:.3f} QPS/GPU={qps_gpu:.4f} "
            f"avg_lat={avg_lat:.1f}s ptoks={avg_pt:.0f}"
        )
        with open(args.output, "w") as f:
            json.dump(
                {"label": args.label, "n_gpus": N_GPUS, "results": results}, f, indent=2
            )

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
