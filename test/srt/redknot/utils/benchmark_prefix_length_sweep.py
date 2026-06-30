#!/usr/bin/env python3
"""RedKnot DeepSeek-V4: prefix-length sweep (8K/16K/32K/64K).

PART 1 — ACCURACY (full vs compressed prefix)
    full     : dsv4, native SWA+compressed, full prefix visible.
    compress : redknot_mla head-sparse (local heads see only SWA window).
    Metric: avg F1 over QA datasets at each prefix length.

PART 2 — QPS at FIXED MEMORY (lossless reuse vs recompute)
    Workload: ONE shared long prefix + many different short questions.
    reuse    : radix cache ON -> shared prefix KV is prefilled once and reused
               across all requests (the user's verified lossless idea).
    recompute: radix cache OFF -> every request re-prefills the whole prefix.
    Metric: QPS (completed req/s) under fixed concurrency. Longer prefixes make
            recompute pay the full prefill each time -> reuse's QPS edge grows.

Long prompts are built by tiling LongBench context up to the target length.

Smoke:
  python benchmark_prefix_length_sweep.py --lengths 8000,16000 \
      --n-samples 2 --do-accuracy --do-qps --port 32030
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
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[3]
HEAD_CLASS_DIR = Path(__file__).resolve().parent / "head_class"
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
COMPRESS_CFG = str(HEAD_CLASS_DIR / "dsv4_extreme_local128.json")


# ── accuracy helpers ────────────────────────────────────────────────────────
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
    t = re.split(r"(?i)(?:\n\s*question\s*:|\n\s*document\s*:|<\|)", t)[0]
    return t.strip()


# ── data ────────────────────────────────────────────────────────────────────
def load_rows(ds, n, seed=2026):
    rows = []
    with open(os.path.join(LONGBENCH_DIR, f"{ds}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("context") and r.get("answers"):
                rows.append(r)
    import random

    random.Random(seed).shuffle(rows)
    return rows[:n]


def tile_context(ctx, target_tokens):
    target_chars = target_tokens * 4
    body = ctx
    while len(body) < target_chars:
        body += "\n\n" + ctx
    return body[:target_chars]


def build_prompt(row, target_tokens):
    body = tile_context(row["context"], target_tokens)
    q = row.get("input", "").strip()
    prompt = (
        f"Read the documents and answer the question concisely.\n\n"
        f"{body}\n\nQuestion: {q}\nAnswer:"
    )
    golds = (
        row["answers"] if isinstance(row["answers"], list) else [str(row["answers"])]
    )
    return prompt, [str(x) for x in golds]


# ── server ──────────────────────────────────────────────────────────────────
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


def start_server(
    backend,
    port,
    rank_log_dir,
    server_log,
    *,
    radix,
    head_cfg,
    max_total_tokens,
    mem_frac,
):
    os.makedirs(rank_log_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--attention-backend",
        backend,
        "--tp-size",
        "1",
        "--pp-size",
        str(PP_SIZE),
        "--mem-fraction-static",
        str(mem_frac),
        "--disable-cuda-graph",
        "--skip-server-warmup",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "8192",
        "--trust-remote-code",
        "--port",
        str(port),
    ]
    if max_total_tokens and max_total_tokens > 0:
        cmd += ["--max-total-tokens", str(max_total_tokens)]
    if not radix:
        cmd.append("--disable-radix-cache")
    if head_cfg:
        cmd += ["--redknot-head-config-path", head_cfg]
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
                "SIGQUIT received" in txt
                or "Received sigquit" in txt
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


def gen(port, prompt, max_new, timeout=1200):
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
            gen(port, "Hello. Answer:", 3, timeout=120)
            return True
        except Exception as e:
            print("  warmup retry:", e, flush=True)
            time.sleep(2)
    return False


def flush(port):
    try:
        requests.post(f"http://127.0.0.1:{port}/flush_cache", timeout=30)
        time.sleep(1)
    except Exception:
        pass


# ── QPS load: shared prefix + distinct questions ────────────────────────────
def measure_qps(
    port, shared_prefix, questions, max_new, concurrency, prime_reuse, timeout=1200
):
    """If prime_reuse: send one request first to populate the prefix KV cache,
    then fire the rest concurrently (they reuse the cached prefix).
    Else: fire all concurrently with no priming (each pays full prefill)."""
    if prime_reuse:
        # prime: prefill the shared prefix once (radix cache stores it)
        print("    priming shared prefix...", flush=True)
        try:
            _, mi = gen(port, shared_prefix + questions[0], max_new, timeout=timeout)
            print(
                f"    primed (ptok={mi.get('prompt_tokens')} "
                f"cached={mi.get('cached_tokens')})",
                flush=True,
            )
        except Exception as e:
            print("    prime err:", e, flush=True)

    lat, errors, done_ct, cached_ct = [], [0], [0], []
    lock = threading.Lock()

    def one(qi):
        prompt = shared_prefix + questions[qi % len(questions)]
        t0 = time.time()
        try:
            _, mi = gen(port, prompt, max_new, timeout=timeout)
            with lock:
                lat.append(time.time() - t0)
                cached_ct.append(mi.get("cached_tokens", 0))
                done_ct[0] += 1
                print(
                    f"      req done {done_ct[0]} lat={time.time() - t0:.1f}s "
                    f"cached={mi.get('cached_tokens')}",
                    flush=True,
                )
        except Exception as e:
            with lock:
                errors[0] += 1
                print(f"      req ERR: {e}", flush=True)

    n = len(questions)
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(one, range(n)))
    wall = time.time() - t_start
    done = len(lat)
    return {
        "concurrency": concurrency,
        "n_requests": n,
        "completed": done,
        "errors": errors[0],
        "wall_s": round(wall, 2),
        "qps": round(done / wall, 4) if wall > 0 else 0,
        "avg_lat_s": round(sum(lat) / done, 2) if done else None,
        "avg_cached": round(sum(cached_ct) / len(cached_ct), 1) if cached_ct else 0,
    }


# ── main ────────────────────────────────────────────────────────────────────
def run_accuracy(
    backend, head_cfg, length, samples, max_new, port, load_timeout, mem_frac, mtt, tag
):
    rank_log_dir = f"/tmp/{tag}_ranklogs"
    server_log = f"/tmp/{tag}_server.log"
    subprocess.run(f"rm -rf {rank_log_dir} {server_log}", shell=True)
    proc, lf = start_server(
        backend,
        port,
        rank_log_dir,
        server_log,
        radix=False,
        head_cfg=head_cfg,
        max_total_tokens=mtt,
        mem_frac=mem_frac,
    )
    entry = {"backend": backend}
    try:
        if not wait_ready(port, server_log, load_timeout):
            print(f"  [{tag}] server FAILED", flush=True)
            return {"error": "server_failed"}
        warmup(port)
        ds_f1 = {}
        for ds, plist in samples.items():
            f1s = []
            for i, (prompt, golds) in enumerate(plist):
                try:
                    text, mi = gen(port, prompt, max_new)
                except Exception as e:
                    print(f"    [{tag}] {ds} #{i} err: {e}", flush=True)
                    continue
                v = best_f1(clean(text), golds)
                f1s.append(v)
                print(
                    f"    [{tag}] {ds} #{i + 1} ptok={mi.get('prompt_tokens')} "
                    f"F1={v:.2f}",
                    flush=True,
                )
            if f1s:
                ds_f1[ds] = sum(f1s) / len(f1s)
        entry["f1_per_ds"] = ds_f1
        entry["f1_avg"] = sum(ds_f1.values()) / len(ds_f1) if ds_f1 else None
        print(f"  [{tag}] avg F1 = {entry['f1_avg']}", flush=True)
    finally:
        stop_server(proc, lf)
    return entry


def run_qps(
    mode,
    length,
    shared_prefix,
    questions,
    max_new,
    concurrency,
    port,
    load_timeout,
    mem_frac,
    mtt,
    tag,
):
    """mode: 'reuse' (radix ON, prime) or 'recompute' (radix OFF)."""
    radix = mode == "reuse"
    rank_log_dir = f"/tmp/{tag}_ranklogs"
    server_log = f"/tmp/{tag}_server.log"
    subprocess.run(f"rm -rf {rank_log_dir} {server_log}", shell=True)
    proc, lf = start_server(
        "dsv4",
        port,
        rank_log_dir,
        server_log,
        radix=radix,
        head_cfg=None,
        max_total_tokens=mtt,
        mem_frac=mem_frac,
    )
    entry = {"mode": mode, "radix": radix}
    try:
        if not wait_ready(port, server_log, load_timeout):
            print(f"  [{tag}] server FAILED", flush=True)
            return {"error": "server_failed"}
        warmup(port)
        if not radix:
            flush(port)
        q = measure_qps(
            port, shared_prefix, questions, max_new, concurrency, prime_reuse=radix
        )
        entry.update(q)
        print(
            f"  [{tag}] QPS={q['qps']} completed={q['completed']}/{q['n_requests']} "
            f"errors={q['errors']} avg_lat={q['avg_lat_s']}s",
            flush=True,
        )
    finally:
        stop_server(proc, lf)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="8000,16000,32000,64000")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--port", type=int, default=32030)
    ap.add_argument("--load-timeout", type=int, default=1500)
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--max-total-tokens", type=int, default=65536)
    ap.add_argument("--do-accuracy", action="store_true")
    ap.add_argument("--do-qps", action="store_true")
    ap.add_argument("--qps-concurrency", type=int, default=8)
    ap.add_argument("--qps-requests", type=int, default=16)
    ap.add_argument("--out", default="/tmp/redknot_prefix_sweep.json")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]

    results = {}
    for length in lengths:
        results[str(length)] = {}
        # accuracy samples (per dataset)
        samples = {
            ds: [build_prompt(r, length) for r in load_rows(ds, args.n_samples)]
            for ds in datasets
        }
        # qps workload: one shared long prefix + distinct short questions
        base_row = load_rows(datasets[0], 1)[0]
        shared_prefix = (
            "Read the documents and answer the question concisely.\n\n"
            + tile_context(base_row["context"], length)
            + "\n\n"
        )
        questions = [
            f"Question: Briefly, what is fact #{k}?\nAnswer:"
            for k in range(args.qps_requests)
        ]

        print(f"\n{'#' * 70}\n#  PREFIX LENGTH = {length}\n{'#' * 70}", flush=True)

        if args.do_accuracy:
            results[str(length)]["acc_full"] = run_accuracy(
                "dsv4",
                None,
                length,
                samples,
                args.max_new,
                args.port,
                args.load_timeout,
                args.mem_frac,
                args.max_total_tokens,
                f"acc_full_{length}",
            )
            json.dump(results, open(args.out, "w"), indent=2)
            results[str(length)]["acc_compress"] = run_accuracy(
                "redknot_mla",
                COMPRESS_CFG,
                length,
                samples,
                args.max_new,
                args.port,
                args.load_timeout,
                args.mem_frac,
                args.max_total_tokens,
                f"acc_comp_{length}",
            )
            json.dump(results, open(args.out, "w"), indent=2)

        if args.do_qps:
            results[str(length)]["qps_recompute"] = run_qps(
                "recompute",
                length,
                shared_prefix,
                questions,
                args.max_new,
                args.qps_concurrency,
                args.port,
                args.load_timeout,
                args.mem_frac,
                args.max_total_tokens,
                f"qps_recomp_{length}",
            )
            json.dump(results, open(args.out, "w"), indent=2)
            results[str(length)]["qps_reuse"] = run_qps(
                "reuse",
                length,
                shared_prefix,
                questions,
                args.max_new,
                args.qps_concurrency,
                args.port,
                args.load_timeout,
                args.mem_frac,
                args.max_total_tokens,
                f"qps_reuse_{length}",
            )
            json.dump(results, open(args.out, "w"), indent=2)

    # ── report ──
    print("\n" + "=" * 78)
    print("PREFIX-LENGTH SWEEP SUMMARY")
    print("=" * 78)
    print("\n[ACCURACY] full vs compressed prefix")
    print(f"{'length':>8} | {'full F1':>8} {'comp F1':>8} {'retention':>9}")
    for length in lengths:
        r = results[str(length)]
        ff = (r.get("acc_full") or {}).get("f1_avg")
        cf = (r.get("acc_compress") or {}).get("f1_avg")
        ret = f"{100 * cf / ff:.0f}%" if (ff and cf) else "-"
        print(
            f"{length:>8} | {str(round(ff, 3) if ff else '-'):>8} "
            f"{str(round(cf, 3) if cf else '-'):>8} {ret:>9}"
        )

    print("\n[QPS @ fixed mem] lossless reuse vs recompute")
    print(
        f"{'length':>8} | {'recomp QPS':>11} {'reuse QPS':>10} {'speedup':>8} "
        f"| {'recomp lat':>11} {'reuse lat':>10}"
    )
    for length in lengths:
        r = results[str(length)]
        rc = r.get("qps_recompute") or {}
        ru = r.get("qps_reuse") or {}
        sp = f"{ru.get('qps', 0) / rc['qps']:.2f}x" if rc.get("qps") else "-"
        print(
            f"{length:>8} | {str(rc.get('qps', '-')):>11} {str(ru.get('qps', '-')):>10} "
            f"{sp:>8} | {str(rc.get('avg_lat_s', '-')):>11} {str(ru.get('avg_lat_s', '-')):>10}"
        )

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
