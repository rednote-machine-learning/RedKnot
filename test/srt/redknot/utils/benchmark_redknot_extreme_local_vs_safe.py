#!/usr/bin/env python3
"""RedKnot DeepSeek-V4 experiment: does the "store only last 128 tokens"
(extreme local-head) assumption hurt accuracy?

Three modes share the SAME server config except attention policy:

  A. dense    : --attention-backend dsv4              (full reference)
  B. extreme  : redknot_mla + head-config where ~98% heads are local(w=128)
                => prefix KV is effectively unused by almost every head
                   (the strong "only last 128 tokens" hypothesis)
  C. safe     : redknot_mla + head-config stride-8 global + 2 dense-prefix layers
                => the realistic RedKnot policy

Metric: LongBench QA F1 / EM on the same samples.

Run::

    python benchmark_redknot_extreme_local_vs_safe.py \
        --modes dense,extreme,safe --n-samples 8 --port 31997
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
HEAD_CLASS_DIR = Path(__file__).resolve().parent / "head_class"
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

DATASETS = ["hotpotqa", "2wikimqa", "musique"]

MODE_HEAD_CFG = {
    "dense": None,  # uses dsv4 backend, no head cfg
    "extreme": str(HEAD_CLASS_DIR / "dsv4_extreme_local128.json"),
    "safe": str(HEAD_CLASS_DIR / "dsv4_safe_stride8.json"),
}


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_dataset(
    ds_name: str, n_samples: int, min_context_tokens: int, seed: int = 2026
):
    path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("context") and row.get("answers"):
                if len(row["context"]) // 4 >= min_context_tokens:
                    rows.append(row)
    import random

    random.Random(seed).shuffle(rows)
    return rows[:n_samples]


def build_prompt(row: dict, n_blocks: int, max_tokens_per_block: int):
    context = row["context"]
    question = row.get("input", "").strip()
    target_chars = max_tokens_per_block * 4
    blocks, remaining = [], context
    for i in range(n_blocks):
        blocks.append(remaining[:target_chars])
        remaining = remaining[target_chars:]
    prompt = (
        "\n\n".join(f"Block {i + 1}:\n{b}" for i, b in enumerate(blocks))
        + f"\n\nQuestion: {question}\nAnswer:"
    )
    golds = row["answers"]
    gold = str(golds[0]) if isinstance(golds, list) and golds else str(golds)
    return prompt, gold


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
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


def start_server(mode: str, port: int, rank_log_dir: str, server_log: str):
    os.makedirs(rank_log_dir, exist_ok=True)
    env = _server_env(rank_log_dir)
    backend = "dsv4" if mode == "dense" else "redknot_mla"
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--attention-backend",
        backend,
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
    head_cfg = MODE_HEAD_CFG[mode]
    if head_cfg:
        cmd += ["--redknot-head-config-path", head_cfg]
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
            if "EXIT=" in txt or "SIGQUIT received" in txt or "Traceback" in txt:
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


def generate(port: int, prompt: str, max_new: int, timeout: int = 900):
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
    }
    r = requests.post(
        url, json=payload, timeout=timeout, headers={"Connection": "close"}
    )
    r.raise_for_status()
    obj = r.json()
    return obj.get("text", "")


def parse_head_policy(server_log: str):
    if not os.path.exists(server_log):
        return None
    for line in Path(server_log).read_text(errors="ignore").splitlines():
        m = re.search(r"RedKnot MLA policy loaded: (\{[^}]+\})", line)
        if m:
            try:
                import ast

                return ast.literal_eval(m.group(1))
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_mode(mode: str, all_samples, args):
    port = args.port
    rank_log_dir = f"/tmp/extreme_{mode}_ranklogs"
    server_log = f"/tmp/extreme_{mode}_server.log"
    subprocess.run(f"rm -rf {rank_log_dir} {server_log}", shell=True)

    print(f"\n{'=' * 64}")
    print(
        f"  MODE: {mode}  backend={'dsv4' if mode == 'dense' else 'redknot_mla'}"
        f"  head_cfg={MODE_HEAD_CFG[mode]}"
    )
    print(f"{'=' * 64}", flush=True)

    proc, lf = start_server(mode, port, rank_log_dir, server_log)
    try:
        if not wait_ready(port, server_log, args.load_timeout):
            print(f"[{mode}] server failed; tail of log:")
            if os.path.exists(server_log):
                print(
                    "\n".join(
                        Path(server_log).read_text(errors="ignore").splitlines()[-30:]
                    )
                )
            return None
        # Warmup with retries: under PP=8 the first request after "ready" can
        # race with the scheduler loop and hang. Use short per-attempt timeouts
        # and retry until one succeeds (this reliably wakes the loop).
        warmed = False
        for attempt in range(8):
            try:
                generate(port, "Hello. Answer:", 3, timeout=90)
                warmed = True
                print(f"[{mode}] warmup ok (attempt {attempt + 1})", flush=True)
                break
            except Exception as e:
                print(f"[{mode}] warmup attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(2)
        if not warmed:
            print(
                f"[{mode}] WARNING: warmup never succeeded; proceeding anyway",
                flush=True,
            )

        results = {}
        for ds_name, samples in all_samples.items():
            print(f"\n  [{mode}] {ds_name} ({len(samples)} samples)", flush=True)
            rows = []
            for i, s in enumerate(samples):
                try:
                    text = generate(port, s["prompt"], args.max_new, timeout=400)
                except Exception as e:
                    print(f"    #{i} error: {e}")
                    continue
                ans = _clean_answer(text)
                f1 = f1_score(ans, s["gold"])
                em = em_score(ans, s["gold"])
                rows.append({"f1": f1, "em": em})
                print(
                    f"    #{i + 1:2d} F1={f1:.2f} EM={em:.0f}  ans={ans[:40]!r}  gold={s['gold'][:30]!r}",
                    flush=True,
                )
            if rows:
                n = len(rows)
                results[ds_name] = {
                    "f1": sum(r["f1"] for r in rows) / n,
                    "em": sum(r["em"] for r in rows) / n,
                    "n": n,
                }
        head_policy = parse_head_policy(server_log)
        return {"mode": mode, "datasets": results, "head_policy": head_policy}
    finally:
        stop_server(proc, lf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str, default="dense,extreme,safe")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--port", type=int, default=31997)
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument(
        "--seq-len",
        type=int,
        default=4000,
        help="total context tokens; split into n-blocks",
    )
    ap.add_argument("--datasets", type=str, default=None)
    ap.add_argument("--out", type=str, default="/tmp/redknot_extreme_local_exp.json")
    args = ap.parse_args()

    datasets = (
        [d.strip() for d in args.datasets.split(",")] if args.datasets else DATASETS
    )
    modes = [m.strip() for m in args.modes.split(",")]
    max_tpb = args.seq_len // args.n_blocks
    min_ctx = int(args.seq_len * 0.9)

    all_samples = {}
    for ds in datasets:
        rows = load_dataset(ds, args.n_samples, min_ctx)
        if not rows:
            print(f"  SKIP {ds}: no rows >= {min_ctx} tokens")
            continue
        all_samples[ds] = [
            {"prompt": p, "gold": g}
            for row in rows
            for p, g in [build_prompt(row, args.n_blocks, max_tpb)]
        ]
        print(f"  Loaded {ds}: {len(all_samples[ds])} samples (seq_len~{args.seq_len})")

    all_results = {}
    for mode in modes:
        all_results[mode] = run_mode(mode, all_samples, args)

    # ---- Report ----
    print("\n" + "=" * 72)
    print(f"RedKnot DSV4: extreme-local(only last 128) vs safe vs dense")
    print(f"seq_len~{args.seq_len}  n_blocks={args.n_blocks}  datasets={datasets}")
    print("=" * 72)
    for mode in modes:
        r = all_results.get(mode)
        if r:
            print(f"  [{mode}] head_policy = {r['head_policy']}")

    print(f"\n  {'dataset':<14}" + "".join(f"{m:>16}" for m in modes))
    for ds in datasets:
        line = f"  {ds:<14}"
        for mode in modes:
            r = all_results.get(mode)
            if r and ds in r["datasets"]:
                d = r["datasets"][ds]
                cell = "F1=%.3f" % d["f1"]
                line += f"{cell:>16}"
            else:
                line += f"{'-':>16}"
        print(line)
    # averages
    line = f"  {'AVG':<14}"
    for mode in modes:
        r = all_results.get(mode)
        if r and r["datasets"]:
            avg = sum(d["f1"] for d in r["datasets"].values()) / len(r["datasets"])
            cell = "F1=%.3f" % avg
            line += f"{cell:>16}"
        else:
            line += f"{'-':>16}"
        print(line)
    # averages
    line = f"  {'AVG':<14}"
    for mode in modes:
        r = all_results.get(mode)
        if r and r["datasets"]:
            avg = sum(d["f1"] for d in r["datasets"].values()) / len(r["datasets"])
            line += f"{f'F1={avg:.3f}':>16}"
        else:
            line += f"{'-':>16}"
    print(line)

    json.dump(
        {
            "config": {
                "seq_len": args.seq_len,
                "n_blocks": args.n_blocks,
                "datasets": datasets,
                "modes": modes,
                "n_samples": args.n_samples,
            },
            "results": all_results,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
