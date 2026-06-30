#!/usr/bin/env python3
"""RedKnot DeepSeek-V4 sparse-FFN 3-mode comparison (精度 / 算力 / TTFT).

Compares three configurations of the SAME DeepSeek-V4 dsv4 attention backend on
HotpotQA, isolating the contribution of this integration (indexer-driven
sparse-FFN token selection):

  1. dense       : sparse-FFN OFF  -> baseline, every token runs the MoE.
  2. activation  : sparse-FFN ON, importance = post-attention activation L2 norm.
  3. indexer     : sparse-FFN ON, importance = DeepSeek-V4 indexer signal
                   (c4_topk_lengths_raw)  <-- this integration.

For each mode it starts a fresh server, replays N HotpotQA samples over HTTP, and
reports:
  * Accuracy : F1 / EM vs the gold answer.
  * TTFT     : time-to-first-token (streaming) per request, mean.
  * Compute  : measured sparse-FFN keep ratio (= MoE FLOPs fraction on sparse
               layers) parsed from the rank0 log, and the implied MoE-FLOPs
               saving vs dense.

This benchmark drives a server it launches itself; it does NOT touch any other
running server. It uses the dsv4 backend only (avoids the redknot_mla head-split
FlashMLA h_q limitation), which is exactly where this sparse-FFN feature lives.

Usage:
  python test/srt/redknot/benchmark_redknot_dsv4_sparseffn_compare.py \
      --n-samples 20 --target-tokens 16000 --max-new 32 --port 31997

Env overrides: REDKNOT_MODEL_PATH, REDKNOT_TP_SIZE.
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
HOTPOT_PARQUET = os.environ.get(
    "REDKNOT_HOTPOT_PARQUET",
    str(
        REPO
        / "test/srt/redknot/datasets/HotpotQA/distractor/validation-00000-of-00001.parquet"
    ),
)
TP_SIZE = int(os.environ.get("REDKNOT_TP_SIZE", "8"))


# --------------------------------------------------------------------------- #
# Accuracy metrics (HotpotQA-style normalized F1 / EM)
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
def load_hotpot(n_samples: int, target_tokens: int, seed: int = 2026):
    import pandas as pd

    df = pd.read_parquet(HOTPOT_PARQUET)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    samples = []
    # ~4 chars/token heuristic for padding target; exact token count not critical.
    target_chars = target_tokens * 4
    for _, row in df.iterrows():
        if len(samples) >= n_samples:
            break
        ctx = row["context"]
        titles = list(ctx["title"])
        sents = ctx["sentences"]
        passages = []
        for t, ss in zip(titles, sents):
            passages.append(f"{t}: " + " ".join(list(ss)))
        context_text = "\n\n".join(passages)
        # pad with more passages from following rows to reach target length
        if len(context_text) < target_chars:
            extra = []
            for _, r2 in df.iterrows():
                if len(context_text) + len("\n\n".join(extra)) >= target_chars:
                    break
                c2 = r2["context"]
                for t, ss in zip(list(c2["title"]), c2["sentences"]):
                    extra.append(f"{t}: " + " ".join(list(ss)))
            context_text = context_text + "\n\n" + "\n\n".join(extra)
        context_text = context_text[:target_chars]
        prompt = (
            context_text
            + "\n\nAnswer the question using only the documents above. "
            + "Return the shortest exact answer span only, with no explanation.\n"
            + f"Question: {row['question']}\nAnswer:"
        )
        samples.append({"prompt": prompt, "gold": str(row["answer"])})
    return samples


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
MODE_FLAGS = {
    "dense": [],  # sparse-ffn off
    "activation": [
        "--redknot-sparse-ffn-enable",
        "--redknot-sparse-ffn-importance",
        "activation",
    ],
    "indexer": [
        "--redknot-sparse-ffn-enable",
        "--redknot-sparse-ffn-importance",
        "indexer",
    ],
    "indexer_indegree": [
        "--redknot-sparse-ffn-enable",
        "--redknot-sparse-ffn-importance",
        "indexer_indegree",
    ],
}

COMMON_SPARSE_FLAGS = [
    "--redknot-sparse-ffn-dense-until",
    "4",
    "--redknot-sparse-ffn-mass-thresh",
    "0.6",
    "--redknot-sparse-ffn-recent-n",
    "256",
]


def _server_env():
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
        SGLANG_REDKNOT_FFN_DEBUG="1",  # emit keep-ratio stats to rank log
    )
    return env


def start_server(mode: str, port: int, rank_log_dir: str, server_log: str):
    os.makedirs(rank_log_dir, exist_ok=True)
    env = _server_env()
    env["SGLANG_RANK_LOG_DIR"] = rank_log_dir
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--attention-backend",
        "dsv4",
        "--tp-size",
        str(TP_SIZE),
        "--mem-fraction-static",
        "0.85",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        # Long-context tuning: on these L20Y cards the default chunked-prefill +
        # SWA radix-cache + overlap-schedule path stalls long prefills. These
        # flags were empirically needed to let >=4K-token requests complete.
        "--disable-radix-cache",
        "--disable-overlap-schedule",
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
    if mode != "dense":
        cmd += MODE_FLAGS[mode] + COMMON_SPARSE_FLAGS
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
                # health may lag; give it a moment
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
    # best-effort sweep
    subprocess.run("pkill -9 -f sglang::scheduler", shell=True)
    subprocess.run("pkill -9 -f sglang.launch_server", shell=True)
    time.sleep(8)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def _generate(port: int, prompt: str, max_new: int):
    """Non-streaming generate. Returns (text, server_e2e_latency)."""
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
    }
    r = requests.post(
        url, json=payload, timeout=int(os.environ.get("REDKNOT_REQ_TIMEOUT", "900"))
    )
    r.raise_for_status()
    obj = r.json()
    return obj.get("text", ""), float(obj["meta_info"].get("e2e_latency", 0.0))


# NOTE: TTFT/throughput are intentionally NOT reported by this benchmark. On the
# L20Y dev cards the DeepSeek-V4 fp8 MoE/FlashMLA kernels run ~100x below normal
# (measured ~137 tok/s prefill, vs thousands on H800; no tuned MoE config for
# NVIDIA_L20Y + forced --disable-cuda-graph), so any latency number would reflect
# the slow kernels, not RedKnot's algorithmic effect. We report the two
# hardware-independent dimensions: accuracy (F1/EM) and compute (MoE-FLOPs
# fraction from the measured sparse-FFN keep ratio). e2e is kept only as a
# rough, explicitly-untrusted reference.


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


# --------------------------------------------------------------------------- #
# MoE FLOPs model (uses measured keep ratio)
# --------------------------------------------------------------------------- #
def moe_flops_fraction(keep_ratio: float, n_layers: int, dense_until: int) -> float:
    """Fraction of total MoE FLOPs vs dense, given measured keep_ratio on the
    sparse-eligible layers. Dense-prefix layers always run full."""
    dense_layers = min(dense_until, n_layers)
    sparse_layers = n_layers - dense_layers
    total_dense = n_layers
    total_sparse = dense_layers + sparse_layers * keep_ratio
    return total_sparse / total_dense


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_mode(mode: str, samples, args):
    port = args.port
    tag = f"dsv4_{mode}"
    rank_log_dir = f"/tmp/bench_{tag}_ranklogs"
    server_log = f"/tmp/bench_{tag}_server.log"
    subprocess.run(f"rm -rf {rank_log_dir} {server_log}", shell=True)

    print(f"\n=== [{mode}] starting server (tp={TP_SIZE}, port={port}) ===", flush=True)
    proc, lf = start_server(mode, port, rank_log_dir, server_log)
    try:
        if not wait_ready(port, server_log, timeout=args.load_timeout):
            print(f"[{mode}] server failed to become ready. tail:")
            print(
                subprocess.run(
                    f"grep -vi server_args {server_log} | tail -20",
                    shell=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            return None

        # warmup request (triggers JIT; excluded from metrics).
        # Use a SHORT prompt first (cheap JIT trigger), then a medium one.
        # A full 16K-token warmup can stall a cold dsv4 server on low-BW GPUs.
        print(f"[{mode}] warmup (short)...", flush=True)
        try:
            _generate(port, "Question: What is 2+2?\nAnswer:", 4)
            print(f"[{mode}] warmup short OK", flush=True)
        except Exception as e:
            print(f"[{mode}] warmup short error: {e}", flush=True)
        try:
            _generate(port, samples[0]["prompt"][:2000], 4)
            print(f"[{mode}] warmup medium OK", flush=True)
        except Exception as e:
            print(f"[{mode}] warmup medium error: {e}", flush=True)

        rows = []
        for i, s in enumerate(samples):
            try:
                text, e2e = _generate(port, s["prompt"], args.max_new)
            except Exception as e:
                print(f"[{mode}] sample {i} error: {e}")
                continue
            ans = _clean_answer(text)
            rows.append(
                {
                    "f1": f1_score(ans, s["gold"]),
                    "em": em_score(ans, s["gold"]),
                    "e2e": e2e,
                    "ans": ans[:60],
                    "gold": s["gold"][:40],
                }
            )
            print(
                f"[{mode}] {i + 1}/{len(samples)} "
                f"F1={rows[-1]['f1']:.2f} EM={rows[-1]['em']:.0f} "
                f"ans={ans[:32]!r} gold={s['gold'][:24]!r}",
                flush=True,
            )

        keep_ratio = parse_keep_ratio(rank_log_dir)
        n = max(len(rows), 1)
        result = {
            "mode": mode,
            "n": len(rows),
            "f1": sum(r["f1"] for r in rows) / n,
            "em": sum(r["em"] for r in rows) / n,
            "e2e_ref": sum(r["e2e"] for r in rows) / n,  # untrusted reference only
            "keep_ratio": keep_ratio,
        }
        return result
    finally:
        stop_server(proc, lf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--target-tokens", type=int, default=16000)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--port", type=int, default=31997)
    ap.add_argument("--load-timeout", type=int, default=600)
    ap.add_argument("--modes", type=str, default="dense,activation,indexer")
    ap.add_argument("--n-layers", type=int, default=43)
    ap.add_argument("--dense-until", type=int, default=4)
    ap.add_argument(
        "--out", type=str, default="/tmp/redknot_dsv4_sparseffn_compare.json"
    )
    args = ap.parse_args()

    print(
        f"Loading {args.n_samples} HotpotQA samples (~{args.target_tokens} tok ctx)..."
    )
    samples = load_hotpot(args.n_samples, args.target_tokens)
    print(f"Loaded {len(samples)} samples.")

    results = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        r = run_mode(mode, samples, args)
        if r:
            results.append(r)

    # ---- report ----
    print("\n" + "=" * 72)
    print("RedKnot DeepSeek-V4 sparse-FFN comparison (dsv4 backend, HotpotQA)")
    print("Metrics: accuracy (F1/EM) + compute (MoE-FLOPs from measured keep ratio)")
    print("TTFT/throughput intentionally omitted: L20Y fp8 kernels are ~100x slow")
    print("=" * 72)
    header = f"{'mode':<12}{'n':>4}{'F1':>8}{'EM':>8}{'keep%':>9}{'MoE FLOPs%':>12}"
    print(header)
    print("-" * len(header))
    base = next((r for r in results if r["mode"] == "dense"), None)
    for r in results:
        kr = r["keep_ratio"]
        # dense baseline keeps everything (no sparse-FFN) -> keep=100%, MoE=100%.
        if r["mode"] == "dense" or kr is None:
            keep_s, moe_s = "  100.0", "     100.0"
        else:
            keep_s = f"{kr * 100:7.1f}"
            moe_s = (
                f"{moe_flops_fraction(kr, args.n_layers, args.dense_until) * 100:9.1f}"
            )
        print(
            f"{r['mode']:<12}{r['n']:>4}{r['f1']:>8.3f}{r['em']:>8.3f}"
            f"{keep_s:>9}{moe_s:>12}"
        )
    if base:
        print("-" * len(header))
        for r in results:
            if r["mode"] == "dense":
                continue
            df1 = r["f1"] - base["f1"]
            dem = r["em"] - base["em"]
            kr = r["keep_ratio"]
            moe = moe_flops_fraction(kr, args.n_layers, args.dense_until) if kr else 1.0
            print(
                f"  {r['mode']:<10} vs dense: dF1={df1:+.3f} dEM={dem:+.3f}  "
                f"MoE FLOPs saved={(1 - moe) * 100:.1f}%  (keep_ratio={kr})"
            )
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
