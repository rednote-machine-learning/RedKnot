#!/usr/bin/env python3
"""benchmark_RedKnot_Qwen35_sglang_TP.py — sglang-native tensor-parallel RAG
benchmark for Qwen3.5-35B-A3B.

Unlike the older HF-monkey-patch benchmark, this drives the **sglang Engine**
on its native TP path:

  * baseline : attention_backend=flashinfer  (standard full attention,
               hybrid GDN linear layers run natively)
  * RedKnot  : attention_backend=redknot     (head-classified KV sparsity on
               the 10 full-attention layers, driven by a head-config JSON)

Both run under tensor parallelism (tp_size), so per-layer weights are sharded
across GPUs (real TP), not the naive device_map="auto" pipeline split.

It loads LongBench QA datasets, pads each sample to a target context length by
concatenating contexts, then asks one question per sample and scores F1 / EM.
It prints per-dataset and overall averages for BOTH methods.

One-click:
  PYTHONPATH=python CUDA_VISIBLE_DEVICES=0,1 \
    REDKNOT_TP_SIZE=2 \
    .venv_tf5/bin/python test/srt/redknot/benchmark_RedKnot_Qwen35_sglang_TP.py
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

# ── config ──
MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
# Only datasets that carry input/context/answers are usable by this QA harness.
DEFAULT_DATASETS = (
    "triviaqa,hotpotqa,2wikimqa,multifieldqa_en,multifieldqa_zh,musique,"
    "narrativeqa,qasper,dureader,passage_retrieval_en,passage_retrieval_zh,"
    "lsht,qmsum,samsum,trec,repobench-p"
)
DATASETS = [
    x.strip()
    for x in os.environ.get("REDKNOT_DATASETS", DEFAULT_DATASETS).split(",")
    if x.strip()
]
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "50"))
MAX_NEW_TOKENS = int(os.environ.get("REDKNOT_MAX_NEW", "32"))
TARGET_TOKENS = int(os.environ.get("REDKNOT_TARGET_TOKENS", "32000"))
CHUNK_TOKENS = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "8000"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
TP_SIZE = int(os.environ.get("REDKNOT_TP_SIZE", "2"))
MEM_FRACTION_STATIC = os.environ.get("REDKNOT_MEM_FRACTION_STATIC", "0.85")
# Large FP8 MoE (e.g. 397B) needs a longer init watchdog (DeepGEMM warmup) and
# optionally skipping the server warmup pass.
WATCHDOG_TIMEOUT = os.environ.get("REDKNOT_WATCHDOG_TIMEOUT", "")
SKIP_WARMUP = os.environ.get("REDKNOT_SKIP_WARMUP", "0") == "1"
HEAD_CFG = os.environ.get(
    "REDKNOT_HEAD_CFG",
    str(
        Path(__file__).resolve().parent
        / "head_class"
        / "qwen3.5-35B-A3B_redknot_server.json"
    ),
)
BASELINE_BACKEND = os.environ.get("REDKNOT_BASELINE_BACKEND", "flashinfer")
METHODS = [
    m.strip()
    for m in os.environ.get("REDKNOT_METHODS", "baseline,redknot").split(",")
    if m.strip()
]


# ── version-assert / numa bypass for this dev env ──
def _patch_env():
    import sglang.srt.utils.common as _c

    _orig = _c.assert_pkg_version

    def _lenient(pkg, min_version, message):
        try:
            return _orig(pkg, min_version, message)
        except Exception as e:  # noqa: BLE001
            print(f"[bench] bypass version assert {pkg!r}: {e}")
            return None

    _c.assert_pkg_version = _lenient
    try:
        import sglang.srt.entrypoints.engine as _eng

        _eng.assert_pkg_version = _lenient
    except Exception:
        pass


# ── metrics ──
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
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_max(pred: str, golds: Iterable[str]) -> float:
    return max((f1_score(pred, g) for g in golds), default=0.0)


def em_max(pred: str, golds: Iterable[str]) -> float:
    return max((float(_normalize(pred) == _normalize(g)) for g in golds), default=0.0)


def _short_ans(text: str) -> str:
    text = text or ""
    if not text.strip():
        return ""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"(?i)^\s*(the answer is|answer)\b[:\s]*", "", text, count=1)
    text = re.split(r"(?i)(?:\n\s*question\s*:|\n\s*q\s*:|<\|)", text)[0]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cand = (lines[0] if lines else text.strip()).strip().strip('"').strip("'")
    return re.sub(r"\s*[.。]\s*$", "", cand)


def _trunc(s: str, n: int = 72) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _query_text(question: str) -> str:
    return (
        "\n\nAnswer the question using only the documents above. "
        "Return the shortest exact answer span only, with no explanation.\n"
        f"Question: {question}\nAnswer:"
    )


# ── data loading ──
def _load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_longbench_padded(ds_name: str, tok, n_samples: int, target_tokens: int):
    path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
    if not os.path.exists(path):
        return []
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("input") and row.get("context") and row.get("answers"):
                raw.append(row)
    if not raw:
        return []
    rng = random.Random(SEED)
    rng.shuffle(raw)

    out = []
    n = len(raw)
    for i, base in enumerate(raw):
        if len(out) >= n_samples:
            break
        ctx_ids = tok(base["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % n
        while len(ctx_ids) < target_tokens and j != i:
            ctx_ids.extend(
                tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            )
            j = (j + 1) % n
        ctx_ids = ctx_ids[:target_tokens]
        if len(ctx_ids) < target_tokens:
            continue
        docs = []
        for start in range(0, len(ctx_ids), CHUNK_TOKENS):
            piece = ctx_ids[start : start + CHUNK_TOKENS]
            if len(piece) < 64:
                break
            docs.append(tok.decode(piece, skip_special_tokens=True))
        if len(docs) < 2:
            continue
        out.append(
            {
                "question": base["input"],
                "golds": [str(x) for x in base["answers"]],
                "docs": docs,
            }
        )
    return out


# ── engine ──
def _engine_kwargs(method: str):
    backend = BASELINE_BACKEND if method == "baseline" else "redknot"
    kwargs = dict(
        model_path=MODEL,
        attention_backend=backend,
        tp_size=TP_SIZE,
        random_seed=SEED,
        trust_remote_code=True,
        enable_multimodal=False,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        log_level="error",
    )
    if MEM_FRACTION_STATIC:
        kwargs["mem_fraction_static"] = float(MEM_FRACTION_STATIC)
    if WATCHDOG_TIMEOUT:
        kwargs["watchdog_timeout"] = float(WATCHDOG_TIMEOUT)
    if SKIP_WARMUP:
        kwargs["skip_server_warmup"] = True
    if method == "redknot" and HEAD_CFG and os.path.exists(HEAD_CFG):
        kwargs["redknot_head_config_path"] = HEAD_CFG
    return kwargs


def _run_method(method: str, tasks):
    """Bring up one engine for `method`, generate for all tasks, tear down.

    Returns a list of (text, ttft_seconds) per task. TTFT is measured with
    streaming: wall-clock time from submitting the request to receiving the
    first decoded token (i.e. the prefill / first-token latency, which is
    exactly what RedKnot sparse attention is meant to reduce on long context).
    """
    import sglang as sgl

    print(f"\n[bench] launching engine for method={method} ...")
    t0 = time.perf_counter()
    engine = sgl.Engine(**_engine_kwargs(method))
    print(f"[bench] engine ({method}) up in {time.perf_counter() - t0:.1f}s")
    sampling = {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS}
    results = []
    try:
        for label, sample, prompt in tasks:
            tstart = time.perf_counter()
            ttft = None
            text = ""
            stream = engine.generate(prompt, sampling, stream=True)
            for chunk in stream:
                if ttft is None:
                    # first chunk carrying any generated text == first token
                    piece = chunk["text"] if isinstance(chunk, dict) else str(chunk)
                    if piece:
                        ttft = time.perf_counter() - tstart
                text = chunk["text"] if isinstance(chunk, dict) else str(chunk)
            if ttft is None:
                ttft = time.perf_counter() - tstart
            results.append((text, ttft))
    finally:
        engine.shutdown()
    return results


# ── main ──
def main():
    _patch_env()
    tok = _load_tokenizer()

    # Build the task list once (identical prompts across methods).
    tasks = []  # (dataset_label, sample, prompt)
    per_ds_counts = {}
    for ds_name in DATASETS:
        samples = _load_longbench_padded(ds_name, tok, N_SAMPLES, TARGET_TOKENS)
        per_ds_counts[ds_name] = len(samples)
        if not samples:
            print(f"[bench] skip {ds_name}: no usable samples")
            continue
        for sample in samples:
            prompt = "\n\n".join(sample["docs"]) + _query_text(sample["question"])
            tasks.append((ds_name, sample, prompt))

    W = 110
    print("=" * W)
    print(" Qwen3.5 RedKnot RAG Benchmark (sglang-native tensor parallel)")
    print(f" model={MODEL}")
    print(
        f" tp={TP_SIZE} ctx={TARGET_TOKENS} chunk={CHUNK_TOKENS} "
        f"n/dataset={N_SAMPLES} max_new={MAX_NEW_TOKENS} methods={METHODS}"
    )
    print(f" head_cfg={HEAD_CFG}")
    print(f" total tasks={len(tasks)} across {len(per_ds_counts)} datasets")
    print("=" * W)

    if not tasks:
        print("[bench] no tasks; aborting")
        return

    # Run each method in its own engine (different attention backend).
    method_outputs = {}
    for method in METHODS:
        method_outputs[method] = _run_method(method, tasks)

    # ── score ──
    # Accumulators: per (dataset, method) -> [f1_sum, em_sum, ttft_sum, count]
    agg = {}
    for idx, (ds_name, sample, _prompt) in enumerate(tasks):
        for method in METHODS:
            text, ttft = method_outputs[method][idx]
            ans = _short_ans(text)
            f1 = f1_max(ans, sample["golds"])
            em = em_max(ans, sample["golds"])
            key = (ds_name, method)
            a = agg.setdefault(key, [0.0, 0.0, 0.0, 0])
            a[0] += f1
            a[1] += em
            a[2] += ttft
            a[3] += 1

    # Per-dataset table
    print("\n" + "=" * W)
    print(" Per-dataset results (F1 / EM / TTFT[s])")
    print("=" * W)
    header = f" {'dataset':20} {'n':>4}"
    for m in METHODS:
        header += f" {m + '_F1':>11} {m + '_EM':>9} {m + '_TTFT':>11}"
    print(header)
    print("-" * W)

    overall = {m: [0.0, 0.0, 0.0, 0] for m in METHODS}
    for ds_name in DATASETS:
        if (ds_name, METHODS[0]) not in agg:
            continue
        n = agg[(ds_name, METHODS[0])][3]
        row = f" {ds_name:20} {n:>4}"
        for m in METHODS:
            f1s, ems, tts, c = agg[(ds_name, m)]
            row += f" {f1s / c:11.3f} {ems / c:9.3f} {tts / c:11.3f}"
            overall[m][0] += f1s
            overall[m][1] += ems
            overall[m][2] += tts
            overall[m][3] += c
        print(row)

    print("-" * W)
    avg_row = f" {'AVERAGE (micro)':20} {overall[METHODS[0]][3]:>4}"
    for m in METHODS:
        f1s, ems, tts, c = overall[m]
        avg_row += (
            f" {f1s / max(c, 1):11.3f} {ems / max(c, 1):9.3f} {tts / max(c, 1):11.3f}"
        )
    print(avg_row)
    print("=" * W)

    # TTFT speedup line (redknot vs baseline), if both present
    if "baseline" in METHODS and "redknot" in METHODS:
        b_tt = overall["baseline"][2] / max(overall["baseline"][3], 1)
        r_tt = overall["redknot"][2] / max(overall["redknot"][3], 1)
        if r_tt > 0:
            print(
                f" TTFT: baseline={b_tt:.3f}s  redknot={r_tt:.3f}s  "
                f"speedup={b_tt / r_tt:.2f}x"
            )
            print("=" * W)

    # Machine-parseable summary lines
    for m in METHODS:
        f1s, ems, tts, c = overall[m]
        print(
            f"SUMMARY method={m} datasets={len(per_ds_counts)} n={c} "
            f"avg_f1={f1s / max(c, 1):.4f} avg_em={ems / max(c, 1):.4f} "
            f"avg_ttft={tts / max(c, 1):.4f}"
        )


if __name__ == "__main__":
    main()
