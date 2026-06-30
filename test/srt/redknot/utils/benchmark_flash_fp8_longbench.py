#!/usr/bin/env python3
"""LongBench QA evaluation for DeepSeek-V4-Flash-FP8 (instruct model).

Uses the DeepSeek instruct prompt format (<|User|>...<|Assistant|>) since
Flash is an instruction-tuned model (unlike Base). Assumes a server is already
running (use --port).

Usage::

    python benchmark_flash_fp8_longbench.py --port 31995 --n-samples 20 \
        --datasets hotpotqa 2wikimqa musique narrativeqa triviaqa \
        --target-tokens 4096 --output /tmp/flash_fp8_dense.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import Counter

import requests

LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)

# DeepSeek instruct special tokens
BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"


# ---------------------------------------------------------------------------
# Metrics
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


def best_f1(pred: str, golds) -> float:
    if isinstance(golds, str):
        golds = [golds]
    return max((f1_score(pred, g) for g in golds), default=0.0)


def best_em(pred: str, golds) -> float:
    if isinstance(golds, str):
        golds = [golds]
    return max((em_score(pred, g) for g in golds), default=0.0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(ds_name: str, n_samples: int, min_ctx_tokens: int, seed: int = 2026):
    path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("context") and row.get("answers"):
                if len(row["context"]) // 4 >= min_ctx_tokens:
                    rows.append(row)
    import random

    random.Random(seed).shuffle(rows)
    return rows[:n_samples]


def build_instruct_prompt(row: dict, target_tokens: int) -> tuple[str, list]:
    """Build a DeepSeek-instruct QA prompt with truncated context."""
    context = row["context"]
    question = row.get("input", "").strip()
    # Reserve ~200 tokens for question + instructions
    ctx_chars = max(target_tokens - 200, 256) * 4
    context = context[:ctx_chars]

    user_msg = (
        f"Read the following context and answer the question concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer with only the answer, no explanation."
    )
    prompt = f"{BOS}{USER}{user_msg}{ASSISTANT}"

    golds = row["answers"]
    if not isinstance(golds, list):
        golds = [str(golds)]
    return prompt, golds


def clean_answer(text: str) -> str:
    # Strip common instruct preambles and stop at newlines/markers
    text = text.strip()
    # Remove leading "The answer is", etc.
    text = re.split(r"(?i)\n\s*(?:question|context|note)\s*:", text)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate(port: int, prompt: str, max_new: int, timeout: int = 600):
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
    return obj.get("text", ""), float(obj["meta_info"].get("e2e_latency", 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31995)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "2wikimqa", "musique", "narrativeqa", "triviaqa"],
    )
    ap.add_argument("--output", type=str, default="/tmp/flash_fp8_dense.json")
    args = ap.parse_args()

    min_ctx = int(args.target_tokens * 0.5)
    all_results = {}
    all_f1, all_em = [], []

    for ds in args.datasets:
        print(f"\n{'=' * 60}\nDataset: {ds}\n{'=' * 60}", flush=True)
        rows = load_dataset(ds, args.n_samples, min_ctx)
        if not rows:
            print(f"  No samples for {ds}")
            continue

        ds_f1, ds_em, ds_lat = [], [], []
        for i, row in enumerate(rows):
            prompt, golds = build_instruct_prompt(row, args.target_tokens)
            try:
                text, lat = generate(args.port, prompt, args.max_new)
            except Exception as e:
                print(f"  #{i + 1} ERROR: {e}")
                continue
            ans = clean_answer(text)
            f1 = best_f1(ans, golds)
            em = best_em(ans, golds)
            ds_f1.append(f1)
            ds_em.append(em)
            ds_lat.append(lat)
            print(
                f"  #{i + 1:2d} F1={f1:.2f} EM={em:.0f} lat={lat:.1f}s "
                f"ans={ans[:45]!r} gold={str(golds[0])[:30]!r}",
                flush=True,
            )

        if ds_f1:
            n = len(ds_f1)
            res = {
                "n": n,
                "f1": sum(ds_f1) / n,
                "em": sum(ds_em) / n,
                "avg_latency": sum(ds_lat) / n,
            }
            all_results[ds] = res
            all_f1.extend(ds_f1)
            all_em.extend(ds_em)
            print(
                f"\n  {ds}: F1={res['f1']:.4f} EM={res['em']:.4f} "
                f"(n={n}, avg_lat={res['avg_latency']:.1f}s)"
            )

    # Overall
    if all_f1:
        overall = {
            "n": len(all_f1),
            "avg_f1": sum(all_f1) / len(all_f1),
            "avg_em": sum(all_em) / len(all_em),
        }
        print(f"\n{'=' * 60}")
        print(
            f"OVERALL: F1={overall['avg_f1']:.4f} EM={overall['avg_em']:.4f} "
            f"(n={overall['n']})"
        )
        print(f"{'=' * 60}")
        all_results["_overall"] = overall

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
