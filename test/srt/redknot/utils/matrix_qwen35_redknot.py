#!/usr/bin/env python3
"""Accuracy matrix: RedKnot Step-1 vs native, across small Qwen3.5 models x datasets.

Uses the small bf16 Qwen3.5 models (same hybrid linear/full architecture +
Q-gating + head_dim 256 as the 397B, but no FP8 and fast to load) to validate
RedKnot head-class attention accuracy across multiple models and LongBench QA
datasets. The 397B FP8 model is excluded here (its FP8 projection path needs a
separate fix; including it would report misleading F1=0).

For each (model, dataset): run N random samples, compare native full-recompute
vs RedKnot (full_attention layers only, sweet-spot frac_global) — report F1/EM
for both and the next-token agreement.

Run:
  HF_ENDPOINT=https://huggingface.co \
    PYTHONPATH=python:.venv_tf5/...:<sys-sp> CUDA_VISIBLE_DEVICES=0 \
    .venv_tf5/bin/python test/srt/redknot/matrix_qwen35_redknot.py
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODELS = os.environ.get(
    "REDKNOT_MODELS",
    # ~30B-class bf16 models (no FP8): dense 27B + MoE 35B-A3B (same
    # qwen3_5_moe architecture as the 397B). Both hybrid linear/full attention.
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-27B,"
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
).split(",")
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa,multifieldqa_en,qasper"
).split(",")
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "4"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
MAX_CTX = int(os.environ.get("REDKNOT_MAX_CTX", "8000"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
FRAC_GLOBAL = float(os.environ.get("REDKNOT_FRAC_GLOBAL", "0.10"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "4096"))


def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, golds):
    best = 0.0
    for g in golds:
        p, gg = _norm(pred).split(), _norm(g).split()
        if not p or not gg:
            best = max(best, float(p == gg))
            continue
        common = Counter(p) & Counter(gg)
        ns = sum(common.values())
        if ns == 0:
            continue
        prec, rec = ns / len(p), ns / len(gg)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def em(pred, golds):
    return max((float(_norm(pred) == _norm(g)) for g in golds), default=0.0)


def short(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    cand = (lines[0] if lines else t).strip().strip('"').strip("'")
    return re.sub(r"\s*[.。]\s*$", "", cand)


def load_ds(name, tok, n, seed):
    path = os.path.join(LONGBENCH_DIR, f"{name}.jsonl")
    raw = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(seed).shuffle(raw)
    out = []
    for r in raw[: n * 3]:
        ids = tok(r["context"], add_special_tokens=False)["input_ids"][:MAX_CTX]
        docs = [
            tok.decode(ids[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, len(ids), CHUNK)
        ]
        out.append({"q": r["input"], "golds": r["answers"], "docs": docs})
        if len(out) >= n:
            break
    return out


QPROMPT = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


@torch.no_grad()
def gen_native(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    out = model.generate(
        ids,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        build_full_attention_head_config,
        run_redknot_qwen35,
    )

    W = 96
    print("=" * W)
    print(" ACCURACY MATRIX: RedKnot Step-1 vs native (small Qwen3.5, bf16)")
    print(f" models={[Path(m).name for m in MODELS]}")
    print(
        f" datasets={DATASETS} | N={N_SAMPLES} frac_global={FRAC_GLOBAL} window={WINDOW}"
    )
    print("=" * W)

    results = []
    for mpath in MODELS:
        mname = Path(mpath).name
        tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            mpath, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
        ).eval()
        head_cfg = build_full_attention_head_config(
            model.config, frac_global=FRAC_GLOBAL, local_window=WINDOW
        )

        for ds in DATASETS:
            try:
                samples = load_ds(ds, tok, N_SAMPLES, SEED)
            except FileNotFoundError:
                print(f" [skip] {mname}/{ds}: not found")
                continue
            nb_f1 = nb_em = rk_f1 = rk_em = 0.0
            for s in samples:
                qt = QPROMPT.format(q=s["q"])
                ctx = "\n\n".join(s["docs"])
                nb = short(gen_native(model, tok, ctx + qt))
                rk_text, _ = run_redknot_qwen35(
                    model,
                    tok,
                    context_text=ctx,
                    query_text=qt,
                    head_cfg=head_cfg,
                    max_new_tokens=MAX_NEW,
                )
                rk = short(rk_text)
                nb_f1 += f1(nb, s["golds"])
                nb_em += em(nb, s["golds"])
                rk_f1 += f1(rk, s["golds"])
                rk_em += em(rk, s["golds"])
            n = len(samples)
            row = (mname, ds, n, nb_f1 / n, nb_em / n, rk_f1 / n, rk_em / n)
            results.append(row)
            print(
                f" {mname:20} {ds:16} N={n} | native F1={row[3]:.3f} EM={row[4]:.3f} "
                f"| RedKnot F1={row[5]:.3f} EM={row[6]:.3f}"
            )
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * W)
    print(" SUMMARY")
    print("=" * W)
    print(
        f" {'model':20} {'dataset':16} {'N':>2} {'natF1':>6} {'natEM':>6} {'rkF1':>6} {'rkEM':>6} {'dF1':>6}"
    )
    for (
        mname,
        ds,
        n,
        nf,
        ne,
        rf,
        re_,
    ) in results:
        print(
            f" {mname:20} {ds:16} {n:>2} {nf:6.3f} {ne:6.3f} {rf:6.3f} {re_:6.3f} {rf - nf:+6.3f}"
        )
    if results:
        avg_nat = sum(r[3] for r in results) / len(results)
        avg_rk = sum(r[5] for r in results) / len(results)
        print("-" * W)
        print(
            f" AVG native F1={avg_nat:.3f} | RedKnot F1={avg_rk:.3f} | delta={avg_rk - avg_nat:+.3f}"
        )
    print("=" * W)


if __name__ == "__main__":
    main()
