#!/usr/bin/env python3
"""Verify Plan-B chunked driver: full-attn RedKnot sparsity + chunk-wise linear
state accumulation, vs native, on Qwen3.5-35B-A3B.

Plan B (user's design):
  * chunk 1 = exact reusable prefix (linear state + full KV).
  * chunk k>=2 forwards only ITS OWN tokens, continuing linear state from the
    accumulated prefix (no re-scan); full layers run head-class sparsity.
This is realised by chunk-by-chunk prefill through one shared cache.

We compare next-token logits of the chunked driver vs native full-recompute on
several real LongBench samples and report agreement. (Loss vs native is the
unavoidable FULL-attn sparsity cost; the chunked linear relay should add no
extra error beyond that — confirmed earlier that relay==single-pass.)
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
DATASETS = os.environ.get("REDKNOT_DATASETS", "hotpotqa,2wikimqa,triviaqa").split(",")
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
MAX_CTX = int(os.environ.get("REDKNOT_MAX_CTX", "8000"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
FRAC = float(os.environ.get("REDKNOT_FRAC_GLOBAL", "0.10"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "4096"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


def load_ds(name, tok, n, seed):
    raw = []
    with open(os.path.join(LB, f"{name}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(seed).shuffle(raw)
    out = []
    for r in raw[: n * 2]:
        ids = tok(r["context"], add_special_tokens=False)["input_ids"][:MAX_CTX]
        chunks = [
            tok.decode(ids[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, len(ids), CHUNK)
        ]
        out.append({"q": r["input"], "chunks": chunks})
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        build_full_attention_head_config,
        run_redknot_qwen35_chunked,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    head_cfg = build_full_attention_head_config(
        model.config, frac_global=FRAC, local_window=WINDOW
    )

    W = 80
    print("=" * W)
    print(f" PLAN-B chunked driver vs native — {Path(MODEL).name}")
    print(f" frac_global={FRAC} window={WINDOW} chunk={CHUNK}")
    print("=" * W)
    print(f" {'dataset':14} {'chunks':>6} {'match':>6} {'rk_text':<30}")
    matches = []
    for ds in DATASETS:
        for s in load_ds(ds, tok, N, SEED):
            qt = QP.format(q=s["q"])
            # native reference next token
            ctx = "\n\n".join(s["chunks"])
            rids = tok(ctx + qt, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(model.device)
            ref_tok = int(
                model(input_ids=rids, use_cache=False).logits[0, -1, :].argmax()
            )
            # plan-b chunked
            text, _ = run_redknot_qwen35_chunked(
                model,
                tok,
                segments=s["chunks"],
                query_text=qt,
                head_cfg=head_cfg,
                max_new_tokens=8,
            )
            rk_first = tok(text, add_special_tokens=False)["input_ids"]
            rk_tok = rk_first[0] if rk_first else -1
            m = int(rk_tok == ref_tok)
            matches.append(m)
            print(f" {ds:14} {len(s['chunks']):>6} {str(bool(m)):>6} {text[:30]!r}")
    print("-" * W)
    print(
        f" next-token match rate: {sum(matches)}/{len(matches)} = {sum(matches) / len(matches):.2f}"
    )
    print("=" * W)


if __name__ == "__main__":
    main()
