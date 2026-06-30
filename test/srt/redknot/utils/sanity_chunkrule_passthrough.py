#!/usr/bin/env python3
"""SANITY: minimal-invasive linear patch fidelity on Qwen3.5-35B-A3B.

Wrap each linear layer's chunk_gated_delta_rule with a pass-through (calls the
ORIGINAL kernel, unchanged). The native forward is otherwise untouched. If
next-token logits / F1 differ from baseline, the wrapping mechanism itself is
broken. They must match EXACTLY. This separates wiring fidelity from the
windowing algorithm.

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/sanity_chunkrule_passthrough.py
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
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
DS = os.environ.get("REDKNOT_DATASETS", "hotpotqa").split(",")[0]
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
MAX_CTX = int(os.environ.get("REDKNOT_MAX_CTX", "8000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        install_linear_chunkrule_passthrough,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()

    raw = []
    with open(os.path.join(LB, f"{DS}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context"):
                raw.append(r)
    random.Random(0).shuffle(raw)

    print("=" * 64)
    print(" SANITY: chunk_gated_delta_rule pass-through vs native")
    print("=" * 64)
    diffs = []
    matches = []
    for r in raw[:N]:
        ids = tok(r["context"], add_special_tokens=False)["input_ids"][:MAX_CTX]
        text = tok.decode(ids, skip_special_tokens=True) + QP.format(q=r["input"])
        t = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        ref = model(input_ids=t, use_cache=False).logits[0, -1, :].float()
        restore = install_linear_chunkrule_passthrough(model)
        try:
            wr = model(input_ids=t, use_cache=False).logits[0, -1, :].float()
        finally:
            restore()
        d = (wr - ref).abs().max().item()
        m = int(wr.argmax() == ref.argmax())
        diffs.append(d)
        matches.append(m)
        print(f"   sample: max_logit_diff={d:.6f}  next-token match={bool(m)}")
    print("-" * 64)
    md = sum(diffs) / len(diffs)
    mm = sum(matches) / len(matches)
    print(f" AVG max_logit_diff={md:.6f}  match_rate={mm:.2f}")
    print(
        f" VERDICT: {'PASS (wrapping faithful)' if md < 1e-3 and mm == 1.0 else 'FAIL (wrapping changes outputs!)'}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
