#!/usr/bin/env python3
"""Ablation: is the RedKnot accuracy drop from the FULL-attention sparsity or
the linear-attention layers? (Qwen3.5-35B-A3B)

The driver only patches full_attention layers; linear layers run native. To
prove where the loss comes from we compare three configs on the same samples:

  A) native            : no patching at all (reference).
  B) rk_allglobal      : patch full layers but with frac_global=1.0
                         (head-class attention == exact full causal attention).
                         If B == A, then (i) linear layers are untouched/correct
                         and (ii) the patch wiring is correct -> ANY loss is due
                         to sparsity, NOT linear layers or wiring.
  C) rk_sweetspot      : patch full layers with frac_global=0.10, window=4096.
                         (C vs B) isolates the cost of head-class SPARSITY on the
                         full-attention layers.

Per dataset we report next-token-level logit agreement (argmax match + max abs
logit diff vs native) which is a cleaner signal than end F1 for localisation.

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/ablate_qwen35_35b.py
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
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "hotpotqa,2wikimqa,multifieldqa_en"
).split(",")
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
MAX_CTX = int(os.environ.get("REDKNOT_MAX_CTX", "8000"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")


def load_ds(name, tok, n, seed):
    raw = []
    with open(os.path.join(LONGBENCH_DIR, f"{name}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context") and r.get("answers"):
                raw.append(r)
    random.Random(seed).shuffle(raw)
    out = []
    for r in raw[: n * 2]:
        ids = tok(r["context"], add_special_tokens=False)["input_ids"][:MAX_CTX]
        docs = [
            tok.decode(ids[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, len(ids), CHUNK)
        ]
        out.append({"q": r["input"], "docs": docs})
        if len(out) >= n:
            break
    return out


QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


@torch.no_grad()
def prefill_logits(model, ids):
    return model(input_ids=ids, use_cache=False).logits[0, -1, :].float()


def patch_full(model, head_cfg):
    """Patch full layers with given head_cfg; return restore fn."""
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _qwen35_full_attn_headclass,
        full_attention_layer_indices,
    )

    bm = model.model if hasattr(model, "model") else model
    full_idx = sorted(full_attention_layer_indices(model.config))
    hc_row = {li: k for k, li in enumerate(full_idx)}
    saved = {}
    for li in full_idx:
        attn = bm.layers[li].self_attn
        saved[li] = attn.forward

        def mk(_attn, _row, _orig):
            def fwd(
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                position_ids=None,
                **kw,
            ):
                if hidden_states.shape[1] == 1:
                    return _orig(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        **kw,
                    )
                out, k, v = _qwen35_full_attn_headclass(
                    _attn,
                    hidden_states,
                    position_embeddings,
                    head_cfg=head_cfg,
                    hc_layer_idx=_row,
                    seg0_k=None,
                    seg0_v=None,
                    seg0_len=0,
                )
                if past_key_values is not None:
                    past_key_values.update(k, v, _attn.layer_idx)
                return out, None

            return fwd

        attn.forward = mk(attn, hc_row[li], saved[li])

    def restore():
        for li, f in saved.items():
            bm.layers[li].self_attn.forward = f

    return restore


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        build_full_attention_head_config,
    )

    W = 92
    print("=" * W)
    print(f" ABLATION: localise RedKnot accuracy loss — {Path(MODEL).name}")
    print(" A=native  B=rk_allglobal(==exact full attn)  C=rk_sweetspot(frac0.1,w4096)")
    print("=" * W)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()

    hc_global = build_full_attention_head_config(
        model.config, frac_global=1.0, local_window=4096
    )
    hc_sweet = build_full_attention_head_config(
        model.config, frac_global=0.10, local_window=4096
    )

    print(
        f" {'dataset':16} {'ctx':>6} | {'B match':>8} {'B diff':>8} | {'C match':>8} {'C diff':>8}"
    )
    agg = {"B_match": [], "B_diff": [], "C_match": [], "C_diff": []}
    for ds in DATASETS:
        samples = load_ds(ds, tok, N, SEED)
        for s in samples:
            text = "\n\n".join(s["docs"]) + QP.format(q=s["q"])
            ids = tok(text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(model.device)
            T = ids.shape[1]

            ref = prefill_logits(model, ids)

            r = patch_full(model, hc_global)
            try:
                b = prefill_logits(model, ids)
            finally:
                r()
            r = patch_full(model, hc_sweet)
            try:
                c = prefill_logits(model, ids)
            finally:
                r()

            bm_ = int(b.argmax() == ref.argmax())
            bd = (b - ref).abs().max().item()
            cm_ = int(c.argmax() == ref.argmax())
            cd = (c - ref).abs().max().item()
            agg["B_match"].append(bm_)
            agg["B_diff"].append(bd)
            agg["C_match"].append(cm_)
            agg["C_diff"].append(cd)
            print(
                f" {ds:16} {T:6d} | {str(bool(bm_)):>8} {bd:8.3f} | {str(bool(cm_)):>8} {cd:8.3f}"
            )

    print("-" * W)
    bm = sum(agg["B_match"]) / len(agg["B_match"])
    cm = sum(agg["C_match"]) / len(agg["C_match"])
    bd = sum(agg["B_diff"]) / len(agg["B_diff"])
    cd = sum(agg["C_diff"]) / len(agg["C_diff"])
    print(
        f" AVG  B(all-global) match={bm:.2f} diff={bd:.3f} | C(sweet) match={cm:.2f} diff={cd:.3f}"
    )
    print("=" * W)
    print(" INTERPRETATION:")
    if bm > 0.95 and bd < 2.0:
        print(
            "  B≈native -> linear layers + wiring CORRECT. Loss is from FULL-attn SPARSITY."
        )
    else:
        print(
            "  B != native -> wiring/linear-layer issue on full layers (NOT just sparsity)."
        )
    print(
        f"  Sparsity cost (C vs native): match drop = {bm - cm:+.2f}, extra logit diff = {cd - bd:+.3f}"
    )
    print("=" * W)


if __name__ == "__main__":
    main()
