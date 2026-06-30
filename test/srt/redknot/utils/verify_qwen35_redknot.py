#!/usr/bin/env python3
"""Fast correctness check for RedKnot Step-1 on small Qwen3.5 (0.8B).

0.8B has the SAME hybrid architecture as 397B (linear+full attention, head_dim
256, Q-gating) but is bf16 (no FP8) and loads in seconds — ideal for debugging
the patch wiring + numerics without the 397B cost.

Test strategy:
  1. ALL-GLOBAL head config: RedKnot head-class attention then reduces to exact
     full causal attention, so RedKnot prefill logits MUST match the native
     model's prefill logits. Any mismatch is a wiring bug (patch not firing,
     Q-gating wrong, rope wrong, ...), NOT a sparsity effect.
  2. Report next-token agreement + max logit diff.

Run:
  HF_ENDPOINT=https://huggingface.co \
    PYTHONPATH=python:.venv_tf5/lib/python3.11/site-packages:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0 .venv_tf5/bin/python \
    test/srt/redknot/verify_qwen35_redknot.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-0.8B",
)


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        _qwen35_full_attn_headclass,
        build_full_attention_head_config,
        full_attention_layer_indices,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()

    text = (
        "The capital of France is Paris. The capital of Japan is Tokyo. "
        "Question: what is the capital of France? Answer:"
    ) * 4  # make it a few hundred tokens
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    T = ids.shape[1]
    print(f"prompt tokens T={T}")

    # ── Native reference logits ──
    ref = model(input_ids=ids, use_cache=False).logits[0, -1, :].float()

    # ── RedKnot ALL-GLOBAL: must equal native ──
    cfg = model.config
    full_idx = sorted(full_attention_layer_indices(cfg))
    hc_row = {li: k for k, li in enumerate(full_idx)}
    # all-global: frac_global=1.0
    head_cfg = build_full_attention_head_config(cfg, frac_global=1.0, local_window=4096)
    s = head_cfg.summary()
    print(
        f"full layers={full_idx} | head_cfg layers={head_cfg.num_layers} "
        f"global={s.get('global', 0)} local={s.get('local', 0)}"
    )

    bm = model.model if hasattr(model, "model") else model
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

    try:
        rk = model(input_ids=ids, use_cache=False).logits[0, -1, :].float()
    finally:
        for li, f in saved.items():
            bm.layers[li].self_attn.forward = f

    diff = (rk - ref).abs().max().item()
    same_tok = int(rk.argmax() == ref.argmax())
    print("=" * 60)
    print(f" ALL-GLOBAL RedKnot vs native:")
    print(
        f"   next-token match : {bool(same_tok)} "
        f"(native={int(ref.argmax())}, rk={int(rk.argmax())})"
    )
    print(f"   max logit diff   : {diff:.4f}")
    print(
        f"   verdict          : {'PASS (wiring correct)' if same_tok and diff < 2.0 else 'FAIL (wiring bug)'}"
    )
    print("=" * 60)

    # ── Sweet-spot sparse config (frac_global=0.10): generate + compare ──
    sp_cfg = build_full_attention_head_config(cfg, frac_global=0.10, local_window=4096)
    ss = sp_cfg.summary()
    print(
        f" SPARSE frac_global=0.10: global={ss.get('global', 0)} local={ss.get('local', 0)}"
    )
    saved2 = {}
    for li in full_idx:
        attn = bm.layers[li].self_attn
        saved2[li] = attn.forward

        def mk2(_attn, _row, _orig):
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
                    head_cfg=sp_cfg,
                    hc_layer_idx=_row,
                    seg0_k=None,
                    seg0_v=None,
                    seg0_len=0,
                )
                if past_key_values is not None:
                    past_key_values.update(k, v, _attn.layer_idx)
                return out, None

            return fwd

        attn.forward = mk2(attn, hc_row[li], saved2[li])
    try:
        rk2 = model(input_ids=ids, use_cache=False).logits[0, -1, :].float()
    finally:
        for li, f in saved2.items():
            bm.layers[li].self_attn.forward = f
    diff2 = (rk2 - ref).abs().max().item()
    same2 = int(rk2.argmax() == ref.argmax())
    print(f"   sparse next-token match : {bool(same2)} (rk={int(rk2.argmax())})")
    print(f"   sparse max logit diff   : {diff2:.4f}")
    print("=" * 60)
    print(f" ALL-GLOBAL RedKnot vs native:")
    print(
        f"   next-token match : {bool(same_tok)} "
        f"(native={int(ref.argmax())}, rk={int(rk.argmax())})"
    )
    print(f"   max logit diff   : {diff:.4f}")
    print(
        f"   verdict          : {'PASS (wiring correct)' if same_tok and diff < 2.0 else 'FAIL (wiring bug)'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
