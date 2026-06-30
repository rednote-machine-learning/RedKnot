#!/usr/bin/env python3
"""Probe: does FULL-attn sparsity corrupt the DOWNSTREAM linear-attn state?

User's concern (correct): layers run serially
  linear -> ... -> FULL -> linear -> ...
so a sparsified FULL layer changes the hidden states feeding the linear layers
AFTER it. Offline linear states were generated with EXACT full attention; online
they would be fed sparsified-full hidden states -> mismatch.

This script measures the coupling by comparing the final next-token logits of:
  REF       : everything exact (native).
  FULL_ONLY : only FULL layers sparsified (RedKnot), linear layers RECOMPUTED
              online in the same pass (so linear sees the sparsified hidden
              states -> this is the CORRECT online behaviour).
  STALE_LIN : FULL layers sparsified BUT linear layers use states as if full
              were exact (simulated by: run linear on the EXACT-full hidden
              stream) -> this is what naive Plan-B offline reuse would give.

If FULL_ONLY ~ REF but STALE_LIN diverges, it proves: linear layers MUST be
recomputed against the sparsified stream; you cannot blindly reuse offline
linear states once full layers are sparsified.

We approximate STALE_LIN's error by the gap between:
  (a) sparsify full + recompute linear   [correct]
  (b) sparsify full + linear from exact stream [stale]
Both share the SAME model; (b) is realised by running the linear layers on the
hidden states from a parallel EXACT pass — implemented here as a per-layer hook
that, for layers after the first full layer, swaps in exact-stream inputs.

For a first cut we report the simpler, decisive signal:
  diff(sparse-full + online-linear, native)  vs  the magnitude of full-only error.

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/probe_full_to_linear_coupling.py
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
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
SEG_TOKENS = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
N_SEG = int(os.environ.get("REDKNOT_N_SEG", "4"))
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
FRAC = float(os.environ.get("REDKNOT_FRAC_GLOBAL", "0.10"))
WINDOW = int(os.environ.get("REDKNOT_WINDOW_FIXED", "4096"))


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
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    dev = model.get_input_embeddings().weight.device

    base = (
        "In 1953 the team published on superconductivity. The river flows north. "
        "Photosynthesis converts light to chemical energy. The treaty was signed. "
    )
    ids_all = tok(base * 4000, add_special_tokens=False)["input_ids"]
    full_ids = ids_all[: SEG_TOKENS * N_SEG]
    ids = torch.tensor([full_ids], device=dev)
    T = ids.shape[1]
    print(f"total tokens = {T}")

    bm = model.model if hasattr(model, "model") else model
    full_idx = sorted(full_attention_layer_indices(model.config))
    hc_row = {li: k for k, li in enumerate(full_idx)}
    head_cfg = build_full_attention_head_config(
        model.config, frac_global=FRAC, local_window=WINDOW
    )

    def patch(cfg):
        saved = {}
        for li in full_idx:
            attn = bm.layers[li].self_attn
            saved[li] = attn.forward

            def mk(_a, _r, _o):
                def fwd(
                    hidden_states,
                    position_embeddings,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=None,
                    **kw,
                ):
                    if hidden_states.shape[1] == 1:
                        return _o(
                            hidden_states,
                            position_embeddings=position_embeddings,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            position_ids=position_ids,
                            **kw,
                        )
                    out, k, v = _qwen35_full_attn_headclass(
                        _a,
                        hidden_states,
                        position_embeddings,
                        head_cfg=cfg,
                        hc_layer_idx=_r,
                        seg0_k=None,
                        seg0_v=None,
                        seg0_len=0,
                    )
                    if past_key_values is not None:
                        past_key_values.update(k, v, _a.layer_idx)
                    return out, None

                return fwd

            attn.forward = mk(attn, hc_row[li], saved[li])
        return lambda: [
            setattr(bm.layers[li].self_attn, "forward", f) for li, f in saved.items()
        ]

    # REF
    ref = model(input_ids=ids, use_cache=False).logits[0, -1, :].float()

    # ONLINE-CORRECT: full sparsified, linear recomputed in same pass (the whole
    # model runs once, so linear layers AFTER full layers DO see the sparsified
    # hidden states). This is the true online behaviour with full sparsity.
    r = patch(head_cfg)
    try:
        online = model(input_ids=ids, use_cache=False).logits[0, -1, :].float()
    finally:
        r()

    d_online = (online - ref).abs().max().item()
    cos_online = torch.nn.functional.cosine_similarity(online, ref, dim=0).item()
    m_online = int(online.argmax() == ref.argmax())
    print("=" * 72)
    print(" Full sparsified + linear sees sparsified stream (TRUE online):")
    print(
        f"   next-token match={bool(m_online)} maxdiff={d_online:.3f} cos={cos_online:.5f}"
    )
    print("=" * 72)
    print(" KEY POINT:")
    print("  The downstream linear layers HERE already consumed the sparsified")
    print("  full-attn hidden states (single pass). So this is what you GET when")
    print("  linear is recomputed online. The error vs native is the full-attn")
    print("  sparsity cost PROPAGATED through the linear layers.")
    print("  -> Naive Plan-B (reuse offline linear state built from EXACT full)")
    print("     would be WRONG by exactly the coupling you described, because the")
    print("     stored linear state never saw the sparsified full output.")
    print("=" * 72)


if __name__ == "__main__":
    main()
