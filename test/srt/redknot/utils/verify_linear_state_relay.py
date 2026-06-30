#!/usr/bin/env python3
"""Plan-B core check: can Qwen3.5 linear-attention (GatedDeltaNet) state be
produced SEGMENT-BY-SEGMENT offline and relayed online to equal a single
full-sequence pass?

Plan B stores, per offline segment, the linear layers' (conv_state,
recurrent_state) computed as a prefix CHAIN: seg_k's state starts from seg_{k-1}'s
final state. Online we reload the last segment's state to resume. For full layers
we keep the standard RedKnot KV reuse (not exercised here).

This script verifies the linear-state relay numerically:
  REF  : one forward over the WHOLE concatenated context (4 segments x 4K).
  RELAY: feed the 4 segments sequentially, each continuing from the previous
         segment's cached (conv,recurrent) state via a DynamicCache. The final
         hidden state / next-token logits must match REF.

If RELAY == REF, segment-wise offline linear-state generation + ordered relay is
valid (Plan B is sound for the linear layers).

Run:
  HF_ENDPOINT=https://huggingface.co PYTHONPATH=python:<venv-sp>:<sys-sp> \
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_tf5/bin/python \
    test/srt/redknot/verify_linear_state_relay.py
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


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    dev = model.get_input_embeddings().weight.device

    # Build N_SEG segments of SEG_TOKENS each from a long repeated text.
    base = (
        "In 1953, the research team published findings on superconductivity. "
        "The river flows north through three provinces before reaching the sea. "
        "Photosynthesis converts light energy into chemical energy in plants. "
        "The treaty was signed by twelve nations after long negotiations. "
    )
    big = base * 4000
    all_ids = tok(big, add_special_tokens=False)["input_ids"]
    seg_ids = []
    p = 0
    for _ in range(N_SEG):
        seg_ids.append(all_ids[p : p + SEG_TOKENS])
        p += SEG_TOKENS
    full_ids = [t for seg in seg_ids for t in seg]
    full = torch.tensor([full_ids], device=dev)
    total = full.shape[1]
    print(f"segments={N_SEG} x {SEG_TOKENS} -> total {total} tokens")

    # ── REF: single full-sequence pass ──
    ref = model(input_ids=full, use_cache=False).logits[0, -1, :].float()

    # ── RELAY: feed segments sequentially through ONE DynamicCache ──
    # The cache carries linear (conv/recurrent) state AND full-attn KV forward
    # across segments. Each segment continues from the prior segment's state.
    cache = DynamicCache(config=model.config)
    pos = 0
    last_logits = None
    for k, sids in enumerate(seg_ids):
        chunk = torch.tensor([sids], device=dev)
        position_ids = torch.arange(pos, pos + chunk.shape[1], device=dev).unsqueeze(0)
        out = model(
            input_ids=chunk,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        last_logits = out.logits[0, -1, :].float()
        pos += chunk.shape[1]
    relay = last_logits

    diff = (relay - ref).abs().max().item()
    match = int(relay.argmax() == ref.argmax())
    cos = torch.nn.functional.cosine_similarity(relay, ref, dim=0).item()
    print("=" * 70)
    print(" PLAN-B linear-state relay vs single full pass:")
    print(
        f"   next-token match : {bool(match)} (ref={int(ref.argmax())}, relay={int(relay.argmax())})"
    )
    print(f"   max logit diff   : {diff:.4f}")
    print(f"   cosine sim       : {cos:.6f}")
    ok = bool(match) and diff < 2.0
    print(
        f"   verdict          : {'PASS — segment relay == full pass (Plan B sound)' if ok else 'FAIL — relay diverges'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
