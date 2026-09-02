#!/usr/bin/env python3
"""Conservative DeepSeek-V4 RedKnot prefill arithmetic ledger.

The ledger deliberately gives Indexer/compressor/router/norm no speculative
saving credit.  It counts the model's expert GEMMs, MLA Q/attention/wo_a, and
fixed MLA projections, then applies only runtime-measurable head-row and
token/routed-expert keep ratios.  It is a FLOPs proxy, not elapsed time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PrefillLedger:
    token_full_ratio: float
    moe_arithmetic_saving: float
    mla_head_arithmetic_saving: float
    total_online_saving: float
    full_input_saving_with_first_document_prefix: float
    online_compute_ratio: float


def estimate_prefill_saving(
    *,
    token_full_ratio: float,
    mla_head_row_saving: float = 0.752733048994,
    total_tokens: int = 262197,
    first_document_tokens: int = 32768,
) -> PrefillLedger:
    if not 0.0 <= token_full_ratio <= 1.0:
        raise ValueError("token_full_ratio must be in [0, 1]")
    if not 0.0 <= mla_head_row_saving <= 1.0:
        raise ValueError("mla_head_row_saving must be in [0, 1]")
    if not 0 <= first_document_tokens < total_tokens:
        raise ValueError("first-document geometry is invalid")

    # DeepSeek-V4-Flash-0731: H=4096, I=2048, 64 heads, D=512,
    # q/o LoRA rank=1024, native Top-6 + one shared expert.
    hidden, inter, heads, head_dim = 4096, 2048, 64, 512
    q_lora = o_lora = 1024
    topk_candidates = 512
    native_experts = 7
    layers, sparse_layers, fence_layers = 43, 37, 6

    moe = native_experts * 6.0 * hidden * inter
    q_b = 2.0 * q_lora * heads * head_dim
    wo_a = 2.0 * heads * head_dim * o_lora
    sparse_attention = 4.0 * heads * head_dim * topk_candidates
    mla_head_dependent = q_b + wo_a + sparse_attention
    mla_fixed = (
        2.0 * hidden * q_lora
        + 2.0 * hidden * head_dim
        + 2.0 * o_lora * hidden
    )
    baseline = moe + mla_head_dependent + mla_fixed

    # Fence layers retain 6 routed + 1 shared.  Middle layers retain the
    # shared expert on every token and run physical K3 routed experts only on
    # token_full_ratio of rows.
    online_expert_slots = (
        fence_layers * native_experts
        + sparse_layers * (1.0 + 3.0 * token_full_ratio)
    )
    native_expert_slots = layers * native_experts
    moe_ratio = online_expert_slots / native_expert_slots
    moe_saving = 1.0 - moe_ratio
    head_ratio = 1.0 - mla_head_row_saving
    online_ratio = (
        moe * moe_ratio
        + mla_head_dependent * head_ratio
        + mla_fixed
    ) / baseline
    online_saving = 1.0 - online_ratio

    remaining_fraction = (total_tokens - first_document_tokens) / total_tokens
    full_input_ratio = remaining_fraction * online_ratio
    return PrefillLedger(
        token_full_ratio=token_full_ratio,
        moe_arithmetic_saving=moe_saving,
        mla_head_arithmetic_saving=mla_head_row_saving,
        total_online_saving=online_saving,
        full_input_saving_with_first_document_prefix=1.0 - full_input_ratio,
        online_compute_ratio=online_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-full-ratio", type=float, required=True)
    parser.add_argument("--target-saving", type=float, default=0.70)
    args = parser.parse_args()
    ledger = estimate_prefill_saving(token_full_ratio=args.token_full_ratio)
    for key, value in asdict(ledger).items():
        print(f"{key}={value:.9f}")
    print(f"target_pass={ledger.total_online_saving >= args.target_saving}")


if __name__ == "__main__":
    main()
