#!/usr/bin/env python3
"""B300/SM103 numerical oracle for the Pro-0813 groups=2 W_OA merge.

The probe is intentionally model-independent: it exercises the exact TP8
rank-local production geometry (16 heads, head_dim=512, two output groups,
o_lora_rank=1024) before the 893 GB checkpoint finishes downloading.
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence

import torch

from sglang.srt.layers.attention.redknot.dsv4_fused_z_merge import (
    PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN,
    PersistentProjectionView,
    ProjectionSpanGeometry,
    build_persistent_projection_plan,
    preflight_persistent_headsplit_woa_merge,
    project_merge_persistent_headsplit,
)


OWNED_HEADS = 16
HEAD_DIM = 512
GROUPS = 2
O_LORA_RANK = 1024
HEADS_PER_GROUP = OWNED_HEADS // GROUPS
GLOBAL_HEAD_AXES = (0, 8)
LOCAL_HEAD_AXES = tuple(
    axis for axis in range(OWNED_HEADS) if axis not in GLOBAL_HEAD_AXES
)


def _bf16_mm_fp32(left: torch.Tensor, right_t: torch.Tensor) -> torch.Tensor:
    try:
        return torch.mm(left, right_t, out_dtype=torch.float32)
    except TypeError:
        return left.float().mm(right_t.float())


def _head_projection(
    attention: torch.Tensor,
    weight: torch.Tensor,
    axes: Sequence[int],
) -> torch.Tensor:
    rows = int(attention.shape[0])
    grouped_weight = weight.reshape(
        GROUPS, O_LORA_RANK, HEADS_PER_GROUP, HEAD_DIM
    )
    projected = torch.zeros(
        rows,
        GROUPS,
        O_LORA_RANK,
        dtype=torch.float32,
        device=attention.device,
    )
    for group_id in range(GROUPS):
        group_start = group_id * HEADS_PER_GROUP
        selected_axes = tuple(
            axis for axis in axes if group_start <= axis < group_start + HEADS_PER_GROUP
        )
        if not selected_axes:
            continue
        group_axes = tuple(axis - group_start for axis in selected_axes)
        selected_attention = attention[:, selected_axes, :].reshape(
            rows, -1
        ).contiguous()
        selected_weight = grouped_weight[group_id, :, group_axes, :].reshape(
            O_LORA_RANK, -1
        ).contiguous()
        projected[:, group_id, :] = _bf16_mm_fp32(
            selected_attention, selected_weight.T
        )
    return projected


def _complement_runs(total_rows: int, dirty_rows: Iterable[int]) -> list[tuple[int, int]]:
    dirty = set(int(row) for row in dirty_rows)
    runs: list[tuple[int, int]] = []
    start = None
    for row in range(total_rows + 1):
        clean = row < total_rows and row not in dirty
        if clean and start is None:
            start = row
        elif not clean and start is not None:
            runs.append((start, row))
            start = None
    return runs


def _split_run(start: int, end: int, pieces: int) -> list[tuple[int, int]]:
    length = end - start
    pieces = min(max(1, pieces), length)
    base, extra = divmod(length, pieces)
    runs = []
    cursor = start
    for index in range(pieces):
        width = base + (1 if index < extra else 0)
        runs.append((cursor, cursor + width))
        cursor += width
    return runs


def _run_case(
    *,
    rows: int,
    dirty_values: tuple[int, ...],
    split_clean_into: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)
    attention = (
        torch.randn(rows, OWNED_HEADS, HEAD_DIM, device=device) * 0.02
    ).to(torch.bfloat16).contiguous()
    weight = (
        torch.randn(
            GROUPS * O_LORA_RANK,
            HEADS_PER_GROUP * HEAD_DIM,
            device=device,
        )
        * 0.02
    ).to(torch.bfloat16).contiguous()

    global_projection = _head_projection(attention, weight, GLOBAL_HEAD_AXES)
    local_projection = _head_projection(attention, weight, LOCAL_HEAD_AXES)
    stored_local = local_projection.to(torch.bfloat16)

    clean_runs = _complement_runs(rows, dirty_values)
    if not clean_runs:
        raise AssertionError("persistent merge probe requires at least one clean row")
    if len(clean_runs) == 1 and split_clean_into > 1:
        clean_runs = _split_run(*clean_runs[0], split_clean_into)
    if len(clean_runs) > 16:
        raise AssertionError("probe generated more than sixteen persistent views")

    views = []
    for view_index, (output_start, output_end) in enumerate(clean_runs):
        span = output_end - output_start
        # Non-zero local_start proves that the kernel honors each view's source
        # stride instead of accidentally indexing z_off by the forward row.
        values = torch.zeros(
            span + 2,
            GROUPS,
            O_LORA_RANK,
            dtype=torch.bfloat16,
            device=device,
        )
        values[1 : span + 1].copy_(stored_local[output_start:output_end])
        geometry = ProjectionSpanGeometry(
            output_rows=tuple(range(output_start, output_end)),
            local_rows=tuple(range(1, span + 1)),
        )
        views.append(
            PersistentProjectionView.bind(
                seg_hash=f"sm103-case-{seed}-view-{view_index}",
                layer_id=3,
                commit_epoch=1,
                geometry=geometry,
                values=values,
                generation_token=f"sm103-generation-{seed}",
            )
        )

    projection_plan = build_persistent_projection_plan(
        total_rows=rows,
        tail_shape=(GROUPS, O_LORA_RANK),
        views=views,
    )
    dirty_cpu = torch.tensor(dirty_values, dtype=torch.long)
    dirty_device = dirty_cpu.to(device=device)
    plan = preflight_persistent_headsplit_woa_merge(
        projection_plan=projection_plan,
        dirty_rows=dirty_device,
        dirty_rows_cpu=dirty_cpu,
        local_head_axes=LOCAL_HEAD_AXES,
        wo_a_weight=weight,
        owned_heads=OWNED_HEADS,
        groups=GROUPS,
        head_dim=HEAD_DIM,
        o_lora_rank=O_LORA_RANK,
    )
    if plan.kernel_token != PERSISTENT_HEADSPLIT_WOA_MERGE_KERNEL_TOKEN:
        raise AssertionError(f"unexpected groups=2 kernel token: {plan.kernel_token}")
    if plan.uses_cublas_woa_fp32_fastpath:
        raise AssertionError("groups=2 must not enter the groups=1 cuBLAS certificate")

    actual = project_merge_persistent_headsplit(attention, plan)
    expected = global_projection.to(torch.bfloat16)
    if dirty_values:
        dirty_index = torch.tensor(dirty_values, dtype=torch.long, device=device)
        expected[dirty_index] = (
            expected[dirty_index].float()
            + local_projection[dirty_index]
        ).to(torch.bfloat16)
    clean_mask = torch.ones(rows, dtype=torch.bool, device=device)
    if dirty_values:
        clean_mask[dirty_index] = False
    expected[clean_mask] = (
        global_projection[clean_mask]
        + stored_local[clean_mask].float()
    ).to(torch.bfloat16)

    absolute_error = (actual.float() - expected.float()).abs()
    max_abs = float(absolute_error.max().item())
    mean_abs = float(absolute_error.mean().item())
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=5.0e-3,
        rtol=5.0e-2,
    )
    return {
        "rows": rows,
        "dirty_rows": len(dirty_values),
        "views": len(views),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "kernel_token": plan.kernel_token,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = tuple(torch.cuda.get_device_capability(0))
    device_name = torch.cuda.get_device_name(0)
    if capability != (10, 3) or "B300" not in device_name:
        raise RuntimeError(
            f"this certificate is B300/SM103-only, got {device_name} SM{capability}"
        )
    torch.backends.cuda.matmul.allow_tf32 = False

    cases = [
        dict(rows=1, dirty_values=(), split_clean_into=1, seed=10301),
        dict(rows=127, dirty_values=tuple(range(7)), split_clean_into=1, seed=10302),
        dict(rows=128, dirty_values=(), split_clean_into=4, seed=10303),
        dict(rows=129, dirty_values=(0, 64, 128), split_clean_into=1, seed=10304),
        dict(rows=4096, dirty_values=tuple(range(128)), split_clean_into=8, seed=10305),
    ]
    results = [_run_case(**case) for case in cases]
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "pass",
                "device": device_name,
                "compute_capability": "sm_103",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cases": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
