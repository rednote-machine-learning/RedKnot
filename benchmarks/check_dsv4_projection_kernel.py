"""Opt-in CUDA check of native FP8 DSV4 projection with RedKnot masked z_off.

Example (choose an actually free device yourself):
  python benchmarks/check_dsv4_projection_kernel.py --run --device cuda:0

Import and --help do not import Torch/vLLM or initialize CUDA. Random, small
weights exercise the pinned native inverse-RoPE, activation quantization, grouped
FP8 wo_a and its weight post-load layout. This is NOT a model/attention/KV/TTFT
qualification. Both masked branches still execute a full-width wo_a GEMM.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time


def _ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated head IDs") from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any(head < 0 or head >= 64 for head in result)
    ):
        raise argparse.ArgumentTypeError(
            "head IDs must be nonempty, unique and in [0,64)"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="explicitly authorize GPU run"
    )
    parser.add_argument("--device", help="explicit indexed CUDA device, e.g. cuda:0")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--rank", type=int, choices=(128, 256, 1024), default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--relative-rms", type=float, default=0.02)
    parser.add_argument("--max-absolute", type=float, default=0.05)
    parser.add_argument(
        "--local-head-ids",
        type=_ids,
        default=tuple(head for head in range(64) if head % 8),
    )
    return parser


def _error(actual, expected) -> dict:
    import torch

    difference = actual.float() - expected.float()
    rms = difference.square().mean().sqrt()
    norm = expected.float().square().mean().sqrt()
    return {
        "finite": bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()),
        "max_absolute": float(difference.abs().max()),
        "rms": float(rms),
        "relative_rms": float(rms / norm.clamp_min(1e-12)),
    }


def _measure(call, device, warmup: int, iterations: int) -> dict:
    import torch

    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    device_ms, wall_ms = [], []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        call()
        end.record()
        end.synchronize()
        device_ms.append(begin.elapsed_time(end))
        wall_ms.append((time.perf_counter() - start) * 1000)
    return {
        "warmup_iterations_excluded": warmup,
        "iterations": iterations,
        "cuda_event_median_ms": statistics.median(device_ms),
        "synchronized_wall_median_ms": statistics.median(wall_ms),
        "synchronized_wall_min_ms": min(wall_ms),
    }


def run(args) -> dict:
    # The CLI validates the explicit opt-in before reaching any GPU dependency.
    if (
        not args.run
        or not args.device
        or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", args.device) is None
    ):
        raise ValueError("GPU execution requires --run and an explicit --device cuda:N")
    from vllm_redknot.compat import verify_vllm_sources

    source_contract = verify_vllm_sources(engine_family="deepseek_v4_flash")
    import torch

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    import vllm
    from vllm.config import VllmConfig, set_current_vllm_config

    # Native CustomOps require the same default config context used by vLLM's
    # default_vllm_config test fixture. No model/engine is constructed here.
    with (
        torch.inference_mode(),
        torch.cuda.device(device),
        set_current_vllm_config(VllmConfig()),
    ):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
            deepgemm_post_process_fp8_weight_block,
        )
        from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
            DeepseekV4ScalingRotaryEmbedding,
        )
        from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
            compute_fp8_einsum_recipe,
            deep_gemm_fp8_o_proj,
        )
        from vllm.platforms import current_platform
        from vllm.utils.deep_gemm import (
            is_deep_gemm_e8m0_used,
            is_deep_gemm_supported,
            per_block_cast_to_fp8,
        )

        from vllm_redknot.dsv4_projection import (
            CachedContribution,
            capture_local_z,
            merge_cached_z_and_project,
        )

        if not is_deep_gemm_supported():
            raise RuntimeError(
                "native DeepGEMM is not supported/enabled on this device"
            )
        # Pinned recipe selection queries platform device 0. Do not silently use
        # its recipe on a different architecture selected by --device cuda:N.
        native_cap = current_platform.get_device_capability()
        selected_cap = torch.cuda.get_device_capability(device)
        if native_cap is None or tuple(native_cap) != tuple(selected_cap):
            raise RuntimeError(
                "selected device capability differs from native platform device 0; "
                "launch a fresh process with only the chosen device visible"
            )
        generator = torch.Generator(device=device).manual_seed(args.seed)
        groups, heads, head_dim, rope_dim = 8, 64, 512, 64
        heads_per_group, input_width, output_width = 8, 4096, 128
        recipe, aligned_scales = compute_fp8_einsum_recipe()
        if recipe[2] != 128 or head_dim % recipe[2]:
            raise RuntimeError(
                "native quantization blocks no longer partition logical heads"
            )

        def random(shape, scale=1.0):
            return (
                torch.randn(
                    shape, generator=generator, device=device, dtype=torch.bfloat16
                )
                * scale
            )

        # Flash's ColumnParallelLinear checkpoint wo_a is [G*R, H/G*D]. Its
        # DeepGemmFp8BlockScaledMMKernel post-load calls precisely this helper
        # with is_bmm=True and bmm_batch_size=G, producing [G,R,H/G*D].
        checkpoint_weight = random((groups * args.rank, input_width), input_width**-0.5)
        weight, power2_scale = per_block_cast_to_fp8(
            checkpoint_weight, block_size=[128, 128], use_ue8m0=True
        )
        del checkpoint_weight
        # Model Flash checkpoint scales are E8M0 exponents. Pack exact native
        # power-of-two scales (no weight dequantization/requantization). This is
        # the inverse of vLLM's _upcast_e8m0_to_fp32, not a GPU-layout guess.
        scale_bits = power2_scale.contiguous().view(torch.int32)
        if bool((scale_bits & ((1 << 23) - 1)).ne(0).any()):
            raise RuntimeError(
                "native per-block quantizer did not emit power-of-two scales"
            )
        checkpoint_scale = (scale_bits >> 23).to(torch.uint8).view(torch.float8_e8m0fnu)
        if not torch.equal(_upcast_e8m0_to_fp32(checkpoint_scale), power2_scale):
            raise RuntimeError("E8M0 checkpoint scale roundtrip failed")
        checkpoint_scale_shape = list(checkpoint_scale.shape)
        weight, native_scale = deepgemm_post_process_fp8_weight_block(
            weight,
            checkpoint_scale,
            quant_block_shape=(128, 128),
            use_e8m0=is_deep_gemm_e8m0_used(),
            is_bmm=True,
            bmm_batch_size=groups,
        )
        if weight.shape != (groups, args.rank, input_width):
            raise RuntimeError(
                f"unexpected native post-load wo_a shape: {weight.shape}"
            )
        if weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError("native wo_a is not FP8 E4M3")
        wo_a = torch.nn.Module()
        wo_a.register_buffer("weight", weight)
        wo_a.register_buffer("weight_scale_inv", native_scale)
        weight_before = weight.view(torch.uint8).clone()
        scale_before = native_scale.clone()

        # Synthetic but native DSV4 YaRN cache, with FP32 cos/sin and no magnitude
        # scaling. Nonzero positions exercise inverse interleaved RoPE on the
        # LAST 64 dimensions, exactly inside deep_gemm_fp8_o_proj.
        rope = DeepseekV4ScalingRotaryEmbedding(
            head_size=head_dim,
            rotary_dim=rope_dim,
            max_position_embeddings=512,
            base=10000,
            is_neox_style=False,
            scaling_factor=16,
            dtype=torch.bfloat16,
            mscale=0,
            mscale_all_dim=0,
        ).to(device=device)
        if rope.cos_sin_cache.dtype != torch.float32:
            raise RuntimeError("native DSV4 RoPE cache is no longer FP32")
        identity = torch.nn.Identity()
        calls = {"native_project_z": 0, "wo_b": 0}
        last = {}
        final_weight = random(
            (groups * args.rank, output_width), (groups * args.rank) ** -0.5
        )

        def project_z(o, positions):
            calls["native_project_z"] += 1
            return deep_gemm_fp8_o_proj(
                o,
                positions,
                rope.cos_sin_cache,
                wo_a,
                identity,
                n_groups=groups,
                heads_per_group=heads_per_group,
                nope_dim=head_dim - rope_dim,
                rope_dim=rope_dim,
                o_lora_rank=args.rank,
                einsum_recipe=recipe,
                tma_aligned_scales=aligned_scales,
            )

        def wo_b(z):
            calls["wo_b"] += 1
            last["z"] = z
            return z @ final_weight

        base_o = random((args.tokens, heads, head_dim), 0.25)
        positions = torch.arange(args.tokens, device=device, dtype=torch.long) + 17
        base_before, positions_before = base_o.clone(), positions.clone()
        doc_rows = args.tokens - 2  # The final two new/query rows must never reuse.
        policy_key = f"synthetic-native-fp8-tp1-seed{args.seed}-rank{args.rank}"

        def capture(start, end):
            before = dict(calls)
            result = capture_local_z(
                base_o[start:end],
                positions[start:end],
                args.local_head_ids,
                project_z,
                policy_key=policy_key,
            )
            if (
                calls["native_project_z"] - before["native_project_z"] != 1
                or calls["wo_b"] != before["wo_b"]
            ):
                raise AssertionError(
                    "capture did not use one native wo_a and zero real wo_b"
                )
            return result

        cached = capture(0, doc_rows)
        cached_before = cached.z_off.clone()
        source_positions_before = cached.source_positions.clone()
        split = doc_rows // 2
        chunks = (capture(0, split), capture(split, doc_rows))
        chunk_snapshots = [
            (c.z_off.clone(), c.source_positions.clone()) for c in chunks
        ]
        head_index = torch.tensor(args.local_head_ids, device=device, dtype=torch.long)
        report = {
            "scope": (
                "native FP8 output-projection micro-check; not model qualification"
            ),
            "source_contract": source_contract,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "geometry": {
                "tokens": args.tokens,
                "logical_heads": heads,
                "groups": groups,
                "head_dim": head_dim,
                "rope_dim": rope_dim,
                "rank": args.rank,
                "wo_b_output_width": output_width,
                "tp": 1,
                "local_head_ids": args.local_head_ids,
            },
            "native_fp8_contract": {
                "checkpoint_weight_shape": [groups * args.rank, input_width],
                "checkpoint_scale_dtype": "torch.float8_e8m0fnu",
                "checkpoint_scale_shape": checkpoint_scale_shape,
                "postload_weight_shape": list(weight.shape),
                "postload_weight_dtype": str(weight.dtype),
                "postload_scale_shape": list(native_scale.shape),
                "postload_scale_stride": list(native_scale.stride()),
                "postload_scale_dtype": str(native_scale.dtype),
                "postload": "deepgemm_post_process_fp8_weight_block(is_bmm=True)",
                "einsum_recipe": recipe,
                "tma_aligned_scales": aligned_scales,
                "quant_block_width": 128,
                "blocks_per_logical_head": 4,
                "weight_slicing_or_dequantization": False,
            },
            "rope": {
                "implementation": "DeepseekV4ScalingRotaryEmbedding",
                "synthetic_parameters_not_checkpoint": True,
                "base": 10000,
                "factor": 16,
                "original_context": 512,
                "inverse_application": "native deep_gemm_fp8_o_proj only",
                "cached_z_re_rotated": False,
            },
            "artifact_bytes": cached.nbytes,
            "tolerances": {
                "relative_rms": args.relative_rms,
                "max_absolute": args.max_absolute,
            },
            "timing_scope": (
                "Full Python projection calls, allocation and final BF16 torch wo_b. "
                "Reuse also includes validation synchronization, CPU artifact row "
                "gathers/H2D and addition. CUDA events are not pure kernel time. "
                "Explicit warmups and validation/compilation excluded."
            ),
            "limitations": [
                "Random weights and synthetic RoPE, not a real Flash model run.",
                "Same-context decomposition only; independent chunk/position "
                "reuse remains approximate.",
                "Extra BF16 partial rounding prevents bitwise clean-row equivalence.",
                "Both masked branches execute full-width wo_a; "
                "no wo_a/KV/FFN compute savings claimed.",
                "No attention kernel, KV/indexer state, TTFT, "
                "generation quality or end-to-end speed claim.",
            ],
            "cases": [],
        }
        for case in (
            "all_document_clean",
            "dirty_boundary_and_query",
            "all_dirty",
            "multi_chunk",
        ):
            clean_ids = list(range(doc_rows))
            if case == "dirty_boundary_and_query":
                clean_ids = [row for row in clean_ids if row not in {0, doc_rows // 2}]
            elif case == "all_dirty":
                clean_ids = []
            clean = torch.tensor(clean_ids, device=device, dtype=torch.long)
            clean_cpu = clean.cpu()
            dirty_ids = [row for row in range(args.tokens) if row not in clean_ids]
            dirty = torch.tensor(dirty_ids, device=device, dtype=torch.long)
            current = base_o.clone()
            current.index_copy_(
                0, dirty, random((len(dirty_ids), heads, head_dim), 0.25)
            )
            online = current.clone()
            if clean.numel():
                masked_rows = online.index_select(0, clean)
                masked_rows.index_fill_(1, head_index, 0)
                online.index_copy_(0, clean, masked_rows)
            online_before = online.clone()
            contributions = []
            if case == "multi_chunk":
                for chunk, start, end in (
                    (chunks[0], 0, split),
                    (chunks[1], split, doc_rows),
                ):
                    rows = torch.arange(start, end, dtype=torch.long)
                    contributions.append(CachedContribution(chunk, rows, rows - start))
            elif clean_ids:
                contributions = [CachedContribution(cached, clean_cpu, clean_cpu)]

            def dense():
                return wo_b(project_z(current, positions))

            def reuse():
                return merge_cached_z_and_project(
                    online,
                    positions,
                    contributions,
                    project_z,
                    wo_b,
                    local_head_ids=args.local_head_ids,
                    policy_key=policy_key,
                )

            expected = dense()
            dense_z = last["z"].clone()
            # Keep a separate native online result to prove the cached add really
            # reaches clean rows, not just a reported cache hit/callback counter.
            online_z = project_z(online, positions)
            exact_merged_z = online_z.clone()
            for contribution in contributions:
                row_ids = contribution.clean_rows.to(device=device)
                z_off = contribution.cached.z_off.index_select(
                    0, contribution.cache_rows
                ).to(device)
                exact_merged_z.index_add_(0, row_ids, z_off)
            before = dict(calls)
            actual = reuse()
            merged_z = last["z"]
            delta_calls = {key: calls[key] - before[key] for key in calls}
            error, z_error = _error(actual, expected), _error(merged_z, dense_z)
            contracts = {
                "one_native_project_z": delta_calls["native_project_z"] == 1,
                "one_final_wo_b": delta_calls["wo_b"] == 1,
                "cached_add_matches_row_map": torch.equal(merged_z, exact_merged_z),
                "dirty_query_z_bitwise_native": torch.equal(
                    merged_z[dirty], dense_z[dirty]
                ),
                "dirty_query_output_bitwise_native": torch.equal(
                    actual[dirty], expected[dirty]
                ),
                "attention_input_unchanged": torch.equal(online, online_before),
                "source_attention_unchanged": torch.equal(base_o, base_before),
                "positions_unchanged": torch.equal(positions, positions_before),
                "cached_clean_addition_nonzero": not clean_ids
                or bool((merged_z[clean] != online_z[clean]).any()),
            }
            checks_pass = all(contracts.values()) and all(
                metric["finite"]
                and metric["relative_rms"] <= args.relative_rms
                and metric["max_absolute"] <= args.max_absolute
                for metric in (error, z_error)
            )
            case_report = {
                "case": case,
                "clean_rows": clean_ids,
                "dirty_query_rows": dirty_ids,
                "query_rows": [args.tokens - 2, args.tokens - 1],
                "chunk_contributions": len(contributions),
                "contracts": contracts,
                "output_error_vs_native_dense": error,
                "z_error_vs_native_dense": z_error,
                "dense_timing": _measure(dense, device, args.warmup, args.iterations),
                "reuse_timing": _measure(reuse, device, args.warmup, args.iterations),
                "passed": checks_pass,
            }
            report["cases"].append(case_report)
        report["immutable_state"] = {
            "fp8_weight_bytes_unchanged": torch.equal(
                weight.view(torch.uint8), weight_before
            ),
            "native_weight_scales_unchanged": torch.equal(native_scale, scale_before),
            "cached_z_unchanged": torch.equal(cached.z_off, cached_before),
            "source_positions_unchanged": torch.equal(
                cached.source_positions, source_positions_before
            ),
            "multi_chunk_artifacts_unchanged": all(
                torch.equal(chunk.z_off, snapshot[0])
                and torch.equal(chunk.source_positions, snapshot[1])
                for chunk, snapshot in zip(chunks, chunk_snapshots)
            ),
        }
        report["passed"] = all(report["immutable_state"].values()) and all(
            case["passed"] for case in report["cases"]
        )
        return report


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (
        not args.run
        or not args.device
        or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", args.device) is None
    ):
        parser.error(
            "GPU execution requires both --run and an explicit --device cuda:N"
        )
    if not 4 <= args.tokens <= 4096:
        parser.error("--tokens must be between 4 and 4096")
    if args.warmup < 1 or args.iterations < 1 or args.seed < 0:
        parser.error("--warmup and --iterations must be >=1; --seed must be >=0")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.relative_rms, args.max_absolute)
    ):
        parser.error("error tolerances must be positive and finite")
    try:
        report = run(args)
    except Exception as exc:
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
