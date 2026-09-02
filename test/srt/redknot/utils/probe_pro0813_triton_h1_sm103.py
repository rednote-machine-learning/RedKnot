#!/usr/bin/env python3
"""Adversarial B300/SM103 oracle for Pro-0813's fused Triton H1 path.

The probe constructs the physical DeepSeek-V4 cache ABI directly: 448 FP8
E4M3 bytes, 64 BF16 RoPE values (128 bytes), and seven UE8M0 scales per
token. Every case enters the production
``triton_fp8_attention_fwd(..., force_fused_headwise=True)`` wrapper and is
checked against an independent PyTorch reference on every query row/head.

The cases deliberately make the common false-positive modes observable:

* MAIN and EXTRA use different non-affine index permutations;
* every valid prefix contains a ``-1`` sentinel while the suffix after
  ``topk_length`` contains legal poison indices;
* MAIN B256 is paired with both C4 B64 and C128 B2 physical page strides;
* C4 covers split-K H2 and H14 routes, including the production uint8
  per-head mask and per-head MAIN limits;
* attention sinks affect the output denominator, while the returned LSE is
  compared with token-only LSE, matching the fused kernel contract; and
* UE8M0 exponents vary per token/tile, so scale decoding cannot degenerate to
  the all-ones case.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


_D_NOPE = 448
_D_ROPE = 64
_D_QK = 512
_DATA_BYTES = 576
_SCALE_BYTES = 8
_TILES = 7

_WRAPPER_MODULE = "sglang.srt.layers.attention.nsa.triton_decode"
_FUSED_MODULE = (
    "sglang.srt.layers.attention.nsa.triton_decode."
    "triton_mla_kernels_decode_fused"
)
_BACKEND_MODULE = "sglang.srt.layers.attention.redknot_mla_backend"

_EXPECTED_RELATIVE_PATHS = {
    "wrapper": Path(
        "sglang/srt/layers/attention/nsa/triton_decode/__init__.py"
    ),
    "fused": Path(
        "sglang/srt/layers/attention/nsa/triton_decode/"
        "triton_mla_kernels_decode_fused.py"
    ),
    "backend": Path("sglang/srt/layers/attention/redknot_mla_backend.py"),
}
_PRODUCTION_PAGE_BYTES = {
    # block_size: (logical bytes, padded physical stride bytes)
    256: (149_504, 149_760),
    64: (37_376, 37_440),
    2: (1_168, 1_728),
}

_MAX_OUTPUT_ABS = 0.02
_MIN_OUTPUT_COSINE = 0.999
_MAX_LSE_ABS = 0.02


@dataclass(frozen=True)
class _Scenario:
    name: str
    batch: int
    sequence: int
    heads: int
    main_num_blocks: int
    main_block_size: int
    main_index_width: int
    main_lengths: tuple[int, ...]
    extra_num_blocks: int
    extra_block_size: int
    extra_index_width: int
    extra_lengths: tuple[int, ...]
    expected_split_k: int
    sink_values: tuple[float, ...]
    extra_head_mask: tuple[int, ...] | None = None
    main_head_lengths: tuple[int, ...] | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenario.name for scenario in _scenarios()),
        help=(
            "run only this scenario; repeat to select multiple scenarios "
            "(the default runs the complete oracle)"
        ),
    )
    parser.add_argument(
        "--expected-source-root",
        default=os.environ.get("REDKNOT_PROBE_EXPECTED_SOURCE_ROOT"),
        help=(
            "optional resolved Python source root (for example "
            "/workspace/RedKnot_Pro0813/python)"
        ),
    )
    parser.add_argument(
        "--expected-wrapper-sha256",
        default=os.environ.get("REDKNOT_PROBE_EXPECTED_WRAPPER_SHA256"),
    )
    parser.add_argument(
        "--expected-fused-sha256",
        default=os.environ.get("REDKNOT_PROBE_EXPECTED_FUSED_SHA256"),
    )
    parser.add_argument(
        "--expected-backend-sha256",
        default=os.environ.get("REDKNOT_PROBE_EXPECTED_BACKEND_SHA256"),
    )
    return parser.parse_args()


def _require_b300() -> dict:
    if os.environ.get("REDKNOT_PRO0813_B300_TRITON_H1_PROBE") != "1":
        raise RuntimeError(
            "set REDKNOT_PRO0813_B300_TRITON_H1_PROBE=1 to run this GPU probe"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "B300" not in name or capability != (10, 3):
        raise RuntimeError(
            f"expected NVIDIA B300 compute capability 10.3, got {name!r} "
            f"{capability!r}"
        )
    return {
        "device_name": name,
        "compute_capability": list(capability),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _module_source_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"cannot resolve source for {module_name}")
    path = Path(spec.origin).resolve()
    if not path.is_file():
        raise RuntimeError(f"module source is not a file: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(digest) != 64:
        raise AssertionError(f"invalid SHA256 result for {path}")
    return digest


def _validate_expected_sha(label: str, actual: str, expected: str | None) -> None:
    if expected is None:
        return
    normalized = expected.strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"expected {label} SHA256 must be 64 lowercase hex digits")
    if actual != normalized:
        raise RuntimeError(
            f"{label} source SHA256 mismatch: actual={actual} expected={normalized}"
        )


def _source_identity(args: argparse.Namespace) -> tuple[dict, Callable]:
    wrapper_module = importlib.import_module(_WRAPPER_MODULE)
    entrypoint = getattr(wrapper_module, "triton_fp8_attention_fwd", None)
    if not callable(entrypoint):
        raise RuntimeError(f"{_WRAPPER_MODULE} has no callable Triton entry point")

    paths = {
        "wrapper": _module_source_path(_WRAPPER_MODULE),
        "fused": _module_source_path(_FUSED_MODULE),
        "backend": _module_source_path(_BACKEND_MODULE),
    }
    package_spec = importlib.util.find_spec("sglang")
    package_locations = (
        () if package_spec is None else package_spec.submodule_search_locations
    )
    roots = []
    for location in package_locations or ():
        package_root = Path(location).resolve()
        try:
            paths["wrapper"].relative_to(package_root)
        except ValueError:
            continue
        roots.append(package_root.parent)
    if len(roots) != 1:
        raise RuntimeError(
            "cannot identify one Python source root for the imported wrapper: "
            f"candidates={[str(root) for root in roots]}"
        )
    source_root = roots[0]

    relative_paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for label, path in paths.items():
        try:
            relative = path.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError(
                f"{label} source {path} escapes wrapper source root {source_root}"
            ) from exc
        expected_relative = _EXPECTED_RELATIVE_PATHS[label]
        if relative != expected_relative:
            raise RuntimeError(
                f"unexpected {label} source under {source_root}: "
                f"{relative} != {expected_relative}"
            )
        relative_paths[label] = str(relative)
        hashes[label] = _sha256(path)

    if args.expected_source_root is not None:
        expected_root = Path(args.expected_source_root).resolve()
        if source_root != expected_root:
            raise RuntimeError(
                f"Python source root mismatch: actual={source_root} "
                f"expected={expected_root}"
            )
    _validate_expected_sha(
        "wrapper", hashes["wrapper"], args.expected_wrapper_sha256
    )
    _validate_expected_sha("fused", hashes["fused"], args.expected_fused_sha256)
    _validate_expected_sha(
        "backend", hashes["backend"], args.expected_backend_sha256
    )

    identity = {
        "source_root": str(source_root),
        "paths": {label: str(path) for label, path in paths.items()},
        "relative_paths": relative_paths,
        "sha256": hashes,
        "expected_source_root": args.expected_source_root,
        "expected_sha256_supplied": {
            "wrapper": args.expected_wrapper_sha256 is not None,
            "fused": args.expected_fused_sha256 is not None,
            "backend": args.expected_backend_sha256 is not None,
        },
    }
    print(
        "REDKNOT_TRITON_H1_SOURCE_IDENTITY "
        + json.dumps(identity, sort_keys=True),
        flush=True,
    )
    return identity, entrypoint


def _pack_cache(
    *,
    num_blocks: int,
    block_size: int,
    generator: torch.Generator,
    scope_id: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    device = torch.device("cuda:0")
    raw_nope = (
        torch.randn(
            num_blocks,
            block_size,
            _D_NOPE,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.15625
    )
    # Make mask/length/sink mistakes numerically loud instead of relying on a
    # chance random average. MAIN and EXTRA have opposite signed marker
    # channels; token zero is a large sentinel-clamp trap, and the tail is a
    # legal-poison signature band. The remaining dimensions stay random.
    scope_sign = 1.0 if scope_id % 2 else -1.0
    raw_nope[..., 0] = scope_sign * 0.75
    raw_nope[0, 0, 1:9] = scope_sign * 4.0
    raw_nope_flat = raw_nope.view(-1, _D_NOPE)
    poison_band = min(256, raw_nope_flat.shape[0])
    raw_nope_flat[-poison_band:, 9:17] = scope_sign * 2.0
    nope_fp8 = raw_nope.to(torch.float8_e4m3fn)
    rope = (
        torch.randn(
            num_blocks,
            block_size,
            _D_ROPE,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.109375
    ).to(torch.bfloat16)

    block_axis = torch.arange(num_blocks, device=device)[:, None, None]
    token_axis = torch.arange(block_size, device=device)[None, :, None]
    tile_axis = torch.arange(_TILES, device=device)[None, None, :]
    # Exponents 125..129 yield exact power-of-two scales 1/4..4. Varying all
    # three axes catches code that reads the wrong token, tile, or scale page.
    scale_exponents = (
        125
        + (
            block_axis * 3
            + token_axis * 2
            + tile_axis * 4
            + scope_id * 3
            + (token_axis % 7) * (tile_axis + 1)
        )
        % 5
    ).to(torch.uint8)
    unique_exponents = sorted(
        int(value) for value in scale_exponents.unique().tolist()
    )
    if unique_exponents == [127] or len(unique_exponents) < 4:
        raise AssertionError(f"degenerate UE8M0 exponent pattern: {unique_exponents}")

    logical_bytes = block_size * (_DATA_BYTES + _SCALE_BYTES)
    # DeepSeekV4SingleKVPool pads each physical page to a 576-byte boundary.
    # The exposed [page, page_size, 1, 584] view retains that padded stride(0).
    padded_bytes = ((logical_bytes + _DATA_BYTES - 1) // _DATA_BYTES) * _DATA_BYTES
    expected_page_bytes = _PRODUCTION_PAGE_BYTES.get(block_size)
    if expected_page_bytes is None:
        raise AssertionError(f"probe block size is not a production geometry: {block_size}")
    if (logical_bytes, padded_bytes) != expected_page_bytes:
        raise AssertionError(
            "production page byte geometry mismatch: "
            f"block_size={block_size} actual={(logical_bytes, padded_bytes)} "
            f"expected={expected_page_bytes}"
        )
    packed_storage = torch.zeros(
        (num_blocks, padded_bytes), dtype=torch.uint8, device=device
    )
    for block in range(num_blocks):
        data = packed_storage[block, : block_size * _DATA_BYTES].view(
            block_size, _DATA_BYTES
        )
        data[:, :_D_NOPE].copy_(nope_fp8[block].contiguous().view(torch.uint8))
        data[:, _D_NOPE:].copy_(
            rope[block].contiguous().view(torch.uint8).view(block_size, -1)
        )
        scales = packed_storage[
            block, block_size * _DATA_BYTES : logical_bytes
        ].view(block_size, _SCALE_BYTES)
        scales[:, :_TILES].copy_(scale_exponents[block])
        # Byte seven is ABI padding and intentionally differs from a scale.
        scales[:, _TILES] = 0

    scale_values = torch.exp2(scale_exponents.float() - 127.0).to(torch.bfloat16)
    nope_reference = (
        nope_fp8.to(torch.bfloat16).view(
            num_blocks, block_size, _TILES, _D_NOPE // _TILES
        )
        * scale_values.unsqueeze(-1)
    ).reshape(num_blocks, block_size, _D_NOPE)
    reference = torch.cat((nope_reference, rope), dim=-1).contiguous()
    packed_view = packed_storage[:, :logical_bytes].view(
        num_blocks, block_size, 1, _DATA_BYTES + _SCALE_BYTES
    )
    if packed_view.stride(0) != padded_bytes:
        raise AssertionError(
            f"physical page stride mismatch: {packed_view.stride(0)} != {padded_bytes}"
        )
    metadata = {
        "num_blocks": num_blocks,
        "block_size": block_size,
        "population": num_blocks * block_size,
        "logical_page_bytes": logical_bytes,
        "physical_page_stride_bytes": packed_view.stride(0),
        "view_shape": list(packed_view.shape),
        "ue8m0_exponents": unique_exponents,
    }
    return packed_view, reference, metadata


def _make_indices(
    *,
    batch: int,
    sequence: int,
    width: int,
    population: int,
    valid_lengths: tuple[int, ...],
    scope: str,
) -> tuple[torch.Tensor, dict]:
    if len(valid_lengths) != batch:
        raise ValueError(f"{scope}: valid length count must equal batch")
    if population <= 1:
        raise ValueError(f"{scope}: population must be greater than one")
    rows = batch * sequence
    result = torch.empty((rows, width), dtype=torch.int32)
    sentinel_positions: list[int] = []
    poison_samples: list[list[int]] = []
    if scope == "main":
        quadratic, linear, row_term, cross, modulus, offset = (
            13,
            17,
            29,
            7,
            5,
            3,
        )
    elif scope == "extra":
        quadratic, linear, row_term, cross, modulus, offset = (
            19,
            5,
            31,
            11,
            7,
            23,
        )
    else:
        raise ValueError(f"unknown index scope {scope!r}")

    for row in range(rows):
        length = int(valid_lengths[row // sequence])
        if length < 16 or length >= width:
            raise ValueError(
                f"{scope}: require 16 <= valid length < width, got {length}/{width}"
            )
        for position in range(width):
            value = (
                quadratic * position * position
                + linear * position
                + row_term * row
                + cross * (position % modulus) * (row + 1)
                + offset
            ) % population
            result[row, position] = value

        # A valid, high-impact suffix makes ignoring topk_length observably
        # wrong. It contains no -1 and intentionally follows another
        # non-affine sequence rather than padding with a constant.
        for position in range(length, width):
            suffix_position = position - length
            result[row, position] = (
                population
                - 1
                - row * (17 if scope == "main" else 19)
                - suffix_position * suffix_position * (5 if scope == "main" else 9)
                - suffix_position * (3 if scope == "main" else 13)
            ) % population

        sentinel = 2 + (row * (3 if scope == "main" else 5)) % 11
        if sentinel >= length:
            raise AssertionError(f"{scope}: in-prefix sentinel escaped valid length")
        result[row, sentinel] = -1
        sentinel_positions.append(sentinel)
        poison_samples.append(result[row, length : length + 4].tolist())

        prefix = result[row, :length]
        suffix = result[row, length:]
        if int((prefix == -1).sum().item()) != 1:
            raise AssertionError(f"{scope}: expected exactly one in-prefix -1")
        if bool((suffix < 0).any().item()) or bool(
            (suffix >= population).any().item()
        ):
            raise AssertionError(f"{scope}: suffix poison must contain legal indices")

    metadata = {
        "recipe": scope,
        "non_affine": True,
        "coefficients": {
            "quadratic": quadratic,
            "linear": linear,
            "row": row_term,
            "cross": cross,
            "modulus_term": modulus,
            "offset": offset,
        },
        "width": width,
        "population": population,
        "valid_lengths_by_batch": list(valid_lengths),
        "sentinel_positions_by_row": sentinel_positions,
        "legal_poison_samples_by_row": poison_samples,
    }
    return result.to(device="cuda:0"), metadata


def _reference(
    *,
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_lengths: torch.Tensor,
    scale: float,
    extra_cache: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    extra_head_mask: torch.Tensor | None,
    main_head_lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, heads, _ = q.shape
    q_flat = q.reshape(batch * sequence, heads, _D_QK).float()
    out = torch.empty_like(q_flat)
    token_lse = torch.empty(
        (batch * sequence, heads), dtype=torch.float32, device=q.device
    )
    main_flat = main_cache.reshape(-1, _D_QK).float()
    extra_flat = extra_cache.reshape(-1, _D_QK).float()

    for row in range(batch * sequence):
        batch_index = row // sequence
        batch_main_length = int(main_lengths[batch_index].item())
        batch_extra_length = int(extra_lengths[batch_index].item())
        for head in range(heads):
            effective_main_length = batch_main_length
            if main_head_lengths is not None:
                effective_main_length = min(
                    effective_main_length, int(main_head_lengths[head].item())
                )
            main_selection = main_indices[row, :effective_main_length].to(torch.int64)
            main_selection = main_selection[main_selection >= 0]
            selected = [main_flat.index_select(0, main_selection)]

            use_extra = (
                extra_head_mask is None or int(extra_head_mask[head].item()) != 0
            )
            if use_extra:
                extra_selection = extra_indices[
                    row, :batch_extra_length
                ].to(torch.int64)
                extra_selection = extra_selection[extra_selection >= 0]
                selected.append(extra_flat.index_select(0, extra_selection))

            kv = torch.cat(selected, dim=0)
            if kv.shape[0] == 0:
                raise AssertionError(f"reference row={row} head={head} has no KV tokens")
            scores = torch.mv(kv, q_flat[row, head]) * scale
            current_token_lse = torch.logsumexp(scores, dim=0)
            # The sink is a zero-value virtual token: it changes the output
            # denominator but not the numerator. The Triton API returns the
            # token-only LSE rather than logaddexp(token_lse, sink).
            log_denominator = torch.logaddexp(current_token_lse, attn_sink[head])
            token_weights = torch.exp(scores - log_denominator)
            out[row, head] = torch.mv(kv.transpose(0, 1), token_weights)
            token_lse[row, head] = current_token_lse

    return (
        out.to(torch.bfloat16).view(batch, sequence, heads, _D_QK),
        token_lse.view(batch, sequence, heads),
    )


def _finite_summary(name: str, value: torch.Tensor) -> dict:
    value_f = value.float()
    finite = torch.isfinite(value_f)
    result = {
        "name": name,
        "numel": value_f.numel(),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(value_f).sum().item()),
        "positive_inf": int(torch.isposinf(value_f).sum().item()),
        "negative_inf": int(torch.isneginf(value_f).sum().item()),
    }
    if bool(finite.any().item()):
        result["finite_min"] = float(value_f[finite].min().item())
        result["finite_max"] = float(value_f[finite].max().item())
    return result


def _row_head_metrics(
    *,
    actual_output: torch.Tensor,
    expected_output: torch.Tensor,
    actual_lse: torch.Tensor,
    expected_lse: torch.Tensor,
) -> dict:
    if actual_output.shape != expected_output.shape:
        raise AssertionError(
            f"output shape mismatch {actual_output.shape} != {expected_output.shape}"
        )
    if actual_lse.shape != expected_lse.shape:
        raise AssertionError(
            f"LSE shape mismatch {actual_lse.shape} != {expected_lse.shape}"
        )
    batch, sequence, heads, _ = actual_output.shape
    actual_flat = actual_output.float().reshape(batch * sequence, heads, _D_QK)
    expected_flat = expected_output.float().reshape(batch * sequence, heads, _D_QK)
    max_abs = (actual_flat - expected_flat).abs().amax(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(
        actual_flat, expected_flat, dim=-1
    )
    lse_abs = (
        actual_lse.float().reshape(batch * sequence, heads)
        - expected_lse.float().reshape(batch * sequence, heads)
    ).abs()

    finite = (
        torch.isfinite(max_abs) & torch.isfinite(cosine) & torch.isfinite(lse_abs)
    )
    passed = (
        finite
        & (max_abs <= _MAX_OUTPUT_ABS)
        & (cosine >= _MIN_OUTPUT_COSINE)
        & (lse_abs <= _MAX_LSE_ABS)
    )
    row_head: list[dict] = []
    for row in range(batch * sequence):
        for head in range(heads):
            row_head.append(
                {
                    "row": row,
                    "batch_index": row // sequence,
                    "sequence_index": row % sequence,
                    "head": head,
                    "output_max_abs": float(max_abs[row, head].item()),
                    "output_cosine": float(cosine[row, head].item()),
                    "token_lse_abs": float(lse_abs[row, head].item()),
                    "pass": bool(passed[row, head].item()),
                }
            )
    if not bool(passed.all().item()):
        failures = [entry for entry in row_head if not entry["pass"]]
        raise AssertionError(
            "per-row/head numerical mismatch; first failures="
            + json.dumps(failures[:8], sort_keys=True)
        )
    return {
        "limits": {
            "output_max_abs": _MAX_OUTPUT_ABS,
            "output_min_cosine": _MIN_OUTPUT_COSINE,
            "token_lse_max_abs": _MAX_LSE_ABS,
        },
        "worst_output_max_abs": float(max_abs.max().item()),
        "minimum_output_cosine": float(cosine.min().item()),
        "worst_token_lse_abs": float(lse_abs.max().item()),
        "row_head": row_head,
        "pass": True,
    }


def _validate_route_geometry(scenario: _Scenario) -> dict:
    total_tokens = scenario.batch * scenario.sequence
    total_topk = scenario.main_index_width + scenario.extra_index_width
    small_batch_route = total_tokens <= 8 and (
        scenario.heads >= 128 or total_topk >= 1024
    )
    h64_large_topk_route = (
        scenario.heads <= 64
        and total_topk >= 1024
        and 8 < total_tokens <= 128
    )
    large_topk_route = total_tokens > 64 and total_topk >= 2048
    large_head_route = (
        scenario.heads > 64 and total_tokens > 8 and total_topk >= 256
    )
    if small_batch_route:
        computed_split_k = 8 if total_topk >= 512 and total_tokens <= 4 else 4
    elif large_head_route:
        computed_split_k = 4 if total_topk >= 512 else 2
    elif h64_large_topk_route:
        computed_split_k = 2
    elif large_topk_route:
        # None of the fixed probe cases uses this heuristic branch.
        computed_split_k = -1
    else:
        computed_split_k = 0
    if computed_split_k != scenario.expected_split_k:
        raise AssertionError(
            f"{scenario.name}: route geometry produced split_k={computed_split_k}, "
            f"expected {scenario.expected_split_k}"
        )
    return {
        "total_tokens": total_tokens,
        "total_topk_width": total_topk,
        "expected_split_k": computed_split_k,
        "small_batch_route": small_batch_route,
        "h64_large_topk_route": h64_large_topk_route,
        "large_topk_route": large_topk_route,
        "large_head_route": large_head_route,
    }


def _run_scenario(
    scenario: _Scenario,
    *,
    entrypoint: Callable,
    scenario_index: int,
) -> dict:
    if scenario.batch == scenario.sequence:
        raise AssertionError(f"{scenario.name}: batch/sequence must be asymmetric")
    if len(scenario.main_lengths) != scenario.batch:
        raise AssertionError(f"{scenario.name}: invalid MAIN length vector")
    if len(scenario.extra_lengths) != scenario.batch:
        raise AssertionError(f"{scenario.name}: invalid EXTRA length vector")
    if len(scenario.sink_values) != scenario.heads:
        raise AssertionError(f"{scenario.name}: invalid sink vector")
    if (
        scenario.extra_head_mask is not None
        and len(scenario.extra_head_mask) != scenario.heads
    ):
        raise AssertionError(f"{scenario.name}: invalid uint8 mask vector")
    if (
        scenario.main_head_lengths is not None
        and len(scenario.main_head_lengths) != scenario.heads
    ):
        raise AssertionError(f"{scenario.name}: invalid per-head MAIN vector")

    route = _validate_route_geometry(scenario)
    scenario_seed = 20260813 + scenario_index
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(scenario_seed)
    main_packed, main_reference, main_cache_metadata = _pack_cache(
        num_blocks=scenario.main_num_blocks,
        block_size=scenario.main_block_size,
        generator=generator,
        scope_id=scenario_index * 2 + 1,
    )
    extra_packed, extra_reference, extra_cache_metadata = _pack_cache(
        num_blocks=scenario.extra_num_blocks,
        block_size=scenario.extra_block_size,
        generator=generator,
        scope_id=scenario_index * 2 + 2,
    )
    main_indices, main_index_metadata = _make_indices(
        batch=scenario.batch,
        sequence=scenario.sequence,
        width=scenario.main_index_width,
        population=main_cache_metadata["population"],
        valid_lengths=scenario.main_lengths,
        scope="main",
    )
    extra_indices, extra_index_metadata = _make_indices(
        batch=scenario.batch,
        sequence=scenario.sequence,
        width=scenario.extra_index_width,
        population=extra_cache_metadata["population"],
        valid_lengths=scenario.extra_lengths,
        scope="extra",
    )
    common_width = min(scenario.main_index_width, scenario.extra_index_width)
    if torch.equal(main_indices[:, :common_width], extra_indices[:, :common_width]):
        raise AssertionError(f"{scenario.name}: MAIN and EXTRA indices are identical")

    q = (
        torch.randn(
            scenario.batch,
            scenario.sequence,
            scenario.heads,
            _D_QK,
            device="cuda:0",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.0546875
    ).to(torch.bfloat16)
    scale = _D_QK**-0.5
    main_lengths = torch.tensor(
        scenario.main_lengths, dtype=torch.int32, device="cuda:0"
    )
    extra_lengths = torch.tensor(
        scenario.extra_lengths, dtype=torch.int32, device="cuda:0"
    )
    attn_sink = torch.tensor(
        scenario.sink_values, dtype=torch.float32, device="cuda:0"
    )
    extra_head_mask = (
        None
        if scenario.extra_head_mask is None
        else torch.tensor(
            scenario.extra_head_mask, dtype=torch.uint8, device="cuda:0"
        )
    )
    main_head_lengths = (
        None
        if scenario.main_head_lengths is None
        else torch.tensor(
            scenario.main_head_lengths, dtype=torch.int32, device="cuda:0"
        )
    )
    if extra_head_mask is not None and extra_head_mask.dtype != torch.uint8:
        raise AssertionError("production per-head mask must be uint8")

    actual_output, actual_lse = entrypoint(
        q=q,
        k_cache=main_packed,
        head_dim_v=_D_QK,
        softmax_scale=scale,
        indices=main_indices.view(
            scenario.batch, scenario.sequence, scenario.main_index_width
        ),
        topk_length=main_lengths,
        attn_sink=attn_sink,
        extra_k_cache=extra_packed,
        extra_indices_in_kvcache=extra_indices.view(
            scenario.batch, scenario.sequence, scenario.extra_index_width
        ),
        extra_topk_length=extra_lengths,
        extra_head_mask=extra_head_mask,
        main_head_lengths=main_head_lengths,
        force_fused_headwise=True,
    )
    torch.cuda.synchronize()
    expected_output, expected_lse = _reference(
        q=q,
        main_cache=main_reference,
        main_indices=main_indices,
        main_lengths=main_lengths,
        scale=scale,
        extra_cache=extra_reference,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=attn_sink,
        extra_head_mask=extra_head_mask,
        main_head_lengths=main_head_lengths,
    )

    diagnostics = [
        _finite_summary(name, value)
        for name, value in (
            ("main_reference", main_reference),
            ("extra_reference", extra_reference),
            ("q", q),
            ("actual_output", actual_output),
            ("expected_output", expected_output),
            ("actual_token_lse", actual_lse),
            ("expected_token_lse", expected_lse),
        )
    ]
    if any(item["finite"] != item["numel"] for item in diagnostics):
        raise AssertionError(
            f"{scenario.name}: non-finite diagnostic "
            + json.dumps(diagnostics, sort_keys=True)
        )
    metrics = _row_head_metrics(
        actual_output=actual_output,
        expected_output=expected_output,
        actual_lse=actual_lse,
        expected_lse=expected_lse,
    )
    report = {
        "name": scenario.name,
        "canonical_scenario_index": scenario_index,
        "scenario_seed": scenario_seed,
        "geometry": {
            "batch": scenario.batch,
            "sequence": scenario.sequence,
            "heads": scenario.heads,
            "head_dim": _D_QK,
            "main": main_cache_metadata,
            "extra": extra_cache_metadata,
            "main_index_width": scenario.main_index_width,
            "extra_index_width": scenario.extra_index_width,
            "main_lengths": list(scenario.main_lengths),
            "extra_lengths": list(scenario.extra_lengths),
            "extra_head_mask_dtype": (
                None if extra_head_mask is None else str(extra_head_mask.dtype)
            ),
            "extra_head_mask": (
                None if extra_head_mask is None else extra_head_mask.tolist()
            ),
            "main_head_lengths": (
                None if main_head_lengths is None else main_head_lengths.tolist()
            ),
            "attn_sink": attn_sink.tolist(),
            "route": route,
        },
        "index_adversary": {
            "main": main_index_metadata,
            "extra": extra_index_metadata,
            "main_extra_nonidentical": True,
        },
        "contract": {
            "force_fused_headwise": True,
            "sink_output_denominator": True,
            "returned_lse": "token_only",
        },
        "finite_diagnostics": diagnostics,
        "metrics": metrics,
        "pass": True,
    }
    print(
        "REDKNOT_TRITON_H1_SCENARIO " + json.dumps(report, sort_keys=True),
        flush=True,
    )
    return report


def _scenarios() -> tuple[_Scenario, ...]:
    return (
        _Scenario(
            name="pro0813_main_b256_c4_b64_splitk_h2_full_scope",
            batch=2,
            sequence=3,
            heads=2,
            main_num_blocks=3,
            main_block_size=256,
            main_index_width=512,
            main_lengths=(137, 229),
            extra_num_blocks=6,
            extra_block_size=64,
            extra_index_width=512,
            extra_lengths=(83, 173),
            expected_split_k=4,
            sink_values=(5.15, 6.10),
        ),
        _Scenario(
            name="pro0813_main_b256_c4_b64_splitk_h14_headwise",
            batch=3,
            sequence=4,
            heads=14,
            main_num_blocks=3,
            main_block_size=256,
            main_index_width=512,
            main_lengths=(101, 223, 317),
            extra_num_blocks=6,
            extra_block_size=64,
            extra_index_width=512,
            extra_lengths=(61, 139, 251),
            expected_split_k=2,
            sink_values=(
                4.10,
                4.28,
                4.46,
                4.64,
                4.82,
                5.00,
                5.18,
                5.36,
                5.54,
                5.72,
                5.90,
                6.08,
                6.26,
                6.44,
            ),
            extra_head_mask=(0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0),
            main_head_lengths=(
                17,
                31,
                47,
                63,
                89,
                113,
                149,
                191,
                257,
                333,
                401,
                127,
                271,
                455,
            ),
        ),
        _Scenario(
            name="pro0813_main_b256_c128_b2_h14_full_scope",
            batch=2,
            sequence=5,
            heads=14,
            main_num_blocks=3,
            main_block_size=256,
            main_index_width=512,
            main_lengths=(173, 347),
            extra_num_blocks=32,
            extra_block_size=2,
            extra_index_width=64,
            extra_lengths=(17, 49),
            expected_split_k=0,
            sink_values=(
                4.35,
                4.53,
                4.71,
                4.89,
                5.07,
                5.25,
                5.43,
                5.61,
                5.79,
                5.97,
                6.15,
                6.33,
                6.51,
                6.69,
            ),
        ),
    )


def _select_scenarios(
    requested: list[str] | None,
    available: tuple[_Scenario, ...],
) -> tuple[_Scenario, ...]:
    """Return a fail-closed, order-preserving scenario selection."""
    if requested is None:
        return available
    if len(requested) != len(set(requested)):
        raise ValueError("each --scenario selection must be unique")
    by_name = {scenario.name: scenario for scenario in available}
    if len(by_name) != len(available):
        raise RuntimeError("probe scenario names must be unique")
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"unknown probe scenarios: {unknown}")
    if not requested:
        raise ValueError("at least one probe scenario must be selected")
    return tuple(by_name[name] for name in requested)


def main() -> None:
    args = _parse_args()
    runtime = _require_b300()
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    source_identity, entrypoint = _source_identity(args)
    available_scenarios = _scenarios()
    selected_scenarios = _select_scenarios(args.scenario, available_scenarios)
    scenario_indices = {
        scenario.name: index
        for index, scenario in enumerate(available_scenarios)
    }

    scenario_reports = [
        _run_scenario(
            scenario,
            entrypoint=entrypoint,
            scenario_index=scenario_indices[scenario.name],
        )
        for scenario in selected_scenarios
    ]
    report = {
        "schema": "redknot_pro0813_triton_h1_sm103_probe_v2",
        **runtime,
        "source_identity": source_identity,
        "scenarios": scenario_reports,
        "scenario_count": len(scenario_reports),
        "available_scenario_count": len(available_scenarios),
        "selected_scenario_names": [
            scenario.name for scenario in selected_scenarios
        ],
        "pass": True,
    }
    print("REDKNOT_TRITON_H1_RESULT " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
