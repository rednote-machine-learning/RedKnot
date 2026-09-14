"""Explicit CUDA micro-check: selected Triton vs pinned native FlashMLA + Torch.

Example (choose an actually free device yourself):
  python benchmarks/check_dsv4_sparse_kernel.py --run --device cuda:0

Import and --help do not import Torch/vLLM or initialize CUDA. This does not load
model weights or measure end-to-end RedKnot quality, TTFT, or cache reuse.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time


def _ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integer IDs"
        ) from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("IDs must be nonempty and unique")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="explicitly authorize GPU run"
    )
    parser.add_argument("--device", help="explicit CUDA device, e.g. cuda:0")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--kv-tokens", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--heads", type=int, choices=(64, 128), default=64)
    parser.add_argument("--rows", type=_ids, default=(17, 0, 63))
    parser.add_argument("--head-ids", type=_ids, default=(63, 0, 17))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.03)
    return parser


def _error(actual, expected) -> dict:
    import torch

    difference = actual.float() - expected.float()
    rms = difference.square().mean().sqrt()
    norm = expected.float().square().mean().sqrt()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_absolute": float(difference.abs().max()),
        "rms": float(rms),
        "relative_rms": float(rms / norm.clamp_min(1e-12)),
    }


def _native_oracle_indices(indices, kv_tokens):
    """Normalize only the native oracle copy to the conservative -1 sentinel.

    Pinned SM90's load predicate already checks 0 <= index < S. Normalization
    avoids exercising an installed native binary's positive-OOB load behavior.
    The selected kernel and Torch oracle still see the original OOB entries.
    """
    return indices.masked_fill((indices < 0) | (indices >= kv_tokens), -1)


def _measure(call, device, warmup, iterations) -> dict:
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
        "iterations": iterations,
        "cuda_event_median_ms": statistics.median(device_ms),
        "synchronized_wall_median_ms": statistics.median(wall_ms),
        "synchronized_wall_min_ms": min(wall_ms),
    }


def run(args) -> dict:
    import torch
    import triton
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd,
        is_flashmla_sparse_supported,
    )

    from vllm_redknot.dsv4_sparse import (
        selected_sparse_mla,
        selected_sparse_mla_reference,
        sparse_workload,
    )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    available, reason = is_flashmla_sparse_supported()
    if not available:
        raise RuntimeError(f"pinned native sparse FlashMLA is unavailable: {reason}")
    torch.manual_seed(args.seed)
    q = torch.randn(args.tokens, args.heads, 512, device=device, dtype=torch.bfloat16)
    kv = torch.randn(args.kv_tokens, 1, 512, device=device, dtype=torch.bfloat16)
    indices = torch.randint(
        args.kv_tokens,
        (args.tokens, 1, args.candidates),
        device=device,
        dtype=torch.int32,
    )
    lens = torch.randint(
        0, args.candidates + 1, (args.tokens,), device=device, dtype=torch.int32
    )
    sink = torch.randn(args.heads, device=device, dtype=torch.float32)
    indices[:, :, 0] = -1
    indices[:, :, 1] = args.kv_tokens + 7
    indices[:, :, 2:4] = 0  # Repeated candidates must both contribute.
    scale = 512**-0.5
    row_ids = torch.tensor(args.rows, device=device, dtype=torch.long)
    head_ids = torch.tensor(args.head_ids, device=device, dtype=torch.long)
    report = {
        "scope": "sparse attention micro-check only; not model qualification",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "shape": {"q": list(q.shape), "kv": list(kv.shape)},
        "selection": {"rows": args.rows, "head_ids": args.head_ids},
        "triton_workload": sparse_workload(
            len(args.rows), len(args.head_ids), args.candidates
        ),
        "native_workload": {
            "query_rows": len(args.rows),
            "heads_per_row": args.heads,
            "query_head_rows": len(args.rows) * args.heads,
            "minimum_supported_heads": 64,
            "selected_head_padding_is_not_a_reduction": True,
            "out_of_range_indices_normalized_to_minus_one": True,
        },
        "timing_scope": (
            "native: selected-row Q/indices/lens copies prepared before timing, all "
            "original heads; Triton: wrapper allocation, selection transfer, lens "
            "validation synchronization, and kernel included. Both read same KV. "
            "CUDA events are not pure kernel time because host validation can idle "
            "the stream. Warmup/JIT compilation excluded. No TTFT claim."
        ),
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "cases": [],
    }
    for case in ("mixed", "all_invalid", "zero_length", "infinite_sink"):
        case_indices, case_lens, case_sink = indices.clone(), lens.clone(), sink.clone()
        if case == "all_invalid":
            case_indices.fill_(-1)
            case_lens.fill_(args.candidates)
            case_sink.fill_(-float("inf"))
        elif case == "zero_length":
            case_lens.zero_()
        elif case == "infinite_sink":
            case_sink[args.head_ids[0]] = float("inf")
            if len(args.head_ids) > 1:
                case_sink[args.head_ids[1]] = -float("inf")
        if case in {"mixed", "infinite_sink"}:
            case_lens[args.rows[0]] = args.candidates
        # Only the oracle copies rows: the selected kernel addresses originals.
        native_q = q.index_select(0, row_ids).contiguous()
        native_indices = _native_oracle_indices(
            case_indices.index_select(0, row_ids), args.kv_tokens
        ).contiguous()
        native_lens = case_lens.index_select(0, row_ids).contiguous()
        native_out = torch.empty_like(native_q)

        def native():
            flash_mla_sparse_fwd(
                q=native_q,
                kv=kv,
                indices=native_indices,
                sm_scale=scale,
                attn_sink=case_sink,
                topk_length=native_lens,
                out=native_out,
            )
            return native_out

        def selected():
            return selected_sparse_mla(
                q,
                kv,
                case_indices,
                case_lens,
                case_sink,
                args.rows,
                args.head_ids,
                scale,
            )

        originals = [
            value.clone() for value in (q, kv, case_indices, case_lens, case_sink)
        ]
        actual = selected()
        expected_native = native().index_select(1, head_ids)
        expected_reference = selected_sparse_mla_reference(
            q.cpu(),
            kv.cpu(),
            case_indices.cpu(),
            case_lens.cpu(),
            case_sink.cpu(),
            args.rows,
            args.head_ids,
            scale,
        ).to(device)
        native_error = _error(actual, expected_native)
        reference_error = _error(actual, expected_reference)
        passed = True
        failures = []
        for name, expected in (
            ("native", expected_native),
            ("torch_reference", expected_reference),
        ):
            try:
                torch.testing.assert_close(
                    actual, expected, atol=args.atol, rtol=args.rtol
                )
            except AssertionError as exc:
                passed = False
                failures.append(f"{name}: {exc}")
        unchanged = all(
            torch.equal(old, new)
            for old, new in zip(
                originals, (q, kv, case_indices, case_lens, case_sink), strict=True
            )
        )
        passed = passed and unchanged
        lengths = native_lens.cpu().tolist()
        head_tiles = (len(args.head_ids) + 15) // 16
        report["cases"].append(
            {
                "name": case,
                "passed": passed,
                "inputs_unchanged": unchanged,
                "selected_candidate_lengths": lengths,
                "triton_candidate_tiles": sum((value + 63) // 64 for value in lengths)
                * head_tiles,
                "native_error": native_error,
                "torch_reference_error": reference_error,
                "failures": failures,
                "selected_timing": _measure(
                    selected, device, args.warmup, args.iterations
                ),
                "native_all_heads_timing": _measure(
                    native, device, args.warmup, args.iterations
                ),
            }
        )
    report["passed"] = all(case["passed"] for case in report["cases"])
    return report


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.run or not args.device or not args.device.startswith("cuda:"):
        parser.error("GPU execution requires --run and an explicit --device cuda:N")
    if args.tokens < 1 or args.kv_tokens < 1 or args.candidates < 128:
        parser.error("tokens/kv-tokens must be positive; candidates must be >= 128")
    if args.candidates % 128:
        parser.error("pinned SM90 native oracle requires capacity divisible by 128")
    if args.warmup < 0 or args.iterations < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("invalid warmup, iterations, or tolerances")
    if any(row < 0 or row >= args.tokens for row in args.rows):
        parser.error("selected rows must be in [0, tokens)")
    if any(head < 0 or head >= args.heads for head in args.head_ids):
        parser.error("selected head IDs must be in [0, heads)")
    try:
        report = run(args)
    except Exception as exc:
        print(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
