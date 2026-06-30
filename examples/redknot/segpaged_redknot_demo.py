#!/usr/bin/env python3
"""Reproducible RedKnot / SegPagedAttention demo.

This single file consolidates the demo experiments used by the paper's
evaluation section. It intentionally does not copy or vendor the SGLang
framework; it lives under the SGLang fork and only depends on PyTorch plus
the already-built ``sgl_kernel.flash_attn`` package.

What it measures
----------------
We compare three ways to implement the same DuoAttention-like policy on a
Qwen3-32B-class attention shape (Hq=32, Hkv=8, D=128, bf16):

  A1_manual:
      Dense KV storage [B, Hkv, L, D] + manual matmul + (-inf) mask + softmax.
      This mirrors the slow research path in __REDKNOT_V02__/redknot/core.py.

  A2_sdpa_mask:
      Dense KV storage + torch.scaled_dot_product_attention(attn_mask=...).
      On PyTorch 2.9.1 this falls off the FlashAttention path.

  B_segpaged_fused:
      Per-head ragged KV (retrieval heads keep full L; streaming heads keep
      sink+recent) + one fused FA-3 varlen call per layer.

The goal is to show that the odd throughput behavior in the experiments is
an engine/layout issue, not an algorithmic issue: once the physical KV layout
matches the per-head sparsity, prefill speedups grow with context length and
decode becomes both faster and more stable.

Example
-------
  CUDA_VISIBLE_DEVICES=0 python examples/redknot/segpaged_redknot_demo.py \
      --mode smoke --output /tmp/redknot_demo.json

  CUDA_VISIBLE_DEVICES=0 python examples/redknot/segpaged_redknot_demo.py \
      --mode paper --output /tmp/redknot_paper.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

# Set a default before importing torch. Users can override from the shell.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func, is_fa3_supported
except Exception as exc:  # pragma: no cover - environment-specific
    flash_attn_varlen_func = None

    def is_fa3_supported() -> bool:
        return False

    _FA3_IMPORT_ERROR = repr(exc)
else:
    _FA3_IMPORT_ERROR = ""


PathName = Literal["A1_manual", "A2_sdpa_mask", "B_segpaged_fused"]
Workload = Literal["decode", "prefill"]


@dataclass
class DemoConfig:
    hq: int = 32
    hkv: int = 8
    head_dim: int = 128
    sink: int = 64
    recent: int = 256
    retrieval_heads: tuple[int, ...] = (0, 1, 2, 3)
    dtype: str = "bf16"
    seed: int = 7

    @property
    def gqa_group(self) -> int:
        assert self.hq % self.hkv == 0
        return self.hq // self.hkv

    @property
    def streaming_len(self) -> int:
        return self.sink + self.recent


@dataclass
class BenchResult:
    path: str
    workload: str
    L: int
    batch: int
    q_len: int
    layers: int
    warmup: int
    iters: int
    p50_ms: float
    min_ms: float
    mean_ms: float
    std_ms: float
    cov_pct: float
    notes: str = "ok"


@dataclass
class QualityResult:
    L: int
    q_len: int
    cos_a1_a2: float
    cos_a1_b: float
    cos_a2_b: float


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")
    return torch.device("cuda:0")


def cuda_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "fa3_supported": bool(is_fa3_supported()),
        "fa3_import_error": _FA3_IMPORT_ERROR,
    }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dense_kv(
    cfg: DemoConfig, L: int, batch: int, *, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = device()
    dtype = torch_dtype(cfg.dtype)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=dev).manual_seed(seed)
    k = torch.randn(
        batch, cfg.hkv, L, cfg.head_dim, device=dev, dtype=dtype, generator=generator
    )
    v = torch.randn(
        batch, cfg.hkv, L, cfg.head_dim, device=dev, dtype=dtype, generator=generator
    )
    return k, v


def build_queries(
    cfg: DemoConfig, q_len: int, batch: int, *, seed: int | None = None
) -> torch.Tensor:
    dev = device()
    dtype = torch_dtype(cfg.dtype)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=dev).manual_seed(seed)
    return torch.randn(
        batch, cfg.hq, q_len, cfg.head_dim, device=dev, dtype=dtype, generator=generator
    )


def dense_to_ragged_views(
    cfg: DemoConfig, k_dense: torch.Tensor, v_dense: torch.Tensor, L: int
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Convert dense KV to per-sample, per-head ragged views.

    Retrieval heads keep full L. Streaming heads keep [0:sink) plus
    [L-recent:L]. The content is exactly the visible subset used by the mask
    paths, so quality comparisons are meaningful.
    """
    batch = k_dense.shape[0]
    output: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    retrieval = set(cfg.retrieval_heads)
    for b in range(batch):
        views: list[tuple[torch.Tensor, torch.Tensor]] = []
        for h in range(cfg.hkv):
            if h in retrieval:
                k_h = k_dense[b, h, :, :]
                v_h = v_dense[b, h, :, :]
            else:
                k_h = torch.cat(
                    [k_dense[b, h, : cfg.sink, :], k_dense[b, h, L - cfg.recent :, :]],
                    dim=0,
                )
                v_h = torch.cat(
                    [v_dense[b, h, : cfg.sink, :], v_dense[b, h, L - cfg.recent :, :]],
                    dim=0,
                )
            views.append((k_h.contiguous(), v_h.contiguous()))
        output.append(views)
    return output


def _duo_mask(cfg: DemoConfig, q_len: int, L: int) -> torch.Tensor:
    """Build [Hq, q_len, L] bool mask for retrieval/full + streaming heads."""
    dev = device()
    j_idx = torch.arange(L, device=dev)
    i_idx = torch.arange(q_len, device=dev) + (L - q_len)
    causal = j_idx.unsqueeze(0) <= i_idx.unsqueeze(1)
    streaming = (j_idx < cfg.sink) | (j_idx >= L - cfg.recent)
    streaming_mask = causal & streaming.unsqueeze(0)

    head_mask = torch.empty(cfg.hq, q_len, L, dtype=torch.bool, device=dev)
    retrieval = set(cfg.retrieval_heads)
    for h in range(cfg.hkv):
        mask = causal if h in retrieval else streaming_mask
        for g in range(cfg.gqa_group):
            head_mask[h * cfg.gqa_group + g] = mask
    return head_mask


def path_a1_manual(
    cfg: DemoConfig, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L: int
) -> torch.Tensor:
    """Dense KV + explicit mask + matmul/softmax/matmul."""
    k_rep = k.repeat_interleave(cfg.gqa_group, dim=1)
    v_rep = v.repeat_interleave(cfg.gqa_group, dim=1)
    head_mask = _duo_mask(cfg, q.shape[2], L)
    add_mask = torch.where(head_mask, 0.0, float("-inf")).to(q.dtype).unsqueeze(0)
    if q.shape[0] > 1:
        add_mask = add_mask.expand(q.shape[0], -1, -1, -1)

    scores = torch.matmul(q, k_rep.transpose(2, 3)) * (1.0 / math.sqrt(cfg.head_dim))
    scores = scores + add_mask
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v_rep)


def path_a2_sdpa_mask(
    cfg: DemoConfig, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L: int
) -> torch.Tensor:
    """Dense KV + explicit mask + torch SDPA.

    On current PyTorch this mask generally routes away from the FlashAttention
    fast path. That is exactly the implementation trap this demo exposes.
    """
    k_rep = k.repeat_interleave(cfg.gqa_group, dim=1)
    v_rep = v.repeat_interleave(cfg.gqa_group, dim=1)
    head_mask = _duo_mask(cfg, q.shape[2], L)
    add_mask = torch.where(head_mask, 0.0, float("-inf")).to(q.dtype).unsqueeze(0)
    if q.shape[0] > 1:
        add_mask = add_mask.expand(q.shape[0], -1, -1, -1)
    return F.scaled_dot_product_attention(
        q, k_rep, v_rep, attn_mask=add_mask, dropout_p=0.0, is_causal=False
    )


def path_b_segpaged_fused(
    cfg: DemoConfig,
    q: torch.Tensor,
    ragged: list[list[tuple[torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    """Fused SegPagedAttention path: one FA-3 varlen call per layer.

    We create one varlen sequence slot per (batch item, KV head). Each slot has
    q_len query tokens and a head-specific key length L_h. K/V head count is 1
    in the kernel, and GQA query heads are packed into the head dimension.
    """
    if not is_fa3_supported() or flash_attn_varlen_func is None:
        raise RuntimeError(
            "sgl_kernel.flash_attn FA-3 is required for B_segpaged_fused"
        )
    batch, _, q_len, _ = q.shape
    q_packs: list[torch.Tensor] = []
    k_packs: list[torch.Tensor] = []
    v_packs: list[torch.Tensor] = []
    cu_q = [0]
    cu_k = [0]
    max_k = 0

    for b in range(batch):
        for h in range(cfg.hkv):
            k_h, v_h = ragged[b][h]
            L_h = k_h.shape[0]
            q_h = q[b, h * cfg.gqa_group : (h + 1) * cfg.gqa_group, :, :]
            q_packs.append(q_h.transpose(0, 1).contiguous())  # [q_len, GQA, D]
            k_packs.append(k_h.unsqueeze(1).contiguous())
            v_packs.append(v_h.unsqueeze(1).contiguous())
            cu_q.append(cu_q[-1] + q_len)
            cu_k.append(cu_k[-1] + L_h)
            max_k = max(max_k, L_h)

    Q = torch.cat(q_packs, dim=0)
    K = torch.cat(k_packs, dim=0)
    V = torch.cat(v_packs, dim=0)
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=q.device)
    cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=q.device)
    out = flash_attn_varlen_func(
        q=Q,
        k=K,
        v=V,
        cu_seqlens_q=cu_q_t,
        cu_seqlens_k=cu_k_t,
        max_seqlen_q=q_len,
        max_seqlen_k=max_k,
        softmax_scale=1.0 / math.sqrt(cfg.head_dim),
        causal=True,
    )
    # [B*Hkv*q_len, GQA, D] -> [B, Hq, q_len, D]
    out = out.view(batch, cfg.hkv, q_len, cfg.gqa_group, cfg.head_dim)
    return out.permute(0, 1, 3, 2, 4).reshape(batch, cfg.hq, q_len, cfg.head_dim)


def make_layer_fn(
    cfg: DemoConfig,
    path: PathName,
    workload: Workload,
    L: int,
    batch: int,
    seed: int,
) -> Callable[[], torch.Tensor]:
    q_len = 1 if workload == "decode" else 2048
    q = build_queries(cfg, q_len, batch, seed=seed + 101)
    k, v = build_dense_kv(cfg, L, batch, seed=seed + 202)
    ragged = dense_to_ragged_views(cfg, k, v, L)
    if path == "A1_manual":
        return lambda: path_a1_manual(cfg, q, k, v, L)
    if path == "A2_sdpa_mask":
        return lambda: path_a2_sdpa_mask(cfg, q, k, v, L)
    if path == "B_segpaged_fused":
        return lambda: path_b_segpaged_fused(cfg, q, ragged)
    raise ValueError(path)


def bench(
    fn: Callable[[], torch.Tensor], layers: int, warmup: int, iters: int
) -> tuple[list[float], str]:
    try:
        for _ in range(warmup):
            for _layer in range(layers):
                fn()
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _layer in range(layers):
                fn()
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
        return times, "ok"
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return [], f"OOM: {exc}"
    except Exception as exc:  # pragma: no cover - environment-specific
        torch.cuda.empty_cache()
        return [], f"ERR: {exc}"


def summarize_times(times: list[float]) -> tuple[float, float, float, float, float]:
    if not times:
        nan = float("nan")
        return nan, nan, nan, nan, nan
    sorted_times = sorted(times)
    p50 = sorted_times[len(sorted_times) // 2]
    minimum = min(sorted_times)
    mean = statistics.mean(sorted_times)
    std = statistics.stdev(sorted_times) if len(sorted_times) > 1 else 0.0
    cov = 100.0 * std / mean if mean else 0.0
    return p50, minimum, mean, std, cov


def run_bench_cell(
    cfg: DemoConfig,
    path: PathName,
    workload: Workload,
    L: int,
    batch: int,
    layers: int,
    warmup: int,
    iters: int,
) -> BenchResult:
    fn = make_layer_fn(cfg, path, workload, L, batch, cfg.seed)
    times, notes = bench(fn, layers, warmup, iters)
    p50, minimum, mean, std, cov = summarize_times(times)
    return BenchResult(
        path=path,
        workload=workload,
        L=L,
        batch=batch,
        q_len=1 if workload == "decode" else 2048,
        layers=layers,
        warmup=warmup,
        iters=iters,
        p50_ms=p50,
        min_ms=minimum,
        mean_ms=mean,
        std_ms=std,
        cov_pct=cov,
        notes=notes,
    )


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.flatten().float()
    yf = y.flatten().float()
    return float((xf @ yf / (xf.norm() * yf.norm())).item())


def run_quality(
    cfg: DemoConfig, L: int, q_len: int = 1, batch: int = 1
) -> QualityResult:
    q = build_queries(cfg, q_len, batch, seed=cfg.seed + 11)
    k, v = build_dense_kv(cfg, L, batch, seed=cfg.seed + 22)
    ragged = dense_to_ragged_views(cfg, k, v, L)
    with torch.no_grad():
        a1 = path_a1_manual(cfg, q, k, v, L)
        a2 = path_a2_sdpa_mask(cfg, q, k, v, L)
        b = path_b_segpaged_fused(cfg, q, ragged)
    return QualityResult(
        L=L,
        q_len=q_len,
        cos_a1_a2=cosine(a1, a2),
        cos_a1_b=cosine(a1, b),
        cos_a2_b=cosine(a2, b),
    )


def mode_plan(mode: str) -> dict:
    if mode == "smoke":
        return {
            "quality_Ls": [2048],
            "bench": [
                ("decode", 2048, 1, 2, ["A2_sdpa_mask", "B_segpaged_fused"]),
                ("prefill", 2048, 1, 2, ["A2_sdpa_mask", "B_segpaged_fused"]),
            ],
            "warmup": 1,
            "iters": 2,
        }
    if mode == "quick":
        return {
            "quality_Ls": [8192],
            "bench": [
                (
                    "decode",
                    8192,
                    1,
                    64,
                    ["A1_manual", "A2_sdpa_mask", "B_segpaged_fused"],
                ),
                ("prefill", 8192, 1, 64, ["A2_sdpa_mask", "B_segpaged_fused"]),
                ("prefill", 8192, 4, 64, ["A2_sdpa_mask", "B_segpaged_fused"]),
            ],
            "warmup": 1,
            "iters": 3,
        }
    if mode == "paper":
        return {
            "quality_Ls": [8192, 32768],
            "bench": [
                (
                    "decode",
                    8192,
                    1,
                    64,
                    ["A1_manual", "A2_sdpa_mask", "B_segpaged_fused"],
                ),
                (
                    "decode",
                    32768,
                    1,
                    64,
                    ["A1_manual", "A2_sdpa_mask", "B_segpaged_fused"],
                ),
                (
                    "decode",
                    131072,
                    1,
                    64,
                    ["A1_manual", "A2_sdpa_mask", "B_segpaged_fused"],
                ),
                ("prefill", 8192, 1, 64, ["A2_sdpa_mask", "B_segpaged_fused"]),
                ("prefill", 32768, 1, 64, ["A2_sdpa_mask", "B_segpaged_fused"]),
                ("prefill", 131072, 1, 64, ["A2_sdpa_mask", "B_segpaged_fused"]),
                ("prefill", 8192, 4, 64, ["A2_sdpa_mask", "B_segpaged_fused"]),
                ("prefill", 32768, 4, 64, ["A2_sdpa_mask", "B_segpaged_fused"]),
            ],
            "warmup": 1,
            "iters": 4,
        }
    raise ValueError(f"Unknown mode: {mode}")


def print_quality_table(results: Iterable[QualityResult]) -> None:
    print("\nQuality check: identical dense/masked vs SegPaged outputs")
    print(
        f"  {'L':>8s} {'q_len':>6s} {'cos(A1,A2)':>12s} {'cos(A1,B)':>12s} {'cos(A2,B)':>12s}"
    )
    for r in results:
        print(
            f"  {r.L:>8d} {r.q_len:>6d} {r.cos_a1_a2:>12.6f} {r.cos_a1_b:>12.6f} {r.cos_a2_b:>12.6f}"
        )


def print_bench_table(results: Iterable[BenchResult]) -> None:
    print("\nLatency results")
    print(
        f"  {'path':<18s} {'workload':<8s} {'L':>8s} {'B':>2s} {'layers':>6s} "
        f"{'p50 ms':>10s} {'mean ms':>10s} {'CoV':>8s} notes"
    )
    for r in results:
        print(
            f"  {r.path:<18s} {r.workload:<8s} {r.L:>8d} {r.batch:>2d} {r.layers:>6d} "
            f"{r.p50_ms:>10.3f} {r.mean_ms:>10.3f} {r.cov_pct:>7.2f}% {r.notes}"
        )


def speedup_summary(results: list[BenchResult]) -> list[dict]:
    summary: list[dict] = []
    grouped: dict[tuple[str, int, int, int], dict[str, BenchResult]] = {}
    for r in results:
        grouped.setdefault((r.workload, r.L, r.batch, r.layers), {})[r.path] = r
    for key, cells in grouped.items():
        b = cells.get("B_segpaged_fused")
        if b is None or not math.isfinite(b.p50_ms) or b.p50_ms <= 0:
            continue
        item = {
            "workload": key[0],
            "L": key[1],
            "batch": key[2],
            "layers": key[3],
            "B_ms": b.p50_ms,
        }
        for base in ("A1_manual", "A2_sdpa_mask"):
            a = cells.get(base)
            item[f"speedup_vs_{base}"] = (
                (a.p50_ms / b.p50_ms) if a and math.isfinite(a.p50_ms) else None
            )
            item[f"{base}_ms"] = a.p50_ms if a else None
        summary.append(item)
    return summary


def print_speedup_summary(summary: Iterable[dict]) -> None:
    print("\nSpeedup summary (B_segpaged_fused is denominator)")
    print(
        f"  {'workload':<8s} {'L':>8s} {'B':>2s} {'layers':>6s} {'B ms':>10s} {'vs A1':>10s} {'vs A2':>10s}"
    )
    for s in summary:
        sp_a1 = s.get("speedup_vs_A1_manual")
        sp_a2 = s.get("speedup_vs_A2_sdpa_mask")
        print(
            f"  {s['workload']:<8s} {s['L']:>8d} {s['batch']:>2d} {s['layers']:>6d} {s['B_ms']:>10.3f} "
            f"{(str(round(sp_a1, 2)) + 'x') if sp_a1 else 'n/a':>10s} "
            f"{(str(round(sp_a2, 2)) + 'x') if sp_a2 else 'n/a':>10s}"
        )


def write_outputs(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = output.with_suffix(".md")
    lines = [
        "# RedKnot SegPagedAttention Demo Results",
        "",
        f"Generated at: `{payload['run']['started_at']}`",
        f"Mode: `{payload['run']['mode']}`",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(payload["environment"], indent=2),
        "```",
        "",
        "## Speedup Summary",
        "",
        "| workload | L | B | layers | B ms | vs A1_manual | vs A2_sdpa_mask |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in payload["speedup_summary"]:
        sp1 = s.get("speedup_vs_A1_manual")
        sp2 = s.get("speedup_vs_A2_sdpa_mask")
        lines.append(
            f"| {s['workload']} | {s['L']} | {s['batch']} | {s['layers']} | {s['B_ms']:.3f} | "
            f"{sp1:.2f}x"
            if sp1
            else f"| {s['workload']} | {s['L']} | {s['batch']} | {s['layers']} | {s['B_ms']:.3f} | n/a"
        )
        # Fix the row assembly above to avoid nested conditional formatting.
        if sp1:
            lines[-1] += f" | {sp2:.2f}x |" if sp2 else " | n/a |"
        else:
            lines[-1] += f" | {sp2:.2f}x |" if sp2 else " | n/a |"
    lines.extend(["", "Full JSON output is stored next to this file.", ""])
    md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RedKnot SegPagedAttention reproducibility demo"
    )
    parser.add_argument("--mode", choices=["smoke", "quick", "paper"], default="smoke")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/redknot_segpaged_demo.json")
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--iters", type=int, default=None, help="Override timed iterations"
    )
    parser.add_argument(
        "--warmup", type=int, default=None, help="Override warmup iterations"
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DemoConfig(seed=args.seed, dtype=args.dtype)
    env = cuda_info()
    if not env.get("fa3_supported"):
        raise RuntimeError(f"FA-3 is not available: {env.get('fa3_import_error', '')}")
    set_seed(cfg.seed)
    plan = mode_plan(args.mode)
    warmup = args.warmup if args.warmup is not None else plan["warmup"]
    iters = args.iters if args.iters is not None else plan["iters"]

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"RedKnot SegPagedAttention demo | mode={args.mode} | output={args.output}")
    print(json.dumps(env, indent=2))

    quality_results: list[QualityResult] = []
    for L in plan["quality_Ls"]:
        quality_results.append(run_quality(cfg, L))
        torch.cuda.empty_cache()
    print_quality_table(quality_results)

    bench_results: list[BenchResult] = []
    for workload, L, batch, layers, paths in plan["bench"]:
        for path in paths:
            # The manual prefill mask at very long L is intentionally omitted from
            # all paper-mode plans because it is both unrealistic and often OOM.
            result = run_bench_cell(
                cfg, path, workload, L, batch, layers, warmup, iters
            )
            bench_results.append(result)
            torch.cuda.empty_cache()
    print_bench_table(bench_results)

    summary = speedup_summary(bench_results)
    print_speedup_summary(summary)

    payload = {
        "run": {
            "mode": args.mode,
            "started_at": started,
            "seed": args.seed,
            "warmup": warmup,
            "iters": iters,
        },
        "environment": env,
        "config": asdict(cfg),
        "quality": [asdict(q) for q in quality_results],
        "benchmarks": [asdict(r) for r in bench_results],
        "speedup_summary": summary,
    }
    write_outputs(args.output, payload)
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
