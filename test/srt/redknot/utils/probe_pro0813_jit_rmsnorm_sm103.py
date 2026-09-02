#!/usr/bin/env python3
"""Fail-closed B300 oracle for the Pro-0813 JIT RMSNorm fallback.

The shared ``sgl_kernel`` wheel has no SM103 code object.  Importing it is
still required by SGLang, but every Pro-0813 RMSNorm invocation must dispatch
through ``sglang.jit_kernel.norm``.  This probe exercises both ordinary and
fused residual forms at the three widths used by DeepSeek-V4-Pro-0813 before a
model worker is launched.
"""

from __future__ import annotations

import json
import os

if os.environ.get("SGLANG_USE_JIT_RMSNORM") != "1":
    raise RuntimeError("SGLANG_USE_JIT_RMSNORM must be exactly 1")

import torch

from sglang.srt.layers import layernorm as layernorm_module


EXPECTED_CAPABILITY = (10, 3)
EXPECTED_DEVICE_SUBSTRING = "B300"
PRO0813_RMSNORM_WIDTHS = (512, 1536, 7168)
EPS = 1.0e-6


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS)
    return (normalized * weight.float()).to(dtype=x.dtype)


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    max_abs = float((actual_fp32 - expected_fp32).abs().max().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_fp32.reshape(1, -1), expected_fp32.reshape(1, -1)
        ).item()
    )
    if not torch.isfinite(actual_fp32).all():
        raise RuntimeError("JIT RMSNorm produced a non-finite value")
    if max_abs > 0.04 or cosine < 0.9999:
        raise RuntimeError(
            "JIT RMSNorm differs from the BF16 reference: "
            f"max_abs={max_abs:.8f} cosine={cosine:.8f}"
        )
    return {"max_abs": max_abs, "cosine": cosine}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the B300 JIT RMSNorm oracle requires CUDA")
    if not bool(getattr(layernorm_module, "_use_jit_rmsnorm", False)):
        raise RuntimeError("SGLang did not bind its JIT RMSNorm implementation")

    device = torch.device("cuda", 0)
    device_name = torch.cuda.get_device_name(device)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability != EXPECTED_CAPABILITY or EXPECTED_DEVICE_SUBSTRING not in device_name:
        raise RuntimeError(
            "this certificate is B300/SM103-only, got "
            f"{device_name!r} capability={capability!r}"
        )

    torch.manual_seed(20260813)
    torch.cuda.manual_seed_all(20260813)
    records = []
    with torch.inference_mode():
        for width in PRO0813_RMSNORM_WIDTHS:
            norm = layernorm_module.RMSNorm(width, eps=EPS).to(
                device=device, dtype=torch.bfloat16
            )
            weight = torch.linspace(
                0.75, 1.25, width, device=device, dtype=torch.bfloat16
            )
            norm.weight.copy_(weight)

            source = torch.randn(
                (2, width), device=device, dtype=torch.bfloat16
            )
            ordinary = norm(source.clone())
            ordinary_metrics = _error_metrics(
                ordinary, _reference(source, weight)
            )

            fused_x_input = torch.randn(
                (2, width), device=device, dtype=torch.bfloat16
            )
            fused_residual_input = torch.randn(
                (2, width), device=device, dtype=torch.bfloat16
            )
            fused_x = fused_x_input.clone()
            fused_residual = fused_residual_input.clone()
            fused_output, fused_residual_output = norm(
                fused_x, fused_residual
            )
            expected_residual = fused_x_input + fused_residual_input
            residual_max_abs = float(
                (fused_residual_output.float() - expected_residual.float())
                .abs()
                .max()
                .item()
            )
            if residual_max_abs != 0.0:
                raise RuntimeError(
                    "JIT fused RMSNorm changed BF16 residual-add semantics: "
                    f"max_abs={residual_max_abs:.8f}"
                )
            fused_metrics = _error_metrics(
                fused_output, _reference(expected_residual, weight)
            )
            records.append(
                {
                    "width": width,
                    "ordinary": ordinary_metrics,
                    "fused": fused_metrics,
                    "fused_residual_max_abs": residual_max_abs,
                }
            )

    torch.cuda.synchronize(device)
    print(
        json.dumps(
            {
                "schema": "redknot-pro0813-b300-jit-rmsnorm-oracle-v1",
                "accelerator": device_name,
                "compute_capability": "sm_103",
                "jit_rmsnorm": True,
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "records": records,
                "status": "pass",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
