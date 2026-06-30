#!/usr/bin/env python3
"""FP8 offline/online enablement for transformers finegrained-FP8 on this box.

Two problems block block-quant FP8 checkpoints (e.g. Qwen3.5-397B-A17B-FP8) here:

  1. HF offline / restricted egress -> the FP8 matmul kernel cannot be pulled
     from the Hub at the first forward.
  2. transformers 5.x expects the kernel module to expose
     ``matmul / matmul_batched / matmul_grouped``, but the
     ``kernels-community/finegrained-fp8`` ``torch-cuda`` build exposes the older
     ``w8a8_*`` names -> ``_load_finegrained_fp8_kernel`` raises "missing
     required symbols" even when the kernel is present.

``apply()`` does, in order:

  * Try to load the locally-cached Triton ``finegrained-fp8`` kernel
    (``local_files_only``) and build a small shim object exposing the
    ``matmul / matmul_batched / matmul_grouped`` entry points transformers wants.
    The shim does dynamic per-block activation quant (``fp8_act_quant``) then the
    Triton block FP8 GEMM (``w8a8_block_fp8_matmul``) — i.e. the *real fused
    Triton path*, just under the right names. It is injected into
    ``finegrained_fp8._load_finegrained_fp8_kernel``'s cache so the stock
    ``fp8_linear`` uses it.
  * Regardless, install a pure-PyTorch ``fp8_linear`` fallback (block dequant +
    bf16 matmul) so that any call site for which the Triton shim is unavailable
    still runs offline. The Triton path is preferred when it loads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── pure-torch fallback (always correct, slower) ──────────────────────────
def _dequantize_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor, block_size):
    N, K = weight.shape
    if block_size is None:
        return (weight.to(torch.float32) * scale_inv.to(torch.float32)).to(
            torch.bfloat16
        )
    bn, bk = int(block_size[0]), int(block_size[1])
    w = weight.to(torch.float32)
    sf = scale_inv.to(torch.float32)
    rows = (torch.arange(N, device=w.device) // bn).clamp_(max=sf.shape[0] - 1)
    cols = (torch.arange(K, device=w.device) // bk).clamp_(max=sf.shape[1] - 1)
    return (w * sf[rows][:, cols]).to(torch.bfloat16)


def _fp8_linear_torch(
    input,
    weight,
    weight_scale_inv,
    block_size=None,
    bias=None,
    activation_scale=None,
    output_dtype=None,
):
    if weight.element_size() > 1:
        return F.linear(input, weight, bias)
    out_dtype = output_dtype or input.dtype
    w_bf16 = _dequantize_fp8_block(weight, weight_scale_inv, block_size)
    out = F.linear(input.to(w_bf16.dtype), w_bf16, None)
    if bias is not None:
        out = out + bias
    return out.to(out_dtype)


# ── Triton kernel shim: adapt w8a8_* -> matmul/matmul_batched/matmul_grouped ─
def _load_local_triton_module():
    """Import the locally-cached finegrained-fp8 ``torch-cuda`` Triton build.

    The ``kernels`` library only resolves exact build variants
    (e.g. ``torch29-cxx11-cu128-x86_64-linux``) or ``torch-universal``; this
    repo only ships ``torch-cuda`` (pure-Triton, no .so), so we import it by
    path instead. Returns the loaded module (exposing ``fp8_act_quant`` and
    ``w8a8_block_fp8_matmul``) or None.
    """
    import importlib.util
    import glob
    import os
    import sys

    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache = os.environ.get("HF_HUB_CACHE", os.path.join(home, "hub"))
    pattern = os.path.join(
        cache,
        "models--kernels-community--finegrained-fp8",
        "snapshots",
        "*",
        "build",
        "torch-cuda",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(
            f"[RedKnot] local finegrained-fp8 torch-cuda build not found under {cache}"
        )
        return None
    build_dir = matches[-1]
    # Register the build dir as an importable package root so the relative
    # imports inside (``from .act_quant import ...``) resolve.
    pkg_name = "_redknot_finegrained_fp8"
    init_path = os.path.join(build_dir, "__init__.py")
    if not os.path.exists(init_path):
        return None
    try:
        if pkg_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                pkg_name,
                init_path,
                submodule_search_locations=[build_dir],
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[pkg_name] = mod
            spec.loader.exec_module(mod)
        return sys.modules[pkg_name]
    except Exception as e:  # noqa: BLE001
        print(
            f"[RedKnot] failed to import local Triton FP8 build: {type(e).__name__}: {str(e)[:120]}"
        )
        return None


def _try_build_triton_shim():
    """Return a shim object with matmul/matmul_batched/matmul_grouped, or None.

    Loads the locally-cached ``kernels-community/finegrained-fp8`` Triton build
    and wraps its ``w8a8_block_fp8_matmul`` + ``fp8_act_quant`` into the high
    level ``matmul(input_bf16, weight_fp8, weight_scale, block_size, ...)``
    contract transformers expects (it does the dynamic activation quant first).
    """
    k = _load_local_triton_module()
    if k is None:
        return None

    needed = ["fp8_act_quant", "w8a8_block_fp8_matmul"]
    if not all(hasattr(k, n) for n in needed):
        avail = [a for a in dir(k) if "matmul" in a or "quant" in a]
        print(f"[RedKnot] Triton FP8 kernel missing entry points; available={avail}")
        return None

    fp8_act_quant = k.fp8_act_quant
    w8a8_block = k.w8a8_block_fp8_matmul

    _FP8_DT = torch.float8_e4m3fn

    def _safe_act_quant(A_bf16, bk):
        """Per-block dynamic FP8 activation quant, SAFE against all-zero blocks.

        The upstream Triton ``fp8_act_quant`` computes ``scale = max(|x|)/448`` and
        ``y = x / scale``; for an all-zero block ``scale == 0`` -> ``0/0 = NaN``,
        which then poisons the whole forward (logits -> NaN -> argmax token 0,
        i.e. the '!' degeneration). Real activations frequently contain zero
        blocks (padding / dead channels), so we clamp the scale to a tiny
        positive floor before dividing. Returns (q_fp8, scales_fp32) matching the
        kernel's layout: scales shape (M, K//bk).
        """
        M, K = A_bf16.shape
        x = A_bf16.to(torch.float32).reshape(M, K // bk, bk)
        amax = x.abs().amax(dim=-1, keepdim=True)  # (M, K//bk, 1)
        scale = (amax / 448.0).clamp_min(1e-12)
        q = (x / scale).clamp_(-448.0, 448.0).to(_FP8_DT)
        q = q.reshape(M, K)
        s = scale.squeeze(-1).to(torch.float32).contiguous()  # (M, K//bk)
        return q.contiguous(), s

    def _matmul_2d(A_bf16, weight_fp8, weight_scale, block_size, output_dtype):
        bk = int(block_size[1]) if block_size is not None else A_bf16.shape[-1]
        M, K = A_bf16.shape
        if K % bk != 0:
            return None
        A_q, A_s = _safe_act_quant(A_bf16.contiguous(), bk)
        out = w8a8_block(
            A_q,
            weight_fp8,
            A_s,
            weight_scale,
            list(block_size),
            output_dtype or A_bf16.dtype,
        )
        return out

    def matmul(
        input,
        weight,
        weight_scale_inv,
        block_size,
        output_dtype=None,
        activation_scale=None,
    ):
        out_dtype = output_dtype or input.dtype
        if weight.element_size() > 1 or block_size is None:
            return _fp8_linear_torch(
                input,
                weight,
                weight_scale_inv,
                block_size,
                None,
                activation_scale,
                out_dtype,
            )
        orig_shape = input.shape
        A = input.reshape(-1, orig_shape[-1]).contiguous().to(torch.bfloat16)
        res = _matmul_2d(A, weight, weight_scale_inv, block_size, out_dtype)
        if res is None:
            return _fp8_linear_torch(
                input,
                weight,
                weight_scale_inv,
                block_size,
                None,
                activation_scale,
                out_dtype,
            )
        return res.reshape(*orig_shape[:-1], res.shape[-1]).to(out_dtype)

    def matmul_batched(
        input,
        weight,
        weight_scale_inv,
        block_size=None,
        expert_ids=None,
        output_dtype=None,
        activation_scale=None,
    ):
        out_dtype = output_dtype or input.dtype
        return _fp8_linear_torch(
            input,
            weight,
            weight_scale_inv,
            block_size,
            None,
            activation_scale,
            out_dtype,
        )

    def matmul_grouped(*args, **kwargs):
        return matmul_batched(*args, **kwargs)

    class _Shim:
        pass

    shim = _Shim()
    shim.matmul = matmul
    shim.matmul_batched = matmul_batched
    shim.matmul_grouped = matmul_grouped
    return shim


def apply():
    """Install FP8 enablement (Triton shim if possible, torch fallback always)."""
    from transformers.integrations import finegrained_fp8 as ffp8

    if getattr(ffp8, "_REDKNOT_OFFLINE_FP8_PATCHED", False):
        return

    shim = _try_build_triton_shim()
    if shim is not None:
        fp8_obj = ffp8.FineGrainedFP8(
            matmul=shim.matmul,
            batched_matmul=shim.matmul_batched,
            grouped_matmul=shim.matmul_grouped,
        )
        try:
            ffp8._load_finegrained_fp8_kernel.cache_clear()
        except Exception:
            pass

        def _loader():
            return fp8_obj

        ffp8._load_finegrained_fp8_kernel = _loader
        ffp8.load_finegrained_fp8_kernel = _loader
        print(
            "[RedKnot] online FP8 enabled via local Triton kernel shim "
            "(matmul/batched/grouped)"
        )

    def fp8_linear_patched(
        input,
        weight,
        weight_scale_inv,
        block_size=None,
        bias=None,
        activation_scale=None,
        output_dtype=None,
    ):
        if shim is not None and weight.element_size() == 1 and block_size is not None:
            try:
                out = shim.matmul(
                    input,
                    weight,
                    weight_scale_inv,
                    block_size,
                    output_dtype=output_dtype or input.dtype,
                    activation_scale=activation_scale,
                )
                if bias is not None:
                    out = out + bias
                return out
            except Exception:
                pass
        return _fp8_linear_torch(
            input,
            weight,
            weight_scale_inv,
            block_size,
            bias,
            activation_scale,
            output_dtype,
        )

    ffp8.fp8_linear = fp8_linear_patched
    ffp8._REDKNOT_OFFLINE_FP8_PATCHED = True
    if shim is None:
        print(
            "[RedKnot] offline FP8 fallback installed (pure-torch block dequant matmul)"
        )
