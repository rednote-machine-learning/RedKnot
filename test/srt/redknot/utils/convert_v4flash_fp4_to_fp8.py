#!/usr/bin/env python3
"""Convert DeepSeek-V4-Flash routed experts from FP4 (e2m1, packed int8) to FP8
(e4m3) IN-PLACE of the HuggingFace/SGLang weight layout.

Unlike the official inference/convert.py (which rewrites all keys into the
private generate.py format and sharded model{i}-mp{N}.safetensors), this script
PRESERVES the original HF key naming and per-file sharding so the result loads
directly with SGLang's dsv4 backend — exactly like DeepSeek-V4-Flash-Base does.

Why: H800 is Hopper (sm_90) with NO FP4 Tensor Core. FP4 experts only have
hardware acceleration on Blackwell (sm_100). Upconverting experts to FP8 lets
Hopper run them natively at full speed while keeping the instruct model's
quality.

The FP4->FP8 upconversion (cast_e2m1fn_to_e4m3fn) is copied verbatim from the
official convert.py and is lossless: e2m1 values * per-block scale are exactly
representable in e4m3 with an e8m0 block scale.

Output format per expert weight:
    weight: [out, in]      float8_e4m3fn   (was [out, in//2] int8)
    scale:  [out//128, in//128] float8_e8m0fnu  (was [out, in//32] e8m0)

This matches config.json quantization_config: fmt=e4m3, scale_fmt=ue8m0,
weight_block_size=[128,128] — identical to the Base checkpoint.

Usage::

    python convert_v4flash_fp4_to_fp8.py \
        --src /path/to/DeepSeek-V4-Flash \
        --dst /path/to/DeepSeek-V4-Flash-FP8 \
        [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# ---------------------------------------------------------------------------
# FP4 (e2m1) -> FP8 (e4m3) lossless upconversion (from official convert.py)
# ---------------------------------------------------------------------------
FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def cast_e2m1fn_to_e4m3fn(
    x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Casts a tensor from e2m1fn (packed int8) to e4m3fn losslessly.

    x:     [out, in//2] int8 (two fp4 values packed per byte)
    scale: [out, in//32] e8m0 (per-32-element block scale)

    Returns:
        weight: [out, in] float8_e4m3fn
        scale:  [out//128, in//128] float8_e8m0fnu
    """
    assert x.dtype == torch.int8, f"expected int8, got {x.dtype}"
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    # max_fp4 (6.0) * MAX_OFFSET must fit in e4m3fn (max 448)
    # 6.0 * 2^6 = 384 < 448; 6.0 * 2^7 = 768 > 448; so MAX_OFFSET_BITS = 6
    MAX_OFFSET_BITS = 6

    bOut = out_dim // fp8_block_size
    bIn = in_dim // fp8_block_size
    # bOut, bIn, 128, 128
    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)
    # bOut, bIn, 128*4
    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)
    # bOut, bIn, 1
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
    # bOut, bIn, 128*4
    offset = scale / scale_max_offset_bits
    # bOut, bIn, 128, 128
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(
        fp4_block_size, dim=-1
    )
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(
        torch.float8_e8m0fnu
    )


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------
def is_expert_weight(name: str) -> bool:
    """Routed expert weight (not shared expert)."""
    return (
        "experts." in name and "shared_experts" not in name and name.endswith(".weight")
    )


def convert(src: str, dst: str, num_threads: int = 8):
    torch.set_num_threads(num_threads)
    os.makedirs(dst, exist_ok=True)

    files = sorted(glob(os.path.join(src, "*.safetensors")))
    print(f"Found {len(files)} safetensors files in {src}")

    n_converted = 0
    n_copied = 0

    for fp in tqdm(files, desc="Files"):
        fname = os.path.basename(fp)
        out_path = os.path.join(dst, fname)

        new_tensors = {}
        with safe_open(fp, framework="pt", device="cpu") as f:
            keys = list(f.keys())

            # First pass: identify expert weight/scale pairs in this file
            for name in keys:
                tensor = f.get_tensor(name)

                if is_expert_weight(name) and tensor.dtype == torch.int8:
                    # This is an FP4-packed expert weight; find its scale
                    scale_name = name[: -len(".weight")] + ".scale"
                    if scale_name not in keys:
                        raise KeyError(
                            f"Expert weight {name} (int8/fp4) has no scale "
                            f"{scale_name} in {fname}"
                        )
                    scale = f.get_tensor(scale_name)
                    # scale is e8m0 -> view as uint8 then to float for math
                    if scale.dtype == torch.float8_e8m0fnu:
                        scale_f = scale.float()
                    else:
                        scale_f = scale.float()

                    w_fp8, s_e8m0 = cast_e2m1fn_to_e4m3fn(tensor, scale_f)
                    new_tensors[name] = w_fp8.contiguous()
                    new_tensors[scale_name] = s_e8m0.contiguous()
                    n_converted += 1
                elif is_expert_weight(
                    name.replace(".scale", ".weight")
                ) and name.endswith(".scale"):
                    # Expert scale: already handled together with its weight above.
                    # Skip here (will be set when we process the weight).
                    # But if weight was NOT int8 (already fp8), keep scale as-is.
                    weight_name = name[: -len(".scale")] + ".weight"
                    if weight_name in new_tensors:
                        continue  # already converted
                    # weight not int8 -> keep original
                    if name not in new_tensors:
                        new_tensors[name] = tensor.contiguous()
                    n_copied += 1
                else:
                    # Non-expert weight (or already-fp8 expert): copy verbatim
                    if name not in new_tensors:
                        new_tensors[name] = tensor.contiguous()
                        n_copied += 1

        # Ensure all converted scales overwrote any copied originals
        save_file(new_tensors, out_path, metadata={"format": "pt"})

    print(f"Converted {n_converted} expert weights, copied {n_copied} tensors")

    # Copy config + tokenizer + index (config quant_config already fp8/ue8m0)
    aux_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "configuration_deepseek.py",
        "modeling_deepseek.py",
    ]
    # Also copy any *.py custom code and special_tokens
    for extra in glob(os.path.join(src, "*.py")):
        aux_files.append(os.path.basename(extra))
    for extra in glob(os.path.join(src, "*special_tokens*")):
        aux_files.append(os.path.basename(extra))

    for fn in set(aux_files):
        s = os.path.join(src, fn)
        d = os.path.join(dst, fn)
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copyfile(s, d)
            print(f"  copied aux: {fn}")

    print(f"\nDone. FP8 checkpoint at: {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source FP4 checkpoint dir")
    ap.add_argument("--dst", required=True, help="Destination FP8 checkpoint dir")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    convert(args.src, args.dst, num_threads=args.workers)


if __name__ == "__main__":
    main()
