#!/usr/bin/env python3
"""Numerical B300 gate for the isolated SM100-family FlashMLA build.

FlashMLA consumes the final sparse indices, so these two cases certify the
Pro-0813 attention geometry (128 query heads, D=512, TopK=1024).  The model's
64-head indexer remains a separate integration gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


FLASHMLA_ROOT = Path("/data/temp/FlashMLA-sm103-src").resolve()
FLASHMLA_TESTS = FLASHMLA_ROOT / "tests"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = tuple(torch.cuda.get_device_capability(0))
    device_name = torch.cuda.get_device_name(0)
    if capability != (10, 3) or "B300" not in device_name:
        raise RuntimeError(
            f"this certificate is B300/SM103-only, got {device_name} SM{capability}"
        )

    cuda_spec = importlib.util.find_spec("flash_mla.cuda")
    cuda_origin = Path(cuda_spec.origin).resolve() if cuda_spec and cuda_spec.origin else None
    if cuda_origin is None or FLASHMLA_ROOT not in cuda_origin.parents:
        raise RuntimeError(
            f"FlashMLA did not resolve from the isolated SM100-family build: {cuda_origin}"
        )

    sys.path.insert(0, str(FLASHMLA_TESTS))
    import lib
    import test_flash_mla_sparse_decoding as sparse_decode
    import test_flash_mla_sparse_prefill as sparse_prefill

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.set_float32_matmul_precision("high")

    decode_param = lib.RawTestParamForDecode(
        b=2,
        h_q=128,
        s_q=1,
        h_kv=1,
        s_kv=16_384,
        is_varlen=True,
        topk=1_024,
        block_size=256,
        d_qk=512,
        d_v=512,
        enable_attn_sink=False,
        check_correctness=True,
        num_runs=0,
        seed=2026,
    ).to_test_param()
    decode_result = sparse_decode.test_flash_mla(decode_param)
    if not decode_result.is_correct:
        raise AssertionError("FlashMLA Pro-0813 sparse decode oracle failed")

    prefill_param = lib.TestParam(
        s_q=62,
        s_kv=8_192,
        topk=1_024,
        h_q=128,
        h_kv=1,
        d_qk=512,
        d_v=512,
        seed=2026,
        check_correctness=True,
        num_runs=0,
        have_attn_sink=False,
        have_topk_length=False,
    )
    prefill_correct = sparse_prefill.run_test(prefill_param)
    if not prefill_correct:
        raise AssertionError("FlashMLA Pro-0813 sparse prefill oracle failed")

    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "pass",
                "device": device_name,
                "compute_capability": "sm_103",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "flash_mla_cuda": str(cuda_origin),
                "decode": {
                    "h_q": 128,
                    "h_kv": 1,
                    "d_qk": 512,
                    "d_v": 512,
                    "topk": 1024,
                    "correct": True,
                },
                "prefill": {
                    "h_q": 128,
                    "h_kv": 1,
                    "d_qk": 512,
                    "d_v": 512,
                    "topk": 1024,
                    "correct": True,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
