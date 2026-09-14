"""Explicit local Flash-0731 native-generation smoke, not a reuse/quality benchmark.

Before --run, the operator must independently verify all 74 manifest files at
revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 with verify-only exit status 0.
--verified-model records that operator confirmation; this script does NOT hash
the weights, consume download progress as verification, or create verification
evidence. Pause any owned GPU keeper and verify the selected GPU is free first.

Example, in the separately prepared vLLM environment with the plugin installed:
  CUDA_VISIBLE_DEVICES=GPU-<selected-full-uuid> python check_dsv4_model_smoke.py \
      --run --verified-model --model /models/DeepSeek-V4-Flash-0731 \
      --config /absolute/deepseek_v4_flash_policy.json \
      --cases-output /absolute/new-smoke-cases.json \
      --report-output /absolute/new-native-smoke.json

Only explicit --run plus --verified-model can import vLLM or initialize a GPU.
The one native request uses mode=recomputed, greedy generation, and exactly
eight output tokens. Worker RPC evidence must show TP/PP/DP/context parallelism
all equal to one, a recomputed prefill, and no capture or reuse. Successful
generation then creates two original fictional English fact records for the
separate paired benchmark. References remain empty: these are smoke inputs,
not a quality dataset and not evidence of RedKnot speed or accuracy.

The native tokenizer's chat template is used with thinking disabled. The case
prompt is DEFINED as the concatenation of independently encoded document and
query fragments (without added special tokens). It is not claimed to equal a
single tokenization of the rendered full text. No offsets API, token truncation,
padding, or guessed special token IDs are used; all exact fragments are recorded.
Both documents exceed the configured dirty boundary and 128 tokens. New files
only: existing outputs, including symlinks, are never overwritten.
The owned engine's pinned EngineCoreClient.shutdown(timeout=30) is called in
finally, including on case-preparation failure. If LLM construction never returns
an instance, only vLLM's constructor cleanup and process exit are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

FLASH_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
FAMILY = "deepseek_v4_flash"
NATIVE_OUTPUT_TOKENS = 8
CASE_OUTPUT_RESERVE = 128
PARALLEL_FIELDS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "decode_context_parallel_size",
    "prefill_context_parallel_size",
)
DOCUMENT_A = """Document A: The fictional Alder observatory field record.
The Alder observatory stands on a low hill beside a freshwater lake. Its field
team studies the movement of clouds using a blue camera mounted on the eastern
roof. Mara is the camera operator, and Ivo maintains the weather notebook. Each
morning Mara photographs the same patch of sky at seven, eight, and nine o'clock.
Ivo writes the wind direction next to each photograph before anyone analyzes
the images. The team stores the morning photographs in a folder called Cedar.
Photographs taken in the afternoon go into a different folder called Willow.
The folders identify the collection time, not the color of the clouds.
On the first trial day, the team used three white reference cards to check the
camera exposure. The second trial used the same cards and an unchanged lens.
Mara rejected one blurred frame but kept the original file in a review folder.
The published table contains only checked frames and lists the exposure setting
beside every entry. Nobody edits the weather notebook after the daily review.
The spare camera is green and stays in a locked cupboard near the western door.
Its battery is charged on Fridays, even during weeks with no afternoon session.
The project coordinator compares the daily tables at the end of each month.
"""
DOCUMENT_B = """Document B: The fictional Birch library delivery record.
The Birch library runs a small delivery service for readers in three nearby
villages. Its coordinator, Lena, uses a yellow bicycle for the northern route.
The southern route uses a red bicycle, while books for the western route travel
in a white van. These vehicle colors never indicate the subject of a book.
Every parcel contains a printed loan receipt and a reusable cloth cover. Staff
place the receipts inside the covers before sealing the parcels for transport.
On Tuesdays, Tomas checks the route labels at the library's wooden sorting desk.
He places northern parcels on the upper shelf and southern parcels on the lower
shelf. Western parcels remain in a separate box beside the desk until the van
arrives. The shelves are cleared after each delivery round and inspected again.
The delivery log records the parcel count, departure time, and driver name.
Readers return their books through village collection boxes on the following
Monday. Lena counts those returns before any new parcels leave the building.
A damaged cover is replaced, but the book keeps its original catalog number.
During the autumn trial, the northern route carried twelve parcels each week.
The trial report tracked late arrivals separately from damaged covers because
the two events require different remedies. A monthly staff meeting reviews both
lists without changing the original delivery entries or the readers' receipts.
"""
QUESTION = (
    "Question: According to Document A, what is the folder for morning photographs? "
    "According to Document B, what color is the northern route bicycle? "
    "Answer both questions using only the records above.\n"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicit GPU opt-in")
    parser.add_argument(
        "--verified-model",
        action="store_true",
        help="Operator confirms independent complete 74-file verify-only exit 0",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser


def new_output_path(path: Path) -> Path:
    result = path.expanduser().absolute()
    if os.path.lexists(result):
        raise FileExistsError(f"output already exists: {result}")
    if not result.parent.is_dir():
        raise ValueError(f"output parent directory must already exist: {result.parent}")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if not args.run or not args.verified_model:
        raise ValueError("both --run and --verified-model are required")
    if args.max_model_len < 256:
        raise ValueError(
            "--max-model-len must be at least 256; no tokens are truncated"
        )
    if not math.isfinite(args.gpu_memory_utilization) or not (
        0 < args.gpu_memory_utilization <= 1
    ):
        raise ValueError("--gpu-memory-utilization must be finite and in (0, 1]")
    args.cases_output = new_output_path(args.cases_output)
    args.report_output = new_output_path(args.report_output)
    if args.cases_output.resolve() == args.report_output.resolve():
        raise ValueError("cases and report outputs must be distinct new files")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_uuid = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    if re.fullmatch(rf"(?:0|[1-9][0-9]*|{gpu_uuid})", visible) is None:
        raise ValueError(
            "set CUDA_VISIBLE_DEVICES to exactly one full GPU UUID or ordinal; "
            "check that GPU is free before running"
        )
    for name, expected in (
        ("VLLM_DP_SIZE", "1"),
        ("VLLM_DP_RANK", "0"),
        ("VLLM_DP_RANK_LOCAL", "0"),
        ("WORLD_SIZE", "1"),
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
    ):
        if name in os.environ and os.environ[name] != expected:
            raise ValueError(f"unsupported distributed environment: {name}")


def prepare_engine(args: argparse.Namespace) -> tuple[dict, dict]:
    """Local CPU checks and explicit environment setup, before importing vLLM."""
    from vllm_redknot.cli import checkpoint_engine_options, installed_plugin
    from vllm_redknot.compat import verify_vllm_sources
    from vllm_redknot.config import load_config, policy_fingerprint

    model = args.model.expanduser().resolve(strict=True)
    policy_path = args.config.expanduser().resolve(strict=True)
    policy = load_config(policy_path)
    if not policy or not policy["enabled"] or policy.get("engine_family") != FAMILY:
        raise ValueError("an enabled deepseek_v4_flash policy is required")
    if policy["model_revision"] != FLASH_REVISION:
        raise ValueError("only the fixed Flash-0731 revision is supported")
    backend, dtype = checkpoint_engine_options(model, policy, "bfloat16")
    source_contract = verify_vllm_sources(engine_family=FAMILY)
    if not installed_plugin():
        raise ValueError("install the redknot plugin in this vLLM interpreter first")
    overrides = {
        "VLLM_REDKNOT_CONFIG": str(policy_path),
        "VLLM_PLUGINS": "redknot",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    os.environ.update(overrides)
    return {
        "model": str(model),
        "dtype": dtype,
        "trust_remote_code": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "max_num_seqs": 1,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "attention_config": {"backend": backend},
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": False,
        "generation_config": "vllm",
        "seed": 0,
    }, {
        "engine_family": FAMILY,
        "model_revision": FLASH_REVISION,
        "configuration": policy,
        "config_path": str(policy_path),
        "policy_fingerprint": policy_fingerprint(policy),
        "source_contract": source_contract,
        "environment_overrides": overrides,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "model_verification": {
            "method": "operator_confirmation_only",
            "confirmed_by_explicit_verified_model_flag": True,
            "independent_manifest_file_count_expected": 74,
            "independent_verify_only_exit_code_expected": 0,
            "weights_verified_by_this_script": False,
            "download_progress_used_as_evidence": False,
        },
    }


def token_ids(raw: Any) -> list[int]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("tokenizer must return a nonempty list of token IDs")
    if any(type(token) is not int or token < 0 for token in raw):
        raise ValueError("expected nonnegative integer token IDs")
    return list(raw)


def chat_prompt(tokenizer: Any, content: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False
    )
    if not isinstance(rendered, str) or rendered.count(content) != 1:
        raise ValueError("native chat template must retain the complete user content")
    tokens = token_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False)
    )
    if tokens != token_ids(tokenizer.encode(rendered, add_special_tokens=False)):
        raise ValueError(
            "native chat template tokens differ from its rendered encoding"
        )
    return rendered, tokens


def read_worker_state(worker: Any) -> dict[str, Any]:
    """Module-level callable serialized by collective_rpc into the actual worker."""
    runner = worker.model_runner
    runtime = runner.redknot_runtime
    config = worker.vllm_config
    return {
        "worker_pid": os.getpid(),
        "runtime_class": f"{type(runtime).__module__}.{type(runtime).__name__}",
        "runner_hook_installed": bool(
            getattr(runner, "_redknot_dsv4_installed", False)
        ),
        "parallel": {
            name: getattr(config.parallel_config, name) for name in PARALLEL_FIELDS
        },
        "model": config.model_config.model,
        "dtype": str(config.model_config.dtype),
        "model_revision": runtime.settings.model_revision,
        "attention_backend": config.attention_config.backend.name,
        "stats": runtime.stats(),
    }


def snapshot(llm: Any) -> dict[str, Any]:
    replies = llm.collective_rpc(read_worker_state, timeout=10)
    if not isinstance(replies, list) or len(replies) != 1:
        raise RuntimeError("expected exactly one actual TP1 worker RPC reply")
    result = replies[0]
    if not isinstance(result, Mapping) or result.get("runtime_class") != (
        "vllm_redknot.dsv4_runtime.DSV4Runtime"
    ):
        raise RuntimeError("actual DSV4 worker runtime is missing")
    if result.get("runner_hook_installed") is not True or (
        result.get("model_revision") != FLASH_REVISION
        or result.get("dtype") != "torch.bfloat16"
        or result.get("attention_backend") != "FLASHMLA_SPARSE_DSV4"
    ):
        raise RuntimeError(
            "actual worker does not match the native Flash configuration"
        )
    parallel = result.get("parallel", {})
    if any(
        type(parallel.get(name)) is not int or parallel[name] != 1
        for name in PARALLEL_FIELDS
    ):
        raise RuntimeError(
            "actual worker must use TP/PP/DP and context parallelism = 1"
        )
    stats = result.get("stats")
    if not isinstance(stats, Mapping):
        raise RuntimeError("actual worker counters are missing")
    for name in ("runtime", "cache"):
        group = stats.get(name)
        if not isinstance(group, Mapping) or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in group.items()
        ):
            raise RuntimeError("actual worker counters must be nonnegative integers")
    if any(name not in stats["cache"] for name in ("hits", "entries", "bytes")):
        raise RuntimeError("actual worker cache accounting is missing")
    return dict(result)


def native_delta(before: dict, after: dict) -> dict[str, Any]:
    for key in ("worker_pid", "model", "model_revision", "parallel"):
        if before[key] != after[key]:
            raise RuntimeError(
                f"worker identity changed across native generation: {key}"
            )
    delta = {}
    for group in ("runtime", "cache"):
        left, right = before["stats"][group], after["stats"][group]
        delta[group] = {
            key: right.get(key, 0) - left.get(key, 0) for key in set(left) | set(right)
        }
        if any(value < 0 for value in delta[group].values()):
            raise RuntimeError("worker counters reset across native generation")
    if delta["runtime"].get("recomputed_steps", 0) <= 0:
        raise RuntimeError("worker did not observe a recomputed complete prefill")
    for key, value in delta["runtime"].items():
        if value and (
            key.startswith(("capture", "reuse", "restored_"))
            or key in {"native_state_token_rows", "launched_sparse_head_rows"}
        ):
            raise RuntimeError(
                f"native smoke unexpectedly used RedKnot capture/reuse: {key}"
            )
    if any(delta["cache"].get(key, 0) for key in ("hits", "entries", "bytes")):
        raise RuntimeError("native smoke unexpectedly changed RedKnot cache state")
    return delta


def run_native(
    llm: Any,
    sampling_factory: Callable[..., Any],
    *,
    max_model_len: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], Any]:
    tokenizer = llm.get_tokenizer()
    text, tokens = chat_prompt(
        tokenizer,
        "Reply with one short sentence stating that a triangle has three sides.",
    )
    if len(tokens) < 2 or len(tokens) + NATIVE_OUTPUT_TOKENS > max_model_len:
        raise ValueError("native smoke prompt does not fit; no truncation is allowed")
    params = sampling_factory(
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_tokens=NATIVE_OUTPUT_TOKENS,
        min_tokens=NATIVE_OUTPUT_TOKENS,
        ignore_eos=True,
        extra_args={
            "redknot": {
                "mode": "recomputed",
                "namespace": "native-smoke-" + uuid.uuid4().hex,
                "chunks": [{"start": 0, "end": len(tokens)}],
                "allow_approximate": False,
            }
        },
    )
    before = snapshot(llm)
    started = clock()
    outputs = llm.generate([{"prompt_token_ids": tokens}], params, use_tqdm=False)
    elapsed = clock() - started
    after = snapshot(llm)
    delta = native_delta(before, after)
    if (
        not isinstance(outputs, list)
        or len(outputs) != 1
        or len(outputs[0].outputs) != 1
    ):
        raise RuntimeError("expected one native request with one completion")
    output = outputs[0]
    completion = output.outputs[0]
    generated = token_ids(completion.token_ids)
    if getattr(output, "prompt_token_ids", None) != tokens:
        raise RuntimeError("native generation did not retain every prompt token")
    if (
        getattr(output, "finished", None) is not True
        or len(generated) != NATIVE_OUTPUT_TOKENS
    ):
        raise RuntimeError(
            "native generation must finish with exactly eight output tokens"
        )
    if not isinstance(completion.text, str) or not completion.text.strip():
        raise RuntimeError("native generation returned no visible output text")
    if getattr(getattr(output, "metrics", None), "is_corrupted", False):
        raise RuntimeError("vLLM marked the native output as corrupted")
    return {
        "passed": True,
        "mode": "recomputed",
        "prompt_text": text,
        "prompt_token_ids": tokens,
        "output_text": completion.text,
        "output_token_ids": generated,
        "finish_reason": getattr(completion, "finish_reason", None),
        "finished": output.finished,
        "e2e_seconds": elapsed,
        "worker_before": before,
        "worker_after": after,
        "worker_delta": delta,
        "quality_assessed": False,
        "reuse_assessed": False,
    }, tokenizer


def build_cases(tokenizer: Any, *, boundary_tokens: int, max_model_len: int) -> dict:
    """Define all case tokens by explicit fragment concatenation, never slicing IDs."""
    content = DOCUMENT_A + "\n" + DOCUMENT_B + "\n" + QUESTION
    rendered, single_encoding = chat_prompt(tokenizer, content)
    second = rendered.index(DOCUMENT_B)
    query = rendered.index(QUESTION)
    fragments = [rendered[:second], rendered[second:query], rendered[query:]]
    pieces = [
        token_ids(tokenizer.encode(part, add_special_tokens=False))
        for part in fragments
    ]
    minimum = max(128, boundary_tokens)
    document_lengths = [
        len(token_ids(tokenizer.encode(document, add_special_tokens=False)))
        for document in (DOCUMENT_A, DOCUMENT_B)
    ]
    if any(
        length <= minimum for length in document_lengths + list(map(len, pieces[:2]))
    ):
        raise ValueError(
            "both complete documents must exceed 128 tokens and the dirty boundary"
        )
    prompt = pieces[0] + pieces[1] + pieces[2]
    if len(prompt) + CASE_OUTPUT_RESERVE > max_model_len:
        raise ValueError(
            "complete case plus 128 output tokens does not fit; no truncation allowed"
        )
    digest = hashlib.sha256(
        json.dumps(prompt, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "cases": [
            {
                "id": "original-english-smoke",
                "chunks": pieces[:2],
                "query": pieces[2],
                "references": [],
            }
        ],
        "provenance": {
            "purpose": "synthetic smoke input, not a quality benchmark",
            "references_available": False,
            "prompt_definition": (
                "concatenate the exact independently encoded fragments in order"
            ),
            "tokenizer_class": (
                f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
            ),
            "thinking_enabled": False,
            "add_special_tokens_for_fragments": False,
            "rendered_text": rendered,
            "fragment_texts": fragments,
            "document_token_counts": document_lengths,
            "fragment_token_counts": list(map(len, pieces)),
            "full_prompt_token_ids": prompt,
            "full_prompt_sha256_json_compact": digest,
            "matches_single_full_text_encoding": prompt == single_encoding,
            "truncated_tokens": 0,
            "boundary_tokens": boundary_tokens,
            "output_token_reserve": CASE_OUTPUT_RESERVE,
        },
    }


def write_json(path: Path, document: dict) -> None:
    serialized = (
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        engine, metadata = prepare_engine(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    report = {
        "schema_version": 1,
        "status": "FAILED",
        "native_smoke_passed": False,
        "full_redknot_benchmark_passed": False,
        "engine": engine,
        "metadata": metadata,
    }
    llm = None
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(**engine)
        native, tokenizer = run_native(
            llm, SamplingParams, max_model_len=args.max_model_len
        )
        report["native"] = native
        report["native_smoke_passed"] = True
        cases = build_cases(
            tokenizer,
            boundary_tokens=metadata["configuration"]["boundary_tokens"],
            max_model_len=args.max_model_len,
        )
        write_json(args.cases_output, cases)
        report["cases_output"] = str(args.cases_output)
        report["cases_prompt_sha256"] = cases["provenance"][
            "full_prompt_sha256_json_compact"
        ]
        report["status"] = "NATIVE_SMOKE_PASSED_CASES_WRITTEN"
    except Exception as error:  # noqa: BLE001 - record failures, never claim a passed smoke.
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if llm is None:
            report["shutdown"] = {
                "attempted": False,
                "reason": "LLM construction did not return an owned instance",
            }
        else:
            try:
                # Pinned core_client.py:377,740; LLM itself has no shutdown API.
                llm.llm_engine.engine_core.shutdown(timeout=30)
                report["shutdown"] = {
                    "attempted": True,
                    "call_returned": True,
                    "gpu_free_independently_verified": False,
                }
            except Exception as error:  # noqa: BLE001 - preserve cleanup failures.
                report["shutdown"] = {
                    "attempted": True,
                    "call_returned": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
                report.setdefault("error", report["shutdown"]["error"])
                report["status"] = "FAILED"
    write_json(args.report_output, report)
    print(
        json.dumps(
            {"status": report["status"], "report_output": str(args.report_output)}
        )
    )
    if "error" in report:
        print(f"native smoke failed: {report['error']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
