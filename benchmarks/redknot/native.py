"""Measure native request admission through observed token completion.

This module has no accelerator dependencies. The engine protocol mirrors
DeepInfer; the runner's bridge prepares NumPy input arrays before timing starts.
An output reader must copy fields from DeepInfer's pooled pointers immediately:
those pointers cease to be valid on the next engine step.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any

BOUNDARY = "native_engine_enqueue_to_native_token_observation"
ADMISSION_POLICY = "manifest_order_sequential_batches_v1"
_PHASE_IDS = itertools.count()


def run_admission_phase(
    engine: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    admission_batch_size: int | None = None,
    sampling_factory: Callable[[Mapping[str, Any]], Any],
    output_reader: Callable[[Any], Mapping[str, Any]],
    clock: Callable[[], float] = time.perf_counter,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict:
    """Drain each fixed, ordered batch before admitting the next batch.

    Each run_phase synchronizes ranks after CPU input preparation and before
    admission. Local elapsed_s is the sum of local batch durations; the runner
    must sum the cross-rank maximum separately for each batch, not take the
    maximum of these local sums. None retains the original single-phase path.
    """
    if admission_batch_size is None:
        return run_phase(engine, cases, sampling_factory=sampling_factory, output_reader=output_reader, clock=clock)
    if type(admission_batch_size) is not int or admission_batch_size < 1:
        raise ValueError("admission_batch_size must be a positive integer")
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("a phase requires a nonempty case sequence")
    sample_ids = [item.get("sample_id") if isinstance(item, Mapping) else None for item in cases]
    if not all(isinstance(sample_id, str) and sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be nonempty and unique across admission batches")
    batches, requests = [], []
    for batch_index, start in enumerate(range(0, len(cases), admission_batch_size)):
        selected = cases[start : start + admission_batch_size]
        phase = run_phase(engine, selected, sampling_factory=sampling_factory, output_reader=output_reader, clock=clock)
        batches.append(
            {
                "batch_index": batch_index,
                "sample_ids": [case["sample_id"] for case in selected],
                "elapsed_s": phase["elapsed_s"],
                "completed_requests": phase["completed_requests"],
            }
        )
        requests.extend({**record, "batch_index": batch_index} for record in phase["requests"])
        if progress_callback is not None:
            progress_callback(dict(batches[-1]))
    return {
        "measurement_boundary": BOUNDARY,
        "elapsed_s": sum(batch["elapsed_s"] for batch in batches),
        "requests": requests,
        "completed_requests": len(requests),
        "batches": batches,
    }


def run_phase(
    engine: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    sampling_factory: Callable[[Mapping[str, Any]], Any],
    output_reader: Callable[[Any], Mapping[str, Any]],
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Run a complete, pre-tokenized cohort, including the final async flush.

    ``engine.prepare_input_ids(list[int])`` is optional and runs outside the
    measurement window. All ranks must call this function for the same phase;
    ``has_unfinished_requests`` can perform distributed collectives even when
    a rank has completed its local work. Errors are raised, never converted to
    successful measurements. Cross-rank failure supervision belongs to runner.
    """
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("a phase requires a nonempty case sequence")
    prepared: list[tuple[str, Mapping[str, Any], Any, Any]] = []
    sample_ids: set[str] = set()
    phase_id = next(_PHASE_IDS)
    prepare_ids = getattr(engine, "prepare_input_ids", list)
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("case must be an object")
        sample_id = case.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("sample IDs must be nonempty and unique within a phase")
        sample_ids.add(sample_id)
        tokens = case.get("input_ids")
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or not 0 <= token < 2**31 for token in tokens):
            raise ValueError(f"{sample_id}: invalid int32 input token IDs")
        if type(case.get("max_new_tokens")) is not int or case["max_new_tokens"] < 1:
            raise ValueError(f"{sample_id}: max_new_tokens must be positive")
        if "ignore_eos" in case and type(case["ignore_eos"]) is not bool:
            raise ValueError(f"{sample_id}: ignore_eos must be boolean")
        prepared.append((f"redknot-{phase_id}-{index}", case, prepare_ids(tokens), sampling_factory(case)))
    if hasattr(engine, "get_num_unfinished_requests") and engine.get_num_unfinished_requests():
        raise RuntimeError("benchmark phase requires an idle engine")
    # The native bridge synchronizes DP ranks here, after CPU preparation.
    if hasattr(engine, "before_phase"):
        engine.before_phase()

    records: dict[str, dict] = {}
    previous_time: float | None = None

    def timestamp() -> float:
        nonlocal previous_time
        value = clock()
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError("measurement clock returned a non-finite timestamp")
        if previous_time is not None and value < previous_time:
            raise RuntimeError("measurement clock moved backwards")
        previous_time = value
        return float(value)

    phase_start = timestamp()
    try:
        for request_id, case, input_ids, sampling in prepared:
            admitted_at = timestamp()
            records[request_id] = {
                "sample_id": case["sample_id"],
                "output_token_ids": [],
                "ttft_s": None,
                "latency_s": None,
                "finished": False,
                "_admitted_at": admitted_at,
                "_max_new_tokens": case["max_new_tokens"],
                "_ignore_eos": case.get("ignore_eos", False),
            }
            sequence = engine.add_request(
                user_id="redknot-benchmark",
                request_id=request_id,
                prompt_token_ids=input_ids,
                sampling_params=sampling,
                arrival_time=admitted_at,
            )
            if sequence is None:
                raise RuntimeError(f"engine rejected request {case['sample_id']}")

        while True:
            unfinished = engine.has_unfinished_requests()
            raw_outputs = engine.step() if unfinished else engine.flush_inflight_reqs()
            observed_at = timestamp()
            for raw_output in raw_outputs:
                output = output_reader(raw_output)
                request_id = output.get("request_id")
                if request_id not in records:
                    raise RuntimeError(f"engine emitted an unknown request: {request_id!r}")
                record = records[request_id]
                if record["finished"]:
                    raise RuntimeError(f"engine emitted output after completion: {record['sample_id']}")
                token_id = output.get("token_id")
                if type(token_id) is not int or token_id < 0:
                    raise RuntimeError(f"engine emitted an invalid token: {record['sample_id']}")
                if type(output.get("finished")) is not bool:
                    raise RuntimeError("engine output requires a boolean finished field")
                record["output_token_ids"].append(token_id)
                if len(record["output_token_ids"]) > record["_max_new_tokens"]:
                    raise RuntimeError(f"engine exceeded output token budget: {record['sample_id']}")
                if record["ttft_s"] is None:
                    record["ttft_s"] = observed_at - record["_admitted_at"]
                if output["finished"]:
                    if "finish_reason" in output and output["finish_reason"] not in {"stop", "length"}:
                        raise RuntimeError(f"engine terminated unsuccessfully: {record['sample_id']}: {output['finish_reason']}")
                    if record["_ignore_eos"] and output.get("finish_reason") == "stop":
                        raise RuntimeError(f"fixed-length request stopped on EOS: {record['sample_id']}")
                    record["finished"] = True
                    record["latency_s"] = observed_at - record["_admitted_at"]
                    if "finish_reason" in output:
                        record["finish_reason"] = str(output["finish_reason"])
            if not unfinished:
                break

        elapsed_s = timestamp() - phase_start
        if elapsed_s <= 0:
            raise RuntimeError("phase duration must be positive")
        for record in records.values():
            if not record["finished"] or not record["output_token_ids"]:
                raise RuntimeError(f"engine ended without completing {record['sample_id']}")
            if record["_ignore_eos"] and len(record["output_token_ids"]) != record["_max_new_tokens"]:
                raise RuntimeError(f"fixed-length request ended early: {record['sample_id']}")
        requests = [{key: value for key, value in record.items() if not key.startswith("_")} for record in records.values()]
        return {
            "measurement_boundary": BOUNDARY,
            "elapsed_s": elapsed_s,
            "requests": requests,
            "completed_requests": len(requests),
        }
    except BaseException:
        if hasattr(engine, "abort_request"):
            with suppress(Exception):
                engine.abort_request(list(records))
        raise
