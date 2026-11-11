"""Persistent DeepInfer token transport shared by Full and S2.

This is a live engine adapter, not a replay executor or a chat formatter. All
eight DP ranks must run the same request command and drain loop. Only rank 0
submits a real request; other ranks participate in native collective steps.
The caller owns process startup, the HTTP queue, and the S2 admission hooks.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable


def token_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class TokenRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    arm: str
    received_s: float
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    # Explicit rendered-token spans are supplied by a verified chat codec.
    documents: tuple[dict, ...] = ()

    def validate(self, max_model_len=270336):
        if not self.request_id or self.arm not in ("full", "s2", "prefix"):
            raise ValueError("request identity/arm missing")
        if not self.prompt_token_ids or any(type(i) is not int or not 0 <= i < 2**31
                                            for i in self.prompt_token_ids):
            raise ValueError("nonempty int32 token IDs required")
        if type(self.max_tokens) is not int or not 0 < self.max_tokens <= max_model_len-len(self.prompt_token_ids):
            raise ValueError("request exceeds context capacity")
        if not math.isfinite(self.received_s) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("invalid request clock/temperature")
        if not 0 < self.top_p <= 1 or type(self.seed) is not int:
            raise ValueError("invalid sampling parameters")


class NativeEnginePort:
    """Copies borrowed native output scalars before the next engine step.

    Tests inject the native API boundary. Production defaults import the actual
    DeepInfer RequestOutput getters, SamplingParams, and numpy only at binding.
    """

    def __init__(self, engine, *, output_api=None, array_factory=None, params_factory=None):
        if output_api is None:
            from deepinfer._C.engine import RequestOutput
            output_api = RequestOutput
        if array_factory is None:
            import numpy as np
            array_factory = lambda ids: np.asarray(ids, dtype=np.int32)
        if params_factory is None:
            from deepinfer.sampling_params import SamplingParams
            params_factory = SamplingParams
        self.engine, self.output_api = engine, output_api
        self.array_factory, self.params_factory = array_factory, params_factory

    def submit(self, request):
        params = self.params_factory(temperature=request.temperature, top_p=request.top_p,
            top_k=-1, max_tokens=request.max_tokens, ignore_eos=False, seq_seed=request.seed)
        mm = (dict(image_bytes=[], image_sizes=[])
              if self.engine.multi_modal_processor is not None else None)
        sequence = self.engine.add_request(user_id="s2-agent", request_id=request.request_id,
            sampling_params=params, prompt_token_ids=self.array_factory(request.prompt_token_ids),
            multi_modal_data=mm)
        if sequence is None:
            raise RuntimeError("native engine rejected request " + request.request_id)
        return sequence

    def has_unfinished(self):
        # Native implementation performs an all-rank collective. Call on ALL ranks.
        return self.engine.has_unfinished_requests()

    def advance(self, unfinished):
        pointers = self.engine.step() if unfinished else self.engine.flush_inflight_reqs()
        api = self.output_api
        return [dict(request_id=api.get_request_id(p), token_id=api.get_token_id(p),
            prompt_len=api.get_prompt_len(p), finished=api.get_finished(p),
            finish_reason=api.get_finish_reason(p), logprob=api.get_token_logprobs(p))
            for p in pointers.tolist()]


@dataclass
class RequestMetrics:
    request_id: str
    arm: str
    received_s: float
    admitted_s: float
    prepared_s: float | None = None
    submitted_s: float | None = None
    first_raw_token_s: float | None = None
    terminal_s: float | None = None
    drained_s: float | None = None
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None

    def report(self, request):
        if self.first_raw_token_s is None or self.terminal_s is None or self.drained_s is None:
            raise RuntimeError("cannot close an incomplete request")
        return dict(schema="s2-live-native-request-v1", request_id=self.request_id, arm=self.arm,
            input_ids_sha256=token_digest(request.prompt_token_ids), input_tokens=len(request.prompt_token_ids),
            output_ids=self.output_ids, output_ids_sha256=token_digest(self.output_ids),
            output_tokens=len(self.output_ids), finish_reason=self.finish_reason,
            received_s=self.received_s, admitted_s=self.admitted_s, prepared_s=self.prepared_s,
            submitted_s=self.submitted_s, first_raw_token_s=self.first_raw_token_s,
            terminal_s=self.terminal_s, drained_s=self.drained_s,
            queue_s=self.admitted_s-self.received_s,
            prepare_s=self.prepared_s-self.admitted_s,
            raw_ttft_s=self.first_raw_token_s-self.received_s,
            engine_raw_ttft_s=self.first_raw_token_s-self.submitted_s,
            request_elapsed_s=self.drained_s-self.received_s,
            raw_token_is_not_http_first_byte=True, saturated_qps_claim=False,
            live_engine=True, replay=False)


def run_serial_request(port, request, *, rank, barrier, prepare, finish,
                       emit: Callable[[dict], None], clock=time.perf_counter):
    """Execute one broadcast request against a previously created engine.

    `prepare`/`finish` execute on every rank and own controller/bank/batch state.
    Their cost is inside the request clock. `emit` must enqueue locally, never
    wait for a network peer. HTTP-first-byte/content times belong to transport.
    Any exception is fatal to this worker group: the supervising process must
    stop all ranks; independently retrying one collective rank is unsafe.
    """
    request.validate()
    metrics = RequestMetrics(request.request_id, request.arm, request.received_s, clock())
    preparation = prepare(request)
    if preparation is not None and not isinstance(preparation, dict):
        raise TypeError("prepare metadata must be JSON mapping")
    barrier()
    metrics.prepared_s = clock()
    if rank == 0:
        metrics.submitted_s = clock()
        port.submit(request)
    # Prevent an idle rank entering native has_unfinished before rank0 enqueue.
    barrier()
    terminal = False
    while True:
        unfinished = port.has_unfinished()
        outputs = port.advance(unfinished)
        arrived = clock()
        if rank != 0 and outputs:
            raise RuntimeError("inactive DP rank emitted a real request")
        for output in outputs:
            if output["request_id"] != request.request_id or terminal:
                raise RuntimeError("foreign/late native output")
            if output["prompt_len"] != len(request.prompt_token_ids):
                raise RuntimeError("native prompt length changed")
            if not math.isfinite(output["logprob"]):
                raise RuntimeError("nonfinite native output logprob")
            if metrics.first_raw_token_s is None:
                metrics.first_raw_token_s = arrived
            metrics.output_ids.append(output["token_id"])
            if len(metrics.output_ids) > request.max_tokens:
                raise RuntimeError("native output exceeded request budget")
            emit(dict(event="raw_token", arrival_s=arrived, **output))
            if output["finished"]:
                terminal = True
                metrics.terminal_s, metrics.finish_reason = arrived, output["finish_reason"]
        if not unfinished:
            break  # final native ZOS flush has completed on all ranks
    if rank == 0 and (not terminal or metrics.finish_reason not in ("stop", "length")):
        raise RuntimeError("engine drained without a successful terminal notification")
    completion = finish(request)
    if completion is not None and not isinstance(completion, dict):
        raise TypeError("finish metadata must be JSON mapping")
    barrier()
    metrics.drained_s = clock()
    if rank == 0:
        result = metrics.report(request)
        result.update(preparation=preparation, completion=completion)
        emit(dict(event="request_complete", report=result))
        return result
    return None


def serial_completed_qps(reports, *, observation_start_s=None, observation_end_s=None,
                         failed_requests=0, fallback_requests=0):
    """Completed requests over an explicit, failure-inclusive observation window.

    With no explicit window this is only a successful-report interval diagnostic.
    A benchmark must use its actual start/end even when no request succeeds.
    """
    explicit = observation_start_s is not None or observation_end_s is not None
    if explicit:
        if observation_start_s is None or observation_end_s is None:
            raise ValueError("both observation bounds required")
        start, end = observation_start_s, observation_end_s
    elif reports:
        start = min(r["received_s"] for r in reports)
        end = max(r["drained_s"] for r in reports)
    else:
        return dict(completed_requests=0, window_s=0.0, qps=None, concurrency=1,
                    diagnostic_success_interval=True, benchmark_qps_eligible=False)
    if not all(math.isfinite(v) for v in (start, end)) or end <= start:
        raise ValueError("nonpositive measurement window")
    if any(r["received_s"] < start or r["drained_s"] > end for r in reports):
        raise ValueError("request outside observation window")
    return dict(completed_requests=len(reports), failed_requests=failed_requests,
                fallback_requests=fallback_requests, window_s=end-start,
                observation_start_s=start, observation_end_s=end,
                qps=len(reports)/(end-start), concurrency=1, saturated_qps_claim=False,
                diagnostic_success_interval=not explicit, benchmark_qps_eligible=explicit)
