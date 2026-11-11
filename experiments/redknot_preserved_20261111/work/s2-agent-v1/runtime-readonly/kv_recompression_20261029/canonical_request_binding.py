"""Bind canonical KV hooks to verified native scheduled prefill requests.

This isolated instance patch does not change native scheduling or KV policy.
An active controller alone is NOT proof of ownership: EP workers execute a
one-row native placeholder while another worker has a real request. Only the
native SequenceMetadata request ID plus exact scheduled token/position checks
can produce a binding. Decode remains native and reads the persistent cache
populated during the verified candidate prefill.

The marker is attached after _prepare_input_driver returns because native
recursive_fn reconstructs InputMetadata during its CPU-to-GPU transfer and
would discard an earlier dynamic attribute. Imports of torch/native runtime
are lazy; CPU tests inject operations and make no GPU parity claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import Any, Callable


_MARKER = "canonical_request_binding"
_INSTALLATION = "_canonical_request_binding_installation"
_MISSING = object()


@dataclass(frozen=True)
class RequestBindingOps:
    context_parallel_size: Callable[[], int]
    allow_cuda_sync: Callable[[], Any]
    read_integer_vector: Callable[[Any, str], tuple[int, ...]]

    @classmethod
    def from_deepinfer(cls) -> "RequestBindingOps":
        import torch
        from deepinfer.model_executor.parallel_utils.parallel_state import PG
        from deepinfer.utils import allow_cuda_sync

        def read_integer_vector(value: Any, label: str) -> tuple[int, ...]:
            if (not isinstance(value, torch.Tensor) or value.ndim != 1
                    or value.dtype not in (torch.int32, torch.int64)):
                raise ValueError(f"native {label} must be a one-dimensional integer tensor")
            return tuple(value.detach().cpu().tolist())

        return cls(lambda: PG.context_parallel_group().size(), allow_cuda_sync,
                   read_integer_vector)


class RequestBindingHandle:
    """Own one instance override; never overwrite a later unrelated override."""

    def __init__(self, runner: Any, previous: Any, wrapper: Any):
        self.runner = runner
        self.previous = previous
        self.wrapper = wrapper
        self.closed = False

    def restore(self) -> None:
        if self.closed:
            return
        if self.runner.__dict__.get("_prepare_input_driver") is not self.wrapper:
            raise RuntimeError("canonical request binding no longer owns the runner method")
        if self.previous is _MISSING:
            delattr(self.runner, "_prepare_input_driver")
        else:
            self.runner._prepare_input_driver = self.previous
        if self.runner.__dict__.get(_INSTALLATION) is self:
            delattr(self.runner, _INSTALLATION)
        self.closed = True

    def __enter__(self) -> "RequestBindingHandle":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.restore()


def install_request_binding(
    model_runner: Any,
    get_request: Callable[[], dict[str, Any] | None],
    *,
    native_ops: RequestBindingOps | None = None,
) -> RequestBindingHandle:
    """Install a bounded, instance-only verified-prefill marker.

    get_request returns None or {request_id, phase, token_ids}. token_ids is
    the exact frozen complete prompt, not the current chunk. A verified marker
    is {request_id, phase, start, end}, with half-open absolute prompt offsets.
    Dummy, unrelated, and decode steps receive marker=None without GPU reads.
    Matching prefill with invalid ownership/layout fails closed before forward.
    """
    if not callable(get_request):
        raise TypeError("get_request must be callable")
    if getattr(model_runner, _INSTALLATION, None) is not None:
        raise ValueError("canonical request binding is already installed")
    original = model_runner._prepare_input_driver
    previous = model_runner.__dict__.get("_prepare_input_driver", _MISSING)
    ops = native_ops

    @wraps(original)
    def wrapped(runner: Any, *args: Any, **kwargs: Any):
        nonlocal ops
        sequences = args[0] if args else kwargs.get("seq_metadata_list")
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("unexpected native _prepare_input_driver result contract")
        metadata, total_q, run_cuda_graph, _graph, _graph_size = result
        # Reused metadata must never retain a prior real request's marker.
        setattr(metadata, _MARKER, None)
        request = get_request()
        if request is None:
            return result
        if not isinstance(request, dict):
            raise TypeError("scheduled request accessor must return a dict or None")
        request_id = request.get("request_id")
        phase = request.get("phase")
        if (not isinstance(request_id, str) or not request_id or request_id == "placeholder"
                or not isinstance(phase, str) or not phase):
            raise ValueError("scheduled request needs a non-placeholder ID and explicit phase")
        if not isinstance(sequences, (list, tuple)):
            raise ValueError("native scheduled sequences must be a list or tuple")
        matching = [seq for seq in sequences
                    if getattr(seq, "request_id", None) == request_id
                    and getattr(seq, "is_prompt", False)
                    and not getattr(seq, "is_placeholder", False)]
        if not matching:
            return result
        if len(sequences) != 1 or len(matching) != 1:
            raise ValueError("canonical prefill binding requires exactly one native real request")
        seq = matching[0]
        if getattr(seq, "is_placeholder", None) is not False:
            raise ValueError("native sequence must explicitly identify itself as non-placeholder")
        if (getattr(runner, "use_cuda_graph", None) is not False or run_cuda_graph is not False
                or getattr(metadata, "is_prompt", None) is not True
                or getattr(metadata, "num_reqs", None) != 1):
            raise ValueError("canonical prefill binding requires eager single-request metadata")
        if ops is None:
            ops = RequestBindingOps.from_deepinfer()
        if ops.context_parallel_size() != 1:
            raise ValueError("canonical prefill binding requires CP1")
        tokens = request.get("token_ids")
        if not isinstance(tokens, (list, tuple)) or not tokens:
            raise ValueError("scheduled request requires frozen nonempty token IDs")
        if any(type(token) is not int or not 0 <= token < 2**31 for token in tokens):
            raise ValueError("frozen token IDs must be nonnegative int32 values")
        seqlen_q, seqlen_k = getattr(seq, "seqlen_q", None), getattr(seq, "seqlen_k", None)
        if type(seqlen_q) is not int or type(seqlen_k) is not int:
            raise ValueError("native scheduled sequence lengths must be integers")
        start, end = seqlen_k - seqlen_q, seqlen_k
        if not 0 <= start < end <= len(tokens) or total_q != end - start:
            raise ValueError("native scheduled prefill range is outside the frozen prompt")
        with ops.allow_cuda_sync():
            actual_tokens = ops.read_integer_vector(metadata.input_ids, "input_ids")
            actual_positions = ops.read_integer_vector(metadata.positions, "positions")
        if (actual_tokens != tuple(tokens[start:end])
                or actual_positions != tuple(range(start, end))):
            raise ValueError("native scheduled tokens/positions differ from the frozen request")
        setattr(metadata, _MARKER,
                {"request_id": request_id, "phase": phase, "start": start, "end": end})
        return result

    wrapper = MethodType(wrapped, model_runner)
    handle = RequestBindingHandle(model_runner, previous, wrapper)
    model_runner._prepare_input_driver = wrapper
    setattr(model_runner, _INSTALLATION, handle)
    return handle
