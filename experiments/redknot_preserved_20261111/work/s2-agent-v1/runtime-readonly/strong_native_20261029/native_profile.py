"""Isolated SM90 native *candidate* profiles for the pinned DeepInfer PR148.

This module imports no GPU package and changes nothing on import.  It does not
patch native attention, CSA2 maps, model weights, or frozen benchmark files.
The profile preserves native YOCO compact and offers matched MTP5/MTP0 arms;
CUDA graphs and variable-length speculation require a separate qualification.

Use ``scoped_base_profile`` around preflight, and AGAIN inside every spawned
worker, before importing DeepInfer.  Spawn does not inherit Python monkeypatches.
The caller must construct create_llm with ``profile.llm_kwargs()``: the frozen
base.worker hardcodes lookahead_tokens=0 and is not an MTP5 worker.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping


SCHEMA = "deepinfer-pr148-sm90-native-candidate-v1"
_SCOPE_LOCK = threading.RLock()
_MISSING = object()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _arg_settings(args: Any = None) -> tuple[int, int]:
    batch = getattr(args, "batch_invariant", 1)
    interval = getattr(args, "prefill_attempt_interval", 1)
    if type(batch) is not int or batch not in (0, 1):
        raise ValueError("batch_invariant must be integer 0 or 1")
    return batch, _positive_int(interval, "prefill_attempt_interval")


@dataclass(frozen=True)
class NativeProfile:
    """Small, serializable allowlist, not a claim of globally best performance."""

    mtp_tokens: int = 5
    keep_tokens: int = 140
    cuda_graphs: bool = False
    mtp_varlen: bool = False

    def __post_init__(self) -> None:
        if type(self.mtp_tokens) is not int or self.mtp_tokens not in (0, 5):
            raise ValueError("this candidate supports only matched MTP0 or native MTP5")
        if type(self.keep_tokens) is not int or self.keep_tokens != 140:
            raise ValueError("this candidate fixes YOCO keep_tokens=140; qualify other windows separately")
        if self.cuda_graphs is not False:
            raise ValueError("whole-model SM90 CUDA graphs are pending separate qualification")
        if self.mtp_varlen is not False:
            raise ValueError("MTP varlen and decode profiling are pending separate qualification")

    @property
    def name(self) -> str:
        return f"sm90-compact140-eager-mtp{self.mtp_tokens}-candidate"

    def environment(self, args: Any = None) -> dict[str, str]:
        batch, interval = _arg_settings(args)
        return {
            "DEEPINFER_USE_NEXTN_DECODE": str(self.mtp_tokens),
            "DEEPINFER_ENABLE_MTP_VARLEN": "0",
            "DEEPINFER_MAX_BACKTRACKING_TOKENS": str(self.mtp_tokens + 1),
            "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE": "compact",
            "DEEPINFER_YOCO_PREFILL_KEEP_TOKENS": str(self.keep_tokens),
            "DEEPINFER_ENFORCE_EAGER": "1",
            "DEEPINFER_DFLASH_ENFORCE_EAGER": "1",
            "DEEPINFER_ZERO_OVERHEAD_SCHEDULING": "1",
            "DEEPINFER_BATCH_INVARIANT": str(batch),
            "DEEPINFER_PREFILL_ATTEMPT_INTERVAL": str(interval),
            "DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL": str(interval),
            # Keep expert placement identical to the frozen experiment.  These
            # controls are disclosed, not presented as globally optimal EPLB.
            "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "0",
            "DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE": "0",
            "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "0",
        }

    def llm_kwargs(self) -> dict[str, Any]:
        # Native examples use min(1,nextn), NOT nextn and NOT unconditional 0.
        # decode_sequence=False disables Python detokenization, not generation.
        return {
            "lookahead_tokens": min(1, self.mtp_tokens),
            "chunked_prefill": True,
            "decode_sequence": False,
        }

    def identity(self, args: Any = None) -> dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "name": self.name,
            "profile": asdict(self),
            "environment": self.environment(args),
            "llm_kwargs": self.llm_kwargs(),
            "profile_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "csa2_policy": "preserve_checkpoint_native_maps_without_overrides",
            "qualification": "SM90 native candidate; runtime, quality, and performance must be measured",
            "cuda_graph_status": "disabled_pending_separate_whole_model_qualification",
            "quality_qualified": False,
            "performance_qualified": False,
        }
        return {**body, "sha256": _sha(body)}


def validate_args(profile: NativeProfile, args: Any) -> None:
    """Fail early on layouts not covered by this isolated eight-GPU candidate."""
    profile.environment(args)
    if getattr(args, "cp", 1) != 1 or getattr(args, "ep", 8) != 8:
        raise ValueError("this SM90 candidate requires EP8/CP1")
    if getattr(args, "dp", None) not in (None, 8):
        raise ValueError("this SM90 candidate requires DP8")


@contextmanager
def scoped_base_profile(
    base: Any,
    profile: NativeProfile,
    *,
    args: Any = None,
    apply_environment: bool = False,
) -> Iterator[dict[str, str]]:
    """Temporarily bind base preflight/worker configuration, restoring on errors.

    ``apply_environment=False`` is for CPU preflight.  Spawned worker entrypoints
    must enter with ``apply_environment=True`` before native imports and retain
    this context for the lifetime of that worker's engine.  Runtime observations
    must subsequently pass ``validate_runtime``.  No import-cache rearming is
    attempted: a worker that already imported DeepInfer is rejected.

    The lock serializes scopes within this process only; processes remain fully
    independent.  The old mapping/function objects are restored, not rewritten.
    """
    validate_args(profile, args)
    if not isinstance(getattr(base, "FIXED_ENV", None), dict):
        raise TypeError("base.FIXED_ENV must be a dictionary")
    if not callable(getattr(base, "effective_env", None)):
        raise TypeError("base.effective_env must be callable")
    if type(apply_environment) is not bool:
        raise TypeError("apply_environment must be bool")
    with _SCOPE_LOCK:
        if apply_environment and any(n == "deepinfer" or n.startswith("deepinfer.") for n in sys.modules):
            raise RuntimeError("apply native profile before importing DeepInfer; cached envs cannot be changed safely")
        old_fixed, old_effective = base.FIXED_ENV, base.effective_env
        saved_env: dict[str, object] = {}

        def effective_env(runtime_args: Any) -> dict[str, str]:
            validate_args(profile, runtime_args)
            previous = old_effective(runtime_args)
            if not isinstance(previous, dict):
                raise TypeError("base.effective_env must return a dictionary")
            return {**previous, **profile.environment(runtime_args)}

        try:
            base.FIXED_ENV = {**old_fixed, **profile.environment(args)}
            base.effective_env = effective_env
            selected = effective_env(args)
            if apply_environment:
                saved_env = {key: os.environ.get(key, _MISSING) for key in selected}
                os.environ.update(selected)
            yield dict(selected)
        finally:
            base.FIXED_ENV, base.effective_env = old_fixed, old_effective
            for key, value in saved_env.items():
                if value is _MISSING:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = str(value)


def validate_runtime(
    profile: NativeProfile, observed: Mapping[str, Any], *, args: Any = None,
) -> dict[str, Any]:
    """Validate measured native fields; never infers them from requested env.

    Caller records actual runner.nextn/use_cuda_graph/dflash_use_cuda_graph/
    yoco_prefill_truncation_mode/window_size; resolved native envs; cache_config
    lookahead_tokens; MTP_module presence; real device compute capability;
    engine/layout; and independent checkpoint vs. loaded-model CSA2 maps.
    Passing this function certifies configuration, NOT quality or performance.
    """
    validate_args(profile, args)
    required = {
        "nextn", "use_cuda_graph", "dflash_use_cuda_graph",
        "yoco_prefill_truncation_mode", "window_size", "keep_tokens",
        "max_backtracking_tokens", "mtp_varlen", "batch_invariant",
        "lookahead_tokens", "mtp_module_present", "zero_overhead_scheduling",
        "cp_size", "compute_capability", "engine_type",
        "csa2_maps", "expected_csa2_maps",
    }
    missing = sorted(required.difference(observed))
    if missing:
        raise ValueError(f"runtime evidence is incomplete: {missing}")
    batch, _ = _arg_settings(args)
    expected = {
        "nextn": profile.mtp_tokens,
        "use_cuda_graph": False,
        "dflash_use_cuda_graph": False,
        "yoco_prefill_truncation_mode": "compact",
        "keep_tokens": profile.keep_tokens,
        "max_backtracking_tokens": profile.mtp_tokens + 1,
        "mtp_varlen": False,
        "batch_invariant": bool(batch),
        "lookahead_tokens": min(1, profile.mtp_tokens),
        "mtp_module_present": bool(profile.mtp_tokens),
        "zero_overhead_scheduling": True,
        "cp_size": 1,
        "engine_type": "NORMAL",
    }
    for key, value in expected.items():
        actual = observed[key]
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"runtime {key}={actual!r} does not match expected {value!r}")
    capability = observed["compute_capability"]
    if (not isinstance(capability, (list, tuple)) or len(capability) != 2
            or any(type(x) is not int for x in capability) or tuple(capability) != (9, 0)):
        raise ValueError("this candidate must be runtime-verified on SM90")
    window = _positive_int(observed["window_size"], "runtime window_size")
    minimum_keep = window + observed["max_backtracking_tokens"]
    if profile.keep_tokens < minimum_keep:
        raise ValueError(f"compact keep_tokens={profile.keep_tokens} cannot cover window+backtracking={minimum_keep}")
    expected_maps, maps = observed["expected_csa2_maps"], observed["csa2_maps"]
    if not isinstance(expected_maps, (dict, list)) or not expected_maps:
        raise ValueError("expected_csa2_maps must be nonempty checkpoint-derived JSON maps")
    if _json(maps) != _json(expected_maps):
        raise ValueError("loaded CSA2 maps differ from the native checkpoint maps")
    # JSON roundtrip takes a detached, JSON-safe snapshot of supplied evidence.
    snapshot = json.loads(_json(dict(observed)))
    body = {
        "schema": SCHEMA,
        "profile": profile.identity(args),
        "observed": snapshot,
        "native_csa2_maps_sha256": _sha(expected_maps),
        "validated_runtime_configuration": True,
        "quality_qualified": False,
        "performance_qualified": False,
    }
    return {**body, "sha256": _sha(body)}
