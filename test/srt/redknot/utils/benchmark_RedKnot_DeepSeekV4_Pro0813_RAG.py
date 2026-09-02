#!/usr/bin/env python3
"""Isolated RAG benchmark for DeepSeek-V4-Pro-0813 / RedKnot MLA.

The default path uses SGLang Engine twice: baseline ``attention_backend=dsv4``
and RedKnot ``attention_backend=redknot_mla``. The RedKnot path keeps the
physical DeepSeek V4 MLA cache and applies RedKnot at logical attention
head granularity inside the MLA backend.

This copy is fail-closed to the official Pro-0813 checkpoint, TP8, 61 logical
layers, 128 attention heads, Indexer Top-K 1024, and 8x B300/Blackwell SM103.
It must never be used as the Flash-0731 reproduction entry.

Examples:
  # One-sample RAG smoke with standard vs RedKnot MLA output comparison.
  CUDA_VISIBLE_DEVICES=0 REDKNOT_N_SAMPLES=1 REDKNOT_LENGTHS=8K \
    python test/srt/redknot/benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py

  # Tune the logical-head MLA policy.
  REDKNOT_MLA_LOCAL_WINDOW=256 REDKNOT_MLA_GLOBAL_HEAD_STRIDE=8 \
    python test/srt/redknot/benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py

  # Pure MLA qualification: capture each 8K artifact only after its real
  # cumulative prefix, then restore it at identical source positions with
  # native DSV4 full candidate scope and skip_first=0.
  REDKNOT_ENGINE_MODE=indexer_hot \
    python test/srt/redknot/benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py

  # Freeze a disjoint data selection, then replay it exactly on later runs.
  REDKNOT_ENGINE_MODE=indexer_hot REDKNOT_IH_DATA_ROW_OFFSET=100 \
    REDKNOT_IH_DATA_MANIFEST_OUT=/tmp/redknot-data-100.json \
    python test/srt/redknot/benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py
  REDKNOT_ENGINE_MODE=indexer_hot \
    REDKNOT_IH_DATA_MANIFEST=/tmp/redknot-data-100.json \
    python test/srt/redknot/benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import re
import secrets
import signal
import stat
import string
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[4]


def _pin_pro0813_python_sources() -> tuple[Path, Path]:
    """Discard ambient Python source overlays before importing SGLang."""

    pro_root = (REPO / "python").resolve()
    flash_raw = os.environ.get(
        "REDKNOT_FLASHMLA_SM103_ROOT", "/data/temp/FlashMLA-sm103-src"
    )
    flash_root = Path(flash_raw).expanduser()
    if not flash_root.is_absolute() or os.pathsep in flash_raw:
        raise RuntimeError(
            "REDKNOT_FLASHMLA_SM103_ROOT must be an absolute colon-free path"
        )
    flash_root = flash_root.resolve()

    def _resolved(entry: str) -> Path:
        return Path(entry or os.getcwd()).expanduser().resolve()

    ambient_raw = os.environ.get("PYTHONPATH", "")
    ambient_entries = {
        _resolved(entry)
        for entry in ambient_raw.split(os.pathsep)
        if entry or ambient_raw
    }
    certified = (pro_root, flash_root)
    retained: list[str] = []
    retained_resolved: set[Path] = set()
    for entry in sys.path:
        resolved = _resolved(entry)
        if resolved in certified:
            continue
        if resolved in ambient_entries or resolved in retained_resolved:
            continue
        retained.append(entry)
        retained_resolved.add(resolved)
    sys.path[:] = [str(pro_root), str(flash_root), *retained]
    os.environ["PYTHONPATH"] = os.pathsep.join(str(root) for root in certified)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONSAFEPATH"] = "1"
    return certified


PRO0813_PYTHON_ROOT, PRO0813_FLASHMLA_ROOT = _pin_pro0813_python_sources()

import torch

from sglang.srt.layers.attention.redknot import (
    dsv4_composite_commit as _dsv4_composite_commit_module,
)
from sglang.srt.layers.attention.redknot.dsv4_composite_commit import (
    COMMIT_SCOPE_FORWARD_RESERVED,
)
from sglang.srt.layers.attention.redknot.pro0813 import (
    scale_policy as _pro0813_scale_policy,
)

_COMMIT_SOURCE = Path(_dsv4_composite_commit_module.__file__).resolve()
if not _COMMIT_SOURCE.is_relative_to(PRO0813_PYTHON_ROOT / "sglang"):
    raise RuntimeError(
        "Pro-0813 benchmark resolved SGLang outside the certified Pro source "
        f"root: {_COMMIT_SOURCE}"
    )
_SCALE_POLICY_SOURCE = Path(_pro0813_scale_policy.__file__).resolve()
if not _SCALE_POLICY_SOURCE.is_relative_to(PRO0813_PYTHON_ROOT / "sglang"):
    raise RuntimeError(
        "Pro-0813 scale policy resolved outside the certified Pro source "
        f"root: {_SCALE_POLICY_SOURCE}"
    )

PRO0813_MODEL_ROOT = Path("/workspace/Models/DeepSeek-V4-Pro-0813")
_requested_model = Path(
    os.path.abspath(
        os.path.expanduser(
            os.environ.get("REDKNOT_MODEL_PATH", str(PRO0813_MODEL_ROOT))
        )
    )
)
if _requested_model != PRO0813_MODEL_ROOT:
    raise ValueError(
        "the Pro-0813 reproduction is pinned to "
        f"{PRO0813_MODEL_ROOT}, got {_requested_model}"
    )
MODEL_PATH = str(PRO0813_MODEL_ROOT)
PRO0813_VARIANT = "deepseek_v4_pro_0813"
PRO0813_TP8_GEOMETRY_DIGEST = (
    "sha256:adca138e64f2da316e94dd62394a51bbf5a89ab0651475579ce1977c59497819"
)
PRO0813_OFFICIAL_CONFIG_SHA256 = (
    "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0"
)
# Compatibility aliases for the broader benchmark result schema. Calibration
# plans and manifests below deliberately use the certified names above.
PRO0813_GEOMETRY_DIGEST = PRO0813_TP8_GEOMETRY_DIGEST
PRO0813_CONFIG_SHA256 = PRO0813_OFFICIAL_CONFIG_SHA256
PRO0813_NUM_LAYERS = 61
PRO0813_NUM_HEADS = 128
PRO0813_INDEX_TOPK = 1024
PRO0813_TP_SIZE = 8
PRO0813_HEADS_PER_TP_RANK = 16
PRO0813_GLOBAL_OUTPUT_GROUPS = 16
PRO0813_DENSE_LAYER_IDS = (0, 1, 2, 58, 59, 60)
PRO0813_REUSABLE_LAYER_IDS = tuple(range(3, 58))
PRO0813_DENSE_PREFIX_LAYERS = 3
PRO0813_DENSE_SUFFIX_LAYERS = 3
PRO0813_O_GROUPS_PER_TP_RANK = 2
PRO0813_HEADS_PER_OUTPUT_GROUP = 8
PRO0813_O_LORA_RANK = 1024
PRO0813_SCALE_POLICY_VERSION = (
    _pro0813_scale_policy.PRO0813_SCALE_POLICY_VERSION
)
PRO0813_SCALE_POLICY_DIGEST = (
    _pro0813_scale_policy.PRO0813_SCALE_POLICY_DIGEST
)
_HEAD_CFG_RAW = os.environ.get("REDKNOT_HEAD_CFG", "")
HEAD_CFG_PATH = (
    str(Path(_HEAD_CFG_RAW).expanduser().resolve()) if _HEAD_CFG_RAW else ""
)
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    str(Path(__file__).resolve().parent.parent / "datasets" / "LongBench" / "data"),
)
DATASETS = [
    x.strip()
    for x in os.environ.get(
        "REDKNOT_DATASETS", "hotpotqa,2wikimqa,musique,multifieldqa_en"
    ).split(",")
    if x.strip()
]
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "1"))
MAX_NEW_TOKENS = int(os.environ.get("REDKNOT_MAX_NEW", "32"))
ENGINE_MODE = os.environ.get("REDKNOT_ENGINE_MODE", "both").lower()

# ── Indexer-hot offline-reuse one-click configuration ────────────────────────
# Metrics from this path are deliberately labelled as measured values.  In
# particular, online-row saving is not reported as total-model FLOPs saving.
IH_PORT = int(os.environ.get("REDKNOT_IH_PORT", "31998"))
IH_CHUNK_TOKENS = int(os.environ.get("REDKNOT_IH_CHUNK_TOKENS", "8192"))
IH_MERGED_PREFILL_TOKENS = int(os.environ.get("REDKNOT_IH_MERGED_PREFILL_TOKENS", "0"))
IH_NUM_CHUNKS = int(os.environ.get("REDKNOT_IH_NUM_CHUNKS", "8"))
IH_QUALIFICATION_PROFILE_PATH = os.environ.get(
    "REDKNOT_IH_QUALIFICATION_PROFILE_PATH", ""
).strip()
IH_QUALIFICATION_PROFILE_SHA256 = os.environ.get(
    "REDKNOT_IH_QUALIFICATION_PROFILE_SHA256", ""
).strip()
if bool(IH_QUALIFICATION_PROFILE_PATH) != bool(
    IH_QUALIFICATION_PROFILE_SHA256
):
    raise ValueError(
        "qualification profile path and SHA-256 must be provided together"
    )
if IH_QUALIFICATION_PROFILE_SHA256 and re.fullmatch(
    r"[0-9a-f]{64}", IH_QUALIFICATION_PROFILE_SHA256
) is None:
    raise ValueError(
        "REDKNOT_IH_QUALIFICATION_PROFILE_SHA256 must be lowercase 64-hex"
    )
IH_HOT_FRAC = float(os.environ.get("REDKNOT_IH_HOT_FRAC", "0.50"))
IH_SELECTION_POLICY = os.environ.get(
    "REDKNOT_IH_SELECTION_POLICY", "checkpoint_islands"
)
IH_CHECKPOINT_STRIDE = int(os.environ.get("REDKNOT_IH_CHECKPOINT_STRIDE", "512"))
_IH_TARGET_DOCUMENT_TOKENS = IH_NUM_CHUNKS * IH_CHUNK_TOKENS
IH_CHECKPOINT_MAX_ISLANDS = int(
    os.environ.get(
        "REDKNOT_IH_CHECKPOINT_MAX_ISLANDS",
        str(
            _pro0813_scale_policy.PRO0813_CHECKPOINT_MAX_ISLANDS.get(
                _IH_TARGET_DOCUMENT_TOKENS, 64
            )
        ),
    )
)
IH_ACTIVE_BUDGET_RATIO = float(
    os.environ.get(
        "REDKNOT_IH_ACTIVE_BUDGET_RATIO",
        str(_pro0813_scale_policy.PRO0813_STANDARD_ACTIVE_RATIO),
    )
)
IH_MIN_REALIZED_ACTIVE_RATIO = float(
    os.environ.get("REDKNOT_IH_MIN_REALIZED_ACTIVE_RATIO", "0")
)
IH_QUERY_PROTECTION_TOKENS = int(
    os.environ.get(
        "REDKNOT_IH_QUERY_PROTECTION_TOKENS",
        str(_pro0813_scale_policy.PRO0813_QUERY_PROTECTION_TOKENS),
    )
)
_IH_GENERALIZED_ADAPTIVE_CONTROLLER_RAW = os.environ.get(
    "REDKNOT_IH_GENERALIZED_ADAPTIVE_CONTROLLER", "0"
)
if _IH_GENERALIZED_ADAPTIVE_CONTROLLER_RAW not in ("0", "1"):
    raise ValueError(
        "REDKNOT_IH_GENERALIZED_ADAPTIVE_CONTROLLER must be exactly 0 or 1"
    )
IH_GENERALIZED_ADAPTIVE_CONTROLLER = (
    _IH_GENERALIZED_ADAPTIVE_CONTROLLER_RAW == "1"
)
_IH_GENERALIZED_ADAPTIVE_CONTROLLER_VERSION = "lexical_entropy_top2_bucket_v4"
_IH_GENERALIZED_STRONG_ACTIVE_RATIO = float(
    os.environ.get(
        "REDKNOT_IH_GENERALIZED_STRONG_ACTIVE_RATIO",
        str(_pro0813_scale_policy.PRO0813_STRONG_ACTIVE_RATIO),
    )
)
_IH_GENERALIZED_MEDIUM_ACTIVE_RATIO = float(
    os.environ.get(
        "REDKNOT_IH_GENERALIZED_MEDIUM_ACTIVE_RATIO",
        str(_pro0813_scale_policy.PRO0813_STANDARD_ACTIVE_RATIO),
    )
)
_IH_GENERALIZED_DIFFUSE_ACTIVE_RATIO = float(
    os.environ.get(
        "REDKNOT_IH_GENERALIZED_DIFFUSE_ACTIVE_RATIO",
        str(_pro0813_scale_policy.PRO0813_DIFFUSE_ACTIVE_RATIO),
    )
)
_IH_GENERALIZED_ADAPTIVE_ROW_RATIOS = (
    _IH_GENERALIZED_STRONG_ACTIVE_RATIO,
    _IH_GENERALIZED_MEDIUM_ACTIVE_RATIO,
    _IH_GENERALIZED_DIFFUSE_ACTIVE_RATIO,
)
if any(
    not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0
    for ratio in _IH_GENERALIZED_ADAPTIVE_ROW_RATIOS
):
    raise ValueError("generalized adaptive row ratios must be in (0, 1]")
if tuple(sorted(_IH_GENERALIZED_ADAPTIVE_ROW_RATIOS)) != (
    _IH_GENERALIZED_ADAPTIVE_ROW_RATIOS
):
    raise ValueError(
        "generalized adaptive row ratios must be monotonic: strong <= medium <= diffuse"
    )
_IH_GENERALIZED_ADAPTIVE_QUERY_TOKENS = (8192, 16384, 32768)
_IH_GENERALIZED_ADAPTIVE_POLICY = {
    "version": _IH_GENERALIZED_ADAPTIVE_CONTROLLER_VERSION,
    "inputs": "query_token_ids_and_frozen_document_token_ids_only",
    "forbidden_inputs": (
        "dataset_name",
        "gold_answer",
        "expected_chunk",
        "model_output",
        "hidden_state",
    ),
    "strong": {
        "min_top1_share": 0.40,
        "min_margin": 0.60,
        "max_normalized_entropy": 0.88,
        "active_token_budget_ratio": _IH_GENERALIZED_STRONG_ACTIVE_RATIO,
        "query_protection_tokens": 8192,
        "query_protection_documents": 1,
    },
    "medium": {
        "min_top1_share": 0.27,
        "min_margin": 0.35,
        # The old 0.95 boundary admitted nearly-uniform sketches into the
        # one-document shape.  Promote the high-entropy tail to the existing
        # conservative diffuse shape.  This remains output-blind and adds no
        # fourth kernel/graph specialization.
        "max_normalized_entropy": 0.93,
        "active_token_budget_ratio": _IH_GENERALIZED_MEDIUM_ACTIVE_RATIO,
        "query_protection_tokens": 16384,
        "query_protection_documents": 1,
    },
    "diffuse": {
        "active_token_budget_ratio": _IH_GENERALIZED_DIFFUSE_ACTIVE_RATIO,
        # Diffuse questions deliberately protect two 16K evidence windows.
        # The v2 8K-per-document windows covered the expected source but still
        # missed a multi-hop answer in the frozen MuSiQue qualification set.
        # This remains output-blind: the two documents and their windows are
        # selected solely by the pre-inference lexical sketch.
        "query_protection_tokens": 32768,
        "query_protection_documents": 2,
    },
}
IH_HOT_MAX_PER_SEGMENT_RATIO = float(
    os.environ.get("REDKNOT_IH_HOT_MAX_PER_SEGMENT_RATIO", "0.75")
)
IH_BOUNDARY = int(os.environ.get("REDKNOT_IH_BOUNDARY", "128"))
IH_RELEVANCE_LAST = os.environ.get("REDKNOT_IH_RELEVANCE_LAST", "0") == "1"
IH_RELEVANCE_FIRST = os.environ.get("REDKNOT_IH_RELEVANCE_FIRST", "1") == "1"
IH_SKIP_PREFIX_RECOMPUTE = (
    os.environ.get("REDKNOT_IH_SKIP_PREFIX_RECOMPUTE", "0") == "1"
)
IH_NUM_QUERIES = int(os.environ.get("REDKNOT_IH_NUM_QUERIES", "5"))
# Dataset selection is deliberately independent from the model seed.  The
# default remains the historical behaviour (start at JSONL row zero), while a
# row offset or a previously emitted manifest makes disjoint/replayed runs
# explicit.  Exclusion manifests are separated with the platform path
# separator (":" on Linux/macOS).
IH_DATA_ROW_OFFSET = int(
    os.environ.get(
        "REDKNOT_IH_DATA_ROW_OFFSET",
        os.environ.get("REDKNOT_IH_DATA_OFFSET", "0"),
    )
)
_IH_DATA_MANIFEST_RAW = os.environ.get("REDKNOT_IH_DATA_MANIFEST", "")
IH_DATA_MANIFEST = (
    str(Path(_IH_DATA_MANIFEST_RAW).expanduser().resolve())
    if _IH_DATA_MANIFEST_RAW
    else ""
)
_IH_DATA_MANIFEST_OUT_RAW = os.environ.get(
    "REDKNOT_IH_DATA_MANIFEST_OUT", ""
)
IH_DATA_MANIFEST_OUT = (
    str(Path(_IH_DATA_MANIFEST_OUT_RAW).expanduser().resolve())
    if _IH_DATA_MANIFEST_OUT_RAW
    else ""
)
IH_DATA_EXCLUDE_MANIFESTS = tuple(
    str(Path(value).expanduser().resolve())
    for value in os.environ.get(
        "REDKNOT_IH_DATA_EXCLUDE_MANIFESTS", ""
    ).split(os.pathsep)
    if value.strip()
)
IH_PURE_PROMPT_MODE = os.environ.get(
    "REDKNOT_IH_PURE_PROMPT_MODE", "raw_suffix"
).strip()
if IH_PURE_PROMPT_MODE not in {"raw_suffix", "official_rag_v1"}:
    raise ValueError(
        "REDKNOT_IH_PURE_PROMPT_MODE must be raw_suffix or official_rag_v1"
    )
IH_LONG_OUTPUT_TOKENS = int(
    os.environ.get("REDKNOT_IH_LONG_OUTPUT_TOKENS", "0")
)
if IH_LONG_OUTPUT_TOKENS not in {0, 30, 50}:
    raise ValueError(
        "REDKNOT_IH_LONG_OUTPUT_TOKENS must be exactly 0, 30, or 50"
    )
_IH_PROMPT_MANIFEST_RAW = os.environ.get(
    "REDKNOT_IH_PROMPT_MANIFEST", ""
)
IH_PROMPT_MANIFEST = (
    str(Path(_IH_PROMPT_MANIFEST_RAW).expanduser().resolve())
    if _IH_PROMPT_MANIFEST_RAW
    else ""
)
_IH_PROMPT_MANIFEST_OUT_RAW = os.environ.get(
    "REDKNOT_IH_PROMPT_MANIFEST_OUT", ""
)
IH_PROMPT_MANIFEST_OUT = (
    str(Path(_IH_PROMPT_MANIFEST_OUT_RAW).expanduser().resolve())
    if _IH_PROMPT_MANIFEST_OUT_RAW
    else ""
)
IH_EXPECTED_DATA_SELECTION_SHA256 = os.environ.get(
    "REDKNOT_IH_EXPECTED_DATA_SELECTION_SHA256", ""
).strip()
IH_EXPECTED_QUERY_ROW_ID = int(
    os.environ.get("REDKNOT_IH_EXPECTED_QUERY_ROW_ID", "-1")
)
_IH_EXPECTED_QUERY_ROW_IDS_RAW = os.environ.get(
    "REDKNOT_IH_EXPECTED_QUERY_ROW_IDS", ""
).strip()
if _IH_EXPECTED_QUERY_ROW_IDS_RAW:
    try:
        IH_EXPECTED_QUERY_ROW_IDS = tuple(
            int(value.strip())
            for value in _IH_EXPECTED_QUERY_ROW_IDS_RAW.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise ValueError(
            "REDKNOT_IH_EXPECTED_QUERY_ROW_IDS must be comma-separated integers"
        ) from error
else:
    IH_EXPECTED_QUERY_ROW_IDS = (
        (IH_EXPECTED_QUERY_ROW_ID,) if IH_EXPECTED_QUERY_ROW_ID >= 0 else ()
    )
if any(value < 0 for value in IH_EXPECTED_QUERY_ROW_IDS) or len(
    set(IH_EXPECTED_QUERY_ROW_IDS)
) != len(IH_EXPECTED_QUERY_ROW_IDS):
    raise ValueError(
        "REDKNOT_IH_EXPECTED_QUERY_ROW_IDS must contain unique non-negative rows"
    )
IH_EXPECTED_DATASET = os.environ.get(
    "REDKNOT_IH_EXPECTED_DATASET", "musique"
).strip()
if not IH_EXPECTED_DATASET:
    raise ValueError("REDKNOT_IH_EXPECTED_DATASET must be non-empty")
IH_EXPECTED_PROMPT_MANIFEST_SHA256 = os.environ.get(
    "REDKNOT_IH_EXPECTED_PROMPT_MANIFEST_SHA256", ""
).strip()
IH_EXPECTED_PROMPT_TEXT_SHA256 = os.environ.get(
    "REDKNOT_IH_EXPECTED_PROMPT_TEXT_SHA256", ""
).strip()
IH_EXPECTED_FULL_INPUT_IDS_SHA256 = os.environ.get(
    "REDKNOT_IH_EXPECTED_FULL_INPUT_IDS_SHA256", ""
).strip()
IH_EXPECTED_FULL_INPUT_TOKENS = int(
    os.environ.get("REDKNOT_IH_EXPECTED_FULL_INPUT_TOKENS", "0")
)
IH_QUALITY_REPEATS = int(os.environ.get("REDKNOT_IH_QUALITY_REPEATS", "3"))
IH_TTFT_ITERS = int(os.environ.get("REDKNOT_IH_TTFT_ITERS", "20"))
IH_TTFT_WARMUP = int(os.environ.get("REDKNOT_IH_TTFT_WARMUP", "3"))
IH_REQUIRE_MODEL_TTFT = (
    os.environ.get("REDKNOT_IH_REQUIRE_MODEL_TTFT", "0") == "1"
)


def _ih_strict_binary_env(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be exactly 0 or 1, got {value!r}")
    return value == "1"


IH_STRICT_PERFORMANCE_CLAIMS = _ih_strict_binary_env(
    "REDKNOT_IH_STRICT_PERFORMANCE_CLAIMS"
)
IH_PERFORMANCE_DIAGNOSTIC_ONLY = not IH_STRICT_PERFORMANCE_CLAIMS
IH_MEASURE_QPS = os.environ.get("REDKNOT_IH_MEASURE_QPS", "1") == "1"
IH_QPS_WARMUP_WAVES = int(
    os.environ.get("REDKNOT_IH_QPS_WARMUP_WAVES", "3")
)
IH_QPS_WAVES = int(os.environ.get("REDKNOT_IH_QPS_WAVES", "10"))


def _ih_parse_qps_concurrencies(raw: str) -> Tuple[int, ...]:
    text = str(raw).strip()
    if not text:
        raise ValueError("REDKNOT_IH_QPS_CONCURRENCIES must not be empty")
    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError(
            "REDKNOT_IH_QPS_CONCURRENCIES must be a comma-separated list "
            "of positive integers"
        ) from exc
    if any(value <= 0 for value in values):
        raise ValueError("QPS concurrencies must all be positive")
    if tuple(sorted(set(values))) != values:
        raise ValueError(
            "QPS concurrencies must be unique and strictly increasing"
        )
    return values


# Default to the only concurrency currently certified with first-document
# prefix isolation. Higher concurrency requires a wave-level seed fence; an
# explicit override remains valid for modes that do not use that prefix.
# A wave contains one simultaneous request per worker, so attempted requests at
# one point are ``concurrency * IH_QPS_WAVES``.
IH_QPS_CONCURRENCIES = _ih_parse_qps_concurrencies(
    os.environ.get("REDKNOT_IH_QPS_CONCURRENCIES", "1")
)
IH_MAX_NEW = int(os.environ.get("REDKNOT_IH_MAX_NEW", "64"))
IH_MIN_TOP1_RATE = float(os.environ.get("REDKNOT_IH_MIN_TOP1_RATE", "1.0"))
IH_MIN_COSINE = float(os.environ.get("REDKNOT_IH_MIN_COSINE", "0.999999"))
IH_MIN_F1_RETENTION = float(os.environ.get("REDKNOT_IH_MIN_F1_RETENTION", "1.0"))
IH_MIN_EM_RETENTION = float(os.environ.get("REDKNOT_IH_MIN_EM_RETENTION", "1.0"))
IH_MIN_DENSE_F1 = float(os.environ.get("REDKNOT_IH_MIN_DENSE_F1", "0.0"))
IH_MIN_REUSE_F1 = float(os.environ.get("REDKNOT_IH_MIN_REUSE_F1", "0.0"))
IH_MIN_DENSE_EM = float(os.environ.get("REDKNOT_IH_MIN_DENSE_EM", "0.0"))
IH_MIN_REUSE_EM = float(os.environ.get("REDKNOT_IH_MIN_REUSE_EM", "0.0"))
IH_MIN_TOKEN_AGREEMENT = float(
    os.environ.get("REDKNOT_IH_MIN_TOKEN_AGREEMENT", "1.0")
)
IH_MIN_SPEEDUP = float(os.environ.get("REDKNOT_IH_MIN_SPEEDUP", "5.0"))
IH_MIN_QPS_SPEEDUP = float(
    os.environ.get("REDKNOT_IH_MIN_QPS_SPEEDUP", "1.05")
)
IH_MIN_ROW_SAVING = float(os.environ.get("REDKNOT_IH_MIN_ROW_SAVING", "0.85"))
IH_MLA_OFFLOAD = os.environ.get("REDKNOT_IH_MLA_OFFLOAD", "0") == "1"
_IH_COMBINED_HEADSPLIT_ROW_SPARSE_RAW = os.environ.get(
    "REDKNOT_IH_COMBINED_HEADSPLIT_ROW_SPARSE", "0"
)
if _IH_COMBINED_HEADSPLIT_ROW_SPARSE_RAW not in ("0", "1"):
    raise ValueError(
        "REDKNOT_IH_COMBINED_HEADSPLIT_ROW_SPARSE must be exactly 0 or 1"
    )
IH_COMBINED_HEADSPLIT_ROW_SPARSE = (
    _IH_COMBINED_HEADSPLIT_ROW_SPARSE_RAW == "1"
)
IH_MLA_OFF_DIAGNOSTIC_ABLATION = os.environ.get(
    "REDKNOT_IH_MLA_OFF_DIAGNOSTIC_ABLATION", "full"
)
if IH_MLA_OFF_DIAGNOSTIC_ABLATION not in (
    "full",
    "zoff_only",
    "shared_only",
):
    raise ValueError(
        "REDKNOT_IH_MLA_OFF_DIAGNOSTIC_ABLATION must be exactly full, "
        "zoff_only, or shared_only"
    )
if IH_COMBINED_HEADSPLIT_ROW_SPARSE and (
    IH_MLA_OFF_DIAGNOSTIC_ABLATION not in {"full", "zoff_only"}
):
    raise ValueError(
        "combined headsplit/row-sparse supports only full production or "
        "zoff_only diagnostic execution"
    )
_IH_ROW_SPARSE_CLOSURE_RAW = os.environ.get(
    "REDKNOT_IH_ROW_SPARSE_CLOSURE", "0"
)
if _IH_ROW_SPARSE_CLOSURE_RAW not in ("0", "1"):
    raise ValueError(
        "REDKNOT_IH_ROW_SPARSE_CLOSURE must be exactly 0 or 1"
    )
IH_ROW_SPARSE_CLOSURE = _IH_ROW_SPARSE_CLOSURE_RAW == "1"
if (
    IH_ROW_SPARSE_CLOSURE
    and IH_MLA_OFFLOAD
    and not IH_COMBINED_HEADSPLIT_ROW_SPARSE
):
    raise ValueError(
        "row-sparse closure is a standalone qualification arm and cannot "
        "silently masquerade as pure MLA-offload"
    )
IH_PREFIX_MATERIALIZATION = (
    os.environ.get("REDKNOT_IH_PREFIX_MATERIALIZATION", "0") == "1"
)
IH_PREFIX_MATERIALIZATION_SCOPE = os.environ.get(
    "REDKNOT_IH_PREFIX_MATERIALIZATION_SCOPE",
    "full" if IH_PREFIX_MATERIALIZATION else "none",
).strip().lower()
if IH_PREFIX_MATERIALIZATION_SCOPE not in {"none", "full", "first_document"}:
    raise ValueError(
        "REDKNOT_IH_PREFIX_MATERIALIZATION_SCOPE must be none, full, or "
        "first_document"
    )
if (IH_PREFIX_MATERIALIZATION_SCOPE != "none") != IH_PREFIX_MATERIALIZATION:
    raise ValueError(
        "prefix materialization scope and enable flag disagree"
    )
IH_FULL_PREFIX_MATERIALIZATION = (
    IH_PREFIX_MATERIALIZATION_SCOPE == "full"
)
IH_FIRST_DOCUMENT_PREFIX = (
    IH_PREFIX_MATERIALIZATION_SCOPE == "first_document"
)
if (
    IH_FIRST_DOCUMENT_PREFIX
    and IH_MEASURE_QPS
    and IH_QPS_CONCURRENCIES != (1,)
):
    raise ValueError(
        "first-document prefix isolation currently certifies QPS only at "
        "concurrency=1; higher concurrency requires a wave-level seed fence"
    )
if IH_PREFIX_MATERIALIZATION and not IH_MLA_OFFLOAD:
    raise ValueError("prefix materialization requires pure MLA-offload")
IH_RADIX_EVICTION_POLICY = os.environ.get(
    "REDKNOT_RADIX_EVICTION_POLICY", "lru"
).strip().lower()
if IH_RADIX_EVICTION_POLICY not in {"lru", "lfu", "slru", "priority"}:
    raise ValueError(
        "REDKNOT_RADIX_EVICTION_POLICY must be lru, lfu, slru, or priority"
    )
if IH_PREFIX_MATERIALIZATION and IH_RADIX_EVICTION_POLICY != "lfu":
    raise ValueError(
        "prefix materialization requires REDKNOT_RADIX_EVICTION_POLICY=lfu"
    )
IH_MLA_OFF_EXECUTION_PROFILE = (
    (
        (
            "combined_headsplit_pro0813_independent_rope_full_checkpoint_"
            "rowsparse_3_55_3_v1"
        )
        if IH_MLA_OFF_DIAGNOSTIC_ABLATION == "full"
        else (
            "combined_headsplit_pro0813_independent_rope_zoff_checkpoint_"
            "rowsparse_3_55_3_v1"
        )
    )
    if IH_COMBINED_HEADSPLIT_ROW_SPARSE
    else (
        "pure_headsplit_pro0813_independent_rope_relocation_fullscope_"
        "boundary128_3_55_3_v1"
    )
)
IH_MLA_OFF_HEAD_SCOPE_POLICY = "native_dsv4_full_candidate_scope_v1"
IH_MLA_OFF_GLOBAL_ATTN_IMPL = os.environ.get(
    "REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL", "triton_h1"
).strip()
if IH_MLA_OFF_GLOBAL_ATTN_IMPL != "triton_h1":
    raise ValueError(
        "B300/SM103 Pro-0813 requires the arbitrary-head triton_h1 provider"
    )
_IH_MLA_OFF_GEOMETRY_TEMPLATE_CACHE_RAW = os.environ.get(
    "REDKNOT_MLA_OFF_GEOMETRY_TEMPLATE_CACHE", "0"
)
if _IH_MLA_OFF_GEOMETRY_TEMPLATE_CACHE_RAW not in ("0", "1"):
    raise ValueError(
        "REDKNOT_MLA_OFF_GEOMETRY_TEMPLATE_CACHE must be exactly 0 or 1"
    )
IH_MLA_OFF_GEOMETRY_TEMPLATE_CACHE = (
    _IH_MLA_OFF_GEOMETRY_TEMPLATE_CACHE_RAW == "1"
)
try:
    IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS = int(
        os.environ.get(
            "REDKNOT_IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS", "0"
        )
    )
except ValueError as error:
    raise ValueError(
        "REDKNOT_IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS must be an integer"
    ) from error
if not 0 <= IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS <= 55:
    raise ValueError(
        "REDKNOT_IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS must be in [0, 55]"
    )
if IH_MLA_OFF_DIAGNOSTIC_ABLATION != "full" and not IH_MLA_OFFLOAD:
    raise ValueError("MLA-off diagnostic ablation requires MLA offload")
IH_MLA_OFF_DIAGNOSTIC_ONLY = bool(
    IH_MLA_OFFLOAD and IH_MLA_OFF_DIAGNOSTIC_ABLATION != "full"
)
_IH_MLA_OFF_DIAGNOSTIC_CLAIM_REASON = (
    "non_full_mla_off_diagnostic_ablation"
)
_IH_PERFORMANCE_DIAGNOSTIC_CLAIM_REASON = (
    "diagnostic_performance_opt_out"
)
_IH_MLA_OFF_COMPACT_WOA_RAW = os.environ.get(
    "REDKNOT_IH_MLA_OFF_COMPACT_WOA", "0"
)
if _IH_MLA_OFF_COMPACT_WOA_RAW not in ("0", "1"):
    raise ValueError(
        "REDKNOT_IH_MLA_OFF_COMPACT_WOA must be exactly 0 or 1"
    )
IH_MLA_OFF_COMPACT_WOA = _IH_MLA_OFF_COMPACT_WOA_RAW == "1"
if IH_MLA_OFFLOAD and IH_MLA_OFF_COMPACT_WOA:
    raise ValueError(
        "pure MLA headsplit forbids the legacy selected-row compact wo_a path"
    )
IH_REUSE_HEADS_FULL_SCOPE = (
    os.environ.get("REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE", "1") == "1"
)
IH_MLA_OFF_MAX_BYTES = int(
    os.environ.get("REDKNOT_IH_MLA_OFF_MAX_BYTES", str(8 * 1024**3))
)
IH_TP_SIZE = int(os.environ.get("REDKNOT_IH_TP_SIZE", "8"))
IH_MIN_GPU_FREE_MIB = int(
    os.environ.get(
        "REDKNOT_IH_MIN_GPU_FREE_MIB",
        str(_pro0813_scale_policy.PRO0813_MIN_FREE_BEFORE_LAUNCH_MIB),
    )
)
IH_MLA_OFF_REFRESH_LAYER_STRIDE = int(
    os.environ.get("REDKNOT_IH_MLA_OFF_REFRESH_LAYER_STRIDE", "0")
)
IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS = int(
    os.environ.get(
        "REDKNOT_IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS", "0"
    )
)
_IH_MLA_OFF_QUALIFICATION_ONLY_RAW = os.environ.get(
    "REDKNOT_IH_MLA_OFF_QUALIFICATION_ONLY", "0"
)
if _IH_MLA_OFF_QUALIFICATION_ONLY_RAW not in ("0", "1"):
    raise ValueError(
        "REDKNOT_IH_MLA_OFF_QUALIFICATION_ONLY must be exactly 0 or 1"
    )
IH_MLA_OFF_QUALIFICATION_ONLY = (
    _IH_MLA_OFF_QUALIFICATION_ONLY_RAW == "1"
)
IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS = int(
    os.environ.get(
        "REDKNOT_IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS", "0"
    )
)
_IH_MLA_OFF_QUALIFICATION_CLAIM_REASON = (
    "pure_mla_qualification_only"
)
IH_MLA_OFF_HOT_EXPAND_TOKENS = int(
    os.environ.get("REDKNOT_IH_MLA_OFF_HOT_EXPAND_TOKENS", "0")
)
IH_MIN_HEAD_ROW_SAVING = float(
    os.environ.get("REDKNOT_IH_MIN_HEAD_ROW_SAVING", "0.80")
)
IH_NO_LAUNCH = os.environ.get("REDKNOT_IH_NO_LAUNCH", "0") == "1"
_IH_SERVER_IDENTITY_PATH_RAW = os.environ.get(
    "REDKNOT_IH_SERVER_IDENTITY_PATH", ""
).strip()
IH_SERVER_IDENTITY_NONCE = os.environ.get(
    "REDKNOT_IH_SERVER_IDENTITY_NONCE", ""
).strip()
if bool(_IH_SERVER_IDENTITY_PATH_RAW) != bool(IH_SERVER_IDENTITY_NONCE):
    raise ValueError(
        "REDKNOT_IH_SERVER_IDENTITY_PATH and "
        "REDKNOT_IH_SERVER_IDENTITY_NONCE must be supplied together"
    )
if IH_SERVER_IDENTITY_NONCE and not re.fullmatch(
    r"[0-9a-f]{64}", IH_SERVER_IDENTITY_NONCE
):
    raise ValueError(
        "REDKNOT_IH_SERVER_IDENTITY_NONCE must be 32 random bytes in hex"
    )
if _IH_SERVER_IDENTITY_PATH_RAW:
    if not os.path.isabs(_IH_SERVER_IDENTITY_PATH_RAW):
        raise ValueError("REDKNOT_IH_SERVER_IDENTITY_PATH must be absolute")
    IH_SERVER_IDENTITY_PATH = Path(
        os.path.abspath(_IH_SERVER_IDENTITY_PATH_RAW)
    )
else:
    IH_SERVER_IDENTITY_PATH = None
if IH_NO_LAUNCH and IH_SERVER_IDENTITY_PATH is not None:
    raise ValueError(
        "an owned-server identity path is invalid with REDKNOT_IH_NO_LAUNCH=1"
    )
IH_SERVER_INSTANCE_NONCE = os.environ.get(
    "REDKNOT_IH_SERVER_INSTANCE_NONCE", ""
)
if not IH_NO_LAUNCH and not IH_SERVER_INSTANCE_NONCE:
    IH_SERVER_INSTANCE_NONCE = secrets.token_hex(16)
_IH_SERVER_POLICY_MANIFEST_RAW = os.environ.get(
    "REDKNOT_IH_SERVER_POLICY_MANIFEST", ""
)
IH_SERVER_POLICY_MANIFEST = (
    str(Path(_IH_SERVER_POLICY_MANIFEST_RAW).expanduser().resolve())
    if _IH_SERVER_POLICY_MANIFEST_RAW
    else ""
)
IH_SERVER_LOG = os.environ.get(
    "REDKNOT_IH_SERVER_LOG", "/tmp/redknot_pro0813_ih_server.log"
)
IH_RANK_LOG_DIR = str(
    Path(
        os.environ.get(
            "REDKNOT_IH_RANK_LOG_DIR",
            (
                "/tmp/ranklogs_redknot_pro0813"
                if IH_NO_LAUNCH
                else f"/tmp/redknot_pro0813_ih_ranklogs_{os.getpid()}"
            ),
        )
    )
    .expanduser()
    .resolve()
)
_PRO0813_SERVER_SCRIPT = (
    REPO / "server" / "start_server_redknot_pro0813.sh"
).resolve()
IH_SERVER_SCRIPT = str(
    Path(
        os.environ.get("REDKNOT_IH_SERVER_SCRIPT", str(_PRO0813_SERVER_SCRIPT))
    ).expanduser().resolve()
)
if Path(IH_SERVER_SCRIPT) != _PRO0813_SERVER_SCRIPT:
    raise ValueError(
        "the Pro-0813 benchmark must use start_server_redknot_pro0813.sh"
    )
IH_VENV_PY = os.environ.get(
    "REDKNOT_IH_VENV_PY", "/workspace/RedKnot/.venv_sm103/bin/python"
)
SPARSE_FFN = os.environ.get("REDKNOT_SPARSE_FFN", "0") == "1"
THREE_WAY_CLOSURE = os.environ.get("REDKNOT_THREE_WAY_CLOSURE", "0") == "1"
FFN_IMPORTANCE = os.environ.get("REDKNOT_FFN_IMPORTANCE", "activation")
FFN_MASS = float(os.environ.get("REDKNOT_FFN_MASS", "0.30"))
FFN_MASS_DEEP = float(os.environ.get("REDKNOT_FFN_MASS_DEEP", "0.10"))
FFN_MIN_FULL_RATIO = float(
    os.environ.get("REDKNOT_FFN_MIN_FULL_RATIO", "0.20")
)
FFN_MAX_FULL_RATIO = float(
    os.environ.get("REDKNOT_FFN_MAX_FULL_RATIO", "0.80")
)
FFN_DENSE_SUFFIX_LAYERS = int(
    os.environ.get("REDKNOT_FFN_DENSE_SUFFIX_LAYERS", "0")
)
FFN_BOUNDARY_TOKENS = int(
    os.environ.get("REDKNOT_FFN_BOUNDARY_TOKENS", "128")
)
FFN_BLOCK_TOKENS = int(os.environ.get("REDKNOT_FFN_BLOCK_TOKENS", "0") or 0)
FFN_FREEZE_BLOCK_SELECTION = (
    os.environ.get("REDKNOT_FFN_FREEZE_BLOCK_SELECTION", "0") == "1"
)
if FFN_BLOCK_TOKENS < 0 or (
    FFN_BLOCK_TOKENS and FFN_BLOCK_TOKENS % 128 != 0
):
    raise ValueError(
        "REDKNOT_FFN_BLOCK_TOKENS must be 0 or a positive multiple of 128"
    )
if FFN_FREEZE_BLOCK_SELECTION and not FFN_BLOCK_TOKENS:
    raise ValueError(
        "REDKNOT_FFN_FREEZE_BLOCK_SELECTION requires block selection"
    )
if THREE_WAY_CLOSURE != SPARSE_FFN:
    raise ValueError(
        "qualification three-way closure and token-sparse FFN must be enabled together"
    )
PROGRESSIVE_TOPK_SCHEDULE = os.environ.get(
    "REDKNOT_PROGRESSIVE_TOPK_SCHEDULE", ""
).strip()
RESULT_OUT = os.environ.get("REDKNOT_RESULT_OUT", "")
CHUNK_TOKENS = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "4000"))
SEED = int(os.environ.get("REDKNOT_SEED", "2026"))
ENABLE_REUSE = os.environ.get("REDKNOT_ENABLE_REUSE", "0") == "1"
RUNTIME = os.environ.get("REDKNOT_RUNTIME", "sglang").lower()
USE_COMPILE = os.environ.get("REDKNOT_COMPILE", "0") == "1"
REUSE_KERNEL = os.environ.get("REDKNOT_KERNEL", "fa3_parallel")
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
DTYPE = os.environ.get("REDKNOT_DTYPE", "bf16").lower()
DRY_RUN = os.environ.get("REDKNOT_DRY_RUN", "0") == "1"
TP_SIZE = int(os.environ.get("REDKNOT_TP_SIZE", "8"))
if TP_SIZE != PRO0813_TP_SIZE or IH_TP_SIZE != PRO0813_TP_SIZE:
    raise ValueError("DeepSeek-V4-Pro-0813 reproduction requires TP=8")
MAX_TOTAL_TOKENS = int(os.environ.get("REDKNOT_MAX_TOTAL_TOKENS", "0"))
ENGINE_SWA_FULL_TOKENS_RATIO = float(
    os.environ.get("REDKNOT_SWA_FULL_TOKENS_RATIO", "0")
)
if not 0.0 <= ENGINE_SWA_FULL_TOKENS_RATIO <= 1.0:
    raise ValueError("REDKNOT_SWA_FULL_TOKENS_RATIO must be in [0, 1]")
DISABLE_CUDA_GRAPH = os.environ.get("REDKNOT_DISABLE_CUDA_GRAPH", "0") == "1"
SKIP_SERVER_WARMUP = os.environ.get("REDKNOT_SKIP_SERVER_WARMUP", "0") == "1"
MOE_RUNNER_BACKEND = os.environ.get(
    "REDKNOT_MOE_RUNNER_BACKEND", "flashinfer_mxfp4"
)
MEM_FRACTION_STATIC = os.environ.get("REDKNOT_MEM_FRACTION_STATIC", "")
MLA_DENSE_PREFIX = int(os.environ.get("REDKNOT_MLA_DENSE_PREFIX_LAYERS", "3"))
MLA_DENSE_SUFFIX = int(os.environ.get("REDKNOT_MLA_DENSE_SUFFIX_LAYERS", "3"))
MLA_LOCAL_WINDOW = int(
    os.environ.get(
        "REDKNOT_MLA_LOCAL_WINDOW", os.environ.get("REDKNOT_LOCAL_WINDOW", "128")
    )
)
MLA_GLOBAL_HEAD_STRIDE = int(os.environ.get("REDKNOT_MLA_GLOBAL_HEAD_STRIDE", "8"))
MLA_GLOBAL_LAYER_STRIDE = int(os.environ.get("REDKNOT_MLA_GLOBAL_LAYER_STRIDE", "0"))
MLA_PASS_MODE = os.environ.get("REDKNOT_MLA_PASS_MODE", "headwise")
THINKING_MODE = os.environ.get("REDKNOT_THINKING_MODE", "chat")
REASONING_EFFORT = os.environ.get("REDKNOT_REASONING_EFFORT", "low")
_OFFICIAL_ENCODER = None
_OFFICIAL_ENCODER_IDENTITY = None

# Offline MLA head-locality profiling (analysis-before-compression step).
PROFILE = os.environ.get("REDKNOT_MLA_PROFILE", "0") == "1"
PROFILE_OUT = os.environ.get(
    "REDKNOT_MLA_PROFILE_OUT",
    str(
        Path(__file__).resolve().parent.parent
        / "head_class"
        / "dsv4_pro0813_mla_head_config.json"
    ),
)
PROFILE_COVERAGE = float(os.environ.get("REDKNOT_MLA_PROFILE_COVERAGE", "0.95"))
PROFILE_SAMPLE_Q = int(os.environ.get("REDKNOT_MLA_PROFILE_SAMPLE_Q", "256"))
PROFILE_EXPECTED_HEADS = int(
    os.environ.get("REDKNOT_MLA_PROFILE_EXPECTED_HEADS", "128")
)
PROFILE_GLOBAL_RATIO = float(os.environ.get("REDKNOT_MLA_PROFILE_GLOBAL_RATIO", "0.5"))
PROFILE_WINDOW_SAFETY = float(
    os.environ.get("REDKNOT_MLA_PROFILE_WINDOW_SAFETY", "1.5")
)
PROFILE_SWA_FULL_TOKENS_RATIO = float(
    os.environ.get("REDKNOT_MLA_PROFILE_SWA_FULL_TOKENS_RATIO", "0.30")
)
PROFILE_CHUNKED_PREFILL_SIZE = int(
    os.environ.get("REDKNOT_MLA_PROFILE_CHUNKED_PREFILL_SIZE", "4096")
)

# Paired projected-output drift calibration.  Unlike the attention-distance
# profiler above, this compares the same sampled token rows when a block is run
# alone and when it is embedded in a composed context.  It is the accuracy-first
# selector for expanding local heads across many layers.
DRIFT_PROFILE = os.environ.get("REDKNOT_MLA_DRIFT_PROFILE", "0") == "1"
DRIFT_CHUNK_TOKENS = int(
    os.environ.get("REDKNOT_MLA_DRIFT_CHUNK_TOKENS", "1024")
)
DRIFT_NUM_CHUNKS = int(
    os.environ.get("REDKNOT_MLA_DRIFT_NUM_CHUNKS", "4")
)
DRIFT_SAMPLE_ROWS = int(
    os.environ.get("REDKNOT_MLA_DRIFT_SAMPLE_ROWS", "16")
)
DRIFT_BOUNDARY_ROWS = int(
    os.environ.get("REDKNOT_MLA_DRIFT_BOUNDARY_ROWS", "128")
)
DRIFT_OUT_DIR = os.environ.get("REDKNOT_MLA_DRIFT_OUT_DIR", "")
DRIFT_MAX_CAPTURE_BYTES = int(
    os.environ.get("REDKNOT_MLA_DRIFT_MAX_CAPTURE_BYTES", str(16 << 30))
)
DRIFT_READY_DELAY_SECONDS = float(
    os.environ.get("REDKNOT_MLA_DRIFT_READY_DELAY_SECONDS", "3")
)
DRIFT_HTTP_PORT = int(os.environ.get("REDKNOT_MLA_DRIFT_HTTP_PORT", "0"))
DRIFT_HTTP_TIMEOUT = int(
    os.environ.get("REDKNOT_MLA_DRIFT_HTTP_TIMEOUT", "1800")
)
DRIFT_ARTIFACT_TIMEOUT_SECONDS = float(
    os.environ.get("REDKNOT_MLA_DRIFT_ARTIFACT_TIMEOUT_SECONDS", "60")
)

_LEN_MAP = {
    "4K": 4000,
    "8K": 8000,
    "16K": 16000,
    "24K": 24000,
    "32K": 32000,
    "40K": 40000,
    "64K": 64000,
    "128K": 128000,
    "256K": 256000,
}
LENGTHS = [
    x.strip()
    for x in os.environ.get("REDKNOT_LENGTHS", "8K").split(",")
    if x.strip() in _LEN_MAP
]


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_max(pred: str, golds: Iterable[str]) -> float:
    return max((f1_score(pred, g) for g in golds), default=0.0)


def em_max(pred: str, golds: Iterable[str]) -> float:
    return max((float(_normalize(pred) == _normalize(g)) for g in golds), default=0.0)


def _short_ans(text: str) -> str:
    text = text or ""
    if not text.strip():
        return ""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = re.sub(r"(?i)^\s*(the answer is|answer)\b[:\s]*", "", text, count=1)
    text = re.split(r"(?i)(?:\n\s*question\s*:|\n\s*q\s*:|<\|)", text)[0]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cand = (lines[0] if lines else text.strip()).strip().strip('"').strip("'")
    return re.sub(r"\s*[.。]\s*$", "", cand)


def _trunc(s: str, n: int = 72) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _query_text(question: str) -> str:
    if IH_LONG_OUTPUT_TOKENS:
        lower = IH_LONG_OUTPUT_TOKENS - 5
        upper = IH_LONG_OUTPUT_TOKENS + 5
        return (
            "\n\nAnswer the question using only the documents above. "
            "Put the direct answer alone on the first line. Then give one or "
            "two concise sentences explaining the supporting evidence from "
            "the documents. Do not use bullets or introduce outside facts. "
            f"Keep the complete response between {lower} and {upper} tokens "
            f"(target: about {IH_LONG_OUTPUT_TOKENS} tokens).\n"
            f"Question: {question}\nAnswer:"
        )
    return (
        "\n\nAnswer the question using only the documents above. "
        "Return the shortest exact answer span only, with no explanation.\n"
        f"Question: {question}\nAnswer:"
    )


def _load_official_encoder():
    global _OFFICIAL_ENCODER, _OFFICIAL_ENCODER_IDENTITY
    if _OFFICIAL_ENCODER is not None:
        return _OFFICIAL_ENCODER
    encoding_path = Path(MODEL_PATH) / "encoding" / "encoding_dsv4.py"
    if not encoding_path.is_file():
        raise FileNotFoundError(
            f"DeepSeek V4 official encoder is missing: {encoding_path}"
        )
    # Hash and execute the exact same immutable byte string.  Reading the file
    # once for import and again for the manifest would leave a TOCTOU window in
    # which the recorded encoder differed from the code that built the prompt.
    source_bytes = encoding_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    namespace = {
        "__name__": "redknot_encoding_dsv4_frozen",
        "__file__": str(encoding_path.resolve()),
    }
    exec(compile(source_bytes, str(encoding_path), "exec"), namespace)
    encode_messages = namespace.get("encode_messages")
    if not callable(encode_messages):
        raise RuntimeError(
            f"official DeepSeek V4 encoder has no encode_messages: {encoding_path}"
        )
    required_parameters = {
        "thinking_mode",
        "reasoning_effort",
        "drop_thinking",
        "add_default_bos_token",
    }
    if not required_parameters.issubset(
        inspect.signature(encode_messages).parameters
    ):
        raise RuntimeError(
            "official DeepSeek V4 encoder lacks the frozen chat protocol "
            f"parameters: {sorted(required_parameters)}"
        )
    _OFFICIAL_ENCODER = encode_messages
    _OFFICIAL_ENCODER_IDENTITY = {
        "path": str(encoding_path.resolve()),
        "sha256": "sha256:" + source_sha256,
    }
    return _OFFICIAL_ENCODER


def _encode_rag_prompt(docs: list[str], question: str) -> str:
    """Encode one RAG turn with the checkpoint's official prompt protocol."""

    content = "\n\n".join(docs) + _query_text(question)
    encode_messages = _load_official_encoder()
    kwargs = {
        "thinking_mode": THINKING_MODE,
        "drop_thinking": True,
        "add_default_bos_token": True,
    }
    if "reasoning_effort" in inspect.signature(encode_messages).parameters:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    return encode_messages([{"role": "user", "content": content}], **kwargs)


def _encode_pure_official_rag_prompt(docs: list[str], question: str) -> str:
    """Frozen chat/low protocol used only by pure MLA qualification."""

    content = "\n\n".join(docs) + _query_text(question)
    encode_messages = _load_official_encoder()
    return encode_messages(
        [{"role": "user", "content": content}],
        thinking_mode="chat",
        reasoning_effort="low",
        drop_thinking=True,
        add_default_bos_token=True,
    )


def _chunk_token_ids(ids: list[int], tok, chunk_tokens: int) -> list[str]:
    docs = []
    for start in range(0, len(ids), chunk_tokens):
        piece = ids[start : start + chunk_tokens]
        if len(piece) < 64:
            break
        docs.append(tok.decode(piece, skip_special_tokens=True))
    return docs


def _load_longbench_padded(ds_name: str, tok, n_samples: int, target_tokens: int):
    path = os.path.join(LONGBENCH_DIR, f"{ds_name}.jsonl")
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("input") and row.get("context") and row.get("answers"):
                raw.append(row)
    rng = random.Random(SEED)
    rng.shuffle(raw)

    out = []
    n = len(raw)
    for i, base in enumerate(raw):
        if len(out) >= n_samples:
            break
        ctx_ids = tok(base["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % n
        while len(ctx_ids) < target_tokens and j != i:
            extra = tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            ctx_ids.extend(extra)
            j = (j + 1) % n
        ctx_ids = ctx_ids[:target_tokens]
        docs = _chunk_token_ids(ctx_ids, tok, CHUNK_TOKENS)
        if len(docs) < 2:
            continue
        out.append(
            {
                "question": base["input"],
                "golds": [str(x) for x in base["answers"]],
                "docs": docs,
            }
        )
    return out


@torch.no_grad()
def standard_prefill(model, tok, full_text: str, query_text: str):
    device = getattr(model, "device", None) or next(model.parameters()).device
    ids = tok(full_text + query_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    next_id = out.logits[0, -1, :].argmax().view(1, 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    past = out.past_key_values
    generated = [int(next_id[0, 0])]
    t1 = time.perf_counter()
    for _ in range(MAX_NEW_TOKENS - 1):
        og = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = og.past_key_values
        next_id = og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(next_id[0, 0])
        generated.append(tid)
        if tid == tok.eos_token_id:
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_t = max(time.perf_counter() - t1, 1e-3)
    return (
        tok.decode(generated, skip_special_tokens=True),
        ttft,
        len(generated) / decode_t,
        ids.shape[1],
    )


def _model_dims(cfg):
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    return {
        "L": int(cfg.num_hidden_layers),
        "hidden": int(cfg.hidden_size),
        "Hq": int(cfg.num_attention_heads),
        "Hkv": int(getattr(cfg, "num_key_value_heads", 1)),
        "D": head_dim,
        "q_lora": int(getattr(cfg, "q_lora_rank", 0) or 0),
        "o_lora": int(getattr(cfg, "o_lora_rank", 0) or 0),
        "moe_inter": int(
            getattr(cfg, "moe_intermediate_size", getattr(cfg, "intermediate_size", 0))
        ),
        "experts_per_tok": int(getattr(cfg, "num_experts_per_tok", 1)),
        "shared_experts": int(getattr(cfg, "n_shared_experts", 0)),
    }


def _proj_flops_per_token(d):
    if d["q_lora"] and d["o_lora"]:
        q = 2.0 * d["hidden"] * d["q_lora"] + 2.0 * d["q_lora"] * d["Hq"] * d["D"]
        kv = 2.0 * d["hidden"] * d["D"]
        o = 2.0 * d["Hq"] * d["D"] * d["o_lora"] + 2.0 * d["o_lora"] * d["hidden"]
        return q + kv + o
    return (
        2.0 * d["hidden"] * (d["Hq"] + 2 * d["Hkv"]) * d["D"]
        + 2.0 * d["Hq"] * d["D"] * d["hidden"]
    )


def _ffn_flops_per_token(d):
    if d["moe_inter"]:
        active = max(1, d["experts_per_tok"] + d["shared_experts"])
        return active * 6.0 * d["hidden"] * d["moe_inter"]
    return 0.0


def _attn_dense(d, T):
    return d["L"] * d["Hq"] * 4.0 * d["D"] * (T * (T + 1) / 2.0)


def _attn_hc(d, T, frac_global, window):
    h_global = d["Hq"] * frac_global
    h_local = d["Hq"] - h_global
    full = T * (T + 1) / 2.0
    local = T * min(window, T)
    return d["L"] * 4.0 * d["D"] * (h_global * full + h_local * local)


def compute_flops(d, T, frac_global, ffn_selected, dense_until, window):
    proj = d["L"] * T * _proj_flops_per_token(d)
    ffn_dense = d["L"] * T * _ffn_flops_per_token(d)
    dense_layers = min(dense_until, d["L"])
    ffn_hc = (
        dense_layers * T * _ffn_flops_per_token(d)
        + (d["L"] - dense_layers) * T * _ffn_flops_per_token(d) * ffn_selected
    )
    attn_d = _attn_dense(d, T)
    attn_h = _attn_hc(d, T, frac_global, window)
    return {
        "attn": (attn_d, attn_h),
        "ffn": (ffn_dense, ffn_hc),
        "proj": (proj, proj),
        "total": (proj + ffn_dense + attn_d, proj + ffn_hc + attn_h),
    }


def _looks_redknot_hf_compatible(model) -> tuple[bool, str]:
    base = model.model if hasattr(model, "model") else model
    layers = getattr(base, "layers", None)
    if not layers:
        return False, "model.model.layers is missing"
    attn = getattr(layers[0], "self_attn", None)
    if attn is None:
        return False, "layers[0].self_attn is missing"
    required = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "num_key_value_groups",
        "head_dim",
    ]
    missing = [name for name in required if not hasattr(attn, name)]
    if missing:
        return False, "missing attention attrs: " + ", ".join(missing)
    return True, "compatible"


def _make_default_head_config(cfg):
    from sglang.srt.layers.attention.redknot import (
        DeepSeekV4MLAHeadConfig,
        HeadClassConfig,
        is_deepseek_v4_mla_config,
    )

    if is_deepseek_v4_mla_config(cfg):
        return DeepSeekV4MLAHeadConfig.from_model_config(
            cfg,
            dense_prefix_layers=MLA_DENSE_PREFIX,
            dense_suffix_layers=MLA_DENSE_SUFFIX,
            local_window=MLA_LOCAL_WINDOW,
            global_head_stride=max(1, MLA_GLOBAL_HEAD_STRIDE),
            global_layer_stride=MLA_GLOBAL_LAYER_STRIDE,
        )
    n_layers = int(cfg.num_hidden_layers)
    n_kv = int(getattr(cfg, "num_key_value_heads", 1))
    window = int(os.environ.get("REDKNOT_LOCAL_WINDOW", "4096"))
    dense_prefix = int(os.environ.get("REDKNOT_DENSE_PREFIX_LAYERS", "2"))
    global_stride = max(1, int(os.environ.get("REDKNOT_GLOBAL_LAYER_STRIDE", "8")))
    head_class = []
    head_distance = []
    for layer in range(n_layers):
        if layer < dense_prefix or layer % global_stride == 0:
            row = ["global"] * n_kv
            dist = [-1] * n_kv
        else:
            row = ["local"] * n_kv
            dist = [window] * n_kv
        head_class.append(row)
        head_distance.append(dist)
    return HeadClassConfig(
        head_class=head_class,
        head_max_distance=head_distance,
        num_layers=n_layers,
        num_kv_heads=n_kv,
        dense_prefix_layers=dense_prefix,
        local_default_window=window,
    )


def _load_head_config(cfg):
    path = HEAD_CFG_PATH
    if path:
        from sglang.srt.layers.attention.redknot import (
            DeepSeekV4MLAHeadConfig,
            HeadClassConfig,
            is_deepseek_v4_mla_config,
        )

        if is_deepseek_v4_mla_config(cfg):
            return DeepSeekV4MLAHeadConfig.from_json(
                path,
                dense_prefix_layers=MLA_DENSE_PREFIX,
                dense_suffix_layers=MLA_DENSE_SUFFIX,
            )
        hc = HeadClassConfig.from_json(path)
        hc.merge_retrieval_to_global()
        return hc
    return _make_default_head_config(cfg)


def _make_sparse_ffn_schedule():
    from sglang.srt.layers.attention.redknot import SparseFFNSchedule

    dense_until = int(os.environ.get("REDKNOT_FFN_DENSE_UNTIL", "4"))
    mass = float(os.environ.get("REDKNOT_FFN_MASS", "0.30"))
    deep_start = int(
        os.environ.get(
            "REDKNOT_FFN_DEEP_START",
            str(_pro0813_scale_policy.PRO0813_TOKEN_SPARSE_DEEP_START),
        )
    )
    mass_deep = float(os.environ.get("REDKNOT_FFN_MASS_DEEP", "0.10"))
    recent_n = int(os.environ.get("REDKNOT_FFN_RECENT_N", "256"))
    return SparseFFNSchedule(
        dense_until=dense_until,
        mass_thresh=mass,
        deep_layer_start=deep_start,
        mass_thresh_deep=mass_deep,
        recent_n=recent_n,
    )


@torch.no_grad()
def redknot_prefill(model, tok, docs: list[str], query_text: str):
    from sglang.srt.layers.attention.redknot import (
        offline_prefill_segments,
        run_redknot_offlinekv,
    )

    hc = _load_head_config(model.config)
    if hc.__class__.__name__ == "DeepSeekV4MLAHeadConfig":
        raise RuntimeError(
            "DeepSeek V4 MLA needs a FlashMLA-native RedKnot prefill path; "
            "the generic HF offline-KV driver expects materialized k_proj/v_proj KV."
        )
    sched = _make_sparse_ffn_schedule()
    segs = offline_prefill_segments(
        model,
        tok,
        docs,
        chunk_size=max(4096, CHUNK_TOKENS + 96),
        model_id=MODEL_PATH,
    )
    stats = []
    t0 = time.perf_counter()
    _, text, _, ttft = run_redknot_offlinekv(
        model,
        tok,
        segments_offline=segs,
        query_text=query_text,
        head_cfg=hc,
        max_new_tokens=MAX_NEW_TOKENS,
        kernel=REUSE_KERNEL,
        sparse_ffn_schedule=sched,
        sparse_ffn_stats=stats,
        use_compile=USE_COMPILE,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = max(time.perf_counter() - t0, 1e-3)
    n_dec = len(tok(text, add_special_tokens=False)["input_ids"]) or 1
    return text, ttft, n_dec / max(total - ttft, 1e-3), stats


def _load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM

    tok = _load_tokenizer()

    kwargs = {
        "device_map": DEVICE_MAP,
        "trust_remote_code": True,
    }
    if DTYPE == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif DTYPE == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif DTYPE == "fp32":
        kwargs["torch_dtype"] = torch.float32
    print(f"Loading {MODEL_PATH} (dtype={DTYPE}, device_map={DEVICE_MAP})...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs).eval()
    return model, tok


def _engine_kwargs(attention_backend: str, *, sparse_ffn: bool):
    kwargs = {
        "model_path": MODEL_PATH,
        "attention_backend": attention_backend,
        "tp_size": TP_SIZE,
        "random_seed": SEED,
        "log_level": os.environ.get("REDKNOT_LOG_LEVEL", "info"),
        "watchdog_timeout": int(os.environ.get("REDKNOT_WATCHDOG_TIMEOUT", "1800")),
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        # Required for SchedulerReqTimeStats.forward_entry_time and
        # prefill_finished_time to be returned with each result.  This gives
        # a model-serving-internal first-token interval that excludes client,
        # tokenization and response-delivery time.
        "enable_metrics": True,
        "chunked_prefill_size": 16384,
        "max_prefill_tokens": 16384,
    }
    if MAX_TOTAL_TOKENS > 0:
        kwargs["max_total_tokens"] = MAX_TOTAL_TOKENS
    if ENGINE_SWA_FULL_TOKENS_RATIO > 0:
        # DSV4's automatic 0.1 ratio can be smaller than one long-prefill
        # admission budget (chunk + allocator page), leaving a valid request
        # permanently queued.  Keep the legacy default when unset, but allow
        # correctness/quality gates to provision the SWA pool explicitly.
        kwargs["swa_full_tokens_ratio"] = ENGINE_SWA_FULL_TOKENS_RATIO
    if DISABLE_CUDA_GRAPH:
        kwargs["disable_cuda_graph"] = True
    if SKIP_SERVER_WARMUP:
        kwargs["skip_server_warmup"] = True
    if MOE_RUNNER_BACKEND:
        kwargs["moe_runner_backend"] = MOE_RUNNER_BACKEND
    if MEM_FRACTION_STATIC:
        kwargs["mem_fraction_static"] = float(MEM_FRACTION_STATIC)
    if attention_backend == "redknot_mla":
        kwargs.update(
            {
                "redknot_sparse_ffn_enable": sparse_ffn,
                "redknot_sparse_ffn_dense_until": int(
                    os.environ.get("REDKNOT_FFN_DENSE_UNTIL", "4")
                ),
                "redknot_sparse_ffn_mass_thresh": float(
                    os.environ.get("REDKNOT_FFN_MASS", "0.30")
                ),
                "redknot_sparse_ffn_deep_start": int(
                    os.environ.get(
                        "REDKNOT_FFN_DEEP_START",
                        str(
                            _pro0813_scale_policy.PRO0813_TOKEN_SPARSE_DEEP_START
                        ),
                    )
                ),
                "redknot_sparse_ffn_mass_thresh_deep": float(
                    os.environ.get("REDKNOT_FFN_MASS_DEEP", "0.10")
                ),
                "redknot_sparse_ffn_recent_n": int(
                    os.environ.get("REDKNOT_FFN_RECENT_N", "256")
                ),
                "redknot_mla_dense_prefix_layers": MLA_DENSE_PREFIX,
                "redknot_mla_dense_suffix_layers": MLA_DENSE_SUFFIX,
                "redknot_mla_local_window": MLA_LOCAL_WINDOW,
                "redknot_mla_global_head_stride": MLA_GLOBAL_HEAD_STRIDE,
                "redknot_mla_global_layer_stride": MLA_GLOBAL_LAYER_STRIDE,
                "redknot_mla_pass_mode": MLA_PASS_MODE,
            }
        )
        if HEAD_CFG_PATH:
            kwargs["redknot_head_config_path"] = HEAD_CFG_PATH
    return kwargs


def _profile_engine_kwargs(context_tokens: int):
    """Engine kwargs for the offline MLA head-locality analysis run."""
    kwargs = {
        "model_path": MODEL_PATH,
        # Profiling hooks the dsv4 non-fused attention path.
        "attention_backend": "dsv4",
        "tp_size": TP_SIZE,
        "random_seed": SEED,
        "redknot_mla_profile_enable": True,
        "redknot_mla_profile_out": PROFILE_OUT,
        "redknot_mla_profile_coverage": PROFILE_COVERAGE,
        "redknot_mla_profile_sample_queries": PROFILE_SAMPLE_Q,
        "redknot_mla_profile_global_window_ratio": PROFILE_GLOBAL_RATIO,
        "redknot_mla_profile_window_safety": PROFILE_WINDOW_SAFETY,
        "redknot_mla_dense_prefix_layers": MLA_DENSE_PREFIX,
        "redknot_mla_dense_suffix_layers": MLA_DENSE_SUFFIX,
        # Keep enough hybrid-SWA capacity for admission while the logical
        # request is processed as several native DSV4 prefill chunks.
        "swa_full_tokens_ratio": PROFILE_SWA_FULL_TOKENS_RATIO,
        # Make a lost/stalled request visible in the rank logs instead of
        # leaving eight TP workers silently polling forever.  Keep the native
        # DSV4 scheduler settings here: some hybrid-SWA initialization paths
        # size internal state from those defaults even for a single request.
        "log_level": os.environ.get("REDKNOT_PROFILE_LOG_LEVEL", "info"),
        "log_requests": True,
        "show_time_cost": True,
        "watchdog_timeout": float(
            os.environ.get("REDKNOT_PROFILE_WATCHDOG_TIMEOUT", "300")
        ),
        # The profiler accumulates latent K across request chunks, retaining Q
        # only at request-global sampled rows, and scores after the exact
        # expected context is complete. Cap the aggregate prefill budget to the
        # same value so FlashMLA never receives a mixed/oversized metadata
        # batch even if another request appears while profiling.
        "chunked_prefill_size": PROFILE_CHUNKED_PREFILL_SIZE,
        "max_prefill_tokens": PROFILE_CHUNKED_PREFILL_SIZE,
        # The profiler reconstructs the complete latent stream itself. Prefix
        # cache reuse would make the first observed prefix non-zero and omit K
        # rows that are required for an exact full-context score.
        "disable_radix_cache": True,
    }
    if MAX_TOTAL_TOKENS > 0:
        kwargs["max_total_tokens"] = MAX_TOTAL_TOKENS
    if DISABLE_CUDA_GRAPH:
        kwargs["disable_cuda_graph"] = True
    if SKIP_SERVER_WARMUP:
        kwargs["skip_server_warmup"] = True
    if MOE_RUNNER_BACKEND:
        kwargs["moe_runner_backend"] = MOE_RUNNER_BACKEND
    if MEM_FRACTION_STATIC:
        kwargs["mem_fraction_static"] = float(MEM_FRACTION_STATIC)
    return kwargs


def _profile_rank_path(rank: int, *, distance: bool = False) -> Path:
    path = Path(PROFILE_OUT)
    stem = path.stem + ("_distance" if distance else "")
    return path.with_name(f"{stem}_rank{rank}{path.suffix or '.json'}")


def _merge_tp_profile_outputs() -> None:
    """Merge contiguous TP query-head shards into one 128-head profile."""
    if TP_SIZE <= 1:
        return

    from sglang.srt.layers.attention.redknot import DeepSeekV4MLAHeadConfig

    rank_paths = [_profile_rank_path(rank) for rank in range(TP_SIZE)]
    missing = [str(path) for path in rank_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "MLA profiler did not export every TP head shard; missing="
            + ", ".join(missing)
        )
    shards = [DeepSeekV4MLAHeadConfig.from_json(str(path)) for path in rank_paths]
    first = shards[0]
    for rank, shard in enumerate(shards[1:], start=1):
        if shard.num_layers != first.num_layers:
            raise RuntimeError(
                f"rank {rank} profile layers={shard.num_layers}, "
                f"expected {first.num_layers}"
            )
        if shard.dense_prefix_layers != first.dense_prefix_layers:
            raise RuntimeError("TP profile shards disagree on dense-prefix layers")
        if shard.dense_suffix_layers != first.dense_suffix_layers:
            raise RuntimeError("TP profile shards disagree on dense-suffix layers")
        if shard.num_attention_heads != first.num_attention_heads:
            raise RuntimeError(
                f"rank {rank} local heads={shard.num_attention_heads}, "
                f"expected {first.num_attention_heads}"
            )

    head_class = []
    head_distance = []
    head_sinks = []
    for layer in range(first.num_layers):
        head_class.append(
            [value for shard in shards for value in shard.head_class[layer]]
        )
        head_distance.append(
            [value for shard in shards for value in shard.head_max_distance[layer]]
        )
        head_sinks.append(
            [
                value
                for shard in shards
                for value in (
                    shard.head_sink_size[layer]
                    if shard.head_sink_size is not None
                    else [shard.default_sink_size] * shard.num_attention_heads
                )
            ]
        )

    merged = DeepSeekV4MLAHeadConfig(
        head_class=head_class,
        head_max_distance=head_distance,
        head_sink_size=head_sinks,
        num_layers=first.num_layers,
        num_attention_heads=sum(x.num_attention_heads for x in shards),
        physical_kv_heads=first.physical_kv_heads,
        default_sink_size=first.default_sink_size,
        local_default_window=first.local_default_window,
        dense_prefix_layers=first.dense_prefix_layers,
        dense_suffix_layers=first.dense_suffix_layers,
    )
    if PROFILE_EXPECTED_HEADS > 0 and (
        merged.num_attention_heads != PROFILE_EXPECTED_HEADS
    ):
        raise RuntimeError(
            f"merged profile has {merged.num_attention_heads} logical heads, "
            f"expected {PROFILE_EXPECTED_HEADS}"
        )
    merged.to_json(PROFILE_OUT)

    distance_paths = [
        _profile_rank_path(rank, distance=True) for rank in range(TP_SIZE)
    ]
    missing_distance = [str(path) for path in distance_paths if not path.exists()]
    if missing_distance:
        raise RuntimeError(
            "MLA profiler did not export every TP distance shard; missing="
            + ", ".join(missing_distance)
        )
    with open(distance_paths[0], "r", encoding="utf-8") as f:
        distance_shards = [json.load(f)]
    for path in distance_paths[1:]:
        with open(path, "r", encoding="utf-8") as f:
            distance_shards.append(json.load(f))

    reference_distance = distance_shards[0]
    comparable_fields = (
        "num_layers",
        "num_attention_heads",
        "dense_prefix_layers",
        "dense_suffix_layers",
        "max_context",
        "sample_queries",
        "bin_edges",
        "query_window_quantile_for_policy",
        "sampled_query_positions",
    )
    for rank, shard in enumerate(distance_shards):
        for field in comparable_fields:
            if shard.get(field) != reference_distance.get(field):
                raise RuntimeError(
                    f"rank {rank} distance profile disagrees on {field}"
                )
        query_rows = shard.get("query_rows_per_layer", [])
        if len(query_rows) != first.num_layers or any(
            float(value) <= 0 for value in query_rows
        ):
            raise RuntimeError(
                f"rank {rank} has missing per-layer query observations: "
                f"{query_rows}"
            )

    distance = dict(distance_shards[0])
    distance["num_attention_heads"] = sum(
        int(x["num_attention_heads"]) for x in distance_shards
    )
    distance["tp_size"] = TP_SIZE
    distance["head_order"] = "contiguous_tp_rank_then_local_head"
    for name in distance["coverage_windows"]:
        distance["coverage_windows"][name] = [
            [
                value
                for shard in distance_shards
                for value in shard["coverage_windows"][name][layer]
            ]
            for layer in range(first.num_layers)
        ]
    for name in distance["per_query_coverage_window_quantiles"]:
        distance["per_query_coverage_window_quantiles"][name] = [
            [
                value
                for shard in distance_shards
                for value in shard["per_query_coverage_window_quantiles"][name][
                    layer
                ]
            ]
            for layer in range(first.num_layers)
        ]
    distance["normalized_mass_by_distance_bin"] = [
        [
            value
            for shard in distance_shards
            for value in shard["normalized_mass_by_distance_bin"][layer]
        ]
        for layer in range(first.num_layers)
    ]
    distance["query_rows_per_layer_by_rank"] = [
        shard["query_rows_per_layer"] for shard in distance_shards
    ]
    distance_out = Path(PROFILE_OUT).with_name(
        Path(PROFILE_OUT).stem + "_distance" + (Path(PROFILE_OUT).suffix or ".json")
    )
    with open(distance_out, "w", encoding="utf-8") as f:
        json.dump(distance, f, indent=2)
        f.write("\n")

    print(
        f"[profile] merged {TP_SIZE} TP shards -> {merged.num_attention_heads} "
        f"logical heads; distance evidence: {distance_out}"
    )


def _run_profile():
    """Run the analysis-before-compression step and export a head config JSON.

    Prefills one long single-sequence context with the profiler enabled, then
    reads back the exported ``DeepSeekV4MLAHeadConfig`` JSON and prints a
    per-layer summary of global vs local heads and the local window sizes.
    """
    # Make profiler failures visible immediately.  The runtime hook remains
    # fail-open for normal serving unless this explicit offline-analysis flag
    # is present.
    os.environ.setdefault("REDKNOT_MLA_PROFILE_STRICT", "1")
    # Stock sgl-kernel does not carry DSV4's private DeepGEMM prenorm symbol.
    # The built-in fallback performs the real MHC computation and is the
    # repository-supported compatibility path.
    os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0")
    # The default eagerly compiles every GEMM M=1..16384 on the first request.
    # Profiling needs only its real chunk shapes, which DeepGEMM JITs on demand.
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

    import sglang as sgl

    tok = _load_tokenizer()
    # Build the longest available single context to make distance stats robust.
    length_label = LENGTHS[-1] if LENGTHS else "8K"
    target = _LEN_MAP[length_label]
    sample = None
    for ds_name in DATASETS:
        cand = _load_longbench_padded(ds_name, tok, 1, target)
        if cand:
            sample = cand[0]
            break
    if sample is None:
        print("[profile] no usable sample found; aborting")
        return

    prompt = _encode_rag_prompt(sample["docs"], sample["question"])
    n_ctx = len(tok(prompt, add_special_tokens=False)["input_ids"])
    # Model workers inherit this before Engine startup.  It is the validity
    # boundary that prevents engine warmups or incomplete requests from being
    # exported as real profiles.
    os.environ["REDKNOT_MLA_PROFILE_EXPECTED_CONTEXT"] = str(n_ctx)

    W = 108
    print("=" * W)
    print(" REDKNOT MLA HEAD-LOCALITY PROFILE (analysis before compression)")
    print(f" Model: {MODEL_PATH}")
    print(
        f" ctx≈{n_ctx:,} tok | coverage={PROFILE_COVERAGE} "
        f"global_ratio={PROFILE_GLOBAL_RATIO} window_safety={PROFILE_WINDOW_SAFETY}"
    )
    print(f" out: {PROFILE_OUT}")
    print("=" * W)

    os.makedirs(os.path.dirname(PROFILE_OUT) or ".", exist_ok=True)
    rank_log_dir = os.environ.get("SGLANG_RANK_LOG_DIR", "")
    if rank_log_dir:
        os.makedirs(rank_log_dir, exist_ok=True)
    # Do not allow stale shard files from an earlier failed run to masquerade
    # as a complete new TP profile.
    if TP_SIZE > 1:
        for rank in range(TP_SIZE):
            _profile_rank_path(rank).unlink(missing_ok=True)
            _profile_rank_path(rank, distance=True).unlink(missing_ok=True)
        Path(PROFILE_OUT).unlink(missing_ok=True)
    engine = sgl.Engine(**_profile_engine_kwargs(n_ctx))
    try:
        # No decode token is needed: the final prefill chunk seals and exports
        # the profile from the last transformer layer.
        print(
            f"[profile] engine ready; submitting {n_ctx:,}-token prefill",
            flush=True,
        )
        request_t0 = time.perf_counter()
        engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": 0})
        print(
            f"[profile] prefill returned in {time.perf_counter() - request_t0:.3f}s",
            flush=True,
        )
    finally:
        engine.shutdown()

    _merge_tp_profile_outputs()

    if not os.path.exists(PROFILE_OUT):
        print(f"[profile] expected head config not found at {PROFILE_OUT}")
        return

    from sglang.srt.layers.attention.redknot import DeepSeekV4MLAHeadConfig

    hc = DeepSeekV4MLAHeadConfig.from_json(PROFILE_OUT)
    summary = hc.summary()
    print("\n head classification summary:")
    print(f"   {summary}")
    print("\n per-layer (global / local / dense, local window min..max):")
    from sglang.srt.layers.attention.redknot.head_config import (
        HEAD_GLOBAL,
        HEAD_LOCAL,
    )

    for layer in range(hc.num_layers):
        global_count = local_count = dense_count = 0
        wins = []
        for head in range(hc.num_attention_heads):
            t = hc.head_class[layer][head]
            if t == HEAD_GLOBAL:
                global_count += 1
            elif t == HEAD_LOCAL:
                local_count += 1
                wins.append(hc.head_max_distance[layer][head])
            else:
                dense_count += 1
        wtxt = f"{min(wins)}..{max(wins)}" if wins else "-"
        print(
            f"   layer {layer:3d}: global={global_count:3d} "
            f"local={local_count:3d} dense={dense_count:3d} win={wtxt}"
        )
    print("=" * W)
    print(f" head config written to: {PROFILE_OUT}")
    print(" Use it via REDKNOT_HEAD_CFG=<path> with attention_backend=redknot_mla")
    print("=" * W)


_DRIFT_MANIFEST_FORMAT = (
    "redknot_dsv4_pro0813_drift_calibration_manifest_v3"
)
_DRIFT_PAYLOAD_SCHEMA = "redknot_dsv4_pro0813_drift_calibration_payload_v2"
_DRIFT_RANK_CAPTURE_FORMAT = (
    "redknot_deepseek_v4_pro0813_mla_head_drift_rank_capture_v2"
)
_DRIFT_REPORT_FORMAT = "redknot_deepseek_v4_pro0813_mla_head_drift_v2"
_DRIFT_HEAD_CONFIG_FORMAT = "redknot_deepseek_v4_pro0813_mla_head_config_v3"
_DRIFT_DIGEST_SCOPE = {
    "algorithm": "sha256",
    "payload_field": "canonical_payload",
    "canonicalization": "utf8_json_sort_keys_compact_no_nan_v1",
}


def _validate_pro0813_drift_module_contract() -> None:
    """Bind this producer to the current Pro runtime/analyzer/profiler APIs."""

    from sglang.srt.layers.attention.redknot.mla_head_drift_analyze import (
        CALIBRATION_DIGEST_SCOPE,
        CALIBRATION_MANIFEST_FORMAT,
        CALIBRATION_PAYLOAD_SCHEMA,
        RANK_CAPTURE_FORMAT as ANALYZER_RANK_CAPTURE_FORMAT,
    )
    from sglang.srt.layers.attention.redknot.mla_head_drift_profiler import (
        DRIFT_REPORT_FORMAT,
        HEAD_CONFIG_FORMAT,
    )
    from sglang.srt.layers.attention.redknot.mla_head_drift_runtime import (
        RANK_CAPTURE_FORMAT as RUNTIME_RANK_CAPTURE_FORMAT,
    )
    from sglang.srt.layers.attention.redknot.pro0813.profile import (
        PRO0813_OFFICIAL_CONFIG_SHA256 as PROFILE_CONFIG_SHA256,
        PRO0813_TP8_GEOMETRY,
        PRO0813_TP8_GEOMETRY_DIGEST as PROFILE_GEOMETRY_DIGEST,
        PRO0813_VARIANT as PROFILE_VARIANT,
    )

    observed_formats = {
        "manifest": CALIBRATION_MANIFEST_FORMAT,
        "payload": CALIBRATION_PAYLOAD_SCHEMA,
        "runtime_capture": RUNTIME_RANK_CAPTURE_FORMAT,
        "analyzer_capture": ANALYZER_RANK_CAPTURE_FORMAT,
        "report": DRIFT_REPORT_FORMAT,
        "head_config": HEAD_CONFIG_FORMAT,
    }
    expected_formats = {
        "manifest": _DRIFT_MANIFEST_FORMAT,
        "payload": _DRIFT_PAYLOAD_SCHEMA,
        "runtime_capture": _DRIFT_RANK_CAPTURE_FORMAT,
        "analyzer_capture": _DRIFT_RANK_CAPTURE_FORMAT,
        "report": _DRIFT_REPORT_FORMAT,
        "head_config": _DRIFT_HEAD_CONFIG_FORMAT,
    }
    if observed_formats != expected_formats:
        raise RuntimeError(
            "Pro-0813 drift producer/runtime/analyzer schema mismatch: "
            f"observed={observed_formats!r} expected={expected_formats!r}"
        )
    if CALIBRATION_DIGEST_SCOPE != _DRIFT_DIGEST_SCOPE:
        raise RuntimeError(
            "Pro-0813 calibration digest scope differs between producer and analyzer"
        )
    identity = (
        PROFILE_VARIANT,
        PROFILE_GEOMETRY_DIGEST,
        PROFILE_CONFIG_SHA256,
    )
    expected_identity = (
        PRO0813_VARIANT,
        PRO0813_TP8_GEOMETRY_DIGEST,
        PRO0813_OFFICIAL_CONFIG_SHA256,
    )
    if identity != expected_identity:
        raise RuntimeError(
            "Pro-0813 calibration model identity differs from the certified profile"
        )
    exact_geometry = {
        "num_layers": PRO0813_TP8_GEOMETRY.num_target_layers,
        "num_attention_heads": PRO0813_TP8_GEOMETRY.num_attention_heads,
        "tp_size": PRO0813_TP8_GEOMETRY.tp_size,
        "heads_per_rank": PRO0813_TP8_GEOMETRY.heads_per_tp_rank,
        "num_output_groups": PRO0813_TP8_GEOMETRY.o_groups,
        "groups_per_rank": PRO0813_TP8_GEOMETRY.o_groups_per_tp_rank,
        "heads_per_output_group": (
            PRO0813_TP8_GEOMETRY.heads_per_tp_rank
            // PRO0813_TP8_GEOMETRY.o_groups_per_tp_rank
        ),
        "dense_prefix_layers": PRO0813_TP8_GEOMETRY.dense_prefix_layers,
        "dense_suffix_layers": PRO0813_TP8_GEOMETRY.dense_suffix_layers,
        "dense_layer_ids": tuple(PRO0813_TP8_GEOMETRY.dense_layer_ids),
        "reusable_layer_ids": tuple(PRO0813_TP8_GEOMETRY.reusable_layer_ids),
    }
    expected_geometry = {
        "num_layers": PRO0813_NUM_LAYERS,
        "num_attention_heads": PRO0813_NUM_HEADS,
        "tp_size": PRO0813_TP_SIZE,
        "heads_per_rank": PRO0813_HEADS_PER_TP_RANK,
        "num_output_groups": PRO0813_GLOBAL_OUTPUT_GROUPS,
        "groups_per_rank": PRO0813_O_GROUPS_PER_TP_RANK,
        "heads_per_output_group": PRO0813_HEADS_PER_OUTPUT_GROUP,
        "dense_prefix_layers": PRO0813_DENSE_PREFIX_LAYERS,
        "dense_suffix_layers": PRO0813_DENSE_SUFFIX_LAYERS,
        "dense_layer_ids": PRO0813_DENSE_LAYER_IDS,
        "reusable_layer_ids": PRO0813_REUSABLE_LAYER_IDS,
    }
    if exact_geometry != expected_geometry:
        raise RuntimeError(
            "Pro-0813 drift topology differs from the certified 3+55+3 TP8 layout"
        )


def _validate_pro0813_drift_plan_identity(plan: Mapping[str, Any]) -> None:
    expected = {
        "variant": PRO0813_VARIANT,
        "geometry_digest": PRO0813_TP8_GEOMETRY_DIGEST,
        "official_config_sha256": PRO0813_OFFICIAL_CONFIG_SHA256,
    }
    for name, value in expected.items():
        if plan.get(name) != value:
            raise ValueError(
                f"Pro-0813 drift plan {name}={plan.get(name)!r}, expected {value!r}"
            )


def _build_pro0813_drift_plan(
    *,
    role: str,
    run_id: str,
    calibration_digest: str,
    out_path: str,
    segments: Sequence[Mapping[str, Any]],
    sample_rows: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    if role not in ("snapshot", "oracle"):
        raise ValueError("Pro-0813 drift role must be snapshot or oracle")
    plan = {
        "mode": "drift_profile",
        "variant": PRO0813_VARIANT,
        "geometry_digest": PRO0813_TP8_GEOMETRY_DIGEST,
        "official_config_sha256": PRO0813_OFFICIAL_CONFIG_SHA256,
        "run_id": run_id,
        "calibration_digest": calibration_digest,
        "role": role,
        "out_path": out_path,
        "segments": [dict(item) for item in segments],
        "sample_rows": {
            str(segment_id): list(rows)
            for segment_id, rows in sample_rows.items()
        },
    }
    _validate_pro0813_drift_plan_identity(plan)
    return plan


def _build_pro0813_drift_payload(
    *,
    run_id: str,
    config_sha256: str,
    num_layers: int,
    num_attention_heads: int,
    o_lora_rank: int,
    segments: Sequence[Mapping[str, Any]],
    oracles: Sequence[Mapping[str, Any]],
    estimated_capture_tensor_bytes: int,
    max_capture_tensor_bytes: int,
) -> dict[str, Any]:
    if config_sha256 != PRO0813_OFFICIAL_CONFIG_SHA256:
        raise ValueError("drift calibration config is not official Pro-0813")
    if (
        num_layers != PRO0813_NUM_LAYERS
        or num_attention_heads != PRO0813_NUM_HEADS
        or o_lora_rank != PRO0813_O_LORA_RANK
    ):
        raise ValueError("drift calibration model geometry is not Pro-0813")
    if estimated_capture_tensor_bytes <= 0:
        raise ValueError("drift capture estimate must be positive")
    if max_capture_tensor_bytes < estimated_capture_tensor_bytes:
        raise ValueError("drift capture estimate exceeds its explicit budget")
    payload = {
        "schema": _DRIFT_PAYLOAD_SCHEMA,
        "run_id": run_id,
        "model": {
            "variant": PRO0813_VARIANT,
            "geometry_digest": PRO0813_TP8_GEOMETRY_DIGEST,
            "config_sha256": config_sha256,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "o_lora_rank": o_lora_rank,
        },
        "topology": {
            "tp_size": PRO0813_TP_SIZE,
            "heads_per_rank": PRO0813_HEADS_PER_TP_RANK,
            "num_output_groups": PRO0813_GLOBAL_OUTPUT_GROUPS,
            "groups_per_rank": PRO0813_O_GROUPS_PER_TP_RANK,
            "heads_per_output_group": PRO0813_HEADS_PER_OUTPUT_GROUP,
            "dense_prefix_layers": PRO0813_DENSE_PREFIX_LAYERS,
            "dense_suffix_layers": PRO0813_DENSE_SUFFIX_LAYERS,
        },
        "segments": [dict(item) for item in segments],
        "oracles": [dict(item) for item in oracles],
        "resource_guard": {
            "estimated_capture_tensor_bytes": estimated_capture_tensor_bytes,
            "max_capture_tensor_bytes": max_capture_tensor_bytes,
        },
    }
    return payload


def _drift_token_ids_sha256(input_ids) -> str:
    """Use the runtime capture's architecture-independent token hash."""

    values = []
    for index, token_id in enumerate(input_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"drift input_ids[{index}] must be an integer")
        if token_id < 0:
            raise ValueError(f"drift input_ids[{index}] must be non-negative")
        values.append(token_id)
    digest = hashlib.sha256()
    digest.update(f"{len(values)}:".encode("ascii"))
    digest.update(",".join(str(value) for value in values).encode("ascii"))
    return digest.hexdigest()


def _drift_canonical_manifest_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _drift_capture_tensor_bytes(
    *, num_segments: int, sampled_rows: int, o_lora_rank: int
) -> int:
    # One standalone capture per segment plus N full-segment cyclic oracles.
    if num_segments <= 0 or sampled_rows <= 0:
        raise ValueError("drift segment and sampled-row counts must be positive")
    if o_lora_rank != PRO0813_O_LORA_RANK:
        raise ValueError("drift projection rank must be Pro-0813 o_lora_rank=1024")
    sampled_row_instances = num_segments * sampled_rows * (num_segments + 1)
    return (
        PRO0813_TP_SIZE
        * PRO0813_NUM_LAYERS
        * sampled_row_instances
        * PRO0813_HEADS_PER_TP_RANK
        * o_lora_rank
        * 4  # runtime artifacts are materialized as float32
    )


def _drift_sample_rows(segment_length: int) -> tuple[int, ...]:
    """Choose deterministic interior rows without sampling the boundary replay."""

    length = int(segment_length)
    boundary = int(DRIFT_BOUNDARY_ROWS)
    count = int(DRIFT_SAMPLE_ROWS)
    if length <= boundary:
        raise ValueError(
            f"drift segment length={length} must exceed boundary={boundary}"
        )
    if count <= 0:
        raise ValueError("REDKNOT_MLA_DRIFT_SAMPLE_ROWS must be positive")
    available = length - boundary
    count = min(count, available)
    if count == 1:
        return (length - 1,)
    rows = tuple(
        boundary + (index * (available - 1)) // (count - 1)
        for index in range(count)
    )
    if len(set(rows)) != count:
        raise RuntimeError("drift row sampler produced duplicate rows")
    return rows


def _drift_engine_kwargs(total_tokens: int) -> dict:
    """Build the one-forward, all-online engine required by drift capture."""

    total = int(total_tokens)
    if TP_SIZE != 8:
        raise ValueError("projected-output drift calibration requires TP_SIZE=8")
    if total <= 0:
        raise ValueError("drift calibration total token count must be positive")
    kwargs = {
        "model_path": MODEL_PATH,
        "attention_backend": "dsv4",
        "tp_size": TP_SIZE,
        "random_seed": SEED,
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "disable_cuda_graph": True,
        # The runtime hook deliberately rejects scheduler chunking.  The 4x1K
        # default quick screen therefore executes as one real 4K prefill.
        "chunked_prefill_size": total,
        "max_prefill_tokens": total,
        "redknot_sparse_ffn_enable": False,
        "swa_full_tokens_ratio": PROFILE_SWA_FULL_TOKENS_RATIO,
        "log_level": os.environ.get("REDKNOT_DRIFT_LOG_LEVEL", "info"),
        "watchdog_timeout": int(
            os.environ.get("REDKNOT_DRIFT_WATCHDOG_TIMEOUT", "900")
        ),
    }
    if MAX_TOTAL_TOKENS > 0:
        if MAX_TOTAL_TOKENS < total:
            raise ValueError(
                f"REDKNOT_MAX_TOTAL_TOKENS={MAX_TOTAL_TOKENS} is below "
                f"drift context={total}"
            )
        kwargs["max_total_tokens"] = MAX_TOTAL_TOKENS
    if SKIP_SERVER_WARMUP:
        kwargs["skip_server_warmup"] = True
    if MOE_RUNNER_BACKEND:
        kwargs["moe_runner_backend"] = MOE_RUNNER_BACKEND
    if MEM_FRACTION_STATIC:
        kwargs["mem_fraction_static"] = float(MEM_FRACTION_STATIC)
    return kwargs


def _verify_drift_rank_artifacts(path_template: str) -> list[str]:
    paths = [path_template.replace("{tp_rank}", str(rank)) for rank in range(8)]
    deadline = time.monotonic() + DRIFT_ARTIFACT_TIMEOUT_SECONDS
    while True:
        # Each worker publishes with os.replace, so existence means the whole
        # rank artifact is visible.  The HTTP response follows TP0 completion;
        # other ranks can finish their independent fsync/rename milliseconds
        # later and therefore need a bounded publication grace period.
        missing = [path for path in paths if not Path(path).is_file()]
        if not missing:
            return paths
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "drift request returned without every TP rank artifact after "
                f"{DRIFT_ARTIFACT_TIMEOUT_SECONDS:.3f}s: " + ", ".join(missing)
            )
        time.sleep(min(0.05, remaining))


def _drift_http_request(method: str, path: str, **kwargs):
    """Call an explicitly selected loopback drift server without shell proxies."""

    import requests

    if not 1 <= DRIFT_HTTP_PORT <= 65535:
        raise ValueError("REDKNOT_MLA_DRIFT_HTTP_PORT must be in [1, 65535]")
    with requests.Session() as session:
        session.trust_env = False
        return session.request(
            method,
            f"http://127.0.0.1:{DRIFT_HTTP_PORT}{path}",
            **kwargs,
        )


def _drift_http_generate(input_ids, sampling_params: dict, plan: dict) -> None:
    response = _drift_http_request(
        "POST",
        "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "redknot_reuse_plan": plan,
        },
        timeout=DRIFT_HTTP_TIMEOUT,
        headers={"Connection": "close"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise RuntimeError("drift HTTP generate response must be a JSON object")
    if payload.get("error") is not None:
        raise RuntimeError(f"drift HTTP generate failed: {payload['error']!r}")
    output_ids = payload.get("output_ids")
    if not isinstance(output_ids, list) or len(output_ids) != 1:
        raise RuntimeError(
            "drift HTTP generate must return exactly one real output token"
        )


def _drift_analyzer_command(
    analyzer: Path, output_dir: Path, manifest_path: Path
) -> list[str]:
    """Build the manifest-bound analyzer CLI without launching a subprocess."""

    return [
        sys.executable,
        str(analyzer),
        "--snapshot-glob",
        str(output_dir / "snapshot-*-rank*.pt"),
        "--oracle-glob",
        str(output_dir / "oracle-*-rank*.pt"),
        "--manifest",
        str(manifest_path),
        "--out-dir",
        str(output_dir / "analysis"),
    ]


def _run_projected_head_drift_profile() -> None:
    """Capture standalone/composed head projections and emit nested policies."""

    _validate_pro0813_drift_module_contract()
    if (
        not math.isfinite(DRIFT_READY_DELAY_SECONDS)
        or DRIFT_READY_DELAY_SECONDS < 0
    ):
        raise ValueError(
            "REDKNOT_MLA_DRIFT_READY_DELAY_SECONDS must be finite and non-negative"
        )
    if DRIFT_HTTP_TIMEOUT <= 0:
        raise ValueError("REDKNOT_MLA_DRIFT_HTTP_TIMEOUT must be positive")
    if (
        not math.isfinite(DRIFT_ARTIFACT_TIMEOUT_SECONDS)
        or DRIFT_ARTIFACT_TIMEOUT_SECONDS <= 0
    ):
        raise ValueError(
            "REDKNOT_MLA_DRIFT_ARTIFACT_TIMEOUT_SECONDS must be finite and positive"
        )
    if DRIFT_HTTP_PORT < 0 or DRIFT_HTTP_PORT > 65535:
        raise ValueError("REDKNOT_MLA_DRIFT_HTTP_PORT must be 0 or in [1, 65535]")
    if not 4 <= DRIFT_NUM_CHUNKS <= 16:
        raise ValueError(
            "drift calibration requires 4..16 chunks; the default is four "
            "cyclic rotations"
        )
    if DRIFT_CHUNK_TOKENS <= DRIFT_BOUNDARY_ROWS:
        raise ValueError("drift chunk size must exceed the excluded boundary rows")

    # These are read during SGLang/model import, so set them before importing
    # Engine.  The runtime hook independently revalidates the resolved backend,
    # sparse-FFN setting, and wo_a path inside every worker.
    os.environ["SGLANG_OPT_FP8_WO_A_GEMM"] = "0"
    os.environ.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
    os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0")
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

    tok = _ih_load_tokenizer()
    chunks, _ = _ih_load(
        tok,
        DRIFT_CHUNK_TOKENS,
        DRIFT_NUM_CHUNKS,
        num_queries=1,
    )
    if any(len(chunk) != DRIFT_CHUNK_TOKENS for chunk in chunks):
        raise RuntimeError("drift calibration chunks are not equal length")
    sample_rows = _drift_sample_rows(DRIFT_CHUNK_TOKENS)
    token_digests = [_drift_token_ids_sha256(chunk) for chunk in chunks]
    segment_ids = [
        f"seg{index:02d}-{token_digest[:16]}"
        for index, token_digest in enumerate(token_digests)
    ]
    run_id = f"dsv4pro0813drift-{int(time.time())}-{secrets.token_hex(4)}"
    config_path = Path(MODEL_PATH) / "config.json"
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != PRO0813_OFFICIAL_CONFIG_SHA256:
        raise ValueError(
            "drift calibration requires the byte-exact official Pro-0813 config: "
            f"observed={config_sha256} expected={PRO0813_OFFICIAL_CONFIG_SHA256}"
        )
    model_config = json.loads(config_bytes)
    from sglang.srt.layers.attention.redknot.pro0813.profile import (
        inspect_pro0813_config,
    )

    model_geometry = inspect_pro0813_config(model_config, tp_size=PRO0813_TP_SIZE)
    if (
        model_geometry.variant != PRO0813_VARIANT
        or model_geometry.geometry_digest != PRO0813_TP8_GEOMETRY_DIGEST
        or tuple(model_geometry.dense_layer_ids) != PRO0813_DENSE_LAYER_IDS
        or tuple(model_geometry.reusable_layer_ids) != PRO0813_REUSABLE_LAYER_IDS
    ):
        raise ValueError("drift calibration model failed the Pro-0813 identity gate")
    num_layers = int(model_config.get("num_hidden_layers", 0))
    num_heads = int(model_config.get("num_attention_heads", 0))
    index_topk = int(model_config.get("index_topk", 0))
    o_lora_rank = int(model_config.get("o_lora_rank", 0))
    if (
        num_layers != PRO0813_NUM_LAYERS
        or num_heads != PRO0813_NUM_HEADS
        or index_topk != PRO0813_INDEX_TOPK
        or o_lora_rank != PRO0813_O_LORA_RANK
    ):
        raise ValueError(
            "drift calibration requires the official Pro-0813 61-layer/"
            "128-head/TopK-1024 geometry, got "
            f"layers={num_layers}, heads={num_heads}, index_topk={index_topk}, "
            f"o_lora_rank={o_lora_rank}"
        )
    if DRIFT_MAX_CAPTURE_BYTES <= 0:
        raise ValueError("REDKNOT_MLA_DRIFT_MAX_CAPTURE_BYTES must be positive")
    estimated_capture_bytes = _drift_capture_tensor_bytes(
        num_segments=DRIFT_NUM_CHUNKS,
        sampled_rows=len(sample_rows),
        o_lora_rank=o_lora_rank,
    )
    if estimated_capture_bytes > DRIFT_MAX_CAPTURE_BYTES:
        raise ValueError(
            "drift capture estimate exceeds REDKNOT_MLA_DRIFT_MAX_CAPTURE_BYTES: "
            f"estimated={estimated_capture_bytes} limit={DRIFT_MAX_CAPTURE_BYTES}; "
            "reduce chunks/sample rows or raise the explicit resource budget"
        )
    oracle_orders = [
        [
            segment_ids[(offset + index) % len(segment_ids)]
            for index in range(len(segment_ids))
        ]
        for offset in range(len(segment_ids))
    ]
    segment_length_by_id = dict(
        zip(segment_ids, (len(chunk) for chunk in chunks))
    )
    manifest_segments = [
        {
            "id": segment_id,
            "length": len(chunk),
            "sample_rows": list(sample_rows),
            "token_ids_sha256": token_digest,
        }
        for segment_id, chunk, token_digest in zip(
            segment_ids, chunks, token_digests
        )
    ]
    manifest_oracles = [
        {
            "id": f"oracle-{oracle_index:02d}",
            "logical_seq_len": sum(
                segment_length_by_id[segment_id]
                for segment_id in oracle_order
            ),
            "segments": [
                {
                    "id": segment_id,
                    "slot": slot,
                    "start": sum(
                        segment_length_by_id[previous]
                        for previous in oracle_order[:slot]
                    ),
                    "length": segment_length_by_id[segment_id],
                }
                for slot, segment_id in enumerate(oracle_order)
            ],
        }
        for oracle_index, oracle_order in enumerate(oracle_orders)
    ]
    canonical_payload = _build_pro0813_drift_payload(
        run_id=run_id,
        config_sha256=config_sha256,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        o_lora_rank=o_lora_rank,
        segments=manifest_segments,
        oracles=manifest_oracles,
        estimated_capture_tensor_bytes=estimated_capture_bytes,
        max_capture_tensor_bytes=DRIFT_MAX_CAPTURE_BYTES,
    )
    calibration_digest = _drift_canonical_manifest_digest(canonical_payload)
    output_dir = Path(
        DRIFT_OUT_DIR or f"/tmp/redknot_pro0813_mla_head_drift_{run_id}"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": _DRIFT_MANIFEST_FORMAT,
        "digest_scope": _DRIFT_DIGEST_SCOPE,
        "canonical_payload": canonical_payload,
        "calibration_digest": calibration_digest,
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "calibration_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    # Fail closed before constructing Engine (and therefore before touching a
    # GPU): parse the bytes we actually published with the same strict v3/v2
    # consumer that will later bind every capture and analyzer result.
    from sglang.srt.layers.attention.redknot.mla_head_drift_analyze import (
        load_calibration_manifest,
    )

    validated_manifest = load_calibration_manifest(manifest_path)
    validated_identity = (
        validated_manifest.variant,
        validated_manifest.geometry_digest,
        validated_manifest.official_config_sha256,
    )
    expected_identity = (
        PRO0813_VARIANT,
        PRO0813_TP8_GEOMETRY_DIGEST,
        PRO0813_OFFICIAL_CONFIG_SHA256,
    )
    if validated_identity != expected_identity:
        raise RuntimeError(
            "published drift manifest failed the Pro-0813 identity gate"
        )
    if (
        validated_manifest.run_id != run_id
        or validated_manifest.calibration_digest != calibration_digest
        or validated_manifest.estimated_capture_tensor_bytes
        != estimated_capture_bytes
        or validated_manifest.max_capture_tensor_bytes
        != DRIFT_MAX_CAPTURE_BYTES
    ):
        raise RuntimeError(
            "published drift manifest differs from the producer run/resource plan"
        )

    total_tokens = DRIFT_NUM_CHUNKS * DRIFT_CHUNK_TOKENS
    print("=" * 100)
    print(" REDKNOT DSV4 PAIRED PROJECTED-HEAD DRIFT CALIBRATION")
    print(
        f" model={MODEL_PATH} geometry={DRIFT_NUM_CHUNKS}x"
        f"{DRIFT_CHUNK_TOKENS} total={total_tokens} TP={TP_SIZE}"
    )
    print(
        f" sampled_rows/block={len(sample_rows)} boundary_excluded="
        f"{DRIFT_BOUNDARY_ROWS} out={output_dir}"
    )
    print(
        " capture_tensor_estimate="
        f"{estimated_capture_bytes / (1 << 30):.3f}GiB "
        f"budget={DRIFT_MAX_CAPTURE_BYTES / (1 << 30):.3f}GiB"
    )
    print("=" * 100, flush=True)

    engine = None
    snapshot_paths = []
    oracle_paths = []
    try:
        if DRIFT_HTTP_PORT:
            # launch_server does not advertise readiness until its own real
            # model warm-up succeeds.  Prove this client is talking to that
            # live loopback endpoint, then remove health-probe KV state.
            ready_started = time.perf_counter()
            response = _drift_http_request(
                "GET", "/health_generate", timeout=DRIFT_HTTP_TIMEOUT
            )
            response.raise_for_status()
            response = _drift_http_request(
                "POST", "/flush_cache", timeout=DRIFT_HTTP_TIMEOUT
            )
            response.raise_for_status()
            print(
                f"[drift] HTTP TP ready probe on port {DRIFT_HTTP_PORT} and "
                f"cache flush completed in {time.perf_counter() - ready_started:.3f}s",
                flush=True,
            )

            def generate(input_ids, sampling_params, plan):
                # Every snapshot/oracle is an independent dense-context
                # observation.  Reset allocator/tree state between requests so
                # a preceding context length cannot affect the paired capture.
                response = _drift_http_request(
                    "POST", "/flush_cache", timeout=DRIFT_HTTP_TIMEOUT
                )
                response.raise_for_status()
                _drift_http_generate(input_ids, sampling_params, plan)

        else:
            import sglang as sgl

            engine = sgl.Engine(**_drift_engine_kwargs(total_tokens))
            # Embedded Engine acknowledges worker construction before the TP
            # event loops have proved that they can execute a request.  Exercise
            # a real one-token path, then clear its KV state before calibration.
            if DRIFT_READY_DELAY_SECONDS:
                time.sleep(DRIFT_READY_DELAY_SECONDS)
            ready_started = time.perf_counter()
            engine.generate(
                input_ids=[0, 1, 2],
                sampling_params={"temperature": 0.0, "max_new_tokens": 1},
            )
            engine.flush_cache()
            print(
                "[drift] embedded TP ready probe and cache flush completed in "
                f"{time.perf_counter() - ready_started:.3f}s",
                flush=True,
            )

            def generate(input_ids, sampling_params, plan):
                # Match the HTTP path: each paired observation starts from an
                # empty KV/tree cache rather than inheriting request order.
                engine.flush_cache()
                engine.generate(
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    redknot_reuse_plan=plan,
                )

        # The rank artifact is sealed at the end of the complete EXTEND
        # forward.  One generated token only exercises the supported scheduler
        # path afterwards; forward_decode never enters the drift capture hook.
        sampling = {"temperature": 0.0, "max_new_tokens": 1}
        for index, (segment_id, chunk) in enumerate(zip(segment_ids, chunks)):
            path_template = str(
                output_dir / f"snapshot-{index:02d}-rank{{tp_rank}}.pt"
            )
            plan = _build_pro0813_drift_plan(
                role="snapshot",
                run_id=run_id,
                calibration_digest=calibration_digest,
                out_path=path_template,
                segments=[
                    {
                        "id": segment_id,
                        "start": 0,
                        "length": len(chunk),
                        "token_ids_sha256": token_digests[index],
                    }
                ],
                sample_rows={segment_id: list(sample_rows)},
            )
            started = time.perf_counter()
            generate(chunk, sampling, plan)
            snapshot_paths.extend(_verify_drift_rank_artifacts(path_template))
            print(
                f"[drift] snapshot {index + 1}/{len(chunks)} captured in "
                f"{time.perf_counter() - started:.3f}s",
                flush=True,
            )

        chunk_by_id = dict(zip(segment_ids, chunks))
        token_digest_by_id = dict(zip(segment_ids, token_digests))
        for oracle_index, oracle_order in enumerate(oracle_orders):
            oracle_segments = []
            oracle_start = 0
            for segment_id in oracle_order:
                segment_length = len(chunk_by_id[segment_id])
                oracle_segments.append(
                    {
                        "id": segment_id,
                        "start": oracle_start,
                        "length": segment_length,
                        "token_ids_sha256": token_digest_by_id[segment_id],
                    }
                )
                oracle_start += segment_length
            oracle_ids = [
                token
                for segment_id in oracle_order
                for token in chunk_by_id[segment_id]
            ]
            oracle_template = str(
                output_dir / f"oracle-{oracle_index:02d}-rank{{tp_rank}}.pt"
            )
            oracle_plan = _build_pro0813_drift_plan(
                role="oracle",
                run_id=run_id,
                calibration_digest=calibration_digest,
                out_path=oracle_template,
                segments=oracle_segments,
                sample_rows={
                    segment_id: list(sample_rows) for segment_id in oracle_order
                },
            )
            started = time.perf_counter()
            generate(oracle_ids, sampling, oracle_plan)
            oracle_paths.extend(_verify_drift_rank_artifacts(oracle_template))
            print(
                f"[drift] composed oracle {oracle_index + 1}/"
                f"{len(oracle_orders)} captured in "
                f"{time.perf_counter() - started:.3f}s",
                flush=True,
            )
    finally:
        if engine is not None:
            engine.shutdown()

    analyzer = (
        REPO
        / "python/sglang/srt/layers/attention/redknot/"
        "mla_head_drift_analyze.py"
    )
    if not analyzer.is_file():
        raise RuntimeError(f"drift analyzer is missing: {analyzer}")
    analysis_dir = output_dir / "analysis"
    subprocess.run(
        _drift_analyzer_command(analyzer, output_dir, manifest_path),
        check=True,
        cwd=REPO,
    )
    print(f"[drift] rank captures: {len(snapshot_paths) + len(oracle_paths)}")
    print(
        "[drift] nested 16/32/48/64/96/112-head Pro candidates: "
        f"{analysis_dir}"
    )


def _engine_generate_all(
    prompt_ids: list[list[int]], attention_backend: str, *, sparse_ffn: bool
):
    import sglang as sgl

    engine = sgl.Engine(**_engine_kwargs(attention_backend, sparse_ffn=sparse_ffn))
    sampling_params = {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS}
    warmup_iters = int(os.environ.get("REDKNOT_ENGINE_WARMUP_ITERS", "0"))
    if warmup_iters < 0:
        raise ValueError("REDKNOT_ENGINE_WARMUP_ITERS must be non-negative")
    profile_dir_raw = os.environ.get("REDKNOT_INTERNAL_PROFILE_DIR", "").strip()
    profile_dir = Path(profile_dir_raw).expanduser().resolve() if profile_dir_raw else None
    profile_prefix = os.environ.get(
        "REDKNOT_INTERNAL_PROFILE_PREFIX", "redknot-model-prefill"
    ).strip()
    if profile_dir is not None:
        if profile_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite internal profile directory: {profile_dir}"
            )
        if not profile_prefix:
            raise ValueError("REDKNOT_INTERNAL_PROFILE_PREFIX must be non-empty")
    try:
        if prompt_ids and warmup_iters:
            warmup_params = {"temperature": 0.0, "max_new_tokens": 1}
            for warmup_index in range(warmup_iters):
                engine.generate(
                    input_ids=prompt_ids[0], sampling_params=warmup_params
                )
                print(
                    "[engine-warmup] "
                    f"iteration={warmup_index + 1}/{warmup_iters} "
                    f"prompt_tokens={len(prompt_ids[0])} excluded_from_metrics=true",
                    flush=True,
                )
        outputs = []
        for prompt_index, input_ids in enumerate(prompt_ids):
            profile_this_request = profile_dir is not None and prompt_index == 0
            if profile_this_request:
                engine.start_profile(
                    output_dir=str(profile_dir),
                    activities=["CPU", "GPU"],
                    with_stack=False,
                    record_shapes=False,
                    profile_prefix=profile_prefix,
                )
                print(
                    "[internal-profile] "
                    f"start request_index={prompt_index} output_dir={profile_dir}",
                    flush=True,
                )
            started_at = time.perf_counter()
            try:
                output = engine.generate(
                    input_ids=input_ids, sampling_params=sampling_params
                )
            finally:
                if profile_this_request:
                    engine.stop_profile()
                    print(
                        "[internal-profile] "
                        f"stop request_index={prompt_index} output_dir={profile_dir}",
                        flush=True,
                    )
            finished_at = time.perf_counter()

            text = output.get("text", "") if isinstance(output, dict) else str(output)
            final_meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
            wall_e2e = finished_at - started_at
            e2e = float(final_meta.get("e2e_latency", wall_e2e) or wall_e2e)
            prompt_tokens = int(final_meta.get("prompt_tokens", 0) or 0)
            completion_tokens = int(final_meta.get("completion_tokens", 0) or 0)
            decode_throughput = float(final_meta.get("decode_throughput", 0.0) or 0.0)
            decode_seconds = (
                (completion_tokens - 1) / decode_throughput
                if completion_tokens > 1 and decode_throughput > 0
                else 0.0
            )
            forward_entry_time = final_meta.get("forward_entry_time")
            prefill_finished_time = final_meta.get("prefill_finished_time")
            queue_time = final_meta.get("queue_time")
            if not isinstance(forward_entry_time, (int, float)) or isinstance(
                forward_entry_time, bool
            ):
                raise RuntimeError(
                    "engine metrics omitted scheduler forward_entry_time"
                )
            if not isinstance(prefill_finished_time, (int, float)) or isinstance(
                prefill_finished_time, bool
            ):
                raise RuntimeError(
                    "engine metrics omitted scheduler prefill_finished_time"
                )
            if not isinstance(queue_time, (int, float)) or isinstance(
                queue_time, bool
            ):
                raise RuntimeError("engine metrics omitted scheduler queue_time")
            model_prefill_ttft = float(prefill_finished_time) - float(
                forward_entry_time
            )
            if not math.isfinite(model_prefill_ttft) or model_prefill_ttft <= 0:
                raise RuntimeError(
                    "engine returned an invalid internal model prefill interval: "
                    f"forward_entry={forward_entry_time!r} "
                    f"prefill_finished={prefill_finished_time!r}"
                )
            if not math.isfinite(float(queue_time)) or float(queue_time) < 0:
                raise RuntimeError(
                    f"engine returned an invalid scheduler queue time: {queue_time!r}"
                )
            # Scheduler records decode throughput from first-token to finish.
            # Therefore TTFT = server E2E - decode duration, using timestamps
            # from the same request clock rather than client-side estimation.
            ttft = max(e2e - decode_seconds, 0.0)
            outputs.append(
                {
                    "text": text,
                    "meta_info": final_meta,
                    "bench_metrics": {
                        "ttft": ttft,
                        "e2e": e2e,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "prompt_throughput": (
                            prompt_tokens / ttft if ttft > 0 else 0.0
                        ),
                        "decode_throughput": decode_throughput,
                        "wall_e2e": wall_e2e,
                        "model_prefill_ttft": model_prefill_ttft,
                        "scheduler_queue_time": float(queue_time),
                        "server_non_model_ttft": max(
                            ttft - model_prefill_ttft, 0.0
                        ),
                        "engine_warmup_iters": warmup_iters,
                    },
                }
            )
        return outputs
    finally:
        engine.shutdown()


# ══════════════════════════════════════════════════════════════════════════
# Legacy Indexer-hot and context-bound pure-MLA qualification benchmark.
#
# Two-phase HTTP protocol against a live RedKnot MLA server:
#   1) PURE SNAPSHOT: prefill cumulative prefixes of 8K..64K and publish only
#      the final exact 8K microforward of each request.
#   2) PURE RESTORE: authenticate the frozen full token stream and restore all
#      context-qualified document rows at their exact source positions.
# The separate legacy mode retains selected-row/Indexer-hot behavior.
# Reports TTFT speedup (max_new=1), compute-saving (active-row ratio), and
# first-token / long-text agreement vs standard full recompute.
# ══════════════════════════════════════════════════════════════════════════


def _ih_local_request(method: str, path: str, **kwargs):
    """Send one loopback request without inheriting shell proxy settings."""
    import requests

    # Remote developer shells commonly export http_proxy for model/data access.
    # requests otherwise sends even 127.0.0.1 probes through that proxy when
    # NO_PROXY is absent, making a healthy local SGLang server look unavailable.
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method, f"http://127.0.0.1:{IH_PORT}{path}", **kwargs)


def _ih_http_post(path: str, payload: dict, timeout: int = 1800):
    r = _ih_local_request(
        "POST",
        path,
        json=payload,
        timeout=timeout,
        headers={"Connection": "close"},
    )
    r.raise_for_status()
    return r.json()


class _IHPerformanceMeasurementError(RuntimeError):
    """Raised when a latency stream cannot prove that a token was observed."""


def _ih_apply_diagnostic_claim_gate(
    qualification: Mapping[str, object], diagnostic_ablation: str
) -> dict:
    """Make non-full attribution runs ineligible for production claims."""

    if diagnostic_ablation not in ("full", "zoff_only", "shared_only"):
        raise ValueError("unknown MLA-off diagnostic ablation")
    result = dict(qualification)
    if diagnostic_ablation == "full":
        return result
    reasons = list(result.get("reasons", ()))
    reasons.append(_IH_MLA_OFF_DIAGNOSTIC_CLAIM_REASON)
    result.update(
        {
            "eligible": False,
            "reasons": sorted(set(reasons)),
            "diagnostic_only": True,
            "diagnostic_ablation": diagnostic_ablation,
            "claim_status": "diagnostic_only",
            "claim_ineligible_reason": (
                _IH_MLA_OFF_DIAGNOSTIC_CLAIM_REASON
            ),
        }
    )
    return result


def _ih_apply_qualification_only_claim_gate(
    qualification: Mapping[str, object], qualification_only: bool
) -> dict:
    """Make a non-certified pure-MLA qualification run claim-ineligible."""

    if type(qualification_only) is not bool:
        raise ValueError("qualification-only flag must be a boolean")
    result = dict(qualification)
    if not qualification_only:
        return result
    reasons = list(result.get("reasons", ()))
    reasons.append(_IH_MLA_OFF_QUALIFICATION_CLAIM_REASON)
    result.update(
        {
            "eligible": False,
            "reasons": sorted(set(reasons)),
            "qualification_only": True,
            "claim_status": "qualification_only",
            "claim_ineligible_reason": (
                _IH_MLA_OFF_QUALIFICATION_CLAIM_REASON
            ),
        }
    )
    return result


def _ih_result_claim_mode(
    *,
    mla_diagnostic_only: bool,
    qualification_only: bool,
    performance_diagnostic_only: bool,
) -> Tuple[str, str]:
    """Resolve the one mutually exclusive claim label written to results."""

    if any(
        type(value) is not bool
        for value in (
            mla_diagnostic_only,
            qualification_only,
            performance_diagnostic_only,
        )
    ):
        raise ValueError("result claim-mode flags must be booleans")
    if mla_diagnostic_only:
        return "diagnostic_only", _IH_MLA_OFF_DIAGNOSTIC_CLAIM_REASON
    if qualification_only:
        return "qualification_only", _IH_MLA_OFF_QUALIFICATION_CLAIM_REASON
    if performance_diagnostic_only:
        return "diagnostic_only", _IH_PERFORMANCE_DIAGNOSTIC_CLAIM_REASON
    return "formal_candidate", ""


def _ih_qps_execution_complete(
    qps_results,
    *,
    required: bool,
    expected_concurrencies: Sequence[int],
    expected_warmup_waves: int,
    expected_waves: int,
) -> bool:
    """Prove that every requested paired-QPS request completed successfully.

    This is an execution-integrity gate, not a throughput claim gate.  In
    particular, it deliberately does not inspect the observed speedup or the
    formal minimum sampling sizes.  A short diagnostic sweep can therefore be
    execution-complete without becoming eligible for a performance claim.
    """

    if type(required) is not bool:
        raise ValueError("QPS execution requirement must be a boolean")
    if not required:
        return True
    concurrencies = tuple(expected_concurrencies)
    if (
        not concurrencies
        or any(type(value) is not int or value <= 0 for value in concurrencies)
        or type(expected_warmup_waves) is not int
        or expected_warmup_waves < 0
        or type(expected_waves) is not int
        or expected_waves <= 0
        or not isinstance(qps_results, Mapping)
    ):
        return False
    if (
        qps_results.get("protocol") != "paired_ab_ba_v1"
        or tuple(qps_results.get("concurrencies", ())) != concurrencies
        or qps_results.get("warmup_waves") != expected_warmup_waves
        or qps_results.get("paired_waves") != expected_waves
        or qps_results.get("all_requests_succeeded") is not True
    ):
        return False

    paired_points = qps_results.get("paired_points")
    if not isinstance(paired_points, list) or len(paired_points) != len(
        concurrencies
    ):
        return False
    paired_request_ids = {}
    paired_wall_seconds = {}
    all_request_ids = set()
    for concurrency, point in zip(concurrencies, paired_points):
        if (
            not isinstance(point, Mapping)
            or point.get("concurrency") != concurrency
            or point.get("warmup_waves_per_mode") != expected_warmup_waves
            or point.get("paired_waves") != expected_waves
        ):
            return False
        pairs = point.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != expected_waves:
            return False
        request_ids_by_mode = {"dense": [], "reuse": []}
        wall_seconds_by_mode = {"dense": [], "reuse": []}
        previous_order = None
        for pair_index, pair in enumerate(pairs):
            order = (
                tuple(pair.get("order", ()))
                if isinstance(pair, Mapping)
                else ()
            )
            if (
                not isinstance(pair, Mapping)
                or pair.get("pair_index") != pair_index
                or len(order) != 2
                or set(order) != {"dense", "reuse"}
                or (
                    previous_order is not None
                    and order != previous_order[::-1]
                )
            ):
                return False
            previous_order = order
            for mode in ("dense", "reuse"):
                record = pair.get(mode)
                wall_seconds = (
                    record.get("wall_seconds")
                    if isinstance(record, Mapping)
                    else None
                )
                request_ids = (
                    record.get("request_ids")
                    if isinstance(record, Mapping)
                    else None
                )
                if (
                    not isinstance(record, Mapping)
                    or record.get("attempted") != concurrency
                    or record.get("succeeded") != concurrency
                    or record.get("failed") != 0
                    or record.get("failed_requests") not in ([], ())
                    or not isinstance(wall_seconds, (int, float))
                    or isinstance(wall_seconds, bool)
                    or not math.isfinite(float(wall_seconds))
                    or float(wall_seconds) <= 0.0
                    or not isinstance(request_ids, list)
                    or len(request_ids) != concurrency
                    or len(set(request_ids)) != concurrency
                    or any(
                        not isinstance(request_id, str) or not request_id
                        for request_id in request_ids
                    )
                    or all_request_ids.intersection(request_ids)
                ):
                    return False
                all_request_ids.update(request_ids)
                request_ids_by_mode[mode].extend(request_ids)
                wall_seconds_by_mode[mode].append(float(wall_seconds))
        for mode in ("dense", "reuse"):
            paired_request_ids[(concurrency, mode)] = request_ids_by_mode[mode]
            paired_wall_seconds[(concurrency, mode)] = wall_seconds_by_mode[mode]

    for mode in ("dense", "reuse"):
        sweep = qps_results.get(mode)
        if (
            not isinstance(sweep, Mapping)
            or tuple(sweep.get("concurrencies", ())) != concurrencies
            or sweep.get("warmup_waves") != expected_warmup_waves
            or sweep.get("waves") != expected_waves
            or sweep.get("all_requests_succeeded") is not True
        ):
            return False
        points = sweep.get("points")
        if not isinstance(points, list) or len(points) != len(concurrencies):
            return False
        for point_index, (concurrency, point) in enumerate(
            zip(concurrencies, points)
        ):
            expected_attempted = concurrency * expected_waves
            expected_warmup_attempted = concurrency * expected_warmup_waves
            request_ids = (
                point.get("request_ids")
                if isinstance(point, Mapping)
                else None
            )
            warmup_request_ids = (
                point.get("warmup_request_ids")
                if isinstance(point, Mapping)
                else None
            )
            wave_seconds = (
                point.get("wave_seconds")
                if isinstance(point, Mapping)
                else None
            )
            wall_seconds = (
                point.get("wall_seconds")
                if isinstance(point, Mapping)
                else None
            )
            requests_per_second = (
                point.get("requests_per_second")
                if isinstance(point, Mapping)
                else None
            )
            expected_wall_seconds = sum(
                paired_wall_seconds.get((concurrency, mode), ())
            )
            expected_measured_request_ids = paired_request_ids.get(
                (concurrency, mode)
            )
            if (
                not isinstance(point, Mapping)
                or point.get("concurrency") != concurrency
                or point.get("warmup_waves") != expected_warmup_waves
                or point.get("warmup_attempted")
                != expected_warmup_attempted
                or point.get("warmup_failed") != 0
                or point.get("warmup_failures") != []
                or not isinstance(warmup_request_ids, list)
                or len(warmup_request_ids) != expected_warmup_attempted
                or len(set(warmup_request_ids)) != expected_warmup_attempted
                or any(
                    not isinstance(request_id, str) or not request_id
                    for request_id in warmup_request_ids
                )
                or all_request_ids.intersection(warmup_request_ids)
                or point.get("waves") != expected_waves
                or point.get("attempted") != expected_attempted
                or point.get("succeeded") != expected_attempted
                or point.get("failed") != 0
                or point.get("failed_requests") not in ([], ())
                or not isinstance(request_ids, list)
                or len(request_ids) != expected_attempted
                or len(set(request_ids)) != expected_attempted
                or request_ids != expected_measured_request_ids
                or paired_points[point_index].get(mode) != point
                or not isinstance(wave_seconds, list)
                or len(wave_seconds) != expected_waves
                or wave_seconds
                != paired_wall_seconds.get((concurrency, mode))
                or any(
                    not isinstance(seconds, (int, float))
                    or isinstance(seconds, bool)
                    or not math.isfinite(float(seconds))
                    or float(seconds) <= 0.0
                    for seconds in wave_seconds
                )
                or not isinstance(wall_seconds, (int, float))
                or isinstance(wall_seconds, bool)
                or not math.isclose(
                    float(wall_seconds),
                    expected_wall_seconds,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                or not isinstance(requests_per_second, (int, float))
                or isinstance(requests_per_second, bool)
                or not math.isclose(
                    float(requests_per_second),
                    expected_attempted / expected_wall_seconds,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                return False
            all_request_ids.update(warmup_request_ids)
            if mode == "reuse" and (
                point.get("formal_evidence_request_ids") != request_ids
                or point.get("audit_request_ids")
                != warmup_request_ids + request_ids
            ):
                return False
    return True


def _ih_execution_qualification(
    *,
    diagnostic_only: bool,
    quality_pass: bool,
    runtime_pass: bool,
    runtime_evidence_pass: bool,
    quality_pair_count: int,
    expected_quality_pair_count: int,
    dense_ttft_samples: Sequence[float],
    reuse_ttft_samples: Sequence[float],
    expected_ttft_iters: int,
    qps_required: bool,
    qps_results,
    expected_qps_concurrencies: Sequence[int],
    expected_qps_warmup_waves: int,
    expected_qps_waves: int,
) -> dict:
    """Build claim-independent execution and diagnostic success gates."""

    for name, value in (
        ("diagnostic_only", diagnostic_only),
        ("quality_pass", quality_pass),
        ("runtime_pass", runtime_pass),
        ("runtime_evidence_pass", runtime_evidence_pass),
        ("qps_required", qps_required),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    quality_pairs_complete = bool(
        type(quality_pair_count) is int
        and type(expected_quality_pair_count) is int
        and expected_quality_pair_count > 0
        and quality_pair_count == expected_quality_pair_count
    )

    def valid_ttft_stream(samples: Sequence[float]) -> bool:
        return bool(
            type(expected_ttft_iters) is int
            and expected_ttft_iters > 0
            and len(samples) == expected_ttft_iters
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in samples
            )
        )

    dense_ttft_complete = valid_ttft_stream(dense_ttft_samples)
    reuse_ttft_complete = valid_ttft_stream(reuse_ttft_samples)
    qps_complete = _ih_qps_execution_complete(
        qps_results,
        required=qps_required,
        expected_concurrencies=expected_qps_concurrencies,
        expected_warmup_waves=expected_qps_warmup_waves,
        expected_waves=expected_qps_waves,
    )
    gates = {
        "quality": quality_pass,
        "runtime": runtime_pass,
        "runtime_evidence": runtime_evidence_pass,
        "quality_pairs_complete": quality_pairs_complete,
        "dense_ttft_complete": dense_ttft_complete,
        "reuse_ttft_complete": reuse_ttft_complete,
        "qps_complete": qps_complete,
    }
    reasons = sorted(name for name, passed in gates.items() if not passed)
    execution_pass = not reasons
    return {
        "execution_pass": execution_pass,
        "diagnostic_success": bool(diagnostic_only and execution_pass),
        "diagnostic_only": diagnostic_only,
        "claim_independent": True,
        "gates": gates,
        "reasons": reasons,
        "observed": {
            "quality_pairs": quality_pair_count,
            "expected_quality_pairs": expected_quality_pair_count,
            "dense_ttft_samples": len(dense_ttft_samples),
            "reuse_ttft_samples": len(reuse_ttft_samples),
            "expected_ttft_samples_per_arm": expected_ttft_iters,
            "qps_required": qps_required,
        },
    }


def _ih_performance_claim_qualification(
    *,
    ttft_warmup: int,
    ttft_iters: int,
    qps_warmup_waves: int = IH_QPS_WARMUP_WAVES,
    qps_waves: int = IH_QPS_WAVES,
    measure_qps: bool = IH_MEASURE_QPS,
    min_ttft_speedup: float = IH_MIN_SPEEDUP,
    min_qps_speedup: float = IH_MIN_QPS_SPEEDUP,
    strict: bool = False,
    diagnostic_performance_opt_out: bool | None = None,
) -> dict:
    """Qualify the sampling plan before it may support a performance claim.

    A short diagnostic run is still useful for correctness. Non-strict mode
    records an explicit opt-out and is never eligible for a speedup/QPS claim;
    only a complete result explicitly labelled ``diagnostic_only`` may then
    return zero under the outer diagnostic opt-out. Strict mode rejects an
    ineligible sampling plan before model launch.
    """

    if type(strict) is not bool:
        raise ValueError("performance-claim strictness must be a boolean")
    if diagnostic_performance_opt_out is None:
        diagnostic_performance_opt_out = not strict
    if type(diagnostic_performance_opt_out) is not bool:
        raise ValueError("diagnostic performance opt-out must be a boolean")
    if diagnostic_performance_opt_out != (not strict):
        raise ValueError(
            "diagnostic performance opt-out must be exactly the inverse of "
            "strict performance"
        )
    reasons = []
    if diagnostic_performance_opt_out:
        reasons.append("diagnostic_performance_opt_out")
    if type(ttft_warmup) is not int or ttft_warmup < 3:
        reasons.append("ttft_warmup_below_3")
    if type(ttft_iters) is not int or ttft_iters < 10:
        reasons.append("ttft_iters_below_10")
    if type(measure_qps) is not bool or not measure_qps:
        reasons.append("qps_measurement_disabled")
    if type(qps_warmup_waves) is not int or qps_warmup_waves < 3:
        reasons.append("qps_warmup_waves_below_3")
    if type(qps_waves) is not int or qps_waves < 10:
        reasons.append("qps_measurement_waves_below_10")
    if (
        not isinstance(min_ttft_speedup, (int, float))
        or isinstance(min_ttft_speedup, bool)
        or not math.isfinite(float(min_ttft_speedup))
        or float(min_ttft_speedup) <= 1.0
    ):
        reasons.append("ttft_claim_threshold_not_above_1")
    if (
        not isinstance(min_qps_speedup, (int, float))
        or isinstance(min_qps_speedup, bool)
        or not math.isfinite(float(min_qps_speedup))
        or float(min_qps_speedup) <= 1.0
    ):
        reasons.append("qps_claim_threshold_not_above_1")
    result = {
        "eligible": not reasons,
        "strict": strict,
        "reasons": reasons,
        "requirements": {
            "minimum_ttft_warmup": 3,
            "minimum_ttft_iters": 10,
            "minimum_qps_warmup_waves": 3,
            "minimum_qps_measurement_waves": 10,
            "ttft_speedup_must_exceed": 1.0,
            "qps_speedup_must_exceed": 1.0,
        },
        "observed": {
            "ttft_warmup": ttft_warmup,
            "ttft_iters": ttft_iters,
            "qps_warmup_waves": qps_warmup_waves,
            "qps_measurement_waves": qps_waves,
            "measure_qps": measure_qps,
            "diagnostic_performance_opt_out": (
                diagnostic_performance_opt_out
            ),
            "min_ttft_speedup": min_ttft_speedup,
            "min_qps_speedup": min_qps_speedup,
        },
    }
    if strict and reasons:
        raise ValueError(
            "performance-claim sampling is ineligible: " + ",".join(reasons)
        )
    return result


def _ih_generate_payload(
    input_ids,
    max_new: int,
    *,
    plan,
    request_id: str,
    collect_logprobs: bool,
    stream: bool,
    cache_key: str | None = None,
) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
        "return_logprob": bool(collect_logprobs),
        "stream": bool(stream),
        # A unique radix key prevents dense/reuse measurements from becoming
        # prefix-cache hits without repeatedly flushing the CUDA allocator.
        "extra_key": (
            str(cache_key)
            if cache_key is not None
            else f"redknot-bench-{request_id}"
        ),
    }
    if collect_logprobs:
        payload["top_logprobs_num"] = 10
    if plan is not None:
        request_plan = dict(plan)
        request_plan["benchmark_request_id"] = request_id
        payload["redknot_reuse_plan"] = request_plan
    return payload


def _ih_stream_json_events(response):
    """Yield proven JSON events from native SSE or an explicit JSONL stream.

    The native SGLang ``/generate`` endpoint is SSE with one ``data:`` JSON
    object per event and a final ``data: [DONE]`` marker.  JSON-lines media
    types are accepted for compatible proxies.  Unknown framing, malformed
    lines, non-object JSON, and incomplete SSE all fail closed.
    """

    content_type = str(response.headers.get("content-type", ""))
    media_type = content_type.split(";", 1)[0].strip().lower()
    is_sse = media_type == "text/event-stream"
    is_jsonl = media_type in {
        "application/x-ndjson",
        "application/ndjson",
        "application/jsonl",
    }
    if not is_sse and not is_jsonl:
        raise _IHPerformanceMeasurementError(
            f"unsupported streaming content-type {content_type!r}"
        )

    saw_done = False
    for raw_line in response.iter_lines(decode_unicode=False):
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise _IHPerformanceMeasurementError(
                    "stream contains non-UTF-8 data"
                ) from error
        elif isinstance(raw_line, str):
            line = raw_line
        else:
            raise _IHPerformanceMeasurementError(
                "stream iterator returned a non-text line"
            )
        line = line.strip()
        if not line or (is_sse and line.startswith(":")):
            continue
        if is_sse:
            if not line.startswith("data:"):
                raise _IHPerformanceMeasurementError(
                    f"unsupported SSE field in line {line!r}"
                )
            encoded = line[len("data:") :].lstrip(" ")
            if encoded == "[DONE]":
                saw_done = True
                break
        else:
            encoded = line
        try:
            event = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise _IHPerformanceMeasurementError(
                "stream contains malformed JSON"
            ) from error
        if not isinstance(event, dict):
            raise _IHPerformanceMeasurementError(
                "stream event must be a JSON object"
            )
        yield event
    if is_sse and not saw_done:
        raise _IHPerformanceMeasurementError(
            "SSE stream ended without data: [DONE]"
        )


def _ih_generate_streaming_ttft(
    input_ids,
    plan=None,
    request_tag="ttft",
    *,
    request_id=None,
    timeout: int = 1800,
    request_post=None,
    clock=time.perf_counter,
    cache_key: str | None = None,
):
    """Measure client-observed latency to a proven first output token.

    A non-streaming E2E timestamp is never substituted for TTFT.  The helper
    requires an ``output_ids`` event containing exactly the one token requested
    here, consumes the terminal stream, and fails closed otherwise.
    ``request_post`` is an injectable CPU-test seam accepting the JSON payload.
    """

    request_id = str(request_id or f"{request_tag}-{time.time_ns()}")
    payload = _ih_generate_payload(
        input_ids,
        1,
        plan=plan,
        request_id=request_id,
        collect_logprobs=False,
        stream=True,
        cache_key=cache_key,
    )
    session = None
    response = None
    started = clock()
    try:
        if request_post is None:
            import requests

            session = requests.Session()
            session.trust_env = False
            response = session.post(
                f"http://127.0.0.1:{IH_PORT}/generate",
                json=payload,
                timeout=timeout,
                stream=True,
                headers={"Connection": "close"},
            )
        else:
            response = request_post(payload)
        if response is None:
            raise _IHPerformanceMeasurementError(
                "streaming request returned no response"
            )
        response.raise_for_status()
        first_token_seconds = None
        first_output_ids = None
        event_count = 0
        cached_tokens = None
        final_meta_info = {}
        for event in _ih_stream_json_events(response):
            event_count += 1
            if event.get("error") is not None:
                raise _IHPerformanceMeasurementError(
                    f"server returned a streaming error: {event['error']!r}"
                )
            meta_info = event.get("meta_info")
            if meta_info is not None:
                if not isinstance(meta_info, Mapping):
                    raise _IHPerformanceMeasurementError(
                        "stream meta_info must be a JSON object"
                    )
                observed_cached = meta_info.get("cached_tokens")
                if observed_cached is not None:
                    if type(observed_cached) is not int or observed_cached < 0:
                        raise _IHPerformanceMeasurementError(
                            "stream cached_tokens must be a non-negative integer"
                        )
                    if (
                        cached_tokens is not None
                        and cached_tokens != observed_cached
                    ):
                        raise _IHPerformanceMeasurementError(
                            "stream cached_tokens changed within one request"
                        )
                    cached_tokens = observed_cached
                final_meta_info.update(meta_info)
            if "output_ids" not in event:
                continue
            output_ids = event["output_ids"]
            if not isinstance(output_ids, list) or any(
                type(token_id) is not int for token_id in output_ids
            ):
                raise _IHPerformanceMeasurementError(
                    "stream output_ids must be a list of integer token ids"
                )
            if output_ids:
                if len(output_ids) != 1:
                    raise _IHPerformanceMeasurementError(
                        "max_new_tokens=1 stream emitted a non-singleton token event"
                    )
                if first_token_seconds is not None:
                    raise _IHPerformanceMeasurementError(
                        "max_new_tokens=1 stream emitted more than one token event"
                    )
                first_token_seconds = clock() - started
                first_output_ids = list(output_ids)
        wall_seconds = clock() - started
        if first_token_seconds is None:
            raise _IHPerformanceMeasurementError(
                "stream completed without a proven output_ids token"
            )
        model_prefill_ttft = None
        scheduler_queue_time = None
        server_non_model_ttft = None
        forward_entry_time = final_meta_info.get("forward_entry_time")
        prefill_finished_time = final_meta_info.get("prefill_finished_time")
        queue_time = final_meta_info.get("queue_time")
        if forward_entry_time is not None or prefill_finished_time is not None:
            if (
                not isinstance(forward_entry_time, (int, float))
                or isinstance(forward_entry_time, bool)
                or not isinstance(prefill_finished_time, (int, float))
                or isinstance(prefill_finished_time, bool)
            ):
                raise _IHPerformanceMeasurementError(
                    "stream returned an incomplete model-internal TTFT interval"
                )
            model_prefill_ttft = float(prefill_finished_time) - float(
                forward_entry_time
            )
            if not math.isfinite(model_prefill_ttft) or model_prefill_ttft <= 0:
                raise _IHPerformanceMeasurementError(
                    "stream returned an invalid model-internal TTFT interval"
                )
        if queue_time is not None:
            if (
                not isinstance(queue_time, (int, float))
                or isinstance(queue_time, bool)
                or not math.isfinite(float(queue_time))
                or float(queue_time) < 0
            ):
                raise _IHPerformanceMeasurementError(
                    "stream returned an invalid scheduler queue_time"
                )
            scheduler_queue_time = float(queue_time)
        if IH_REQUIRE_MODEL_TTFT and (
            model_prefill_ttft is None or scheduler_queue_time is None
        ):
            raise _IHPerformanceMeasurementError(
                "strict run requires forward_entry_time, prefill_finished_time, "
                "and queue_time in stream meta_info"
            )
        if model_prefill_ttft is not None:
            server_non_model_ttft = max(
                first_token_seconds - model_prefill_ttft, 0.0
            )
        return {
            "ttft": first_token_seconds,
            "model_prefill_ttft": model_prefill_ttft,
            "scheduler_queue_time": scheduler_queue_time,
            "server_non_model_ttft": server_non_model_ttft,
            "wall_e2e": wall_seconds,
            "first_output_ids": first_output_ids,
            "event_count": event_count,
            "request_id": request_id,
            "cached_tokens": cached_tokens,
        }
    finally:
        if response is not None:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if session is not None:
            session.close()


def _ih_run_qps_wave(
    input_ids,
    max_new: int,
    plan,
    *,
    concurrency: int,
    wave: int,
    phase: str,
    request_tag: str,
    generate_fn,
    clock,
):
    """Run one synchronized request wave and return primitive observations."""

    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(concurrency + 1)

    def invoke(slot: int):
        request_id = (
            f"{request_tag}-{phase}-c{concurrency}-w{wave}-s{slot}-"
            f"{time.time_ns()}"
        )
        barrier.wait()
        try:
            output = generate_fn(
                input_ids,
                max_new,
                plan,
                request_tag=request_tag,
                request_id=request_id,
            )
            observed_id = output.get("request_id")
            if observed_id != request_id:
                raise _IHPerformanceMeasurementError(
                    "QPS request returned a mismatched request_id"
                )
            return {
                "success": True,
                "request_id": request_id,
            }
        except Exception as error:
            return {
                "success": False,
                "request_id": request_id,
                "error_type": type(error).__name__,
                "error": str(error),
            }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(invoke, slot) for slot in range(concurrency)]
        # Start before releasing the barrier.  Starting afterwards lets a fast
        # worker perform unmeasured HTTP work and inflates QPS.
        wave_started = clock()
        barrier.wait()
        records = [future.result() for future in futures]
        elapsed = max(0.0, clock() - wave_started)
    return records, elapsed


def _ih_measure_qps(
    input_ids,
    max_new: int,
    plan=None,
    *,
    concurrencies: Sequence[int] = IH_QPS_CONCURRENCIES,
    waves: int = 1,
    warmup_waves: int = 0,
    request_tag: str = "qps",
    generate_fn=None,
    clock=time.perf_counter,
) -> dict:
    """Run simultaneous request waves and report successful requests/wall time."""

    if type(waves) is not int or waves <= 0:
        raise ValueError("QPS waves must be a positive integer")
    if type(warmup_waves) is not int or warmup_waves < 0:
        raise ValueError("QPS warmup waves must be a non-negative integer")
    normalized_concurrencies = tuple(concurrencies)
    if (
        not normalized_concurrencies
        or any(type(value) is not int or value <= 0 for value in normalized_concurrencies)
        or len(set(normalized_concurrencies)) != len(normalized_concurrencies)
    ):
        raise ValueError("QPS concurrencies must be unique positive integers")
    generate_fn = generate_fn or _ih_generate
    points = []
    for concurrency in normalized_concurrencies:
        successes = []
        failures = []
        warmup_failures = []
        wall_seconds = 0.0
        wave_seconds = []

        for wave in range(warmup_waves):
            records, _ = _ih_run_qps_wave(
                input_ids,
                max_new,
                plan,
                concurrency=concurrency,
                wave=wave,
                phase="warmup",
                request_tag=request_tag,
                generate_fn=generate_fn,
                clock=clock,
            )
            warmup_failures.extend(
                record for record in records if not record["success"]
            )
        for wave in range(waves):
            records, elapsed = _ih_run_qps_wave(
                input_ids,
                max_new,
                plan,
                concurrency=concurrency,
                wave=wave,
                phase="measure",
                request_tag=request_tag,
                generate_fn=generate_fn,
                clock=clock,
            )
            wall_seconds += elapsed
            wave_seconds.append(elapsed)
            successes.extend(record for record in records if record["success"])
            failures.extend(record for record in records if not record["success"])
        attempted = concurrency * waves
        succeeded = len(successes)
        point = {
            "concurrency": concurrency,
            "warmup_waves": warmup_waves,
            "warmup_attempted": concurrency * warmup_waves,
            "warmup_failed": len(warmup_failures),
            "warmup_failures": warmup_failures,
            "waves": waves,
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": len(failures),
            "wall_seconds": wall_seconds,
            "wave_seconds": wave_seconds,
            "per_wave_requests_per_second": [
                concurrency / seconds if seconds > 0 else None
                for seconds in wave_seconds
            ],
            "requests_per_second": (
                succeeded / wall_seconds if wall_seconds > 0 else None
            ),
            "request_ids": [record["request_id"] for record in successes],
            "failed_requests": failures,
        }
        if plan is not None:
            point["formal_evidence_request_ids"] = list(point["request_ids"])
        points.append(point)
    return {
        "concurrencies": list(normalized_concurrencies),
        "warmup_waves": warmup_waves,
        "waves": waves,
        "points": points,
        "all_requests_succeeded": all(
            point["succeeded"] == point["attempted"]
            and point["warmup_failed"] == 0
            for point in points
        ),
    }


def _ih_measure_paired_qps(
    input_ids,
    max_new: int,
    reuse_plan,
    *,
    concurrencies: Sequence[int] = IH_QPS_CONCURRENCIES,
    paired_waves: int = IH_QPS_WAVES,
    warmup_waves: int = IH_QPS_WARMUP_WAVES,
    request_tag: str = "qps-paired",
    seed: int = SEED,
    generate_fn=None,
    clock=time.perf_counter,
    wave_runner=None,
) -> dict:
    """Measure dense/reuse QPS as alternating AB/BA request-wave pairs.

    Each pair contains the same number of dense and reuse requests.  The raw
    wall time for both halves is retained so qualification can derive every
    throughput and speedup independently instead of trusting summary fields.
    """

    if type(paired_waves) is not int or paired_waves <= 0:
        raise ValueError("paired QPS waves must be a positive integer")
    if type(warmup_waves) is not int or warmup_waves < 0:
        raise ValueError("paired QPS warmup waves must be non-negative")
    normalized_concurrencies = tuple(concurrencies)
    if (
        not normalized_concurrencies
        or any(
            type(value) is not int or value <= 0
            for value in normalized_concurrencies
        )
        or len(set(normalized_concurrencies)) != len(normalized_concurrencies)
    ):
        raise ValueError("paired QPS concurrencies must be unique positive integers")
    if reuse_plan is None:
        raise ValueError("paired QPS requires an explicit reuse plan")
    generate_fn = generate_fn or _ih_generate
    wave_runner = wave_runner or _ih_run_qps_wave
    mode_plans = {"dense": None, "reuse": reuse_plan}
    mode_points = {"dense": [], "reuse": []}
    paired_points = []

    for concurrency in normalized_concurrencies:
        first_mode = (
            "dense"
            if random.Random(seed + 104729 * concurrency).randrange(2) == 0
            else "reuse"
        )
        second_mode = "reuse" if first_mode == "dense" else "dense"
        warmup_records_by_mode = {"dense": [], "reuse": []}
        measured_records_by_mode = {"dense": [], "reuse": []}
        measured_wall_by_mode = {"dense": [], "reuse": []}

        # Warm both paths in balanced AB/BA order.  Warmup requests are not part
        # of formal performance evidence, but any warmup failure disqualifies the
        # point through its primitive failure count.
        for wave in range(warmup_waves):
            order = (
                (first_mode, second_mode)
                if wave % 2 == 0
                else (second_mode, first_mode)
            )
            for mode in order:
                records, _ = wave_runner(
                    input_ids,
                    max_new,
                    mode_plans[mode],
                    concurrency=concurrency,
                    wave=wave,
                    phase=f"warmup-{mode}",
                    request_tag=request_tag,
                    generate_fn=generate_fn,
                    clock=clock,
                )
                warmup_records_by_mode[mode].extend(records)

        pairs = []
        for pair_index in range(paired_waves):
            order = (
                (first_mode, second_mode)
                if pair_index % 2 == 0
                else (second_mode, first_mode)
            )
            pair = {"pair_index": pair_index, "order": list(order)}
            for mode in order:
                records, elapsed = wave_runner(
                    input_ids,
                    max_new,
                    mode_plans[mode],
                    concurrency=concurrency,
                    wave=pair_index,
                    phase=f"measure-{mode}",
                    request_tag=request_tag,
                    generate_fn=generate_fn,
                    clock=clock,
                )
                successes = [record for record in records if record["success"]]
                failures = [record for record in records if not record["success"]]
                pair[mode] = {
                    "attempted": concurrency,
                    "succeeded": len(successes),
                    "failed": len(failures),
                    "wall_seconds": elapsed,
                    "request_ids": [record["request_id"] for record in successes],
                    "failed_requests": failures,
                }
                measured_records_by_mode[mode].extend(records)
                measured_wall_by_mode[mode].append(elapsed)
            dense_wall = pair["dense"]["wall_seconds"]
            reuse_wall = pair["reuse"]["wall_seconds"]
            pair["speedup_from_wall_seconds"] = (
                dense_wall / reuse_wall
                if dense_wall > 0 and reuse_wall > 0
                else None
            )
            pairs.append(pair)

        point = {
            "concurrency": concurrency,
            "warmup_waves_per_mode": warmup_waves,
            "paired_waves": paired_waves,
            "pairs": pairs,
        }
        for mode in ("dense", "reuse"):
            warmup_successes = [
                record
                for record in warmup_records_by_mode[mode]
                if record["success"]
            ]
            warmup_failures = [
                record
                for record in warmup_records_by_mode[mode]
                if not record["success"]
            ]
            successes = [
                record
                for record in measured_records_by_mode[mode]
                if record["success"]
            ]
            failures = [
                record
                for record in measured_records_by_mode[mode]
                if not record["success"]
            ]
            total_wall = sum(measured_wall_by_mode[mode])
            mode_point = {
                "concurrency": concurrency,
                "warmup_waves": warmup_waves,
                "warmup_attempted": concurrency * warmup_waves,
                "warmup_failed": len(warmup_failures),
                "warmup_failures": warmup_failures,
                "warmup_request_ids": [
                    record["request_id"] for record in warmup_successes
                ],
                "waves": paired_waves,
                "attempted": concurrency * paired_waves,
                "succeeded": len(successes),
                "failed": len(failures),
                "wall_seconds": total_wall,
                "wave_seconds": list(measured_wall_by_mode[mode]),
                "requests_per_second": (
                    len(successes) / total_wall if total_wall > 0 else None
                ),
                "request_ids": [record["request_id"] for record in successes],
                "failed_requests": failures,
            }
            if mode == "reuse":
                # Warmup reuse requests execute the same audited MLA-off path
                # and therefore emit per-rank controller events.  Keep them
                # out of formal QPS evidence, but return every attempted ID so
                # the runtime audit can validate (rather than ignore) them.
                mode_point["audit_request_ids"] = [
                    record["request_id"]
                    for record in (
                        warmup_records_by_mode[mode]
                        + measured_records_by_mode[mode]
                    )
                ]
                mode_point["formal_evidence_request_ids"] = list(
                    mode_point["request_ids"]
                )
            mode_points[mode].append(mode_point)
            point[mode] = mode_point
        paired_points.append(point)

    result = {
        "protocol": "paired_ab_ba_v1",
        "concurrencies": list(normalized_concurrencies),
        "warmup_waves": warmup_waves,
        "paired_waves": paired_waves,
        "paired_points": paired_points,
    }
    for mode in ("dense", "reuse"):
        result[mode] = {
            "concurrencies": list(normalized_concurrencies),
            "warmup_waves": warmup_waves,
            "waves": paired_waves,
            "points": mode_points[mode],
            "all_requests_succeeded": all(
                point["succeeded"] == point["attempted"]
                and point["warmup_failed"] == 0
                for point in mode_points[mode]
            ),
        }
    result["all_requests_succeeded"] = all(
        result[mode]["all_requests_succeeded"] for mode in ("dense", "reuse")
    )
    return result


def _ih_finalize_performance_claim_qualification(
    sampling_qualification: Mapping[str, object],
    *,
    dense_ttft_samples: Sequence[float],
    reuse_ttft_samples: Sequence[float],
    expected_ttft_iters: int,
    qps_results,
    min_qps_speedup: float = IH_MIN_QPS_SPEEDUP,
) -> dict:
    """Combine the static sampling gate with fail-closed runtime evidence."""

    reasons = list(sampling_qualification.get("reasons", ()))
    for name, samples in (
        ("dense", dense_ttft_samples),
        ("reuse", reuse_ttft_samples),
    ):
        if len(samples) != expected_ttft_iters:
            reasons.append(f"{name}_ttft_sample_count_mismatch")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in samples
        ):
            reasons.append(f"{name}_ttft_contains_invalid_sample")
    qps_speedup_by_concurrency = {}
    qps_aggregate_speedup_by_concurrency = {}
    qps_pair_q25_speedup_by_concurrency = {}
    if qps_results is None:
        reasons.append("qps_sweep_not_measured")
    elif not isinstance(qps_results, Mapping):
        reasons.append("paired_qps_result_invalid")
    else:
        if qps_results.get("protocol") != "paired_ab_ba_v1":
            reasons.append("paired_qps_protocol_mismatch")
        if tuple(qps_results.get("concurrencies", ())) != IH_QPS_CONCURRENCIES:
            reasons.append("paired_qps_concurrency_set_mismatch")
        warmup_waves = qps_results.get("warmup_waves")
        paired_waves = qps_results.get("paired_waves")
        if type(warmup_waves) is not int or warmup_waves < 3:
            reasons.append("paired_qps_warmup_waves_below_3")
        if type(paired_waves) is not int or paired_waves < 10:
            reasons.append("paired_qps_measurement_waves_below_10")
        paired_points = qps_results.get("paired_points")
        if not isinstance(paired_points, list) or len(paired_points) != len(
            IH_QPS_CONCURRENCIES
        ):
            reasons.append("paired_qps_points_incomplete")
            paired_points = []

        wrapper_points = {}
        for mode in ("dense", "reuse"):
            sweep = qps_results.get(mode)
            if not isinstance(sweep, Mapping):
                reasons.append(f"{mode}_qps_sweep_missing")
                wrapper_points[mode] = []
                continue
            if tuple(sweep.get("concurrencies", ())) != IH_QPS_CONCURRENCIES:
                reasons.append(f"{mode}_qps_concurrency_set_mismatch")
            if sweep.get("warmup_waves") != warmup_waves:
                reasons.append(f"{mode}_qps_warmup_waves_mismatch")
            if sweep.get("waves") != paired_waves:
                reasons.append(f"{mode}_qps_measurement_waves_mismatch")
            if sweep.get("all_requests_succeeded") is not True:
                reasons.append(f"{mode}_qps_has_failed_requests")
            points = sweep.get("points")
            if not isinstance(points, list) or len(points) != len(
                IH_QPS_CONCURRENCIES
            ):
                reasons.append(f"{mode}_qps_points_incomplete")
                points = []
            wrapper_points[mode] = points

        all_request_ids = set()
        observed_concurrencies = []
        for point_index, point in enumerate(paired_points):
            if not isinstance(point, Mapping):
                reasons.append("paired_qps_point_invalid")
                continue
            concurrency = point.get("concurrency")
            if type(concurrency) is not int:
                reasons.append("paired_qps_point_concurrency_invalid")
                continue
            observed_concurrencies.append(concurrency)
            point_key = f"c{concurrency}"
            if point.get("warmup_waves_per_mode") != warmup_waves:
                reasons.append(f"paired_qps_{point_key}_warmup_mismatch")
            if point.get("paired_waves") != paired_waves:
                reasons.append(f"paired_qps_{point_key}_wave_count_mismatch")
            pairs = point.get("pairs")
            if not isinstance(pairs, list) or len(pairs) != paired_waves:
                reasons.append(f"paired_qps_{point_key}_pairs_incomplete")
                continue

            walls_by_mode = {"dense": [], "reuse": []}
            request_ids_by_mode = {"dense": [], "reuse": []}
            pair_ratios = []
            previous_order = None
            for pair_index, pair in enumerate(pairs):
                pair_key = f"{point_key}_p{pair_index}"
                if not isinstance(pair, Mapping):
                    reasons.append(f"paired_qps_{pair_key}_invalid")
                    continue
                if pair.get("pair_index") != pair_index:
                    reasons.append(f"paired_qps_{pair_key}_index_mismatch")
                order = tuple(pair.get("order", ()))
                if len(order) != 2 or set(order) != {"dense", "reuse"}:
                    reasons.append(f"paired_qps_{pair_key}_order_invalid")
                elif previous_order is not None and order != previous_order[::-1]:
                    reasons.append(f"paired_qps_{pair_key}_order_not_ab_ba")
                previous_order = order
                pair_walls = {}
                for mode in ("dense", "reuse"):
                    record = pair.get(mode)
                    if not isinstance(record, Mapping):
                        reasons.append(f"paired_qps_{pair_key}_{mode}_missing")
                        continue
                    if (
                        record.get("attempted") != concurrency
                        or record.get("succeeded") != concurrency
                        or record.get("failed") != 0
                        or record.get("failed_requests") not in ([], ())
                    ):
                        reasons.append(
                            f"paired_qps_{pair_key}_{mode}_request_geometry_invalid"
                        )
                    wall = record.get("wall_seconds")
                    if (
                        not isinstance(wall, (int, float))
                        or isinstance(wall, bool)
                        or not math.isfinite(float(wall))
                        or float(wall) <= 0
                    ):
                        reasons.append(f"paired_qps_{pair_key}_{mode}_wall_invalid")
                        continue
                    ids = record.get("request_ids")
                    if (
                        not isinstance(ids, list)
                        or len(ids) != concurrency
                        or any(not isinstance(value, str) or not value for value in ids)
                        or len(set(ids)) != len(ids)
                        or all_request_ids.intersection(ids)
                    ):
                        reasons.append(f"paired_qps_{pair_key}_{mode}_ids_invalid")
                    else:
                        all_request_ids.update(ids)
                        request_ids_by_mode[mode].extend(ids)
                    pair_walls[mode] = float(wall)
                    walls_by_mode[mode].append(float(wall))
                if set(pair_walls) == {"dense", "reuse"}:
                    # Both halves contain the same request count, so the paired
                    # QPS ratio reduces exactly to dense_wall / reuse_wall.
                    ratio = pair_walls["dense"] / pair_walls["reuse"]
                    pair_ratios.append(ratio)
                    reported_ratio = pair.get("speedup_from_wall_seconds")
                    if (
                        not isinstance(reported_ratio, (int, float))
                        or isinstance(reported_ratio, bool)
                        or not math.isfinite(float(reported_ratio))
                        or not math.isclose(
                            float(reported_ratio),
                            ratio,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                    ):
                        reasons.append(f"paired_qps_{pair_key}_ratio_mismatch")

            if all(
                len(walls_by_mode[mode]) == paired_waves
                for mode in ("dense", "reuse")
            ):
                total_walls = {
                    mode: sum(walls_by_mode[mode]) for mode in ("dense", "reuse")
                }
                aggregate_speedup = total_walls["dense"] / total_walls["reuse"]
                pair_q25 = _ih_percentile(pair_ratios, 0.25)
                conservative_speedup = min(aggregate_speedup, pair_q25)
                key = str(concurrency)
                qps_aggregate_speedup_by_concurrency[key] = aggregate_speedup
                qps_pair_q25_speedup_by_concurrency[key] = pair_q25
                qps_speedup_by_concurrency[key] = conservative_speedup
                if conservative_speedup < float(min_qps_speedup):
                    reasons.append(f"qps_speedup_below_threshold_c{concurrency}")

                expected_attempted = concurrency * paired_waves
                expected_warmup_attempted = concurrency * warmup_waves
                for mode in ("dense", "reuse"):
                    summary = point.get(mode)
                    if not isinstance(summary, Mapping):
                        reasons.append(f"{mode}_qps_{point_key}_summary_missing")
                        continue
                    expected_qps = expected_attempted / total_walls[mode]
                    summary_wall = summary.get("wall_seconds")
                    summary_qps = summary.get("requests_per_second")
                    valid_summary = (
                        summary.get("concurrency") == concurrency
                        and summary.get("warmup_waves") == warmup_waves
                        and summary.get("warmup_attempted")
                        == expected_warmup_attempted
                        and summary.get("warmup_failed") == 0
                        and summary.get("waves") == paired_waves
                        and summary.get("attempted") == expected_attempted
                        and summary.get("succeeded") == expected_attempted
                        and summary.get("failed") == 0
                        and summary.get("failed_requests") in ([], ())
                        and summary.get("request_ids") == request_ids_by_mode[mode]
                        and isinstance(summary_wall, (int, float))
                        and not isinstance(summary_wall, bool)
                        and math.isclose(
                            float(summary_wall),
                            total_walls[mode],
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                        and isinstance(summary_qps, (int, float))
                        and not isinstance(summary_qps, bool)
                        and math.isclose(
                            float(summary_qps),
                            expected_qps,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                        and summary.get("wave_seconds") == walls_by_mode[mode]
                    )
                    if mode == "reuse":
                        valid_summary = valid_summary and summary.get(
                            "formal_evidence_request_ids"
                        ) == request_ids_by_mode[mode]
                    if not valid_summary:
                        reasons.append(f"{mode}_qps_{point_key}_summary_mismatch")
                    if (
                        point_index >= len(wrapper_points[mode])
                        or wrapper_points[mode][point_index] != summary
                    ):
                        reasons.append(f"{mode}_qps_{point_key}_wrapper_mismatch")

        if tuple(observed_concurrencies) != IH_QPS_CONCURRENCIES:
            reasons.append("paired_qps_point_order_mismatch")
        if set(qps_speedup_by_concurrency) != {
            str(value) for value in IH_QPS_CONCURRENCIES
        }:
            reasons.append("paired_qps_speedup_unavailable")
    result = dict(sampling_qualification)
    result["eligible"] = not reasons
    result["reasons"] = sorted(set(reasons))
    result["runtime_evidence"] = {
        "dense_ttft_samples": len(dense_ttft_samples),
        "reuse_ttft_samples": len(reuse_ttft_samples),
        "qps_measured": qps_results is not None,
        "qps_speedup_by_concurrency": qps_speedup_by_concurrency,
        "qps_aggregate_speedup_by_concurrency": (
            qps_aggregate_speedup_by_concurrency
        ),
        "qps_pair_q25_speedup_by_concurrency": (
            qps_pair_q25_speedup_by_concurrency
        ),
        "qps_conservative_rule": "min(aggregate_speedup,pair_q25_speedup)",
        "minimum_qps_speedup": min_qps_speedup,
        "minimum_observed_qps_speedup": (
            min(qps_speedup_by_concurrency.values())
            if qps_speedup_by_concurrency
            else None
        ),
    }
    return result


def _ih_flush():
    # A non-streaming generation response can reach the client a few scheduler
    # ticks before the request is retired from ``running_reqs``.  SGLang then
    # returns 400 ("pending requests") even though the request itself is done.
    # Retry within a bounded window so the next isolated snapshot does not race
    # that asynchronous cleanup; persistent 400s still fail closed.
    deadline = time.monotonic() + 60.0
    while True:
        response = _ih_local_request("POST", "/flush_cache", timeout=60)
        if response.status_code != 400 or time.monotonic() >= deadline:
            response.raise_for_status()
            return
        time.sleep(0.25)


def _ih_generate(
    input_ids,
    max_new,
    plan=None,
    request_tag="request",
    collect_logprobs=False,
    *,
    request_id=None,
    cache_key: str | None = None,
):
    request_id = str(request_id or f"{request_tag}-{time.time_ns()}")
    payload = _ih_generate_payload(
        input_ids,
        max_new,
        plan=plan,
        request_id=request_id,
        collect_logprobs=collect_logprobs,
        stream=False,
        cache_key=cache_key,
    )
    t0 = time.perf_counter()
    out = _ih_http_post("/generate", payload)
    if not isinstance(out, Mapping):
        raise _IHPerformanceMeasurementError(
            "generate response must be a JSON object"
        )
    if out.get("error") is not None:
        raise _IHPerformanceMeasurementError(
            f"generate response contains an error: {out['error']!r}"
        )
    output_ids = out.get("output_ids")
    if not isinstance(output_ids, list) or any(
        type(token_id) is not int for token_id in output_ids
    ):
        raise _IHPerformanceMeasurementError(
            "generate response output_ids must be a list of integer token ids"
        )
    if max_new > 0 and not 1 <= len(output_ids) <= max_new:
        raise _IHPerformanceMeasurementError(
            f"generate response produced {len(output_ids)} tokens for "
            f"max_new_tokens={max_new}"
        )
    if max_new == 1 and len(output_ids) != 1:
        raise _IHPerformanceMeasurementError(
            "one-token QPS/quality request did not return exactly one token"
        )
    text_value = out.get("text", "")
    if not isinstance(text_value, str):
        raise _IHPerformanceMeasurementError("generate response text must be a string")
    meta = out.get("meta_info", {})
    if not isinstance(meta, Mapping):
        raise _IHPerformanceMeasurementError(
            "generate response meta_info must be a JSON object"
        )
    cached_tokens = meta.get("cached_tokens")
    if cached_tokens is not None and (
        type(cached_tokens) is not int or cached_tokens < 0
    ):
        raise _IHPerformanceMeasurementError(
            "generate response cached_tokens must be a non-negative integer"
        )
    top = meta.get("output_top_logprobs") or []
    return {
        "text": text_value,
        "output_ids": output_ids,
        "e2e": meta.get("e2e_latency") or (time.perf_counter() - t0),
        "first_top_logprobs": top[0] if top else [],
        "cached_tokens": cached_tokens,
        # This is the client correlation id embedded in the request plan.  A
        # valid returned token proves HTTP execution; formal reuse is bound to
        # the same id independently by rank runtime evidence.
        "request_id": request_id,
    }


def _ih_first_topk_cosine(ref, cand):
    import math

    def as_map(entries):
        return {int(t): float(lp) for lp, t, *_ in entries}

    rm = as_map(ref.get("first_top_logprobs") or [])
    cm = as_map(cand.get("first_top_logprobs") or [])
    keys = set(rm) | set(cm)
    if not keys:
        return 0.0
    rv = [math.exp(rm[k]) if k in rm else 0.0 for k in keys]
    cv = [math.exp(cm[k]) if k in cm else 0.0 for k in keys]
    dot = sum(a * b for a, b in zip(rv, cv))
    rn = math.sqrt(sum(a * a for a in rv))
    cn = math.sqrt(sum(b * b for b in cv))
    return dot / max(rn * cn, 1e-12)


def _ih_chunk_hash(input_ids):
    digest = hashlib.sha256()
    for token_id in input_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return f"sha256:{digest.hexdigest()}"


def _ih_query_chunk_weights(chunks, query_ids):
    """Cheap query sketch used before the model's hidden state exists.

    This is intentionally a chunk-level lexical prior, not a claim that native
    DSV4 Indexer Q is available before chunked prefill.  Native offline Indexer
    frequency remains the block-level score inside each chunk.
    """

    if not chunks:
        return []
    chunk_counts = [Counter(chunk) for chunk in chunks]
    document_frequency = Counter()
    for counts in chunk_counts:
        document_frequency.update(counts.keys())
    query_counts = Counter(query_ids)
    raw = []
    n_chunks = len(chunks)
    for counts in chunk_counts:
        score = 0.0
        for token_id, query_tf in query_counts.items():
            if token_id not in counts:
                continue
            idf = math.log((n_chunks + 1.0) / (document_frequency[token_id] + 0.5))
            score += query_tf * min(counts[token_id], 4) * max(0.0, idf)
        raw.append(score / math.sqrt(max(1, len(chunks[0]))))
    peak = max(raw, default=0.0)
    if peak <= 0.0:
        return [1.0] * len(chunks)
    return [0.05 + 0.95 * score / peak for score in raw]


def _ih_generalized_adaptive_decision(weights):
    """Choose one of three frozen systems shapes without task labels.

    The controller consumes only the lexical sketch already computed before
    inference.  It never observes dataset identity, gold answers, the expected
    evidence chunk, hidden states, logits, or generated text.  Keeping exactly
    three row/protection buckets bounds kernel/graph specialization while a
    diffuse query automatically receives the conservative shape.
    """

    if not isinstance(weights, (list, tuple)) or len(weights) < 2:
        raise ValueError("generalized controller requires at least two chunks")
    values = [float(value) for value in weights]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("generalized controller weights must be finite/nonnegative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("generalized controller weights have zero mass")
    probabilities = [value / total for value in values]
    ranked = sorted(probabilities, reverse=True)
    top1_share = ranked[0]
    top2_share = ranked[1]
    margin = (top1_share - top2_share) / max(top1_share, 1e-12)
    normalized_entropy = -sum(
        probability * math.log(max(probability, 1e-12))
        for probability in probabilities
    ) / math.log(len(probabilities))

    strong = _IH_GENERALIZED_ADAPTIVE_POLICY["strong"]
    medium = _IH_GENERALIZED_ADAPTIVE_POLICY["medium"]
    if (
        top1_share >= strong["min_top1_share"]
        and margin >= strong["min_margin"]
        and normalized_entropy <= strong["max_normalized_entropy"]
    ):
        bucket = "strong"
    elif (
        top1_share >= medium["min_top1_share"]
        and margin >= medium["min_margin"]
        and normalized_entropy <= medium["max_normalized_entropy"]
    ):
        bucket = "medium"
    else:
        bucket = "diffuse"
    selected = _IH_GENERALIZED_ADAPTIVE_POLICY[bucket]
    return {
        "version": _IH_GENERALIZED_ADAPTIVE_CONTROLLER_VERSION,
        "bucket": bucket,
        "active_token_budget_ratio": float(
            selected["active_token_budget_ratio"]
        ),
        "query_protection_tokens": int(selected["query_protection_tokens"]),
        "query_protection_documents": int(
            selected["query_protection_documents"]
        ),
        "features": {
            "top1_share": top1_share,
            "top2_share": top2_share,
            "relative_margin": margin,
            "normalized_entropy": normalized_entropy,
        },
    }


def _ih_query_protected_ranges(
    chunk,
    query_ids,
    *,
    global_offset,
    budget_tokens,
    block_tokens=512,
):
    """Select output-blind lexical windows inside one protected document.

    Query-token TF-IDF identifies anchor blocks.  Each anchor first contributes
    its immediate neighbourhood, retaining local evidence around the lexical
    hit without consulting gold answers, expected chunks, or model output.
    Returned intervals are absolute, sorted, disjoint, and block aligned.
    """

    chunk = tuple(int(token) for token in chunk)
    query_ids = tuple(int(token) for token in query_ids)
    global_offset = int(global_offset)
    budget_tokens = int(budget_tokens)
    block_tokens = int(block_tokens)
    if (
        not chunk
        or block_tokens <= 0
        or len(chunk) % block_tokens != 0
        or budget_tokens <= 0
        or budget_tokens > len(chunk)
        or budget_tokens % block_tokens != 0
        or global_offset < 0
    ):
        raise ValueError("query-protection window geometry is invalid")
    blocks = [
        chunk[begin : begin + block_tokens]
        for begin in range(0, len(chunk), block_tokens)
    ]
    block_counts = [Counter(block) for block in blocks]
    document_frequency = Counter()
    for counts in block_counts:
        document_frequency.update(counts.keys())
    query_counts = Counter(query_ids)
    n_blocks = len(blocks)
    scores = []
    for index, counts in enumerate(block_counts):
        score = 0.0
        for token_id, query_tf in query_counts.items():
            if token_id not in counts:
                continue
            idf = math.log((n_blocks + 1.0) / (document_frequency[token_id] + 0.5))
            score += query_tf * min(counts[token_id], 4) * max(0.0, idf)
        scores.append((score, index))
    anchor_order = [
        index for _, index in sorted(scores, key=lambda item: (-item[0], item[1]))
    ]
    target_blocks = budget_tokens // block_tokens
    selected = set()
    if max((score for score, _ in scores), default=0.0) <= 0.0:
        selected.update(
            min(n_blocks - 1, (slot * n_blocks) // target_blocks)
            for slot in range(target_blocks)
        )
    for anchor in anchor_order:
        if len(selected) == target_blocks:
            break
        for radius in (0, -1, 1, -2, 2):
            candidate = anchor + radius
            if 0 <= candidate < n_blocks:
                selected.add(candidate)
                if len(selected) == target_blocks:
                    break
    if len(selected) != target_blocks:
        raise RuntimeError("query-protection selector did not fill its budget")

    ranges = []
    for block_index in sorted(selected):
        begin = global_offset + block_index * block_tokens
        end = begin + block_tokens
        if ranges and ranges[-1]["end"] == begin:
            ranges[-1]["end"] = end
        else:
            ranges.append({"start": begin, "end": end})
    return ranges


class _IHFastTokenizer:
    """Minimal callable adapter around a local ``tokenizer.json``.

    The HTTP indexer-hot benchmark only needs raw text-to-id conversion.  Using
    ``AutoTokenizer`` for that one operation makes recent Transformers releases
    discover every model package at import time; on a FUSE-mounted checkout that
    can add several minutes to every benchmark process without changing tokens.
    """

    def __init__(self, tokenizer_json):
        from tokenizers import Tokenizer

        tokenizer_path = Path(tokenizer_json).expanduser().resolve()
        tokenizer_bytes = tokenizer_path.read_bytes()
        # As with the official encoder, parse and hash one byte string so the
        # prompt manifest identifies the tokenizer that actually produced ids.
        self._tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        self.source_identity = {
            "path": str(tokenizer_path),
            "sha256": "sha256:" + hashlib.sha256(tokenizer_bytes).hexdigest(),
        }

    def __call__(self, text, add_special_tokens=False):
        return {
            "input_ids": self._tokenizer.encode(
                text, add_special_tokens=bool(add_special_tokens)
            ).ids
        }

    def decode(self, input_ids, skip_special_tokens=True):
        return self._tokenizer.decode(
            list(map(int, input_ids)),
            skip_special_tokens=bool(skip_special_tokens),
        )


def _ih_load_tokenizer():
    tokenizer_json = Path(MODEL_PATH) / "tokenizer.json"
    if tokenizer_json.is_file():
        return _IHFastTokenizer(tokenizer_json)
    # Preserve compatibility with checkpoints that do not expose a standalone
    # tokenizer.json, while keeping the common DeepSeek-V4 path lightweight.
    return _load_tokenizer()


def _ih_prefix_materialization_sentinel(queries) -> int:
    """Choose a valid token that diverges immediately after the offline prefix.

    SGLang may retain only prompt tokens that have a following token in the
    radix tree.  Appending one deterministic, nonmatching token to the native
    materialization request makes every offline document token an internal
    radix node, while ensuring that no query token is accidentally cached.
    """

    if not queries or not queries[0][1]:
        raise ValueError("prefix materialization requires a non-empty suffix")
    first_online_token = int(queries[0][1][0])
    sentinel = 0 if first_online_token != 0 else 1
    if sentinel == first_online_token or sentinel < 0:
        raise AssertionError("failed to choose a divergent sentinel token")
    return sentinel


_IH_DATA_MANIFEST_FORMAT = "redknot_indexer_hot_data_selection_v1"


def _ih_canonical_json_sha256(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_IH_PURE_PROMPT_MANIFEST_FORMAT = "redknot_pure_mla_prompt_v1"
_IH_PURE_PROMPT_MULTI_MANIFEST_FORMAT = "redknot_pure_mla_prompt_multi_v1"
_IH_PURE_SNAPSHOT_PLAN_KEYS = frozenset(
    {
        "mode",
        "capture_mla_off",
        "mla_off_execution_profile",
        "mla_off_head_scope_policy",
        "allow_approximate",
        "seg_hash",
        "token_hash",
        "length",
        "canonical_start_pos",
        "model_compat_hash",
        "head_policy_hash",
        "snapshot_generation_id",
    }
)
_IH_PURE_RESTORE_PLAN_KEYS = frozenset(
    {
        "mode",
        "reuse_mla_off",
        "mla_off_execution_profile",
        "mla_off_head_scope_policy",
        "allow_approximate",
        "query_start",
        "total_tokens",
        "segments",
        "mla_off_qualification_only",
        "model_compat_hash",
        "head_policy_hash",
    }
)
_IH_PURE_RESTORE_MERGED_KEYS = frozenset({"merged_prefill_tokens"})
_IH_COMBINED_SNAPSHOT_KEYS = _IH_PURE_SNAPSHOT_PLAN_KEYS | frozenset(
    {"reuse_window_kv", "checkpoint_stride_tokens"}
)
_IH_COMBINED_RESTORE_KEYS = (
    _IH_PURE_RESTORE_PLAN_KEYS - {"mla_off_qualification_only"}
) | frozenset(
    {
        "mla_off_diagnostic_ablation",
        "skip_forward",
        "inject_full_blocks",
        "refresh_selected_c4_rows",
        "reuse_window_kv",
        "skip_prefix_recompute",
        "selection_policy",
        "active_token_budget_ratio",
        "hot_max_per_segment_ratio",
        "checkpoint_stride_tokens",
        "checkpoint_max_islands",
        "interior_stride",
        "hot_frac",
        "mla_off_use_indexer_hot",
        "row_sparse_closure",
        "query_protection_policy",
        "query_protected_segment_index",
        "query_protected_ranges",
    }
)
_IH_PURE_RADIX_PREFIX_KEYS = frozenset(
    {
        "radix_prefix_role",
        "radix_prefix_tokens",
        "radix_prefix_input_hash",
        "radix_prefix_receipt_key",
    }
)
_IH_PURE_SEGMENT_KEYS = frozenset(
    {
        "seg_hash",
        "token_hash",
        "global_offset",
        "length",
        "canonical_start_pos",
        "skip_first",
    }
)
_IH_PURE_LEGACY_CONFIG_KEYS = frozenset(
    {
        "selection_policy",
        "checkpoint_stride_tokens",
        "checkpoint_max_islands",
        "active_token_budget_ratio",
        "hot_max_per_segment_ratio",
        "relevance_last",
        "relevance_first",
        "skip_prefix_recompute",
        "mla_off_refresh_layer_stride",
        "mla_off_hot_expand_tokens",
        "mla_off_compact_woa",
        "ttft_chunk_order",
    }
)


def _ih_validate_exact_pure_plan(plan: Mapping[str, object], *, mode: str) -> None:
    if IH_COMBINED_HEADSPLIT_ROW_SPARSE:
        expected = (
            _IH_COMBINED_SNAPSHOT_KEYS
            if mode == "snapshot"
            else _IH_COMBINED_RESTORE_KEYS
        )
    else:
        expected = (
            _IH_PURE_SNAPSHOT_PLAN_KEYS
            if mode == "snapshot"
            else _IH_PURE_RESTORE_PLAN_KEYS
        )
    radix_role = plan.get("radix_prefix_role") if mode == "restore" else None
    if (
        mode == "restore"
        and IH_MERGED_PREFILL_TOKENS > 0
        and radix_role != "seed"
    ):
        expected = expected | _IH_PURE_RESTORE_MERGED_KEYS
    if radix_role is not None:
        expected = expected | _IH_PURE_RADIX_PREFIX_KEYS
    observed = frozenset(plan)
    if observed != expected:
        raise ValueError(
            f"pure {mode} plan schema mismatch: "
            f"missing={sorted(expected - observed)} "
            f"unexpected={sorted(observed - expected)}"
        )
    if plan.get("allow_approximate") is not True:
        raise ValueError(
            "independent-document relocation requires allow_approximate=true"
        )
    if plan.get("mla_off_execution_profile") != IH_MLA_OFF_EXECUTION_PROFILE:
        raise ValueError("pure MLA execution profile differs from benchmark contract")
    if plan.get("mla_off_head_scope_policy") != IH_MLA_OFF_HEAD_SCOPE_POLICY:
        raise ValueError("pure MLA must preserve native DSV4 full candidate scope")
    if mode == "snapshot" and plan.get("capture_mla_off") is not True:
        raise ValueError("pure snapshot requires capture_mla_off=true")
    if mode == "snapshot":
        if (
            int(plan["canonical_start_pos"]) != 0
            or int(plan["length"]) != IH_CHUNK_TOKENS
            or not str(plan["snapshot_generation_id"])
        ):
            raise ValueError("independent snapshot geometry/generation is invalid")
    if mode == "restore":
        if plan.get("reuse_mla_off") is not True:
            raise ValueError("pure restore requires reuse_mla_off=true")
        if IH_COMBINED_HEADSPLIT_ROW_SPARSE:
            if "mla_off_qualification_only" in plan:
                raise ValueError(
                    "combined formal restore forbids the qualification marker"
                )
            required_true = (
                "skip_forward",
                "inject_full_blocks",
                "refresh_selected_c4_rows",
                "reuse_window_kv",
                "skip_prefix_recompute",
                "mla_off_use_indexer_hot",
                "row_sparse_closure",
            )
            if any(plan.get(name) is not True for name in required_true):
                raise ValueError("combined restore closure flags are incomplete")
            if (
                plan.get("mla_off_diagnostic_ablation")
                != IH_MLA_OFF_DIAGNOSTIC_ABLATION
                or plan.get("selection_policy") != "checkpoint_islands"
            ):
                raise ValueError(
                    "combined restore diagnostic/checkpoint policy differs "
                    "from the benchmark contract"
                )
            active_ratio = float(plan["active_token_budget_ratio"])
            if IH_GENERALIZED_ADAPTIVE_CONTROLLER:
                if not any(
                    math.isclose(active_ratio, allowed, rel_tol=0.0, abs_tol=1e-9)
                    for allowed in _IH_GENERALIZED_ADAPTIVE_ROW_RATIOS
                ):
                    raise ValueError(
                        "generalized controller selected an uncertified row ratio"
                    )
            elif not math.isclose(
                active_ratio, IH_ACTIVE_BUDGET_RATIO, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    "combined restore row ratio differs from benchmark contract"
                )
            protection_policy = str(plan.get("query_protection_policy", ""))
            protected_index = plan.get("query_protected_segment_index")
            protected_ranges = plan.get("query_protected_ranges")
            if type(protected_index) is not int:
                raise ValueError("combined query-protected segment index is invalid")
            if protection_policy == "none":
                if protected_index != -1 or protected_ranges != []:
                    raise ValueError(
                        "disabled query protection requires index=-1/ranges=[]"
                    )
            elif protection_policy not in {
                "lexical_top1_full_segment_v1",
                "lexical_top1_block_windows_v1",
                "lexical_topk_block_windows_v2",
            }:
                raise ValueError("combined query protection policy is unsupported")
        segments = plan.get("segments")
        expected_segment_count = 1 if radix_role == "seed" else IH_NUM_CHUNKS
        if (
            not isinstance(segments, list)
            or len(segments) != expected_segment_count
        ):
            raise ValueError(
                "pure restore plan segment count differs from the frozen "
                f"geometry: expected={expected_segment_count} "
                f"actual={len(segments) if isinstance(segments, list) else None}"
            )
        if (
            IH_COMBINED_HEADSPLIT_ROW_SPARSE
            and str(plan["query_protection_policy"])
            in {
                "lexical_top1_full_segment_v1",
                "lexical_top1_block_windows_v1",
                "lexical_topk_block_windows_v2",
            }
            and not 0 <= int(plan["query_protected_segment_index"]) < len(segments)
        ):
            raise ValueError(
                "query-protected segment is outside the restore chain"
            )
        if IH_COMBINED_HEADSPLIT_ROW_SPARSE:
            protected_ranges = plan["query_protected_ranges"]
            if not isinstance(protected_ranges, list):
                raise ValueError("query-protected ranges must be a list")
            if str(plan["query_protection_policy"]) != "none":
                protected_index = int(plan["query_protected_segment_index"])
                protected_segment = segments[protected_index]
                protected_begin = int(protected_segment["global_offset"])
                protected_end = protected_begin + int(protected_segment["length"])
                cursor = 0
                total = 0
                protected_segment_indices = set()
                for item in protected_ranges:
                    if not isinstance(item, dict) or frozenset(item) != {
                        "start",
                        "end",
                    }:
                        raise ValueError("query-protected range schema is invalid")
                    begin, end = int(item["start"]), int(item["end"])
                    containing = [
                        segment_index
                        for segment_index, segment in enumerate(segments)
                        if begin >= int(segment["global_offset"])
                        and end
                        <= int(segment["global_offset"]) + int(segment["length"])
                    ]
                    if (
                        len(containing) != 1
                        or begin < cursor
                        or begin >= end
                        or begin % 512 != 0
                        or end % 512 != 0
                    ):
                        raise ValueError("query-protected range geometry is invalid")
                    cursor = end
                    total += end - begin
                    protected_segment_indices.add(containing[0])
                if str(plan["query_protection_policy"]) == "lexical_top1_full_segment_v1":
                    if protected_ranges != [
                        {"start": protected_begin, "end": protected_end}
                    ]:
                        raise ValueError("full-segment protection range is incomplete")
                elif str(plan["query_protection_policy"]) == "lexical_top1_block_windows_v1":
                    if protected_segment_indices != {protected_index}:
                        raise ValueError("top1 protection escaped its selected segment")
                elif str(plan["query_protection_policy"]) == "lexical_topk_block_windows_v2" and (
                    protected_index not in protected_segment_indices
                    or len(protected_segment_indices) != 2
                ):
                    raise ValueError(
                        "topk protection must cover exactly two segments including top1"
                    )
                if str(plan["query_protection_policy"]) != "lexical_top1_full_segment_v1":
                    expected_tokens = IH_QUERY_PROTECTION_TOKENS
                    if IH_GENERALIZED_ADAPTIVE_CONTROLLER:
                        ratio_to_tokens = dict(
                            zip(
                                _IH_GENERALIZED_ADAPTIVE_ROW_RATIOS,
                                _IH_GENERALIZED_ADAPTIVE_QUERY_TOKENS,
                            )
                        )
                        selected_ratio = float(
                            plan["active_token_budget_ratio"]
                        )
                        expected_tokens = next(
                            tokens
                            for ratio, tokens in ratio_to_tokens.items()
                            if math.isclose(
                                selected_ratio,
                                ratio,
                                rel_tol=0.0,
                                abs_tol=1e-9,
                            )
                        )
                    if total != expected_tokens:
                        raise ValueError(
                            "query-protected window budget differs from "
                            "benchmark/controller contract"
                        )
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict) or frozenset(segment) != (
                _IH_PURE_SEGMENT_KEYS
            ):
                raise ValueError(
                    f"pure restore segment {index} schema differs from the "
                    "pure offline/online merge contract"
                )
        cursor = 0
        for index, segment in enumerate(segments):
            expected_skip = (
                IH_BOUNDARY
                if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                else (0 if index == 0 else IH_BOUNDARY)
            )
            if (
                int(segment["global_offset"]) != cursor
                or int(segment["length"]) != IH_CHUNK_TOKENS
                or int(segment["canonical_start_pos"]) != 0
                or int(segment["skip_first"]) != expected_skip
            ):
                raise ValueError(
                    "independent restore segment geometry differs from "
                    f"position-0/boundary contract at index={index}"
                )
            cursor += int(segment["length"])
        if radix_role is not None:
            if radix_role not in {"seed", "consume"}:
                raise ValueError("radix-prefix role must be seed or consume")
            prefix_tokens = plan.get("radix_prefix_tokens")
            prefix_hash = plan.get("radix_prefix_input_hash")
            receipt_key = plan.get("radix_prefix_receipt_key")
            if (
                type(prefix_tokens) is not int
                or prefix_tokens != IH_CHUNK_TOKENS
                or segments[0]["length"] != prefix_tokens
                or segments[0]["token_hash"] != prefix_hash
            ):
                raise ValueError(
                    "radix-prefix contract differs from the frozen first document"
                )
            for value in (prefix_hash, receipt_key):
                if (
                    not isinstance(value, str)
                    or len(value) != 71
                    or not value.startswith("sha256:")
                    or any(
                        char not in "0123456789abcdef" for char in value[7:]
                    )
                ):
                    raise ValueError("radix-prefix digest is not canonical SHA-256")
            if radix_role == "seed" and plan["query_start"] != prefix_tokens:
                raise ValueError("radix-prefix seed query boundary is invalid")
        if IH_MERGED_PREFILL_TOKENS > 0 and radix_role != "seed":
            from sglang.srt.layers.attention.redknot.v4.merged_prefill import (
                _validated_plan_request,
            )

            if (
                plan.get("merged_prefill_tokens")
                != IH_MERGED_PREFILL_TOKENS
                or _validated_plan_request(plan) != IH_MERGED_PREFILL_TOKENS
            ):
                raise ValueError(
                    "pure restore merged-prefill plan differs from the frozen request"
                )


def _ih_build_context_segment_contracts(
    chunks: Sequence[Sequence[int]],
    *,
    model_compat_hash: str,
    head_policy_hash: str,
) -> tuple[dict, ...]:
    """Bind every independent position-0 document artifact."""

    contracts = []
    for index, chunk in enumerate(chunks):
        normalized_chunk = tuple(chunk)
        if not normalized_chunk:
            raise ValueError("independent snapshot document must be non-empty")
        token_hash = _ih_chunk_hash(normalized_chunk)
        contract = {
            "token_hash": token_hash,
            "length": len(normalized_chunk),
            "canonical_start_pos": 0,
        }
        contract["seg_hash"] = "sha256:" + hashlib.sha256(
            json.dumps(
                {
                    "schema": "redknot-independent-document-artifact-v1",
                    "document_index": index,
                    "execution_profile": IH_MLA_OFF_EXECUTION_PROFILE,
                    "head_scope_policy": IH_MLA_OFF_HEAD_SCOPE_POLICY,
                    "model_compat_hash": model_compat_hash,
                    "head_policy_hash": head_policy_hash,
                    **contract,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        contracts.append(contract)
    return tuple(contracts)


def _ih_build_context_snapshot_request(
    chunks: Sequence[Sequence[int]],
    contracts: Sequence[Mapping[str, object]],
    *,
    index: int,
    model_compat_hash: str,
    head_policy_hash: str,
) -> tuple[tuple[int, ...], dict]:
    """Build one independent position-0 document snapshot request."""

    if type(index) is not int or index < 0 or index >= len(contracts):
        raise ValueError("context snapshot index is outside the contract chain")
    snapshot_input = tuple(chunks[index])
    contract = dict(contracts[index])
    if len(snapshot_input) != contract.get("length"):
        raise ValueError("independent snapshot length differs from contract")
    if _ih_chunk_hash(snapshot_input) != contract.get("token_hash"):
        raise ValueError("snapshot target tokens differ from contract")
    plan = {
        "mode": "snapshot",
        "capture_mla_off": True,
        "mla_off_execution_profile": IH_MLA_OFF_EXECUTION_PROFILE,
        "mla_off_head_scope_policy": IH_MLA_OFF_HEAD_SCOPE_POLICY,
        "allow_approximate": True,
        **contract,
        "model_compat_hash": model_compat_hash,
        "head_policy_hash": head_policy_hash,
        "snapshot_generation_id": (
            "independent-doc-v1:" + str(contract["seg_hash"])
        ),
    }
    if IH_COMBINED_HEADSPLIT_ROW_SPARSE:
        plan.update(
            {
                "reuse_window_kv": True,
                "checkpoint_stride_tokens": IH_CHECKPOINT_STRIDE,
            }
        )
    _ih_validate_exact_pure_plan(plan, mode="snapshot")
    return snapshot_input, plan


def _ih_build_context_restore_plan(
    composed_prefix: Sequence[int],
    query_ids: Sequence[int],
    contracts: Sequence[Mapping[str, object]],
    *,
    model_compat_hash: str,
    head_policy_hash: str,
    merged_prefill_tokens: int = 0,
    radix_prefix_role: str | None = None,
    radix_prefix_receipt_key: str | None = None,
    segment_limit: int | None = None,
    query_protection_policy: str = "none",
    query_protected_segment_index: int = -1,
    query_protected_ranges: Sequence[Mapping[str, int]] = (),
    active_token_budget_ratio: float | None = None,
) -> dict:
    """Relocate independent position-0 documents into one online RAG prefix."""

    if not contracts:
        raise ValueError("context restore needs at least one offline segment")
    selected_contracts = (
        contracts if segment_limit is None else contracts[: int(segment_limit)]
    )
    segments = [
        {
            **dict(contract),
            "global_offset": index * int(contract["length"]),
            "skip_first": (
                IH_BOUNDARY
                if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                else (0 if index == 0 else IH_BOUNDARY)
            ),
        }
        for index, contract in enumerate(selected_contracts)
    ]
    query_start = len(composed_prefix)
    cursor = 0
    for index, contract in enumerate(selected_contracts):
        length = int(contract["length"])
        document_ids = tuple(composed_prefix[cursor : cursor + length])
        if len(document_ids) != length or _ih_chunk_hash(document_ids) != str(
            contract["token_hash"]
        ):
            raise ValueError(
                "restore prefix differs from independent document "
                f"at index={index}"
            )
        cursor += length
    if cursor != len(composed_prefix):
        raise ValueError("restore prefix has tokens beyond independent documents")
    full_input = tuple(composed_prefix) + tuple(query_ids)
    plan = {
        "mode": "restore",
        "reuse_mla_off": True,
        "mla_off_execution_profile": IH_MLA_OFF_EXECUTION_PROFILE,
        "mla_off_head_scope_policy": IH_MLA_OFF_HEAD_SCOPE_POLICY,
        "allow_approximate": True,
        "query_start": query_start,
        "total_tokens": len(full_input),
        "segments": segments,
        "model_compat_hash": model_compat_hash,
        "head_policy_hash": head_policy_hash,
    }
    if not IH_COMBINED_HEADSPLIT_ROW_SPARSE:
        plan["mla_off_qualification_only"] = True
    if IH_COMBINED_HEADSPLIT_ROW_SPARSE:
        selected_active_ratio = (
            IH_ACTIVE_BUDGET_RATIO
            if active_token_budget_ratio is None
            else float(active_token_budget_ratio)
        )
        plan.update(
            {
                "mla_off_diagnostic_ablation": IH_MLA_OFF_DIAGNOSTIC_ABLATION,
                "skip_forward": True,
                "inject_full_blocks": True,
                "refresh_selected_c4_rows": True,
                "reuse_window_kv": True,
                "skip_prefix_recompute": IH_SKIP_PREFIX_RECOMPUTE,
                "selection_policy": "checkpoint_islands",
                "active_token_budget_ratio": selected_active_ratio,
                "hot_max_per_segment_ratio": IH_HOT_MAX_PER_SEGMENT_RATIO,
                "checkpoint_stride_tokens": IH_CHECKPOINT_STRIDE,
                "checkpoint_max_islands": IH_CHECKPOINT_MAX_ISLANDS,
                "interior_stride": 0,
                "hot_frac": IH_HOT_FRAC,
                "mla_off_use_indexer_hot": True,
                "row_sparse_closure": True,
                "query_protection_policy": str(query_protection_policy),
                "query_protected_segment_index": int(
                    query_protected_segment_index
                ),
                "query_protected_ranges": [
                    {"start": int(item["start"]), "end": int(item["end"])}
                    for item in query_protected_ranges
                ],
            }
        )
    if radix_prefix_role is not None:
        if radix_prefix_role not in {"seed", "consume"}:
            raise ValueError("radix-prefix role must be seed or consume")
        if not isinstance(radix_prefix_receipt_key, str):
            raise ValueError("radix-prefix receipt key is absent")
        plan.update(
            {
                "radix_prefix_role": radix_prefix_role,
                "radix_prefix_tokens": int(contracts[0]["length"]),
                "radix_prefix_input_hash": str(contracts[0]["token_hash"]),
                "radix_prefix_receipt_key": radix_prefix_receipt_key,
            }
        )
    if merged_prefill_tokens and radix_prefix_role != "seed":
        plan["merged_prefill_tokens"] = int(merged_prefill_tokens)
    _ih_validate_exact_pure_plan(plan, mode="restore")
    return plan


def _ih_finalize_pure_result_config(
    config: dict, *, offline_chunk_order: Sequence[int]
) -> None:
    for legacy_key in _IH_PURE_LEGACY_CONFIG_KEYS:
        config.pop(legacy_key, None)
    config["offline_chunk_order"] = [
        int(value) for value in offline_chunk_order
    ]
    if IH_MERGED_PREFILL_TOKENS > 0:
        config["merged_prefill_tokens"] = int(IH_MERGED_PREFILL_TOKENS)
    else:
        config.pop("merged_prefill_tokens", None)
    leaked = _IH_PURE_LEGACY_CONFIG_KEYS.intersection(config)
    if leaked:
        raise AssertionError(
            f"pure result leaked legacy config keys: {sorted(leaked)}"
        )


def _ih_build_official_pure_prompt(
    tok,
    chunks,
    queries,
    data_selection,
    *,
    chunk_tokens,
):
    """Freeze one official DSV4 RAG prompt before any model request.

    The source selection remains the independently replayed LongBench manifest.
    This layer adds explicit document boundaries and the checkpoint's official
    chat protocol, tokenizes the complete prompt once, then splits its frozen
    offline prefix into 8K segments captured under cumulative causal prefixes.
    """

    if not queries:
        raise ValueError("official_rag_v1 requires at least one frozen query")
    supported_geometry = {
        (8, 8192),
        (16, 8192),
        (8, 32768),
        (8, 56320),
        (8, 65536),
    }
    if (len(chunks), int(chunk_tokens)) not in supported_geometry:
        raise ValueError(
            "official_rag_v1 requires a frozen 8x8192, 16x8192, "
            "8x32768, 8x56320, or 8x65536 geometry"
        )
    if data_selection.get("format") != _IH_DATA_MANIFEST_FORMAT:
        raise ValueError("official_rag_v1 requires a frozen data-selection manifest")
    dataset_identity = data_selection.get("dataset")
    if (
        not isinstance(dataset_identity, dict)
        or dataset_identity.get("name") != IH_EXPECTED_DATASET
    ):
        raise ValueError(
            "official_rag_v1 dataset differs from the pre-registered dataset: "
            f"expected={IH_EXPECTED_DATASET!r} actual={dataset_identity!r}"
        )
    if data_selection.get("selection_sha256") != IH_EXPECTED_DATA_SELECTION_SHA256:
        raise ValueError(
            "official_rag_v1 data-selection digest differs from the pre-registered "
            f"digest: expected={IH_EXPECTED_DATA_SELECTION_SHA256} "
            f"actual={data_selection.get('selection_sha256')}"
        )
    query_entries = data_selection.get("selection", {}).get("queries", [])
    query_row_ids = tuple(
        entry.get("row_id") for entry in query_entries
        if isinstance(entry, Mapping)
    )
    if (
        len(query_entries) != len(queries)
        or query_row_ids != IH_EXPECTED_QUERY_ROW_IDS
    ):
        raise ValueError(
            "official_rag_v1 query rows differ from the pre-registered rows: "
            f"expected={IH_EXPECTED_QUERY_ROW_IDS} actual={query_row_ids}"
        )
    if not isinstance(tok, _IHFastTokenizer) or not isinstance(
        getattr(tok, "source_identity", None), dict
    ):
        raise ValueError(
            "official_rag_v1 requires the byte-frozen tokenizer.json adapter"
        )
    source_order = list(range(len(chunks)))
    documents = [
        tok.decode(chunks[source], skip_special_tokens=False)
        for source in source_order
    ]
    query_start = len(chunks) * int(chunk_tokens)
    frozen_prompt_chunks = None
    prompt_cases = []
    rebuilt_queries = []
    for query_index, (
        question,
        raw_query_ids,
        answers,
        expected_source_chunk,
    ) in enumerate(queries):
        del raw_query_ids
        prompt_text = _encode_pure_official_rag_prompt(documents, question)
        full_input_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        if len(full_input_ids) <= query_start:
            raise ValueError(
                "official RAG prompt did not leave a non-empty online suffix"
            )
        prompt_chunks = [
            full_input_ids[start : start + int(chunk_tokens)]
            for start in range(0, query_start, int(chunk_tokens))
        ]
        if len(prompt_chunks) != len(chunks) or any(
            len(chunk) != int(chunk_tokens) for chunk in prompt_chunks
        ):
            raise AssertionError(
                "official prompt split lost the frozen chunk geometry"
            )
        if frozen_prompt_chunks is None:
            frozen_prompt_chunks = prompt_chunks
        elif prompt_chunks != frozen_prompt_chunks:
            raise ValueError(
                "multi-query official prompts do not share one exact offline "
                "prefix; use separate frozen corpora"
            )
        online_suffix_ids = full_input_ids[query_start:]
        if int(expected_source_chunk) not in source_order:
            raise ValueError("query evidence chunk is absent from prompt order")
        prompt_cases.append(
            {
                "query_index": query_index,
                "query_row_id": int(query_row_ids[query_index]),
                "expected_source_chunk": int(expected_source_chunk),
                "online_suffix_tokens": len(online_suffix_ids),
                "total_tokens": len(full_input_ids),
                "text_sha256": "sha256:"
                + hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "full_input_ids_sha256": _ih_chunk_hash(full_input_ids),
                "online_suffix_hash": _ih_chunk_hash(online_suffix_ids),
                "question_sha256": "sha256:"
                + hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "answers_sha256": "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        list(answers),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        rebuilt_queries.append(
            (
                question,
                online_suffix_ids,
                answers,
                int(expected_source_chunk),
            )
        )
    if frozen_prompt_chunks is None:
        raise AssertionError("official prompt builder produced no frozen prefix")
    prompt_chunks = frozen_prompt_chunks

    encoder_path = Path(MODEL_PATH) / "encoding" / "encoding_dsv4.py"
    tokenizer_path = Path(MODEL_PATH) / "tokenizer.json"
    tokenizer_config_path = Path(MODEL_PATH) / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        raise FileNotFoundError(
            f"DeepSeek V4 tokenizer config is missing: {tokenizer_config_path}"
        )
    if _OFFICIAL_ENCODER_IDENTITY is None:
        raise RuntimeError("official encoder identity was not frozen")
    if tok.source_identity.get("path") != str(tokenizer_path.resolve()):
        raise ValueError("prompt tokenizer path differs from checkpoint tokenizer.json")
    protocol = {
        "encoder_path": _OFFICIAL_ENCODER_IDENTITY["path"],
        "encoder_sha256": _OFFICIAL_ENCODER_IDENTITY["sha256"],
        "tokenizer_path": tok.source_identity["path"],
        "tokenizer_sha256": tok.source_identity["sha256"],
        "tokenizer_config_path": str(tokenizer_config_path.resolve()),
        "tokenizer_config_sha256": (
            "sha256:" + _ih_sha256_file(tokenizer_config_path)
        ),
        "thinking_mode": "chat",
        "reasoning_effort": "low",
        "drop_thinking": True,
        "add_default_bos_token": True,
        "document_delimiter": "\n\n",
    }
    if IH_LONG_OUTPUT_TOKENS:
        protocol.update(
            {
                "answer_style": "direct_answer_plus_document_evidence_v1",
                "target_output_tokens": IH_LONG_OUTPUT_TOKENS,
                "target_output_token_range": [
                    IH_LONG_OUTPUT_TOKENS - 5,
                    IH_LONG_OUTPUT_TOKENS + 5,
                ],
            }
        )
    common_prompt = {
        "offline_chunk_hashes": [
            _ih_chunk_hash(chunk) for chunk in prompt_chunks
        ],
        "offline_prefix_hash": _ih_chunk_hash(
            [token for chunk in prompt_chunks for token in chunk]
        ),
    }
    if len(prompt_cases) == 1:
        case = prompt_cases[0]
        protocol["query_instruction"] = _query_text(queries[0][0])
        payload = {
            "format": _IH_PURE_PROMPT_MANIFEST_FORMAT,
            "prompt_mode": "official_rag_v1",
            "data_selection_sha256": data_selection["selection_sha256"],
            "dataset": dict(data_selection["dataset"]),
            "output_blind": True,
            "protocol": protocol,
            "source": {
                "chunk_hashes": [_ih_chunk_hash(chunk) for chunk in chunks],
                "chunk_order": source_order,
                "expected_source_chunk": case["expected_source_chunk"],
                "query_row_id": case["query_row_id"],
            },
            "geometry": {
                "chunk_tokens": int(chunk_tokens),
                "num_chunks": len(prompt_chunks),
                "query_start": query_start,
                "online_suffix_tokens": case["online_suffix_tokens"],
                "total_tokens": case["total_tokens"],
            },
            "prompt": {
                "text_sha256": case["text_sha256"],
                "full_input_ids_sha256": case["full_input_ids_sha256"],
                **common_prompt,
                "online_suffix_hash": case["online_suffix_hash"],
                "question_sha256": case["question_sha256"],
                "answers_sha256": case["answers_sha256"],
            },
        }
    else:
        protocol["query_instruction_mode"] = (
            "per_case_direct_answer_plus_evidence_v1"
            if IH_LONG_OUTPUT_TOKENS
            else "per_case_short_span_v1"
        )
        payload = {
            "format": _IH_PURE_PROMPT_MULTI_MANIFEST_FORMAT,
            "prompt_mode": "official_rag_v1",
            "data_selection_sha256": data_selection["selection_sha256"],
            "dataset": dict(data_selection["dataset"]),
            "output_blind": True,
            "protocol": protocol,
            "source": {
                "chunk_hashes": [_ih_chunk_hash(chunk) for chunk in chunks],
                "chunk_order": source_order,
                "query_row_ids": list(query_row_ids),
            },
            "geometry": {
                "chunk_tokens": int(chunk_tokens),
                "num_chunks": len(prompt_chunks),
                "num_queries": len(prompt_cases),
                "query_start": query_start,
                "min_total_tokens": min(
                    case["total_tokens"] for case in prompt_cases
                ),
                "max_total_tokens": max(
                    case["total_tokens"] for case in prompt_cases
                ),
            },
            "prompt": common_prompt,
            "cases": prompt_cases,
        }
    prompt_manifest = dict(payload)
    prompt_manifest["prompt_manifest_sha256"] = _ih_canonical_json_sha256(
        payload
    )
    if IH_PROMPT_MANIFEST:
        with Path(IH_PROMPT_MANIFEST).open("r", encoding="utf-8") as handle:
            replay_manifest = json.load(handle)
        replay_payload = dict(replay_manifest)
        replay_digest = replay_payload.pop("prompt_manifest_sha256", None)
        if replay_digest != _ih_canonical_json_sha256(replay_payload):
            raise ValueError("pure prompt replay manifest digest is invalid")
        if replay_manifest != prompt_manifest:
            raise ValueError("pure prompt replay differs from frozen token prompt")
    elif IH_PROMPT_MANIFEST_OUT:
        _ih_write_data_manifest(IH_PROMPT_MANIFEST_OUT, prompt_manifest)
    else:
        raise ValueError(
            "official_rag_v1 requires a prompt manifest input or output path"
        )
    return (
        prompt_chunks,
        rebuilt_queries,
        prompt_manifest,
    )


def _ih_validate_official_prompt_run_identity(
    prompt_manifest: Mapping[str, object], qualification_cap: int
) -> None:
    geometry = prompt_manifest.get("geometry")
    prompt = prompt_manifest.get("prompt")
    if not isinstance(geometry, dict) or not isinstance(prompt, dict):
        raise ValueError("official prompt manifest lacks geometry/token identity")
    if prompt_manifest.get("format") == _IH_PURE_PROMPT_MULTI_MANIFEST_FORMAT:
        cases = prompt_manifest.get("cases")
        if not isinstance(cases, list) or len(cases) != len(
            IH_EXPECTED_QUERY_ROW_IDS
        ):
            raise ValueError("multi-query prompt manifest has invalid cases")
        observed_rows = tuple(case.get("query_row_id") for case in cases)
        if observed_rows != IH_EXPECTED_QUERY_ROW_IDS:
            raise ValueError(
                "multi-query prompt rows differ from their pre-registration"
            )
        if (
            not re.fullmatch(
                r"[0-9a-f]{64}", IH_EXPECTED_PROMPT_MANIFEST_SHA256
            )
            or prompt_manifest.get("prompt_manifest_sha256")
            != IH_EXPECTED_PROMPT_MANIFEST_SHA256
        ):
            raise ValueError(
                "multi-query prompt manifest digest differs from its "
                "pre-registered digest"
            )
        max_total_tokens = max(
            int(case.get("total_tokens", 0)) for case in cases
        )
        if (
            geometry.get("max_total_tokens") != max_total_tokens
            or type(qualification_cap) is not int
            or qualification_cap != max_total_tokens
        ):
            raise ValueError(
                "multi-query qualification cap must equal the largest frozen "
                f"request: required={max_total_tokens} cap={qualification_cap}"
            )
        return
    if (
        prompt.get("text_sha256") != IH_EXPECTED_PROMPT_TEXT_SHA256
        or prompt.get("full_input_ids_sha256")
        != IH_EXPECTED_FULL_INPUT_IDS_SHA256
        or geometry.get("total_tokens") != IH_EXPECTED_FULL_INPUT_TOKENS
    ):
        raise ValueError(
            "official_rag_v1 one-pass prompt differs from its pre-registered "
            "text/token identity"
        )
    if type(qualification_cap) is not int or qualification_cap != geometry.get(
        "total_tokens"
    ):
        raise ValueError(
            "official prompt qualification cap must equal the exact frozen "
            f"request length: required={geometry.get('total_tokens')} "
            f"cap={qualification_cap}"
        )


def _ih_read_data_manifest(path) -> dict:
    """Read either a standalone selection manifest or a benchmark result."""

    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read indexer-hot data manifest {source}: {error}"
        ) from error
    candidates = [document]
    if isinstance(document, dict):
        reproducibility = document.get("reproducibility")
        if isinstance(reproducibility, dict):
            candidates.append(reproducibility.get("source_data_selection"))
            candidates.append(reproducibility.get("indexer_hot_data_selection"))
        candidates.append(document.get("source_data_selection"))
        candidates.append(document.get("indexer_hot_data_selection"))
    manifest = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("format") == _IH_DATA_MANIFEST_FORMAT
        ),
        None,
    )
    if manifest is None:
        raise ValueError(
            f"{source} does not contain {_IH_DATA_MANIFEST_FORMAT}"
        )
    claimed_digest = manifest.get("selection_sha256")
    if not isinstance(claimed_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", claimed_digest
    ):
        raise ValueError(f"{source} has an invalid data-selection digest")
    canonical = dict(manifest)
    canonical.pop("selection_sha256", None)
    observed_digest = _ih_canonical_json_sha256(canonical)
    if observed_digest != claimed_digest:
        raise ValueError(
            f"{source} data-selection digest mismatch: "
            f"expected={claimed_digest} observed={observed_digest}"
        )
    return manifest


def _ih_manifest_row_ids(manifest: dict) -> set[int]:
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("data-selection manifest lacks selection object")
    geometry = manifest.get("geometry")
    chunks = selection.get("chunks")
    queries = selection.get("queries")
    if (
        not isinstance(geometry, dict)
        or not isinstance(chunks, list)
        or not isinstance(queries, list)
        or len(chunks) != geometry.get("num_chunks")
        or len(queries) != geometry.get("num_queries")
    ):
        raise ValueError("data-selection manifest row lists do not match geometry")
    row_ids = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("data-selection chunk must be an object")
        for row in chunk.get("rows", []):
            if not isinstance(row, dict) or type(row.get("row_id")) is not int:
                raise ValueError("data-selection corpus row has an invalid row_id")
            row_ids.add(row["row_id"])
    for query in queries:
        if not isinstance(query, dict) or type(query.get("row_id")) is not int:
            raise ValueError("data-selection query has an invalid row_id")
        row_ids.add(query["row_id"])
    return row_ids


def _ih_dataset_records(dataset_path: Path):
    records = []
    with dataset_path.open("rb") as handle:
        for row_id, raw_line in enumerate(handle):
            row_bytes = raw_line.rstrip(b"\r\n")
            if not row_bytes.strip():
                continue
            try:
                row = json.loads(row_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid JSON in {dataset_path} at zero-based row "
                    f"{row_id}: {error}"
                ) from error
            if row.get("input") and row.get("context") and row.get("answers"):
                records.append(
                    {
                        "row_id": row_id,
                        "row_sha256": hashlib.sha256(row_bytes).hexdigest(),
                        "data": row,
                    }
                )
    return records


def _ih_dataset_identity(dataset_name: str, dataset_path: Path) -> dict:
    return {
        "name": dataset_name,
        "bytes": dataset_path.stat().st_size,
        "sha256": _ih_sha256_file(dataset_path),
        "row_id_base": 0,
    }


def _ih_question_ids(tok, row: dict) -> list[int]:
    return tok(
        "\nQuestion: " + row["input"] + "\nAnswer:",
        add_special_tokens=False,
    )["input_ids"]


def _ih_finalize_data_manifest(payload: dict) -> dict:
    manifest = dict(payload)
    manifest["selection_sha256"] = _ih_canonical_json_sha256(payload)
    return manifest


def _ih_build_data_selection(
    tok,
    records,
    *,
    dataset_identity,
    chunk_tokens,
    num_chunks,
    num_queries,
    row_offset,
    excluded_row_ids,
    excluded_selection_sha256,
):
    eligible = [
        record
        for record in records
        if record["row_id"] >= row_offset
        and record["row_id"] not in excluded_row_ids
    ]
    chunks = []
    chunk_entries = []
    cursor = 0
    while len(chunks) < num_chunks and cursor < len(eligible):
        input_ids = []
        row_entries = []
        while len(input_ids) < chunk_tokens and cursor < len(eligible):
            record = eligible[cursor]
            cursor += 1
            context_ids = tok(
                record["data"]["context"], add_special_tokens=False
            )["input_ids"]
            used_count = min(len(context_ids), chunk_tokens - len(input_ids))
            if used_count <= 0:
                continue
            input_ids.extend(context_ids[:used_count])
            row_entries.append(
                {
                    "row_id": record["row_id"],
                    "row_sha256": record["row_sha256"],
                    "context_token_count": len(context_ids),
                    "context_token_hash": _ih_chunk_hash(context_ids),
                    "used_token_count": used_count,
                }
            )
        if len(input_ids) == chunk_tokens:
            chunk_index = len(chunks)
            chunks.append(input_ids)
            chunk_entries.append(
                {
                    "chunk_index": chunk_index,
                    "token_hash": _ih_chunk_hash(input_ids),
                    "rows": row_entries,
                }
            )
    if len(chunks) != num_chunks:
        raise RuntimeError(
            f"only built {len(chunks)}/{num_chunks} disjoint {chunk_tokens}-token "
            f"chunks at row_offset={row_offset} after excluding "
            f"{len(excluded_row_ids)} seen rows"
        )

    record_by_id = {record["row_id"]: record for record in records}
    # Spread queries over chunks first, then use additional contributing rows.
    # This preserves the old one-query-per-chunk order when possible, but never
    # repeats a row when num_queries exceeds num_chunks.
    query_candidates = []
    max_rows = max(len(chunk["rows"]) for chunk in chunk_entries)
    for source_index in range(max_rows):
        for chunk_index, chunk in enumerate(chunk_entries):
            if source_index < len(chunk["rows"]):
                row_entry = chunk["rows"][source_index]
                # A question is valid only when its complete supporting context
                # is present in the selected chunk.  The final row used to fill
                # a fixed-size chunk can be truncated; selecting that row would
                # silently weaken the dense baseline and the quality gate.
                if row_entry["used_token_count"] != row_entry["context_token_count"]:
                    continue
                query_candidates.append(
                    (record_by_id[row_entry["row_id"]], chunk_index)
                )
    if num_queries > len(query_candidates):
        raise RuntimeError(
            f"requested {num_queries} unique queries, but the selected corpus has "
            f"only {len(query_candidates)} fully-contained contributing JSONL rows; "
            "increase chunks, "
            "decrease chunk_tokens, or reduce REDKNOT_IH_NUM_QUERIES"
        )

    queries = []
    query_entries = []
    for query_index, (record, expected_chunk) in enumerate(
        query_candidates[:num_queries]
    ):
        row = record["data"]
        query_ids = _ih_question_ids(tok, row)
        answers = [str(answer) for answer in row["answers"]]
        queries.append((row["input"], query_ids, answers, expected_chunk))
        query_entries.append(
            {
                "query_index": query_index,
                "row_id": record["row_id"],
                "row_sha256": record["row_sha256"],
                "expected_chunk": expected_chunk,
                "question_sha256": hashlib.sha256(
                    row["input"].encode("utf-8")
                ).hexdigest(),
                "answers_sha256": _ih_canonical_json_sha256(answers),
                "query_token_hash": _ih_chunk_hash(query_ids),
            }
        )

    payload = {
        "format": _IH_DATA_MANIFEST_FORMAT,
        "dataset": dataset_identity,
        "geometry": {
            "chunk_tokens": chunk_tokens,
            "num_chunks": num_chunks,
            "num_queries": num_queries,
        },
        "selection": {
            "mode": "offset",
            "row_offset": row_offset,
            "excluded_row_ids": sorted(excluded_row_ids),
            "excluded_selection_sha256": sorted(excluded_selection_sha256),
            "chunks": chunk_entries,
            "queries": query_entries,
        },
    }
    return chunks, queries, _ih_finalize_data_manifest(payload)


def _ih_replay_data_selection(
    tok,
    records,
    manifest,
    *,
    dataset_identity,
    chunk_tokens,
    num_chunks,
    num_queries,
):
    if manifest.get("dataset") != dataset_identity:
        raise ValueError(
            "data-selection manifest dataset identity differs from the current JSONL"
        )
    expected_geometry = {
        "chunk_tokens": chunk_tokens,
        "num_chunks": num_chunks,
        "num_queries": num_queries,
    }
    if manifest.get("geometry") != expected_geometry:
        raise ValueError(
            "data-selection manifest geometry differs from requested benchmark: "
            f"expected={expected_geometry} actual={manifest.get('geometry')}"
        )
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("data-selection manifest lacks selection object")
    chunk_entries = selection.get("chunks")
    query_entries = selection.get("queries")
    if not isinstance(chunk_entries, list) or len(chunk_entries) != num_chunks:
        raise ValueError("data-selection manifest has an invalid chunk list")
    if not isinstance(query_entries, list) or len(query_entries) != num_queries:
        raise ValueError("data-selection manifest has an invalid query list")
    record_by_id = {record["row_id"]: record for record in records}
    chunks = []
    corpus_rows_by_chunk = []
    corpus_row_entries_by_chunk = []
    seen_corpus_rows = set()
    for chunk_index, chunk_entry in enumerate(chunk_entries):
        if (
            not isinstance(chunk_entry, dict)
            or chunk_entry.get("chunk_index") != chunk_index
            or not isinstance(chunk_entry.get("rows"), list)
            or not chunk_entry["rows"]
        ):
            raise ValueError(f"data-selection chunk {chunk_index} is malformed")
        input_ids = []
        corpus_row_ids = set()
        for row_entry in chunk_entry["rows"]:
            if (
                not isinstance(row_entry, dict)
                or type(row_entry.get("row_id")) is not int
            ):
                raise ValueError(
                    f"data-selection chunk {chunk_index} has malformed row"
                )
            row_id = row_entry["row_id"]
            if row_id in seen_corpus_rows:
                raise ValueError(f"data-selection corpus repeats JSONL row {row_id}")
            seen_corpus_rows.add(row_id)
            corpus_row_ids.add(row_id)
            record = record_by_id.get(row_id)
            if record is None or record["row_sha256"] != row_entry.get("row_sha256"):
                raise ValueError(
                    f"data-selection row hash mismatch for JSONL row {row_id}"
                )
            context_ids = tok(
                record["data"]["context"], add_special_tokens=False
            )["input_ids"]
            if (
                len(context_ids) != row_entry.get("context_token_count")
                or _ih_chunk_hash(context_ids) != row_entry.get("context_token_hash")
            ):
                raise ValueError(
                    f"data-selection context token mismatch for JSONL row {row_id}"
                )
            used_count = row_entry.get("used_token_count")
            if (
                type(used_count) is not int
                or used_count <= 0
                or used_count > len(context_ids)
                or len(input_ids) + used_count > chunk_tokens
            ):
                raise ValueError(
                    f"data-selection used-token count is invalid for JSONL row {row_id}"
                )
            input_ids.extend(context_ids[:used_count])
        if (
            len(input_ids) != chunk_tokens
            or _ih_chunk_hash(input_ids) != chunk_entry.get("token_hash")
        ):
            raise ValueError(
                f"data-selection token hash mismatch for chunk {chunk_index}"
            )
        chunks.append(input_ids)
        corpus_rows_by_chunk.append(corpus_row_ids)
        corpus_row_entries_by_chunk.append(
            {row_entry["row_id"]: row_entry for row_entry in chunk_entry["rows"]}
        )

    queries = []
    seen_query_rows = set()
    for query_index, query_entry in enumerate(query_entries):
        if (
            not isinstance(query_entry, dict)
            or query_entry.get("query_index") != query_index
            or type(query_entry.get("row_id")) is not int
            or type(query_entry.get("expected_chunk")) is not int
        ):
            raise ValueError(f"data-selection query {query_index} is malformed")
        row_id = query_entry["row_id"]
        expected_chunk = query_entry["expected_chunk"]
        if row_id in seen_query_rows:
            raise ValueError(f"data-selection queries repeat JSONL row {row_id}")
        seen_query_rows.add(row_id)
        if not 0 <= expected_chunk < num_chunks or row_id not in corpus_rows_by_chunk[
            expected_chunk
        ]:
            raise ValueError(
                f"data-selection query row {row_id} is not in expected "
                f"chunk {expected_chunk}"
            )
        corpus_row_entry = corpus_row_entries_by_chunk[expected_chunk][row_id]
        if corpus_row_entry.get("used_token_count") != corpus_row_entry.get(
            "context_token_count"
        ):
            raise ValueError(
                f"data-selection query row {row_id} is not fully contained in "
                f"chunk {expected_chunk}"
            )
        record = record_by_id.get(row_id)
        if record is None or record["row_sha256"] != query_entry.get("row_sha256"):
            raise ValueError(
                f"data-selection query row hash mismatch for JSONL row {row_id}"
            )
        row = record["data"]
        query_ids = _ih_question_ids(tok, row)
        answers = [str(answer) for answer in row["answers"]]
        if (
            hashlib.sha256(row["input"].encode("utf-8")).hexdigest()
            != query_entry.get("question_sha256")
            or _ih_canonical_json_sha256(answers)
            != query_entry.get("answers_sha256")
            or _ih_chunk_hash(query_ids) != query_entry.get("query_token_hash")
        ):
            raise ValueError(
                f"data-selection query payload mismatch for JSONL row {row_id}"
            )
        queries.append((row["input"], query_ids, answers, expected_chunk))
    return chunks, queries, manifest


def _ih_write_data_manifest(path, manifest):
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # link() is an atomic, no-overwrite publication in the same directory.
        # A killed writer can leave only the hidden temporary file, never a
        # partially-written manifest at the requested replay path.
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _ih_load(
    tok,
    chunk_tokens,
    num_chunks,
    num_queries,
    *,
    row_offset=None,
    manifest_path=None,
    exclude_manifest_paths=None,
    return_manifest=False,
    dataset_name=None,
    dataset_dir=None,
):
    """Load unique corpus/query rows, optionally replaying a frozen selection.

    The two-value return remains the default for callers of the historical
    helper.  ``return_manifest=True`` adds the exact, digest-bound row/token
    selection used by the benchmark.
    """

    row_offset = IH_DATA_ROW_OFFSET if row_offset is None else int(row_offset)
    manifest_path = IH_DATA_MANIFEST if manifest_path is None else manifest_path
    exclude_manifest_paths = (
        IH_DATA_EXCLUDE_MANIFESTS
        if exclude_manifest_paths is None
        else tuple(exclude_manifest_paths)
    )
    if row_offset < 0:
        raise ValueError("indexer-hot data row offset must be non-negative")
    if chunk_tokens <= 0 or num_chunks <= 0 or num_queries <= 0:
        raise ValueError(
            "indexer-hot chunk_tokens, num_chunks and num_queries must be positive"
        )
    if manifest_path and (row_offset != 0 or exclude_manifest_paths):
        raise ValueError(
            "REDKNOT_IH_DATA_MANIFEST is an exact replay and cannot be combined "
            "with a nonzero data offset or exclusion manifests"
        )
    dataset_name = dataset_name or (DATASETS[0] if DATASETS else "hotpotqa")
    dataset_root = Path(dataset_dir or LONGBENCH_DIR).expanduser().resolve()
    dataset_path = dataset_root / f"{dataset_name}.jsonl"
    if not dataset_path.is_file():
        raise ValueError(f"benchmark dataset is missing: {dataset_path}")
    records = _ih_dataset_records(dataset_path)
    dataset_identity = _ih_dataset_identity(dataset_name, dataset_path)
    if manifest_path:
        loaded_manifest = _ih_read_data_manifest(manifest_path)
        result = _ih_replay_data_selection(
            tok,
            records,
            loaded_manifest,
            dataset_identity=dataset_identity,
            chunk_tokens=chunk_tokens,
            num_chunks=num_chunks,
            num_queries=num_queries,
        )
    else:
        excluded_row_ids = set()
        excluded_selection_sha256 = []
        for excluded_path in exclude_manifest_paths:
            excluded_manifest = _ih_read_data_manifest(excluded_path)
            if excluded_manifest.get("dataset") != dataset_identity:
                raise ValueError(
                    f"exclusion manifest {excluded_path} belongs to a different dataset"
                )
            excluded_row_ids.update(_ih_manifest_row_ids(excluded_manifest))
            excluded_selection_sha256.append(
                excluded_manifest["selection_sha256"]
            )
        result = _ih_build_data_selection(
            tok,
            records,
            dataset_identity=dataset_identity,
            chunk_tokens=chunk_tokens,
            num_chunks=num_chunks,
            num_queries=num_queries,
            row_offset=row_offset,
            excluded_row_ids=excluded_row_ids,
            excluded_selection_sha256=excluded_selection_sha256,
        )
    return result if return_manifest else result[:2]


def _ih_wait_ready(
    proc,
    timeout_s=int(os.environ.get("REDKNOT_IH_SERVER_READY_TIMEOUT_S", "900")),
):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("RedKnot server process exited before ready")
        try:
            response = _ih_local_request("GET", "/get_model_info", timeout=5)
            response.raise_for_status()
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    "launcher exited while another process answered the readiness probe"
                )
            return True
        except Exception:
            pass
        try:
            # /health may 404 but a 200 on generate readiness is enough
            response = _ih_local_request("POST", "/flush_cache", timeout=5)
            response.raise_for_status()
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    "launcher exited while another process answered the readiness probe"
                )
            return True
        except Exception:
            time.sleep(5)
    raise TimeoutError("RedKnot server did not become ready in time")


def _ih_assert_gpu_capacity(env: dict) -> None:
    """Fail before launch when the requested TP devices cannot fit the model."""

    if IH_MIN_GPU_FREE_MIB <= 0:
        return
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"cannot verify GPU capacity before DeepSeek launch: {error}"
        ) from error
    gpu_rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        gpu_rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "free_mib": int(fields[2]),
            }
        )
    visible = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible:
        selected = []
        by_index = {str(row["index"]): row for row in gpu_rows}
        by_uuid = {row["uuid"]: row for row in gpu_rows}
        visible_tokens = [item.strip() for item in visible.split(",")]
        if len(set(visible_tokens)) != len(visible_tokens):
            raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate GPUs")
        for token in visible_tokens:
            row = by_index.get(token) or by_uuid.get(token)
            if row is None:
                raise RuntimeError(
                    f"cannot resolve CUDA_VISIBLE_DEVICES entry {token!r}"
                )
            selected.append(row)
    else:
        selected = gpu_rows[:IH_TP_SIZE]
    if len(selected) < IH_TP_SIZE:
        raise RuntimeError(
            f"DeepSeek TP={IH_TP_SIZE} needs at least {IH_TP_SIZE} visible GPUs, "
            f"got {len(selected)}"
        )
    selected = selected[:IH_TP_SIZE]
    if len({row["uuid"] for row in selected}) != len(selected):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES resolves multiple entries to the same GPU"
        )
    low = [
        (row["index"], row["free_mib"])
        for row in selected
        if row["free_mib"] < IH_MIN_GPU_FREE_MIB
    ]
    if low:
        raise RuntimeError(
            "DeepSeek launch refused because target GPUs are already occupied: "
            f"free_mib={low}, required_per_gpu={IH_MIN_GPU_FREE_MIB}; stop the "
            "owning jobs or lower REDKNOT_IH_MIN_GPU_FREE_MIB explicitly"
        )


def _ih_process_group_alive(pgid: int) -> bool:
    """Return whether a process group has any non-zombie member."""

    pgid = int(pgid)
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            # After the final ') ': state, ppid, pgrp, ...
            fields = (proc_dir / "stat").read_text().rsplit(") ", 1)[1].split()
            state = fields[0]
            process_group = int(fields[2])
        except (IndexError, OSError, ValueError):
            continue
        if process_group == pgid and state != "Z":
            return True
    return False


_IH_SERVER_IDENTITY_SCHEMA = "redknot_pro0813_owned_server_identity_v1"
_IH_SERVER_IDENTITY_KEYS = frozenset(
    {
        "schema",
        "nonce",
        "pid",
        "pgid",
        "sid",
        "starttime_ticks",
        "launcher",
        "model_path",
        "port",
    }
)


def _ih_read_proc_cmdline(pid: int) -> tuple[str, ...]:
    raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    argv = tuple(
        item.decode("utf-8", errors="strict")
        for item in raw.rstrip(b"\0").split(b"\0")
        if item
    )
    if not argv:
        raise ValueError(f"/proc/{int(pid)}/cmdline is empty")
    return argv


def _ih_is_owned_pro_server_cmdline(
    argv: Sequence[str], executable: str
) -> bool:
    """Recognize only the certified launcher before or after its final exec."""

    argv = tuple(str(item) for item in argv)
    if len(argv) == 2 and Path(argv[0]).name == "bash":
        try:
            launcher = Path(argv[1]).expanduser().resolve()
            executable_path = Path(executable).expanduser().resolve()
            bash_path = Path("/usr/bin/bash").resolve()
        except OSError:
            return False
        return launcher == _PRO0813_SERVER_SCRIPT and executable_path == bash_path
    try:
        executable_path = Path(executable).expanduser().resolve()
        runtime_path = Path(IH_VENV_PY).expanduser().resolve()
    except OSError:
        return False
    if executable_path != runtime_path or len(argv) < 3:
        return False

    def has_exact_pair(flag: str, value: str) -> bool:
        values = tuple(
            argv[index + 1]
            for index in range(len(argv) - 1)
            if argv[index] == flag
        )
        return values == (value,) and not any(
            item.startswith(f"{flag}=") for item in argv
        )

    return (
        has_exact_pair("-m", "sglang.launch_server")
        and has_exact_pair("--model-path", MODEL_PATH)
        and has_exact_pair("--attention-backend", "redknot_mla")
        and has_exact_pair("--tp-size", str(IH_TP_SIZE))
        and has_exact_pair("--port", str(IH_PORT))
    )


def _ih_validate_server_identity_payload(
    payload: object, *, expected_nonce: str
) -> dict:
    """Validate the immutable, supervisor-bound part of a launch identity."""

    if not isinstance(payload, dict) or set(payload) != _IH_SERVER_IDENTITY_KEYS:
        raise ValueError("owned-server identity has an unexpected JSON schema")
    expected_values = {
        "schema": _IH_SERVER_IDENTITY_SCHEMA,
        "nonce": expected_nonce,
        "launcher": str(_PRO0813_SERVER_SCRIPT),
        "model_path": MODEL_PATH,
        "port": IH_PORT,
    }
    for key, expected in expected_values.items():
        if type(payload.get(key)) is not type(expected) or payload.get(key) != expected:
            raise ValueError(f"owned-server identity has invalid {key}")
    for key in ("pid", "pgid", "sid", "starttime_ticks"):
        if type(payload.get(key)) is not int or int(payload[key]) <= 0:
            raise ValueError(f"owned-server identity has invalid {key}")
    if not (
        payload["pid"] == payload["pgid"] == payload["sid"]
        and payload["pid"] > 1
    ):
        raise ValueError("owned-server identity is not an independent session")
    return dict(payload)


def _ih_read_server_identity_file(path: Path, *, expected_nonce: str) -> dict:
    """Read a small owner-only regular file without following a symlink."""

    path = Path(path)
    if not path.is_absolute():
        raise ValueError("owned-server identity path must be absolute")
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"owned-server identity is absent: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("owned-server identity must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot safely open owned-server identity: {error}") from error
    try:
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or after.st_nlink != 1
            or after.st_uid != os.geteuid()
            or stat.S_IMODE(after.st_mode) != 0o600
            or after.st_size <= 0
            or after.st_size > 4096
        ):
            raise ValueError("owned-server identity file metadata is unsafe")
        raw = b""
        while len(raw) <= 4096:
            chunk = os.read(descriptor, min(4097 - len(raw), 4096))
            if not chunk:
                break
            raw += chunk
        if len(raw) != after.st_size or len(raw) > 4096:
            raise ValueError("owned-server identity changed while being read")
    finally:
        os.close(descriptor)
    def no_duplicate_keys(pairs):
        parsed = {}
        for key, value in pairs:
            if key in parsed:
                raise ValueError(f"duplicate JSON key {key!r}")
            parsed[key] = value
        return parsed

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"owned-server identity is malformed: {error}") from error
    return _ih_validate_server_identity_payload(
        payload, expected_nonce=expected_nonce
    )


def _ih_publish_server_identity_file(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish without ever replacing a stale or attacker file."""

    path = Path(path)
    if not path.is_absolute():
        raise ValueError("owned-server identity path must be absolute")
    parent_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_DIRECTORY", 0
    ) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(path.parent, parent_flags)
    temp_name = f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(16)}"
    temp_fd = None
    temp_exists = False
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("refusing a stale owned-server identity path")
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        temp_exists = True
        encoded = (
            json.dumps(
                dict(payload), sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(encoded):
            offset += os.write(temp_fd, encoded[offset:])
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = None
        # link(2) is the portable no-replace publication primitive: unlike
        # os.replace, it fails if any stale regular file, directory or symlink
        # appeared after the lstat above.
        os.link(
            temp_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.unlink(temp_name, dir_fd=parent_fd)
        temp_exists = False
        os.fsync(parent_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_exists:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _ih_publish_owned_server_identity(proc):
    """Publish the exact Popen leader identity before doing any other work."""

    if IH_SERVER_IDENTITY_PATH is None:
        return None
    pid = int(proc.pid)
    state, starttime_ticks = _ih_linux_process_identity(pid)
    pgid = os.getpgid(pid)
    sid = os.getsid(pid)
    if state in {"Z", "X", "x"} or not (pid == pgid == sid):
        raise RuntimeError(
            "owned Pro server did not enter its independent session before publication"
        )
    argv = _ih_read_proc_cmdline(pid)
    executable = os.path.realpath(f"/proc/{pid}/exe")
    if not _ih_is_owned_pro_server_cmdline(argv, executable):
        raise RuntimeError("Popen child is not the certified Pro-0813 server")
    payload = {
        "schema": _IH_SERVER_IDENTITY_SCHEMA,
        "nonce": IH_SERVER_IDENTITY_NONCE,
        "pid": pid,
        "pgid": pgid,
        "sid": sid,
        "starttime_ticks": starttime_ticks,
        "launcher": str(_PRO0813_SERVER_SCRIPT),
        "model_path": MODEL_PATH,
        "port": IH_PORT,
    }
    _ih_validate_server_identity_payload(
        payload, expected_nonce=IH_SERVER_IDENTITY_NONCE
    )
    _ih_publish_server_identity_file(IH_SERVER_IDENTITY_PATH, payload)
    proc._redknot_server_identity = dict(payload)
    proc._redknot_server_identity_path = str(IH_SERVER_IDENTITY_PATH)
    return dict(payload)


def _ih_stop_launched_server(proc, *, timeout: float = 30.0) -> None:
    """Stop only the independent process group created by this benchmark."""

    if proc is None:
        return
    pgid = int(getattr(proc, "_redknot_server_pgid", proc.pid))
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + float(timeout)
    while _ih_process_group_alive(pgid) and time.monotonic() < deadline:
        proc.poll()
        time.sleep(0.25)
    if _ih_process_group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + 10.0
        while _ih_process_group_alive(pgid) and time.monotonic() < kill_deadline:
            time.sleep(0.1)
    try:
        proc.wait(timeout=max(1.0, float(timeout)))
    except subprocess.TimeoutExpired:
        pass
    if _ih_process_group_alive(pgid):
        raise RuntimeError(
            f"RedKnot server PGID {pgid} still has live members after SIGKILL"
        )


def _ih_launch_server():
    """Launch the isolated pure MLA offline/online head-split server."""
    import socket
    env = dict(os.environ)
    worker_ready_dir = env.get(
        "REDKNOT_GPU_HOLDER_WORKER_READY_DIR", ""
    ).strip()
    worker_go_file = env.get(
        "REDKNOT_GPU_HOLDER_WORKER_GO_FILE", ""
    ).strip()
    worker_count_raw = env.get(
        "REDKNOT_GPU_HOLDER_EXPECTED_WORKERS", ""
    ).strip()
    worker_barrier = (
        bool(worker_ready_dir), bool(worker_go_file), bool(worker_count_raw)
    )
    if any(worker_barrier):
        if not all(worker_barrier):
            raise ValueError(
                "scheduler GPU-holder barrier requires ready dir, go file, "
                "and expected worker count"
            )
        if not os.path.isabs(worker_ready_dir) or not os.path.isabs(
            worker_go_file
        ):
            raise ValueError("scheduler GPU-holder barrier paths must be absolute")
        if int(worker_count_raw) != IH_TP_SIZE:
            raise ValueError(
                "scheduler GPU-holder barrier count differs from benchmark TP"
            )
        if IH_NO_LAUNCH:
            raise ValueError(
                "scheduler GPU-holder barrier is invalid for an external server"
            )
        print(
            "[pure-mla] deferring GPU capacity check to the authenticated "
            "scheduler pre-CUDA release barrier"
        )
    else:
        _ih_assert_gpu_capacity(env)

    # A readiness response alone does not prove that it came from the process
    # launched below. Refuse an occupied port up front; otherwise an old SGLang
    # instance can make the benchmark silently test stale code/config while the
    # new launcher is still loading or about to fail its bind.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", IH_PORT)) == 0:
            raise RuntimeError(
                f"benchmark launch port {IH_PORT} is already occupied; stop the "
                "existing service or explicitly use REDKNOT_IH_NO_LAUNCH=1"
            )

    env["PATH"] = f"{os.path.dirname(IH_VENV_PY)}:{env.get('PATH', '')}"
    env["REDKNOT_V4_MODE"] = "aggressive"
    env["REDKNOT_V4_INDEXER_KV_REUSE"] = (
        "1"
        if IH_COMBINED_HEADSPLIT_ROW_SPARSE or not IH_MLA_OFFLOAD
        else "0"
    )
    # The request-plan transport remains enabled; all legacy KV/row actions in
    # that plan are explicitly false and independently rejected by the backend.
    env["SGLANG_REDKNOT_OFFLINE_REUSE"] = "1"
    env["REDKNOT_MLA_OFFLOAD"] = "1" if IH_MLA_OFFLOAD else "0"
    env["REDKNOT_MLA_OFF_EXECUTION_PROFILE"] = (
        IH_MLA_OFF_EXECUTION_PROFILE
    )
    env["REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL"] = IH_MLA_OFF_GLOBAL_ATTN_IMPL
    env["REDKNOT_MLA_OFF_GEOMETRY_TEMPLATE_CACHE"] = (
        "1" if IH_MLA_OFF_GEOMETRY_TEMPLATE_CACHE else "0"
    )
    env["REDKNOT_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS"] = str(
        IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS
    )
    env["REDKNOT_MLA_OFF_COMPACT_WOA"] = (
        "1" if IH_MLA_OFF_COMPACT_WOA else "0"
    )
    env["REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE"] = (
        "1" if IH_REUSE_HEADS_FULL_SCOPE else "0"
    )
    env["REDKNOT_MLA_OFF_MAX_BYTES"] = str(IH_MLA_OFF_MAX_BYTES)
    env["REDKNOT_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS"] = str(
        IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS
    )
    env["REDKNOT_MLA_OFF_QUALIFICATION_ONLY"] = (
        "1" if IH_MLA_OFF_QUALIFICATION_ONLY else "0"
    )
    env["REDKNOT_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS"] = str(
        IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS
    )
    env["REDKNOT_MLA_OFF_METRICS"] = "1" if IH_MLA_OFFLOAD else "0"
    env["REDKNOT_TP_SIZE"] = str(IH_TP_SIZE)
    env["REDKNOT_MLA_DENSE_PREFIX_LAYERS"] = str(MLA_DENSE_PREFIX)
    env["REDKNOT_MLA_DENSE_SUFFIX_LAYERS"] = str(MLA_DENSE_SUFFIX)
    env["REDKNOT_MLA_LOCAL_WINDOW"] = str(MLA_LOCAL_WINDOW)
    env["REDKNOT_MLA_GLOBAL_HEAD_STRIDE"] = str(MLA_GLOBAL_HEAD_STRIDE)
    env["REDKNOT_MLA_GLOBAL_LAYER_STRIDE"] = str(MLA_GLOBAL_LAYER_STRIDE)
    env["REDKNOT_V4_TIMING"] = os.environ.get("REDKNOT_V4_TIMING", "0")
    env["SGLANG_OPT_USE_COMPRESSOR_V2"] = "0"
    env["REDKNOT_MLA_PREFIX_MATERIALIZATION"] = (
        "1" if IH_PREFIX_MATERIALIZATION else "0"
    )
    env["REDKNOT_DISABLE_RADIX_CACHE"] = (
        "0" if IH_PREFIX_MATERIALIZATION else "1"
    )
    env["REDKNOT_RADIX_EVICTION_POLICY"] = IH_RADIX_EVICTION_POLICY
    env["SGLANG_RANK_LOG_DIR"] = IH_RANK_LOG_DIR
    env["CHUNKED_PREFILL_SIZE"] = str(IH_CHUNK_TOKENS)
    env["MAX_PREFILL_TOKENS"] = str(max(IH_CHUNK_TOKENS, IH_MERGED_PREFILL_TOKENS))
    if IH_MERGED_PREFILL_TOKENS > 0:
        env["REDKNOT_V4_MERGED_PREFILL_MAX_TOKENS"] = str(IH_MERGED_PREFILL_TOKENS)
    else:
        env.pop("REDKNOT_V4_MERGED_PREFILL_MAX_TOKENS", None)
    env["PORT"] = str(IH_PORT)
    env["REDKNOT_MODEL_PATH"] = MODEL_PATH
    if HEAD_CFG_PATH:
        env["REDKNOT_HEAD_CFG"] = HEAD_CFG_PATH
    else:
        env.pop("REDKNOT_HEAD_CFG", None)
    if IH_MLA_OFFLOAD:
        env["REDKNOT_SERVER_POLICY_MANIFEST_OUT"] = str(
            Path(IH_RANK_LOG_DIR) / "server_policy_manifest.json"
        )
        env["REDKNOT_SERVER_INSTANCE_NONCE"] = IH_SERVER_INSTANCE_NONCE
    else:
        env.pop("REDKNOT_SERVER_POLICY_MANIFEST_OUT", None)
        env.pop("REDKNOT_SERVER_INSTANCE_NONCE", None)
    launch_label = "pure-mla" if IH_MLA_OFFLOAD else "indexer-hot"
    print(f"[{launch_label}] launching server: {IH_SERVER_SCRIPT} (port {IH_PORT})")
    os.makedirs(IH_RANK_LOG_DIR, exist_ok=True)
    log_handle = open(IH_SERVER_LOG, "w")
    proc = None
    try:
        proc = subprocess.Popen(
            ["bash", IH_SERVER_SCRIPT],
            env=env,
            start_new_session=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        # start_new_session=True makes the child PID the stable PGID. Preserve
        # it now: querying os.getpgid(proc.pid) after the leader exits can fail
        # while worker processes in the original group are still alive.
        proc._redknot_server_pgid = proc.pid
        _ih_publish_owned_server_identity(proc)
    except BaseException:
        # The caller cannot own `proc` until this function returns. Close the
        # small signal/exception window after Popen by draining it locally.
        if proc is not None:
            _ih_stop_launched_server(proc)
        raise
    finally:
        log_handle.close()
    return proc


def _ih_percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[rank]


def _ih_sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _ih_qualification_profile_provenance(target_tokens: int) -> dict:
    """Validate the immutable target-profile identity before GPU work."""

    target_tokens = int(target_tokens)
    if target_tokens not in {65536, 131072, 262144, 450560, 524288}:
        raise ValueError("qualification provenance has an unsupported target")
    path = IH_QUALIFICATION_PROFILE_PATH
    digest = IH_QUALIFICATION_PROFILE_SHA256
    if not path or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(
            "formal Pro-0813 execution requires a target-profile path and "
            "lowercase SHA-256 provenance"
        )
    builtin_sentinel = f"builtin:pro0813:{target_tokens}"
    if target_tokens in {450560, 524288}:
        if path.startswith("builtin:") or not Path(path).is_absolute():
            raise ValueError(
                "formal 440K/512K requires an absolute verified qualification "
                "profile path and SHA-256"
            )
        profile_path = Path(path)
        if not profile_path.is_file() or _ih_sha256_file(profile_path) != digest:
            raise ValueError(
                "qualification profile bytes differ from the verified SHA-256"
            )
        identity_kind = "verified_qualification_profile"
    elif path == builtin_sentinel:
        http_entry_path = Path(__file__).with_name(
            "benchmark_dsv4_pro0813_redknot_http.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_redknot_pro0813_builtin_profile_identity", http_entry_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"cannot load built-in target profile contract: {http_entry_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = type(
            "_BuiltinTargetArgs",
            (),
            {"target_tokens": target_tokens, "qualification_profile": ""},
        )()
        expected_profile = module._resolve_redknot_target_profile(args)
        if (
            expected_profile.get("qualification_profile_path") != path
            or expected_profile.get("qualification_profile_sha256") != digest
        ):
            raise ValueError(
                "built-in target profile SHA-256 differs from the HTTP entry contract"
            )
        identity_kind = "builtin_target_profile"
    elif IH_STRICT_PERFORMANCE_CLAIMS:
        raise ValueError(
            "formal 64K/128K/256K requires its built-in target profile identity"
        )
    elif not Path(path).is_absolute():
        raise ValueError(
            "diagnostic qualification profile provenance must use an absolute path"
        )
    else:
        profile_path = Path(path)
        if not profile_path.is_file() or _ih_sha256_file(profile_path) != digest:
            raise ValueError(
                "diagnostic profile bytes differ from their SHA-256 provenance"
            )
        identity_kind = "diagnostic_external_profile"
    return {
        "schema": "redknot_pro0813_qualification_provenance_v1",
        "target_tokens": target_tokens,
        "identity_kind": identity_kind,
        "profile_path": path,
        "profile_sha256": digest,
    }


def _ih_reproducibility_manifest(
    dataset_name: str,
    data_selection=None,
    prompt_manifest=None,
    qualification_provenance=None,
) -> dict:
    """Record the exact uncommitted code, model index, and input dataset."""

    source_paths = {
        "benchmark": Path(__file__).resolve(),
        "mla_offload": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_mla_offload.py",
        "context_identity": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_context_identity.py",
        "sparse_q_control": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_sparse_q.py",
        "sparse_q_runtime": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_sparse_q_runtime.py",
        "ragged_reuse_batch": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_reuse_batch.py",
        "composite_commit": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_composite_commit.py",
        "reuse_backend_runtime": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_reuse_backend_runtime.py",
        "shared_latent_cpu": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_shared_latent_cache.py",
        "shared_latent_gpu": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_shared_latent_gpu.py",
        "shared_latent_sglang": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_shared_latent_sglang.py",
        "shared_latent_batch_kernels": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_shared_latent_batch_kernels.py",
        "shared_latent_batch_oracle": REPO
        / "python/sglang/srt/layers/attention/redknot/probe_dsv4_shared_latent_batch_kernels.py",
        "shared_snapshot_runtime": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_shared_snapshot_runtime.py",
        "shared_snapshot_sglang": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_shared_snapshot_sglang.py",
        "fused_z_merge": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_fused_z_merge.py",
        "cuda_event_timing": REPO
        / "python/sglang/srt/layers/attention/redknot/dsv4_timing.py",
        "dirty_compressor": REPO
        / "python/sglang/srt/layers/attention/dsv4/compressor.py",
        "mla_backend": REPO
        / "python/sglang/srt/layers/attention/redknot_mla_backend.py",
        "deepseek_v4_model": REPO / "python/sglang/srt/models/deepseek_v4.py",
        "progressive_topk": REPO
        / "python/sglang/srt/layers/moe/redknot_progressive_topk.py",
        "reuse_planner": REPO
        / "python/sglang/srt/layers/attention/redknot/v4/reuse_planner.py",
        "server_args": REPO / "python/sglang/srt/server_args.py",
        "server_launcher": REPO / "server/start_server_redknot_pro0813.sh",
    }
    optional_source_paths = {
        "http_entry": REPO
        / "test/srt/redknot/utils/benchmark_dsv4_pro0813_redknot_http.py",
        "qualification_profile_verifier": REPO
        / "test/srt/redknot/utils/verify_pro0813_qualification_profile.py",
        "one_target_wrapper": REPO
        / "test/srt/redknot/utils/run_deepseek_v4_pro0813_reproduction.sh",
        "combined_supervisor": REPO
        / "test/srt/redknot/utils/run_pro0813_combined_supervisor.sh",
        "all_target_sequencer": REPO
        / "test/srt/redknot/utils/run_deepseek_v4_pro0813_all_targets.sh",
    }
    source_files = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _ih_sha256_file(path),
        }
        for name, path in source_paths.items()
    }
    for name, path in optional_source_paths.items():
        if path.is_file():
            source_files[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _ih_sha256_file(path),
            }
    model_root = Path(MODEL_PATH)
    model_index_path = model_root / "model.safetensors.index.json"
    if not model_index_path.is_file():
        raise ValueError(
            f"reproducibility manifest needs {model_index_path}"
        )
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(model_index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError("model index contains no safetensors shards")
    shard_files = []
    for shard_name in shard_names:
        shard_path = model_root / shard_name
        if not shard_path.is_file():
            raise ValueError(f"model shard is missing: {shard_path}")
        shard_files.append(
            {"name": shard_name, "bytes": shard_path.stat().st_size}
        )
    dataset_path = Path(LONGBENCH_DIR) / f"{dataset_name}.jsonl"
    if not dataset_path.is_file():
        raise ValueError(f"benchmark dataset is missing: {dataset_path}")
    small_model_files = {}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        path = model_root / name
        if path.is_file():
            small_model_files[name] = {
                "bytes": path.stat().st_size,
                "sha256": _ih_sha256_file(path),
            }
    manifest = {
        "source_files": source_files,
        "model_path": str(model_root),
        "model_files": small_model_files,
        "model_shards": shard_files,
        "model_shard_count": len(shard_files),
        "model_shard_bytes": sum(item["bytes"] for item in shard_files),
        "dataset": {
            "name": dataset_name,
            "path": str(dataset_path.resolve()),
            "bytes": dataset_path.stat().st_size,
            "sha256": _ih_sha256_file(dataset_path),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch_version": str(torch.__version__),
            "torch_path": str(Path(torch.__file__).resolve()),
            "torch_cuda_version": str(torch.version.cuda),
        },
    }
    if not IH_MLA_OFFLOAD:
        manifest["runtime"]["mla_off_compact_woa"] = IH_MLA_OFF_COMPACT_WOA
    if data_selection is not None:
        manifest["source_data_selection"] = data_selection
    if prompt_manifest is not None:
        manifest["pure_mla_prompt"] = prompt_manifest
    if qualification_provenance is not None:
        manifest["qualification_provenance"] = dict(
            qualification_provenance
        )
    return manifest


_IH_MLA_OFF_TRANSFER_AUDIT_SCHEMA = "redknot_mla_off_controller_stats_v1"
_IH_MLA_OFF_TRANSFER_BYTE_SEMANTICS = "logical_cpu_source_payload_v1"
_IH_MLA_OFF_COMPOSITE_RECEIPT_SCHEMA = (
    "redknot_mla_off_composite_execution_receipt_v1"
)
_IH_SHARED_SNAPSHOT_AUDIT_SCHEMA = "redknot_shared_snapshot_publication_v1"
_IH_SHARED_RESTORE_AUDIT_SCHEMA = "redknot_shared_device_restore_stats_v1"
_IH_SHARED_RESTORE_COUNTER_FIELDS = (
    "shared_device_restore_calls",
    "shared_device_restore_operations",
    "shared_device_values_restored",
)
_IH_MLA_OFF_COMPACT_WOA_AUDIT_SCHEMA = "redknot_mla_off_compact_woa_v1"
_IH_MLA_OFF_COMPACT_WOA_MEASUREMENT_SEMANTICS = (
    "successful_indexed_inverse_rope_wo_a_row_geometry_v1"
)
_IH_MLA_OFF_COMPACT_WOA_CLAIM_SCOPE = (
    "activation_evidence_not_flops_or_energy_v1"
)
_IH_MLA_OFF_TRANSFER_COUNTER_FIELDS = (
    "device_restore_calls",
    "device_rows_restored",
    "rows_restored",
    "online_artifact_h2d_calls",
    "online_artifact_h2d_bytes",
    "online_device_gather_index_h2d_calls",
    "online_device_gather_index_h2d_rows",
    "online_device_gather_index_h2d_bytes",
    "online_device_scatter_index_h2d_calls",
    "online_device_scatter_index_h2d_rows",
    "online_device_scatter_index_h2d_bytes",
    "online_dirty_index_h2d_calls",
    "online_dirty_index_h2d_rows",
    "online_dirty_index_h2d_bytes",
    "snapshot_device_index_h2d_calls",
    "snapshot_device_index_h2d_rows",
    "snapshot_device_index_h2d_bytes",
    "online_index_h2d_bytes",
    "online_total_h2d_bytes",
)
_IH_LEGACY_DEVICE_RESTORE_COUNTER_FIELDS = (
    "device_restore_calls",
    "device_rows_restored",
    "online_device_gather_index_h2d_calls",
    "online_device_gather_index_h2d_rows",
    "online_device_gather_index_h2d_bytes",
    "online_device_scatter_index_h2d_calls",
    "online_device_scatter_index_h2d_rows",
    "online_device_scatter_index_h2d_bytes",
)
_IH_MLA_OFF_TRANSFER_GAUGE_FIELDS = (
    "device_cache_enabled",
    "reserved_device_bytes",
    "allocated_device_bytes",
    "max_device_cache_bytes",
)


def _ih_runtime_log_paths():
    # Rank 0 is the authoritative source for model-runner metrics. Reading both
    # it and captured server stdout can double-count the same logging record and
    # make runtime coverage look artificially complete on other launch setups.
    rank0_log = os.path.join(IH_RANK_LOG_DIR, "rank0.log")
    if os.path.isfile(rank0_log):
        return [rank0_log]
    if IH_MLA_OFFLOAD:
        raise RuntimeError(
            f"MLA-off runtime evidence requires the rank-0 log: {rank0_log}"
        )
    return [IH_SERVER_LOG]


def _ih_transfer_audit_log_paths(tp_size=None):
    """Return every expected TP scheduler log, including absent paths."""

    tp_size = IH_TP_SIZE if tp_size is None else int(tp_size)
    if tp_size <= 0:
        raise ValueError("MLA-off transfer audit TP size must be positive")
    return [
        os.path.join(IH_RANK_LOG_DIR, f"rank{rank}.log")
        for rank in range(tp_size)
    ]


def _ih_rank_log_id(path):
    match = re.fullmatch(r"rank(\d+)\.log", os.path.basename(str(path)))
    return int(match.group(1)) if match else None


def _ih_read_shared_snapshot_audit(start_offsets=None):
    """Read only post-confirm snapshot markers from every expected TP log."""

    start_offsets = start_offsets or {}
    paths = (
        tuple(start_offsets)
        if start_offsets
        else tuple(_ih_transfer_audit_log_paths())
    )
    events = []
    parse_errors = []
    log_read_errors = []
    marker = "REDKNOT_SHARED_SNAPSHOT_AUDIT "
    for path in paths:
        path_rank = _ih_rank_log_id(path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(int(start_offsets.get(path, 0)))
                for line in handle:
                    if marker not in line:
                        continue
                    serialized = line.split(marker, 1)[1].strip()
                    try:
                        payload = json.loads(serialized)
                    except json.JSONDecodeError as error:
                        parse_errors.append(
                            f"rank_path={path_rank}:malformed_json:{error.msg}"
                        )
                        continue
                    if not isinstance(payload, dict):
                        parse_errors.append(
                            f"rank_path={path_rank}:payload_not_object"
                        )
                        continue
                    events.append(
                        {
                            "path": str(path),
                            "path_rank": path_rank,
                            "payload": payload,
                        }
                    )
        except OSError as error:
            log_read_errors.append(f"{path}:{error}")
    return {
        "events": events,
        "parse_errors": sorted(set(parse_errors)),
        "log_read_errors": sorted(set(log_read_errors)),
    }


def _ih_validate_shared_snapshot_audit(
    raw_audit,
    *,
    expected_tp_size,
    expected_segment_hashes,
):
    """Prove every rank confirmed every offline segment in all three stores."""

    expected_tp_size = int(expected_tp_size)
    if expected_tp_size <= 0:
        raise ValueError("shared snapshot audit needs a positive TP size")
    expected_hashes = tuple(map(str, expected_segment_hashes))
    if not expected_hashes or any(not value for value in expected_hashes):
        raise ValueError("shared snapshot audit needs non-empty segment hashes")
    expected_ranks = set(range(expected_tp_size))
    raw_audit = raw_audit if isinstance(raw_audit, Mapping) else {}
    raw_events = raw_audit.get("events", ())
    global_errors = list(map(str, raw_audit.get("parse_errors", ())))
    global_errors.extend(map(str, raw_audit.get("log_read_errors", ())))
    if len(set(expected_hashes)) != len(expected_hashes):
        global_errors.append("expected_segment_hashes_not_unique")
    grouped = defaultdict(list)
    for event_index, wrapper in enumerate(raw_events):
        if not isinstance(wrapper, dict) or not isinstance(
            wrapper.get("payload"), dict
        ):
            global_errors.append(f"event={event_index}:malformed_wrapper")
            continue
        payload = wrapper["payload"]
        rank = payload.get("tp_rank")
        if type(rank) is not int:
            global_errors.append(f"event={event_index}:invalid_tp_rank")
            continue
        grouped[rank].append(wrapper)

    unexpected_ranks = sorted(set(grouped) - expected_ranks)
    if unexpected_ranks:
        global_errors.append(f"unexpected_ranks={unexpected_ranks}")
    per_rank = {}
    failed_ranks = []
    required_store_names = ("cpu", "gpu", "z_off")
    for rank in sorted(expected_ranks):
        wrappers = grouped.get(rank, ())
        rank_errors = []
        payloads = []
        observed_hashes = []
        observed_totals = []
        for ordinal, wrapper in enumerate(wrappers, start=1):
            payload = wrapper["payload"]
            payloads.append(payload)
            prefix = f"event={ordinal - 1}"
            if wrapper.get("path_rank") != rank:
                rank_errors.append(f"{prefix}:path_payload_rank_mismatch")
            if payload.get("schema") != _IH_SHARED_SNAPSHOT_AUDIT_SCHEMA:
                rank_errors.append(f"{prefix}:schema_mismatch")
            if payload.get("tp_size") != expected_tp_size:
                rank_errors.append(f"{prefix}:tp_size_mismatch")
            published_total = payload.get("published_total")
            if type(published_total) is not int or published_total <= 0:
                rank_errors.append(f"{prefix}:invalid_published_total")
            else:
                if observed_totals and published_total != observed_totals[-1] + 1:
                    rank_errors.append(f"{prefix}:published_total_not_consecutive")
                observed_totals.append(published_total)
            segment_hash = payload.get("segment_hash")
            if not isinstance(segment_hash, str) or not segment_hash:
                rank_errors.append(f"{prefix}:invalid_segment_hash")
            else:
                observed_hashes.append(segment_hash)
                if ordinal <= len(expected_hashes) and segment_hash != expected_hashes[
                    ordinal - 1
                ]:
                    rank_errors.append(f"{prefix}:segment_order_mismatch")
            stores = payload.get("stores")
            if not isinstance(stores, dict):
                rank_errors.append(f"{prefix}:stores_malformed")
                continue
            if set(stores) != set(required_store_names):
                rank_errors.append(f"{prefix}:store_set_mismatch")
            for store_name in required_store_names:
                state = stores.get(store_name)
                if not isinstance(state, dict):
                    rank_errors.append(f"{prefix}:{store_name}:state_malformed")
                    continue
                if any(
                    type(state.get(field)) is not bool
                    for field in ("active", "ready", "pending")
                ):
                    rank_errors.append(f"{prefix}:{store_name}:state_not_boolean")
                    continue
                if not state["active"]:
                    rank_errors.append(f"{prefix}:{store_name}:not_active")
                if not state["ready"]:
                    rank_errors.append(f"{prefix}:{store_name}:not_ready")
                if state["pending"]:
                    rank_errors.append(f"{prefix}:{store_name}:still_pending")
        if len(wrappers) != len(expected_hashes):
            rank_errors.append(
                "published_count_mismatch="
                f"{len(wrappers)}!=expected={len(expected_hashes)}"
            )
        if tuple(observed_hashes) != expected_hashes:
            rank_errors.append("segment_hashes_mismatch")
        if rank_errors:
            failed_ranks.append(rank)
        per_rank[str(rank)] = {
            "shared_snapshot_published": len(wrappers),
            "expected_shared_snapshot_published": len(expected_hashes),
            "published_counter_first": (
                observed_totals[0] if observed_totals else None
            ),
            "published_counter_last": (
                observed_totals[-1] if observed_totals else None
            ),
            "segment_hashes": observed_hashes,
            "stores_all_active_ready_not_pending": not any(
                ":not_active" in error
                or ":not_ready" in error
                or ":still_pending" in error
                or ":state_" in error
                or ":stores_" in error
                or ":store_set_" in error
                for error in rank_errors
            ),
            "events": payloads,
            "errors": sorted(set(rank_errors)),
            "complete": not rank_errors,
        }
    return {
        "schema": _IH_SHARED_SNAPSHOT_AUDIT_SCHEMA,
        "expected_tp_size": expected_tp_size,
        "expected_segment_hashes": list(expected_hashes),
        "expected_shared_snapshot_published_per_rank": len(expected_hashes),
        "per_rank": per_rank,
        "failed_rank_ids": failed_ranks,
        "global_errors": sorted(set(global_errors)),
        "raw_event_count": len(raw_events),
        "pass": not global_errors and not failed_ranks,
    }


def _ih_runtime_log_offsets():
    offsets = {}
    paths = list(_ih_runtime_log_paths())
    if IH_MLA_OFFLOAD:
        paths.extend(_ih_transfer_audit_log_paths())
    for path in dict.fromkeys(paths):
        try:
            offsets[path] = os.path.getsize(path)
        except OSError:
            offsets[path] = 0
    return offsets


def _ih_valid_forward_position_span(
    *,
    q_rows: int,
    position_start: int,
    position_end: int,
    position_contiguous: bool,
    allow_sparse_positions: bool,
) -> bool:
    """Validate physical rows against their scheduler-owned logical span."""

    if type(position_contiguous) is not bool or type(allow_sparse_positions) is not bool:
        return False
    if any(type(value) is not int for value in (q_rows, position_start, position_end)):
        return False
    logical_span = position_end - position_start
    if q_rows <= 0 or position_start < 0 or logical_span <= 0:
        return False
    if allow_sparse_positions:
        # Combined selected-row execution may represent a whole logical 8K
        # document with one prefix placeholder, or with several non-contiguous
        # islands.  The manifest's q_rows is physical work; start/end remains
        # the exact scheduler-owned logical coverage interval.
        return q_rows <= logical_span
    return position_contiguous and q_rows == logical_span


def _ih_read_runtime_metrics(start_offsets=None):
    metrics = {
        "active_rows": 0,
        "full_rows": 0,
        "samples": 0,
        "fallbacks": 0,
        "mla_off_samples": 0,
        "reused_local_head_rows": 0,
        "online_local_head_rows": 0,
        "online_global_head_rows": 0,
        "mla_log_read_errors": [],
    }
    transfer_audit_events = []
    transfer_audit_parse_errors = []
    compact_woa_events = []
    compact_woa_parse_errors = []
    composite_receipt_events = []
    composite_receipt_parse_errors = []
    request_layers = defaultdict(set)
    request_forwards = defaultdict(set)
    forward_manifests = defaultdict(dict)
    forward_manifest_counts = Counter()
    forward_layers = defaultdict(lambda: defaultdict(set))
    forward_metric_rows = {}
    forward_metric_counts = Counter()
    fallback_request_ids = set()
    fallback_forward_layers = defaultdict(lambda: defaultdict(set))
    full_local_request_layers = defaultdict(set)
    full_local_forward_layers = defaultdict(lambda: defaultdict(set))
    full_local_forward_reasons = defaultdict(lambda: defaultdict(dict))
    full_local_status_counts = Counter()
    evidence_errors = []
    evidence_error_request_ids = set()

    def evidence_error(request_id, detail):
        request_id = str(request_id)
        evidence_error_request_ids.add(request_id)
        evidence_errors.append(f"{request_id}:{detail}")

    start_offsets = start_offsets or {}
    if start_offsets:
        metric_paths = tuple(start_offsets)
    else:
        metric_paths = list(_ih_runtime_log_paths())
        if IH_MLA_OFFLOAD:
            metric_paths.extend(_ih_transfer_audit_log_paths())
        metric_paths = tuple(dict.fromkeys(metric_paths))
    for path in metric_paths:
        path_rank = _ih_rank_log_id(path)
        parse_rank0_metrics = path_rank in (None, 0)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(int(start_offsets.get(path, 0)))
                for line in handle:
                    transfer_marker = "REDKNOT_MLA_OFF_CONTROLLER_STATS "
                    if transfer_marker in line:
                        serialized = line.split(transfer_marker, 1)[1].strip()
                        try:
                            payload = json.loads(serialized)
                        except json.JSONDecodeError as error:
                            transfer_audit_parse_errors.append(
                                f"rank_path={path_rank}:malformed_json:{error.msg}"
                            )
                        else:
                            if not isinstance(payload, dict):
                                transfer_audit_parse_errors.append(
                                    f"rank_path={path_rank}:payload_not_object"
                                )
                            else:
                                observed_diagnostic = payload.get(
                                    "diagnostic_ablation"
                                )
                                if observed_diagnostic != (
                                    IH_MLA_OFF_DIAGNOSTIC_ABLATION
                                ):
                                    transfer_audit_parse_errors.append(
                                        "rank_path="
                                        f"{path_rank}:diagnostic_ablation_mismatch:"
                                        f"observed={observed_diagnostic!r}:"
                                        "expected="
                                        f"{IH_MLA_OFF_DIAGNOSTIC_ABLATION!r}"
                                    )
                                transfer_audit_events.append(
                                    {
                                        "path": str(path),
                                        "path_rank": path_rank,
                                        "payload": payload,
                                    }
                                )
                    # All existing FORWARD/METRIC/REQUEST and sparse-row
                    # counters remain rank-0 authoritative.  Parsing them from
                    # TP1..N would multiply logical head-row evidence.
                    if not parse_rank0_metrics:
                        continue
                    composite_receipt_marker = (
                        "REDKNOT_MLA_OFF_COMPOSITE_RECEIPT "
                    )
                    if composite_receipt_marker in line:
                        serialized = line.split(
                            composite_receipt_marker, 1
                        )[1].strip()
                        try:
                            payload = json.loads(serialized)
                        except json.JSONDecodeError as error:
                            composite_receipt_parse_errors.append(
                                f"rank_path={path_rank}:malformed_json:{error.msg}"
                            )
                        else:
                            if not isinstance(payload, dict):
                                composite_receipt_parse_errors.append(
                                    f"rank_path={path_rank}:payload_not_object"
                                )
                            else:
                                composite_receipt_events.append(
                                    {
                                        "path": str(path),
                                        "path_rank": path_rank,
                                        "payload": payload,
                                    }
                                )
                    compact_marker = "REDKNOT_MLA_OFF_COMPACT_WOA "
                    if compact_marker in line:
                        serialized = line.split(compact_marker, 1)[1].strip()
                        try:
                            payload = json.loads(serialized)
                        except json.JSONDecodeError as error:
                            compact_woa_parse_errors.append(
                                f"rank_path={path_rank}:malformed_json:{error.msg}"
                            )
                        else:
                            if not isinstance(payload, dict):
                                compact_woa_parse_errors.append(
                                    f"rank_path={path_rank}:payload_not_object"
                                )
                            else:
                                compact_woa_events.append(
                                    {
                                        "path": str(path),
                                        "path_rank": path_rank,
                                        "payload": payload,
                                    }
                                )
                    match = re.search(
                        r"REDKNOT_METRIC active_rows .*?active=(\d+) full=(\d+)",
                        line,
                    )
                    if match:
                        metrics["active_rows"] += int(match.group(1))
                        metrics["full_rows"] += int(match.group(2))
                        metrics["samples"] += 1
                    forward_match = re.search(
                        r"REDKNOT_MLA_OFF_FORWARD request_id=(\S+) "
                        r"forward_id=(\S+) forward_mode=(\S+) q_rows=(\d+) "
                        r"position_start=(-?\d+) position_end=(-?\d+) "
                        r"position_contiguous=([01]) plan_mode=(\S+) "
                        r"diagnostic_ablation=(\S+)",
                        line,
                    )
                    if forward_match:
                        request_id, forward_id = forward_match.group(1, 2)
                        manifest = {
                            "forward_mode": forward_match.group(3),
                            "q_rows": int(forward_match.group(4)),
                            "position_start": int(forward_match.group(5)),
                            "position_end": int(forward_match.group(6)),
                            "position_contiguous": bool(
                                int(forward_match.group(7))
                            ),
                            "plan_mode": forward_match.group(8),
                            "diagnostic_ablation": forward_match.group(9),
                        }
                        if manifest["diagnostic_ablation"] != (
                            IH_MLA_OFF_DIAGNOSTIC_ABLATION
                        ):
                            evidence_error(
                                request_id,
                                f"forward={forward_id}:"
                                "diagnostic_ablation_mismatch",
                            )
                        manifest_key = (request_id, forward_id)
                        forward_manifest_counts[manifest_key] += 1
                        if forward_manifest_counts[manifest_key] > 1:
                            evidence_error(
                                request_id,
                                f"forward={forward_id}:duplicate_manifest",
                            )
                        previous = forward_manifests[request_id].get(forward_id)
                        if previous is not None and previous != manifest:
                            evidence_error(
                                request_id,
                                f"forward={forward_id}:conflicting_manifest",
                            )
                        forward_manifests[request_id][forward_id] = manifest
                        request_forwards[request_id].add(forward_id)
                    mla_match = re.search(
                        r"REDKNOT_MLA_OFF_METRIC request_id=(\S+) "
                        r"forward_id=(\S+) forward_mode=(\S+) q_rows=(\d+) "
                        r"layer=(\d+) "
                        r"reused_local_head_rows=(\d+) "
                        r"online_local_head_rows=(\d+) online_global_head_rows=(\d+) "
                        r"diagnostic_ablation=(\S+)",
                        line,
                    )
                    if mla_match:
                        request_id, forward_id = mla_match.group(1, 2)
                        forward_mode = mla_match.group(3)
                        q_rows = int(mla_match.group(4))
                        layer_id = int(mla_match.group(5))
                        row_values = tuple(
                            int(mla_match.group(index)) for index in (6, 7, 8)
                        )
                        if mla_match.group(9) != (
                            IH_MLA_OFF_DIAGNOSTIC_ABLATION
                        ):
                            evidence_error(
                                request_id,
                                f"forward={forward_id}:layer={layer_id}:"
                                "diagnostic_ablation_mismatch",
                            )
                        key = (request_id, forward_id, layer_id)
                        forward_metric_counts[key] += 1
                        if forward_metric_counts[key] > 1:
                            evidence_error(
                                request_id,
                                f"forward={forward_id}:layer={layer_id}:duplicate_metric",
                            )
                        previous_rows = forward_metric_rows.get(key)
                        metric_record = (forward_mode, q_rows, *row_values)
                        if previous_rows is not None and previous_rows != metric_record:
                            evidence_error(
                                request_id,
                                f"forward={forward_id}:layer={layer_id}:conflicting_metric",
                            )
                        forward_metric_rows[key] = metric_record
                        request_forwards[request_id].add(forward_id)
                        request_layers[request_id].add(layer_id)
                        forward_layers[request_id][forward_id].add(layer_id)
                        metrics["reused_local_head_rows"] += row_values[0]
                        metrics["online_local_head_rows"] += row_values[1]
                        metrics["online_global_head_rows"] += row_values[2]
                        metrics["mla_off_samples"] += 1
                    request_match = re.search(
                        r"REDKNOT_MLA_OFF_REQUEST request_id=(\S+) "
                        r"forward_id=(\S+) forward_mode=(\S+) q_rows=(\d+) "
                        r"layer=(\d+) status=(\S+) reason=(\S+) "
                        r"diagnostic_ablation=(\S+)",
                        line,
                    )
                    if request_match:
                        request_id, forward_id = request_match.group(1, 2)
                        forward_mode = request_match.group(3)
                        q_rows = int(request_match.group(4))
                        request_layer = int(request_match.group(5))
                        request_status = request_match.group(6)
                        request_reason = request_match.group(7)
                        if request_match.group(8) != (
                            IH_MLA_OFF_DIAGNOSTIC_ABLATION
                        ):
                            evidence_error(
                                request_id,
                                f"forward={forward_id}:layer={request_layer}:"
                                "diagnostic_ablation_mismatch",
                            )
                        request_forwards[request_id].add(forward_id)
                        manifest = forward_manifests[request_id].get(forward_id)
                        if manifest is not None and (
                            manifest["forward_mode"] != forward_mode
                            or int(manifest["q_rows"]) != q_rows
                        ):
                            evidence_error(
                                request_id,
                                f"forward={forward_id}:request_context_mismatch",
                            )
                        if request_status == "fallback":
                            fallback_request_ids.add(request_id)
                            fallback_forward_layers[request_id][forward_id].add(
                                request_layer
                            )
                        elif request_status == "full_local":
                            status_key = (request_id, forward_id, request_layer)
                            full_local_status_counts[status_key] += 1
                            if full_local_status_counts[status_key] > 1:
                                evidence_error(
                                    request_id,
                                    f"forward={forward_id}:layer={request_layer}:"
                                    "duplicate_full_local_status",
                                )
                            full_local_request_layers[request_id].add(
                                request_layer
                            )
                            full_local_forward_layers[request_id][forward_id].add(
                                request_layer
                            )
                            previous_reason = full_local_forward_reasons[request_id][
                                forward_id
                            ].get(str(request_layer))
                            if (
                                previous_reason is not None
                                and previous_reason != request_reason
                            ):
                                evidence_error(
                                    request_id,
                                    f"forward={forward_id}:layer={request_layer}:"
                                    "conflicting_full_local_reason",
                                )
                            full_local_forward_reasons[request_id][forward_id][
                                str(request_layer)
                            ] = request_reason
                            if request_reason not in (
                                "no_reusable_rows",
                                "query_suffix_only",
                                "refresh_layer",
                                "uncertified_context",
                                "missing_total_tokens",
                                "missing_actual_total_tokens",
                                "total_tokens_mismatch",
                                "context_exceeds_certification",
                            ):
                                evidence_error(
                                    request_id,
                                    f"forward={forward_id}:layer={request_layer}:"
                                    "unexpected_full_local_reason",
                                )
                        else:
                            evidence_error(
                                request_id,
                                f"forward={forward_id}:layer={request_layer}:"
                                f"unknown_status={request_status}",
                            )
                    if any(
                        marker in line
                        for marker in (
                            "RedKnot selected-row dense fallback",
                            "RedKnot selected-row selection failed",
                            "RedKnot V4 dense fallback",
                            "RedKnot segmented compressor fell back to dense",
                            "RedKnot merged prefill fail-closed",
                            "selected-row compressor failed",
                            "RedKnot MLA-off disabled for this forward",
                            "RedKnot MLA-off snapshot aborted",
                            "RedKnot MLA-off policy was not applied",
                        )
                    ):
                        metrics["fallbacks"] += 1
        except OSError as error:
            metrics["mla_log_read_errors"].append(f"{path}:{error}")
    for request_id, manifests in forward_manifests.items():
        for forward_id, manifest in manifests.items():
            if int(manifest["q_rows"]) <= 0:
                evidence_error(request_id, f"forward={forward_id}:nonpositive_q_rows")
            if not _ih_valid_forward_position_span(
                q_rows=int(manifest["q_rows"]),
                position_start=int(manifest["position_start"]),
                position_end=int(manifest["position_end"]),
                position_contiguous=bool(manifest["position_contiguous"]),
                allow_sparse_positions=bool(IH_COMBINED_HEADSPLIT_ROW_SPARSE),
            ):
                evidence_error(
                    request_id, f"forward={forward_id}:invalid_position_span"
                )
    for key, metric_record in forward_metric_rows.items():
        request_id, forward_id, layer_id = key
        forward_mode, q_rows, reused_rows, online_local_rows, online_global_rows = (
            metric_record
        )
        manifest = forward_manifests[request_id].get(forward_id)
        if manifest is None:
            evidence_error(request_id, f"forward={forward_id}:missing_manifest")
        elif (
            manifest["forward_mode"] != forward_mode
            or int(manifest["q_rows"]) != int(q_rows)
        ):
            evidence_error(
                request_id,
                f"forward={forward_id}:layer={layer_id}:metric_context_mismatch",
            )
        if min(reused_rows, online_local_rows, online_global_rows) < 0:
            evidence_error(
                request_id,
                f"forward={forward_id}:layer={layer_id}:negative_rows",
            )
    for request_id, forwards in full_local_forward_layers.items():
        for forward_id, layers in forwards.items():
            for layer_id in layers:
                metric_record = forward_metric_rows.get(
                    (request_id, forward_id, layer_id)
                )
                if metric_record is None:
                    evidence_error(
                        request_id,
                        f"forward={forward_id}:layer={layer_id}:"
                        "full_local_missing_metric",
                    )
                    continue
                _, _, reused_rows, online_local_rows, online_global_rows = (
                    metric_record
                )
                if reused_rows != 0 or online_local_rows + online_global_rows <= 0:
                    evidence_error(
                        request_id,
                        f"forward={forward_id}:layer={layer_id}:"
                        "invalid_full_local_rows",
                    )
    if metrics["full_rows"]:
        metrics["online_row_saving"] = 1.0 - (
            metrics["active_rows"] / metrics["full_rows"]
        )
    else:
        metrics["online_row_saving"] = None
    total_head_rows = (
        metrics["reused_local_head_rows"]
        + metrics["online_local_head_rows"]
        + metrics["online_global_head_rows"]
    )
    metrics["mla_head_row_saving"] = (
        metrics["reused_local_head_rows"] / total_head_rows
        if total_head_rows
        else None
    )
    metrics["mla_request_layers"] = {
        request_id: sorted(layers)
        for request_id, layers in sorted(request_layers.items())
    }
    metrics["mla_fallback_request_ids"] = sorted(fallback_request_ids)
    metrics["mla_request_forwards"] = {
        request_id: sorted(forward_ids)
        for request_id, forward_ids in sorted(request_forwards.items())
    }
    metrics["mla_forward_manifest"] = {
        request_id: {
            forward_id: dict(manifest)
            for forward_id, manifest in sorted(manifests.items())
        }
        for request_id, manifests in sorted(forward_manifests.items())
    }
    metrics["mla_forward_layers"] = {
        request_id: {
            forward_id: sorted(layers)
            for forward_id, layers in sorted(forwards.items())
        }
        for request_id, forwards in sorted(forward_layers.items())
    }
    metric_rows_by_request = defaultdict(lambda: defaultdict(dict))
    for (
        request_id,
        forward_id,
        layer_id,
    ), metric_record in forward_metric_rows.items():
        (
            forward_mode,
            q_rows,
            reused_rows,
            online_local_rows,
            online_global_rows,
        ) = metric_record
        metric_rows_by_request[request_id][forward_id][str(layer_id)] = {
            "forward_mode": forward_mode,
            "q_rows": q_rows,
            "reused_local_head_rows": reused_rows,
            "online_local_head_rows": online_local_rows,
            "online_global_head_rows": online_global_rows,
        }
    metrics["mla_forward_metric_rows"] = {
        request_id: {
            forward_id: dict(sorted(layers.items(), key=lambda item: int(item[0])))
            for forward_id, layers in sorted(forwards.items())
        }
        for request_id, forwards in sorted(metric_rows_by_request.items())
    }
    metrics["mla_fallback_forward_layers"] = {
        request_id: {
            forward_id: sorted(layers)
            for forward_id, layers in sorted(forwards.items())
        }
        for request_id, forwards in sorted(fallback_forward_layers.items())
    }
    metrics["mla_full_local_request_layers"] = {
        request_id: sorted(layers)
        for request_id, layers in sorted(full_local_request_layers.items())
    }
    metrics["mla_full_local_forward_layers"] = {
        request_id: {
            forward_id: sorted(layers)
            for forward_id, layers in sorted(forwards.items())
        }
        for request_id, forwards in sorted(full_local_forward_layers.items())
    }
    metrics["mla_full_local_forward_reasons"] = {
        request_id: {
            forward_id: dict(sorted(reasons.items(), key=lambda item: int(item[0])))
            for forward_id, reasons in sorted(forwards.items())
        }
        for request_id, forwards in sorted(full_local_forward_reasons.items())
    }
    metrics["mla_evidence_errors"] = sorted(set(evidence_errors))
    metrics["mla_evidence_error_request_ids"] = sorted(
        evidence_error_request_ids
    )
    metrics["mla_off_transfer_audit"] = {
        "schema": _IH_MLA_OFF_TRANSFER_AUDIT_SCHEMA,
        "byte_semantics": _IH_MLA_OFF_TRANSFER_BYTE_SEMANTICS,
        "events": transfer_audit_events,
        "parse_errors": sorted(set(transfer_audit_parse_errors)),
        "pass": None,
    }
    metrics["compact_woa_evidence"] = {
        "schema": _IH_MLA_OFF_COMPACT_WOA_AUDIT_SCHEMA,
        "measurement_semantics": (
            _IH_MLA_OFF_COMPACT_WOA_MEASUREMENT_SEMANTICS
        ),
        "claim_scope": _IH_MLA_OFF_COMPACT_WOA_CLAIM_SCOPE,
        "events": compact_woa_events,
        "parse_errors": sorted(set(compact_woa_parse_errors)),
        "required": False,
        "pass": None,
    }
    metrics["mla_off_composite_receipts"] = {
        "schema": _IH_MLA_OFF_COMPOSITE_RECEIPT_SCHEMA,
        "events": composite_receipt_events,
        "parse_errors": sorted(set(composite_receipt_parse_errors)),
        "required": False,
        "pass": None,
    }
    return metrics


def _ih_validate_compact_woa_evidence(
    runtime_metrics,
    *,
    expected_request_ids,
    expected_layer_ids,
    expected_head_counts_by_layer,
    required=None,
    expected_mode=None,
    allowed_request_ids=(),
):
    """Validate rank-0 proof that compact inverse-RoPE/``wo_a`` ran.

    This audit is row-geometry activation evidence only.  Its row counts are
    never promoted to total-model FLOPs, GPU utilization, or energy savings.
    A required formal reuse request must emit exactly one marker for every
    all-local layer/forward whose independent logical-head metric proves at
    least one reused token row. A zero-reuse/full-local forward does not run a
    compact subset and must not emit a marker. Every observed marker must agree
    with the independent row metric.
    """

    valid_modes = {"required", "forbidden", "not_applicable"}
    if expected_mode is None:
        if type(required) is not bool:
            raise ValueError(
                "compact wo_a evidence required flag must be boolean"
            )
        expected_mode = "required" if required else "not_applicable"
    elif expected_mode not in valid_modes:
        raise ValueError(
            "compact wo_a expected_mode must be required, forbidden, or "
            "not_applicable"
        )
    elif required is not None:
        if type(required) is not bool or required != (
            expected_mode == "required"
        ):
            raise ValueError(
                "compact wo_a required flag conflicts with expected_mode"
            )
    required = expected_mode == "required"
    forbidden = expected_mode == "forbidden"
    expected_request_ids = set(map(str, expected_request_ids))
    allowed_request_ids = set(map(str, allowed_request_ids))
    audited_request_ids = expected_request_ids | allowed_request_ids
    expected_layer_ids = tuple(int(value) for value in expected_layer_ids)
    all_local_layers = []
    local_heads_by_layer = {}
    for layer_id in expected_layer_ids:
        counts = expected_head_counts_by_layer.get(
            str(layer_id), expected_head_counts_by_layer.get(layer_id)
        )
        if not isinstance(counts, dict):
            continue
        local_heads = counts.get("local")
        global_heads = counts.get("global")
        if (
            type(local_heads) is int
            and local_heads > 0
            and type(global_heads) is int
            and global_heads == 0
        ):
            all_local_layers.append(layer_id)
            local_heads_by_layer[layer_id] = local_heads

    raw = runtime_metrics.get("compact_woa_evidence", {})
    if not isinstance(raw, dict):
        raw = {}
    raw_events = raw.get("events", ())
    if not isinstance(raw_events, (list, tuple)):
        raw_events = ()
    parse_errors = raw.get("parse_errors", ())
    if not isinstance(parse_errors, (list, tuple)):
        parse_errors = ("parse_errors_not_sequence",)
    global_errors = [str(error) for error in parse_errors]
    if forbidden and raw_events:
        global_errors.append(
            f"compact_marker_forbidden={len(raw_events)}"
        )
    errors_by_request = defaultdict(list)
    grouped = defaultdict(list)
    marker_errors_by_key = defaultdict(list)

    def request_error(request_id, detail):
        errors_by_request[str(request_id)].append(str(detail))

    def strict_nonnegative_int(value):
        return type(value) is int and value >= 0

    for event_index, wrapper in enumerate(raw_events):
        if not isinstance(wrapper, dict) or not isinstance(
            wrapper.get("payload"), dict
        ):
            global_errors.append(f"event={event_index}:malformed_wrapper")
            continue
        payload = wrapper["payload"]
        request_id = payload.get("request_id")
        forward_id = payload.get("forward_id")
        layer_id = payload.get("layer")
        if not isinstance(request_id, str) or not request_id:
            global_errors.append(f"event={event_index}:invalid_request_id")
            continue
        if request_id not in audited_request_ids:
            global_errors.append(
                f"event={event_index}:unexpected_request={request_id}"
            )
        if not isinstance(forward_id, str) or not forward_id:
            request_error(request_id, f"event={event_index}:invalid_forward_id")
            continue
        if type(layer_id) is not int or layer_id < 0:
            request_error(request_id, f"event={event_index}:invalid_layer")
            continue
        key = (request_id, forward_id, layer_id)
        grouped[key].append(wrapper)

        prefix = f"forward={forward_id}:layer={layer_id}"

        def marker_error(detail):
            marker_errors_by_key[key].append(str(detail))
            request_error(request_id, f"{prefix}:{detail}")

        if wrapper.get("path_rank") != 0:
            marker_error("marker_not_from_rank0_log")
        if payload.get("tp_rank") != 0:
            marker_error("payload_rank_not_zero")
        if payload.get("schema") != _IH_MLA_OFF_COMPACT_WOA_AUDIT_SCHEMA:
            marker_error("schema_mismatch")
        if (
            payload.get("measurement_semantics")
            != _IH_MLA_OFF_COMPACT_WOA_MEASUREMENT_SEMANTICS
        ):
            marker_error("measurement_semantics_mismatch")
        if payload.get("claim_scope") != _IH_MLA_OFF_COMPACT_WOA_CLAIM_SCOPE:
            marker_error("claim_scope_mismatch")
        if not isinstance(payload.get("forward_mode"), str):
            marker_error("invalid_forward_mode")
        full_rows = payload.get("full_rows")
        online_rows = payload.get("online_rows")
        if not strict_nonnegative_int(full_rows) or full_rows <= 0:
            marker_error("invalid_full_rows")
        if not strict_nonnegative_int(online_rows):
            marker_error("invalid_online_rows")
        elif strict_nonnegative_int(full_rows) and online_rows >= full_rows:
            marker_error("online_rows_not_strict_subset")

    manifests = runtime_metrics.get("mla_forward_manifest", {})
    metric_rows = runtime_metrics.get("mla_forward_metric_rows", {})
    expected_keys = set()

    def metric_requires_compact_marker(request_id, forward_id, layer_id):
        """Use only the independent head-row metric to decide expectation."""

        local_heads = local_heads_by_layer.get(layer_id)
        if type(local_heads) is not int or local_heads <= 0:
            return False
        record = (
            metric_rows.get(request_id, {})
            .get(forward_id, {})
            .get(str(layer_id))
        )
        if not isinstance(record, dict):
            return False
        reused_head_rows = record.get("reused_local_head_rows")
        online_head_rows = record.get("online_local_head_rows")
        global_head_rows = record.get("online_global_head_rows")
        if not all(
            strict_nonnegative_int(value)
            for value in (
                reused_head_rows,
                online_head_rows,
                global_head_rows,
            )
        ):
            return False
        if (
            global_head_rows != 0
            or reused_head_rows % local_heads
            or online_head_rows % local_heads
        ):
            return False
        reused_token_rows = reused_head_rows // local_heads
        online_token_rows = online_head_rows // local_heads
        return bool(
            reused_token_rows > 0
            and online_token_rows < reused_token_rows + online_token_rows
        )

    if required:
        if not all_local_layers:
            global_errors.append("no_expected_all_local_layers")
        for request_id in expected_request_ids:
            request_manifests = manifests.get(request_id, {})
            if not request_manifests:
                request_error(request_id, "missing_forward_manifest")
            for forward_id in request_manifests:
                expected_keys.update(
                    (request_id, forward_id, layer_id)
                    for layer_id in all_local_layers
                    if metric_requires_compact_marker(
                        request_id, forward_id, layer_id
                    )
                )
        if not expected_keys:
            global_errors.append("no_expected_compact_activation")

    by_request = defaultdict(lambda: defaultdict(dict))
    formal_full_rows = 0
    formal_online_rows = 0
    observed_formal_events = 0
    keys_to_validate = set(grouped) | expected_keys
    for request_id, forward_id, layer_id in sorted(keys_to_validate):
        wrappers = grouped.get((request_id, forward_id, layer_id), ())
        is_formal = request_id in expected_request_ids
        is_expected = (request_id, forward_id, layer_id) in expected_keys
        prefix = f"forward={forward_id}:layer={layer_id}"
        entry_errors = list(
            marker_errors_by_key.get((request_id, forward_id, layer_id), ())
        )

        if request_id not in audited_request_ids:
            entry_errors.append("unexpected_request")
        manifest = manifests.get(request_id, {}).get(forward_id)
        if not isinstance(manifest, dict):
            entry_errors.append("unexpected_or_missing_forward")
        if layer_id not in local_heads_by_layer:
            entry_errors.append("layer_not_expected_all_local")
        if is_expected and not wrappers:
            entry_errors.append("missing_marker")
        if len(wrappers) > 1:
            entry_errors.append(f"duplicate_markers={len(wrappers)}")

        record = (
            metric_rows.get(request_id, {})
            .get(forward_id, {})
            .get(str(layer_id))
        )
        expected_full_rows = None
        expected_online_rows = None
        reused_token_rows = None
        if not isinstance(record, dict):
            entry_errors.append("missing_head_row_metric")
        elif layer_id in local_heads_by_layer:
            local_heads = local_heads_by_layer[layer_id]
            reused_head_rows = record.get("reused_local_head_rows")
            online_head_rows = record.get("online_local_head_rows")
            global_head_rows = record.get("online_global_head_rows")
            if not all(
                strict_nonnegative_int(value)
                for value in (
                    reused_head_rows,
                    online_head_rows,
                    global_head_rows,
                )
            ):
                entry_errors.append("invalid_head_row_metric")
            else:
                if global_head_rows != 0:
                    entry_errors.append("all_local_layer_has_global_rows")
                if (
                    reused_head_rows % local_heads
                    or online_head_rows % local_heads
                ):
                    entry_errors.append("head_rows_not_token_row_aligned")
                else:
                    reused_token_rows = reused_head_rows // local_heads
                    expected_online_rows = online_head_rows // local_heads
                    expected_full_rows = reused_token_rows + expected_online_rows
                    if reused_token_rows <= 0 and wrappers:
                        entry_errors.append(
                            "marker_without_reused_rows"
                        )
        if isinstance(manifest, dict):
            manifest_q_rows = manifest.get("q_rows")
            if type(manifest_q_rows) is not int or manifest_q_rows <= 0:
                entry_errors.append("invalid_manifest_q_rows")
            elif (
                expected_full_rows is not None
                and expected_full_rows != manifest_q_rows
            ):
                entry_errors.append("head_rows_disagree_with_manifest")

        payload = wrappers[0].get("payload", {}) if len(wrappers) == 1 else None
        if isinstance(payload, dict):
            if isinstance(manifest, dict) and payload.get(
                "forward_mode"
            ) != manifest.get("forward_mode"):
                entry_errors.append("forward_mode_mismatch")
            if (
                expected_full_rows is not None
                and payload.get("full_rows") != expected_full_rows
            ):
                entry_errors.append("full_rows_mismatch")
            if (
                expected_online_rows is not None
                and payload.get("online_rows") != expected_online_rows
            ):
                entry_errors.append("online_rows_mismatch")
            if is_formal:
                observed_formal_events += 1
                if strict_nonnegative_int(payload.get("full_rows")):
                    formal_full_rows += payload["full_rows"]
                if strict_nonnegative_int(payload.get("online_rows")):
                    formal_online_rows += payload["online_rows"]

        for error in entry_errors:
            formatted = f"{prefix}:{error}"
            if formatted not in errors_by_request.get(request_id, ()):
                request_error(request_id, formatted)
        by_request[request_id][forward_id][str(layer_id)] = {
            "expected": is_expected,
            "observed_marker_count": len(wrappers),
            "expected_full_rows": expected_full_rows,
            "expected_online_rows": expected_online_rows,
            "reused_token_rows": reused_token_rows,
            "payload": payload,
            "errors": sorted(set(entry_errors)),
            "complete": is_expected and len(wrappers) == 1 and not entry_errors,
        }

    failed_audited_request_ids = sorted(
        request_id
        for request_id in audited_request_ids
        if errors_by_request.get(request_id)
    )
    failed_request_ids = sorted(
        set(failed_audited_request_ids) & expected_request_ids
    )
    evidence_errors = sorted(set(global_errors)) + sorted(
        f"{request_id}:{detail}"
        for request_id, errors in errors_by_request.items()
        for detail in set(errors)
    )
    pass_value = None
    if expected_mode != "not_applicable":
        pass_value = not global_errors and not failed_audited_request_ids
    return {
        "schema": _IH_MLA_OFF_COMPACT_WOA_AUDIT_SCHEMA,
        "measurement_semantics": (
            _IH_MLA_OFF_COMPACT_WOA_MEASUREMENT_SEMANTICS
        ),
        "claim_scope": _IH_MLA_OFF_COMPACT_WOA_CLAIM_SCOPE,
        "metric_scope": (
            "successful compact inverse-RoPE/wo_a input rows; "
            "not total-model FLOPs, utilization, or energy"
        ),
        "required": required,
        "forbidden": forbidden,
        "expected_mode": expected_mode,
        "expected_all_local_layer_ids": sorted(all_local_layers),
        "expected_formal_event_count": len(expected_keys),
        "observed_formal_event_count": observed_formal_events,
        "formal_full_rows": formal_full_rows,
        "formal_online_rows": formal_online_rows,
        "formal_online_row_fraction": (
            formal_online_rows / formal_full_rows if formal_full_rows else None
        ),
        "by_request": {
            request_id: {
                forward_id: dict(
                    sorted(layers.items(), key=lambda item: int(item[0]))
                )
                for forward_id, layers in sorted(forwards.items())
            }
            for request_id, forwards in sorted(by_request.items())
        },
        "parse_errors": sorted(set(map(str, parse_errors))),
        "global_errors": sorted(set(global_errors)),
        "evidence_errors": evidence_errors,
        "errors_by_request": {
            request_id: sorted(set(errors))
            for request_id, errors in sorted(errors_by_request.items())
            if errors
        },
        "failed_request_ids": failed_request_ids,
        "failed_audited_request_ids": failed_audited_request_ids,
        "raw_event_count": len(raw_events),
        "pass": pass_value,
    }


def _ih_validate_mla_transfer_audit(
    runtime_metrics,
    *,
    expected_request_ids,
    expected_tp_size,
    expected_device_cache_enabled,
    expected_device_max_bytes,
    allowed_request_ids=(),
):
    """Validate one controller-delta event from every TP rank and forward.

    Device-backed restore has three mutually exclusive evidence contracts.  The
    current composite shared-cache path proves a restore when all three
    ``shared_restore`` deltas are positive.  Older providers instead prove the
    same operation through the legacy device/gather/scatter counters.  A
    ``zoff_only`` combined path instead consumes a persistent device-resident
    z_off artifact directly, so both cache-restore counter families must stay
    inactive while the runtime head-row/packed-Q receipts prove the omission.
    A partial shared delta or evidence from multiple contracts is ambiguous and
    must fail closed.  For a forward with no reused rows (notably the final
    online suffix), every contract is expected to remain inactive.
    """

    expected_request_ids = set(map(str, expected_request_ids))
    allowed_request_ids = set(map(str, allowed_request_ids))
    audited_request_ids = expected_request_ids | allowed_request_ids
    expected_tp_size = int(expected_tp_size)
    if expected_tp_size <= 0:
        raise ValueError("MLA-off transfer audit needs a positive TP size")
    expected_device_cache_enabled = bool(expected_device_cache_enabled)
    expected_device_max_bytes = int(expected_device_max_bytes)
    if expected_device_max_bytes < 0:
        raise ValueError("MLA-off device cache cap must be non-negative")
    expected_ranks = set(range(expected_tp_size))
    raw = runtime_metrics.get("mla_off_transfer_audit", {})
    raw_events = raw.get("events", ()) if isinstance(raw, dict) else ()
    parse_errors = list(raw.get("parse_errors", ())) if isinstance(raw, dict) else []
    global_errors = [str(error) for error in parse_errors]
    errors_by_request = defaultdict(list)
    grouped = defaultdict(lambda: defaultdict(list))

    def request_error(request_id, detail):
        errors_by_request[str(request_id)].append(str(detail))

    def strict_nonnegative_int(value):
        return type(value) is int and value >= 0

    for event_index, wrapper in enumerate(raw_events):
        if not isinstance(wrapper, dict) or not isinstance(
            wrapper.get("payload"), dict
        ):
            global_errors.append(f"event={event_index}:malformed_wrapper")
            continue
        payload = wrapper["payload"]
        request_id = payload.get("request_id")
        forward_id = payload.get("forward_id")
        tp_rank = payload.get("tp_rank")
        if not isinstance(request_id, str) or not request_id:
            global_errors.append(f"event={event_index}:invalid_request_id")
            continue
        if not isinstance(forward_id, str) or not forward_id:
            request_error(request_id, f"event={event_index}:invalid_forward_id")
            continue
        if request_id not in audited_request_ids:
            global_errors.append(
                f"event={event_index}:unexpected_request={request_id}"
            )
        if type(tp_rank) is not int:
            request_error(request_id, f"forward={forward_id}:invalid_tp_rank")
            continue
        grouped[(request_id, forward_id)][tp_rank].append(wrapper)

    manifests = runtime_metrics.get("mla_forward_manifest", {})
    metric_rows = runtime_metrics.get("mla_forward_metric_rows", {})
    expected_forward_keys = set()
    for request_id in audited_request_ids:
        for forward_id in manifests.get(request_id, {}):
            expected_forward_keys.add((request_id, forward_id))
    observed_forward_keys = set(grouped)
    for request_id, forward_id in sorted(
        observed_forward_keys - expected_forward_keys
    ):
        request_error(request_id, f"forward={forward_id}:unexpected_audit_forward")

    by_request = defaultdict(dict)
    totals = Counter()
    controller_totals_by_rank = defaultdict(Counter)
    shared_restore_totals = Counter()
    shared_restore_totals_by_rank = defaultdict(Counter)
    accepted_device_restore_events_by_rank = Counter()
    zoff_persistent_events_by_rank = Counter()
    restore_evidence_modes = set()
    required_device_restore_forward_count = 0
    for request_id, forward_id in sorted(expected_forward_keys):
        manifest = manifests[request_id][forward_id]
        forward_error_count_before = len(errors_by_request.get(request_id, ()))
        rank_wrappers = grouped.get((request_id, forward_id), {})
        observed_ranks = set(rank_wrappers)
        missing_ranks = sorted(expected_ranks - observed_ranks)
        unexpected_ranks = sorted(observed_ranks - expected_ranks)
        duplicate_ranks = sorted(
            rank for rank, wrappers in rank_wrappers.items() if len(wrappers) != 1
        )
        if missing_ranks:
            request_error(
                request_id,
                f"forward={forward_id}:missing_ranks={missing_ranks}",
            )
        if unexpected_ranks:
            request_error(
                request_id,
                f"forward={forward_id}:unexpected_ranks={unexpected_ranks}",
            )
        if duplicate_ranks:
            request_error(
                request_id,
                f"forward={forward_id}:duplicate_ranks={duplicate_ranks}",
            )
        rank_payloads = {}
        aggregate = Counter()
        shared_restore_aggregate = Counter()
        restore_evidence_modes_by_rank = {}
        device_cache_enabled_by_rank = []
        reused_rows = sum(
            int(record.get("reused_local_head_rows", 0))
            for record in metric_rows.get(request_id, {})
            .get(forward_id, {})
            .values()
        )
        for rank in sorted(expected_ranks & observed_ranks):
            wrappers = rank_wrappers[rank]
            if len(wrappers) != 1:
                continue
            wrapper = wrappers[0]
            payload = wrapper["payload"]
            event_prefix = f"forward={forward_id}:rank={rank}"
            path_rank = wrapper.get("path_rank")
            if path_rank != rank:
                request_error(
                    request_id,
                    f"{event_prefix}:path_payload_rank_mismatch={path_rank}",
                )
            if payload.get("schema") != _IH_MLA_OFF_TRANSFER_AUDIT_SCHEMA:
                request_error(request_id, f"{event_prefix}:schema_mismatch")
            if (
                payload.get("byte_semantics")
                != _IH_MLA_OFF_TRANSFER_BYTE_SEMANTICS
            ):
                request_error(request_id, f"{event_prefix}:byte_semantics_mismatch")
            if payload.get("tp_size") != expected_tp_size:
                request_error(request_id, f"{event_prefix}:tp_size_mismatch")
            if (
                payload.get("forward_mode") != manifest.get("forward_mode")
                or payload.get("q_rows") != manifest.get("q_rows")
            ):
                request_error(request_id, f"{event_prefix}:forward_context_mismatch")
            manifest_ablation = str(
                manifest.get("diagnostic_ablation", "full") or "full"
            )
            payload_ablation = str(
                payload.get("diagnostic_ablation", "full") or "full"
            )
            if payload_ablation != manifest_ablation:
                request_error(
                    request_id,
                    f"{event_prefix}:diagnostic_ablation_mismatch",
                )
            start = payload.get("counter_start")
            end = payload.get("counter_end")
            delta = payload.get("counter_delta")
            gauges = payload.get("gauge_snapshot")
            if not all(isinstance(value, dict) for value in (start, end, delta, gauges)):
                request_error(request_id, f"{event_prefix}:counter_schema_malformed")
                continue
            shared_restore = payload.get("shared_restore")
            shared_counters_valid = True
            if not isinstance(shared_restore, dict):
                request_error(request_id, f"{event_prefix}:shared_restore_missing")
                shared_counters_valid = False
                shared_start = shared_end = shared_delta = {}
            else:
                if shared_restore.get("schema") != _IH_SHARED_RESTORE_AUDIT_SCHEMA:
                    request_error(
                        request_id, f"{event_prefix}:shared_restore_schema_mismatch"
                    )
                    shared_counters_valid = False
                shared_start = shared_restore.get("counter_start")
                shared_end = shared_restore.get("counter_end")
                shared_delta = shared_restore.get("counter_delta")
                if not all(
                    isinstance(value, dict)
                    for value in (shared_start, shared_end, shared_delta)
                ):
                    request_error(
                        request_id,
                        f"{event_prefix}:shared_restore_counter_schema_malformed",
                    )
                    shared_counters_valid = False
                    shared_start = shared_end = shared_delta = {}
            for field in _IH_SHARED_RESTORE_COUNTER_FIELDS:
                values = (
                    shared_start.get(field),
                    shared_end.get(field),
                    shared_delta.get(field),
                )
                if not all(strict_nonnegative_int(value) for value in values):
                    request_error(
                        request_id,
                        f"{event_prefix}:invalid_shared_restore_counter={field}",
                    )
                    shared_counters_valid = False
                    continue
                if shared_end[field] - shared_start[field] != shared_delta[field]:
                    request_error(
                        request_id,
                        f"{event_prefix}:shared_restore_delta_mismatch={field}",
                    )
                    shared_counters_valid = False
            counters_valid = True
            for field in _IH_MLA_OFF_TRANSFER_COUNTER_FIELDS:
                values = (start.get(field), end.get(field), delta.get(field))
                if not all(strict_nonnegative_int(value) for value in values):
                    request_error(
                        request_id, f"{event_prefix}:invalid_counter={field}"
                    )
                    counters_valid = False
                    continue
                if end[field] - start[field] != delta[field]:
                    request_error(
                        request_id, f"{event_prefix}:delta_mismatch={field}"
                    )
                    counters_valid = False
            for snapshot_name, snapshot in (("start", start), ("end", end), ("delta", delta)):
                if (
                    snapshot.get("online_index_h2d_bytes")
                    != snapshot.get("online_device_gather_index_h2d_bytes", 0)
                    + snapshot.get("online_device_scatter_index_h2d_bytes", 0)
                    + snapshot.get("online_dirty_index_h2d_bytes", 0)
                    or snapshot.get("online_total_h2d_bytes")
                    != snapshot.get("online_artifact_h2d_bytes", 0)
                    + snapshot.get("online_index_h2d_bytes", 0)
                ):
                    request_error(
                        request_id,
                        f"{event_prefix}:{snapshot_name}_derived_bytes_mismatch",
                    )
                    counters_valid = False
            for prefix in (
                "online_device_gather_index_h2d",
                "online_device_scatter_index_h2d",
                "online_dirty_index_h2d",
                "snapshot_device_index_h2d",
            ):
                if delta.get(f"{prefix}_bytes") != 8 * delta.get(
                    f"{prefix}_rows", 0
                ) and prefix != "snapshot_device_index_h2d":
                    request_error(
                        request_id, f"{event_prefix}:logical_index_bytes={prefix}"
                    )
                    counters_valid = False
                if (
                    delta.get(f"{prefix}_bytes", 0) > 0
                    and delta.get(f"{prefix}_calls", 0) <= 0
                ):
                    request_error(
                        request_id, f"{event_prefix}:missing_call={prefix}"
                    )
                    counters_valid = False
            # Snapshot masks may be bool while positions are int64; online
            # restore must never perform either kind of snapshot upload.
            if delta.get("snapshot_device_index_h2d_calls", 0) != 0:
                request_error(request_id, f"{event_prefix}:snapshot_copy_during_restore")
                counters_valid = False
            if not all(
                strict_nonnegative_int(gauges.get(field))
                for field in _IH_MLA_OFF_TRANSFER_GAUGE_FIELDS
            ):
                request_error(request_id, f"{event_prefix}:invalid_gauges")
            else:
                if gauges["device_cache_enabled"] != int(
                    expected_device_cache_enabled
                ):
                    request_error(request_id, f"{event_prefix}:device_mode_mismatch")
                if gauges["max_device_cache_bytes"] != expected_device_max_bytes:
                    request_error(request_id, f"{event_prefix}:device_cap_mismatch")
                if expected_device_cache_enabled:
                    if not (
                        0
                        < gauges["allocated_device_bytes"]
                        <= gauges["reserved_device_bytes"]
                        <= gauges["max_device_cache_bytes"]
                    ):
                        request_error(
                            request_id, f"{event_prefix}:device_capacity_inconsistent"
                        )
                elif gauges["reserved_device_bytes"] or gauges[
                    "allocated_device_bytes"
                ]:
                    request_error(
                        request_id, f"{event_prefix}:disabled_device_bytes_nonzero"
                    )
            if counters_valid and reused_rows > 0:
                if expected_device_cache_enabled:
                    if (
                        delta["online_artifact_h2d_bytes"] != 0
                        or delta["online_artifact_h2d_calls"] != 0
                        or delta["rows_restored"] != 0
                    ):
                        request_error(
                            request_id, f"{event_prefix}:device_artifact_h2d_nonzero"
                        )
                else:
                    if (
                        delta["online_artifact_h2d_calls"] <= 0
                        or delta["online_artifact_h2d_bytes"] <= 0
                        or delta["rows_restored"] <= 0
                    ):
                        request_error(
                            request_id, f"{event_prefix}:missing_cpu_artifact_h2d"
                        )
                    if (
                        delta["device_restore_calls"] != 0
                        or delta["device_rows_restored"] != 0
                    ):
                        request_error(
                            request_id, f"{event_prefix}:unexpected_device_restore"
                        )
            shared_values = tuple(
                shared_delta.get(field, 0)
                for field in _IH_SHARED_RESTORE_COUNTER_FIELDS
            )
            shared_active = bool(
                shared_counters_valid and all(value > 0 for value in shared_values)
            )
            shared_inactive = bool(
                shared_counters_valid and all(value == 0 for value in shared_values)
            )
            shared_partial = bool(
                shared_counters_valid and not shared_active and not shared_inactive
            )
            legacy_values = tuple(
                delta.get(field, 0)
                for field in _IH_LEGACY_DEVICE_RESTORE_COUNTER_FIELDS
            )
            legacy_active = bool(counters_valid and any(legacy_values))
            legacy_complete = bool(
                counters_valid
                and delta.get("device_restore_calls", 0) > 0
                and delta.get("device_rows_restored", 0) > 0
                and delta.get("online_device_gather_index_h2d_calls", 0) > 0
                and delta.get("online_device_gather_index_h2d_rows", 0) > 0
                and delta.get("online_device_gather_index_h2d_bytes", 0) > 0
                and delta.get("online_device_scatter_index_h2d_calls", 0) > 0
                and delta.get("online_device_scatter_index_h2d_rows", 0)
                == delta.get("online_device_gather_index_h2d_rows", 0)
                and delta.get("online_device_scatter_index_h2d_bytes", 0)
                == delta.get("online_device_gather_index_h2d_bytes", 0)
            )
            evidence_mode = "none"
            if shared_partial:
                request_error(
                    request_id, f"{event_prefix}:partial_shared_device_restore"
                )
            if reused_rows > 0 and expected_device_cache_enabled:
                if manifest_ablation == "zoff_only":
                    if shared_active or legacy_active:
                        request_error(
                            request_id,
                            f"{event_prefix}:mixed_zoff_cache_restore_evidence",
                        )
                        evidence_mode = "conflict"
                    elif shared_inactive:
                        evidence_mode = "zoff_persistent"
                        zoff_persistent_events_by_rank[rank] += 1
                elif shared_active:
                    if legacy_active:
                        request_error(
                            request_id,
                            f"{event_prefix}:mixed_shared_legacy_device_restore",
                        )
                        evidence_mode = "conflict"
                    else:
                        evidence_mode = "shared_composite"
                elif shared_inactive:
                    if not legacy_complete:
                        if (
                            delta.get("device_restore_calls", 0) <= 0
                            or delta.get("device_rows_restored", 0) <= 0
                        ):
                            request_error(
                                request_id,
                                f"{event_prefix}:missing_device_restore",
                            )
                        if (
                            delta.get(
                                "online_device_gather_index_h2d_bytes", 0
                            )
                            <= 0
                            or delta.get(
                                "online_device_scatter_index_h2d_bytes", 0
                            )
                            != delta.get(
                                "online_device_gather_index_h2d_bytes", 0
                            )
                        ):
                            request_error(
                                request_id,
                                f"{event_prefix}:device_gather_scatter_index_mismatch",
                            )
                        if legacy_active:
                            request_error(
                                request_id,
                                f"{event_prefix}:partial_legacy_device_restore",
                            )
                    else:
                        evidence_mode = "legacy_device"
                if evidence_mode in (
                    "shared_composite",
                    "legacy_device",
                    "zoff_persistent",
                ):
                    accepted_device_restore_events_by_rank[rank] += 1
                    restore_evidence_modes.add(evidence_mode)
            elif reused_rows > 0:
                if shared_active:
                    request_error(
                        request_id,
                        f"{event_prefix}:unexpected_shared_device_restore",
                    )
                    evidence_mode = "conflict"
            elif shared_active or legacy_active:
                request_error(
                    request_id,
                    f"{event_prefix}:unexpected_restore_without_reused_rows",
                )
                evidence_mode = "conflict"
            restore_evidence_modes_by_rank[str(rank)] = evidence_mode
            rank_payloads[str(rank)] = payload
            if isinstance(gauges, dict) and type(
                gauges.get("device_cache_enabled")
            ) is int:
                device_cache_enabled_by_rank.append(
                    gauges["device_cache_enabled"]
                )
            for field in _IH_MLA_OFF_TRANSFER_COUNTER_FIELDS:
                value = delta.get(field)
                if strict_nonnegative_int(value):
                    aggregate[field] += value
                    totals[field] += value
                    controller_totals_by_rank[rank][field] += value
            for field in _IH_SHARED_RESTORE_COUNTER_FIELDS:
                value = shared_delta.get(field)
                if strict_nonnegative_int(value):
                    shared_restore_aggregate[field] += value
                    shared_restore_totals[field] += value
                    shared_restore_totals_by_rank[rank][field] += value
        aggregate_payload = dict(aggregate)
        aggregate_payload.update(
            {
                "online_artifact_h2d_bytes_sum": aggregate.get(
                    "online_artifact_h2d_bytes", 0
                ),
                "online_index_h2d_bytes_sum": aggregate.get(
                    "online_index_h2d_bytes", 0
                ),
                "online_total_h2d_bytes_sum": aggregate.get(
                    "online_total_h2d_bytes", 0
                ),
                "device_restore_calls_sum": aggregate.get(
                    "device_restore_calls", 0
                ),
                "device_cache_enabled_all_ranks": (
                    len(device_cache_enabled_by_rank) == expected_tp_size
                    and all(device_cache_enabled_by_rank)
                ),
                "shared_restore": dict(shared_restore_aggregate),
            }
        )
        by_request[request_id][forward_id] = {
            "rank_count": len(rank_payloads),
            "required_rank_ids": sorted(expected_ranks),
            "observed_rank_ids": sorted(observed_ranks),
            "missing_rank_ids": missing_ranks,
            "duplicate_rank_ids": duplicate_ranks,
            "ranks": rank_payloads,
            "aggregate": aggregate_payload,
            "restore_evidence_modes_by_rank": restore_evidence_modes_by_rank,
            "restore_evidence_mode": (
                next(iter(set(restore_evidence_modes_by_rank.values())))
                if len(set(restore_evidence_modes_by_rank.values())) == 1
                else "mixed"
            ),
            "complete": (
                observed_ranks == expected_ranks
                and not duplicate_ranks
                and len(errors_by_request.get(request_id, ()))
                == forward_error_count_before
            ),
        }
        if reused_rows > 0 and expected_device_cache_enabled:
            required_device_restore_forward_count += 1
    complete_rank_totals = set(controller_totals_by_rank) == expected_ranks
    complete_shared_rank_totals = (
        set(shared_restore_totals_by_rank) == expected_ranks
    )
    online_artifact_h2d_zero_all_ranks = bool(
        complete_rank_totals
        and all(
            controller_totals_by_rank[rank]["online_artifact_h2d_calls"] == 0
            and controller_totals_by_rank[rank]["online_artifact_h2d_bytes"] == 0
            for rank in expected_ranks
        )
    )
    shared_device_restore_positive_all_ranks = bool(
        complete_shared_rank_totals
        and all(
            all(
                shared_restore_totals_by_rank[rank][field] > 0
                for field in _IH_SHARED_RESTORE_COUNTER_FIELDS
            )
            for rank in expected_ranks
        )
    )
    legacy_device_restore_positive_all_ranks = bool(
        complete_rank_totals
        and all(
            controller_totals_by_rank[rank]["device_restore_calls"] > 0
            and controller_totals_by_rank[rank]["device_rows_restored"] > 0
            and controller_totals_by_rank[rank][
                "online_device_gather_index_h2d_bytes"
            ]
            > 0
            and controller_totals_by_rank[rank][
                "online_device_scatter_index_h2d_bytes"
            ]
            == controller_totals_by_rank[rank][
                "online_device_gather_index_h2d_bytes"
            ]
            for rank in expected_ranks
        )
    )
    composite_device_restore_positive_all_ranks = bool(
        required_device_restore_forward_count == 0
        or all(
            accepted_device_restore_events_by_rank[rank]
            == required_device_restore_forward_count
            for rank in expected_ranks
        )
    )
    zoff_persistent_positive_all_ranks = bool(
        zoff_persistent_events_by_rank
        and set(zoff_persistent_events_by_rank) == expected_ranks
        and all(zoff_persistent_events_by_rank[rank] > 0 for rank in expected_ranks)
    )
    if not expected_forward_keys:
        global_errors.append("no_expected_forwards")
    elif expected_device_cache_enabled:
        if not online_artifact_h2d_zero_all_ranks:
            global_errors.append("online_artifact_h2d_not_zero_all_ranks")
        if not composite_device_restore_positive_all_ranks:
            global_errors.append("composite_device_restore_not_positive_all_ranks")
    if global_errors:
        for request_id in expected_request_ids:
            errors_by_request[request_id].append("global_audit_error")
    failed_request_ids = sorted(
        request_id
        for request_id in expected_request_ids
        if errors_by_request.get(request_id)
    )
    evidence_errors = sorted(set(global_errors)) + sorted(
        f"{request_id}:{detail}"
        for request_id, errors in errors_by_request.items()
        for detail in set(errors)
    )
    return {
        "schema": _IH_MLA_OFF_TRANSFER_AUDIT_SCHEMA,
        "byte_semantics": _IH_MLA_OFF_TRANSFER_BYTE_SEMANTICS,
        "expected_tp_size": expected_tp_size,
        "required_rank_ids": sorted(expected_ranks),
        "by_request": {
            request_id: dict(sorted(forwards.items()))
            for request_id, forwards in sorted(by_request.items())
        },
        "totals": dict(totals),
        "totals_by_rank": {
            str(rank): dict(controller_totals_by_rank[rank])
            for rank in sorted(controller_totals_by_rank)
        },
        "shared_restore_totals": dict(shared_restore_totals),
        "shared_restore_totals_by_rank": {
            str(rank): dict(shared_restore_totals_by_rank[rank])
            for rank in sorted(shared_restore_totals_by_rank)
        },
        "positive_evidence": {
            "restore_evidence_contract": (
                "exclusive_shared_composite_legacy_or_persistent_zoff_v3"
            ),
            "required_device_restore_forward_count": (
                required_device_restore_forward_count
            ),
            "restore_evidence_modes": sorted(restore_evidence_modes),
            "composite_device_restore_positive_all_ranks": (
                composite_device_restore_positive_all_ranks
            ),
            "shared_device_restore_positive_all_ranks": (
                shared_device_restore_positive_all_ranks
            ),
            "legacy_device_restore_positive_all_ranks": (
                legacy_device_restore_positive_all_ranks
            ),
            "zoff_persistent_positive_all_ranks": (
                zoff_persistent_positive_all_ranks
            ),
            "online_artifact_h2d_zero_all_ranks": (
                online_artifact_h2d_zero_all_ranks
            ),
            "online_h2d_claim_scope": (
                "z_off_artifact_payload_only_excludes_online_indices_and_"
                "shared_schedule_v1"
            ),
        },
        "global_errors": sorted(set(global_errors)),
        "evidence_errors": evidence_errors,
        "errors_by_request": {
            request_id: sorted(set(errors))
            for request_id, errors in sorted(errors_by_request.items())
            if errors
        },
        "failed_request_ids": failed_request_ids,
        "event_count": len(raw_events),
        "pass": not global_errors and not failed_request_ids,
    }


def _ih_validate_prefix_materialization_runtime(
    runtime_metrics,
    *,
    expected_request_ids,
    online_rows_by_request,
    query_start,
    query_suffix_rows,
    expected_layer_ids,
    expected_head_counts_by_layer,
    tensor_parallel_size,
    expected_device_cache_enabled,
    expected_device_max_bytes,
    materialization_runtime_pass=True,
):
    """Validate the explicit RedKnot prefix-materialization execution class.

    The prefix is produced by the context-bound snapshot chain and matched by
    one server-instance/cache identity.  A divergent sentinel in the producer
    makes all document tokens internal radix nodes.  Formal evidence therefore
    requires the online forward to contain exactly the frozen query suffix and
    rejects even a one-token document recompute.
    """

    del (
        runtime_metrics,
        expected_layer_ids,
        expected_head_counts_by_layer,
        tensor_parallel_size,
        expected_device_cache_enabled,
        expected_device_max_bytes,
    )
    expected_request_ids = set(map(str, expected_request_ids))
    online_rows_by_request = {
        str(key): int(value) for key, value in online_rows_by_request.items()
    }
    query_start = int(query_start)
    query_suffix_rows = int(query_suffix_rows)
    total_rows = query_start + query_suffix_rows
    errors = defaultdict(list)
    if set(online_rows_by_request) != expected_request_ids:
        raise ValueError(
            "prefix materialization online-row map does not span request ids"
        )
    for request_id in sorted(expected_request_ids):
        online_rows = online_rows_by_request[request_id]
        if online_rows != query_suffix_rows:
            errors[request_id].append(
                "online_rows_not_exact_query_suffix:"
                f"observed={online_rows}:expected={query_suffix_rows}"
            )
    if not materialization_runtime_pass:
        for request_id in sorted(expected_request_ids):
            errors[request_id].append("materialization_runtime_failed")
    online_prompt_rows = sum(online_rows_by_request.values())
    total_prompt_rows = len(expected_request_ids) * total_rows
    saved_prompt_rows = total_prompt_rows - online_prompt_rows
    pass_value = bool(
        expected_request_ids
        and not errors
        and materialization_runtime_pass
    )
    return {
        "schema": "redknot_prefix_materialization_runtime_v2",
        "pass": pass_value,
        "failed_request_ids": sorted(errors),
        "errors_by_request": dict(errors),
        "request_count": len(expected_request_ids),
        "query_start": query_start,
        "query_suffix_rows": query_suffix_rows,
        "online_rows_by_request": online_rows_by_request,
        "saved_prompt_rows": saved_prompt_rows,
        "total_prompt_rows": total_prompt_rows,
        "online_prompt_row_saving": (
            saved_prompt_rows / total_prompt_rows if total_prompt_rows else None
        ),
        "execution": "native_query_suffix_on_redknot_materialized_prefix_v1",
        "materialization_runtime_pass": bool(materialization_runtime_pass),
    }


def _ih_validate_mla_off_composite_receipts(
    runtime_metrics,
    *,
    expected_request_ids,
    expected_layer_ids,
    required,
    expected_diagnostic_ablation,
):
    """Bind each measured reuse forward to one rank-0 final certificate."""

    if type(required) is not bool:
        raise ValueError("composite receipt required flag must be boolean")
    expected_profile = {
        "full": "full",
        "zoff_only": "zoff_only",
    }.get(expected_diagnostic_ablation)
    if required and expected_profile is None:
        raise ValueError(
            "required composite receipts support full or zoff_only execution"
        )
    raw = runtime_metrics.get("mla_off_composite_receipts", {})
    events = raw.get("events", []) if isinstance(raw, dict) else []
    parse_errors = (
        raw.get("parse_errors", []) if isinstance(raw, dict) else []
    )
    if not required:
        return {
            "schema": _IH_MLA_OFF_COMPOSITE_RECEIPT_SCHEMA,
            "required": False,
            "expected_diagnostic_ablation": expected_diagnostic_ablation,
            "expected_omission_profile": expected_profile,
            "events": list(events) if isinstance(events, list) else [],
            "parse_errors": (
                list(parse_errors) if isinstance(parse_errors, list) else []
            ),
            "measured_reuse_forwards": [],
            "failed_request_ids": [],
            "errors_by_request": {},
            "global_errors": [],
            "pass": None,
        }

    expected_request_ids = {str(value) for value in expected_request_ids}
    expected_layer_ids = [int(value) for value in expected_layer_ids]
    metric_rows = runtime_metrics.get("mla_forward_metric_rows", {})
    manifests = runtime_metrics.get("mla_forward_manifest", {})
    measured_reuse_keys = set()
    for request_id in expected_request_ids:
        request_rows = metric_rows.get(request_id, {})
        if not isinstance(request_rows, dict):
            continue
        for forward_id, layer_rows in request_rows.items():
            if not isinstance(layer_rows, dict):
                continue
            if any(
                isinstance(record, dict)
                and type(record.get("reused_local_head_rows")) is int
                and record["reused_local_head_rows"] > 0
                for record in layer_rows.values()
            ):
                measured_reuse_keys.add((request_id, str(forward_id)))

    errors_by_request = defaultdict(list)
    global_errors = []
    grouped = defaultdict(list)
    normalized_events = []
    if not isinstance(events, list):
        global_errors.append("events_not_list")
        events = []
    if not isinstance(parse_errors, list):
        global_errors.append("parse_errors_not_list")
        parse_errors = []
    global_errors.extend(f"parse_error:{value}" for value in parse_errors)
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            global_errors.append(f"event={event_index}:wrapper_not_object")
            continue
        payload = event.get("payload")
        path_rank = event.get("path_rank")
        if not isinstance(payload, dict):
            global_errors.append(f"event={event_index}:payload_not_object")
            continue
        request_id = payload.get("request_id")
        forward_id = payload.get("forward_id")
        if type(request_id) is not str or not request_id:
            global_errors.append(f"event={event_index}:invalid_request_id")
            continue
        if type(forward_id) is not str or not forward_id:
            errors_by_request[request_id].append(
                f"event={event_index}:invalid_forward_id"
            )
            continue
        key = (request_id, forward_id)
        grouped[key].append(payload)
        normalized_events.append(
            {
                "path": str(event.get("path", "")),
                "path_rank": path_rank,
                "payload": dict(payload),
            }
        )
        reasons = []
        if path_rank not in (None, 0):
            reasons.append("not_rank0")
        if key not in measured_reuse_keys:
            reasons.append("receipt_for_non_reuse_forward")
        manifest = manifests.get(request_id, {}).get(forward_id)
        if not isinstance(manifest, dict):
            reasons.append("missing_forward_manifest")
        elif manifest.get("diagnostic_ablation") != expected_diagnostic_ablation:
            reasons.append("manifest_diagnostic_ablation_mismatch")
        expected_values = {
            "schema": _IH_MLA_OFF_COMPOSITE_RECEIPT_SCHEMA,
            "diagnostic_ablation": expected_diagnostic_ablation,
            "omission_profile": expected_profile,
            "commit_scope": COMMIT_SCOPE_FORWARD_RESERVED,
        }
        for field, expected_value in expected_values.items():
            if type(payload.get(field)) is not str or payload.get(field) != expected_value:
                reasons.append(f"{field}_mismatch")
        generation_id = payload.get("generation_id")
        if type(generation_id) is not str or not generation_id:
            reasons.append("invalid_generation_id")
        for field in (
            "shared_digest",
            "prepare_certificate_digest",
            "receipt_manifest_digest",
            "execution_certificate_digest",
        ):
            value = payload.get(field)
            if type(value) is not str or re.fullmatch(
                r"sha256:[0-9a-f]{64}", value
            ) is None:
                reasons.append(f"invalid_{field}")
        reusable_layer_ids = payload.get("reusable_layer_ids")
        if (
            type(reusable_layer_ids) is not list
            or reusable_layer_ids != expected_layer_ids
            or any(type(value) is not int for value in reusable_layer_ids)
        ):
            reasons.append("reusable_layer_ids_mismatch")
        omission_slot_count = payload.get("omission_slot_count")
        if type(omission_slot_count) is not int or omission_slot_count <= 0:
            reasons.append("invalid_omission_slot_count")
        elif expected_profile == "zoff_only" and omission_slot_count != (
            2 * len(expected_layer_ids)
        ):
            reasons.append("zoff_only_omission_slot_count_mismatch")
        elif expected_profile == "full" and omission_slot_count <= (
            2 * len(expected_layer_ids)
        ):
            reasons.append("full_omission_slot_count_missing_cache_slots")
        if reasons:
            errors_by_request[request_id].extend(
                f"forward={forward_id}:{reason}" for reason in reasons
            )

    for request_id, forward_id in sorted(measured_reuse_keys):
        count = len(grouped.get((request_id, forward_id), ()))
        if count != 1:
            errors_by_request[request_id].append(
                f"forward={forward_id}:receipt_count={count}:expected=1"
            )
    failed_request_ids = sorted(
        request_id
        for request_id, errors in errors_by_request.items()
        if errors
    )
    return {
        "schema": _IH_MLA_OFF_COMPOSITE_RECEIPT_SCHEMA,
        "required": True,
        "expected_diagnostic_ablation": expected_diagnostic_ablation,
        "expected_omission_profile": expected_profile,
        "events": normalized_events,
        "parse_errors": list(parse_errors),
        "measured_reuse_forwards": [
            {"request_id": request_id, "forward_id": forward_id}
            for request_id, forward_id in sorted(measured_reuse_keys)
        ],
        "failed_request_ids": failed_request_ids,
        "errors_by_request": {
            request_id: sorted(set(errors))
            for request_id, errors in sorted(errors_by_request.items())
            if errors
        },
        "global_errors": sorted(set(global_errors)),
        "pass": bool(
            measured_reuse_keys
            and not failed_request_ids
            and not global_errors
        ),
    }


def _ih_validate_mla_runtime_metrics(
    runtime_metrics,
    *,
    expected_request_ids,
    expected_q_rows_by_request,
    expected_position_start_by_request=None,
    expected_layer_ids,
    expected_request_count,
    expected_head_counts_by_layer,
    max_q_rows_per_forward,
    full_local_sanity=False,
    num_model_layers=None,
    num_attention_heads=None,
    tensor_parallel_size=None,
    transfer_audit_required=False,
    transfer_audit_device_cache_enabled=False,
    transfer_audit_device_max_bytes=0,
    transfer_audit_allowed_request_ids=(),
    compact_woa_required=None,
    compact_woa_expected_mode=None,
    allow_sparse_positions=False,
    composite_receipt_required=False,
    expected_diagnostic_ablation="full",
):
    """Fail closed for every observed chunked prefill forward, not just a request union."""

    expected_request_ids = set(expected_request_ids)
    expected_position_start_by_request = dict(
        expected_position_start_by_request or {}
    )
    expected_layer_ids = [int(layer_id) for layer_id in expected_layer_ids]
    expected_layer_set = set(expected_layer_ids)
    max_q_rows_per_forward = int(max_q_rows_per_forward)
    if not isinstance(full_local_sanity, bool):
        raise ValueError("full_local_sanity must be a boolean")
    if not isinstance(transfer_audit_required, bool):
        raise ValueError("transfer_audit_required must be a boolean")
    if not isinstance(allow_sparse_positions, bool):
        raise ValueError("allow_sparse_positions must be a boolean")
    if not isinstance(composite_receipt_required, bool):
        raise ValueError("composite_receipt_required must be a boolean")
    if compact_woa_required is not None and type(compact_woa_required) is not bool:
        raise ValueError("compact_woa_required must be boolean or None")
    valid_compact_modes = {"required", "forbidden", "not_applicable"}
    if (
        compact_woa_expected_mode is not None
        and compact_woa_expected_mode not in valid_compact_modes
    ):
        raise ValueError("compact_woa_expected_mode is invalid")
    if compact_woa_expected_mode is not None and compact_woa_required is not None:
        if compact_woa_required != (compact_woa_expected_mode == "required"):
            raise ValueError(
                "compact_woa_required conflicts with expected mode"
            )
    request_forwards = runtime_metrics.get("mla_request_forwards", {})
    forward_manifests = runtime_metrics.get("mla_forward_manifest", {})
    forward_layers = runtime_metrics.get("mla_forward_layers", {})
    forward_metric_rows = runtime_metrics.get("mla_forward_metric_rows", {})
    full_local_forward_layers = runtime_metrics.get(
        "mla_full_local_forward_layers", {}
    )
    full_local_forward_reasons = runtime_metrics.get(
        "mla_full_local_forward_reasons", {}
    )
    composite_receipts = _ih_validate_mla_off_composite_receipts(
        runtime_metrics,
        expected_request_ids=expected_request_ids,
        expected_layer_ids=expected_layer_ids,
        required=composite_receipt_required,
        expected_diagnostic_ablation=expected_diagnostic_ablation,
    )
    def expected_all_local(layer_id):
        counts = expected_head_counts_by_layer.get(
            str(layer_id), expected_head_counts_by_layer.get(layer_id)
        )
        return bool(
            isinstance(counts, dict)
            and type(counts.get("local")) is int
            and counts["local"] > 0
            and type(counts.get("global")) is int
            and counts["global"] == 0
        )

    if compact_woa_expected_mode is None:
        if compact_woa_required is None:
            compact_woa_required = bool(
                any(
                    expected_all_local(layer_id)
                    for layer_id in expected_layer_ids
                )
                and not full_local_sanity
            )
        compact_woa_expected_mode = (
            "required" if compact_woa_required else "not_applicable"
        )
    compact_woa_required = compact_woa_expected_mode == "required"
    compact_woa_evidence = _ih_validate_compact_woa_evidence(
        runtime_metrics,
        expected_request_ids=expected_request_ids,
        expected_layer_ids=expected_layer_ids,
        expected_head_counts_by_layer=expected_head_counts_by_layer,
        expected_mode=compact_woa_expected_mode,
        allowed_request_ids=transfer_audit_allowed_request_ids,
    )
    if transfer_audit_required:
        if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0:
            raise ValueError(
                "required MLA-off transfer audit needs tensor_parallel_size"
            )
        transfer_audit = _ih_validate_mla_transfer_audit(
            runtime_metrics,
            expected_request_ids=expected_request_ids,
            expected_tp_size=tensor_parallel_size,
            expected_device_cache_enabled=(
                transfer_audit_device_cache_enabled
            ),
            expected_device_max_bytes=transfer_audit_device_max_bytes,
            allowed_request_ids=transfer_audit_allowed_request_ids,
        )
    else:
        transfer_audit = dict(
            runtime_metrics.get("mla_off_transfer_audit", {})
        )
        transfer_audit["required"] = False
        transfer_audit["pass"] = None
    missing_request_forwards = sorted(
        request_id
        for request_id in expected_request_ids
        if not request_forwards.get(request_id)
    )
    missing_forward_layers = {}
    unexpected_forward_layers = {}
    invalid_forwards = {}
    position_coverage_errors = {}
    row_geometry_errors = {}
    full_local_sanity_errors = {}
    scoped_reused_rows = 0
    scoped_online_local_rows = 0
    scoped_online_global_rows = 0
    scoped_reused_rows_by_request = Counter()
    scoped_metric_samples = 0
    observed_forward_count = 0
    minimum_expected_forward_count = 0
    for request_id in sorted(expected_request_ids):
        intervals = []
        for forward_id in request_forwards.get(request_id, ()):
            observed_forward_count += 1
            key = f"{request_id}/{forward_id}"
            manifest = forward_manifests.get(request_id, {}).get(forward_id)
            reasons = []
            if manifest is None:
                reasons.append("missing_manifest")
            else:
                # This benchmark sends one synchronous request at a time.
                # MIXED can contain rows from multiple scheduler requests, and
                # a single min/max span cannot attribute those rows exactly.
                if manifest.get("forward_mode") != "extend":
                    reasons.append("non_single_request_extend_forward")
                if int(manifest.get("q_rows", 0)) <= 0:
                    reasons.append("nonpositive_q_rows")
                if int(manifest.get("q_rows", 0)) > max_q_rows_per_forward:
                    reasons.append("q_rows_exceeds_chunked_prefill_limit")
                if manifest.get("plan_mode") != "restore":
                    reasons.append("non_restore_plan_mode")
                position_contiguous = manifest.get("position_contiguous") is True
                if not position_contiguous and not allow_sparse_positions:
                    reasons.append("positions_not_contiguous")
                position_start = int(manifest.get("position_start", -1))
                position_end = int(manifest.get("position_end", -1))
                q_rows = int(manifest.get("q_rows", 0))
                valid_span = _ih_valid_forward_position_span(
                    q_rows=q_rows,
                    position_start=position_start,
                    position_end=position_end,
                    position_contiguous=position_contiguous,
                    allow_sparse_positions=allow_sparse_positions,
                )
                if not valid_span:
                    reasons.append("invalid_position_span")
                else:
                    intervals.append((position_start, position_end, forward_id))
            if reasons:
                invalid_forwards[key] = reasons
            observed_layers = set(
                forward_layers.get(request_id, {}).get(forward_id, ())
            )
            missing = expected_layer_set - observed_layers
            if missing:
                missing_forward_layers[key] = sorted(missing)
            unexpected = observed_layers - expected_layer_set
            if unexpected:
                unexpected_forward_layers[key] = sorted(unexpected)
            if manifest is None:
                continue
            q_rows = int(manifest.get("q_rows", 0))
            for layer_id in expected_layer_ids:
                record = forward_metric_rows.get(request_id, {}).get(
                    forward_id, {}
                ).get(str(layer_id))
                if record is None:
                    continue
                counts = expected_head_counts_by_layer.get(
                    str(layer_id),
                    expected_head_counts_by_layer.get(layer_id),
                )
                if not isinstance(counts, dict):
                    row_geometry_errors[f"{key}/layer={layer_id}"] = [
                        "missing_expected_head_counts"
                    ]
                    continue
                local_heads = int(counts.get("local", -1))
                global_heads = int(counts.get("global", -1))
                reused_rows = int(record["reused_local_head_rows"])
                online_local_rows = int(record["online_local_head_rows"])
                online_global_rows = int(record["online_global_head_rows"])
                geometry_reasons = []
                if local_heads <= 0 or global_heads < 0:
                    geometry_reasons.append("invalid_expected_head_counts")
                if reused_rows + online_local_rows != q_rows * local_heads:
                    geometry_reasons.append("local_head_rows_do_not_partition")
                if online_global_rows != q_rows * global_heads:
                    geometry_reasons.append("global_head_rows_mismatch")
                if local_heads > 0 and (
                    reused_rows % local_heads != 0
                    or online_local_rows % local_heads != 0
                ):
                    geometry_reasons.append("local_head_rows_not_row_aligned")
                has_full_local_status = layer_id in set(
                    full_local_forward_layers.get(request_id, {})
                    .get(forward_id, ())
                )
                if reused_rows == 0 and not has_full_local_status:
                    geometry_reasons.append("missing_full_local_status")
                if reused_rows > 0 and has_full_local_status:
                    geometry_reasons.append("unexpected_full_local_status")
                if geometry_reasons:
                    row_geometry_errors[f"{key}/layer={layer_id}"] = (
                        geometry_reasons
                    )
                scoped_reused_rows += reused_rows
                scoped_reused_rows_by_request[request_id] += reused_rows
                scoped_online_local_rows += online_local_rows
                scoped_online_global_rows += online_global_rows
                scoped_metric_samples += 1
        coverage_reasons = []
        expected_q_rows = expected_q_rows_by_request.get(request_id)
        try:
            expected_q_rows = int(expected_q_rows)
        except (TypeError, ValueError):
            expected_q_rows = -1
        if expected_q_rows <= 0:
            coverage_reasons.append("missing_expected_q_rows")
        elif max_q_rows_per_forward > 0:
            minimum_expected_forward_count += (
                expected_q_rows + max_q_rows_per_forward - 1
            ) // max_q_rows_per_forward
        expected_position_start = int(
            expected_position_start_by_request.get(request_id, 0)
        )
        if expected_position_start < 0:
            coverage_reasons.append("negative_expected_position_start")
            expected_position_start = 0
        cursor = expected_position_start
        for position_start, position_end, forward_id in sorted(intervals):
            if position_start != cursor:
                coverage_reasons.append(
                    f"forward={forward_id}:expected_start={cursor}:"
                    f"observed_start={position_start}"
                )
            cursor = max(cursor, position_end)
        expected_position_end = expected_position_start + max(
            0, expected_q_rows
        )
        if expected_q_rows > 0 and cursor != expected_position_end:
            coverage_reasons.append(
                f"expected_end={expected_position_end}:observed_end={cursor}"
            )
        if coverage_reasons:
            position_coverage_errors[request_id] = coverage_reasons
    observed_request_layers = runtime_metrics.get("mla_request_layers", {})
    missing_request_layers = {
        request_id: sorted(
            expected_layer_set - set(observed_request_layers.get(request_id, ()))
        )
        for request_id in sorted(expected_request_ids)
        if expected_layer_set - set(observed_request_layers.get(request_id, ()))
    }
    measured_fallback_request_ids = sorted(
        expected_request_ids
        & set(runtime_metrics.get("mla_fallback_request_ids", ()))
    )
    evidence_error_request_ids = sorted(
        expected_request_ids
        & set(runtime_metrics.get("mla_evidence_error_request_ids", ()))
    )
    requests_without_reuse = sorted(
        request_id
        for request_id in expected_request_ids
        if scoped_reused_rows_by_request[request_id] <= 0
    )
    evidence_failed_request_ids = set(missing_request_forwards)
    evidence_failed_request_ids.update(missing_request_layers)
    evidence_failed_request_ids.update(measured_fallback_request_ids)
    evidence_failed_request_ids.update(evidence_error_request_ids)
    if transfer_audit_required:
        evidence_failed_request_ids.update(
            transfer_audit.get("failed_request_ids", ())
        )
    if compact_woa_required:
        evidence_failed_request_ids.update(
            compact_woa_evidence.get("failed_request_ids", ())
        )
    if composite_receipt_required:
        evidence_failed_request_ids.update(
            composite_receipts.get("failed_request_ids", ())
        )
    for key in (
        *missing_forward_layers,
        *unexpected_forward_layers,
        *invalid_forwards,
    ):
        evidence_failed_request_ids.add(key.split("/", 1)[0])
    evidence_failed_request_ids.update(position_coverage_errors)
    for key in row_geometry_errors:
        evidence_failed_request_ids.add(key.split("/", 1)[0])
    formal_failed_request_ids = set(evidence_failed_request_ids)
    formal_failed_request_ids.update(requests_without_reuse)
    minimum_samples = observed_forward_count * len(expected_layer_ids)
    scoped_total_rows = (
        scoped_reused_rows + scoped_online_local_rows + scoped_online_global_rows
    )
    scoped_head_row_saving = (
        scoped_reused_rows / scoped_total_rows if scoped_total_rows else None
    )
    full_model_head_row_denominator = None
    full_model_head_row_saving = None
    full_model_reused_head_rows = None
    if (
        num_model_layers is not None
        or num_attention_heads is not None
        or tensor_parallel_size is not None
    ):
        if (
            type(num_model_layers) is not int
            or num_model_layers <= 0
            or type(num_attention_heads) is not int
            or num_attention_heads <= 0
            or type(tensor_parallel_size) is not int
            or tensor_parallel_size <= 0
            or num_attention_heads % tensor_parallel_size != 0
        ):
            raise ValueError(
                "full-model head-row geometry requires positive integer "
                "num_model_layers, num_attention_heads, tensor_parallel_size, "
                "and divisible attention heads"
            )
        full_q_rows = 0
        for request_id in expected_request_ids:
            q_rows = expected_q_rows_by_request.get(request_id)
            if type(q_rows) is not int or q_rows <= 0:
                full_q_rows = 0
                break
            full_q_rows += q_rows
        if full_q_rows > 0:
            # Runtime metric rows are emitted for one TP-local contiguous head
            # shard.  Capacity preflight rejects TP-asymmetric head policies,
            # so multiplying the directly observed rank-local numerator by TP
            # reconstructs the exact logical 128-head numerator; without this
            # factor an all-local TP8 policy could never exceed 12.5%.
            full_model_reused_head_rows = (
                scoped_reused_rows * tensor_parallel_size
            )
            full_model_head_row_denominator = (
                num_model_layers * num_attention_heads * full_q_rows
            )
            full_model_head_row_saving = (
                full_model_reused_head_rows
                / full_model_head_row_denominator
            )
    has_expected_global_heads = any(
        int(
            (
                expected_head_counts_by_layer.get(
                    str(layer_id),
                    expected_head_counts_by_layer.get(layer_id, {}),
                )
                or {}
            ).get("global", 0)
        )
        > 0
        for layer_id in expected_layer_ids
    )
    evidence_pass = (
        len(expected_request_ids) == int(expected_request_count)
        and set(expected_q_rows_by_request) == expected_request_ids
        and bool(expected_layer_ids)
        and max_q_rows_per_forward > 0
        and observed_forward_count >= minimum_expected_forward_count
        and not missing_request_forwards
        and not missing_request_layers
        and not missing_forward_layers
        and not unexpected_forward_layers
        and not invalid_forwards
        and not position_coverage_errors
        and not row_geometry_errors
        and not measured_fallback_request_ids
        and not evidence_error_request_ids
        and not runtime_metrics.get("mla_log_read_errors")
        and (
            not transfer_audit_required
            or bool(transfer_audit.get("pass", False))
        )
        and (
            compact_woa_expected_mode == "not_applicable"
            or bool(compact_woa_evidence.get("pass", False))
        )
        and (
            not composite_receipt_required
            or bool(composite_receipts.get("pass", False))
        )
        and scoped_metric_samples == minimum_samples
        and scoped_online_local_rows > 0
        and (
            not has_expected_global_heads or scoped_online_global_rows > 0
        )
        and runtime_metrics["fallbacks"] == 0
    )
    runtime_pass = (
        evidence_pass and not requests_without_reuse and scoped_reused_rows > 0
    )

    if full_local_sanity:
        for request_id in sorted(expected_request_ids):
            for forward_id in sorted(request_forwards.get(request_id, ())):
                for layer_id in expected_layer_ids:
                    key = f"{request_id}/{forward_id}/layer={layer_id}"
                    reasons = []
                    reason = (
                        full_local_forward_reasons.get(request_id, {})
                        .get(forward_id, {})
                        .get(str(layer_id))
                    )
                    if reason != "refresh_layer":
                        reasons.append(
                            "missing_refresh_layer_status"
                            if reason is None
                            else f"unexpected_full_local_reason={reason}"
                        )
                    record = (
                        forward_metric_rows.get(request_id, {})
                        .get(forward_id, {})
                        .get(str(layer_id))
                    )
                    if record is None:
                        reasons.append("missing_metric")
                    elif int(record["reused_local_head_rows"]) != 0:
                        reasons.append("nonzero_reused_rows")
                    if reasons:
                        full_local_sanity_errors[key] = reasons
    full_local_sanity_failed_request_ids = set(evidence_failed_request_ids)
    full_local_sanity_failed_request_ids.update(
        request_id
        for request_id in expected_request_ids
        if request_id not in requests_without_reuse
    )
    for key in full_local_sanity_errors:
        full_local_sanity_failed_request_ids.add(key.split("/", 1)[0])
    full_local_sanity_pass = (
        evidence_pass
        and full_local_sanity
        and scoped_reused_rows == 0
        and set(requests_without_reuse) == expected_request_ids
        and not full_local_sanity_errors
    )
    return {
        "minimum_mla_off_samples": minimum_samples,
        "measured_mla_off_samples": scoped_metric_samples,
        "expected_mla_forward_count": minimum_expected_forward_count,
        "observed_mla_forward_count": observed_forward_count,
        "missing_mla_request_forwards": missing_request_forwards,
        "missing_mla_request_layers": missing_request_layers,
        "missing_mla_forward_layers": missing_forward_layers,
        "unexpected_mla_forward_layers": unexpected_forward_layers,
        "invalid_mla_forwards": invalid_forwards,
        "mla_position_coverage_errors": position_coverage_errors,
        "mla_row_geometry_errors": row_geometry_errors,
        "measured_mla_fallback_request_ids": measured_fallback_request_ids,
        "measured_mla_evidence_error_request_ids": evidence_error_request_ids,
        "mla_requests_without_reuse": requests_without_reuse,
        "measured_reused_local_head_rows": scoped_reused_rows,
        "measured_online_local_head_rows": scoped_online_local_rows,
        "measured_online_global_head_rows": scoped_online_global_rows,
        "measured_mla_head_row_saving": scoped_head_row_saving,
        "measured_scoped_mla_head_row_saving": scoped_head_row_saving,
        "measured_full_model_mla_head_row_saving": (
            full_model_head_row_saving
        ),
        "measured_full_model_reused_local_head_rows": (
            full_model_reused_head_rows
        ),
        "full_model_mla_head_row_denominator": (
            full_model_head_row_denominator
        ),
        "full_model_num_layers": num_model_layers,
        "full_model_num_attention_heads": num_attention_heads,
        "full_model_tensor_parallel_size": tensor_parallel_size,
        "mla_off_transfer_audit": transfer_audit,
        "compact_woa_evidence": compact_woa_evidence,
        "mla_off_composite_receipts": composite_receipts,
        # Compatibility alias: this remains the formal reuse-gate failure set.
        "failed_mla_request_ids": sorted(formal_failed_request_ids),
        "runtime_evidence_failed_request_ids": sorted(evidence_failed_request_ids),
        "formal_failed_mla_request_ids": sorted(formal_failed_request_ids),
        "runtime_evidence_pass": evidence_pass,
        "formal_reuse_pass": runtime_pass,
        "full_local_sanity_requested": full_local_sanity,
        "full_local_sanity_errors": full_local_sanity_errors,
        "full_local_sanity_failed_request_ids": (
            sorted(full_local_sanity_failed_request_ids) if full_local_sanity else []
        ),
        "full_local_sanity_pass": (
            full_local_sanity_pass if full_local_sanity else None
        ),
        "pass": runtime_pass,
    }


def _ih_mla_off_capacity_geometry(total_segment_tokens: int) -> dict:
    """Derive the exact v1 per-rank artifact working set from server policy."""

    from sglang.srt.layers.attention.redknot.dsv4_mla_offload import (
        MLA_OFF_COMBINED_ROW_SPARSE_FULL_PROFILE,
        MLA_OFF_COMBINED_ROW_SPARSE_PROFILE,
        MLA_OFF_INDEPENDENT_RELOCATION_PROFILE,
        MLA_OFF_FORMAT_VERSION,
        MLA_OFF_TOKEN_BYTES_PER_ROW,
        MLA_OFF_TRANSFER_AUDIT_SCHEMA,
        MLA_OFF_TRANSFER_BYTE_SEMANTICS,
        mla_off_expected_bytes,
        mla_off_layer_bytes_per_row,
    )
    expected_profile = (
        (
            MLA_OFF_COMBINED_ROW_SPARSE_FULL_PROFILE
            if IH_MLA_OFF_DIAGNOSTIC_ABLATION == "full"
            else MLA_OFF_COMBINED_ROW_SPARSE_PROFILE
        )
        if IH_COMBINED_HEADSPLIT_ROW_SPARSE
        else MLA_OFF_INDEPENDENT_RELOCATION_PROFILE
    )
    if IH_MLA_OFF_EXECUTION_PROFILE != expected_profile:
        raise ValueError("benchmark/runtime MLA-off execution profiles differ")
    from sglang.srt.layers.attention.deepseek_v4_backend import SWA_WINDOW
    from sglang.srt.layers.attention.redknot.head_config import (
        DEFAULT_LOCAL_WINDOW,
    )

    config_path = Path(MODEL_PATH) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"MLA-off capacity preflight needs {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    from sglang.srt.layers.attention.redknot.pro0813.profile import (
        inspect_pro0813_config,
    )

    geometry = inspect_pro0813_config(model_config, tp_size=PRO0813_TP_SIZE)
    model_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if model_config_sha256 != PRO0813_CONFIG_SHA256:
        raise ValueError("benchmark model config is not the official Pro-0813 file")
    if (
        geometry.variant != PRO0813_VARIANT
        or geometry.geometry_digest != PRO0813_GEOMETRY_DIGEST
    ):
        raise ValueError("benchmark Pro-0813 variant/geometry digest drifted")

    def required_int(name: str) -> int:
        try:
            value = int(model_config[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"MLA-off model config has invalid {name}: {error}"
            ) from error
        if value <= 0:
            raise ValueError(f"MLA-off model config requires positive {name}")
        return value

    num_layers = required_int("num_hidden_layers")
    num_heads = required_int("num_attention_heads")
    physical_kv_heads = required_int("num_key_value_heads")
    o_groups = required_int("o_groups")
    o_lora_rank = required_int("o_lora_rank")
    # Match RedKnotMLAAttnBackend exactly. DeepSeek-V4's paged pool may reserve
    # a wider physical row, but the executable SWA contract is SWA_WINDOW.
    swa_capacity = int(SWA_WINDOW)
    if swa_capacity <= 0:
        raise ValueError("MLA-off model SWA capacity must be positive")
    if physical_kv_heads != 1:
        raise ValueError("MLA-off requires one physical latent KV stream")
    if IH_TP_SIZE <= 0 or num_heads % IH_TP_SIZE != 0:
        raise ValueError(
            f"MLA-off needs num_attention_heads={num_heads} divisible by "
            f"TP={IH_TP_SIZE}"
        )
    if o_groups % IH_TP_SIZE != 0:
        raise ValueError(
            f"MLA-off needs o_groups={o_groups} divisible by TP={IH_TP_SIZE}"
        )

    head_cfg_path = HEAD_CFG_PATH
    if head_cfg_path:
        with open(head_cfg_path, "r", encoding="utf-8") as handle:
            head_cfg = json.load(handle)
        head_class = head_cfg.get("mla_head_classification")
        head_distance = head_cfg.get("mla_head_max_distance")
        # Runtime-owned fences override JSON metadata; external profiling may
        # classify only the middle layers and cannot weaken layers 0..2/58..60.
        dense_prefix = MLA_DENSE_PREFIX
        dense_suffix = MLA_DENSE_SUFFIX
        local_default_window = int(
            head_cfg.get("local_default_window", DEFAULT_LOCAL_WINDOW)
        )
        if int(head_cfg.get("num_layers", -1)) != num_layers:
            raise ValueError("MLA-off head config layer count differs from model")
        if int(head_cfg.get("num_attention_heads", -1)) != num_heads:
            raise ValueError("MLA-off head config head count differs from model")
        if int(head_cfg.get("physical_kv_heads", 1)) != 1:
            raise ValueError("MLA-off head config requires physical_kv_heads=1")
        if not isinstance(head_class, list) or not isinstance(head_distance, list):
            raise ValueError("MLA-off head config matrices are missing")
    else:
        dense_prefix = MLA_DENSE_PREFIX
        dense_suffix = MLA_DENSE_SUFFIX
        local_default_window = MLA_LOCAL_WINDOW
        global_head_stride = max(1, MLA_GLOBAL_HEAD_STRIDE)
        head_class = []
        head_distance = []
        for layer_id in range(num_layers):
            force_global = (
                layer_id < dense_prefix
                or layer_id >= num_layers - dense_suffix
                or (
                MLA_GLOBAL_LAYER_STRIDE > 0
                and layer_id % MLA_GLOBAL_LAYER_STRIDE == 0
                )
            )
            class_row = []
            distance_row = []
            for head_id in range(num_heads):
                is_local = not force_global and (
                    head_id % global_head_stride != 0
                )
                class_row.append("local" if is_local else "global")
                distance_row.append(local_default_window if is_local else -1)
            head_class.append(class_row)
            head_distance.append(distance_row)

    if len(head_class) != num_layers or len(head_distance) != num_layers:
        raise ValueError("MLA-off head config matrix layer count is invalid")
    if not 0 <= dense_prefix <= num_layers:
        raise ValueError("MLA-off dense-prefix layer count is invalid")
    if not 0 <= dense_suffix <= num_layers - dense_prefix:
        raise ValueError("MLA-off dense-suffix layer count is invalid")
    if (
        num_layers != PRO0813_NUM_LAYERS
        or num_heads != PRO0813_NUM_HEADS
        or required_int("index_topk") != PRO0813_INDEX_TOPK
        or dense_prefix != 3
        or dense_suffix != 3
    ):
        raise ValueError(
            "pure Pro-0813 headsplit capacity requires 61 layers, 128 heads, "
            "Indexer Top-K 1024, and 3/3 dense fences"
        )
    if local_default_window <= 0:
        raise ValueError("MLA-off local default window must be positive")
    heads_per_rank = num_heads // IH_TP_SIZE
    local_layers_by_rank = [[] for _ in range(IH_TP_SIZE)]
    asymmetric_layers = []
    head_count_asymmetric_layers = []
    head_counts_by_layer = {}
    effective_policy_payload = []
    for layer_id in range(num_layers):
        if (
            len(head_class[layer_id]) != num_heads
            or len(head_distance[layer_id]) != num_heads
        ):
            raise ValueError(
                f"MLA-off head config layer {layer_id} width is invalid"
            )
        unknown_head_types = set(map(str, head_class[layer_id])) - {
            "local",
            "global",
            "dense",
        }
        if unknown_head_types:
            raise ValueError(
                f"MLA-off head config layer {layer_id} has unknown types "
                f"{sorted(unknown_head_types)}"
            )
        local_by_window = {}
        global_heads = []
        promoted_heads = []
        for head_id, head_type in enumerate(head_class[layer_id]):
            if (
                layer_id < dense_prefix
                or layer_id >= num_layers - dense_suffix
                or str(head_type) != "local"
            ):
                global_heads.append(head_id)
                continue
            distance = int(head_distance[layer_id][head_id])
            window = distance if distance > 0 else local_default_window
            if window > swa_capacity:
                global_heads.append(head_id)
                promoted_heads.append((head_id, window))
                continue
            if window <= 0:
                raise ValueError(
                    f"MLA-off layer {layer_id} head {head_id} has invalid window"
                )
            local_by_window.setdefault(window, []).append(head_id)
        local_groups = tuple(
            (window, tuple(local_by_window[window]))
            for window in sorted(local_by_window)
        )
        if 3 <= layer_id < 58 and (not local_groups or not global_heads):
            raise ValueError(
                "pure MLA headsplit middle layers require both local and global "
                f"heads; layer={layer_id}"
            )
        effective_policy_payload.append(
            {
                "local_groups": local_groups,
                "global_heads": tuple(global_heads),
                "promoted_heads": tuple(promoted_heads),
            }
        )
        executable_local = {
            head for _, heads in local_groups for head in heads
        }
        if not executable_local:
            continue
        rank_coverage = []
        rank_head_counts = []
        for rank in range(IH_TP_SIZE):
            rank_start = rank * heads_per_rank
            rank_end = rank_start + heads_per_rank
            owned_heads = set(range(rank_start, rank_end))
            local_head_count = len(executable_local.intersection(owned_heads))
            global_head_count = heads_per_rank - local_head_count
            rank_head_counts.append(
                {
                    "local": local_head_count,
                    "global": global_head_count,
                }
            )
            covered = local_head_count > 0
            rank_coverage.append(covered)
            if covered:
                local_layers_by_rank[rank].append(layer_id)
        head_counts_by_layer[layer_id] = rank_head_counts
        if not all(rank_coverage):
            asymmetric_layers.append(layer_id)
        if len(
            {(counts["local"], counts["global"]) for counts in rank_head_counts}
        ) != 1:
            head_count_asymmetric_layers.append(layer_id)
    if asymmetric_layers:
        raise ValueError(
            "MLA-off head policy is TP-asymmetric at layers "
            f"{tuple(asymmetric_layers)}"
        )
    if head_count_asymmetric_layers:
        raise ValueError(
            "MLA-off benchmark requires TP-symmetric logical head counts; "
            f"asymmetric layers={tuple(head_count_asymmetric_layers)}"
        )
    if not any(local_layers_by_rank):
        raise ValueError("MLA-off head policy has no executable local heads")
    expected_offline_layers = list(PRO0813_REUSABLE_LAYER_IDS)
    if any(layer_ids != expected_offline_layers for layer_ids in local_layers_by_rank):
        raise ValueError(
            "pure Pro-0813 headsplit requires every TP rank to cache exactly "
            "layers 3..57"
        )

    total_segment_tokens = int(total_segment_tokens)
    if total_segment_tokens <= 0:
        raise ValueError("MLA-off total offline tokens must be positive")
    groups_per_rank = o_groups // IH_TP_SIZE
    layer_bytes = mla_off_layer_bytes_per_row(
        groups_per_rank, o_lora_rank
    )
    projected_bytes_by_rank = [
        mla_off_expected_bytes(
            length=total_segment_tokens,
            local_layer_count=len(layer_ids),
            num_output_groups=groups_per_rank,
            o_lora_rank=o_lora_rank,
        )
        for layer_ids in local_layers_by_rank
    ]
    exact_pro_zoff_bytes = (
        len(PRO0813_REUSABLE_LAYER_IDS)
        * total_segment_tokens
        * PRO0813_O_GROUPS_PER_TP_RANK
        * PRO0813_O_LORA_RANK
        * 2
    )
    if projected_bytes_by_rank != [exact_pro_zoff_bytes] * PRO0813_TP_SIZE:
        raise ValueError(
            "Pro-0813 z_off projection differs from "
            "55 * tokens * 2 groups/rank * 1024 * 2 BF16 bytes"
        )
    if IH_MLA_OFF_MAX_BYTES < exact_pro_zoff_bytes:
        raise ValueError(
            "Pro-0813 z_off cap is below the exact per-rank projection: "
            f"cap={IH_MLA_OFF_MAX_BYTES} projected={exact_pro_zoff_bytes}"
        )
    head_config_sha256 = (
        hashlib.sha256(Path(head_cfg_path).read_bytes()).hexdigest()
        if head_cfg_path
        else ""
    )
    performance_claim_status = (
        "qualification_only_claim_ineligible"
        if IH_MLA_OFF_QUALIFICATION_ONLY
        else "unverified"
    )
    effective_policy_hash = hashlib.sha256(
        json.dumps(
            {
                "redknot_variant": PRO0813_VARIANT,
                "redknot_geometry_digest": PRO0813_GEOMETRY_DIGEST,
                "execution_profile": IH_MLA_OFF_EXECUTION_PROFILE,
                "dense_prefix_layers": MLA_DENSE_PREFIX,
                "dense_suffix_layers": MLA_DENSE_SUFFIX,
                "dense_layer_ids": list(PRO0813_DENSE_LAYER_IDS),
                "offline_online_layer_ids": list(PRO0813_REUSABLE_LAYER_IDS),
                "selected_row_enabled": IH_COMBINED_HEADSPLIT_ROW_SPARSE,
                "indexer_hot_enabled": IH_COMBINED_HEADSPLIT_ROW_SPARSE,
                "disable_radix_cache": not IH_PREFIX_MATERIALIZATION,
                "prefix_materialization": IH_PREFIX_MATERIALIZATION,
                "radix_eviction_policy": IH_RADIX_EVICTION_POLICY,
                "q_projection_scope": (
                    "q_a_checkpoint_selected_rows_headsplit_v1"
                    if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                    else "q_a_full_rows_native_dsv4_fullscope_skip0_v1"
                ),
                "head_scope_policy": IH_MLA_OFF_HEAD_SCOPE_POLICY,
                "online_global_attention_impl": IH_MLA_OFF_GLOBAL_ATTN_IMPL,
                "geometry_template_cache": (
                    IH_MLA_OFF_GEOMETRY_TEMPLATE_CACHE
                ),
                "kv_projection_scope": (
                    "shared_clean_rows_gpu_restore_dirty_rows_wkv_v1"
                ),
                "compressor_projection_scope": (
                    "shared_clean_blocks_gpu_restore_dirty_islands_online_v1"
                ),
                "shared_latent_restore_scope": (
                    "persistent_gpu_layer_group_pipeline_v1"
                    if IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS
                    else "persistent_gpu_ragged_fused_scatter_v1"
                ),
                "restore_pipeline_group_layers": int(
                    IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS
                ),
                "tp_commit_scope": (
                    "composite_forward_prepare_and_full_layer_final_v6"
                ),
                "wo_a_projection_scope": "true_head_column_slices_v1",
                "performance_claim_status": performance_claim_status,
                "three_way_closure": THREE_WAY_CLOSURE,
                "token_sparse_ffn_enabled": SPARSE_FFN,
                "token_sparse_ffn_importance": FFN_IMPORTANCE,
                "token_sparse_ffn_mass_thresh": FFN_MASS,
                "token_sparse_ffn_mass_thresh_deep": FFN_MASS_DEEP,
                "token_sparse_ffn_min_full_ratio": FFN_MIN_FULL_RATIO,
                "token_sparse_ffn_max_full_ratio": FFN_MAX_FULL_RATIO,
                "token_sparse_ffn_dense_suffix_layers": FFN_DENSE_SUFFIX_LAYERS,
                "token_sparse_ffn_boundary_tokens": FFN_BOUNDARY_TOKENS,
                "reuse_heads_full_scope": IH_REUSE_HEADS_FULL_SCOPE,
                "certified_max_context_tokens": (
                    IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS
                ),
                "qualification_only": IH_MLA_OFF_QUALIFICATION_ONLY,
                "qualification_max_context_tokens": (
                    IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS
                ),
                "layers": effective_policy_payload,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    rank0_owned_heads = set(range(heads_per_rank))
    rank0_global_layer_ids = [
        layer_id
        for layer_id in local_layers_by_rank[0]
        if bool(
            rank0_owned_heads.intersection(
                effective_policy_payload[layer_id]["global_heads"]
            )
        )
    ]
    server_policy_manifest = {
        "format": "redknot_mla_server_policy_v4",
        "backend_ready": True,
        "jit_rmsnorm": True,
        "accelerator_device_count": 8,
        "accelerator_capability": [10, 3],
        "server_instance_nonce": IH_SERVER_INSTANCE_NONCE,
        "port": IH_PORT,
        "model_path": MODEL_PATH,
        "model_config_sha256": model_config_sha256,
        "head_config_path": head_cfg_path,
        "head_config_sha256": head_config_sha256,
        "tp_size": IH_TP_SIZE,
        "mla_pass_mode": MLA_PASS_MODE,
        "reuse_heads_full_scope": IH_REUSE_HEADS_FULL_SCOPE,
        "mla_off_certified_max_context_tokens": (
            IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS
        ),
        "mla_off_qualification_only": IH_MLA_OFF_QUALIFICATION_ONLY,
        "mla_off_qualification_max_context_tokens": (
            IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS
        ),
        "mla_off_effective_restore_max_context_tokens": (
            IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS
            if IH_MLA_OFF_QUALIFICATION_ONLY
            else IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS
        ),
        "dense_prefix_layers": MLA_DENSE_PREFIX,
        "dense_suffix_layers": MLA_DENSE_SUFFIX,
        "execution_profile": IH_MLA_OFF_EXECUTION_PROFILE,
        "dense_layer_ids": list(PRO0813_DENSE_LAYER_IDS),
        "offline_online_layer_ids": list(PRO0813_REUSABLE_LAYER_IDS),
        "selected_row_enabled": IH_COMBINED_HEADSPLIT_ROW_SPARSE,
        "indexer_hot_enabled": IH_COMBINED_HEADSPLIT_ROW_SPARSE,
        "disable_radix_cache": not IH_PREFIX_MATERIALIZATION,
        "prefix_materialization": IH_PREFIX_MATERIALIZATION,
        "radix_eviction_policy": IH_RADIX_EVICTION_POLICY,
        "q_projection_scope": (
            "q_a_checkpoint_selected_rows_headsplit_v1"
            if IH_COMBINED_HEADSPLIT_ROW_SPARSE
            else "q_a_full_rows_native_dsv4_fullscope_skip0_v1"
        ),
        "head_scope_policy": IH_MLA_OFF_HEAD_SCOPE_POLICY,
        "online_global_attention_impl": IH_MLA_OFF_GLOBAL_ATTN_IMPL,
        "geometry_template_cache": IH_MLA_OFF_GEOMETRY_TEMPLATE_CACHE,
        "kv_projection_scope": (
            "shared_clean_rows_gpu_restore_dirty_rows_wkv_v1"
        ),
        "compressor_projection_scope": (
            "shared_clean_blocks_gpu_restore_dirty_islands_online_v1"
        ),
        "shared_latent_restore_scope": (
            "persistent_gpu_layer_group_pipeline_v1"
            if IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS
            else "persistent_gpu_ragged_fused_scatter_v1"
        ),
        "restore_pipeline_group_layers": int(
            IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS
        ),
        "tp_commit_scope": (
            "composite_forward_prepare_and_full_layer_final_v6"
        ),
        "wo_a_projection_scope": "true_head_column_slices_v1",
        "performance_claim_status": performance_claim_status,
        "local_window": MLA_LOCAL_WINDOW,
        "global_head_stride": MLA_GLOBAL_HEAD_STRIDE,
        "global_layer_stride": MLA_GLOBAL_LAYER_STRIDE,
        "mla_off_max_bytes": IH_MLA_OFF_MAX_BYTES,
        "mla_offload_enabled": True,
        "mla_off_compact_woa_enabled": IH_MLA_OFF_COMPACT_WOA,
        "redknot_v4_mode": "aggressive",
        "three_way_closure": THREE_WAY_CLOSURE,
        "token_sparse_ffn_enabled": SPARSE_FFN,
        "token_sparse_ffn_importance": FFN_IMPORTANCE,
        "token_sparse_ffn_mass_thresh": FFN_MASS,
        "token_sparse_ffn_mass_thresh_deep": FFN_MASS_DEEP,
        "token_sparse_ffn_min_full_ratio": FFN_MIN_FULL_RATIO,
        "token_sparse_ffn_max_full_ratio": FFN_MAX_FULL_RATIO,
        "token_sparse_ffn_dense_suffix_layers": FFN_DENSE_SUFFIX_LAYERS,
        "token_sparse_ffn_boundary_tokens": FFN_BOUNDARY_TOKENS,
        "attention_backend": "redknot_mla",
        "dp_size": 1,
        "dp_attention": False,
        "cp_size": 1,
        "pp_size": 1,
        "swa_capacity": swa_capacity,
        "o_groups": o_groups,
        "o_lora_rank": o_lora_rank,
        "output_groups_per_rank": groups_per_rank,
        "mla_off_format_version": int(MLA_OFF_FORMAT_VERSION),
        "mla_off_storage_dtype": "bfloat16",
        "mla_off_transfer_audit_format": MLA_OFF_TRANSFER_AUDIT_SCHEMA,
        "mla_off_transfer_byte_semantics": MLA_OFF_TRANSFER_BYTE_SEMANTICS,
        "mla_off_token_bytes_per_row": MLA_OFF_TOKEN_BYTES_PER_ROW,
        "mla_off_layer_bytes_per_row": layer_bytes,
        "effective_head_policy_hash": effective_policy_hash,
        "runtime_local_layer_ids": local_layers_by_rank[0],
    }
    if not IH_NO_LAUNCH:
        # The owned launcher injects these exact scheduler limits. External
        # endpoints must report their own resolved limits in the live manifest.
        server_policy_manifest.update(
            {
                "chunked_prefill_size": IH_CHUNK_TOKENS,
                "max_prefill_tokens": max(
                    IH_CHUNK_TOKENS, IH_MERGED_PREFILL_TOKENS
                ),
            }
        )
    return {
        "accelerator_contract": "8x NVIDIA B300 / Blackwell SM103",
        "redknot_variant": PRO0813_VARIANT,
        "redknot_geometry_digest": PRO0813_GEOMETRY_DIGEST,
        "official_config_sha256": PRO0813_CONFIG_SHA256,
        "index_topk": PRO0813_INDEX_TOPK,
        "dense_layer_ids": list(PRO0813_DENSE_LAYER_IDS),
        "reusable_layer_ids": list(PRO0813_REUSABLE_LAYER_IDS),
        "zoff_formula": "55 * tokens * 2 * 1024 * 2",
        "zoff_projected_bytes_per_rank": exact_pro_zoff_bytes,
        "tp_size": IH_TP_SIZE,
        "num_model_layers": num_layers,
        "num_attention_heads": num_heads,
        "groups_per_rank": groups_per_rank,
        "local_layers_by_rank": local_layers_by_rank,
        "local_layer_counts_by_rank": [
            len(layer_ids) for layer_ids in local_layers_by_rank
        ],
        "layer_bytes_per_token": layer_bytes,
        "token_identity_bytes_per_token": MLA_OFF_TOKEN_BYTES_PER_ROW,
        "bytes_per_token_by_rank": [
            value // total_segment_tokens for value in projected_bytes_by_rank
        ],
        "projected_bytes_by_rank": projected_bytes_by_rank,
        "projected_bytes_max_rank": max(projected_bytes_by_rank),
        "total_segment_tokens": total_segment_tokens,
        "rank0_global_layer_ids": rank0_global_layer_ids,
        "rank0_head_counts_by_layer": {
            str(layer_id): head_counts_by_layer[layer_id][0]
            for layer_id in local_layers_by_rank[0]
        },
        "server_policy_manifest": server_policy_manifest,
    }


def _ih_linux_process_identity(pid: int) -> tuple[str, int]:
    """Read process state and PID-reuse-resistant start time from Linux procfs."""

    pid = int(pid)
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing_paren = raw.rfind(")")
    if closing_paren < 0:
        raise ValueError(f"malformed /proc/{pid}/stat")
    fields = raw[closing_paren + 1 :].split()
    if len(fields) <= 19:
        raise ValueError(f"truncated /proc/{pid}/stat")
    state = fields[0]
    start_time_ticks = int(fields[19])
    if len(state) != 1 or start_time_ticks <= 0:
        raise ValueError(f"invalid /proc/{pid}/stat identity")
    return state, start_time_ticks


def _ih_verify_runtime_server_policy(
    capacity_geometry: dict, manifest_name: str
) -> dict:
    """Verify backend-effective state for this exact live server instance."""

    if not manifest_name:
        raise ValueError(
            "MLA-off cannot verify an external server; set "
            "REDKNOT_IH_SERVER_POLICY_MANIFEST and "
            "REDKNOT_IH_SERVER_INSTANCE_NONCE to the backend-ready manifest"
        )
    manifest_path = Path(manifest_name).expanduser().resolve()
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read external MLA-off server manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(actual, dict):
        raise ValueError("runtime MLA-off manifest must be a JSON object")
    expected = capacity_geometry["server_policy_manifest"]

    def json_value_equal_strict(expected_value, actual_value) -> bool:
        """Compare JSON values without Python's bool/int equality aliasing."""

        if type(actual_value) is not type(expected_value):
            return False
        if isinstance(expected_value, dict):
            return expected_value.keys() == actual_value.keys() and all(
                json_value_equal_strict(value, actual_value[key])
                for key, value in expected_value.items()
            )
        if isinstance(expected_value, list):
            return len(expected_value) == len(actual_value) and all(
                json_value_equal_strict(left, right)
                for left, right in zip(expected_value, actual_value)
            )
        return actual_value == expected_value

    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if key not in actual or not json_value_equal_strict(value, actual[key])
    }
    if mismatches:
        raise ValueError(
            "external MLA-off server policy differs from benchmark preflight: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    if actual.get("jit_rmsnorm") is not True:
        raise ValueError(
            "runtime MLA-off manifest requires the active JIT RMSNorm path"
        )
    accelerator_device_count = actual.get("accelerator_device_count")
    accelerator_name = actual.get("accelerator_name")
    accelerator_capability = actual.get("accelerator_capability")
    accelerator_device_names = actual.get("accelerator_device_names")
    accelerator_device_capabilities = actual.get(
        "accelerator_device_capabilities"
    )
    if accelerator_device_count != 8:
        raise ValueError("runtime MLA-off manifest requires exactly 8 GPUs")
    if (
        type(accelerator_device_names) is not list
        or len(accelerator_device_names) != 8
        or any(
            type(name) is not str or "B300" not in name
            for name in accelerator_device_names
        )
    ):
        raise ValueError(
            "runtime MLA-off manifest requires eight actual B300 device names"
        )
    if accelerator_device_capabilities != [[10, 3]] * 8:
        raise ValueError(
            "runtime MLA-off manifest requires compute capability 10.3 on all GPUs"
        )
    if (
        type(accelerator_name) is not str
        or accelerator_name != accelerator_device_names[0]
        or accelerator_capability != [10, 3]
    ):
        raise ValueError(
            "runtime MLA-off manifest accelerator summary differs from devices"
        )
    model_compat_hash = actual.get("model_compat_hash")
    if not isinstance(model_compat_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", model_compat_hash
    ):
        raise ValueError("runtime MLA-off manifest lacks model compatibility hash")

    if actual.get("batch_restore_provider_ready") is not True:
        raise ValueError(
            "runtime MLA-off manifest has no ready batch restore provider"
        )
    expected_radix_disabled = not IH_PREFIX_MATERIALIZATION
    if actual.get("disable_radix_cache") is not expected_radix_disabled:
        raise ValueError(
            "context-bound MLA radix-cache policy differs from the requested "
            "prefix-materialization mode"
        )
    if actual.get("radix_eviction_policy") != IH_RADIX_EVICTION_POLICY:
        raise ValueError(
            "runtime MLA-off radix eviction policy differs from the frozen "
            "benchmark policy"
        )
    provider_error = actual.get("batch_restore_provider_error")
    if type(provider_error) is not str or provider_error != "":
        raise ValueError(
            "runtime MLA-off manifest reports a batch restore provider error"
        )

    provider_token_pattern = re.compile(r"sha256:[0-9a-f]{64}")
    for token_name in (
        "batch_restore_provider_common_token",
        "batch_restore_provider_local_token",
    ):
        token = actual.get(token_name)
        if type(token) is not str or provider_token_pattern.fullmatch(token) is None:
            raise ValueError(
                f"runtime MLA-off manifest has invalid {token_name}"
            )

    oracle_evidence = actual.get("batch_restore_oracle_evidence")
    if not isinstance(oracle_evidence, dict):
        raise ValueError(
            "runtime MLA-off manifest lacks batch restore oracle evidence"
        )
    if oracle_evidence.get("strict_pass") is not True:
        raise ValueError(
            "runtime MLA-off batch restore oracle did not strictly pass"
        )

    source_evidence = {
        "kernel_source_sha256": (
            REPO
            / "python/sglang/srt/layers/attention/redknot/"
            "dsv4_shared_latent_batch_kernels.py"
        ),
        "oracle_source_sha256": (
            REPO
            / "python/sglang/srt/layers/attention/redknot/"
            "probe_dsv4_shared_latent_batch_kernels.py"
        ),
    }
    for evidence_name, source_path in source_evidence.items():
        recorded_sha = oracle_evidence.get(evidence_name)
        if type(recorded_sha) is not str or re.fullmatch(
            r"sha256:[0-9a-f]{64}", recorded_sha
        ) is None:
            raise ValueError(
                "runtime MLA-off oracle evidence has invalid "
                f"{evidence_name}"
            )
        try:
            local_sha = "sha256:" + _ih_sha256_file(source_path)
        except OSError as error:
            raise ValueError(
                "cannot verify runtime MLA-off oracle source "
                f"{source_path}: {error}"
            ) from error
        if recorded_sha != local_sha:
            raise ValueError(
                "runtime MLA-off oracle evidence source hash differs from "
                f"local {source_path}"
            )

    for runtime_name in (
        "torch_version",
        "triton_version",
        "cuda_runtime_version",
    ):
        runtime_value = oracle_evidence.get(runtime_name)
        if (
            type(runtime_value) is not str
            or runtime_value != runtime_value.strip()
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", runtime_value)
            is None
        ):
            raise ValueError(
                "runtime MLA-off oracle evidence has invalid "
                f"{runtime_name}"
            )
    device_name = oracle_evidence.get("device_name")
    if (
        type(device_name) is not str
        or not device_name
        or device_name != device_name.strip()
        or any(ord(character) < 32 for character in device_name)
    ):
        raise ValueError(
            "runtime MLA-off oracle evidence has invalid device_name"
        )
    device_capability = oracle_evidence.get("device_capability")
    if (
        type(device_capability) is not list
        or len(device_capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in device_capability
        )
    ):
        raise ValueError(
            "runtime MLA-off oracle evidence has invalid device_capability"
        )
    capability_major, capability_minor = device_capability
    if (
        (capability_major, capability_minor) != (10, 3)
        or "B300" not in device_name
    ):
        raise ValueError(
            "runtime MLA-off batch restore provider must be B300/SM103"
        )

    def required_positive_int(name: str) -> int:
        value = actual.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"runtime MLA-off manifest has invalid positive integer {name}"
            )
        return value

    server_pid = required_positive_int("pid")
    manifest_start_time = required_positive_int("pid_start_time_ticks")
    server_port = required_positive_int("port")
    chunked_prefill_size = required_positive_int("chunked_prefill_size")
    max_prefill_tokens = required_positive_int("max_prefill_tokens")
    certified_max_context = actual.get(
        "mla_off_certified_max_context_tokens"
    )
    if (
        isinstance(certified_max_context, bool)
        or not isinstance(certified_max_context, int)
        or certified_max_context < 0
    ):
        raise ValueError(
            "runtime MLA-off manifest has invalid non-negative integer "
            "mla_off_certified_max_context_tokens"
        )
    qualification_only = actual.get("mla_off_qualification_only")
    qualification_max_context = actual.get(
        "mla_off_qualification_max_context_tokens"
    )
    effective_restore_max_context = actual.get(
        "mla_off_effective_restore_max_context_tokens"
    )
    if type(qualification_only) is not bool:
        raise ValueError(
            "runtime MLA-off manifest has invalid qualification-only flag"
        )
    for name, value in (
        (
            "mla_off_qualification_max_context_tokens",
            qualification_max_context,
        ),
        (
            "mla_off_effective_restore_max_context_tokens",
            effective_restore_max_context,
        ),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(
                f"runtime MLA-off manifest has invalid non-negative integer {name}"
            )
    claim_status = actual.get("performance_claim_status")
    if qualification_only:
        if certified_max_context != 0:
            raise ValueError(
                "qualification-only server must keep formal certified max "
                "context at 0"
            )
        if qualification_max_context <= 0:
            raise ValueError(
                "qualification-only server requires a positive qualification "
                "context cap"
            )
        if effective_restore_max_context != qualification_max_context:
            raise ValueError(
                "qualification-only server effective context cap differs from "
                "its qualification cap"
            )
        if claim_status != "qualification_only_claim_ineligible":
            raise ValueError(
                "qualification-only server is not explicitly claim-ineligible"
            )
    else:
        if qualification_max_context != 0:
            raise ValueError(
                "formal server must not publish a qualification context cap"
            )
        if effective_restore_max_context != certified_max_context:
            raise ValueError(
                "formal server effective context cap differs from its certified cap"
            )
        if claim_status != "unverified":
            raise ValueError(
                "formal server has an unexpected performance claim status"
            )
    device_cache_enabled = actual.get("mla_off_device_cache_enabled")
    device_max_bytes = actual.get("mla_off_device_max_bytes")
    transfer_audit_required = (
        expected.get("mla_off_transfer_audit_format")
        == _IH_MLA_OFF_TRANSFER_AUDIT_SCHEMA
    )
    if (
        transfer_audit_required
        or "mla_off_device_cache_enabled" in actual
        or "mla_off_device_max_bytes" in actual
    ):
        if type(device_cache_enabled) is not bool:
            raise ValueError(
                "runtime MLA-off manifest has invalid device-cache enable flag"
            )
        if type(device_max_bytes) is not int or device_max_bytes < 0:
            raise ValueError(
                "runtime MLA-off manifest has invalid device-cache byte cap"
            )
        if device_cache_enabled != (device_max_bytes > 0):
            raise ValueError(
                "runtime MLA-off manifest device-cache enable/cap fields disagree"
            )
    if server_port > 65535:
        raise ValueError("runtime MLA-off manifest port is outside [1, 65535]")
    expected_max_prefill_tokens = max(
        IH_CHUNK_TOKENS, IH_MERGED_PREFILL_TOKENS
    )
    if (
        chunked_prefill_size != IH_CHUNK_TOKENS
        or max_prefill_tokens != expected_max_prefill_tokens
    ):
        raise ValueError(
            "independent-document pure MLA requires the configured snapshot boundary and "
            "the exact configured restore-only max-prefill cap"
        )
    if chunked_prefill_size > max_prefill_tokens:
        raise ValueError(
            "runtime MLA-off manifest has chunked_prefill_size greater than "
            "max_prefill_tokens"
        )
    try:
        os.kill(server_pid, 0)
        process_state, observed_start_time = _ih_linux_process_identity(server_pid)
    except (TypeError, ValueError, OSError) as error:
        raise ValueError(
            f"runtime MLA-off manifest does not identify a live worker: {error}"
        ) from error
    if process_state in {"Z", "X", "x"}:
        raise ValueError(
            "runtime MLA-off manifest identifies a dead worker: "
            f"pid={server_pid} state={process_state}"
        )
    if observed_start_time != manifest_start_time:
        raise ValueError(
            "runtime MLA-off manifest PID start time does not match the live worker"
        )
    return actual


def _run_indexer_hot_benchmark():
    chunk_tokens = (IH_CHUNK_TOKENS // 128) * 128
    qualification_provenance = _ih_qualification_profile_provenance(
        IH_NUM_CHUNKS * chunk_tokens
    )
    tok = _ih_load_tokenizer()
    if (
        chunk_tokens < 512
        or IH_NUM_CHUNKS <= 0
        or IH_NUM_QUERIES <= 0
        or IH_QUALITY_REPEATS <= 0
    ):
        raise ValueError("chunk/query/repeat counts must be positive and chunk >= 512")
    if (
        not IH_MLA_OFFLOAD
        and IH_SELECTION_POLICY not in {"request_global", "checkpoint_islands"}
    ):
        raise ValueError(
            "selection policy must be request_global or checkpoint_islands"
        )
    if IH_MLA_OFF_QUALIFICATION_ONLY and not IH_MLA_OFFLOAD:
        raise ValueError(
            "MLA-off qualification-only mode requires MLA offload"
        )
    if IH_PROMPT_MANIFEST and IH_PROMPT_MANIFEST_OUT:
        raise ValueError("pure prompt manifest input/output are mutually exclusive")
    if IH_MLA_OFFLOAD and IH_PURE_PROMPT_MODE != "official_rag_v1":
        raise ValueError(
            "pure MLA qualification requires official_rag_v1; raw_suffix is "
            "forbidden"
        )
    if IH_PURE_PROMPT_MODE == "official_rag_v1":
        if not (
            (IH_MLA_OFFLOAD and IH_MLA_OFF_QUALIFICATION_ONLY)
            or IH_ROW_SPARSE_CLOSURE
        ):
            raise ValueError(
                "official_rag_v1 requires either pure MLA qualification-only "
                "mode or the explicit row-sparse closure qualification arm"
            )
        if (
            IH_MLA_OFFLOAD
            and not IH_COMBINED_HEADSPLIT_ROW_SPARSE
            and IH_MLA_OFF_DIAGNOSTIC_ABLATION != "full"
        ):
            raise ValueError("official_rag_v1 requires full pure-MLA ablation")
        if DATASETS != [IH_EXPECTED_DATASET]:
            raise ValueError(
                "official_rag_v1 DATASETS differs from the pre-registered "
                f"dataset: expected={[IH_EXPECTED_DATASET]!r} actual={DATASETS!r}"
            )
        if IH_NUM_QUERIES != len(IH_EXPECTED_QUERY_ROW_IDS) or not (
            IH_EXPECTED_QUERY_ROW_IDS
        ):
            raise ValueError(
                "official_rag_v1 query count differs from the pre-registered "
                f"rows: count={IH_NUM_QUERIES} rows={IH_EXPECTED_QUERY_ROW_IDS}"
            )
        if not IH_DATA_MANIFEST or IH_DATA_MANIFEST_OUT:
            raise ValueError(
                "official_rag_v1 requires an existing frozen data manifest "
                "input and forbids selecting/writing rows during the run"
            )
        if IH_DATA_ROW_OFFSET != 0 or IH_DATA_EXCLUDE_MANIFESTS:
            raise ValueError(
                "official_rag_v1 exact replay forbids row offsets/exclusions"
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}", IH_EXPECTED_DATA_SELECTION_SHA256
        ):
            raise ValueError(
                "official_rag_v1 requires a lowercase 64-hex expected data "
                "selection digest"
            )
        if len(IH_EXPECTED_QUERY_ROW_IDS) == 1:
            for label, digest in (
                ("prompt text", IH_EXPECTED_PROMPT_TEXT_SHA256),
                ("full input ids", IH_EXPECTED_FULL_INPUT_IDS_SHA256),
            ):
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                    raise ValueError(
                        f"official_rag_v1 requires a canonical expected {label} "
                        "SHA-256"
                    )
        elif not re.fullmatch(
            r"[0-9a-f]{64}", IH_EXPECTED_PROMPT_MANIFEST_SHA256
        ):
            raise ValueError(
                "multi-query official_rag_v1 requires a lowercase 64-hex "
                "expected prompt-manifest digest"
            )
        if IH_EXPECTED_FULL_INPUT_TOKENS <= 0:
            raise ValueError(
                "official_rag_v1 requires a positive expected full input length"
            )
        if THINKING_MODE != "chat" or REASONING_EFFORT != "low":
            raise ValueError(
                "official_rag_v1 freezes REDKNOT_THINKING_MODE=chat and "
                "REDKNOT_REASONING_EFFORT=low"
            )
        if IH_RELEVANCE_FIRST or IH_RELEVANCE_LAST:
            raise ValueError(
                "official_rag_v1 preserves frozen chunk order and forbids "
                "query-dependent relevance reordering"
            )
    elif IH_PROMPT_MANIFEST or IH_PROMPT_MANIFEST_OUT:
        raise ValueError(
            "pure prompt manifests require REDKNOT_IH_PURE_PROMPT_MODE="
            "official_rag_v1"
        )
    if (
        IH_MLA_OFF_QUALIFICATION_ONLY
        and IH_MLA_OFF_DIAGNOSTIC_ABLATION != "full"
    ):
        raise ValueError(
            "MLA-off qualification-only mode requires the full pure-MLA path"
        )
    if IH_MLA_OFF_QUALIFICATION_ONLY:
        if IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS != 0:
            raise ValueError(
                "MLA-off qualification-only mode requires the formal "
                "certified max context to remain 0"
            )
        if IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS <= 0:
            raise ValueError(
                "MLA-off qualification-only mode requires a positive "
                "qualification max context"
            )
    elif IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS != 0:
        raise ValueError(
            "MLA-off qualification max context is valid only when "
            "qualification-only mode is enabled"
        )
    if IH_MLA_OFF_MAX_BYTES <= 0:
        raise ValueError("MLA-off cache cap must be positive")
    capacity_geometry = None
    projected_mla_off_bytes = 0
    if IH_MLA_OFFLOAD:
        if IH_NO_LAUNCH and not IH_SERVER_INSTANCE_NONCE:
            raise ValueError(
                "external MLA-off server verification requires "
                "REDKNOT_IH_SERVER_INSTANCE_NONCE"
            )
        if MLA_PASS_MODE != "headwise":
            raise ValueError("MLA-off benchmark requires MLA_PASS_MODE=headwise")
        capacity_geometry = _ih_mla_off_capacity_geometry(
            IH_NUM_CHUNKS * chunk_tokens
        )
        projected_mla_off_bytes = int(
            capacity_geometry["projected_bytes_max_rank"]
        )
    if IH_MLA_OFFLOAD and projected_mla_off_bytes > IH_MLA_OFF_MAX_BYTES:
        raise ValueError(
            "MLA-off offline working set exceeds the per-rank cap: "
            f"projected={projected_mla_off_bytes} cap={IH_MLA_OFF_MAX_BYTES}; "
            "raise REDKNOT_IH_MLA_OFF_MAX_BYTES or reduce block count/length"
        )
    if capacity_geometry is not None:
        print(
            "[mla-off-capacity] "
            f"tp={capacity_geometry['tp_size']} "
            f"local_layers={capacity_geometry['local_layer_counts_by_rank']} "
            f"bytes/token={capacity_geometry['bytes_per_token_by_rank']} "
            f"projected={capacity_geometry['projected_bytes_by_rank']}"
        )
    if IH_MLA_OFFLOAD and (
        MLA_DENSE_PREFIX != 3
        or MLA_DENSE_SUFFIX != 3
        or MLA_LOCAL_WINDOW != 128
        or IH_MLA_OFF_REFRESH_LAYER_STRIDE != 0
        or IH_MLA_OFF_HOT_EXPAND_TOKENS != 0
    ):
        raise ValueError(
            "pure MLA headsplit requires dense prefix/suffix 3/3, local "
            "window 128, and forbids refresh or Indexer-hot expansion"
        )
    supported_pure_geometries = {
        (8, 8192),
        (16, 8192),
        (8, 32768),
        (8, 56320),
        (8, 65536),
    }
    if IH_MLA_OFFLOAD and (
        (IH_NUM_CHUNKS, chunk_tokens) not in supported_pure_geometries
    ):
        raise ValueError(
            "context-bound pure MLA qualification requires exactly 8x8192, "
            "16x8192, 8x32768, 8x56320, or 8x65536 offline document tokens"
        )
    if IH_COMBINED_HEADSPLIT_ROW_SPARSE and (
        IH_QUERY_PROTECTION_TOKENS < 512
        or IH_QUERY_PROTECTION_TOKENS > chunk_tokens
        or IH_QUERY_PROTECTION_TOKENS % 512 != 0
    ):
        raise ValueError(
            "combined query-protection tokens must be a 512-token multiple "
            "within one document"
        )
    if IH_GENERALIZED_ADAPTIVE_CONTROLLER:
        if not IH_COMBINED_HEADSPLIT_ROW_SPARSE:
            raise ValueError(
                "generalized adaptive controller requires the combined "
                "headsplit/row-sparse path"
            )
        if chunk_tokens < max(_IH_GENERALIZED_ADAPTIVE_QUERY_TOKENS):
            raise ValueError(
                "generalized adaptive controller v1 requires 32K documents"
            )
    if IH_MLA_OFFLOAD and not IH_REUSE_HEADS_FULL_SCOPE:
        raise ValueError(
            "context-bound pure MLA requires REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE=1"
        )
    if IH_MLA_OFF_REFRESH_LAYER_STRIDE < 0 or IH_MLA_OFF_HOT_EXPAND_TOKENS < 0:
        raise ValueError("MLA-off refresh stride/hot expansion must be non-negative")
    if IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS < 0:
        raise ValueError("MLA-off certified max context must be non-negative")
    from sglang.srt.layers.attention.redknot.v4.merged_prefill import (
        validate_merged_prefill_request,
    )

    merged_prefill_tokens = validate_merged_prefill_request(
        IH_MERGED_PREFILL_TOKENS,
        segment_tokens=chunk_tokens,
        selection_policy=IH_SELECTION_POLICY,
        num_segments=IH_NUM_CHUNKS,
        pure_context_bound=bool(IH_MLA_OFFLOAD),
        # The context-bound MLA path consumes document 1 through a real radix
        # prefix receipt.  Row-sparse replay instead keeps that document inside
        # the scheduler request and reduces it to a pipeline placeholder, so its
        # merged-prefill alignment origin remains logical position zero.
        resident_prefix_segments=(
            1 if IH_FIRST_DOCUMENT_PREFIX and IH_MLA_OFFLOAD else 0
        ),
    )
    if IH_COMBINED_HEADSPLIT_ROW_SPARSE or not IH_MLA_OFFLOAD:
        if IH_CHECKPOINT_STRIDE < 512 or IH_CHECKPOINT_STRIDE % 512 != 0:
            raise ValueError(
                "checkpoint stride must be >= 512 and a multiple of 512"
            )
        if not (
            1
            <= IH_CHECKPOINT_MAX_ISLANDS
            <= _pro0813_scale_policy.PRO0813_CHECKPOINT_DESCRIPTOR_LIMIT
        ):
            raise ValueError("checkpoint max islands must be in [1, 256]")
        if not 0.0 < IH_ACTIVE_BUDGET_RATIO < 0.85:
            raise ValueError("active budget ratio must be in (0, 0.85)")
        if not 0.0 < IH_HOT_MAX_PER_SEGMENT_RATIO <= 1.0:
            raise ValueError("per-segment cap ratio must be in (0, 1]")
    requested_active_ratio_floor = (
        min(_IH_GENERALIZED_ADAPTIVE_ROW_RATIOS)
        if IH_GENERALIZED_ADAPTIVE_CONTROLLER
        else IH_ACTIVE_BUDGET_RATIO
    )
    capacity_active_ratio = (
        max(_IH_GENERALIZED_ADAPTIVE_ROW_RATIOS)
        if IH_GENERALIZED_ADAPTIVE_CONTROLLER
        else IH_ACTIVE_BUDGET_RATIO
    )
    required_checkpoint_islands = (
        _pro0813_scale_policy.pro0813_required_checkpoint_islands(
            _IH_TARGET_DOCUMENT_TOKENS,
            capacity_active_ratio,
            checkpoint_stride=IH_CHECKPOINT_STRIDE,
        )
    )
    if (
        IH_COMBINED_HEADSPLIT_ROW_SPARSE
        and IH_CHECKPOINT_MAX_ISLANDS < required_checkpoint_islands
    ):
        raise ValueError(
            "checkpoint capacity cannot realize the requested row budget: "
            f"required={required_checkpoint_islands} "
            f"configured={IH_CHECKPOINT_MAX_ISLANDS}"
        )
    min_realized_active_ratio = (
        IH_MIN_REALIZED_ACTIVE_RATIO
        if IH_MIN_REALIZED_ACTIVE_RATIO > 0.0
        else _pro0813_scale_policy.pro0813_min_realized_active_ratio(
            _IH_TARGET_DOCUMENT_TOKENS,
            requested_active_ratio_floor,
            checkpoint_stride=IH_CHECKPOINT_STRIDE,
        )
    )
    if IH_COMBINED_HEADSPLIT_ROW_SPARSE and not (
        0.0 < min_realized_active_ratio <= requested_active_ratio_floor
    ):
        raise ValueError(
            "minimum realized active ratio must be in (0, requested ratio]"
        )
    if not IH_MLA_OFFLOAD and (
        IH_BOUNDARY < 128 or IH_BOUNDARY % 128 != 0
    ):
        raise ValueError("boundary must be a positive multiple of 128")
    if IH_MLA_OFFLOAD and IH_BOUNDARY != 128:
        raise ValueError(
            "independent-document RoPE relocation requires boundary=128"
        )
    if (
        not IH_MLA_OFFLOAD
        and IH_SELECTION_POLICY == "checkpoint_islands"
        and IH_BOUNDARY != 128
    ):
        raise ValueError("checkpoint-island mode currently requires boundary=128")
    if IH_TTFT_ITERS <= 0 or IH_TTFT_WARMUP < 0:
        raise ValueError("TTFT iterations/warmup are invalid")
    sampling_qualification = _ih_performance_claim_qualification(
        ttft_warmup=IH_TTFT_WARMUP,
        ttft_iters=IH_TTFT_ITERS,
        strict=IH_STRICT_PERFORMANCE_CLAIMS,
    )
    if IH_MLA_OFF_DIAGNOSTIC_ONLY:
        sampling_qualification = _ih_apply_diagnostic_claim_gate(
            sampling_qualification,
            IH_MLA_OFF_DIAGNOSTIC_ABLATION,
        )
    sampling_qualification = _ih_apply_qualification_only_claim_gate(
        sampling_qualification,
        IH_MLA_OFF_QUALIFICATION_ONLY,
    )
    if type(IH_QPS_WAVES) is not int or IH_QPS_WAVES <= 0:
        raise ValueError("QPS waves must be a positive integer")
    if not 0.0 <= IH_MIN_TOKEN_AGREEMENT <= 1.0:
        raise ValueError("minimum generation token agreement must be in [0, 1]")
    if not 0.0 <= IH_MIN_EM_RETENTION <= 1.0:
        raise ValueError("minimum EM retention must be in [0, 1]")
    for label, threshold in (
        ("dense F1", IH_MIN_DENSE_F1),
        ("reuse F1", IH_MIN_REUSE_F1),
        ("dense EM", IH_MIN_DENSE_EM),
        ("reuse EM", IH_MIN_REUSE_EM),
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"minimum {label} must be in [0, 1]")
    if IH_RELEVANCE_FIRST and IH_RELEVANCE_LAST:
        raise ValueError("relevance-first and relevance-last are mutually exclusive")

    # Freeze every dataset row and final token id before the server starts.
    # This prevents any model output or runtime failure from influencing prompt
    # selection, ordering, truncation, or the qualification context cap.
    chunks, queries, data_selection = _ih_load(
        tok,
        chunk_tokens,
        IH_NUM_CHUNKS,
        IH_NUM_QUERIES,
        return_manifest=True,
    )
    if IH_DATA_MANIFEST_OUT:
        _ih_write_data_manifest(IH_DATA_MANIFEST_OUT, data_selection)
        print(
            "[pure-mla] frozen data selection written to "
            f"{IH_DATA_MANIFEST_OUT}"
        )
    print(
        "[pure-mla] data selection "
        f"sha256={data_selection['selection_sha256']} "
        f"mode={data_selection['selection']['mode']}"
    )
    prompt_manifest = None
    if IH_PURE_PROMPT_MODE == "official_rag_v1":
        chunks, queries, prompt_manifest = _ih_build_official_pure_prompt(
            tok,
            chunks,
            queries,
            data_selection,
            chunk_tokens=chunk_tokens,
        )
        print(
            "[pure-mla] frozen official prompt "
            f"sha256={prompt_manifest['prompt_manifest_sha256']} "
            "max_total_tokens="
            f"{prompt_manifest['geometry'].get('max_total_tokens', prompt_manifest['geometry'].get('total_tokens'))}"
        )
        _ih_validate_official_prompt_run_identity(
            prompt_manifest,
            (
                IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS
                if IH_MLA_OFF_QUALIFICATION_ONLY
                else IH_EXPECTED_FULL_INPUT_TOKENS
            ),
        )
    # Every request below reuses these immutable token tuples.  Dense and pure
    # restore are never permitted to reconstruct or retokenize the prompt.
    chunks = tuple(tuple(map(int, chunk)) for chunk in chunks)
    queries = tuple(
        (
            str(question),
            tuple(map(int, online_suffix)),
            tuple(map(str, answers)),
            int(expected_source_chunk),
        )
        for question, online_suffix, answers, expected_source_chunk in queries
    )
    if any(len(chunk) != chunk_tokens for chunk in chunks):
        raise RuntimeError("all offline chunks must have the exact aligned length")
    query_start = len(chunks) * chunk_tokens
    if IH_MLA_OFF_QUALIFICATION_ONLY:
        largest_request_tokens = query_start + max(
            len(query[1]) for query in queries
        )
        if largest_request_tokens > IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS:
            raise ValueError(
                "qualification context cap is below the largest frozen request: "
                f"required={largest_request_tokens} "
                f"cap={IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS}"
            )
        if (
            IH_PURE_PROMPT_MODE == "official_rag_v1"
            and largest_request_tokens
            != IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS
        ):
            raise ValueError(
                "official prompt qualification cap must equal the frozen "
                "request length: "
                f"required={largest_request_tokens} "
                f"cap={IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS}"
            )
    # Legacy mode keys artifacts by body content.  Pure MLA replaces these
    # after the live server policy is authenticated, because its artifact key
    # additionally binds model/policy and the complete causal prefix.
    segment_hashes = [_ih_chunk_hash(chunk) for chunk in chunks]
    offline_prefix_ids = tuple(token for chunk in chunks for token in chunk)
    materialized_cache_key = None
    radix_prefix_receipt_key = None
    materialization_sentinel_token = None
    if IH_PREFIX_MATERIALIZATION:
        materialization_sentinel_token = _ih_prefix_materialization_sentinel(
            queries
        )
        cached_prefix_ids = (
            chunks[0] if IH_FIRST_DOCUMENT_PREFIX else offline_prefix_ids
        )
        cache_key_digest = hashlib.sha256(
            json.dumps(
                {
                    "schema": "redknot-prefix-materialization-key-v2",
                    "scope": IH_PREFIX_MATERIALIZATION_SCOPE,
                    "server_instance_nonce": IH_SERVER_INSTANCE_NONCE,
                    "offline_prefix_sha256": _ih_chunk_hash(cached_prefix_ids),
                    "data_selection_sha256": data_selection["selection_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        materialized_cache_key = "redknot-prefix-" + cache_key_digest
        radix_prefix_receipt_key = "sha256:" + cache_key_digest
    context_segments = ()
    base_segments = [
        {
            "seg_hash": segment_hashes[i],
            "token_hash": segment_hashes[i],
            "global_offset": i * chunk_tokens,
            "length": chunk_tokens,
            "skip_first": 0 if IH_MLA_OFFLOAD else IH_BOUNDARY,
            "canonical_start_pos": 0,
        }
        for i in range(len(chunks))
    ]

    proc = None
    verified_runtime_server_policy = None
    previous_signal_handlers = {}
    W = 100

    def interrupt_owned_server(signum, _frame):
        raise KeyboardInterrupt(
            f"received signal {signum}; stopping owned RedKnot server"
        )

    try:
        if not IH_NO_LAUNCH:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupt_owned_server)
            proc = _ih_launch_server()
        _ih_wait_ready(proc)
        print("[pure-mla] server ready" if IH_MLA_OFFLOAD else "[indexer-hot] server ready")
        if IH_MLA_OFFLOAD:
            runtime_manifest = (
                IH_SERVER_POLICY_MANIFEST
                if IH_NO_LAUNCH
                else str(Path(IH_RANK_LOG_DIR) / "server_policy_manifest.json")
            )
            verified_runtime_server_policy = _ih_verify_runtime_server_policy(
                capacity_geometry, runtime_manifest
            )
            context_segments = _ih_build_context_segment_contracts(
                chunks,
                model_compat_hash=str(
                    verified_runtime_server_policy["model_compat_hash"]
                ),
                head_policy_hash=str(
                    verified_runtime_server_policy[
                        "effective_head_policy_hash"
                    ]
                ),
            )
            segment_hashes = [
                str(contract["seg_hash"]) for contract in context_segments
            ]
            base_segments = [
                {
                    **contract,
                    "global_offset": index * int(contract["length"]),
                    "skip_first": 0 if index == 0 else IH_BOUNDARY,
                }
                for index, contract in enumerate(context_segments)
            ]
            print(
                "[mla-off-capacity] runtime server policy verified: "
                f"port={verified_runtime_server_policy['port']} "
                "chunked_prefill_size="
                f"{verified_runtime_server_policy['chunked_prefill_size']} "
                "max_prefill_tokens="
                f"{verified_runtime_server_policy['max_prefill_tokens']}"
            )
        # Phase 1: each document is independently prefetched from local
        # position zero.  Its position-bearing cache is canonicalized during
        # capture and rotated to the destination absolute offset on restore.
        snapshot_audit_start = (
            _ih_runtime_log_offsets() if IH_MLA_OFFLOAD else {}
        )
        snapshot_cached_tokens = []
        for i, chunk in enumerate(chunks):
            _ih_flush()
            if IH_MLA_OFFLOAD:
                snapshot_input, snapshot_plan = (
                    _ih_build_context_snapshot_request(
                        chunks,
                        context_segments,
                        index=i,
                        model_compat_hash=str(
                            verified_runtime_server_policy[
                                "model_compat_hash"
                            ]
                        ),
                        head_policy_hash=str(
                            verified_runtime_server_policy[
                                "effective_head_policy_hash"
                            ]
                        ),
                    )
                )
            else:
                snapshot_input = chunk
                snapshot_plan = {
                    "mode": "snapshot",
                    "seg_hash": segment_hashes[i],
                    "length": chunk_tokens,
                    "canonical_start_pos": 0,
                    "allow_approximate": True,
                    "reuse_window_kv": True,
                }
                if IH_SELECTION_POLICY == "checkpoint_islands":
                    snapshot_plan["checkpoint_stride_tokens"] = (
                        IH_CHECKPOINT_STRIDE
                    )
            snapshot_result = _ih_generate(
                snapshot_input,
                2,
                snapshot_plan,
                request_tag=f"snapshot-{i}",
            )
            snapshot_cached_tokens.append(snapshot_result["cached_tokens"])
            print(
                f"[{'pure-mla' if IH_MLA_OFFLOAD else 'indexer-hot'}] "
                f"snapshot document {i} cached independently from position0 "
                f"tokens={len(snapshot_input)}"
            )
        if IH_MLA_OFFLOAD:
            shared_snapshot_audit = _ih_validate_shared_snapshot_audit(
                _ih_read_shared_snapshot_audit(snapshot_audit_start),
                expected_tp_size=IH_TP_SIZE,
                expected_segment_hashes=segment_hashes,
            )
        else:
            shared_snapshot_audit = {
                "schema": _IH_SHARED_SNAPSHOT_AUDIT_SCHEMA,
                "required": False,
                "pass": None,
            }
        materialization_result = None
        materialization_restore_plan = None
        metric_start = None
        expected_mla_request_ids = set()
        expected_mla_q_rows_by_request = {}
        expected_mla_position_start_by_request = {}
        prefix_materialization_request_ids = set()
        prefix_online_rows_by_request = {}
        audit_allowed_mla_request_ids = set()
        if IH_FULL_PREFIX_MATERIALIZATION:
            # Artifact publication remains fully isolated above.  Once all
            # eight context-bound artifacts are certified, build one exact
            # RedKnot-merged cache image of the immutable document prefix under the
            # server-instance/hash-bound key.  The divergent sentinel makes
            # all document tokens internal radix nodes, but can never match
            # the first online token.  Reuse must therefore enter exactly at
            # the query suffix without weakening snapshot certification.
            _ih_flush()
            metric_start = _ih_runtime_log_offsets()
            materialization_restore_plan = _ih_build_context_restore_plan(
                offline_prefix_ids,
                (materialization_sentinel_token,),
                context_segments,
                model_compat_hash=str(
                    verified_runtime_server_policy["model_compat_hash"]
                ),
                head_policy_hash=str(
                    verified_runtime_server_policy[
                        "effective_head_policy_hash"
                    ]
                ),
                merged_prefill_tokens=merged_prefill_tokens,
            )
            materialization_result = _ih_generate(
                offline_prefix_ids + (materialization_sentinel_token,),
                1,
                materialization_restore_plan,
                request_tag="prefix-materialize",
                cache_key=materialized_cache_key,
            )
            if materialization_result["cached_tokens"] not in (0, None):
                raise RuntimeError(
                    "prefix materialization producer unexpectedly reused an "
                    f"older cache image: {materialization_result['cached_tokens']!r}"
                )
            expected_mla_request_ids.add(materialization_result["request_id"])
            expected_mla_q_rows_by_request[
                materialization_result["request_id"]
            ] = len(offline_prefix_ids) + 1
        elif IH_FIRST_DOCUMENT_PREFIX:
            # Seed exactly document 1 (8K + a divergent sentinel) under the
            # stable radix namespace.  In combined selected-row mode the model
            # runner deliberately converts this producer to a full-row forward
            # so the radix entry owns complete layer KV/state; only documents
            # 2..N are pruned by the timed consumer.  The producer remains
            # outside online TTFT/QPS.
            _ih_flush()
            materialization_restore_plan = _ih_build_context_restore_plan(
                chunks[0],
                (materialization_sentinel_token,),
                context_segments,
                model_compat_hash=str(
                    verified_runtime_server_policy["model_compat_hash"]
                ),
                head_policy_hash=str(
                    verified_runtime_server_policy[
                        "effective_head_policy_hash"
                    ]
                ),
                radix_prefix_role="seed",
                radix_prefix_receipt_key=radix_prefix_receipt_key,
                segment_limit=1,
            )
            materialization_result = _ih_generate(
                tuple(chunks[0]) + (materialization_sentinel_token,),
                1,
                materialization_restore_plan,
                request_tag="first-document-prefix-seed",
                cache_key=materialized_cache_key,
            )
            if materialization_result["cached_tokens"] not in (0, None):
                raise RuntimeError(
                    "first-document prefix seed unexpectedly reused an older "
                    f"cache image: {materialization_result['cached_tokens']!r}"
                )

        def refresh_first_document_prefix(request_tag):
            """Restore the immutable first-document radix seed before reuse.

            A completed online request is allowed to insert its remaining
            document rows into the radix tree.  Reusing that expanded entry
            would turn later measurements into full-prefix cache hits.  Flush
            and replay only the certified document-1 seed outside the timed
            request so every observation begins at the exact chunk boundary.
            """

            if not IH_FIRST_DOCUMENT_PREFIX:
                return None
            _ih_flush()
            result = _ih_generate(
                tuple(chunks[0]) + (materialization_sentinel_token,),
                1,
                materialization_restore_plan,
                request_tag=request_tag,
                cache_key=materialized_cache_key,
            )
            if result["cached_tokens"] not in (0, None):
                raise RuntimeError(
                    "first-document prefix refresh unexpectedly reused an "
                    f"older cache image: {result['cached_tokens']!r}"
                )
            if metric_start is not None:
                audit_allowed_mla_request_ids.add(result["request_id"])
            return result

        def make_query_case(question_text, query_ids):
            if IH_MLA_OFFLOAD:
                # Keep document order fixed.  In combined mode, use only the
                # question text (never gold/expected_chunk/model output) to
                # protect one or two lexical documents with a cheap token-ID
                # TF-IDF sketch. Remaining documents stay checkpoint-row sparse.
                chunk_order = tuple(range(len(chunks)))
                composed_prefix = tuple(
                    token for chunk in chunks for token in chunk
                )
                original_weights = None
                controller_decision = None
                selected_active_ratio = IH_ACTIVE_BUDGET_RATIO
                selected_query_protection_tokens = IH_QUERY_PROTECTION_TOKENS
                selected_query_protection_documents = 1
                query_protection_policy = "none"
                query_protected_segment_index = -1
                query_protected_ranges = []
                if IH_COMBINED_HEADSPLIT_ROW_SPARSE:
                    selection_ids = tok(
                        "\nQuestion: " + str(question_text) + "\nAnswer:",
                        add_special_tokens=False,
                    )["input_ids"]
                    original_weights = _ih_query_chunk_weights(
                        chunks, selection_ids
                    )
                    if IH_GENERALIZED_ADAPTIVE_CONTROLLER:
                        controller_decision = _ih_generalized_adaptive_decision(
                            original_weights
                        )
                        selected_active_ratio = float(
                            controller_decision["active_token_budget_ratio"]
                        )
                        selected_query_protection_tokens = int(
                            controller_decision["query_protection_tokens"]
                        )
                        selected_query_protection_documents = int(
                            controller_decision["query_protection_documents"]
                        )
                    protected_segment_indices = sorted(
                        range(len(original_weights)),
                        key=lambda index: (-original_weights[index], index),
                    )[:selected_query_protection_documents]
                    query_protected_segment_index = protected_segment_indices[0]
                    query_protection_policy = (
                        "lexical_topk_block_windows_v2"
                        if selected_query_protection_documents > 1
                        else "lexical_top1_block_windows_v1"
                    )
                    per_document_budget = (
                        selected_query_protection_tokens
                        // selected_query_protection_documents
                    )
                    if (
                        per_document_budget * selected_query_protection_documents
                        != selected_query_protection_tokens
                    ):
                        raise RuntimeError(
                            "generalized query protection budget is not divisible"
                        )
                    for protected_segment_index in protected_segment_indices:
                        query_protected_ranges.extend(
                            _ih_query_protected_ranges(
                                chunks[protected_segment_index],
                                selection_ids,
                                global_offset=(
                                    protected_segment_index * IH_CHUNK_TOKENS
                                ),
                                budget_tokens=per_document_budget,
                            )
                        )
                    query_protected_ranges.sort(
                        key=lambda item: (int(item["start"]), int(item["end"]))
                    )
                    if controller_decision is not None:
                        controller_decision = dict(controller_decision)
                        controller_decision["protected_segment_indices"] = list(
                            protected_segment_indices
                        )
                plan = _ih_build_context_restore_plan(
                    composed_prefix,
                    query_ids,
                    context_segments,
                    model_compat_hash=str(
                        verified_runtime_server_policy[
                            "model_compat_hash"
                        ]
                    ),
                    head_policy_hash=str(
                        verified_runtime_server_policy[
                            "effective_head_policy_hash"
                        ]
                    ),
                    merged_prefill_tokens=merged_prefill_tokens,
                    radix_prefix_role=(
                        "consume" if IH_FIRST_DOCUMENT_PREFIX else None
                    ),
                    radix_prefix_receipt_key=(
                        radix_prefix_receipt_key
                        if IH_FIRST_DOCUMENT_PREFIX
                        else None
                    ),
                    query_protection_policy=query_protection_policy,
                    query_protected_segment_index=(
                        query_protected_segment_index
                    ),
                    query_protected_ranges=query_protected_ranges,
                    active_token_budget_ratio=selected_active_ratio,
                )
                return (
                    composed_prefix,
                    plan,
                    chunk_order,
                    original_weights,
                    controller_decision,
                )

            weights = _ih_query_chunk_weights(chunks, query_ids)
            chunk_order = list(range(len(chunks)))
            if IH_RELEVANCE_LAST:
                # RAG chunks are semantically independent.  Put the strongest
                # lexical match immediately before the query while preserving a
                # deterministic order among the remaining chunks. Dense and
                # reuse requests consume this exact same composed prompt.
                chunk_order.sort(key=lambda index: (weights[index], index))
            elif IH_RELEVANCE_FIRST:
                # Keeping the strongest match at canonical offset zero removes
                # relocation error from the evidence-bearing chunk entirely.
                chunk_order.sort(key=lambda index: (-weights[index], index))
            # Official-prompt query suffixes are frozen tuples.  Keep the
            # selected-row prefix immutable as well so TTFT, quality, and QPS
            # all concatenate the exact same sequence type and token IDs.
            composed_prefix = tuple(
                token for index in chunk_order for token in chunks[index]
            )
            segments = []
            for destination_index, source_index in enumerate(chunk_order):
                segment = {
                    **base_segments[source_index],
                    "global_offset": destination_index * chunk_tokens,
                    "query_score": weights[source_index],
                    "source_chunk_index": source_index,
                }
                segments.append(segment)
            plan = {
                "mode": "restore",
                "skip_forward": True,
                "allow_approximate": True,
                "inject_full_blocks": True,
                "refresh_selected_c4_rows": True,
                "reuse_window_kv": True,
                "skip_prefix_recompute": IH_SKIP_PREFIX_RECOMPUTE,
                "query_start": query_start,
                "total_tokens": query_start + len(query_ids),
                "interior_stride": 0,
                "hot_frac": IH_HOT_FRAC,
                "selection_policy": IH_SELECTION_POLICY,
                "active_token_budget_ratio": IH_ACTIVE_BUDGET_RATIO,
                "hot_max_per_segment_ratio": IH_HOT_MAX_PER_SEGMENT_RATIO,
                "segments": segments,
            }
            if IH_ROW_SPARSE_CLOSURE:
                plan["row_sparse_closure"] = True
            if merged_prefill_tokens:
                plan["merged_prefill_tokens"] = merged_prefill_tokens
            if IH_SELECTION_POLICY == "checkpoint_islands":
                plan.update(
                    {
                        "checkpoint_stride_tokens": IH_CHECKPOINT_STRIDE,
                        "checkpoint_max_islands": IH_CHECKPOINT_MAX_ISLANDS,
                    }
                )
            return composed_prefix, plan, chunk_order, weights, None

        # One allocator flush before warmup; all measured requests use unique
        # radix keys instead of repeatedly emptying the CUDA allocator.
        if not IH_PREFIX_MATERIALIZATION:
            _ih_flush()
        import statistics

        prefix0, restore_plan0, chunk_order0, _, _ = make_query_case(
            queries[0][0], queries[0][1]
        )
        full0 = prefix0 + queries[0][1]
        if prompt_manifest is not None:
            if prompt_manifest.get("format") == _IH_PURE_PROMPT_MULTI_MANIFEST_FORMAT:
                frozen_full_hash = prompt_manifest["cases"][0][
                    "full_input_ids_sha256"
                ]
            else:
                frozen_full_hash = prompt_manifest["prompt"][
                    "full_input_ids_sha256"
                ]
            if _ih_chunk_hash(full0) != frozen_full_hash:
                raise RuntimeError(
                    "composed request differs from frozen pure prompt"
                )
        reuse_cached_tokens = []
        for warmup in range(IH_TTFT_WARMUP):
            _ih_generate_streaming_ttft(
                full0, request_tag=f"warmup-dense-{warmup}"
            )
            refresh_first_document_prefix(
                f"first-document-prefix-refresh-warmup-{warmup}"
            )
            warmup_reuse = _ih_generate_streaming_ttft(
                full0,
                None if IH_FULL_PREFIX_MATERIALIZATION else restore_plan0,
                request_tag=f"warmup-reuse-{warmup}",
                cache_key=materialized_cache_key,
            )
            reuse_cached_tokens.append(warmup_reuse["cached_tokens"])

        if metric_start is None:
            metric_start = _ih_runtime_log_offsets()

        # Phase 2a: client-observed first-token latency from a proven streaming
        # token event.  Non-streaming request E2E is never labelled TTFT.
        d_ttft, r_ttft = [], []
        d_model_ttft, r_model_ttft = [], []
        d_queue, r_queue = [], []
        d_non_model, r_non_model = [], []
        modes = ["dense", "reuse"]
        rng = random.Random(SEED)
        for iteration in range(IH_TTFT_ITERS):
            rng.shuffle(modes)
            for mode in modes:
                if mode == "dense":
                    dense_result = _ih_generate_streaming_ttft(
                        full0,
                        request_tag=f"ttft-dense-{iteration}",
                    )
                    d_ttft.append(dense_result["ttft"])
                    if dense_result["model_prefill_ttft"] is not None:
                        d_model_ttft.append(dense_result["model_prefill_ttft"])
                        d_queue.append(dense_result["scheduler_queue_time"])
                        d_non_model.append(dense_result["server_non_model_ttft"])
                else:
                    refresh_first_document_prefix(
                        f"first-document-prefix-refresh-ttft-{iteration}"
                    )
                    reuse_result = _ih_generate_streaming_ttft(
                        full0,
                        None if IH_FULL_PREFIX_MATERIALIZATION else restore_plan0,
                        request_tag=f"ttft-reuse-{iteration}",
                        cache_key=materialized_cache_key,
                    )
                    r_ttft.append(reuse_result["ttft"])
                    if reuse_result["model_prefill_ttft"] is not None:
                        r_model_ttft.append(reuse_result["model_prefill_ttft"])
                        r_queue.append(reuse_result["scheduler_queue_time"])
                        r_non_model.append(reuse_result["server_non_model_ttft"])
                    reuse_cached_tokens.append(reuse_result["cached_tokens"])
                    if IH_FULL_PREFIX_MATERIALIZATION:
                        prefix_materialization_request_ids.add(
                            reuse_result["request_id"]
                        )
                        prefix_online_rows_by_request[
                            reuse_result["request_id"]
                        ] = len(full0) - int(reuse_result["cached_tokens"])
                    else:
                        expected_mla_request_ids.add(
                            reuse_result["request_id"]
                        )
                        expected_mla_q_rows_by_request[
                            reuse_result["request_id"]
                        ] = len(full0) - (
                            IH_CHUNK_TOKENS if IH_FIRST_DOCUMENT_PREFIX else 0
                        )
                        expected_mla_position_start_by_request[
                            reuse_result["request_id"]
                        ] = (
                            IH_CHUNK_TOKENS if IH_FIRST_DOCUMENT_PREFIX else 0
                        )
        dm, rm = statistics.median(d_ttft), statistics.median(r_ttft)
        if IH_PREFIX_MATERIALIZATION:
            expected_cached_tokens = (
                IH_CHUNK_TOKENS if IH_FIRST_DOCUMENT_PREFIX else query_start
            )
            if not reuse_cached_tokens or any(
                type(value) is not int or value != expected_cached_tokens
                for value in reuse_cached_tokens
            ):
                raise RuntimeError(
                    "RedKnot prefix cache did not preserve exactly the frozen "
                    f"offline prefix: observed={reuse_cached_tokens!r} "
                    f"expected={expected_cached_tokens}"
                )

        # Phase 2b: per-query quality + long-text agreement.
        print("\n" + "=" * W)
        print(
            " REDKNOT PURE MLA OFFLINE/ONLINE MERGE — qualification benchmark"
            if IH_MLA_OFFLOAD
            else " REDKNOT INDEXER-HOT OFFLINE REUSE — one-click benchmark"
        )
        print(f" Model: {MODEL_PATH}")
        if IH_MLA_OFFLOAD:
            print(
                f" chunks={len(chunks)}x{chunk_tokens} total={len(full0)} "
                "strategy=independent_position0_rope_relocation_headsplit_fullscope "
                f"prompt={IH_PURE_PROMPT_MODE} boundary={IH_BOUNDARY} "
                f"quality_repeats={IH_QUALITY_REPEATS}"
            )
        else:
            print(
                f" chunks={len(chunks)}x{chunk_tokens} total={len(full0)} "
                f"physical_merge={merged_prefill_tokens or 'off'} "
                f"policy={IH_SELECTION_POLICY} "
                f"active_budget={IH_ACTIVE_BUDGET_RATIO:.1%} "
                f"boundary={IH_BOUNDARY} relevance_first={IH_RELEVANCE_FIRST} "
                f"relevance_last={IH_RELEVANCE_LAST} "
                f"skip_prefix_recompute={IH_SKIP_PREFIX_RECOMPUTE} "
                f"quality_repeats={IH_QUALITY_REPEATS}"
            )
        print("=" * W)
        print(
            f" streaming TTFT: dense_p50={dm:.3f}s reuse_p50={rm:.3f}s "
            f"speedup={dm / max(rm, 1e-6):.2f}x "
            f"dense_p95={_ih_percentile(d_ttft, .95):.3f}s "
            f"reuse_p95={_ih_percentile(r_ttft, .95):.3f}s"
        )
        print("-" * W)

        top1_hits, quality_pair_count = 0, 0
        cosines, dense_f1s, reuse_f1s = [], [], []
        dense_ems, reuse_ems = [], []
        token_agreements, quality_rows = [], []
        dense_quality_e2e, reuse_quality_e2e, logical_prompt_tokens = [], [], 0
        for qi, (qtext, q, golds, expected_chunk) in enumerate(queries):
            (
                query_prefix,
                restore_plan,
                chunk_order,
                original_weights,
                controller_decision,
            ) = make_query_case(qtext, q)
            full = query_prefix + q
            if (
                prompt_manifest is not None
                and prompt_manifest.get("format")
                == _IH_PURE_PROMPT_MULTI_MANIFEST_FORMAT
                and _ih_chunk_hash(full)
                != prompt_manifest["cases"][qi]["full_input_ids_sha256"]
            ):
                raise RuntimeError(
                    f"quality query {qi} differs from its frozen prompt case"
                )
            repeat_rows = []
            quality_rng = random.Random(SEED + qi * 1000003)
            for repeat in range(IH_QUALITY_REPEATS):
                pair = {}
                pair_modes = ["dense", "reuse"]
                quality_rng.shuffle(pair_modes)
                for mode in pair_modes:
                    if mode == "reuse":
                        refresh_first_document_prefix(
                            "first-document-prefix-refresh-quality-"
                            f"{qi}-{repeat}"
                        )
                    pair[mode] = _ih_generate(
                        full,
                        IH_MAX_NEW,
                        (
                            None
                            if mode != "reuse" or IH_FULL_PREFIX_MATERIALIZATION
                            else restore_plan
                        ),
                        request_tag=f"quality-{mode}-{qi}-{repeat}",
                        collect_logprobs=True,
                        cache_key=(
                            materialized_cache_key if mode == "reuse" else None
                        ),
                    )
                dense, reuse = pair["dense"], pair["reuse"]
                if IH_PREFIX_MATERIALIZATION:
                    expected_cached_tokens = (
                        IH_CHUNK_TOKENS
                        if IH_FIRST_DOCUMENT_PREFIX
                        else query_start
                    )
                    if (
                        type(reuse["cached_tokens"]) is not int
                        or reuse["cached_tokens"] != expected_cached_tokens
                    ):
                        raise RuntimeError(
                            "quality reuse request missed the exact cached prefix: "
                            f"observed={reuse['cached_tokens']!r} "
                            f"expected={expected_cached_tokens}"
                        )
                    reuse_cached_tokens.append(reuse["cached_tokens"])
                if IH_FULL_PREFIX_MATERIALIZATION:
                    prefix_materialization_request_ids.add(reuse["request_id"])
                    prefix_online_rows_by_request[reuse["request_id"]] = (
                        len(full) - int(reuse["cached_tokens"])
                    )
                else:
                    expected_mla_request_ids.add(reuse["request_id"])
                    expected_mla_q_rows_by_request[reuse["request_id"]] = (
                        len(full)
                        - (IH_CHUNK_TOKENS if IH_FIRST_DOCUMENT_PREFIX else 0)
                    )
                    expected_mla_position_start_by_request[
                        reuse["request_id"]
                    ] = IH_CHUNK_TOKENS if IH_FIRST_DOCUMENT_PREFIX else 0
                cos = _ih_first_topk_cosine(dense, reuse)
                t1 = bool(
                    dense["output_ids"]
                    and reuse["output_ids"]
                    and dense["output_ids"][0] == reuse["output_ids"][0]
                )
                d_ans = _short_ans(dense["text"])
                r_ans = _short_ans(reuse["text"])
                df1 = f1_max(d_ans, golds)
                rf1 = f1_max(r_ans, golds)
                dem = em_max(d_ans, golds)
                rem = em_max(r_ans, golds)
                compared = max(len(dense["output_ids"]), len(reuse["output_ids"]))
                agreement = sum(
                    int(a == b)
                    for a, b in zip(dense["output_ids"], reuse["output_ids"])
                ) / max(1, compared)
                length_match = len(dense["output_ids"]) == len(reuse["output_ids"])
                exact_match = (
                    length_match and dense["output_ids"] == reuse["output_ids"]
                )
                top1_hits += int(t1)
                quality_pair_count += 1
                cosines.append(cos)
                dense_f1s.append(df1)
                reuse_f1s.append(rf1)
                dense_ems.append(dem)
                reuse_ems.append(rem)
                token_agreements.append(agreement)
                dense_quality_e2e.append(dense["e2e"])
                reuse_quality_e2e.append(reuse["e2e"])
                logical_prompt_tokens += len(full)
                repeat_rows.append(
                    {
                        "repeat": repeat,
                        "top1_agreement": t1,
                        "top10_probability_cosine": cos,
                        "generation_token_agreement": agreement,
                        "generation_length_match": length_match,
                        "generation_exact_match": exact_match,
                        "dense_f1": df1,
                        "reuse_f1": rf1,
                        "dense_em": dem,
                        "reuse_em": rem,
                        "dense_e2e": dense["e2e"],
                        "reuse_e2e": reuse["e2e"],
                        "dense_output_tokens": len(dense["output_ids"]),
                        "reuse_output_tokens": len(reuse["output_ids"]),
                        "dense_text": dense["text"],
                        "reuse_text": reuse["text"],
                    }
                )

            q_top1_hits = sum(int(row["top1_agreement"]) for row in repeat_rows)
            q_cosines = [row["top10_probability_cosine"] for row in repeat_rows]
            q_agreements = [row["generation_token_agreement"] for row in repeat_rows]
            q_dense_f1 = statistics.median(row["dense_f1"] for row in repeat_rows)
            q_reuse_f1 = statistics.median(row["reuse_f1"] for row in repeat_rows)
            q_dense_em = statistics.median(row["dense_em"] for row in repeat_rows)
            q_reuse_em = statistics.median(row["reuse_em"] for row in repeat_rows)
            representative = repeat_rows[0]
            quality_row = {
                "query_index": qi,
                "question": qtext,
                "golds": golds,
                "top1_agreement": q_top1_hits == IH_QUALITY_REPEATS,
                "top1_agreement_rate": q_top1_hits / IH_QUALITY_REPEATS,
                "top10_probability_cosine": statistics.median(q_cosines),
                "min_top10_probability_cosine": min(q_cosines),
                "generation_token_agreement": statistics.median(q_agreements),
                "min_generation_token_agreement": min(q_agreements),
                "generation_length_match_rate": sum(
                    int(row["generation_length_match"]) for row in repeat_rows
                )
                / IH_QUALITY_REPEATS,
                "generation_exact_match_rate": sum(
                    int(row["generation_exact_match"]) for row in repeat_rows
                )
                / IH_QUALITY_REPEATS,
                "dense_f1": q_dense_f1,
                "reuse_f1": q_reuse_f1,
                "f1_delta": q_reuse_f1 - q_dense_f1,
                "dense_em": q_dense_em,
                "reuse_em": q_reuse_em,
                "em_delta": q_reuse_em - q_dense_em,
                "dense_e2e": statistics.median(
                    row["dense_e2e"] for row in repeat_rows
                ),
                "reuse_e2e": statistics.median(
                    row["reuse_e2e"] for row in repeat_rows
                ),
                "dense_output_tokens": representative["dense_output_tokens"],
                "reuse_output_tokens": representative["reuse_output_tokens"],
                "full_input_ids_sha256": _ih_chunk_hash(full),
                "dense_text": representative["dense_text"],
                "reuse_text": representative["reuse_text"],
                "repeats": repeat_rows,
            }
            if IH_MLA_OFFLOAD:
                quality_row["data_manifest_expected_source_chunk"] = (
                    expected_chunk
                )
                if IH_COMBINED_HEADSPLIT_ROW_SPARSE:
                    quality_row.update(
                        {
                            "query_protection_policy": restore_plan[
                                "query_protection_policy"
                            ],
                            "query_protected_segment_index": restore_plan[
                                "query_protected_segment_index"
                            ],
                            "query_protected_ranges": restore_plan[
                                "query_protected_ranges"
                            ],
                            "query_protection_tokens": sum(
                                int(item["end"]) - int(item["start"])
                                for item in restore_plan[
                                    "query_protected_ranges"
                                ]
                            ),
                            "active_token_budget_ratio": float(
                                restore_plan["active_token_budget_ratio"]
                            ),
                            "generalized_adaptive_controller": (
                                controller_decision
                            ),
                            "query_chunk_weights": original_weights,
                            "query_protection_hit_expected_source": any(
                                int(item["start"])
                                >= expected_chunk * IH_CHUNK_TOKENS
                                and int(item["end"])
                                <= (expected_chunk + 1) * IH_CHUNK_TOKENS
                                for item in restore_plan[
                                    "query_protected_ranges"
                                ]
                            ),
                        }
                    )
            else:
                quality_row.update(
                    {
                        "expected_chunk": expected_chunk,
                        "query_chunk_weights": original_weights,
                        "composed_chunk_order": chunk_order,
                        "expected_chunk_composed_index": chunk_order.index(
                            expected_chunk
                        ),
                    }
                )
            quality_rows.append(quality_row)
            print(f" Q{qi}: {qtext}")
            print("    [DENSE OUTPUT]")
            print(representative["dense_text"])
            print("    [REDKNOT OUTPUT]")
            print(representative["reuse_text"])

        # Phase 2c: a real simultaneous-request QPS sweep.  Every measured wave
        # is a dense/reuse pair, and pair order alternates AB/BA to suppress
        # monotonic thermal/cache drift.
        # Every successful reuse request id is added to the formal runtime
        # evidence set; concurrency that forces an unsupported mixed batch
        # therefore fails visibly instead of being reported as reuse QPS.
        qps_results = None
        if IH_MEASURE_QPS:
            qps_reuse_plan = restore_plan0
            qps_generate_fn = None
            prefix_qps_marker = None
            if IH_PREFIX_MATERIALIZATION:
                # The paired harness distinguishes dense/reuse by the plan
                # object.  In materialized-prefix mode the marker is consumed
                # only by this client closure: no restore plan reaches the
                # server, and every successful reuse request must prove the
                # exact 65K radix hit before it is counted.
                if IH_FULL_PREFIX_MATERIALIZATION:
                    prefix_qps_marker = object()
                    qps_reuse_plan = prefix_qps_marker

                def qps_generate_fn(
                    input_ids,
                    max_new,
                    plan,
                    *,
                    request_tag,
                    request_id,
                ):
                    is_reuse = (
                        plan is prefix_qps_marker
                        if IH_FULL_PREFIX_MATERIALIZATION
                        else plan is qps_reuse_plan
                    )
                    if is_reuse:
                        refresh_first_document_prefix(
                            f"first-document-prefix-refresh-{request_tag}"
                        )
                    output = _ih_generate(
                        input_ids,
                        max_new,
                        (
                            None
                            if is_reuse and IH_FULL_PREFIX_MATERIALIZATION
                            else plan
                        ),
                        request_tag=request_tag,
                        request_id=request_id,
                        cache_key=(
                            materialized_cache_key if is_reuse else None
                        ),
                    )
                    expected_cached_tokens = (
                        IH_CHUNK_TOKENS
                        if IH_FIRST_DOCUMENT_PREFIX
                        else query_start
                    )
                    if (
                        is_reuse
                        and output.get("cached_tokens")
                        != expected_cached_tokens
                    ):
                        raise _IHPerformanceMeasurementError(
                            "QPS reuse request missed the exact cached "
                            f"prefix: observed={output.get('cached_tokens')!r} "
                            f"expected={expected_cached_tokens}"
                        )
                    return output

            qps_results = _ih_measure_paired_qps(
                full0,
                1,
                qps_reuse_plan,
                concurrencies=IH_QPS_CONCURRENCIES,
                paired_waves=IH_QPS_WAVES,
                warmup_waves=IH_QPS_WARMUP_WAVES,
                request_tag="qps-paired",
                seed=SEED,
                generate_fn=qps_generate_fn,
            )
            for point in qps_results["reuse"]["points"]:
                for request_id in point.get("formal_evidence_request_ids", ()):
                    if IH_FULL_PREFIX_MATERIALIZATION:
                        prefix_materialization_request_ids.add(request_id)
                        prefix_online_rows_by_request[request_id] = len(
                            queries[0][1]
                        )
                    else:
                        expected_mla_request_ids.add(request_id)
                        expected_mla_q_rows_by_request[request_id] = len(full0) - (
                            IH_CHUNK_TOKENS if IH_FIRST_DOCUMENT_PREFIX else 0
                        )
                        expected_mla_position_start_by_request[request_id] = (
                            IH_CHUNK_TOKENS if IH_FIRST_DOCUMENT_PREFIX else 0
                        )
                if not IH_FULL_PREFIX_MATERIALIZATION:
                    audit_allowed_mla_request_ids.update(
                        map(str, point.get("audit_request_ids", ()))
                    )

        print("-" * W)
        time.sleep(0.2)
        runtime_metrics = _ih_read_runtime_metrics(metric_start)
        runtime_metrics["shared_snapshot_audit"] = shared_snapshot_audit
        min_cos = min(cosines) if cosines else 0.0
        top1_rate = top1_hits / max(1, quality_pair_count)
        dense_f1 = sum(dense_f1s) / max(1, len(dense_f1s))
        reuse_f1 = sum(reuse_f1s) / max(1, len(reuse_f1s))
        dense_em = sum(dense_ems) / max(1, len(dense_ems))
        reuse_em = sum(reuse_ems) / max(1, len(reuse_ems))
        f1_retention = reuse_f1 / dense_f1 if dense_f1 > 1e-9 else None
        f1_gate = (
            f1_retention >= IH_MIN_F1_RETENTION
            if f1_retention is not None
            else reuse_f1 + 1e-9 >= dense_f1
        )
        em_retention = reuse_em / dense_em if dense_em > 1e-9 else None
        em_gate = (
            em_retention >= IH_MIN_EM_RETENTION
            if em_retention is not None
            else reuse_em + 1e-9 >= dense_em
        )
        min_token_agreement = min(token_agreements) if token_agreements else 0.0
        quality_pass = (
            top1_rate >= IH_MIN_TOP1_RATE
            and min_cos >= IH_MIN_COSINE
            and min_token_agreement >= IH_MIN_TOKEN_AGREEMENT
            and f1_gate
            and em_gate
            and dense_f1 >= IH_MIN_DENSE_F1
            and reuse_f1 >= IH_MIN_REUSE_F1
            and dense_em >= IH_MIN_DENSE_EM
            and reuse_em >= IH_MIN_REUSE_EM
        )
        performance_claim_qualification = (
            _ih_finalize_performance_claim_qualification(
                sampling_qualification,
                dense_ttft_samples=d_ttft,
                reuse_ttft_samples=r_ttft,
                expected_ttft_iters=IH_TTFT_ITERS,
                qps_results=qps_results,
            )
        )
        if IH_MLA_OFF_DIAGNOSTIC_ONLY:
            performance_claim_qualification = (
                _ih_apply_diagnostic_claim_gate(
                    performance_claim_qualification,
                    IH_MLA_OFF_DIAGNOSTIC_ABLATION,
                )
            )
        performance_claim_qualification = (
            _ih_apply_qualification_only_claim_gate(
                performance_claim_qualification,
                IH_MLA_OFF_QUALIFICATION_ONLY,
            )
        )
        expected_runtime_rows = sum(expected_mla_q_rows_by_request.values())
        runtime_coverage = runtime_metrics["full_rows"] / max(1, expected_runtime_rows)
        minimum_mla_off_samples = 0
        expected_mla_layer_ids = []
        missing_mla_request_layers = {}
        missing_mla_request_forwards = []
        missing_mla_forward_layers = {}
        unexpected_mla_forward_layers = {}
        invalid_mla_forwards = {}
        mla_position_coverage_errors = {}
        mla_row_geometry_errors = {}
        measured_mla_fallback_request_ids = []
        measured_mla_evidence_error_request_ids = []
        mla_requests_without_reuse = []
        failed_mla_request_ids = []
        runtime_evidence_failed_request_ids = []
        formal_failed_mla_request_ids = []
        expected_mla_forward_count = 0
        observed_mla_forward_count = 0
        measured_mla_off_samples = 0
        runtime_evidence_pass = False
        full_local_sanity_requested = False
        full_local_sanity_errors = {}
        full_local_sanity_failed_request_ids = []
        full_local_sanity_pass = None
        prefix_materialization_runtime = None
        if IH_MLA_OFFLOAD:
            expected_mla_layer_ids = [
                int(layer_id)
                for layer_id in capacity_geometry["server_policy_manifest"][
                    "runtime_local_layer_ids"
                ]
            ]
            mla_validation = _ih_validate_mla_runtime_metrics(
                runtime_metrics,
                expected_request_ids=expected_mla_request_ids,
                expected_q_rows_by_request=expected_mla_q_rows_by_request,
                expected_position_start_by_request=(
                    expected_mla_position_start_by_request
                ),
                expected_layer_ids=expected_mla_layer_ids,
                expected_request_count=len(expected_mla_request_ids),
                expected_head_counts_by_layer=capacity_geometry[
                    "rank0_head_counts_by_layer"
                ],
                max_q_rows_per_forward=verified_runtime_server_policy[
                    "max_prefill_tokens"
                ],
                full_local_sanity=IH_MLA_OFF_REFRESH_LAYER_STRIDE == 1,
                num_model_layers=capacity_geometry["num_model_layers"],
                num_attention_heads=capacity_geometry[
                    "num_attention_heads"
                ],
                tensor_parallel_size=capacity_geometry["tp_size"],
                transfer_audit_required=True,
                transfer_audit_device_cache_enabled=bool(
                    verified_runtime_server_policy.get(
                        "mla_off_device_cache_enabled", False
                    )
                ),
                transfer_audit_device_max_bytes=int(
                    verified_runtime_server_policy.get(
                        "mla_off_device_max_bytes", 0
                    )
                ),
                transfer_audit_allowed_request_ids=(
                    audit_allowed_mla_request_ids
                    - expected_mla_request_ids
                ),
                compact_woa_expected_mode=(
                    "forbidden"
                    if not IH_MLA_OFF_COMPACT_WOA
                    else (
                        "required"
                        if IH_MLA_OFF_REFRESH_LAYER_STRIDE != 1
                        and any(
                            type(counts.get("local")) is int
                            and counts["local"] > 0
                            and type(counts.get("global")) is int
                            and counts["global"] == 0
                            for counts in capacity_geometry[
                                "rank0_head_counts_by_layer"
                            ].values()
                        )
                        else "not_applicable"
                    )
                ),
                allow_sparse_positions=IH_COMBINED_HEADSPLIT_ROW_SPARSE,
                composite_receipt_required=(
                    IH_COMBINED_HEADSPLIT_ROW_SPARSE
                ),
                expected_diagnostic_ablation=(
                    IH_MLA_OFF_DIAGNOSTIC_ABLATION
                ),
            )
            runtime_metrics["mla_off_transfer_audit"] = mla_validation[
                "mla_off_transfer_audit"
            ]
            runtime_metrics["compact_woa_evidence"] = mla_validation[
                "compact_woa_evidence"
            ]
            runtime_metrics["mla_off_composite_receipts"] = mla_validation[
                "mla_off_composite_receipts"
            ]
            minimum_mla_off_samples = mla_validation[
                "minimum_mla_off_samples"
            ]
            expected_mla_forward_count = mla_validation[
                "expected_mla_forward_count"
            ]
            observed_mla_forward_count = mla_validation[
                "observed_mla_forward_count"
            ]
            measured_mla_off_samples = mla_validation[
                "measured_mla_off_samples"
            ]
            missing_mla_request_forwards = mla_validation[
                "missing_mla_request_forwards"
            ]
            missing_mla_request_layers = mla_validation[
                "missing_mla_request_layers"
            ]
            missing_mla_forward_layers = mla_validation[
                "missing_mla_forward_layers"
            ]
            unexpected_mla_forward_layers = mla_validation[
                "unexpected_mla_forward_layers"
            ]
            invalid_mla_forwards = mla_validation["invalid_mla_forwards"]
            mla_position_coverage_errors = mla_validation[
                "mla_position_coverage_errors"
            ]
            mla_row_geometry_errors = mla_validation[
                "mla_row_geometry_errors"
            ]
            measured_mla_fallback_request_ids = mla_validation[
                "measured_mla_fallback_request_ids"
            ]
            measured_mla_evidence_error_request_ids = mla_validation[
                "measured_mla_evidence_error_request_ids"
            ]
            mla_requests_without_reuse = mla_validation[
                "mla_requests_without_reuse"
            ]
            failed_mla_request_ids = mla_validation[
                "failed_mla_request_ids"
            ]
            runtime_evidence_failed_request_ids = mla_validation[
                "runtime_evidence_failed_request_ids"
            ]
            formal_failed_mla_request_ids = mla_validation[
                "formal_failed_mla_request_ids"
            ]
            runtime_evidence_pass = bool(
                mla_validation["runtime_evidence_pass"]
                and shared_snapshot_audit.get("pass", False)
            )
            full_local_sanity_requested = mla_validation[
                "full_local_sanity_requested"
            ]
            full_local_sanity_errors = mla_validation[
                "full_local_sanity_errors"
            ]
            full_local_sanity_failed_request_ids = mla_validation[
                "full_local_sanity_failed_request_ids"
            ]
            full_local_sanity_pass = mla_validation[
                "full_local_sanity_pass"
            ]
            runtime_metrics["unscoped_mla_head_row_saving"] = runtime_metrics[
                "mla_head_row_saving"
            ]
            for key in (
                "measured_reused_local_head_rows",
                "measured_online_local_head_rows",
                "measured_online_global_head_rows",
                "measured_mla_head_row_saving",
                "measured_scoped_mla_head_row_saving",
                "measured_full_model_mla_head_row_saving",
                "measured_full_model_reused_local_head_rows",
                "full_model_mla_head_row_denominator",
                "full_model_num_layers",
                "full_model_num_attention_heads",
                "full_model_tensor_parallel_size",
            ):
                runtime_metrics[key] = mla_validation[key]
            runtime_metrics["mla_head_row_saving"] = mla_validation[
                "measured_mla_head_row_saving"
            ]
            if IH_FULL_PREFIX_MATERIALIZATION:
                prefix_materialization_runtime = (
                    _ih_validate_prefix_materialization_runtime(
                        runtime_metrics,
                        expected_request_ids=prefix_materialization_request_ids,
                        online_rows_by_request=prefix_online_rows_by_request,
                        query_start=query_start,
                        query_suffix_rows=len(queries[0][1]),
                        expected_layer_ids=expected_mla_layer_ids,
                        expected_head_counts_by_layer=capacity_geometry[
                            "rank0_head_counts_by_layer"
                        ],
                        tensor_parallel_size=capacity_geometry["tp_size"],
                        expected_device_cache_enabled=bool(
                            verified_runtime_server_policy.get(
                                "mla_off_device_cache_enabled", False
                            )
                        ),
                        expected_device_max_bytes=int(
                            verified_runtime_server_policy.get(
                                "mla_off_device_max_bytes", 0
                            )
                        ),
                        materialization_runtime_pass=bool(
                            mla_validation["runtime_evidence_pass"]
                        ),
                    )
                )
                runtime_metrics["prefix_materialization"] = (
                    prefix_materialization_runtime
                )
                runtime_metrics["online_row_saving"] = (
                    prefix_materialization_runtime[
                        "online_prompt_row_saving"
                    ]
                )
                runtime_evidence_pass = bool(
                    runtime_evidence_pass
                    and prefix_materialization_runtime["pass"]
                )
            runtime_pass = bool(
                mla_validation["pass"]
                and shared_snapshot_audit.get("pass", False)
                and (
                    not IH_FULL_PREFIX_MATERIALIZATION
                    or prefix_materialization_runtime["pass"]
                )
            )
        else:
            runtime_pass = (
                runtime_metrics["samples"] > 0
                and runtime_metrics["fallbacks"] == 0
                and runtime_coverage >= 0.99
            )
            runtime_evidence_pass = runtime_pass
        speedup = dm / max(rm, 1e-9)
        row_saving = runtime_metrics["online_row_saving"]
        if IH_COMBINED_HEADSPLIT_ROW_SPARSE:
            row_ratio_evidence = (
                _pro0813_scale_policy.pro0813_realized_active_ratio_gate(
                    requested_active_ratio_floor,
                    min_realized_active_ratio,
                    row_saving,
                    required=True,
                )
            )
            runtime_metrics.update(row_ratio_evidence)
            runtime_pass = bool(
                runtime_pass
                and row_ratio_evidence["realized_active_ratio_pass"]
            )
            runtime_evidence_pass = bool(
                runtime_evidence_pass
                and row_ratio_evidence["realized_active_ratio_pass"]
            )
        else:
            row_ratio_evidence = {
                "requested_active_ratio": None,
                "min_realized_active_ratio": None,
                "actual_realized_active_ratio": None,
                "realized_active_ratio_pass": None,
                "realized_active_ratio_required": False,
            }
            runtime_metrics.update(row_ratio_evidence)
        scoped_head_row_saving = runtime_metrics["mla_head_row_saving"]
        full_model_head_row_saving = runtime_metrics.get(
            "measured_full_model_mla_head_row_saving"
        )
        reported_mla_request_ids = expected_mla_request_ids
        claim_diagnostic_only = bool(
            IH_MLA_OFF_DIAGNOSTIC_ONLY
            or (
                IH_PERFORMANCE_DIAGNOSTIC_ONLY
                and not IH_MLA_OFF_QUALIFICATION_ONLY
            )
        )
        execution_qualification = _ih_execution_qualification(
            diagnostic_only=claim_diagnostic_only,
            quality_pass=bool(quality_pass),
            runtime_pass=bool(runtime_pass),
            runtime_evidence_pass=bool(runtime_evidence_pass),
            quality_pair_count=quality_pair_count,
            expected_quality_pair_count=len(queries) * IH_QUALITY_REPEATS,
            dense_ttft_samples=d_ttft,
            reuse_ttft_samples=r_ttft,
            expected_ttft_iters=IH_TTFT_ITERS,
            qps_required=IH_MEASURE_QPS,
            qps_results=qps_results,
            expected_qps_concurrencies=IH_QPS_CONCURRENCIES,
            expected_qps_warmup_waves=IH_QPS_WARMUP_WAVES,
            expected_qps_waves=IH_QPS_WAVES,
        )
        execution_pass = execution_qualification["execution_pass"]
        diagnostic_success = execution_qualification["diagnostic_success"]
        formal_runtime_pass = bool(
            runtime_pass
            and not claim_diagnostic_only
            and not IH_MLA_OFF_QUALIFICATION_ONLY
        )
        if claim_diagnostic_only or IH_MLA_OFF_QUALIFICATION_ONLY:
            # Attribution arms deliberately remove half of the production
            # algorithm.  Their latency/quality numbers are diagnostic data,
            # never formal TTFT/QPS or production-reuse evidence.
            performance_pass = False
        elif IH_MLA_OFFLOAD:
            performance_pass = (
                runtime_pass
                and performance_claim_qualification["eligible"]
                and speedup >= IH_MIN_SPEEDUP
                and full_model_head_row_saving is not None
                and full_model_head_row_saving >= IH_MIN_HEAD_ROW_SAVING
            )
        else:
            performance_pass = (
                runtime_pass
                and performance_claim_qualification["eligible"]
                and speedup >= IH_MIN_SPEEDUP
                and row_saving is not None
                and row_saving >= IH_MIN_ROW_SAVING
            )
        overall_pass = bool(
            not claim_diagnostic_only
            and not IH_MLA_OFF_QUALIFICATION_ONLY
            and quality_pass
            and formal_runtime_pass
            and performance_pass
        )
        dense_req_s = quality_pair_count / max(sum(dense_quality_e2e), 1e-9)
        reuse_req_s = quality_pair_count / max(sum(reuse_quality_e2e), 1e-9)
        dense_prompt_tps = logical_prompt_tokens / max(sum(dense_quality_e2e), 1e-9)
        reuse_prompt_tps = logical_prompt_tokens / max(sum(reuse_quality_e2e), 1e-9)
        print(
            f" SUMMARY: streaming_ttft_speedup={dm / max(rm, 1e-6):.2f}x "
            f"online_row_saving={runtime_metrics['online_row_saving']} "
            f"scoped_mla_head_row_saving={scoped_head_row_saving} "
            f"full_model_mla_head_row_saving={full_model_head_row_saving} "
            f"mla_samples={measured_mla_off_samples}/"
            f">={minimum_mla_off_samples} "
            f"formal_mla_requests="
            f"{len(reported_mla_request_ids) - len(formal_failed_mla_request_ids)}/"
            f"{len(reported_mla_request_ids)} "
            f"quality={quality_pass} formal_reuse={formal_runtime_pass} "
            f"evidence={runtime_evidence_pass} "
            f"row_ratio={row_ratio_evidence['actual_realized_active_ratio']}/"
            f">={row_ratio_evidence['min_realized_active_ratio']} "
            f"row_ratio_pass="
            f"{row_ratio_evidence['realized_active_ratio_pass']} "
            f"full_local_sanity={full_local_sanity_pass} "
            f"performance_claim_eligible="
            f"{performance_claim_qualification['eligible']} "
            f"execution={execution_pass} "
            f"diagnostic_success={diagnostic_success} "
            f"performance={performance_pass} overall={overall_pass}"
        )
        print(
            f" sequential service rate: dense={dense_req_s:.3f} req/s, "
            f"reuse={reuse_req_s:.3f} req/s; logical prompt "
            f"dense={dense_prompt_tps:.1f}, reuse={reuse_prompt_tps:.1f} tok/s"
        )
        print(
            " online_row_saving is measured from selected transformer rows; it "
            "is not total-model FLOPs or GPU energy saving."
        )
        print(
            " mla_head_row_saving counts omitted logical local-head rows at the "
            "attention output boundary; it is also not total-model FLOPs saving."
        )
        print(
            " full_model_mla_head_row_saving uses all model layers, all 128 "
            "logical heads and every evidence request row; performance claims "
            "are gated on this conservative metric."
        )
        print(
            " compact_woa_evidence certifies successful indexed inverse-RoPE/"
            "wo_a row activation only; it is not a FLOPs, utilization, or "
            "energy-saving measurement."
        )
        print("=" * W)

        try:
            code_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip()
        except Exception:
            code_revision = "unknown"
        reproducibility = _ih_reproducibility_manifest(
            DATASETS[0] if DATASETS else "hotpotqa",
            data_selection=data_selection,
            prompt_manifest=prompt_manifest,
            qualification_provenance=qualification_provenance,
        )
        result_claim_status, result_claim_reason = _ih_result_claim_mode(
            mla_diagnostic_only=IH_MLA_OFF_DIAGNOSTIC_ONLY,
            qualification_only=IH_MLA_OFF_QUALIFICATION_ONLY,
            performance_diagnostic_only=(
                IH_PERFORMANCE_DIAGNOSTIC_ONLY
            ),
        )
        formal_claim_eligible = bool(
            result_claim_status == "formal_candidate"
            and performance_claim_qualification["eligible"]
        )
        claim_ineligible = not formal_claim_eligible
        claim_ineligible_reason = result_claim_reason
        if not claim_ineligible_reason and claim_ineligible:
            claim_ineligible_reason = "performance_claim_qualification_failed"
        result = {
            "test": (
                f"redknot-dsv4-pure-mla-{len(chunks)}x{chunk_tokens}"
                if IH_MLA_OFFLOAD
                else (
                    f"redknot-dsv4-pro0813-{len(chunks)}x{chunk_tokens}-"
                    f"{IH_SELECTION_POLICY}"
                )
            ),
            "model": MODEL_PATH,
            "accelerator_contract": "8x NVIDIA B300 / Blackwell SM103",
            "redknot_variant": PRO0813_VARIANT,
            "redknot_geometry_digest": PRO0813_GEOMETRY_DIGEST,
            "official_config_sha256": PRO0813_CONFIG_SHA256,
            "code_revision": code_revision,
            "reproducibility": reproducibility,
            "claim_status": {
                "status": result_claim_status,
                "reason": result_claim_reason,
                "claim_ineligible": claim_ineligible,
                "claim_ineligible_reason": claim_ineligible_reason,
                "qualification_only": IH_MLA_OFF_QUALIFICATION_ONLY,
                "diagnostic_ablation": (
                    IH_MLA_OFF_DIAGNOSTIC_ABLATION
                    if IH_MLA_OFF_DIAGNOSTIC_ONLY
                    else None
                ),
            },
            "claim_ineligible": claim_ineligible,
            "claim_ineligible_reason": claim_ineligible_reason,
            "formal_claim": {
                "eligible": formal_claim_eligible,
                "pass": overall_pass,
                "status": result_claim_status,
                "ineligible_reason": claim_ineligible_reason,
            },
            "execution_pass": execution_pass,
            "diagnostic_success": diagnostic_success,
            "execution_gate": execution_qualification,
            "config": {
                "dataset": DATASETS[0] if DATASETS else "hotpotqa",
                "qualification_provenance": qualification_provenance,
                "num_model_layers": PRO0813_NUM_LAYERS,
                "num_attention_heads": PRO0813_NUM_HEADS,
                "index_topk": PRO0813_INDEX_TOPK,
                "tp_size": PRO0813_TP_SIZE,
                "dense_layer_ids": list(PRO0813_DENSE_LAYER_IDS),
                "reusable_layer_ids": list(PRO0813_REUSABLE_LAYER_IDS),
                "seed": SEED,
                "chunk_tokens": chunk_tokens,
                "merged_prefill_tokens": merged_prefill_tokens,
                "num_chunks": len(chunks),
                "chunk_hashes": segment_hashes,
                "num_queries": len(queries),
                # Keep the legacy key truthful under manifest replay and expose
                # the environment request separately from the frozen selection.
                "data_row_offset": int(
                    data_selection["selection"].get(
                        "row_offset", IH_DATA_ROW_OFFSET
                    )
                ),
                "data_row_offset_requested": IH_DATA_ROW_OFFSET,
                "data_row_offset_effective": int(
                    data_selection["selection"].get(
                        "row_offset", IH_DATA_ROW_OFFSET
                    )
                ),
                "data_manifest_input": IH_DATA_MANIFEST or None,
                "data_manifest_output": IH_DATA_MANIFEST_OUT or None,
                "data_exclude_manifests": list(IH_DATA_EXCLUDE_MANIFESTS),
                "data_selection_sha256": data_selection["selection_sha256"],
                "pure_prompt_mode": IH_PURE_PROMPT_MODE,
                "long_output_target_tokens": IH_LONG_OUTPUT_TOKENS,
                "answer_style": (
                    "direct_answer_plus_document_evidence_v1"
                    if IH_LONG_OUTPUT_TOKENS
                    else "shortest_exact_span_v1"
                ),
                "prompt_manifest_input": IH_PROMPT_MANIFEST or None,
                "prompt_manifest_output": IH_PROMPT_MANIFEST_OUT or None,
                "prompt_manifest_sha256": (
                    prompt_manifest["prompt_manifest_sha256"]
                    if prompt_manifest is not None
                    else None
                ),
                "full_input_ids_sha256": _ih_chunk_hash(full0),
                "full_input_tokens": len(full0),
                "online_suffix_tokens": len(queries[0][1]),
                "corpus_row_ids": [
                    row["row_id"]
                    for chunk in data_selection["selection"]["chunks"]
                    for row in chunk["rows"]
                ],
                "query_row_ids": [
                    query["row_id"]
                    for query in data_selection["selection"]["queries"]
                ],
                "quality_repeats": IH_QUALITY_REPEATS,
                "boundary": IH_BOUNDARY,
                "boundary_online_document_rows": (
                    (IH_NUM_CHUNKS - 1) * IH_BOUNDARY
                    if IH_MLA_OFFLOAD
                    else None
                ),
                "reusable_document_rows": (
                    (
                        (IH_NUM_CHUNKS - 1)
                        * (chunk_tokens - IH_BOUNDARY)
                        if IH_FIRST_DOCUMENT_PREFIX
                        else (
                            chunk_tokens
                            + (IH_NUM_CHUNKS - 1)
                            * (chunk_tokens - IH_BOUNDARY)
                        )
                    )
                    if IH_MLA_OFFLOAD
                    else None
                ),
                "offline_snapshot_prefill_tokens": (
                    len(chunks) * chunk_tokens
                    if IH_MLA_OFFLOAD
                    else None
                ),
                "reuse_strategy": (
                    (
                        (
                            "first_document_radix_prefix_plus_"
                            "independent_rope_zoff_headsplit_plus_"
                            "checkpoint_row_sparse_indexer_recompute_v1"
                        )
                        if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                        else (
                            (
                            "first_document_radix_prefix_plus_"
                            "fullrow_seed_plus_independent_rope_relocation_"
                            "pure_mla_restore_v1"
                            )
                            if IH_FIRST_DOCUMENT_PREFIX
                            else (
                                "independent_redknot_prefix_materialization_v1"
                                if IH_FULL_PREFIX_MATERIALIZATION
                                else "independent_position0_rope_relocation_"
                                "pure_mla_headsplit_fullscope"
                            )
                        )
                    )
                    if IH_MLA_OFFLOAD
                    else "legacy_selected_row_reuse"
                ),
                "combined_headsplit_row_sparse": (
                    IH_COMBINED_HEADSPLIT_ROW_SPARSE
                ),
                "combined_row_sparse_active_ratio": (
                    IH_ACTIVE_BUDGET_RATIO
                    if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                    else None
                ),
                "combined_row_sparse_requested_active_ratio": (
                    requested_active_ratio_floor
                    if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                    else None
                ),
                "combined_row_sparse_min_realized_active_ratio": (
                    min_realized_active_ratio
                    if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                    else None
                ),
                "combined_row_sparse_checkpoint_max_islands": (
                    IH_CHECKPOINT_MAX_ISLANDS
                    if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                    else None
                ),
                "combined_row_sparse_required_checkpoint_islands": (
                    required_checkpoint_islands
                    if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                    else None
                ),
                "combined_query_protection_tokens": (
                    IH_QUERY_PROTECTION_TOKENS
                    if IH_COMBINED_HEADSPLIT_ROW_SPARSE
                    else None
                ),
                "generalized_adaptive_controller": (
                    {
                        "enabled": True,
                        "policy": _IH_GENERALIZED_ADAPTIVE_POLICY,
                        "bucket_counts": dict(
                            Counter(
                                row["generalized_adaptive_controller"]["bucket"]
                                for row in quality_rows
                                if row.get("generalized_adaptive_controller")
                            )
                        ),
                    }
                    if IH_GENERALIZED_ADAPTIVE_CONTROLLER
                    else {"enabled": False}
                ),
                "prefix_materialization": IH_PREFIX_MATERIALIZATION,
                "prefix_materialization_scope": (
                    IH_PREFIX_MATERIALIZATION_SCOPE
                ),
                "radix_eviction_policy": IH_RADIX_EVICTION_POLICY,
                "materialized_prefix_tokens": (
                    (
                        IH_CHUNK_TOKENS
                        if IH_FIRST_DOCUMENT_PREFIX
                        else query_start
                    )
                    if IH_PREFIX_MATERIALIZATION
                    else 0
                ),
                "materialized_cache_key_sha256": (
                    "sha256:"
                    + hashlib.sha256(
                        str(materialized_cache_key).encode("utf-8")
                    ).hexdigest()
                    if materialized_cache_key is not None
                    else None
                ),
                "radix_prefix_receipt_key": (
                    radix_prefix_receipt_key
                    if IH_FIRST_DOCUMENT_PREFIX
                    else None
                ),
                "materialization_producer_cached_tokens": (
                    materialization_result["cached_tokens"]
                    if materialization_result is not None
                    else None
                ),
                "materialization_sentinel_token": (
                    materialization_sentinel_token
                    if IH_PREFIX_MATERIALIZATION
                    else None
                ),
                "snapshot_cached_tokens": snapshot_cached_tokens,
                "reuse_cached_tokens": reuse_cached_tokens,
                "restore_pipeline_group_layers": int(
                    IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS
                ),
                "progressive_topk_schedule": PROGRESSIVE_TOPK_SCHEDULE or None,
                "progressive_topk_semantics": (
                    "per_layer_routed_expert_assignment_reduction"
                    if PROGRESSIVE_TOPK_SCHEDULE
                    else "native_topk"
                ),
                "token_sparse_ffn_enabled": SPARSE_FFN,
                "three_way_closure": THREE_WAY_CLOSURE,
                "token_sparse_ffn_importance": (
                    FFN_IMPORTANCE if SPARSE_FFN else None
                ),
                "token_sparse_ffn_mass_thresh": FFN_MASS if SPARSE_FFN else None,
                "token_sparse_ffn_mass_thresh_deep": (
                    FFN_MASS_DEEP if SPARSE_FFN else None
                ),
                "token_sparse_ffn_min_full_ratio": (
                    FFN_MIN_FULL_RATIO if SPARSE_FFN else None
                ),
                "token_sparse_ffn_max_full_ratio": (
                    FFN_MAX_FULL_RATIO if SPARSE_FFN else None
                ),
                "token_sparse_ffn_dense_suffix_layers": (
                    FFN_DENSE_SUFFIX_LAYERS if SPARSE_FFN else None
                ),
                "token_sparse_ffn_boundary_tokens": (
                    FFN_BOUNDARY_TOKENS if SPARSE_FFN else None
                ),
                "token_sparse_ffn_block_tokens": (
                    FFN_BLOCK_TOKENS if SPARSE_FFN else None
                ),
                "token_sparse_ffn_freeze_block_selection": (
                    FFN_FREEZE_BLOCK_SELECTION if SPARSE_FFN else None
                ),
                "adaptive_topk": os.environ.get("REDKNOT_ADAPTIVE_TOPK", "0")
                == "1",
                "adaptive_topk_plan_scoped": os.environ.get(
                    "REDKNOT_ADAPTIVE_TOPK_PLAN_SCOPED", "0"
                )
                == "1",
                "adaptive_topk_cumulative_mass": os.environ.get(
                    "REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS"
                ),
                "adaptive_topk_buckets": os.environ.get(
                    "REDKNOT_ADAPTIVE_TOPK_BUCKETS"
                ),
                "adaptive_topk_physical_compaction": os.environ.get(
                    "REDKNOT_ADAPTIVE_TOPK_PHYSICAL_COMPACTION", "0"
                )
                == "1",
                "indexer_query_online": bool(IH_MLA_OFFLOAD),
                "indexer_online_semantics": (
                    "restore_k_and_state_recompute_query_q_score_top1024"
                    if IH_MLA_OFFLOAD
                    else None
                ),
                # These knobs belong only to the legacy selected-row path.
                # Publishing their parsed defaults for a pure run previously
                # made a correct pure plan look like checkpoint-islands reuse.
                "selection_policy": (
                    None if IH_MLA_OFFLOAD else IH_SELECTION_POLICY
                ),
                "checkpoint_stride_tokens": (
                    None if IH_MLA_OFFLOAD else IH_CHECKPOINT_STRIDE
                ),
                "checkpoint_max_islands": (
                    None if IH_MLA_OFFLOAD else IH_CHECKPOINT_MAX_ISLANDS
                ),
                "active_token_budget_ratio": (
                    None if IH_MLA_OFFLOAD else IH_ACTIVE_BUDGET_RATIO
                ),
                "hot_max_per_segment_ratio": (
                    None if IH_MLA_OFFLOAD else IH_HOT_MAX_PER_SEGMENT_RATIO
                ),
                "relevance_last": IH_RELEVANCE_LAST,
                "relevance_first": IH_RELEVANCE_FIRST,
                "skip_prefix_recompute": (
                    None if IH_MLA_OFFLOAD else IH_SKIP_PREFIX_RECOMPUTE
                ),
                "mla_offload": IH_MLA_OFFLOAD,
                "mla_off_execution_profile": (
                    IH_MLA_OFF_EXECUTION_PROFILE if IH_MLA_OFFLOAD else None
                ),
                "mla_off_diagnostic_ablation": (
                    IH_MLA_OFF_DIAGNOSTIC_ABLATION
                    if IH_MLA_OFFLOAD
                    else None
                ),
                "mla_dense_prefix_layers": MLA_DENSE_PREFIX,
                "mla_dense_suffix_layers": MLA_DENSE_SUFFIX,
                "mla_offline_online_layer_ids": (
                    list(PRO0813_REUSABLE_LAYER_IDS) if IH_MLA_OFFLOAD else []
                ),
                "design_targets": {
                    "ttft_speedup": 5.0,
                    "quality_regression_max": 0.02,
                    "compute_saving_min": 0.70,
                    "status": "must_be_verified_by_real_experiment",
                },
                "mla_off_compact_woa": IH_MLA_OFF_COMPACT_WOA,
                "reuse_heads_full_scope": IH_REUSE_HEADS_FULL_SCOPE,
                "mla_off_max_bytes_per_tp_rank": IH_MLA_OFF_MAX_BYTES,
                "min_gpu_free_mib_before_launch": IH_MIN_GPU_FREE_MIB,
                "mla_off_projected_bytes_per_tp_rank": projected_mla_off_bytes,
                "mla_off_capacity_geometry": capacity_geometry,
                "verified_runtime_server_policy": (
                    verified_runtime_server_policy if IH_MLA_OFFLOAD else None
                ),
                "runtime_accelerator": (
                    {
                        "device_count": verified_runtime_server_policy[
                            "accelerator_device_count"
                        ],
                        "device_names": verified_runtime_server_policy[
                            "accelerator_device_names"
                        ],
                        "device_capabilities": verified_runtime_server_policy[
                            "accelerator_device_capabilities"
                        ],
                        "jit_rmsnorm": verified_runtime_server_policy[
                            "jit_rmsnorm"
                        ],
                    }
                    if IH_MLA_OFFLOAD
                    else None
                ),
                "mla_off_refresh_layer_stride": IH_MLA_OFF_REFRESH_LAYER_STRIDE,
                "mla_off_certified_max_context_tokens": (
                    IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS
                ),
                "mla_off_qualification_only": (
                    IH_MLA_OFF_QUALIFICATION_ONLY
                ),
                "mla_off_qualification_max_context_tokens": (
                    IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS
                ),
                "mla_off_hot_expand_tokens": IH_MLA_OFF_HOT_EXPAND_TOKENS,
                "ttft_chunk_order": chunk_order0,
                "ttft_iters": IH_TTFT_ITERS,
                "ttft_warmup": IH_TTFT_WARMUP,
                "strict_performance_claims": IH_STRICT_PERFORMANCE_CLAIMS,
                "require_model_internal_ttft": IH_REQUIRE_MODEL_TTFT,
                "measure_qps": IH_MEASURE_QPS,
                "qps_waves": IH_QPS_WAVES,
                "qps_warmup_waves": IH_QPS_WARMUP_WAVES,
                "qps_concurrencies": list(IH_QPS_CONCURRENCIES),
                "max_new_tokens": IH_MAX_NEW,
                "rank_log_dir": IH_RANK_LOG_DIR,
                "timing_diagnostics": os.environ.get("REDKNOT_V4_TIMING", "0") == "1",
                "system_optimizations": {
                    "tilelang_mhc_pre": os.environ.get(
                        "SGLANG_OPT_USE_TILELANG_MHC_PRE", "0"
                    ) == "1",
                    "tilelang_mhc_post": os.environ.get(
                        "SGLANG_OPT_USE_TILELANG_MHC_POST", "0"
                    ) == "1",
                    "deepgemm_hc_prenorm": os.environ.get(
                        "SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0"
                    ) == "1",
                    "cublas_woa_fastpath": os.environ.get(
                        "REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH", "0"
                    ) == "1",
                },
            },
            "latency": {
                "metric": "client_stream_first_output_token",
                "protocol": "sse_or_explicit_json_lines",
                "fail_closed": True,
                "dense_samples": d_ttft,
                "reuse_samples": r_ttft,
                "dense_p50": dm,
                "reuse_p50": rm,
                "dense_p95": _ih_percentile(d_ttft, 0.95),
                "reuse_p95": _ih_percentile(r_ttft, 0.95),
                "speedup": speedup,
                "model_internal": {
                    "metric": "forward_entry_time_to_prefill_finished_time",
                    "dense_samples": d_model_ttft,
                    "reuse_samples": r_model_ttft,
                    "dense_p50": (
                        statistics.median(d_model_ttft)
                        if d_model_ttft
                        else None
                    ),
                    "reuse_p50": (
                        statistics.median(r_model_ttft)
                        if r_model_ttft
                        else None
                    ),
                    "speedup": (
                        statistics.median(d_model_ttft)
                        / max(statistics.median(r_model_ttft), 1e-9)
                        if d_model_ttft and r_model_ttft
                        else None
                    ),
                    "dense_queue_samples": d_queue,
                    "reuse_queue_samples": r_queue,
                    "dense_server_non_model_samples": d_non_model,
                    "reuse_server_non_model_samples": r_non_model,
                },
            },
            "performance_measurement": {
                "claim_qualification": performance_claim_qualification,
                "qps": qps_results,
            },
            "runtime": {
                **runtime_metrics,
                "expected_full_rows": expected_runtime_rows,
                "minimum_mla_off_samples": minimum_mla_off_samples,
                "expected_mla_forward_count": expected_mla_forward_count,
                "observed_mla_forward_count": observed_mla_forward_count,
                "measured_mla_off_samples": measured_mla_off_samples,
                "expected_mla_request_ids": sorted(reported_mla_request_ids),
                "expected_mla_q_rows_by_request": dict(
                    sorted(
                        (
                            prefix_online_rows_by_request
                            if IH_FULL_PREFIX_MATERIALIZATION
                            else expected_mla_q_rows_by_request
                        ).items()
                    )
                ),
                "expected_mla_position_start_by_request": dict(
                    sorted(expected_mla_position_start_by_request.items())
                ),
                "prefix_materialization_runtime": prefix_materialization_runtime,
                "expected_mla_layer_ids": expected_mla_layer_ids,
                "missing_mla_request_forwards": missing_mla_request_forwards,
                "missing_mla_request_layers": missing_mla_request_layers,
                "missing_mla_forward_layers": missing_mla_forward_layers,
                "unexpected_mla_forward_layers": (
                    unexpected_mla_forward_layers
                ),
                "invalid_mla_forwards": invalid_mla_forwards,
                "mla_position_coverage_errors": mla_position_coverage_errors,
                "mla_row_geometry_errors": mla_row_geometry_errors,
                "measured_mla_fallback_request_ids": (
                    measured_mla_fallback_request_ids
                ),
                "measured_mla_evidence_error_request_ids": (
                    measured_mla_evidence_error_request_ids
                ),
                "mla_requests_without_reuse": mla_requests_without_reuse,
                "failed_mla_request_ids": failed_mla_request_ids,
                "runtime_evidence_failed_request_ids": (
                    runtime_evidence_failed_request_ids
                ),
                "formal_failed_mla_request_ids": formal_failed_mla_request_ids,
                "runtime_evidence_pass": runtime_evidence_pass,
                "full_local_sanity_requested": full_local_sanity_requested,
                "full_local_sanity_errors": full_local_sanity_errors,
                "full_local_sanity_failed_request_ids": (
                    full_local_sanity_failed_request_ids
                ),
                "full_local_sanity_pass": full_local_sanity_pass,
                "coverage": runtime_coverage,
                "formal_reuse_pass": formal_runtime_pass,
                "diagnostic_runtime_pass": (
                    runtime_pass if claim_diagnostic_only else None
                ),
                "qualification_runtime_pass": (
                    runtime_pass if IH_MLA_OFF_QUALIFICATION_ONLY else None
                ),
                "pass": runtime_pass,
            },
            "sequential_service_rate": {
                "dense_requests_per_second": dense_req_s,
                "reuse_requests_per_second": reuse_req_s,
                "dense_logical_prompt_tokens_per_second": dense_prompt_tps,
                "reuse_logical_prompt_tokens_per_second": reuse_prompt_tps,
            },
            "quality": {
                "num_pairs": quality_pair_count,
                "top1_rate": top1_rate,
                "min_top10_probability_cosine": min_cos,
                "dense_f1": dense_f1,
                "reuse_f1": reuse_f1,
                "f1_delta": reuse_f1 - dense_f1,
                "dense_em": dense_em,
                "reuse_em": reuse_em,
                "em_delta": reuse_em - dense_em,
                "f1_retention": f1_retention,
                "em_retention": em_retention,
                "mean_generation_token_agreement": sum(token_agreements)
                / max(1, len(token_agreements)),
                "min_generation_token_agreement": min_token_agreement,
                "pass": quality_pass,
                "thresholds": {
                    "min_top1_rate": IH_MIN_TOP1_RATE,
                    "min_cosine": IH_MIN_COSINE,
                    "min_f1_retention": IH_MIN_F1_RETENTION,
                    "min_em_retention": IH_MIN_EM_RETENTION,
                    "min_dense_f1": IH_MIN_DENSE_F1,
                    "min_reuse_f1": IH_MIN_REUSE_F1,
                    "min_dense_em": IH_MIN_DENSE_EM,
                    "min_reuse_em": IH_MIN_REUSE_EM,
                    "min_generation_token_agreement": IH_MIN_TOKEN_AGREEMENT,
                },
            },
            "performance_gate": {
                "min_speedup": IH_MIN_SPEEDUP,
                "min_qps_speedup": IH_MIN_QPS_SPEEDUP,
                "observed_qps_speedup_by_concurrency": (
                    performance_claim_qualification["runtime_evidence"].get(
                        "qps_speedup_by_concurrency", {}
                    )
                ),
                "minimum_observed_qps_speedup": (
                    performance_claim_qualification["runtime_evidence"].get(
                        "minimum_observed_qps_speedup"
                    )
                ),
                "min_online_row_saving": IH_MIN_ROW_SAVING,
                "requested_active_ratio": row_ratio_evidence[
                    "requested_active_ratio"
                ],
                "min_realized_active_ratio": row_ratio_evidence[
                    "min_realized_active_ratio"
                ],
                "actual_realized_active_ratio": row_ratio_evidence[
                    "actual_realized_active_ratio"
                ],
                "realized_active_ratio_required": row_ratio_evidence[
                    "realized_active_ratio_required"
                ],
                "realized_active_ratio_pass": row_ratio_evidence[
                    "realized_active_ratio_pass"
                ],
                "min_full_model_mla_head_row_saving": (
                    IH_MIN_HEAD_ROW_SAVING
                ),
                "observed_scoped_mla_head_row_saving": (
                    scoped_head_row_saving
                ),
                "observed_full_model_mla_head_row_saving": (
                    full_model_head_row_saving
                ),
                "performance_claim_eligible": (
                    performance_claim_qualification["eligible"]
                ),
                "diagnostic_only": claim_diagnostic_only,
                "qualification_only": IH_MLA_OFF_QUALIFICATION_ONLY,
                "status": result_claim_status,
                "ineligible_reason": result_claim_reason,
                "formal_runtime_reuse_required": True,
                "metric": (
                    "full_model_mla_head_row_saving"
                    if IH_MLA_OFFLOAD
                    else "online_row_saving"
                ),
                "pass": performance_pass,
            },
            "overall_pass": overall_pass,
            "queries": quality_rows,
        }
        if IH_MLA_OFFLOAD:
            # Do not publish parsed legacy defaults as null/false fields in a
            # pure result.  Their absence is part of the pure-only schema.
            _ih_finalize_pure_result_config(
                result["config"], offline_chunk_order=chunk_order0
            )
        output_path = RESULT_OUT or "/tmp/redknot_v4_flash_rag_result.json"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f" structured result: {output_path}")
        return result
    finally:
        # Ignore a repeated Ctrl-C/TERM while the first one is already draining
        # this benchmark's own group, then restore the caller's signal state.
        for signum in previous_signal_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if proc is not None:
                _ih_stop_launched_server(proc)
        finally:
            for signum, handler in previous_signal_handlers.items():
                signal.signal(signum, handler)


_IH_EXECUTION_GATE_KEYS = frozenset(
    {
        "execution_pass",
        "diagnostic_success",
        "diagnostic_only",
        "claim_independent",
        "gates",
        "reasons",
        "observed",
    }
)
_IH_EXECUTION_BOOLEAN_GATE_KEYS = frozenset(
    {
        "quality",
        "runtime",
        "runtime_evidence",
        "quality_pairs_complete",
        "dense_ttft_complete",
        "reuse_ttft_complete",
        "qps_complete",
    }
)
_IH_EXECUTION_OBSERVED_KEYS = frozenset(
    {
        "quality_pairs",
        "expected_quality_pairs",
        "dense_ttft_samples",
        "reuse_ttft_samples",
        "expected_ttft_samples_per_arm",
        "qps_required",
    }
)
_IH_CLAIM_STATUS_KEYS = frozenset(
    {
        "status",
        "reason",
        "claim_ineligible",
        "claim_ineligible_reason",
        "qualification_only",
        "diagnostic_ablation",
    }
)
_IH_FORMAL_CLAIM_KEYS = frozenset(
    {"eligible", "pass", "status", "ineligible_reason"}
)


def _ih_cli_execution_gate_complete(
    result: Mapping[str, object], *, diagnostic_only: bool
) -> bool:
    """Validate the exact execution record consumed by the process exit gate."""

    if type(diagnostic_only) is not bool or not isinstance(result, Mapping):
        return False
    execution_gate = result.get("execution_gate")
    if (
        not isinstance(execution_gate, Mapping)
        or frozenset(execution_gate) != _IH_EXECUTION_GATE_KEYS
    ):
        return False
    gates = execution_gate.get("gates")
    reasons = execution_gate.get("reasons")
    observed = execution_gate.get("observed")
    expected_diagnostic_success = diagnostic_only
    if (
        not isinstance(gates, Mapping)
        or frozenset(gates) != _IH_EXECUTION_BOOLEAN_GATE_KEYS
        or any(
            gates.get(name) is not True
            for name in _IH_EXECUTION_BOOLEAN_GATE_KEYS
        )
        or type(reasons) is not list
        or reasons
        or not isinstance(observed, Mapping)
        or frozenset(observed) != _IH_EXECUTION_OBSERVED_KEYS
        or execution_gate.get("execution_pass") is not True
        or execution_gate.get("diagnostic_success")
        is not expected_diagnostic_success
        or execution_gate.get("diagnostic_only") is not diagnostic_only
        or execution_gate.get("claim_independent") is not True
        or result.get("execution_pass") is not True
        or result.get("diagnostic_success")
        is not expected_diagnostic_success
    ):
        return False
    quality_pairs = observed.get("quality_pairs")
    expected_quality_pairs = observed.get("expected_quality_pairs")
    dense_ttft_samples = observed.get("dense_ttft_samples")
    reuse_ttft_samples = observed.get("reuse_ttft_samples")
    expected_ttft_samples = observed.get("expected_ttft_samples_per_arm")
    return bool(
        type(quality_pairs) is int
        and type(expected_quality_pairs) is int
        and expected_quality_pairs > 0
        and quality_pairs == expected_quality_pairs
        and type(dense_ttft_samples) is int
        and type(reuse_ttft_samples) is int
        and type(expected_ttft_samples) is int
        and expected_ttft_samples > 0
        and dense_ttft_samples == expected_ttft_samples
        and reuse_ttft_samples == expected_ttft_samples
        and type(observed.get("qps_required")) is bool
    )


def _ih_cli_row_ratio_evidence_complete(result: Mapping[str, object]) -> bool:
    """Fail closed on incomplete combined row-budget evidence."""

    config = result.get("config")
    if not isinstance(config, Mapping) or (
        config.get("combined_headsplit_row_sparse") is not True
    ):
        return True
    performance_gate = result.get("performance_gate")
    runtime = result.get("runtime")
    if not isinstance(performance_gate, Mapping) or not isinstance(
        runtime, Mapping
    ):
        return False
    requested = performance_gate.get("requested_active_ratio")
    minimum = performance_gate.get("min_realized_active_ratio")
    actual = performance_gate.get("actual_realized_active_ratio")
    if any(type(value) not in (int, float) for value in (requested, minimum, actual)):
        return False
    requested = float(requested)
    minimum = float(minimum)
    actual = float(actual)
    configured_capacity = config.get(
        "combined_row_sparse_checkpoint_max_islands"
    )
    required_capacity = config.get(
        "combined_row_sparse_required_checkpoint_islands"
    )
    return bool(
        math.isfinite(requested)
        and math.isfinite(minimum)
        and math.isfinite(actual)
        and 0.0 < minimum <= requested < 1.0
        and minimum - 1e-12 <= actual <= 1.0
        and type(configured_capacity) is int
        and type(required_capacity) is int
        and configured_capacity >= required_capacity > 0
        and performance_gate.get("realized_active_ratio_required") is True
        and performance_gate.get("realized_active_ratio_pass") is True
        and runtime.get("requested_active_ratio") == requested
        and runtime.get("min_realized_active_ratio") == minimum
        and runtime.get("actual_realized_active_ratio") == actual
        and runtime.get("realized_active_ratio_required") is True
        and runtime.get("realized_active_ratio_pass") is True
    )


def _ih_cli_result_success(
    result: Mapping[str, object], *, allow_diagnostic: bool = False
) -> bool:
    """Select a fail-closed process exit gate without confusing claims.

    Formal execution never treats a successful diagnostic arm as its own
    success.  The outer entry point must opt out of strict performance before
    a complete, claim-ineligible diagnostic record may return zero.
    """

    if type(allow_diagnostic) is not bool or not isinstance(result, Mapping):
        return False
    claim_status = result.get("claim_status")
    formal_claim = result.get("formal_claim")
    performance_gate = result.get("performance_gate")
    if (
        not isinstance(claim_status, Mapping)
        or frozenset(claim_status) != _IH_CLAIM_STATUS_KEYS
        or not isinstance(formal_claim, Mapping)
        or frozenset(formal_claim) != _IH_FORMAL_CLAIM_KEYS
        or not isinstance(performance_gate, Mapping)
        or performance_gate.get("formal_runtime_reuse_required") is not True
        or not _ih_cli_row_ratio_evidence_complete(result)
    ):
        return False

    status = claim_status.get("status")
    if status == "diagnostic_only":
        if not allow_diagnostic:
            return False
        reason = claim_status.get("reason")
        if reason == _IH_MLA_OFF_DIAGNOSTIC_CLAIM_REASON:
            diagnostic_ablation_valid = (
                claim_status.get("diagnostic_ablation")
                in {"zoff_only", "shared_only"}
            )
        elif reason == _IH_PERFORMANCE_DIAGNOSTIC_CLAIM_REASON:
            diagnostic_ablation_valid = (
                claim_status.get("diagnostic_ablation") is None
            )
        else:
            return False
        return bool(
            _ih_cli_execution_gate_complete(result, diagnostic_only=True)
            and claim_status.get("reason") == reason
            and claim_status.get("claim_ineligible") is True
            and claim_status.get("claim_ineligible_reason") == reason
            and claim_status.get("qualification_only") is False
            and diagnostic_ablation_valid
            and result.get("claim_ineligible") is True
            and result.get("claim_ineligible_reason") == reason
            and formal_claim.get("eligible") is False
            and formal_claim.get("pass") is False
            and formal_claim.get("status") == "diagnostic_only"
            and formal_claim.get("ineligible_reason") == reason
            and performance_gate.get("performance_claim_eligible") is False
            and performance_gate.get("diagnostic_only") is True
            and performance_gate.get("qualification_only") is False
            and performance_gate.get("status") == "diagnostic_only"
            and performance_gate.get("ineligible_reason") == reason
            and performance_gate.get("pass") is False
            and result.get("overall_pass") is False
        )
    if status == "formal_candidate":
        # A successful formal candidate must keep the execution, claim, and
        # performance records mutually consistent.  Qualification-only and any
        # partial/malformed record remain non-zero.
        return bool(
            not allow_diagnostic
            and _ih_cli_execution_gate_complete(result, diagnostic_only=False)
            and claim_status.get("reason") == ""
            and claim_status.get("claim_ineligible") is False
            and claim_status.get("claim_ineligible_reason") == ""
            and claim_status.get("qualification_only") is False
            and claim_status.get("diagnostic_ablation") is None
            and result.get("claim_ineligible") is False
            and result.get("claim_ineligible_reason") == ""
            and formal_claim.get("eligible") is True
            and formal_claim.get("pass") is True
            and formal_claim.get("status") == "formal_candidate"
            and formal_claim.get("ineligible_reason") == ""
            and performance_gate.get("performance_claim_eligible") is True
            and performance_gate.get("diagnostic_only") is False
            and performance_gate.get("qualification_only") is False
            and performance_gate.get("status") == "formal_candidate"
            and performance_gate.get("ineligible_reason") == ""
            and performance_gate.get("pass") is True
            and result.get("overall_pass") is True
        )
    return False


def _run_sglang_engine_benchmark():
    if ENGINE_MODE not in {"baseline", "redknot", "both"}:
        raise ValueError("REDKNOT_ENGINE_MODE must be one of: baseline, redknot, both")
    tok = _load_tokenizer()
    tasks = []
    for ds_name in DATASETS:
        for length_label in LENGTHS:
            target = _LEN_MAP[length_label]
            samples = _load_longbench_padded(ds_name, tok, N_SAMPLES, target)
            if not samples:
                print(f"\n[skip] {ds_name}@{length_label}: no usable samples")
                continue
            for sample in samples:
                prompt = _encode_rag_prompt(sample["docs"], sample["question"])
                tasks.append((f"{ds_name}@{length_label}", sample, prompt))

    W = 108
    print("=" * W)
    print(" BENCHMARK: DeepSeek-V4-Pro-0813 RAG on 8x B300/SM103")
    print(f" Model: {MODEL_PATH}")
    print(
        f" mode={ENGINE_MODE} sparse_ffn={SPARSE_FFN} tasks={len(tasks)} "
        f"tp={TP_SIZE} samples/dataset={N_SAMPLES}"
    )
    print(
        " RedKnot MLA policy: "
        f"dense_prefix={MLA_DENSE_PREFIX} dense_suffix={MLA_DENSE_SUFFIX} "
        f"local_window={MLA_LOCAL_WINDOW} "
        f"global_head_stride={MLA_GLOBAL_HEAD_STRIDE} "
        f"global_layer_stride={MLA_GLOBAL_LAYER_STRIDE} "
        f"mla_pass_mode={MLA_PASS_MODE}"
    )
    print("=" * W)
    if not tasks:
        return

    prompts = [p for _, _, p in tasks]
    prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    base_out = [None] * len(prompts)
    rk_out = [None] * len(prompts)
    base_time = None
    rk_time = None
    if ENGINE_MODE in {"baseline", "both"}:
        print("Running baseline engine: attention_backend=dsv4")
        t0 = time.perf_counter()
        base_out = _engine_generate_all(prompt_ids, "dsv4", sparse_ffn=False)
        base_time = time.perf_counter() - t0

    if ENGINE_MODE in {"redknot", "both"}:
        print(
            "Running RedKnot engine: attention_backend=redknot_mla, "
            f"sparse_ffn={SPARSE_FFN}"
        )
        t0 = time.perf_counter()
        rk_out = _engine_generate_all(prompt_ids, "redknot_mla", sparse_ffn=SPARSE_FFN)
        rk_time = time.perf_counter() - t0

    rows = []
    for i, ((task_name, sample, prompt), b, r) in enumerate(
        zip(tasks, base_out, rk_out)
    ):
        base_text = b.get("text", "") if isinstance(b, dict) else ""
        rk_text = r.get("text", "") if isinstance(r, dict) else ""
        base_ans = _short_ans(base_text)
        rk_ans = _short_ans(rk_text)
        row = {
            "task": task_name,
            "question": sample["question"],
            "golds": sample["golds"],
            "base_answer": base_ans,
            "rk_answer": rk_ans,
            "base_f1": f1_max(base_ans, sample["golds"]),
            "base_em": em_max(base_ans, sample["golds"]),
            "rk_f1": f1_max(rk_ans, sample["golds"]),
            "rk_em": em_max(rk_ans, sample["golds"]),
            "base_perf": b.get("bench_metrics", {}) if isinstance(b, dict) else {},
            "rk_perf": r.get("bench_metrics", {}) if isinstance(r, dict) else {},
        }
        rows.append(row)
        print(
            f"\n [sample {i} {task_name}] ctx≈{len(tok(prompt, add_special_tokens=False)['input_ids']):,} tok"
        )
        print(f"   Q       : {_trunc(sample['question'], 88)}")
        print(f"   gold    : {sample['golds'][0] if sample['golds'] else ''}")
        if b is not None:
            perf = row["base_perf"]
            print(
                f"   standard: {_trunc(base_text, 96)!r} -> "
                f"{_trunc(base_ans, 40)!r} F1={row['base_f1']:.2f} "
                f"TTFT={perf.get('ttft', 0):.3f}s "
                f"model_prefill={perf.get('model_prefill_ttft', 0):.3f}s "
                f"decode={perf.get('decode_throughput', 0):.2f} tok/s"
            )
        if r is not None:
            perf = row["rk_perf"]
            print(
                f"   redknot : {_trunc(rk_text, 96)!r} -> "
                f"{_trunc(rk_ans, 40)!r} F1={row['rk_f1']:.2f} "
                f"TTFT={perf.get('ttft', 0):.3f}s "
                f"model_prefill={perf.get('model_prefill_ttft', 0):.3f}s "
                f"decode={perf.get('decode_throughput', 0):.2f} tok/s"
            )

    print(f"\n{'=' * W}")
    print(" SUMMARY")
    print("=" * W)
    for task_name in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task_name]

        def avg(key):
            return sum(row[key] for row in sub) / len(sub)

        metrics = []
        if ENGINE_MODE in {"baseline", "both"}:
            metrics.append(
                f"std F1={avg('base_f1'):.3f} EM={avg('base_em'):.3f} "
                f"TTFT={sum(r['base_perf']['ttft'] for r in sub) / len(sub):.3f}s "
                f"model_prefill={sum(r['base_perf']['model_prefill_ttft'] for r in sub) / len(sub):.3f}s "
                f"decode={sum(r['base_perf']['decode_throughput'] for r in sub) / len(sub):.2f} tok/s"
            )
        if ENGINE_MODE in {"redknot", "both"}:
            metrics.append(
                f"rk F1={avg('rk_f1'):.3f} EM={avg('rk_em'):.3f} "
                f"TTFT={sum(r['rk_perf']['ttft'] for r in sub) / len(sub):.3f}s "
                f"model_prefill={sum(r['rk_perf']['model_prefill_ttft'] for r in sub) / len(sub):.3f}s "
                f"decode={sum(r['rk_perf']['decode_throughput'] for r in sub) / len(sub):.2f} tok/s"
            )
        if ENGINE_MODE == "both":
            metrics.append(f"dF1={avg('rk_f1') - avg('base_f1'):+.3f}")
        print(f" {task_name:28s} {' | '.join(metrics)}")
    elapsed = []
    if base_time is not None:
        elapsed.append(f"standard={base_time:.1f}s")
    if rk_time is not None:
        elapsed.append(f"redknot={rk_time:.1f}s")
    print(f" elapsed: {' '.join(elapsed)}")
    if RESULT_OUT:
        os.makedirs(os.path.dirname(RESULT_OUT) or ".", exist_ok=True)
        with open(RESULT_OUT, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f" raw results: {RESULT_OUT}")
    print("=" * W)


def main():
    if DRIFT_PROFILE and not DRY_RUN:
        _run_projected_head_drift_profile()
        return

    if PROFILE and not DRY_RUN:
        _run_profile()
        return

    # One-click pure MLA offline/online merge qualification benchmark.
    if ENGINE_MODE == "indexer_hot" and not DRY_RUN:
        benchmark_result = _run_indexer_hot_benchmark()
        if not _ih_cli_result_success(
            benchmark_result,
            allow_diagnostic=not IH_STRICT_PERFORMANCE_CLAIMS,
        ):
            strategy_label = (
                "pure MLA offline/online merge"
                if IH_MLA_OFFLOAD
                else "indexer-hot"
            )
            claim_record = (
                benchmark_result.get("claim_status")
                if isinstance(benchmark_result, Mapping)
                else None
            )
            claim_status = (
                claim_record.get("status")
                if isinstance(claim_record, Mapping)
                else None
            )
            gate_label = (
                "diagnostic quality/runtime/evidence/execution"
                if claim_status == "diagnostic_only"
                else "quality/runtime/performance"
            )
            raise SystemExit(
                f"RedKnot {strategy_label} "
                f"{gate_label} gate failed"
            )
        return

    if RUNTIME == "sglang" and not DRY_RUN:
        _run_sglang_engine_benchmark()
        return

    W = 108
    print("=" * W)
    print(" BENCHMARK: DeepSeek V4 RAG smoke + optional RedKnot offline-KV reuse")
    print(f" Model: {MODEL_PATH}")
    print(
        " RAG: LongBench padded contexts | "
        f"datasets={','.join(DATASETS)} | lengths={','.join(LENGTHS)} | samples={N_SAMPLES}"
    )
    print(f" RedKnot reuse enabled: {ENABLE_REUSE}")
    print("=" * W)

    if DRY_RUN:
        tok = _load_tokenizer()
        for ds_name in DATASETS:
            for length_label in LENGTHS:
                samples = _load_longbench_padded(
                    ds_name, tok, N_SAMPLES, _LEN_MAP[length_label]
                )
                if not samples:
                    print(f" [dry-run] {ds_name}@{length_label}: no usable samples")
                    continue
                first = samples[0]
                doc_lens = [
                    len(tok(doc, add_special_tokens=False)["input_ids"])
                    for doc in first["docs"]
                ]
                print(
                    f" [dry-run] {ds_name}@{length_label}: samples={len(samples)} "
                    f"docs={len(first['docs'])} doc_tokens={doc_lens[:4]} "
                    f"question={_trunc(first['question'], 64)!r}"
                )
        return

    model, tok = _load_model_and_tokenizer()
    from sglang.srt.layers.attention.redknot import (
        deepseek_v4_mla_cache_descriptor,
        is_deepseek_v4_mla_config,
    )

    dims = _model_dims(model.config)
    print(
        f" Config: layers={dims['L']} hidden={dims['hidden']} Hq={dims['Hq']} "
        f"Hkv={dims['Hkv']} D={dims['D']} q_lora={dims['q_lora']} o_lora={dims['o_lora']}"
    )
    if is_deepseek_v4_mla_config(model.config):
        desc = deepseek_v4_mla_cache_descriptor(model.config)
        mla_cfg = _make_default_head_config(model.config)
        print(f" MLA cache: {desc}")
        print(f" MLA logical-head policy: {mla_cfg.summary()}")

    reuse_ok, reuse_reason = _looks_redknot_hf_compatible(model)
    if ENABLE_REUSE and not reuse_ok:
        print(
            f" [skip RedKnot reuse] Current HF model is not supported: {reuse_reason}"
        )
    elif ENABLE_REUSE:
        print(
            " [RedKnot reuse] HF model exposes generic attention attrs; reuse path will run."
        )

    overall = {}
    for ds_name in DATASETS:
        for length_label in LENGTHS:
            target = _LEN_MAP[length_label]
            samples = _load_longbench_padded(ds_name, tok, N_SAMPLES, target)
            task_name = f"{ds_name}@{length_label}"
            if not samples:
                print(f"\n[skip] {task_name}: no usable samples")
                continue
            print(
                f"\n{'=' * W}\n TASK: {task_name} ({len(samples)} sample(s), chunk={CHUNK_TOKENS})\n{'=' * W}"
            )

            rows, ctx_lens, selected_fracs = [], [], []
            for si, sample in enumerate(samples):
                query = _query_text(sample["question"])
                full_text = "\n\n".join(sample["docs"])
                base_text, base_ttft, base_dec, n_ctx = standard_prefill(
                    model, tok, full_text, query
                )
                base_ans = _short_ans(base_text)
                ctx_lens.append(n_ctx)

                rk_text = rk_ans = ""
                rk_ttft = rk_dec = None
                if ENABLE_REUSE and reuse_ok:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    rk_text, rk_ttft, rk_dec, stats = redknot_prefill(
                        model, tok, sample["docs"], query
                    )
                    rk_ans = _short_ans(rk_text)
                    sparse = [x for x in stats if x.get("mode") == "sparse"]
                    if sparse:
                        selected_fracs.append(
                            sum(x["selected_frac"] for x in sparse) / len(sparse)
                        )

                row = {
                    "base_f1": f1_max(base_ans, sample["golds"]),
                    "base_em": em_max(base_ans, sample["golds"]),
                    "base_ttft": base_ttft,
                    "base_dec": base_dec,
                }
                if rk_ttft is not None:
                    row.update(
                        {
                            "rk_f1": f1_max(rk_ans, sample["golds"]),
                            "rk_em": em_max(rk_ans, sample["golds"]),
                            "rk_ttft": rk_ttft,
                            "rk_dec": rk_dec,
                        }
                    )
                rows.append(row)

                print(f"\n [sample {si}] ctx={n_ctx:,} tok docs={len(sample['docs'])}")
                print(f"   Q   : {_trunc(sample['question'], 88)}")
                print(f"   gold: {sample['golds'][0] if sample['golds'] else ''}")
                print(
                    f"   base: {_trunc(base_text, 64)!r} -> {_trunc(base_ans, 32)!r} "
                    f"F1={row['base_f1']:.2f} TTFT={base_ttft:.2f}s dec={base_dec:.1f} tok/s"
                )
                if rk_ttft is not None:
                    print(
                        f"   rk  : {_trunc(rk_text, 64)!r} -> {_trunc(rk_ans, 32)!r} "
                        f"F1={row['rk_f1']:.2f} TTFT={rk_ttft:.2f}s "
                        f"speedup={base_ttft / rk_ttft:.2f}x"
                    )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not rows:
                continue

            def avg(key):
                return sum(row[key] for row in rows if key in row) / max(
                    1, sum(1 for row in rows if key in row)
                )

            avg_ctx = int(sum(ctx_lens) / len(ctx_lens))
            selected = (
                sum(selected_fracs) / len(selected_fracs) if selected_fracs else 1.0
            )
            dense_until = int(os.environ.get("REDKNOT_FFN_DENSE_UNTIL", "4"))
            local_window = int(os.environ.get("REDKNOT_LOCAL_WINDOW", "4096"))
            frac_global = 1.0 / max(
                1, int(os.environ.get("REDKNOT_GLOBAL_LAYER_STRIDE", "8"))
            )
            flops = compute_flops(
                dims, avg_ctx, frac_global, selected, dense_until, local_window
            )

            print(f"\n {'-' * (W - 2)}")
            print(
                f" {task_name} AGGREGATE ({len(rows)} sample(s), avg ctx={avg_ctx:,} tok)"
            )
            print(f" {'-' * (W - 2)}")
            print(
                f"   baseline  F1={avg('base_f1'):.3f} EM={avg('base_em'):.3f} TTFT={avg('base_ttft'):.2f}s"
            )
            if any("rk_ttft" in r for r in rows):
                print(
                    f"   RedKnot   F1={avg('rk_f1'):.3f} EM={avg('rk_em'):.3f} "
                    f"TTFT={avg('rk_ttft'):.2f}s speedup={avg('base_ttft') / avg('rk_ttft'):.2f}x"
                )
            print(
                "   analytic FLOPs proxy for candidate RedKnot policy (not wall time):"
            )
            for name in ["attn", "ffn", "proj", "total"]:
                dense, rk = flops[name]
                saving = (1 - rk / dense) * 100 if dense else 0.0
                print(
                    f"             {name:6s} dense={dense / 1e15:7.3f}P "
                    f"rk={rk / 1e15:7.3f}P saving={saving:5.1f}%"
                )
            overall[task_name] = {
                "base_f1": avg("base_f1"),
                "base_ttft": avg("base_ttft"),
                "rk_f1": avg("rk_f1") if any("rk_f1" in r for r in rows) else None,
                "rk_ttft": (
                    avg("rk_ttft") if any("rk_ttft" in r for r in rows) else None
                ),
            }

    print(f"\n{'=' * W}")
    print(" SUMMARY")
    print("=" * W)
    print(
        f" {'task':28s} {'base F1':>8s} {'base TTFT':>10s} {'rk F1':>8s} {'rk TTFT':>10s} {'speedup':>8s}"
    )
    for name, row in overall.items():
        if row["rk_ttft"] is None:
            print(
                f" {name:28s} {row['base_f1']:>8.3f} {row['base_ttft']:>9.2f}s {'-':>8s} {'-':>10s} {'-':>8s}"
            )
        else:
            print(
                f" {name:28s} {row['base_f1']:>8.3f} {row['base_ttft']:>9.2f}s "
                f"{row['rk_f1']:>8.3f} {row['rk_ttft']:>9.2f}s "
                f"{row['base_ttft'] / row['rk_ttft']:>7.2f}x"
            )
    print("=" * W)


if __name__ == "__main__":
    main()
