#!/usr/bin/env python3
"""Isolated DeepSeek-V4-Pro-0813 RedKnot reproduction entry point.

The RedKnot mode intentionally combines only compatible mechanisms:

1. context-bound pure MLA offline-local / online-global head merge;
2. native routed-expert K=6 for formal runs; progressive/adaptive Top-K is an
   explicit Pro calibration arm only;
3. offline Indexer K/state with query-dependent Q, scoring and Top-1024 online.

This file is intentionally separate from the Flash-0731 benchmark.  Every GPU
run is pinned to the official Pro-0813 checkpoint, TP8, the 61-layer/128-head
geometry, and the B300/Blackwell SM103 launcher.  ``--contract-only`` performs
the same fail-closed geometry and z_off capacity checks without importing
torch or touching a GPU.

An explicit ``--row-sparse-online`` qualification arm reuses the repository's
existing checkpoint/Indexer replay to physically propagate only selected
document rows.  It is deliberately labelled separately from pure MLA-offload;
the arm is used to establish the systems ceiling before the two mechanisms are
combined, never to report a legacy fallback as pure head-split execution.

Token-drop sparse FFN stays disabled so accuracy loss and compute savings remain
attributable to the two approximation levers above.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import runpy
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
DRIVER = REPO / "test/srt/redknot/utils/benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py"
MODEL = Path("/workspace/Models/DeepSeek-V4-Pro-0813")
LAUNCHER = REPO / "server/start_server_redknot_pro0813.sh"
DATA_DIR = REPO / "test/srt/redknot/datasets/LongBench/data"
DEFAULT_DATA_MANIFEST = (
    REPO / "test/srt/redknot/utils/musique_pure_prompt_selection_v1.json"
)
DATA_MANIFEST_128K = (
    REPO / "test/srt/redknot/utils/musique_pure_prompt_selection_128k_v1.json"
)
DATA_MANIFEST_256K_32K = (
    REPO
    / "test/srt/redknot/utils/musique_pure_prompt_selection_256k_32k_v1.json"
)

SELECTION_SHA256 = (
    "586fd683bfe043e1a6aaa1d07c7236ea9d956d99be739be743c4a2ec1728bcd8"
)
PROMPT_TEXT_SHA256 = (
    "sha256:fa33caccb16d22f9df544239de3229c74bf6ce6847148ddeccbdbde371db11c8"
)
FULL_INPUT_IDS_SHA256 = (
    "sha256:9329590a5c2bb87e7689d5d8b81edbadf50394a89f268df97268debd82bea891"
)
FULL_INPUT_TOKENS = 65585
OFFLINE_DOCUMENT_TOKENS = 65536

PRO0813_VARIANT = "deepseek_v4_pro_0813"
PRO0813_CONFIG_SHA256 = (
    "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0"
)
PRO0813_GEOMETRY_DIGEST = (
    "sha256:adca138e64f2da316e94dd62394a51bbf5a89ab0651475579ce1977c59497819"
)
PRO0813_NUM_LAYERS = 61
PRO0813_NUM_HEADS = 128
PRO0813_INDEX_TOPK = 1024
PRO0813_TP_SIZE = 8
PRO0813_DENSE_LAYER_IDS = (0, 1, 2, 58, 59, 60)
PRO0813_REUSABLE_LAYER_IDS = tuple(range(3, 58))
PRO0813_O_GROUPS_PER_TP_RANK = 2
PRO0813_O_LORA_RANK = 1024
PRO0813_ZOFF_BYTES_PER_ELEMENT = 2
FORMAL_QPS_WARMUP_WAVES = 3
FORMAL_QPS_MEASUREMENT_WAVES = 10
PRO0813_PURE_PROFILE = (
    "pure_headsplit_pro0813_independent_rope_relocation_fullscope_"
    "boundary128_3_55_3_v1"
)
PRO0813_COMBINED_DIAGNOSTIC_PROFILE = (
    "combined_headsplit_pro0813_independent_rope_zoff_checkpoint_"
    "rowsparse_3_55_3_v1"
)
PRO0813_COMBINED_FULL_PROFILE = (
    "combined_headsplit_pro0813_independent_rope_full_checkpoint_"
    "rowsparse_3_55_3_v1"
)


def _load_pro0813_scale_policy_module():
    path = (
        REPO
        / "python/sglang/srt/layers/attention/redknot/pro0813/scale_policy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_redknot_pro0813_benchmark_scale_policy", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Pro-0813 scale policy: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PRO0813_SCALE = _load_pro0813_scale_policy_module()
PRO0813_SCALE_POLICY_VERSION = _PRO0813_SCALE.PRO0813_SCALE_POLICY_VERSION
PRO0813_SCALE_POLICY_DIGEST = _PRO0813_SCALE.PRO0813_SCALE_POLICY_DIGEST
PRO0813_PROGRESSIVE_TOPK_SCHEDULE = (
    _PRO0813_SCALE.PRO0813_PROGRESSIVE_TOPK_SCHEDULE
)
PRO0813_STANDARD_ACTIVE_RATIO = _PRO0813_SCALE.PRO0813_STANDARD_ACTIVE_RATIO
PRO0813_STRONG_ACTIVE_RATIO = _PRO0813_SCALE.PRO0813_STRONG_ACTIVE_RATIO
PRO0813_DIFFUSE_ACTIVE_RATIO = _PRO0813_SCALE.PRO0813_DIFFUSE_ACTIVE_RATIO
PRO0813_TOKEN_SPARSE_DEEP_START = (
    _PRO0813_SCALE.PRO0813_TOKEN_SPARSE_DEEP_START
)
PRO0813_MIN_FREE_BEFORE_LAUNCH_MIB = (
    _PRO0813_SCALE.PRO0813_MIN_FREE_BEFORE_LAUNCH_MIB
)


def _pin_pro0813_python_sources() -> tuple[Path, Path]:
    """Discard ambient source overlays before loading any SGLang code.

    Replacing ``PYTHONPATH`` alone is insufficient in an already-running
    interpreter because the original entries are already in ``sys.path``.
    The shared SM103 virtualenv remains the binary dependency provider; the
    only permitted source overlays are this repository and certified FlashMLA.
    """

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


def _add_performance_qualification_args(
    parser: argparse.ArgumentParser,
) -> None:
    """Install fail-closed formal defaults with one explicit diagnostic opt-out."""

    qps_measurement = parser.add_mutually_exclusive_group()
    qps_measurement.add_argument(
        "--measure-qps",
        dest="measure_qps",
        action="store_true",
        default=True,
        help="Measure paired QPS (enabled by default for formal runs).",
    )
    qps_measurement.add_argument(
        "--no-measure-qps",
        dest="measure_qps",
        action="store_false",
        help="Disable QPS only for an explicit diagnostic-performance run.",
    )
    parser.add_argument(
        "--qps-concurrencies",
        default="1",
        help=(
            "Comma-separated closed-loop QPS concurrency points; formal "
            "first-document-prefix runs are currently certified only at 1."
        ),
    )
    parser.add_argument(
        "--qps-warmup-waves",
        type=int,
        default=FORMAL_QPS_WARMUP_WAVES,
        help="Unmeasured QPS waves; formal default and minimum is 3.",
    )
    parser.add_argument(
        "--qps-waves",
        type=int,
        default=FORMAL_QPS_MEASUREMENT_WAVES,
        help="Measured QPS waves; formal default and minimum is 10.",
    )
    performance_mode = parser.add_mutually_exclusive_group()
    performance_mode.add_argument(
        "--strict-performance",
        dest="strict_performance",
        action="store_true",
        default=True,
        help=(
            "Fail before model launch unless TTFT/QPS sampling is eligible "
            "for a formal performance claim (default)."
        ),
    )
    performance_mode.add_argument(
        "--diagnostic-performance",
        dest="strict_performance",
        action="store_false",
        help=(
            "Explicitly opt out of formal performance qualification; the "
            "result remains claim-ineligible diagnostic evidence."
        ),
    )


def _validate_performance_qualification_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Reject an under-sampled formal request before any model launch."""

    if args.strict_performance and (
        args.ttft_warmup < 3
        or args.ttft_iters < 10
        or not args.measure_qps
        or args.qps_warmup_waves < FORMAL_QPS_WARMUP_WAVES
        or args.qps_waves < FORMAL_QPS_MEASUREMENT_WAVES
        or args.quality_repeats < 3
    ):
        parser.error(
            "formal performance requires TTFT warmup/iters >=3/10, "
            "QPS measurement enabled, and QPS warmup/measurement waves "
            ">=3/10, and quality repeats >=3; use --diagnostic-performance "
            "for claim-ineligible short diagnostics"
        )


def _validate_qualification_profile_target_policy(
    args: argparse.Namespace,
) -> None:
    """Keep formal short targets on their immutable built-in cohorts."""

    if (
        args.strict_performance
        and args.target_tokens in (65536, 131072, 262144)
        and args.qualification_profile
    ):
        raise ValueError(
            "formal 64K/128K/256K reproduction uses only its built-in frozen "
            "input; --qualification-profile is reserved for explicit "
            "claim-ineligible diagnostics at those targets"
        )


def _validate_formal_execution_profile_args(args: argparse.Namespace) -> None:
    """Permit formal RedKnot claims only through the full combined arm."""

    if (
        args.mode == "redknot"
        and args.strict_performance
        and not args.combined_headsplit_row_sparse
    ):
        raise ValueError(
            "formal strict RedKnot reproduction requires the full combined "
            "--combined-headsplit-row-sparse; pure head-split, standalone "
            "row-sparse, and zoff-only diagnostic arms are claim-ineligible"
        )


def _pro0813_zoff_bytes_per_rank(total_tokens: int) -> int:
    """BF16 z_off = layers * tokens * TP-local groups * o_lora * 2 bytes."""

    return _PRO0813_SCALE.pro0813_zoff_bytes_per_rank(total_tokens)


def _load_pro0813_profile_module():
    path = (
        REPO
        / "python/sglang/srt/layers/attention/redknot/pro0813/profile.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_redknot_pro0813_benchmark_profile", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Pro-0813 profile contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_pro0813_contract(config_path: Path) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"official Pro-0813 config is absent: {config_path}")
    config_bytes = config_path.read_bytes()
    observed_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if observed_sha256 != PRO0813_CONFIG_SHA256:
        raise ValueError(
            "Pro-0813 config bytes differ from the official 0813 contract: "
            f"observed={observed_sha256} expected={PRO0813_CONFIG_SHA256}"
        )
    module = _load_pro0813_profile_module()
    geometry = module.inspect_pro0813_config(
        json.loads(config_bytes), tp_size=PRO0813_TP_SIZE
    )
    if geometry.variant != PRO0813_VARIANT:
        raise ValueError("Pro-0813 variant token differs from benchmark contract")
    if geometry.geometry_digest != PRO0813_GEOMETRY_DIGEST:
        raise ValueError("Pro-0813 geometry digest differs from benchmark contract")
    if geometry.num_target_layers != PRO0813_NUM_LAYERS:
        raise ValueError("Pro-0813 benchmark requires exactly 61 target layers")
    if geometry.num_attention_heads != PRO0813_NUM_HEADS:
        raise ValueError("Pro-0813 benchmark requires exactly 128 logical heads")
    if tuple(geometry.dense_layer_ids) != PRO0813_DENSE_LAYER_IDS:
        raise ValueError("Pro-0813 dense-layer fence differs from 3+55+3")
    if tuple(geometry.reusable_layer_ids) != PRO0813_REUSABLE_LAYER_IDS:
        raise ValueError("Pro-0813 reusable layers must be exactly 3..57")
    if geometry.index_topk != PRO0813_INDEX_TOPK:
        raise ValueError("Pro-0813 Indexer Top-K must be exactly 1024")
    scale_policy = _PRO0813_SCALE.scale_policy_audit()
    if _PRO0813_SCALE.scale_policy_digest() != PRO0813_SCALE_POLICY_DIGEST:
        raise ValueError("Pro-0813 scale-policy digest is not reproducible")
    return {
        "accelerator": "8x NVIDIA B300 / Blackwell SM103",
        "variant": geometry.variant,
        "geometry_digest": geometry.geometry_digest,
        "official_config_sha256": observed_sha256,
        "tp_size": geometry.tp_size,
        "num_layers": geometry.num_target_layers,
        "num_attention_heads": geometry.num_attention_heads,
        "dense_layer_ids": list(geometry.dense_layer_ids),
        "reusable_layer_ids": [
            geometry.first_reusable_layer,
            geometry.last_reusable_layer,
            len(geometry.reusable_layer_ids),
        ],
        "index_topk": geometry.index_topk,
        "profiles": [
            PRO0813_PURE_PROFILE,
            PRO0813_COMBINED_DIAGNOSTIC_PROFILE,
            PRO0813_COMBINED_FULL_PROFILE,
        ],
        "zoff_64k_bytes_per_rank": _pro0813_zoff_bytes_per_rank(65536),
        "zoff_formula": "55 * tokens * 2 * 1024 * 2",
        "scale_policy_version": PRO0813_SCALE_POLICY_VERSION,
        "scale_policy_digest": PRO0813_SCALE_POLICY_DIGEST,
        "scale_policy": scale_policy,
    }

REDKNOT_TARGET_PROFILES = {
    65536: {
        "num_chunks": 8,
        "chunk_tokens": 8192,
        "query_row_id": 68,
        "data_manifest": DEFAULT_DATA_MANIFEST,
        "selection_sha256": SELECTION_SHA256,
        "prompt_text_sha256": PROMPT_TEXT_SHA256,
        "full_input_ids_sha256": FULL_INPUT_IDS_SHA256,
        "full_input_tokens": FULL_INPUT_TOKENS,
        # Exact Pro BF16 z_off projection is 14,763,950,080 bytes/rank.
        "zoff_projected_bytes_per_rank": _pro0813_zoff_bytes_per_rank(65536),
        "mla_off_max_bytes": 16 * 1024**3,
        "mla_off_device_max_bytes": 16 * 1024**3,
        "mem_fraction_static": _PRO0813_SCALE.PRO0813_MEM_FRACTION_STATIC[65536],
        "row_sparse_checkpoint_max_islands": (
            _PRO0813_SCALE.PRO0813_CHECKPOINT_MAX_ISLANDS[65536]
        ),
    },
    131072: {
        "num_chunks": 16,
        "chunk_tokens": 8192,
        "query_row_id": 68,
        "data_manifest": DATA_MANIFEST_128K,
        "selection_sha256": (
            "caf99890880e0de190f845d0a38e600d760d2153cd1961888bd7776a2044f040"
        ),
        "prompt_text_sha256": (
            "sha256:9959bc0f32f7eb29a4cf61e7d7a20ca8fda937166057510f49ae74056576f4b1"
        ),
        "full_input_ids_sha256": (
            "sha256:3b1ee37110db315a9ba84a3ae55adfce61b2aaa61520fc4c68511313cf96dd87"
        ),
        "full_input_tokens": 131128,
        "zoff_projected_bytes_per_rank": _pro0813_zoff_bytes_per_rank(131072),
        "mla_off_max_bytes": 32 * 1024**3,
        "mla_off_device_max_bytes": 32 * 1024**3,
        "mem_fraction_static": _PRO0813_SCALE.PRO0813_MEM_FRACTION_STATIC[131072],
        "row_sparse_checkpoint_max_islands": (
            _PRO0813_SCALE.PRO0813_CHECKPOINT_MAX_ISLANDS[131072]
        ),
    },
    262144: {
        "num_chunks": 8,
        "chunk_tokens": 32768,
        "query_row_id": 0,
        "data_manifest": DATA_MANIFEST_256K_32K,
        "selection_sha256": (
            "a2524b87a6ff0a91e7f5aef104d3b8eb14b9aa55e2b8c6b5db34ef0dbe1477cc"
        ),
        # Byte-frozen official one-pass prompt identity.  These values were
        # produced before any GPU/model request from the immutable data
        # selection above.
        "prompt_text_sha256": (
            "sha256:c2bb701688ef4cffc4911cf325c1928c2517f2270ea24e84cd9f492877ae6b4e"
        ),
        "full_input_ids_sha256": (
            "sha256:6adc0143211e3c7d4593ceee923dbeaa923e6f53c7ab02c4f42071ff2dffd310"
        ),
        "full_input_tokens": 262197,
        "zoff_projected_bytes_per_rank": _pro0813_zoff_bytes_per_rank(262144),
        "mla_off_max_bytes": 64 * 1024**3,
        "mla_off_device_max_bytes": 64 * 1024**3,
        "mem_fraction_static": _PRO0813_SCALE.PRO0813_MEM_FRACTION_STATIC[262144],
        "row_sparse_checkpoint_max_islands": (
            _PRO0813_SCALE.PRO0813_CHECKPOINT_MAX_ISLANDS[262144]
        ),
    },
    450560: {
        "num_chunks": 8,
        "chunk_tokens": 56320,
        # 440K (binary) keeps the eight-document RAG geometry while avoiding
        # the eighth 64K snapshot's measured 4 GiB publication OOM.
        "query_row_id": 0,
        "data_manifest": Path(""),
        "selection_sha256": "",
        "prompt_text_sha256": "",
        "full_input_ids_sha256": "",
        "full_input_tokens": 0,
        "zoff_projected_bytes_per_rank": _pro0813_zoff_bytes_per_rank(450560),
        "mla_off_max_bytes": 112 * 1024**3,
        "mla_off_device_max_bytes": 0,
        "mem_fraction_static": _PRO0813_SCALE.PRO0813_MEM_FRACTION_STATIC[450560],
        "row_sparse_checkpoint_max_islands": (
            _PRO0813_SCALE.PRO0813_CHECKPOINT_MAX_ISLANDS[450560]
        ),
        "requires_qualification_profile": True,
    },
    524288: {
        "num_chunks": 8,
        "chunk_tokens": 65536,
        # 512K is always driven by an immutable multi-dataset qualification
        # profile.  There is deliberately no mutable/default corpus identity.
        "query_row_id": 0,
        "data_manifest": Path(""),
        "selection_sha256": "",
        "prompt_text_sha256": "",
        "full_input_ids_sha256": "",
        "full_input_tokens": 0,
        # Exact Pro projection: 55 * 524288 * 2 groups * 1024 * 2 BF16 bytes.
        "zoff_projected_bytes_per_rank": _pro0813_zoff_bytes_per_rank(524288),
        "mla_off_max_bytes": 128 * 1024**3,
        # A complete 8x64K Pro BF16 z_off bank is exactly 110 GiB per TP rank.
        # Keeping
        # that bank resident together with the >=512K scheduler KV pool leaves
        # too little temporary headroom once scheduler KV and model weights are
        # included, so keep the bounded CPU-authoritative path. The controller has a
        # fail-closed CPU-authoritative restore path: assemble one layer's
        # clean rows on CPU, transfer that layer once, then release it before
        # advancing.  Use that bounded-memory path for 512K; the follow-up
        # optimization is a two-layer pinned prefetch ring, not an impossible
        # all-segment device mirror.
        "mla_off_device_max_bytes": 0,
        "mem_fraction_static": _PRO0813_SCALE.PRO0813_MEM_FRACTION_STATIC[524288],
        "row_sparse_checkpoint_max_islands": (
            _PRO0813_SCALE.PRO0813_CHECKPOINT_MAX_ISLANDS[524288]
        ),
        "requires_qualification_profile": True,
    },
}


def _builtin_target_profile_identity(target_tokens: int, profile: dict) -> str:
    """Hash the portable, immutable built-in target contract."""

    def portable(value):
        if isinstance(value, Path):
            resolved = value.expanduser().resolve()
            try:
                return "repo:" + resolved.relative_to(REPO).as_posix()
            except ValueError:
                return str(resolved)
        if isinstance(value, dict):
            return {
                str(key): portable(item)
                for key, item in sorted(value.items())
            }
        if isinstance(value, (list, tuple)):
            return [portable(item) for item in value]
        return value

    payload = {
        "schema": "redknot_pro0813_builtin_target_profile_v1",
        "target_tokens": int(target_tokens),
        "profile": portable(profile),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _resolve_redknot_target_profile(args) -> dict:
    base = REDKNOT_TARGET_PROFILES.get(args.target_tokens)
    if base is None:
        raise ValueError(
            "context-bound RedKnot currently supports exactly 65536, "
            "131072, 262144, 450560, or 524288 offline document tokens"
        )
    profile = dict(base)
    projected = _pro0813_zoff_bytes_per_rank(args.target_tokens)
    if int(profile["zoff_projected_bytes_per_rank"]) != projected:
        raise ValueError("Pro-0813 z_off capacity table differs from its formula")
    if int(profile["mla_off_max_bytes"]) < projected:
        raise ValueError("Pro-0813 z_off cap is smaller than the exact projection")
    cpu_reservation = _PRO0813_SCALE.pro0813_cpu_reservation_bytes_per_rank(
        args.target_tokens
    )
    if int(profile["mla_off_max_bytes"]) < cpu_reservation:
        raise ValueError(
            "Pro-0813 CPU artifact cap is smaller than its full reservation"
        )
    profile["dataset"] = "musique"
    profile["num_queries"] = 1
    profile["query_row_ids"] = [int(profile["query_row_id"])]
    profile["prompt_manifest"] = ""
    profile["prompt_manifest_sha256"] = ""
    profile["long_output_tokens"] = 0
    if not args.qualification_profile:
        if profile.get("requires_qualification_profile"):
            raise ValueError(
                "440K/512K requires an immutable multi-dataset qualification profile"
            )
        profile["qualification_profile_path"] = (
            f"builtin:pro0813:{int(args.target_tokens)}"
        )
        profile["qualification_profile_sha256"] = (
            _builtin_target_profile_identity(args.target_tokens, profile)
        )
        return profile
    source = Path(args.qualification_profile).expanduser()
    if not source.is_absolute():
        raise ValueError("qualification profile path must be absolute")
    if args.target_tokens in (450560, 524288):
        verifier_path = Path(__file__).with_name(
            "verify_pro0813_qualification_profile.py"
        )
        verifier_spec = importlib.util.spec_from_file_location(
            "redknot_pro0813_qualification_verifier", verifier_path
        )
        if verifier_spec is None or verifier_spec.loader is None:
            raise RuntimeError(
                f"cannot load Pro qualification verifier: {verifier_path}"
            )
        verifier_module = importlib.util.module_from_spec(verifier_spec)
        verifier_spec.loader.exec_module(verifier_module)

        expected_profile_sha256 = os.environ.get(
            "REDKNOT_EXPECTED_QUALIFICATION_PROFILE_SHA256", ""
        ).strip()
        verification = verifier_module.verify_profile(
            source,
            expected_target_tokens=int(args.target_tokens),
            model_root=MODEL,
            data_dir=DATA_DIR,
            expected_profile_sha256=(
                expected_profile_sha256 or None
            ),
            include_profile_bytes=True,
        )
        source = Path(verification["profile"])
        profile_bytes = verification.get("_verified_profile_bytes")
        if not isinstance(profile_bytes, bytes):
            raise RuntimeError(
                "qualification verifier did not return immutable profile bytes"
            )
        if hashlib.sha256(profile_bytes).hexdigest() != verification.get(
            "profile_sha256"
        ):
            raise RuntimeError(
                "qualification verifier returned profile bytes with a "
                "different SHA-256"
            )
        try:
            document = json.loads(profile_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"verified qualification profile is not UTF-8 JSON: {error}"
            ) from error
    else:
        verification = None
        source = source.resolve()
        profile_bytes = source.read_bytes()
        try:
            document = json.loads(profile_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"qualification profile is not UTF-8 JSON: {error}"
            ) from error
    if not isinstance(document, dict):
        raise ValueError("qualification profile must contain one JSON object")
    profile_format = document.get("format")
    if profile_format not in {
        "redknot_multidataset_256k_profile_v1",
        "redknot_multidataset_profile_v2",
    }:
        raise ValueError("unsupported multi-dataset qualification profile")
    if profile_format == "redknot_multidataset_256k_profile_v1" and (
        args.target_tokens != 262144
    ):
        raise ValueError("v1 multi-dataset profiles are frozen at 256K")
    if profile_format == "redknot_multidataset_profile_v2" and (
        document.get("target_tokens") != args.target_tokens
    ):
        raise ValueError(
            "v2 qualification profile target length differs from the request"
        )
    for key, expected in (
        ("num_chunks", int(base["num_chunks"])),
        ("chunk_tokens", int(base["chunk_tokens"])),
        ("query_start", int(args.target_tokens)),
    ):
        if document.get(key) != expected:
            raise ValueError(
                f"qualification profile has invalid {key}: {document.get(key)!r}"
            )
    query_rows = document.get("query_row_ids")
    num_queries = document.get("num_queries")
    if (
        type(num_queries) is not int
        or num_queries <= 0
        or not isinstance(query_rows, list)
        or len(query_rows) != num_queries
        or any(type(value) is not int or value < 0 for value in query_rows)
        or len(set(query_rows)) != len(query_rows)
    ):
        raise ValueError("qualification profile has invalid query rows")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != num_queries:
        raise ValueError("qualification profile has invalid prompt cases")
    long_output_tokens = int(document.get("long_output_target_tokens", 0))
    if long_output_tokens not in (0, 30, 50):
        raise ValueError("qualification profile has invalid long-output target")
    profile.update(
        {
            "dataset": str(document["dataset"]),
            "num_queries": num_queries,
            "query_row_id": query_rows[0],
            "query_row_ids": query_rows,
            "data_manifest": (
                source.parent / document["data_manifest"]
                if verification is not None
                else Path(document["data_manifest"])
            ),
            "selection_sha256": str(document["data_selection_sha256"]),
            "prompt_manifest": str(
                source.parent / document["prompt_manifest"]
                if verification is not None
                else document["prompt_manifest"]
            ),
            "prompt_manifest_sha256": str(
                document["prompt_manifest_sha256"]
            ),
            "prompt_text_sha256": str(cases[0]["text_sha256"]),
            "full_input_ids_sha256": str(
                cases[0]["full_input_ids_sha256"]
            ),
            "full_input_tokens": int(document["max_total_tokens"]),
            "long_output_tokens": long_output_tokens,
            "qualification_profile_sha256": (
                verification["profile_sha256"]
                if verification is not None
                else hashlib.sha256(profile_bytes).hexdigest()
            ),
            "qualification_profile_path": str(source),
            "intended_execution_profile": document.get(
                "intended_execution_profile", ""
            ),
            "diagnostic_execution_profiles": list(
                document.get("diagnostic_execution_profiles", [])
            ),
        }
    )
    profile.pop("requires_qualification_profile", None)
    return profile


def _set(name: str, value: object) -> None:
    os.environ[name] = str(value)


def _strict_binary_env(name: str) -> str:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be exactly 0 or 1, got {value!r}")
    return value


def _validate_system_optimizer_dependencies() -> None:
    if os.environ.get("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0") == "1":
        import deep_gemm

        if not callable(getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)):
            raise RuntimeError(
                "SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 requires "
                "deep_gemm.tf32_hc_prenorm_gemm; this runtime does not "
                "provide that kernel"
            )


def _unset_legacy_environment() -> None:
    for name in (
        "REDKNOT_C4_TOPK_CLAMP",
        "REDKNOT_IH_DATA_MANIFEST_OUT",
        "REDKNOT_IH_DATA_EXCLUDE_MANIFESTS",
        "REDKNOT_IH_SELECTION_POLICY",
        "REDKNOT_IH_CHECKPOINT_STRIDE",
        "REDKNOT_IH_CHECKPOINT_MAX_ISLANDS",
        "REDKNOT_IH_ACTIVE_BUDGET_RATIO",
        "REDKNOT_IH_QUERY_PROTECTION_TOKENS",
        "REDKNOT_IH_GENERALIZED_ADAPTIVE_CONTROLLER",
        "REDKNOT_IH_GENERALIZED_STRONG_ACTIVE_RATIO",
        "REDKNOT_IH_GENERALIZED_MEDIUM_ACTIVE_RATIO",
        "REDKNOT_IH_GENERALIZED_DIFFUSE_ACTIVE_RATIO",
        "REDKNOT_IH_HOT_MAX_PER_SEGMENT_RATIO",
        "REDKNOT_IH_HOT_FRAC",
        "REDKNOT_IH_SKIP_PREFIX_RECOMPUTE",
        "REDKNOT_IH_SERVER_POLICY_MANIFEST",
        "REDKNOT_IH_SERVER_INSTANCE_NONCE",
        "REDKNOT_IH_PROMPT_MANIFEST",
    ):
        os.environ.pop(name, None)


def _prepare_output_paths(
    args, *, target_profile: dict | None = None
) -> tuple[Path, Path, Path, Path]:
    result = Path(args.out).expanduser().resolve()
    server_log = Path(args.log).expanduser().resolve()
    if args.mode == "redknot" and target_profile is None:
        raise ValueError("RedKnot output preparation requires one resolved profile")
    if args.mode != "redknot" and target_profile is not None:
        raise ValueError("native output preparation cannot consume a RedKnot profile")
    prompt_manifest_input = (
        Path(target_profile["prompt_manifest"]).expanduser().resolve()
        if target_profile and target_profile.get("prompt_manifest")
        else None
    )
    prompt_manifest = prompt_manifest_input or (
        Path(args.prompt_manifest_out).expanduser().resolve()
        if args.prompt_manifest_out
        else result.with_name(result.stem + ".prompt_manifest.json")
    )
    rank_logs = (
        Path(args.rank_log_dir).expanduser().resolve()
        if args.rank_log_dir
        else result.with_name(result.stem + ".ranklogs")
    )
    for path in (result, server_log):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite benchmark artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if prompt_manifest_input is not None:
        if not prompt_manifest_input.is_file():
            raise FileNotFoundError(
                f"frozen prompt manifest is absent: {prompt_manifest_input}"
            )
    else:
        if prompt_manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite benchmark artifact: {prompt_manifest}"
            )
        prompt_manifest.parent.mkdir(parents=True, exist_ok=True)
    rank_logs.mkdir(parents=True, exist_ok=False)
    return result, server_log, prompt_manifest, rank_logs


def _configure_native(args, result: Path) -> None:
    length_label = {
        8000: "8K",
        16000: "16K",
        32000: "32K",
        64000: "64K",
        65536: "64K",
        128000: "128K",
        131072: "128K",
        256000: "256K",
        262144: "256K",
    }.get(args.target_tokens)
    if length_label is None:
        raise ValueError(
            "native target tokens must map to 8K/16K/32K/64K/128K/256K"
        )
    contract = _validate_pro0813_contract(MODEL / "config.json")
    _set("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    _set("REDKNOT_MODEL_PATH", MODEL)
    _set("REDKNOT_DSV4_VARIANT", "pro0813")
    _set("REDKNOT_PRO0813_VARIANT", contract["variant"])
    _set("REDKNOT_PRO0813_GEOMETRY_DIGEST", contract["geometry_digest"])
    _set("REDKNOT_PRO0813_CONFIG_SHA256", contract["official_config_sha256"])
    _set("REDKNOT_PRO0813_INDEX_TOPK", PRO0813_INDEX_TOPK)
    _set("REDKNOT_TP_SIZE", PRO0813_TP_SIZE)
    _set("REDKNOT_MOE_RUNNER_BACKEND", "flashinfer_mxfp4")
    _set("REDKNOT_MEM_FRACTION_STATIC", args.mem_fraction_static)
    _set("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", 1)
    _set("SGLANG_BARE_SUBPROCESS_LAUNCH", 1)
    _set("REDKNOT_ENGINE_MODE", "baseline")
    _set("REDKNOT_DATASETS", args.datasets)
    _set("REDKNOT_LENGTHS", length_label)
    _set("REDKNOT_N_SAMPLES", args.n_samples)
    _set("REDKNOT_MAX_NEW", args.max_new)
    _set("REDKNOT_RESULT_OUT", result)
    _set("REDKNOT_SPARSE_FFN", 0)
    _set("REDKNOT_THREE_WAY_CLOSURE", 0)
    _set("REDKNOT_PROGRESSIVE_TOPK_SCHEDULE", "")
    _set("REDKNOT_ADAPTIVE_TOPK", int(args.adaptive_topk))
    _set("REDKNOT_ADAPTIVE_TOPK_PLAN_SCOPED", 0)
    _set("REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS", args.adaptive_topk_mass)
    _set("REDKNOT_ADAPTIVE_TOPK_BUCKETS", args.adaptive_topk_buckets)
    _set("REDKNOT_ADAPTIVE_TOPK_MIN_TOKENS", args.adaptive_topk_min_tokens)
    _set("REDKNOT_ENGINE_WARMUP_ITERS", args.engine_warmup_iters)
    _set(
        "REDKNOT_ADAPTIVE_TOPK_LOG_FIRST_HISTOGRAM",
        int(args.adaptive_topk_log_histogram),
    )


def _configure_redknot(
    args,
    result: Path,
    server_log: Path,
    prompt_manifest: Path,
    rank_logs: Path,
    target_profile: dict,
) -> None:
    combined_diagnostic_zoff_only = bool(
        getattr(
            args,
            "combined_headsplit_row_sparse_diagnostic_zoff_only",
            False,
        )
    )
    combined_headsplit_row_sparse = bool(
        args.combined_headsplit_row_sparse or combined_diagnostic_zoff_only
    )
    row_sparse_enabled = bool(
        args.row_sparse_online or combined_headsplit_row_sparse
    )
    if args.adaptive_topk and args.progressive_topk_schedule:
        raise ValueError(
            "adaptive Top-K and progressive Top-K are mutually exclusive; "
            "use --progressive-topk-schedule '' for the physical adaptive run"
        )
    if row_sparse_enabled and args.prefix_materialization:
        raise ValueError(
            "row-sparse qualification cannot use full radix materialization"
        )
    if row_sparse_enabled and not args.first_document_prefix:
        raise ValueError(
            "row-sparse qualification requires --first-document-prefix; "
            "document 1 is served entirely from its offline artifact"
        )
    # A combined selected-row request is allowed to consume exactly one
    # scheduler-authenticated document-1 radix prefix.  The runtime binds the
    # seed and consumer to the same token hash, receipt key and compressor
    # terminal-state receipt; documents 2..N remain on the combined
    # head-split + selected-row path.  Full-prefix materialization stays
    # forbidden above for row-sparse qualification.
    radix_prefix_enabled = bool(
        args.prefix_materialization or args.first_document_prefix
    )
    if radix_prefix_enabled and args.radix_eviction_policy != "lfu":
        raise ValueError(
            "prefix materialization requires radix eviction policy=lfu so "
            "the repeatedly reused fixed prefix is not displaced by one-shot "
            "dense namespaces under concurrent load"
        )
    if not isinstance(target_profile, dict):
        raise ValueError("RedKnot configuration requires one resolved profile")
    profile_algorithm = str(
        target_profile.get("intended_execution_profile", "")
    )
    if profile_algorithm:
        if combined_diagnostic_zoff_only:
            diagnostic_profiles = target_profile.get(
                "diagnostic_execution_profiles", []
            )
            if (
                not isinstance(diagnostic_profiles, list)
                or "zoff_only" not in diagnostic_profiles
            ):
                raise ValueError(
                    "qualification profile does not authorize the explicit "
                    "zoff_only diagnostic execution"
                )
        elif args.combined_headsplit_row_sparse:
            if profile_algorithm != "full_combined_production_v1":
                raise ValueError(
                    "qualification profile is not bound to the full combined "
                    "production algorithm"
                )
        else:
            raise ValueError(
                "the frozen long-context qualification profile is bound to "
                "full combined production (or its explicit zoff_only "
                "diagnostic), not another execution arm"
            )
    if (
        not target_profile["prompt_text_sha256"]
        or not target_profile["full_input_ids_sha256"]
        or int(target_profile["full_input_tokens"]) <= 0
    ):
        raise ValueError(
            "the selected target profile is missing its frozen official-prompt "
            "identity"
        )
    data_manifest = Path(
        args.data_manifest or target_profile["data_manifest"]
    ).expanduser().resolve()
    dataset_name = str(target_profile["dataset"])
    for path in (
        DRIVER,
        LAUNCHER,
        MODEL / "config.json",
        DATA_DIR / f"{dataset_name}.jsonl",
        data_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required benchmark input is absent: {path}")
    contract = _validate_pro0813_contract(MODEL / "config.json")

    _unset_legacy_environment()
    _set("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    _set("REDKNOT_MODEL_PATH", MODEL)
    _set("REDKNOT_DSV4_VARIANT", "pro0813")
    _set("REDKNOT_PRO0813_VARIANT", contract["variant"])
    _set("REDKNOT_PRO0813_GEOMETRY_DIGEST", contract["geometry_digest"])
    _set("REDKNOT_PRO0813_CONFIG_SHA256", contract["official_config_sha256"])
    _set("REDKNOT_PRO0813_INDEX_TOPK", PRO0813_INDEX_TOPK)
    _set("REDKNOT_LONGBENCH_DIR", DATA_DIR)
    _set("REDKNOT_DATASETS", dataset_name)
    _set("REDKNOT_ENGINE_MODE", "indexer_hot")
    _set("REDKNOT_IH_NO_LAUNCH", 0)

    # Pure context-bound MLA head split, or the separately-labelled
    # selected-row systems-ceiling arm.  The latter intentionally keeps
    # MLA-off disabled until its accuracy/speed contract is established.
    _set(
        "REDKNOT_IH_MLA_OFFLOAD",
        int(combined_headsplit_row_sparse or not row_sparse_enabled),
    )
    _set("REDKNOT_IH_ROW_SPARSE_CLOSURE", int(row_sparse_enabled))
    _set(
        "REDKNOT_IH_COMBINED_HEADSPLIT_ROW_SPARSE",
        int(combined_headsplit_row_sparse),
    )
    _set("REDKNOT_IH_TP_SIZE", 8)
    _set("REDKNOT_TP_SIZE", 8)
    _set("REDKNOT_V4_MODE", "aggressive")
    _set("REDKNOT_MLA_PASS_MODE", "headwise")
    _set("REDKNOT_MLA_DENSE_PREFIX_LAYERS", 3)
    _set("REDKNOT_MLA_DENSE_SUFFIX_LAYERS", 3)
    _set("REDKNOT_MLA_LOCAL_WINDOW", 128)
    _set("REDKNOT_MLA_GLOBAL_HEAD_STRIDE", 8)
    _set("REDKNOT_MLA_GLOBAL_LAYER_STRIDE", 0)
    _set("REDKNOT_MLA_OFF_GLOBAL_ATTN_IMPL", "triton_h1")
    _set("REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE", 1)
    _set("REDKNOT_SHARED_LATENT_GPU", 1)
    # The current sparse-FFN selector is certified only for an exact pure-MLA
    # restore receipt.  Do not weaken that contract for the standalone
    # selected-row systems-ceiling arm: physical row pruning already removes
    # its skipped tokens, while adaptive routed K remains active.
    sparse_ffn_enabled = bool(
        args.token_sparse_ffn
        and (combined_headsplit_row_sparse or not row_sparse_enabled)
    )
    _set("REDKNOT_SPARSE_FFN", int(sparse_ffn_enabled))
    _set("REDKNOT_THREE_WAY_CLOSURE", int(sparse_ffn_enabled))
    # The Pro-0813 launcher intentionally does not forward sparse-FFN
    # CLI knobs when the feature is disabled.  Normalize inactive policy
    # metadata to the server defaults as well; otherwise benchmark preflight
    # hashes user-supplied-but-inactive values while the runtime manifest
    # hashes its defaults and the pure MLA run fails before the first snapshot.
    sparse_mass = args.token_sparse_mass if sparse_ffn_enabled else 0.60
    sparse_deep_mass = args.token_sparse_deep_mass if sparse_ffn_enabled else 0.60
    sparse_importance = (
        args.token_sparse_importance if sparse_ffn_enabled else "activation"
    )
    sparse_min_full_ratio = (
        args.token_sparse_min_full_ratio if sparse_ffn_enabled else 0.20
    )
    sparse_max_full_ratio = (
        args.token_sparse_max_full_ratio if sparse_ffn_enabled else 0.80
    )
    _set("REDKNOT_FFN_DENSE_UNTIL", 3)
    _set("REDKNOT_FFN_DENSE_SUFFIX_LAYERS", 3)
    _set("REDKNOT_FFN_MASS", sparse_mass)
    _set("REDKNOT_FFN_DEEP_START", args.token_sparse_deep_start)
    _set("REDKNOT_FFN_MASS_DEEP", sparse_deep_mass)
    _set("REDKNOT_FFN_RECENT_N", args.token_sparse_recent_tokens)
    _set("REDKNOT_FFN_BOUNDARY_TOKENS", args.token_sparse_boundary_tokens)
    _set("REDKNOT_FFN_MIN_SEQ_LEN", args.token_sparse_min_seq_len)
    _set("REDKNOT_FFN_IMPORTANCE", sparse_importance)
    _set("REDKNOT_FFN_MIN_FULL_RATIO", sparse_min_full_ratio)
    _set("REDKNOT_FFN_MAX_FULL_RATIO", sparse_max_full_ratio)
    _set("REDKNOT_FFN_BLOCK_TOKENS", args.token_sparse_block_tokens)
    _set(
        "REDKNOT_FFN_FREEZE_BLOCK_SELECTION",
        int(args.token_sparse_freeze_blocks),
    )
    _set("SGLANG_REDKNOT_FFN_DEBUG", int(args.token_sparse_ffn))
    _set(
        "REDKNOT_IH_PREFIX_MATERIALIZATION",
        int(radix_prefix_enabled),
    )
    _set(
        "REDKNOT_IH_PREFIX_MATERIALIZATION_SCOPE",
        (
            "first_document"
            if args.first_document_prefix
            else ("full" if args.prefix_materialization else "none")
        ),
    )
    if row_sparse_enabled:
        _set("REDKNOT_IH_SELECTION_POLICY", "checkpoint_islands")
        _set("REDKNOT_IH_CHECKPOINT_STRIDE", args.row_sparse_checkpoint_stride)
        _set(
            "REDKNOT_IH_CHECKPOINT_MAX_ISLANDS",
            args.row_sparse_checkpoint_max_islands,
        )
        requested_active_ratio_floor = (
            min(
                args.generalized_strong_active_ratio,
                args.generalized_medium_active_ratio,
                args.generalized_diffuse_active_ratio,
            )
            if args.generalized_adaptive_controller
            else args.row_sparse_active_ratio
        )
        _set(
            "REDKNOT_IH_MIN_REALIZED_ACTIVE_RATIO",
            _PRO0813_SCALE.pro0813_min_realized_active_ratio(
                args.target_tokens,
                requested_active_ratio_floor,
                checkpoint_stride=args.row_sparse_checkpoint_stride,
            ),
        )
        _set(
            "REDKNOT_IH_ACTIVE_BUDGET_RATIO",
            args.row_sparse_active_ratio,
        )
        _set(
            "REDKNOT_IH_QUERY_PROTECTION_TOKENS",
            args.query_protection_tokens,
        )
        _set(
            "REDKNOT_IH_HOT_MAX_PER_SEGMENT_RATIO",
            args.row_sparse_segment_cap_ratio,
        )
        _set("REDKNOT_IH_SKIP_PREFIX_RECOMPUTE", 1)
        _set(
            "REDKNOT_IH_GENERALIZED_ADAPTIVE_CONTROLLER",
            int(args.generalized_adaptive_controller),
        )
        _set(
            "REDKNOT_IH_GENERALIZED_STRONG_ACTIVE_RATIO",
            args.generalized_strong_active_ratio,
        )
        _set(
            "REDKNOT_IH_GENERALIZED_MEDIUM_ACTIVE_RATIO",
            args.generalized_medium_active_ratio,
        )
        _set(
            "REDKNOT_IH_GENERALIZED_DIFFUSE_ACTIVE_RATIO",
            args.generalized_diffuse_active_ratio,
        )
    _set("REDKNOT_RADIX_EVICTION_POLICY", args.radix_eviction_policy)
    _set(
        "REDKNOT_MLA_OFF_GEOMETRY_TEMPLATE_CACHE",
        int(args.geometry_template_cache),
    )
    _set(
        "REDKNOT_IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS",
        args.restore_pipeline_group_layers,
    )

    # Assignment-sparse MoE. Every token is preserved; only routed K changes.
    _set("REDKNOT_PROGRESSIVE_TOPK_SCHEDULE", args.progressive_topk_schedule)
    _set("REDKNOT_ADAPTIVE_TOPK", int(args.adaptive_topk))
    _set(
        "REDKNOT_ADAPTIVE_TOPK_PLAN_SCOPED",
        int(combined_headsplit_row_sparse),
    )
    _set("REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS", args.adaptive_topk_mass)
    _set("REDKNOT_ADAPTIVE_TOPK_BUCKETS", args.adaptive_topk_buckets)
    _set("REDKNOT_ADAPTIVE_TOPK_MIN_TOKENS", args.adaptive_topk_min_tokens)
    _set("REDKNOT_ADAPTIVE_TOPK_DENSE_PREFIX_LAYERS", 3)
    _set("REDKNOT_ADAPTIVE_TOPK_DENSE_SUFFIX_LAYERS", 3)
    _set("REDKNOT_ADAPTIVE_TOPK_PHYSICAL_COMPACTION", 1)
    _set(
        "REDKNOT_ADAPTIVE_TOPK_LOG_FIRST_HISTOGRAM",
        int(args.adaptive_topk_log_histogram),
    )

    _set("REDKNOT_IH_NUM_CHUNKS", target_profile["num_chunks"])
    _set("REDKNOT_IH_CHUNK_TOKENS", target_profile["chunk_tokens"])
    _set(
        "REDKNOT_IH_QUALIFICATION_PROFILE_PATH",
        target_profile["qualification_profile_path"],
    )
    _set(
        "REDKNOT_IH_QUALIFICATION_PROFILE_SHA256",
        target_profile["qualification_profile_sha256"],
    )
    if args.native_indexer_doc_cap:
        if int(target_profile["num_chunks"]) != int(args.native_indexer_documents):
            raise ValueError(
                "native Indexer document count must equal the frozen offline "
                "chunk count"
            )
        if int(target_profile["chunk_tokens"]) % 4:
            raise ValueError("native Indexer C4 document geometry is not integral")
        _set("REDKNOT_NATIVE_INDEXER_DOC_CAP", args.native_indexer_doc_cap)
        _set(
            "REDKNOT_NATIVE_INDEXER_DOCUMENTS",
            args.native_indexer_documents,
        )
        _set(
            "REDKNOT_NATIVE_INDEXER_C4_ROWS_PER_DOCUMENT",
            int(target_profile["chunk_tokens"]) // 4,
        )
    else:
        for name in (
            "REDKNOT_NATIVE_INDEXER_DOC_CAP",
            "REDKNOT_NATIVE_INDEXER_DOCUMENTS",
            "REDKNOT_NATIVE_INDEXER_C4_ROWS_PER_DOCUMENT",
        ):
            os.environ.pop(name, None)
    # Pure independent-document artifacts are captured at local position zero
    # and relocated online.  Every nonzero destination segment therefore
    # recomputes its first 128 local-head rows; context-bound boundary=0 is a
    # different algorithm and must never be selected by this entry point.
    _set("REDKNOT_IH_BOUNDARY", 128)
    _set("REDKNOT_IH_MERGED_PREFILL_TOKENS", args.merged_prefill_tokens)
    _set("REDKNOT_IH_MLA_OFF_REFRESH_LAYER_STRIDE", 0)
    _set("REDKNOT_IH_MLA_OFF_HOT_EXPAND_TOKENS", 0)
    _set("REDKNOT_IH_MLA_OFF_COMPACT_WOA", 0)
    _set(
        "REDKNOT_IH_MLA_OFF_DIAGNOSTIC_ABLATION",
        "zoff_only" if combined_diagnostic_zoff_only else "full",
    )
    _set(
        "REDKNOT_IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS",
        target_profile["full_input_tokens"]
        if combined_headsplit_row_sparse
        else 0,
    )
    _set(
        "REDKNOT_IH_MLA_OFF_QUALIFICATION_ONLY",
        int(not row_sparse_enabled),
    )
    _set(
        "REDKNOT_IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS",
        0 if row_sparse_enabled else target_profile["full_input_tokens"],
    )

    # Frozen real MuSiQue row and official DSV4 prompt.
    _set("REDKNOT_IH_DATA_ROW_OFFSET", 0)
    _set("REDKNOT_IH_DATA_MANIFEST", data_manifest)
    _set(
        "REDKNOT_IH_EXPECTED_DATA_SELECTION_SHA256",
        target_profile["selection_sha256"],
    )
    _set("REDKNOT_IH_EXPECTED_DATASET", dataset_name)
    _set("REDKNOT_IH_EXPECTED_QUERY_ROW_ID", target_profile["query_row_id"])
    _set(
        "REDKNOT_IH_EXPECTED_QUERY_ROW_IDS",
        ",".join(map(str, target_profile["query_row_ids"])),
    )
    _set("REDKNOT_IH_PURE_PROMPT_MODE", "official_rag_v1")
    _set(
        "REDKNOT_IH_LONG_OUTPUT_TOKENS",
        target_profile.get("long_output_tokens", 0),
    )
    _set("REDKNOT_THINKING_MODE", "chat")
    _set("REDKNOT_REASONING_EFFORT", "low")
    _set(
        "REDKNOT_IH_EXPECTED_PROMPT_TEXT_SHA256",
        target_profile["prompt_text_sha256"],
    )
    _set(
        "REDKNOT_IH_EXPECTED_FULL_INPUT_IDS_SHA256",
        target_profile["full_input_ids_sha256"],
    )
    _set(
        "REDKNOT_IH_EXPECTED_FULL_INPUT_TOKENS",
        target_profile["full_input_tokens"],
    )
    if target_profile.get("prompt_manifest"):
        _set("REDKNOT_IH_PROMPT_MANIFEST", prompt_manifest)
        os.environ.pop("REDKNOT_IH_PROMPT_MANIFEST_OUT", None)
    else:
        _set("REDKNOT_IH_PROMPT_MANIFEST_OUT", prompt_manifest)
        os.environ.pop("REDKNOT_IH_PROMPT_MANIFEST", None)
    _set(
        "REDKNOT_IH_EXPECTED_PROMPT_MANIFEST_SHA256",
        target_profile.get("prompt_manifest_sha256", ""),
    )
    _set("REDKNOT_IH_NUM_QUERIES", target_profile["num_queries"])
    _set("REDKNOT_IH_QUALITY_REPEATS", args.quality_repeats)
    _set("REDKNOT_IH_MAX_NEW", args.max_new)
    _set("REDKNOT_IH_RELEVANCE_FIRST", 0)
    _set("REDKNOT_IH_RELEVANCE_LAST", 0)

    # Accuracy first; timing is diagnostic until fidelity passes.
    _set("REDKNOT_IH_TTFT_WARMUP", args.ttft_warmup)
    _set("REDKNOT_IH_TTFT_ITERS", args.ttft_iters)
    _set("REDKNOT_IH_MEASURE_QPS", int(args.measure_qps))
    _set("REDKNOT_IH_QPS_CONCURRENCIES", args.qps_concurrencies)
    _set("REDKNOT_IH_QPS_WARMUP_WAVES", args.qps_warmup_waves)
    _set("REDKNOT_IH_QPS_WAVES", args.qps_waves)
    _set("REDKNOT_IH_STRICT_PERFORMANCE_CLAIMS", int(args.strict_performance))
    _set("REDKNOT_IH_MIN_TOP1_RATE", 0.95)
    _set("REDKNOT_IH_MIN_COSINE", args.min_cosine)
    _set("REDKNOT_IH_MIN_F1_RETENTION", 0.98)
    _set("REDKNOT_IH_MIN_EM_RETENTION", 0.98)
    _set("REDKNOT_IH_MIN_DENSE_F1", 0.0)
    _set("REDKNOT_IH_MIN_REUSE_F1", 0.0)
    _set("REDKNOT_IH_MIN_DENSE_EM", 0.0)
    _set("REDKNOT_IH_MIN_REUSE_EM", 0.0)
    _set("REDKNOT_IH_MIN_TOKEN_AGREEMENT", args.min_token_agreement)
    _set("REDKNOT_IH_MIN_SPEEDUP", args.ttft_target_speedup)
    _set("REDKNOT_IH_MIN_QPS_SPEEDUP", 1.5 if row_sparse_enabled else 2.0)
    _set("REDKNOT_IH_MIN_HEAD_ROW_SAVING", 0.70)

    _set("REDKNOT_IH_MLA_OFF_MAX_BYTES", target_profile["mla_off_max_bytes"])
    _set(
        "REDKNOT_IH_MLA_OFF_PROJECTED_ZOFF_BYTES",
        target_profile["zoff_projected_bytes_per_rank"],
    )
    _set(
        "REDKNOT_MLA_OFF_DEVICE_MAX_BYTES",
        target_profile["mla_off_device_max_bytes"],
    )
    _set("REDKNOT_IH_MIN_GPU_FREE_MIB", args.min_gpu_free_mib)
    _set("REDKNOT_MEM_FRACTION_STATIC", args.mem_fraction_static)
    _set(
        "REDKNOT_MOE_RUNNER_BACKEND",
        os.environ.get("REDKNOT_MOE_RUNNER_BACKEND", "flashinfer_mxfp4"),
    )
    _set("REDKNOT_DISABLE_CUDA_GRAPH", 1)
    _set("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", 1)
    _set("SGLANG_BARE_SUBPROCESS_LAUNCH", 1)
    # System-only kernel switches are explicit, binary and preserved from the
    # launcher.  Their exact values are also emitted into the structured
    # benchmark result, so a faster run cannot be mistaken for the default
    # implementation or silently inherit a malformed environment value.
    for name in (
        "SGLANG_OPT_USE_TILELANG_MHC_PRE",
        "SGLANG_OPT_USE_TILELANG_MHC_POST",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
        "REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH",
    ):
        _set(name, _strict_binary_env(name))
    _validate_system_optimizer_dependencies()
    _set("SGLANG_OPT_USE_TOPK_V2", 0)
    # Torch renamed this variable, while the pinned cluster build still accepts
    # the legacy CUDA spelling.  Set both to the identical value so the 256K
    # snapshot tail cannot silently run with the default fragmented allocator.
    _set("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    _set("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _set("PYTHONUNBUFFERED", 1)
    _set("NO_PROXY", "127.0.0.1,localhost")
    _set("REDKNOT_IH_PORT", args.port)
    _set("REDKNOT_IH_VENV_PY", "/workspace/RedKnot/.venv_sm103/bin/python")
    _set("REDKNOT_IH_SERVER_SCRIPT", LAUNCHER)
    _set("REDKNOT_IH_SERVER_LOG", server_log)
    _set("REDKNOT_IH_RANK_LOG_DIR", rank_logs)
    _set("REDKNOT_ENABLE_METRICS", 1)
    _set("REDKNOT_IH_REQUIRE_MODEL_TTFT", 1)
    _set("REDKNOT_RESULT_OUT", result)


def main() -> None:
    _pin_pro0813_python_sources()
    parser = argparse.ArgumentParser(
        description=(
            "DeepSeek-V4-Pro-0813 RedKnot reproduction on 8x B300/SM103; "
            "the 64K arm is the minimum expected benefit point"
        )
    )
    parser.add_argument("--mode", choices=("baseline", "redknot"), default="redknot")
    parser.add_argument("--port", type=int, default=31998)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument(
        "--datasets",
        default="musique,hotpotqa,2wikimqa,multifieldqa_en",
        help="Comma-separated real LongBench datasets for baseline/full-recompute mode.",
    )
    parser.add_argument("--target-tokens", type=int, default=OFFLINE_DOCUMENT_TOKENS)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--out", default="")
    parser.add_argument("--log", default="")
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="CPU-only official config/geometry/profile/z_off audit; never imports torch.",
    )
    parser.add_argument(
        "--contract-config",
        default=str(MODEL / "config.json"),
        help="Config used only by --contract-only; GPU runs remain pinned to MODEL.",
    )
    parser.add_argument(
        "--data-manifest",
        default="",
        help="Optional exact data manifest override; default follows target length.",
    )
    parser.add_argument(
        "--qualification-profile",
        default="",
        help=(
            "Frozen multi-query 256K dataset/prompt profile. RedKnot replays "
            "its exact data and prompt manifests before server launch."
        ),
    )
    parser.add_argument("--prompt-manifest-out", default="")
    parser.add_argument("--rank-log-dir", default="")
    parser.add_argument(
        "--progressive-topk-schedule",
        default=PRO0813_PROGRESSIVE_TOPK_SCHEDULE,
        help="Inclusive layer schedule; empty string keeps native K=6.",
    )
    parser.add_argument("--adaptive-topk", action="store_true")
    parser.add_argument("--adaptive-topk-mass", type=float, default=0.90)
    parser.add_argument("--adaptive-topk-buckets", default="4,5,6")
    parser.add_argument("--adaptive-topk-min-tokens", type=int, default=512)
    parser.add_argument("--adaptive-topk-log-histogram", action="store_true")
    parser.add_argument(
        "--token-sparse-ffn",
        action="store_true",
        help=(
            "Qualification-only third closure lever for certified pure-MLA "
            "restore, including independent position-0/RoPE relocation: "
            "shared expert on every token and compact routed experts only on "
            "protected/high-salience tokens."
        ),
    )
    parser.add_argument(
        "--token-sparse-mass",
        type=float,
        default=_PRO0813_SCALE.PRO0813_TOKEN_SPARSE_MASS,
    )
    parser.add_argument(
        "--token-sparse-deep-start",
        type=int,
        default=PRO0813_TOKEN_SPARSE_DEEP_START,
    )
    parser.add_argument(
        "--token-sparse-deep-mass",
        type=float,
        default=_PRO0813_SCALE.PRO0813_TOKEN_SPARSE_DEEP_MASS,
    )
    parser.add_argument("--token-sparse-recent-tokens", type=int, default=256)
    parser.add_argument("--token-sparse-boundary-tokens", type=int, default=128)
    parser.add_argument("--token-sparse-min-seq-len", type=int, default=32768)
    parser.add_argument(
        "--token-sparse-importance",
        choices=("activation", "blend", "indexer", "indexer_indegree"),
        default="blend",
    )
    parser.add_argument(
        "--token-sparse-min-full-ratio",
        type=float,
        default=_PRO0813_SCALE.PRO0813_TOKEN_SPARSE_MIN_FULL_RATIO,
    )
    parser.add_argument(
        "--token-sparse-max-full-ratio",
        type=float,
        default=_PRO0813_SCALE.PRO0813_TOKEN_SPARSE_MAX_FULL_RATIO,
    )
    parser.add_argument(
        "--token-sparse-block-tokens",
        type=int,
        default=0,
        help=(
            "Select complete absolute-position blocks for routed MoE; 0 keeps "
            "the historical individual-token selector and 128 enables the "
            "RedKnot block-sparse systems path."
        ),
    )
    parser.add_argument(
        "--token-sparse-freeze-blocks",
        action="store_true",
        help=(
            "Freeze the first eligible layer's block selection for the rest "
            "of the current prefill microforward."
        ),
    )
    parser.add_argument(
        "--engine-warmup-iters",
        type=int,
        default=0,
        help="Unmeasured same-prompt engine warmups before baseline/full-recompute metrics.",
    )
    parser.add_argument("--quality-repeats", type=int, default=3)
    parser.add_argument(
        "--prefix-materialization",
        action="store_true",
        help=(
            "Explicit RedKnot context-bound prefix materialization: retain the "
            "hash-bound offline prefix in the device radix tree and execute "
            "only the online suffix. This is audited separately from ordinary "
            "pure head-row restore."
        ),
    )
    parser.add_argument(
        "--first-document-prefix",
        action="store_true",
        help=(
            "Seed only document 1 into radix cache; every online request "
            "must then execute certified pure RedKnot restore/merge for "
            "documents 2-8 and the query suffix."
        ),
    )
    parser.add_argument(
        "--row-sparse-online",
        action="store_true",
        help=(
            "Explicit systems-ceiling arm: restore offline C4/C128/Indexer "
            "state and propagate only checkpoint-selected document rows plus "
            "the query. This is not labelled as pure MLA head-split."
        ),
    )
    parser.add_argument(
        "--combined-headsplit-row-sparse",
        action="store_true",
        help=(
            "Formal full-composite path: independent position-0 local-head "
            "z_off artifacts, shared KV/Indexer restore, online dirty builders "
            "and fused head merge over checkpoint-selected transformer rows."
        ),
    )
    parser.add_argument(
        "--combined-headsplit-row-sparse-diagnostic-zoff-only",
        action="store_true",
        help=(
            "Claim-ineligible attribution arm that keeps WKV/compressor/Indexer "
            "online and omits only reusable z_off head rows. Requires "
            "--diagnostic-performance."
        ),
    )
    parser.add_argument(
        "--row-sparse-active-ratio",
        type=float,
        default=PRO0813_STANDARD_ACTIVE_RATIO,
    )
    parser.add_argument(
        "--generalized-adaptive-controller",
        action="store_true",
        help=(
            "Select one of three frozen row/protection shapes from an "
            "output-blind query/document lexical sketch. The controller "
            "never consumes dataset identity, gold labels, or model output."
        ),
    )
    parser.add_argument(
        "--generalized-strong-active-ratio",
        type=float,
        default=PRO0813_STRONG_ACTIVE_RATIO,
    )
    parser.add_argument(
        "--generalized-medium-active-ratio",
        type=float,
        default=PRO0813_STANDARD_ACTIVE_RATIO,
    )
    parser.add_argument(
        "--generalized-diffuse-active-ratio",
        type=float,
        default=PRO0813_DIFFUSE_ACTIVE_RATIO,
    )
    parser.add_argument(
        "--query-protection-tokens",
        type=int,
        default=_PRO0813_SCALE.PRO0813_QUERY_PROTECTION_TOKENS,
        help=(
            "Output-blind Top1-document token/local-head protection budget; "
            "must be a 512-token multiple within one document."
        ),
    )
    parser.add_argument(
        "--row-sparse-segment-cap-ratio",
        type=float,
        default=_PRO0813_SCALE.PRO0813_SEGMENT_CAP_RATIO,
    )
    parser.add_argument("--row-sparse-checkpoint-stride", type=int, default=512)
    parser.add_argument(
        "--row-sparse-checkpoint-max-islands",
        type=int,
        default=None,
        help=(
            "Checkpoint descriptor capacity. Default follows the Pro target "
            "profile (64/64/128/256/256 for 64K..512K)."
        ),
    )
    parser.add_argument(
        "--radix-eviction-policy",
        choices=("lru", "lfu", "slru", "priority"),
        default="lfu",
        help=(
            "Device radix-tree eviction policy. Prefix materialization "
            "requires LFU so the fixed offline prefix survives interleaved "
            "one-shot dense traffic."
        ),
    )
    parser.add_argument("--ttft-warmup", type=int, default=3)
    parser.add_argument("--ttft-iters", type=int, default=10)
    parser.add_argument(
        "--ttft-target-speedup",
        type=float,
        default=4.0,
        help="Explicit TTFT qualification target; default is the 4x core goal.",
    )
    parser.add_argument(
        "--native-indexer-doc-cap",
        type=int,
        default=0,
        help=(
            "Opt-in per-offline-document C4 Indexer cap. Zero disables the "
            "candidate; online suffix positions are always uncapped."
        ),
    )
    parser.add_argument("--native-indexer-documents", type=int, default=8)
    parser.add_argument(
        "--geometry-template-cache",
        action="store_true",
        help=(
            "Preload immutable fixed-corpus restore geometry after the first "
            "exact request; dynamic slots, leases and TP commits are rebuilt."
        ),
    )
    parser.add_argument(
        "--restore-pipeline-group-layers",
        type=int,
        default=0,
        help=(
            "Qualification-only layer-group restore pipeline; 0 keeps the "
            "certified monolithic three-family launch."
        ),
    )
    parser.add_argument(
        "--merged-prefill-tokens",
        type=int,
        default=0,
        help=(
            "Restore-only physical prefill group; 0 keeps 8K microforwards, "
            "32768 merges four certified segments, and 65536 merges eight."
        ),
    )
    _add_performance_qualification_args(parser)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--min-token-agreement", type=float, default=0.90)
    parser.add_argument(
        "--min-gpu-free-mib",
        type=int,
        default=PRO0813_MIN_FREE_BEFORE_LAUNCH_MIB,
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=(
            "SGLang static-memory fraction. Default is the target-specific "
            "B300 scale-policy value; strict runs reject a different value."
        ),
    )
    args = parser.parse_args()

    if args.contract_only:
        contract = _validate_pro0813_contract(Path(args.contract_config))
        contract["target_zoff_bytes_per_rank"] = {
            str(tokens): _pro0813_zoff_bytes_per_rank(tokens)
            for tokens in sorted(REDKNOT_TARGET_PROFILES)
        }
        print(json.dumps(contract, sort_keys=True))
        return
    if not args.out or not args.log:
        parser.error("--out and --log are required for benchmark execution")
    _validate_performance_qualification_args(args, parser)
    _validate_qualification_profile_target_policy(args)
    _validate_pro0813_contract(MODEL / "config.json")

    expected_mem_fraction = _PRO0813_SCALE.PRO0813_MEM_FRACTION_STATIC.get(
        args.target_tokens
    )
    if expected_mem_fraction is None:
        raise ValueError(
            f"unsupported Pro-0813 memory target: {args.target_tokens}"
        )
    if args.mem_fraction_static is None:
        args.mem_fraction_static = expected_mem_fraction
    elif args.strict_performance and not math.isclose(
        args.mem_fraction_static, expected_mem_fraction, abs_tol=1e-12
    ):
        raise ValueError(
            "strict Pro-0813 reproduction requires the target-specific B300 "
            f"mem fraction {expected_mem_fraction:.2f}, got "
            f"{args.mem_fraction_static:.6g}"
        )

    if args.qualification_profile and args.mode != "redknot":
        raise ValueError("--qualification-profile is valid only in RedKnot mode")
    if args.qualification_profile and args.data_manifest:
        raise ValueError(
            "--qualification-profile already freezes its data manifest; "
            "--data-manifest must be omitted"
        )
    if args.qualification_profile and args.prompt_manifest_out:
        raise ValueError(
            "--qualification-profile already freezes its prompt manifest; "
            "--prompt-manifest-out must be omitted"
        )

    if not 0.0 < args.token_sparse_mass <= 1.0:
        raise ValueError("--token-sparse-mass must be in (0, 1]")
    if not math.isfinite(args.ttft_target_speedup) or args.ttft_target_speedup <= 1.0:
        raise ValueError("--ttft-target-speedup must be finite and > 1")
    if args.native_indexer_doc_cap < 0:
        raise ValueError("--native-indexer-doc-cap cannot be negative")
    if args.native_indexer_doc_cap and args.mode != "redknot":
        raise ValueError("native Indexer document bucketing is RedKnot-only")
    if args.native_indexer_documents <= 0:
        raise ValueError("--native-indexer-documents must be positive")
    if not 0.0 < args.token_sparse_deep_mass <= 1.0:
        raise ValueError("--token-sparse-deep-mass must be in (0, 1]")
    if not 0.0 <= args.token_sparse_min_full_ratio <= 1.0:
        raise ValueError("--token-sparse-min-full-ratio must be in [0, 1]")
    if not 0.0 <= args.token_sparse_max_full_ratio <= 1.0:
        raise ValueError("--token-sparse-max-full-ratio must be in [0, 1]")
    if args.token_sparse_min_full_ratio > args.token_sparse_max_full_ratio:
        raise ValueError("token-sparse min full ratio exceeds max full ratio")
    if (
        args.token_sparse_block_tokens < 0
        or (
            args.token_sparse_block_tokens
            and args.token_sparse_block_tokens % 128 != 0
        )
    ):
        raise ValueError(
            "--token-sparse-block-tokens must be 0 or a positive multiple of 128"
        )
    if args.token_sparse_freeze_blocks and not args.token_sparse_block_tokens:
        raise ValueError(
            "--token-sparse-freeze-blocks requires --token-sparse-block-tokens"
        )
    if args.token_sparse_ffn and args.mode != "redknot":
        raise ValueError("token-sparse FFN is valid only for RedKnot qualification")
    if not 0 <= args.restore_pipeline_group_layers <= 55:
        raise ValueError("--restore-pipeline-group-layers must be in [0, 55]")
    if args.restore_pipeline_group_layers and args.mode != "redknot":
        raise ValueError("restore pipeline is valid only for RedKnot qualification")
    if args.row_sparse_online and args.mode != "redknot":
        raise ValueError("row-sparse qualification is valid only for RedKnot")
    combined_headsplit_row_sparse = bool(
        args.combined_headsplit_row_sparse
        or getattr(
            args,
            "combined_headsplit_row_sparse_diagnostic_zoff_only",
            False,
        )
    )
    if combined_headsplit_row_sparse and args.mode != "redknot":
        raise ValueError("combined headsplit/row-sparse is valid only for RedKnot")
    if (
        args.combined_headsplit_row_sparse
        and getattr(
            args,
            "combined_headsplit_row_sparse_diagnostic_zoff_only",
            False,
        )
    ):
        raise ValueError(
            "formal full combined and diagnostic zoff-only combined modes are "
            "mutually exclusive"
        )
    if (
        getattr(
            args,
            "combined_headsplit_row_sparse_diagnostic_zoff_only",
            False,
        )
        and args.strict_performance
    ):
        raise ValueError(
            "--combined-headsplit-row-sparse-diagnostic-zoff-only requires "
            "--diagnostic-performance"
        )
    _validate_formal_execution_profile_args(args)
    if args.row_sparse_online and combined_headsplit_row_sparse:
        raise ValueError(
            "standalone row-sparse and combined headsplit/row-sparse are "
            "mutually exclusive"
        )
    if args.generalized_adaptive_controller:
        if not combined_headsplit_row_sparse:
            raise ValueError(
                "--generalized-adaptive-controller requires "
                "--combined-headsplit-row-sparse"
            )
        if int(REDKNOT_TARGET_PROFILES[args.target_tokens]["chunk_tokens"]) < 32768:
            raise ValueError(
                "generalized adaptive controller v1 requires 32K documents"
            )
    generalized_ratios = (
        args.generalized_strong_active_ratio,
        args.generalized_medium_active_ratio,
        args.generalized_diffuse_active_ratio,
    )
    if any(
        not math.isfinite(ratio) or ratio <= 0.0 or ratio >= 0.85
        for ratio in generalized_ratios
    ):
        raise ValueError("generalized active ratios must be finite and in (0, 0.85)")
    if tuple(sorted(generalized_ratios)) != generalized_ratios:
        raise ValueError(
            "generalized active ratios must satisfy strong <= medium <= diffuse"
        )
    if not 0.0 < args.row_sparse_active_ratio < 0.85:
        raise ValueError("--row-sparse-active-ratio must be in (0, 0.85)")
    if (
        args.query_protection_tokens < 512
        or args.query_protection_tokens % 512 != 0
        or args.query_protection_tokens
        > int(REDKNOT_TARGET_PROFILES[args.target_tokens]["chunk_tokens"])
    ):
        raise ValueError(
            "--query-protection-tokens must be a 512-token multiple within "
            "one document"
        )
    if not 0.0 < args.row_sparse_segment_cap_ratio <= 1.0:
        raise ValueError(
            "--row-sparse-segment-cap-ratio must be in (0, 1]"
        )
    if (
        args.row_sparse_checkpoint_stride < 512
        or args.row_sparse_checkpoint_stride % 512 != 0
    ):
        raise ValueError(
            "--row-sparse-checkpoint-stride must be a multiple of 512"
        )
    checkpoint_profile = REDKNOT_TARGET_PROFILES.get(args.target_tokens)
    expected_checkpoint_capacity = (
        int(checkpoint_profile["row_sparse_checkpoint_max_islands"])
        if checkpoint_profile is not None
        else 64
    )
    if args.row_sparse_checkpoint_max_islands is None:
        args.row_sparse_checkpoint_max_islands = expected_checkpoint_capacity
    if not (
        1
        <= args.row_sparse_checkpoint_max_islands
        <= _PRO0813_SCALE.PRO0813_CHECKPOINT_DESCRIPTOR_LIMIT
    ):
        raise ValueError(
            "--row-sparse-checkpoint-max-islands must be in [1, 256]"
        )
    capacity_ratio = (
        max(generalized_ratios)
        if args.generalized_adaptive_controller
        else args.row_sparse_active_ratio
    )
    required_checkpoint_capacity = (
        _PRO0813_SCALE.pro0813_required_checkpoint_islands(
            args.target_tokens,
            capacity_ratio,
            checkpoint_stride=args.row_sparse_checkpoint_stride,
        )
        if checkpoint_profile is not None
        else 1
    )
    if args.row_sparse_checkpoint_max_islands < required_checkpoint_capacity:
        raise ValueError(
            "checkpoint descriptor capacity cannot realize the requested Pro "
            "row budget: "
            f"target={args.target_tokens} ratio={capacity_ratio:.6g} "
            f"required={required_checkpoint_capacity} "
            f"configured={args.row_sparse_checkpoint_max_islands}"
        )
    if (
        args.strict_performance
        and combined_headsplit_row_sparse
        and args.row_sparse_checkpoint_max_islands
        != expected_checkpoint_capacity
    ):
        raise ValueError(
            "strict Pro-0813 reproduction requires target checkpoint capacity "
            f"{expected_checkpoint_capacity}, got "
            f"{args.row_sparse_checkpoint_max_islands}"
        )

    if args.prefix_materialization and args.first_document_prefix:
        raise ValueError(
            "--prefix-materialization and --first-document-prefix are mutually exclusive"
        )

    target_profile = (
        _resolve_redknot_target_profile(args) if args.mode == "redknot" else None
    )
    result, server_log, prompt_manifest, rank_logs = _prepare_output_paths(
        args, target_profile=target_profile
    )
    if args.mode == "baseline":
        _configure_native(args, result)
    else:
        _configure_redknot(
            args,
            result,
            server_log,
            prompt_manifest,
            rank_logs,
            target_profile,
        )
    runpy.run_path(str(DRIVER), run_name="__main__")


if __name__ == "__main__":
    main()
