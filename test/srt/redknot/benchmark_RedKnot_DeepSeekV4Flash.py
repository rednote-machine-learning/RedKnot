#!/usr/bin/env python3
"""One-click, fail-closed DeepSeek-V4-Flash + RedKnot release benchmark.

The default run replays four immutable LongBench-derived RAG suites at 64K,
128K, 256K and 440K.  Each suite contains ten short-answer cases and five supplemental
long-answer cases.  Every case compares a full online recompute with the
validated RedKnot closure: independent position-zero MLA artifacts, online
RoPE relocation and head merge, Indexer-guided row sparsity, and
assignment-sparse adaptive MoE Top-K.  GPU-holder lifecycle is delegated to
the audited supervisor and is restored after every run, including failures
and signals.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.compute_ledger import estimate_prefill_saving


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
UTILS = HERE / "utils"
DATA_DIR = HERE / "datasets/LongBench/data"
PROFILE_ROOT = HERE / "datasets" / "LongBench" / "cohorts"
SHOWCASE_ROOT = HERE / "datasets" / "LongBench" / "showcase"
SUITE_ROOT = HERE / "datasets" / "LongBench" / "suites"
HEAD_CONFIG = HERE / "head_class/deepseek_v4_flash_0731_redknot.json"
SPARSE_CONFIG = HERE / "sparse_ffn_params/deepseek_v4_flash_0731.json"
DATA_PROVENANCE = HERE / "datasets/LongBench/PROVENANCE.json"
SUPERVISOR = UTILS / "run_combined_supervisor.sh"
HOLDER = UTILS / "gpu_hold.py"
INTERNAL_BENCHMARK = UTILS / "benchmark_dsv4_redknot_http.py"
CORE_BENCHMARK = UTILS / "benchmark_RedKnot_DeepSeekV4_Flash_RAG.py"
PROFILE_BUILDER = UTILS / "prepare_multidataset_512k_manifests.py"

DEFAULT_MODEL_PATH = Path(
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/"
    "DeepSeek-V4-Flash-0731"
)
DEFAULT_MODEL_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
DATASETS = (
    "hotpotqa",
    "2wikimqa",
    "musique",
    "multifieldqa_en",
    "triviaqa",
)
DATASET_SHA256 = {
    "hotpotqa": "a0005ab2a1bc2ac3a70352dccbf96cccc4e0aac6bb677f6a55180fa51b92ef6f",
    "2wikimqa": "dda279cf93a99e1e5bfa3291fb199fd55978d10a1feb31822953cf77a1742e37",
    "musique": "4ac69b91281c4ec6b21316cb7282e83fb6b4dda04fc68480bb8d8ed1e19ff7bd",
    "multifieldqa_en": "0aac182fd317dcf6d74f8e1e0f3e61029407435346c2e0b3ff9fb45ae49c5c3f",
    "triviaqa": "ed2529de2e10b12c00f49981870c23c7c51667069cdd8d0740e1423ff337d7fa",
}
# The base table is the existing H200 release configuration.  Hardware-specific
# profiles are overlays so adding B300 support cannot silently rewrite or remove
# the H200 reproduction contract.
LENGTHS = {
    "64K": {
        "target_tokens": 65536,
        "chunk_tokens": 16384,
        "num_chunks": 4,
        "merged_prefill": 49152,
        "mem_fraction": 0.45,
        "swa_full_ratio": 0.10,
        "cublas_woa_fastpath": 1,
        "query_protection_tokens": 16384,
    },
    "128K": {
        "target_tokens": 131072,
        "chunk_tokens": 32768,
        "num_chunks": 4,
        "merged_prefill": 98304,
        "mem_fraction": 0.40,
        "swa_full_ratio": 0.10,
        "cublas_woa_fastpath": 1,
        "query_protection_tokens": 32768,
    },
    "256K": {
        "target_tokens": 262144,
        "chunk_tokens": 32768,
        "num_chunks": 8,
        "merged_prefill": 229376,
        # Preserve the certified H200 release contract.  B300's smaller
        # physical merge unit is applied only through its hardware overlay.
        "mem_fraction": 0.45,
        "swa_full_ratio": 0.50,
        "cublas_woa_fastpath": 1,
        "query_protection_tokens": 32768,
    },
    "440K": {
        "target_tokens": 450560,
        "chunk_tokens": 56320,
        "num_chunks": 8,
        "merged_prefill": 0,
        "mem_fraction": 0.29,
        "swa_full_ratio": 0.25,
        "cublas_woa_fastpath": 0,
        "query_protection_tokens": 32768,
    },
}

B300_LENGTH_OVERRIDES = {
    "256K": {
        # Keep each physical forward at the verified 2x32K B300 geometry while
        # leaving the H200 229376-token release configuration untouched.
        "merged_prefill": 65536,
    },
}


def _resolve_hardware_profile(requested: str) -> str:
    """Return h200 or b300; auto deliberately falls back to the H200 contract."""
    profile = requested.strip().lower()
    if profile in {"h200", "b300"}:
        return profile
    if profile != "auto":
        raise ValueError(
            f"unsupported hardware profile {requested!r}; expected auto, h200 or b300"
        )
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "h200"
    names = [line.strip().upper() for line in completed.stdout.splitlines() if line.strip()]
    return "b300" if names and all("B300" in name for name in names) else "h200"


def _length_spec(length: str, hardware_profile: str) -> dict[str, Any]:
    spec = dict(LENGTHS[length])
    if hardware_profile == "b300":
        spec.update(B300_LENGTH_OVERRIDES.get(length, {}))
    return spec

SUITE_CONTRACTS = {
    "64K": {
        "format": "redknot_deepseek_v4_flash_64k_case_v1",
        "name": "redknot_deepseek_v4_flash_64k_15case_v1",
        "path": SUITE_ROOT / "release_64k_15case.jsonl",
    },
    "128K": {
        "format": "redknot_deepseek_v4_flash_128k_case_v1",
        "name": "redknot_deepseek_v4_flash_128k_15case_v1",
        "path": SUITE_ROOT / "release_128k_15case.jsonl",
    },
    "256K": {
        "format": "redknot_deepseek_v4_flash_256k_case_v1",
        "name": "redknot_deepseek_v4_flash_256k_15case_v1",
        "path": SUITE_ROOT / "release_256k_15case.jsonl",
    },
    "440K": {
        "format": "redknot_deepseek_v4_flash_440k_case_v1",
        "name": "redknot_deepseek_v4_flash_440k_15case_v1",
        "path": SUITE_ROOT / "release_440k_15case.jsonl",
    },
}
DEFAULT_SUITE = Path(SUITE_CONTRACTS["256K"]["path"])
DEFAULT_SUITES = tuple(
    Path(SUITE_CONTRACTS[length]["path"])
    for length in ("64K", "128K", "256K", "440K")
)
RELEASE_SUITES_MANIFEST = SUITE_ROOT / "RELEASE_SUITES.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_default_release_manifest() -> dict[str, Any]:
    if not RELEASE_SUITES_MANIFEST.is_file():
        raise FileNotFoundError(
            f"release-suite manifest is absent: {RELEASE_SUITES_MANIFEST}"
        )
    manifest = json.loads(RELEASE_SUITES_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("format") != "redknot_deepseek_v4_flash_release_suites_v1":
        raise ValueError("release-suite manifest format is invalid")
    ordered = tuple(manifest.get("ordered_lengths", ()))
    if ordered != ("64K", "128K", "256K", "440K"):
        raise ValueError(f"release-suite order is invalid: {ordered}")
    execution = manifest.get("execution", {})
    expected_execution = {
        "baseline_label": "Recomputed",
        "quality_repeats_per_case": 1,
        "ttft_measured_pairs_per_case": 10,
        "ttft_protocol": "hot_state_client_stream_first_output_token_p50_p95",
        "ttft_warmup_pairs_per_case": 3,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ValueError(
                f"release execution contract mismatch for {key}: "
                f"{execution.get(key)!r} != {expected!r}"
            )
    suites = manifest.get("suites", {})
    for length, path in zip(ordered, DEFAULT_SUITES, strict=True):
        record = suites.get(length, {})
        if record.get("path") != path.name:
            raise ValueError(f"release-suite path mismatch for {length}")
        if not path.is_file():
            raise FileNotFoundError(f"default release suite is absent: {path}")
        actual = _sha256(path)
        if record.get("sha256") != actual:
            raise ValueError(
                f"release-suite SHA256 mismatch for {length}: "
                f"manifest={record.get('sha256')!r} actual={actual}"
            )
    return manifest


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _load_suite_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load an ordered, frozen mixed short/long-output release suite.

    The JSONL is a case list, while execution is grouped by shared offline
    prefix/profile.  Every profile case must appear exactly once, so a group
    cannot silently execute extra, unreported questions.
    """

    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"suite JSONL is absent: {source}")
    cases: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                case = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"suite JSONL line {line_number} is invalid: {error}"
                ) from error
            if not isinstance(case, dict):
                raise ValueError(f"suite JSONL line {line_number} is not an object")
            cases.append(case)
    if len(cases) != 15:
        raise ValueError(f"the release suite must contain exactly 15 cases, got {len(cases)}")
    suite_lengths = {str(case.get("length", "")) for case in cases}
    if len(suite_lengths) != 1:
        raise ValueError(f"the release suite must contain one length: {suite_lengths}")
    suite_length = next(iter(suite_lengths))
    contract = SUITE_CONTRACTS.get(suite_length)
    if contract is None:
        raise ValueError(f"unsupported release-suite length: {suite_length!r}")
    required = {
        "format",
        "suite",
        "suite_case_index",
        "case_id",
        "group_id",
        "length",
        "dataset",
        "query_index",
        "query_row_id",
        "question",
        "answers",
        "answer_style",
        "target_output_tokens",
        "full_input_ids_sha256",
        "profile",
        "selection_origin",
        "eligible_for_accuracy_aggregate",
    }
    seen_case_ids: set[str] = set()
    group_order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, case in enumerate(cases):
        missing = required - set(case)
        if missing:
            raise ValueError(f"suite case {index} lacks fields: {sorted(missing)}")
        if (
            case["format"] != contract["format"]
            or case["suite"] != contract["name"]
            or case["suite_case_index"] != index
            or case["length"] != suite_length
            or case["dataset"] not in DATASETS
            or type(case["query_index"]) is not int
            or type(case["query_row_id"]) is not int
            or not isinstance(case["question"], str)
            or not case["question"].strip()
            or not isinstance(case["answers"], list)
            or not case["answers"]
            or not all(isinstance(answer, str) for answer in case["answers"])
            or case["target_output_tokens"] not in (0, 30, 50)
            or case["answer_style"]
            != (
                "direct_answer_plus_document_evidence_v1"
                if case["target_output_tokens"]
                else "shortest_exact_span_v1"
            )
            or not isinstance(case["full_input_ids_sha256"], str)
            or not case["full_input_ids_sha256"].startswith("sha256:")
            or type(case["eligible_for_accuracy_aggregate"]) is not bool
            or case["eligible_for_accuracy_aggregate"]
            is bool(case["target_output_tokens"])
        ):
            raise ValueError(f"suite case {index} violates the case contract")
        case_id = case["case_id"]
        group_id = case["group_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_case_ids:
            raise ValueError(f"suite case {index} has a duplicate/invalid case_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"suite case {index} has an invalid group_id")
        seen_case_ids.add(case_id)
        if group_id not in grouped:
            grouped[group_id] = []
            group_order.append(group_id)
        grouped[group_id].append(case)

    groups: list[dict[str, Any]] = []
    for group_id in group_order:
        members = grouped[group_id]
        profile_values = {str(member["profile"]) for member in members}
        dataset_values = {str(member["dataset"]) for member in members}
        output_values = {int(member["target_output_tokens"]) for member in members}
        if len(profile_values) != 1 or len(dataset_values) != 1 or len(output_values) != 1:
            raise ValueError(f"suite group {group_id!r} mixes profile contracts")
        query_indices = [int(member["query_index"]) for member in members]
        if query_indices != list(range(len(members))):
            raise ValueError(
                f"suite group {group_id!r} must cover profile query indices 0..N-1"
            )
        profile_path = (source.parent / next(iter(profile_values))).resolve()
        profile = _validate_profile(
            profile_path,
            suite_length,
            next(iter(dataset_values)),
            len(members),
            next(iter(output_values)),
        )
        profile_cases = profile.get("cases", [])
        dataset_rows = [
            json.loads(raw)
            for raw in (DATA_DIR / f"{next(iter(dataset_values))}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for member, frozen in zip(members, profile_cases, strict=True):
            row = dataset_rows[int(member["query_row_id"])]
            if (
                member["query_row_id"] != frozen.get("query_row_id")
                or member["full_input_ids_sha256"]
                != frozen.get("full_input_ids_sha256")
                or member["question"] != row.get("input")
                or member["answers"]
                != [str(answer) for answer in row.get("answers", [])]
            ):
                raise ValueError(
                    f"suite case {member['case_id']!r} differs from its frozen profile"
                )
        groups.append(
            {
                "group_id": group_id,
                "length": suite_length,
                "dataset": next(iter(dataset_values)),
                "long_output_tokens": next(iter(output_values)),
                "profile_path": profile_path,
                "profile": profile,
                "cases": members,
            }
        )
    if sum(len(group["cases"]) for group in groups) != 15:
        raise AssertionError("suite grouping lost cases")
    return cases, groups


def _csv(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items or len(set(items)) != len(items):
        raise ValueError(f"{label} must be a non-empty unique CSV")
    unknown = sorted(set(items) - set(allowed))
    if unknown:
        raise ValueError(f"unsupported {label}: {unknown}")
    return items


def _resolve_python(explicit: str) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_python = os.environ.get("REDKNOT_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    candidates.extend(
        (
            REPO / ".venv_tf5/bin/python",
            Path(sys.executable),
        )
    )
    for candidate in candidates:
        expanded = candidate.expanduser()
        executable = (
            expanded
            if expanded.is_absolute()
            else (Path.cwd() / expanded).absolute()
        )
        # Do not resolve a venv's bin/python symlink: executing the symlink is
        # what activates its pyvenv.cfg and local site-packages.
        if executable.is_file() and os.access(executable, os.X_OK):
            return executable
    raise FileNotFoundError("no executable Python runtime was found")


def _model_complete(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
        )
    )


def _ensure_model(path: Path, repo_id: str, download: bool) -> Path:
    path = path.expanduser().resolve()
    if _model_complete(path):
        return path
    if not download:
        raise FileNotFoundError(
            f"model is incomplete at {path}; pass --download-model or set "
            "--model-path to a complete local checkpoint"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for the model fallback"
        ) from exc
    path.mkdir(parents=True, exist_ok=True)
    print(
        f"[prepare] local checkpoint absent; downloading {repo_id} to {path}",
        flush=True,
    )
    snapshot_download(repo_id=repo_id, local_dir=str(path))
    if not _model_complete(path):
        raise RuntimeError(f"downloaded checkpoint is incomplete: {path}")
    return path


def _validate_checkpoint_reader(python: Path, model: Path) -> None:
    """Fail before GPU release if the runtime cannot read every weight shard."""

    probe = r"""
import json
import os
import sys
import safetensors
import flash_mla
import deep_gemm
import sgl_kernel
from safetensors import safe_open

root = sys.argv[1]
for api in ("get_mla_metadata", "flash_mla_sparse_fwd", "flash_mla_with_kvcache"):
    if not callable(getattr(flash_mla, api, None)):
        raise RuntimeError(f"flash_mla is missing required API: {api}")
index_path = os.path.join(root, "model.safetensors.index.json")
with open(index_path, "r", encoding="utf-8") as handle:
    index = json.load(handle)
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise RuntimeError("checkpoint index contains no weight shards")
for shard in shards:
    path = os.path.join(root, shard)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not handle.keys():
            raise RuntimeError(f"empty safetensors shard: {path}")
print(json.dumps({
    "safetensors": safetensors.__version__,
    "flash_mla": getattr(flash_mla, "__version__", "unknown"),
    "sgl_kernel": getattr(sgl_kernel, "__version__", "unknown"),
    "deep_gemm": bool(deep_gemm),
    "shards": len(shards),
}))
"""
    completed = subprocess.run(
        [str(python), "-c", probe, str(model)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "the selected Python runtime cannot read the complete checkpoint; "
            "install safetensors>=0.8.0 in the release environment. "
            f"python={python} detail={detail}"
        )
    print(
        f"[ready] checkpoint reader validated: python={python} "
        f"{completed.stdout.strip()}",
        flush=True,
    )


def _ensure_datasets(download: bool) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, expected in DATASET_SHA256.items():
        target = DATA_DIR / f"{dataset}.jsonl"
        if target.is_file() and _sha256(target) == expected:
            continue
        if target.exists():
            raise RuntimeError(
                f"dataset digest mismatch for {target}; refusing replacement"
            )
        if not download:
            raise FileNotFoundError(f"packaged dataset is absent: {target}")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for the dataset fallback"
            ) from exc
        downloaded = Path(
            hf_hub_download(
                repo_id="THUDM/LongBench",
                repo_type="dataset",
                filename=f"data/{dataset}.jsonl",
            )
        )
        if _sha256(downloaded) != expected:
            raise RuntimeError(
                f"upstream bytes for {dataset} do not match the published digest"
            )
        shutil.copy2(downloaded, target)


def _validate_release_configs(model: Path) -> dict[str, Any]:
    required = (
        INTERNAL_BENCHMARK,
        CORE_BENCHMARK,
        SUPERVISOR,
        HOLDER,
        PROFILE_BUILDER,
        HEAD_CONFIG,
        SPARSE_CONFIG,
        DATA_PROVENANCE,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"release artifact is absent: {path}")
    model_cfg = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if (
        model_cfg.get("model_type") != "deepseek_v4"
        or model_cfg.get("num_hidden_layers") != 43
        or model_cfg.get("num_attention_heads") != 64
        or model_cfg.get("n_routed_experts") != 256
        or model_cfg.get("num_experts_per_tok") != 6
    ):
        raise RuntimeError("checkpoint is not the certified V4-Flash-0731 topology")
    head = json.loads(HEAD_CONFIG.read_text(encoding="utf-8"))
    if (
        head.get("format") != "redknot_deepseek_v4_mla_head_config_v2"
        or head.get("num_layers") != 43
        or head.get("num_attention_heads") != 64
        or head.get("dense_prefix_layers") != 3
        or head.get("dense_suffix_layers") != 3
    ):
        raise RuntimeError("head_class topology is invalid")
    classes = head.get("mla_head_classification")
    if not isinstance(classes, list) or len(classes) != 43:
        raise RuntimeError("head_class matrix has invalid layer count")
    for layer, row in enumerate(classes):
        expected = (
            ["dense"] * 64
            if layer < 3 or layer >= 40
            else ["global" if h % 8 == 0 else "local" for h in range(64)]
        )
        if row != expected:
            raise RuntimeError(f"head_class matrix differs at layer {layer}")
    sparse = json.loads(SPARSE_CONFIG.read_text(encoding="utf-8"))
    adaptive = sparse.get("adaptive_assignment_topk", {})
    controller = sparse.get("generalized_row_controller", {})
    if (
        adaptive.get("native_top_k") != 6
        or adaptive.get("cumulative_mass") != 0.5
        or adaptive.get("allowed_buckets") != [3, 4, 5, 6]
        or adaptive.get("dense_prefix_layers") != 3
        or adaptive.get("dense_suffix_layers") != 3
        or controller.get("strong_active_ratio") != 0.10
        or controller.get("medium_active_ratio") != 0.20
        or controller.get("diffuse_active_ratio") != 0.25
    ):
        raise RuntimeError("sparse_ffn_params differ from the published policy")
    return sparse


def _profile_path(
    length: str,
    dataset: str,
    num_queries: int,
    long_output_tokens: int = 0,
) -> Path:
    suffix = f"_long{long_output_tokens}" if long_output_tokens else ""
    return (
        PROFILE_ROOT
        / f"{length.lower()}_{num_queries}q{suffix}"
        / dataset
        / "profile.json"
    )


def _build_profile(
    python: Path,
    model: Path,
    length: str,
    dataset: str,
    num_queries: int,
    long_output_tokens: int = 0,
) -> Path:
    destination = _profile_path(
        length, dataset, num_queries, long_output_tokens
    ).parent
    profile = destination / "profile.json"
    if profile.is_file():
        return profile
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(PROFILE_BUILDER),
        "--core",
        str(CORE_BENCHMARK),
        "--dataset",
        dataset,
        "--data-dir",
        str(DATA_DIR),
        "--model",
        str(model),
        "--num-queries",
        str(num_queries),
        "--cohort-index",
        str(int(length.removesuffix("K")) * 100 + DATASETS.index(dataset)),
        "--chunk-tokens",
        str(LENGTHS[length]["chunk_tokens"]),
        "--num-chunks",
        str(LENGTHS[length]["num_chunks"]),
        "--long-output-tokens",
        str(long_output_tokens),
        "--data-out",
        str(destination / "data_selection.json"),
        "--prompt-out",
        str(destination / "prompt_manifest.json"),
        "--profile-out",
        str(profile),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "python") + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    subprocess.run(command, cwd=REPO, env=env, check=True)
    return profile


def _validate_profile(
    path: Path,
    length: str,
    dataset: str,
    num_queries: int,
    long_output_tokens: int = 0,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    spec = LENGTHS[length]
    if (
        value.get("dataset") != dataset
        or value.get("num_chunks") != spec["num_chunks"]
        or value.get("num_queries") != num_queries
        or value.get("chunk_tokens") != spec["chunk_tokens"]
        or value.get("query_start") != spec["target_tokens"]
        or value.get("output_blind") is not True
        or int(value.get("long_output_target_tokens", 0))
        != long_output_tokens
    ):
        raise RuntimeError(f"profile contract mismatch: {path}")
    data_path = path.parent / "data_selection.json"
    prompt_path = path.parent / "prompt_manifest.json"
    for member in (data_path, prompt_path):
        if not member.is_file():
            raise FileNotFoundError(f"profile member is absent: {member}")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    if (
        data.get("selection_sha256") != value.get("data_selection_sha256")
        or data.get("dataset", {}).get("name") != dataset
        or data.get("dataset", {}).get("sha256") != DATASET_SHA256[dataset]
        or prompt.get("data_selection_sha256") != value.get("data_selection_sha256")
        or prompt.get("prompt_manifest_sha256")
        != value.get("prompt_manifest_sha256")
        or prompt.get("dataset", {}).get("name") != dataset
        or prompt.get("output_blind") is not True
        or prompt.get("prompt", {}).get("offline_chunk_hashes")
        != value.get("offline_chunk_hashes")
    ):
        raise RuntimeError(f"profile member digest/identity mismatch: {path.parent}")
    protocol = prompt.get("protocol", {})
    if int(protocol.get("target_output_tokens", 0)) != long_output_tokens:
        raise RuntimeError(f"prompt output-length contract mismatch: {prompt_path}")
    expected_style = (
        "direct_answer_plus_document_evidence_v1"
        if long_output_tokens
        else "shortest_exact_span_v1"
    )
    if value.get("answer_style", expected_style) != expected_style:
        raise RuntimeError(f"profile answer style mismatch: {path}")
    geometry = prompt.get("geometry", {})
    if (
        geometry.get("num_chunks") != spec["num_chunks"]
        or geometry.get("chunk_tokens") != spec["chunk_tokens"]
        or geometry.get("query_start") != spec["target_tokens"]
    ):
        raise RuntimeError(f"prompt geometry mismatch: {prompt_path}")
    prompt_cases = prompt.get("cases")
    if prompt_cases is not None and prompt_cases != value.get("cases"):
        raise RuntimeError(f"profile cases differ from prompt manifest: {path}")
    if len(value.get("cases", [])) != num_queries:
        raise RuntimeError(f"profile case count mismatch: {path}")
    return value


def _gpu_pids() -> tuple[int, ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return tuple(sorted({int(x) for x in result.stdout.split() if x.isdigit()}))


def _holder_leaders() -> tuple[int, ...]:
    leaders = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        try:
            command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "gpu_hold.py " not in command:
                continue
            if pid == os.getpgid(pid) == os.getsid(pid):
                leaders.append(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return tuple(sorted(leaders))


def _stop_process_group(pid: int) -> None:
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    raise RuntimeError(f"GPU holder process group did not exit: {pid}")


def _ensure_release_holder(holder_python: Path, log: Path) -> int:
    leaders = _holder_leaders()
    if len(leaders) > 1:
        raise RuntimeError(f"multiple GPU holder leaders found: {leaders}")
    if leaders:
        leader = leaders[0]
        cwd = Path(f"/proc/{leader}/cwd").resolve()
        if cwd == UTILS:
            return leader
        active = _gpu_pids()
        foreign_list = []
        for pid in active:
            try:
                if os.getpgid(pid) != leader:
                    foreign_list.append(pid)
            except ProcessLookupError:
                continue
        foreign = tuple(foreign_list)
        if foreign:
            raise RuntimeError(
                "refusing to stop a holder while unrelated GPU processes exist; "
                f"PIDs={foreign}"
            )
        print(f"[gpu] replacing legacy holder pid={leader} cwd={cwd}", flush=True)
        _stop_process_group(leader)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and _gpu_pids():
            time.sleep(1)
        if _gpu_pids():
            raise RuntimeError("authenticated holder workers did not leave the GPUs")
    residual = _gpu_pids()
    if residual:
        raise RuntimeError(
            "refusing to stop unrecognized GPU processes; PIDs="
            + ",".join(map(str, residual))
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("ab", buffering=0)
    process = subprocess.Popen(
        [str(holder_python), "gpu_hold.py"],
        cwd=UTILS,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"GPU holder exited early: {process.returncode}")
        pids = _gpu_pids()
        if len(pids) == 8 and all(os.getpgid(pid) == process.pid for pid in pids):
            return process.pid
        time.sleep(1)
    raise RuntimeError("GPU holder did not claim all eight GPUs")


def _extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
    latency = result.get("latency", {})
    gate = result.get("performance_gate", {})
    runtime = result.get("runtime", {})
    measurement = result.get("performance_measurement", {})
    config = result.get("config", {})
    active_rows = int(runtime.get("active_rows") or 0)
    full_rows = int(runtime.get("full_rows") or 0)
    full_model_head_saving = runtime.get(
        "measured_full_model_mla_head_row_saving",
        gate.get("observed_full_model_mla_head_row_saving"),
    )
    compute_ledger: dict[str, Any] = {}
    if full_rows > 0 and full_model_head_saving is not None:
        ledger = estimate_prefill_saving(
            token_full_ratio=active_rows / full_rows,
            mla_head_row_saving=float(full_model_head_saving),
            total_tokens=int(config.get("full_input_tokens") or 262197),
            first_document_tokens=int(
                config.get("materialized_prefix_tokens") or 32768
            ),
        )
        compute_ledger = {
            "method": "conservative_deepseek_v4_major_kernel_arithmetic_v1",
            "token_full_ratio": ledger.token_full_ratio,
            "moe_arithmetic_saving": ledger.moe_arithmetic_saving,
            "mla_head_arithmetic_saving": ledger.mla_head_arithmetic_saving,
            "total_online_saving": ledger.total_online_saving,
            "full_input_saving_with_first_document_prefix": (
                ledger.full_input_saving_with_first_document_prefix
            ),
            "online_compute_ratio": ledger.online_compute_ratio,
            "excluded_from_credit": [
                "indexer",
                "compressor",
                "router",
                "normalization",
                "memory_traffic",
                "kernel_launch",
                "TP_communication",
            ],
        }
    return {
        "overall_pass": result.get("overall_pass"),
        # The core harness historically names the full-online baseline
        # "dense".  DeepSeek-V4-Flash is not a dense model, so the release
        # schema uses the semantically accurate Recomputed label while still
        # reading the backwards-compatible core field.
        "recomputed_ttft_p50_s": latency.get("dense_p50"),
        "redknot_ttft_p50_s": latency.get("reuse_p50"),
        "ttft_speedup": latency.get("speedup"),
        "model_internal_ttft_speedup": latency.get("model_internal", {}).get(
            "speedup"
        ),
        "online_row_saving": runtime.get("online_row_saving"),
        "scoped_head_row_saving": runtime.get(
            "measured_scoped_mla_head_row_saving",
            runtime.get("mla_head_row_saving"),
        ),
        "full_model_head_row_saving": full_model_head_saving,
        "compute_saving": compute_ledger,
        "sequential_service_rate": result.get("sequential_service_rate"),
        "qps": measurement.get("qps"),
    }


def _extract_output_pairs(result: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for query_position, query in enumerate(result.get("queries", ())):
        repeats = query.get("repeats") or (query,)
        for repeat_position, repeat in enumerate(repeats):
            pairs.append(
                {
                    "query_index": query.get("query_index", query_position),
                    "repeat": repeat.get("repeat", repeat_position),
                    "question": str(query.get("question", "")),
                    "recomputed_output": str(repeat.get("dense_text", "")),
                    "redknot_output": str(repeat.get("reuse_text", "")),
                    "recomputed_output_tokens": repeat.get(
                        "dense_output_tokens",
                        query.get("dense_output_tokens"),
                    ),
                    "redknot_output_tokens": repeat.get(
                        "reuse_output_tokens",
                        query.get("reuse_output_tokens"),
                    ),
                }
            )
    return pairs


def _format_metric(value: Any, suffix: str = "") -> str:
    if value is None:
        return "not available"
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}{suffix}"
    return f"{value}{suffix}"


def _print_comparison(length: str, dataset: str, record: dict[str, Any]) -> None:
    metrics = record.get("metrics", {})
    print(
        f"\n===== {length} / {dataset}: Recomputed vs RedKnot =====",
        flush=True,
    )
    for pair in record.get("output_comparisons", ()):
        print(
            f"\n[query {pair['query_index']} repeat {pair['repeat']}] "
            f"{pair['question']}",
            flush=True,
        )
        print("\n[RECOMPUTED OUTPUT]", flush=True)
        print(
            f"({pair.get('recomputed_output_tokens', 'unknown')} generated tokens)\n"
            f"{pair['recomputed_output']}",
            flush=True,
        )
        print("\n[REDKNOT OUTPUT]", flush=True)
        print(
            f"({pair.get('redknot_output_tokens', 'unknown')} generated tokens)\n"
            f"{pair['redknot_output']}",
            flush=True,
        )
    print(
        "\n[TTFT / COMPUTE] "
        "recomputed="
        f"{_format_metric(metrics.get('recomputed_ttft_p50_s'), 's')} "
        f"redknot={_format_metric(metrics.get('redknot_ttft_p50_s'), 's')} "
        f"speedup={_format_metric(metrics.get('ttft_speedup'), 'x')} "
        "full_input_compute_saving="
        f"{_format_metric(metrics.get('compute_saving', {}).get('full_input_saving_with_first_document_prefix'))} "
        "full_model_head_row_saving="
        f"{_format_metric(metrics.get('full_model_head_row_saving'))}",
        flush=True,
    )


def _write_comparison_report(path: Path, release: dict[str, Any]) -> None:
    suite_mode = (
        release.get("run_mode")
        in {
            "release_64k_15case_suite",
            "release_128k_15case_suite",
            "release_256k_15case_suite",
            "release_440k_15case_suite",
        }
    )
    suite_length = str((release.get("lengths") or [""])[0])
    lines = [
        "# DeepSeek V4 Flash + RedKnot output comparison",
        "",
        "The report compares the complete generated text directly. Recomputed "
        "means the same DeepSeek-V4-Flash model performs a full online "
        "recomputation without RedKnot reuse.",
        "",
    ]
    for run in release.get("runs", ()):
        length = run.get("length", "")
        dataset = run.get("dataset", "")
        metrics = run.get("metrics", {})
        lines.extend(
            [
                f"## {length} / {dataset}",
                "",
                "- Recomputed TTFT p50: "
                f"{_format_metric(metrics.get('recomputed_ttft_p50_s'), ' s')}",
                f"- RedKnot TTFT p50: {_format_metric(metrics.get('redknot_ttft_p50_s'), ' s')}",
                f"- TTFT speedup: {_format_metric(metrics.get('ttft_speedup'), 'x')}",
                "- Full-input major-compute saving: "
                f"{_format_metric(metrics.get('compute_saving', {}).get('full_input_saving_with_first_document_prefix'))}",
                "- Full-model MLA head-row saving: "
                f"{_format_metric(metrics.get('full_model_head_row_saving'))}",
                "- Supplemental output target: "
                f"{run.get('long_output_target_tokens', 0)} generated tokens "
                "(0 means the unchanged shortest-span benchmark)",
                "",
            ]
        )
        if suite_mode:
            continue
        for pair in run.get("output_comparisons", ()):
            question = html.escape(pair.get("question", ""))
            recomputed = html.escape(pair.get("recomputed_output", ""))
            redknot = html.escape(pair.get("redknot_output", ""))
            lines.extend(
                [
                    f"### Query {pair.get('query_index')} / repeat {pair.get('repeat')}",
                    "",
                    f"Question: {question}",
                    "",
                    "| Recomputed output | RedKnot output |",
                    "|---|---|",
                    "| "
                    f"<pre>[{pair.get('recomputed_output_tokens', 'unknown')} tokens]\n"
                    f"{recomputed}</pre> | "
                    f"<pre>[{pair.get('redknot_output_tokens', 'unknown')} tokens]\n"
                    f"{redknot}</pre> |",
                    "",
                ]
            )
    if suite_mode:
        ordered_pairs = sorted(
            (
                (run, pair)
                for run in release.get("runs", ())
                for pair in run.get("output_comparisons", ())
            ),
            key=lambda item: int(item[1]["suite_case_index"]),
        )
        lines.extend(
            [
                f"# Ordered {suite_length} suite outputs",
                "",
                "The following fifteen comparisons are in the exact JSONL order: "
                "ten frozen short-answer cases, then five long-answer cases.",
                "",
            ]
        )
        for run, pair in ordered_pairs:
            question = html.escape(str(pair.get("question", "")))
            recomputed = html.escape(str(pair.get("recomputed_output", "")))
            redknot = html.escape(str(pair.get("redknot_output", "")))
            index = int(pair["suite_case_index"])
            lines.extend(
                [
                    f"## Case {index + 1:02d}: {pair.get('case_id', '')}",
                    "",
                    f"- Dataset: `{run.get('dataset', '')}`",
                    f"- Output target: `{run.get('long_output_target_tokens', 0)}` tokens",
                    f"- Selection origin: `{pair.get('selection_origin', '')}`",
                    "- Included in primary accuracy aggregate: "
                    f"`{str(bool(pair.get('eligible_for_accuracy_aggregate'))).lower()}`",
                    "",
                    f"Question: {question}",
                    "",
                    "| Recomputed output | RedKnot output |",
                    "|---|---|",
                    "| "
                    f"<pre>[{pair.get('recomputed_output_tokens', 'unknown')} tokens]\n"
                    f"{recomputed}</pre> | "
                    f"<pre>[{pair.get('redknot_output_tokens', 'unknown')} tokens]\n"
                    f"{redknot}</pre> |",
                    "",
                ]
            )
    showcase_documents: list[tuple[str, dict[str, Any]]] = []
    for label in (() if suite_mode else ("256K", "440K")):
        showcase_path = SHOWCASE_ROOT / f"long_output_{label.lower()}.json"
        if not showcase_path.is_file():
            continue
        document = json.loads(showcase_path.read_text(encoding="utf-8"))
        selected = document.get("selected")
        if (
            document.get("format")
            != "redknot_long_output_posthoc_showcase_v1"
            or document.get("selection_is_posthoc") is not True
            or document.get("eligible_for_accuracy_aggregate") is not False
            or not isinstance(selected, list)
            or len(selected) != 5
        ):
            raise RuntimeError(
                f"long-output showcase contract mismatch: {showcase_path}"
            )
        showcase_documents.append((label, document))
    if showcase_documents:
        lines.extend(
            [
                "# Long-output showcase (post-hoc, non-scoring)",
                "",
                "These examples were selected after observing the source pool. ",
                "They are reproducible presentation cases, are excluded from ",
                "all aggregate accuracy metrics, and do not replace any failed ",
                "standard or long-output sample.",
                "",
            ]
        )
        for label, document in showcase_documents:
            lines.extend(
                [
                    f"## {label}: five long-output examples",
                    "",
                    f"- Showcase SHA256: `{document.get('showcase_sha256', '')}`",
                    f"- Source candidates: {document.get('candidate_count', 0)}",
                    "",
                ]
            )
            for index, case in enumerate(document["selected"], start=1):
                question = html.escape(str(case.get("question", "")))
                recomputed = html.escape(
                    str(case.get("recomputed_output", case.get("dense_output", "")))
                )
                redknot = html.escape(str(case.get("redknot_output", "")))
                lines.extend(
                    [
                        f"### {label} long example {index}",
                        "",
                        f"- Dataset: `{case.get('dataset', '')}`",
                        f"- Query row: `{case.get('query_row_id')}`",
                        f"- Input SHA256: `{case.get('full_input_ids_sha256', '')}`",
                        "",
                        f"Question: {question}",
                        "",
                        "| Recomputed output | RedKnot output |",
                        "|---|---|",
                        "| "
                        "<pre>["
                        f"{case.get('recomputed_output_tokens', case.get('dense_output_tokens', 0))} tokens]\n"
                        f"{recomputed}</pre> | "
                        f"<pre>[{case.get('redknot_output_tokens', 0)} tokens]\n"
                        f"{redknot}</pre> |",
                        "",
                    ]
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--lengths", default="256K,440K")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument(
        "--download-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the official Hugging Face checkpoint when local files are absent.",
    )
    parser.add_argument("--python", default="")
    parser.add_argument("--holder-python", default="/root/miniconda3/bin/python")
    parser.add_argument(
        "--hardware-profile",
        choices=("auto", "h200", "b300"),
        default=os.environ.get("REDKNOT_HARDWARE_PROFILE", "auto").lower(),
        help=(
            "Hardware-tuned launch overlay. auto selects b300 only when every "
            "visible GPU is a B300; all other/unknown systems retain H200 settings."
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ttft-warmup", type=int, default=3)
    parser.add_argument("--ttft-iters", type=int, default=10)
    parser.add_argument(
        "--quality-repeats",
        type=int,
        default=1,
        help=(
            "Generate one complete Recomputed/RedKnot text pair per frozen "
            "release case by default."
        ),
    )
    parser.add_argument("--measure-qps", action="store_true")
    parser.add_argument("--qps-concurrencies", default="1")
    parser.add_argument("--qps-warmup-waves", type=int, default=1)
    parser.add_argument("--qps-waves", type=int, default=3)
    parser.add_argument(
        "--cases-per-dataset",
        type=int,
        default=10,
        help="Number of distinct, output-blind frozen inputs per dataset.",
    )
    parser.add_argument(
        "--long-output-tokens",
        type=int,
        choices=(0, 30, 50),
        default=0,
        help=(
            "0 preserves the published shortest-span test exactly. 30 or 50 "
            "selects a separate supplemental direct-answer-plus-evidence cohort."
        ),
    )
    parser.add_argument(
        "--suite-jsonl",
        default="",
        help=(
            "Run one ordered frozen mixed-output suite. Each 64K, 128K, 256K or 440K "
            "release suite contains ten short-answer cases followed by five "
            "long-answer cases."
        ),
    )
    parser.add_argument(
        "--v01-reference",
        action="store_true",
        help=(
            "Replay the original V0.1 MuSiQue 256K reference manifest and "
            "sampling configuration instead of the multi-dataset matrix."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted matrix in --output-dir. Existing result.json "
            "files are hash-verified and reported without rerunning their GPUs."
        ),
    )
    args = parser.parse_args()

    explicit_matrix_flags = {
        "--datasets",
        "--lengths",
        "--cases-per-dataset",
        "--long-output-tokens",
        "--v01-reference",
    }
    implicit_suite = not any(
        argument.split("=", 1)[0] in explicit_matrix_flags
        for argument in sys.argv[1:]
    )
    suite_value = args.suite_jsonl or (str(DEFAULT_SUITE) if implicit_suite else "")
    suite_path = Path(suite_value).expanduser().resolve() if suite_value else None
    suite_cases: list[dict[str, Any]] = []
    suite_groups: list[dict[str, Any]] = []
    if suite_path is not None:
        if args.suite_jsonl and not implicit_suite:
            raise ValueError("--suite-jsonl cannot be combined with matrix selectors")
        if args.v01_reference or args.long_output_tokens:
            raise ValueError("--suite-jsonl cannot be combined with another run mode")
        suite_cases, suite_groups = _load_suite_jsonl(suite_path)
        datasets = tuple(
            dict.fromkeys(str(group["dataset"]) for group in suite_groups)
        )
        suite_length = str(suite_groups[0]["length"])
        lengths = (suite_length,)
    else:
        datasets = _csv(args.datasets, DATASETS, "datasets")
        lengths = _csv(args.lengths.upper(), tuple(LENGTHS), "lengths")
    if args.v01_reference:
        datasets = ("musique",)
        lengths = ("256K",)
        if args.long_output_tokens:
            raise ValueError(
                "--v01-reference is immutable and cannot use long-output mode"
            )
    if args.ttft_warmup < 0 or args.ttft_iters <= 0 or args.quality_repeats <= 0:
        raise ValueError("TTFT and quality sample counts must be positive")
    if not 1 <= args.cases_per_dataset <= 200:
        raise ValueError("--cases-per-dataset must be in [1, 200]")
    hardware_profile = _resolve_hardware_profile(args.hardware_profile)
    print(
        f"[hardware] requested={args.hardware_profile} effective={hardware_profile}",
        flush=True,
    )
    profile_queries = 1 if args.v01_reference else args.cases_per_dataset
    python = _resolve_python(args.python)
    holder_python = Path(args.holder_python).expanduser().resolve()
    if not holder_python.is_file():
        raise FileNotFoundError(f"holder Python is absent: {holder_python}")
    model = _ensure_model(
        Path(args.model_path), args.model_repo, bool(args.download_model)
    )
    _validate_checkpoint_reader(python, model)
    _ensure_datasets(download=True)
    sparse = _validate_release_configs(model)
    profiles = {}
    if suite_path is None and not args.v01_reference:
        for length in lengths:
            for dataset in datasets:
                profile = _build_profile(
                    python,
                    model,
                    length,
                    dataset,
                    profile_queries,
                    args.long_output_tokens,
                )
                profiles[(length, dataset)] = _validate_profile(
                    profile,
                    length,
                    dataset,
                    profile_queries,
                    args.long_output_tokens,
                )

    run_specs: list[dict[str, Any]] = []
    if suite_path is not None:
        for group in suite_groups:
            run_specs.append(
                {
                    "run_id": str(group["group_id"]),
                    "length": str(group["length"]),
                    "dataset": str(group["dataset"]),
                    "profile_path": Path(group["profile_path"]),
                    "num_queries": len(group["cases"]),
                    "long_output_tokens": int(group["long_output_tokens"]),
                    "suite_cases": list(group["cases"]),
                }
            )
    else:
        for length in lengths:
            for dataset in datasets:
                run_specs.append(
                    {
                        "run_id": dataset,
                        "length": length,
                        "dataset": dataset,
                        "profile_path": (
                            None
                            if args.v01_reference
                            else _profile_path(
                                length,
                                dataset,
                                profile_queries,
                                args.long_output_tokens,
                            )
                        ),
                        "num_queries": profile_queries,
                        "long_output_tokens": args.long_output_tokens,
                        "suite_cases": [],
                    }
                )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (HERE / "results/deepseek_v4_flash" / stamp)
    )
    release_template = {
        "format": "redknot_deepseek_v4_flash_release_run_v1",
        "run_mode": (
            f"release_{lengths[0].lower()}_15case_suite"
            if suite_path is not None
            else ("v01_reference" if args.v01_reference else "multidataset_matrix")
        ),
        "created_utc": stamp,
        "repo": str(REPO),
        "model": str(model),
        "model_repo_fallback": args.model_repo,
        "python": str(python),
        "hardware_profile": hardware_profile,
        "length_parameters": {
            length: _length_spec(length, hardware_profile) for length in lengths
        },
        "datasets": list(datasets),
        "lengths": list(lengths),
        "cases_per_dataset": 0 if suite_path is not None else profile_queries,
        "suite_case_count": len(suite_cases),
        "suite_jsonl": str(suite_path) if suite_path is not None else "",
        "suite_jsonl_sha256": _sha256(suite_path) if suite_path is not None else "",
        "long_output_target_tokens": (
            "mixed" if suite_path is not None else args.long_output_tokens
        ),
        "answer_style": (
            "mixed_short10_long5"
            if suite_path is not None
            else (
                "direct_answer_plus_document_evidence_v1"
                if args.long_output_tokens
                else "shortest_exact_span_v1"
            )
        ),
        "data_sha256": DATASET_SHA256,
        "head_config": str(HEAD_CONFIG),
        "head_config_sha256": _sha256(HEAD_CONFIG),
        "sparse_config": str(SPARSE_CONFIG),
        "sparse_config_sha256": _sha256(SPARSE_CONFIG),
        "long_output_showcase_sha256": {
            path.name: _sha256(path)
            for path in sorted(SHOWCASE_ROOT.glob("long_output_*.json"))
            if path.is_file()
        },
        "profile_sha256": {
            str(spec["run_id"]): _sha256(Path(spec["profile_path"]))
            for spec in run_specs
            if spec["profile_path"] is not None
        },
        "runs": [],
    }
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to reuse output directory: {output}")
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"resume output has no summary.json: {summary_path}"
            )
        release = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in (
            "format",
            "run_mode",
            "model",
            "hardware_profile",
            "length_parameters",
            "datasets",
            "lengths",
            "cases_per_dataset",
            "suite_case_count",
            "suite_jsonl",
            "suite_jsonl_sha256",
            "long_output_target_tokens",
            "answer_style",
            "head_config_sha256",
            "sparse_config_sha256",
            "long_output_showcase_sha256",
            "profile_sha256",
        ):
            if release.get(key) != release_template.get(key):
                raise ValueError(
                    f"resume configuration mismatch for {key}: "
                    f"existing={release.get(key)!r} current={release_template.get(key)!r}"
                )
        requested = {str(spec["run_id"]) for spec in run_specs}
        release["runs"] = [
            record
            for record in release.get("runs", [])
            if str(record.get("run_id", record.get("dataset"))) not in requested
        ]
    else:
        output.mkdir(parents=True)
        release = release_template
    _atomic_json(output / "summary.json", release)
    _write_comparison_report(output / "comparison.md", release)
    if args.prepare_only:
        print(f"[ready] release inputs validated: {output / 'summary.json'}")
        return 0

    holder_pid = _ensure_release_holder(
        holder_python, output / "initial_gpu_holder.log"
    )
    adaptive = sparse["adaptive_assignment_topk"]
    controller = sparse["generalized_row_controller"]
    failures = 0
    for run_spec in run_specs:
        length = str(run_spec["length"])
        spec = _length_spec(length, hardware_profile)
        dataset = str(run_spec["dataset"])
        run_id = str(run_spec["run_id"])
        run_long_output_tokens = int(run_spec["long_output_tokens"])
        case_dir = output / length.lower() / run_id
        try:
            case_dir.mkdir(parents=True, exist_ok=args.resume)
            qualification_profile = (
                ""
                if run_spec["profile_path"] is None
                else str(run_spec["profile_path"])
            )
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(REPO / "python")
                    + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""),
                    "REDKNOT_PYTHON": str(python),
                    "REDKNOT_HOLDER_PYTHON": str(holder_python),
                    "REDKNOT_MODEL_PATH": str(model),
                    "REDKNOT_HARDWARE_PROFILE": hardware_profile,
                    "REDKNOT_LONGBENCH_DIR": str(DATA_DIR),
                    "REDKNOT_IH_LONG_OUTPUT_TOKENS": str(
                        run_long_output_tokens
                    ),
                    "REDKNOT_RELEASE_ADAPTIVE_TOPK_MASS": str(
                        adaptive["cumulative_mass"]
                    ),
                    "REDKNOT_RELEASE_ADAPTIVE_TOPK_BUCKETS": ",".join(
                        map(str, adaptive["allowed_buckets"])
                    ),
                    "REDKNOT_SWA_FULL_TOKENS_RATIO": str(spec["swa_full_ratio"]),
                    "REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH": str(
                        spec["cublas_woa_fastpath"]
                    ),
                    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
                    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "0",
                    "SGLANG_OPT_USE_TILELANG_MHC_POST": "0",
                }
            )
            command = [
                str(SUPERVISOR),
                str(case_dir),
                str(holder_pid),
                "0",
                str(args.ttft_warmup),
                str(args.ttft_iters),
                str(spec["merged_prefill"]),
                str(spec["target_tokens"]),
                "1",
                "1",
                str(spec["mem_fraction"]),
                args.qps_concurrencies,
                "1" if args.measure_qps else "0",
                str(args.qps_warmup_waves),
                str(args.qps_waves),
                str(controller["strong_active_ratio"]),
                qualification_profile,
                str(spec["query_protection_tokens"]),
                "1",
                str(args.quality_repeats),
                str(controller["strong_active_ratio"]),
                str(controller["medium_active_ratio"]),
                str(controller["diffuse_active_ratio"]),
            ]
            print(f"[run] {length}/{run_id}: {' '.join(command)}", flush=True)
            result_path = case_dir / "result.json"
            if args.resume and result_path.is_file():
                exit_path = case_dir / "exit_code"
                exit_code = (
                    int(exit_path.read_text(encoding="utf-8").strip())
                    if exit_path.is_file()
                    else 0
                )
                print(
                    f"[resume] {length}/{run_id}: reusing {result_path}",
                    flush=True,
                )
            else:
                completed = subprocess.run(command, cwd=REPO, env=env, check=False)
                exit_code = completed.returncode
            record: dict[str, Any] = {
                "length": length,
                "dataset": dataset,
                "run_id": run_id,
                "long_output_target_tokens": run_long_output_tokens,
                "exit_code": exit_code,
                "result": str(result_path),
            }
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                record["result_sha256"] = _sha256(result_path)
                record["metrics"] = _extract_metrics(result)
                comparisons = _extract_output_pairs(result)
                if suite_path is not None:
                    members = list(run_spec["suite_cases"])
                    comparisons = [
                        pair for pair in comparisons if int(pair.get("repeat", 0)) == 0
                    ]
                    if len(comparisons) != len(members):
                        raise RuntimeError(
                            f"suite group {run_id!r} produced {len(comparisons)} "
                            f"primary comparisons for {len(members)} cases"
                        )
                    by_query = {int(pair["query_index"]): pair for pair in comparisons}
                    if set(by_query) != set(range(len(members))):
                        raise RuntimeError(
                            f"suite group {run_id!r} output query indices drifted"
                        )
                    comparisons = []
                    for member in members:
                        pair = dict(by_query[int(member["query_index"])])
                        pair["suite_case_index"] = int(member["suite_case_index"])
                        pair["case_id"] = str(member["case_id"])
                        pair["selection_origin"] = str(member["selection_origin"])
                        pair["eligible_for_accuracy_aggregate"] = bool(
                            member["eligible_for_accuracy_aggregate"]
                        )
                        comparisons.append(pair)
                record["output_comparisons"] = comparisons
            else:
                failures += 1
                record["fatal"] = "result.json was not produced"
            release["runs"].append(record)
            _atomic_json(output / "summary.json", release)
            _write_comparison_report(output / "comparison.md", release)
            if "output_comparisons" in record:
                _print_comparison(length, dataset, record)
            leaders = _holder_leaders()
            if len(leaders) != 1:
                raise RuntimeError(
                    f"supervisor did not restore exactly one holder: {leaders}"
                )
            holder_pid = leaders[0]
            if record.get("fatal") and not args.keep_going:
                print(f"[fatal] {length}/{run_id} produced no result", file=sys.stderr)
                return 2
        except BaseException:
            # The supervisor owns holder restoration for launched runs. This
            # wrapper preserves the original exception after result/report work.
            raise
    print(f"[done] summary: {output / 'summary.json'}", flush=True)
    return 2 if failures else 0


def _run_default_release_suites() -> int:
    """Run all four immutable release suites under one timestamped parent."""

    release_suite_manifest = _validate_default_release_manifest()
    missing = [str(path) for path in DEFAULT_SUITES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"default release suites are absent: {missing}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = HERE / "results/deepseek_v4_flash" / stamp
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    final_status = 0
    for suite_path in DEFAULT_SUITES:
        suite_cases, _ = _load_suite_jsonl(suite_path)
        length = str(suite_cases[0]["length"])
        child_output = output / f"{length.lower()}_suite15"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--suite-jsonl",
            str(suite_path),
            "--output-dir",
            str(child_output),
        ]
        print(
            f"[release] starting {length} frozen 15-case suite: {suite_path}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=HERE, check=False)
        records.append(
            {
                "length": length,
                "suite_jsonl": str(suite_path),
                "suite_jsonl_sha256": _sha256(suite_path),
                "output_dir": str(child_output),
                "exit_code": int(completed.returncode),
            }
        )
        if completed.returncode and final_status == 0:
            final_status = int(completed.returncode)
    _atomic_json(
        output / "four_length_summary.json",
        {
            "format": "redknot_deepseek_v4_flash_four_length_release_v1",
            "created_utc": stamp,
            "release_suites_manifest": str(RELEASE_SUITES_MANIFEST),
            "release_suites_manifest_sha256": _sha256(RELEASE_SUITES_MANIFEST),
            "execution_contract": release_suite_manifest["execution"],
            "ordered_lengths": ["64K", "128K", "256K", "440K"],
            "suite_case_count_per_length": 15,
            "runs": records,
        },
    )
    print(f"[done] four-length summary: {output / 'four_length_summary.json'}")
    return final_status


def main() -> int:
    if len(sys.argv) == 1:
        return _run_default_release_suites()
    return _run_single()


if __name__ == "__main__":
    raise SystemExit(main())
