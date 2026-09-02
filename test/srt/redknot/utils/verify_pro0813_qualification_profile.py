#!/usr/bin/env python3
"""Fail-closed CPU verifier for a frozen Pro-0813 qualification profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


OFFICIAL_REPO_ID = "deepseek-ai/DeepSeek-V4-Pro-0813"
OFFICIAL_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
OFFICIAL_MODEL_ROOT = Path("/workspace/Models/DeepSeek-V4-Pro-0813")
PINNED_TOKENIZERS_VERSION = "0.22.1"
TOKENIZER_FILES = {
    "encoding/encoding_dsv4.py": (
        "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6"
    ),
    "tokenizer.json": (
        "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
    ),
    "tokenizer_config.json": (
        "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547"
    ),
}
TARGET_GEOMETRIES = {
    450560: {"chunk_tokens": 56320, "num_chunks": 8, "cohort_index": 44000},
    524288: {"chunk_tokens": 65536, "num_chunks": 8, "cohort_index": 51200},
}
HEX64 = re.compile(r"[0-9a-f]{64}")
PREFIXED_HEX64 = re.compile(r"sha256:[0-9a-f]{64}")


class ProfileVerificationError(ValueError):
    pass


def _fail(message: str) -> None:
    raise ProfileVerificationError(message)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(content: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            content.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is not strict UTF-8 JSON: {error}")
    if not isinstance(document, dict):
        _fail(f"{label} must contain one JSON object")
    return document


def _require_regular_nonsymlink(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        _fail(f"{label} is absent: {path}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a non-symlink regular file: {path}")


def _require_hex64(value: Any, label: str, *, prefixed: bool = False) -> str:
    pattern = PREFIXED_HEX64 if prefixed else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolve_co_located_artifact(
    profile_path: Path, value: Any, label: str
) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{label} path must be a non-empty string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or len(relative.parts) != 1 or value != relative.name:
        _fail(
            f"{label} must be one same-directory relative filename, got {value!r}"
        )
    candidate = profile_path.parent / value
    _require_regular_nonsymlink(candidate, label)
    if candidate.resolve(strict=True).parent != profile_path.parent:
        _fail(f"{label} escapes the frozen profile directory")
    return candidate


def _validate_model_identity(
    profile: dict[str, Any], model_root: Path
) -> dict[str, str]:
    expected = {
        "repo_id": OFFICIAL_REPO_ID,
        "revision": OFFICIAL_REVISION,
        "canonical_root": str(OFFICIAL_MODEL_ROOT),
        "encoding_sha256": TOKENIZER_FILES["encoding/encoding_dsv4.py"],
        "tokenizer_sha256": TOKENIZER_FILES["tokenizer.json"],
        "tokenizer_config_sha256": TOKENIZER_FILES["tokenizer_config.json"],
        "tokenizers_version": PINNED_TOKENIZERS_VERSION,
    }
    if profile.get("model_identity") != expected:
        _fail("profile model_identity is not the pinned official Pro revision")
    for relative, expected_digest in TOKENIZER_FILES.items():
        source = model_root / relative
        _require_regular_nonsymlink(source, f"model tokenizer input {relative}")
        observed = _sha256_file(source)
        if observed != expected_digest:
            _fail(
                f"model tokenizer digest mismatch for {relative}: "
                f"expected={expected_digest} observed={observed}"
            )
    try:
        observed_version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        _fail("the pinned tokenizers runtime is not installed")
    if observed_version != PINNED_TOKENIZERS_VERSION:
        _fail(
            "tokenizers runtime differs from profile identity: "
            f"expected={PINNED_TOKENIZERS_VERSION} observed={observed_version}"
        )
    return expected


def _load_raw_rows(dataset_path: Path, required_rows: set[int]) -> dict[int, dict]:
    rows = {}
    with dataset_path.open("rb") as handle:
        for row_id, raw_line in enumerate(handle):
            if row_id not in required_rows:
                continue
            row_bytes = raw_line.rstrip(b"\r\n")
            try:
                payload = json.loads(row_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                _fail(f"dataset row {row_id} is invalid JSON: {error}")
            rows[row_id] = {
                "row_sha256": _sha256_bytes(row_bytes),
                "payload": payload,
            }
    missing = required_rows.difference(rows)
    if missing:
        _fail(f"dataset is missing frozen rows: {sorted(missing)}")
    return rows


def _validate_data_selection(
    document: dict[str, Any],
    profile: dict[str, Any],
    dataset_path: Path,
) -> tuple[list[dict], dict[str, Any]]:
    if document.get("format") != "redknot_indexer_hot_data_selection_v1":
        _fail("data manifest format is invalid")
    claimed = _require_hex64(
        document.get("selection_sha256"), "data selection digest"
    )
    payload = dict(document)
    payload.pop("selection_sha256", None)
    if _canonical_sha256(payload) != claimed:
        _fail("data selection canonical digest mismatch")
    if claimed != profile.get("data_selection_sha256"):
        _fail("profile and data selection digests differ")

    dataset = document.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("name") != profile["dataset"]:
        _fail("data manifest dataset identity differs from profile")
    if dataset.get("row_id_base") != 0:
        _fail("data manifest row_id_base must be zero")
    _require_regular_nonsymlink(dataset_path, "LongBench dataset")
    dataset_sha256 = _sha256_file(dataset_path)
    if (
        dataset.get("bytes") != dataset_path.stat().st_size
        or dataset.get("sha256") != dataset_sha256
        or profile.get("dataset_source_sha256") != dataset_sha256
    ):
        _fail("raw LongBench dataset byte identity differs from profile")

    expected_geometry = {
        "chunk_tokens": profile["chunk_tokens"],
        "num_chunks": profile["num_chunks"],
        "num_queries": profile["num_queries"],
    }
    if document.get("geometry") != expected_geometry:
        _fail("data selection geometry differs from profile")
    selection = document.get("selection")
    if not isinstance(selection, dict) or selection.get("mode") != "offset":
        _fail("data selection policy is invalid")
    if (
        selection.get("row_offset") != profile["row_offset"]
        or selection.get("excluded_row_ids") != []
        or selection.get("excluded_selection_sha256") != []
        or profile.get("exclude_manifests") != []
    ):
        _fail("canonical profile must use the unexcluded zero-offset cohort")
    chunks = selection.get("chunks")
    queries = selection.get("queries")
    if (
        not isinstance(chunks, list)
        or len(chunks) != profile["num_chunks"]
        or not isinstance(queries, list)
        or len(queries) != profile["num_queries"]
    ):
        _fail("data selection chunk/query counts are invalid")

    required_rows: set[int] = set()
    row_to_chunk = {}
    row_entries = {}
    for chunk_index, chunk in enumerate(chunks):
        if (
            not isinstance(chunk, dict)
            or chunk.get("chunk_index") != chunk_index
            or not isinstance(chunk.get("rows"), list)
            or not chunk["rows"]
        ):
            _fail(f"data selection chunk {chunk_index} is malformed")
        _require_hex64(chunk.get("token_hash"), "chunk token hash", prefixed=True)
        used_total = 0
        for entry in chunk["rows"]:
            if not isinstance(entry, dict) or type(entry.get("row_id")) is not int:
                _fail(f"chunk {chunk_index} has a malformed row entry")
            row_id = entry["row_id"]
            if row_id < 0 or row_id in row_to_chunk:
                _fail(f"data selection repeats or has invalid row {row_id}")
            context_count = entry.get("context_token_count")
            used_count = entry.get("used_token_count")
            if (
                type(context_count) is not int
                or type(used_count) is not int
                or context_count <= 0
                or used_count <= 0
                or used_count > context_count
            ):
                _fail(f"row {row_id} has invalid context token counts")
            _require_hex64(entry.get("row_sha256"), f"row {row_id} digest")
            _require_hex64(
                entry.get("context_token_hash"),
                f"row {row_id} context token hash",
                prefixed=True,
            )
            used_total += used_count
            row_to_chunk[row_id] = chunk_index
            row_entries[row_id] = entry
            required_rows.add(row_id)
        if used_total != profile["chunk_tokens"]:
            _fail(f"chunk {chunk_index} does not fill the frozen token quota")

    query_rows = []
    for query_index, query in enumerate(queries):
        if (
            not isinstance(query, dict)
            or query.get("query_index") != query_index
            or type(query.get("row_id")) is not int
            or type(query.get("expected_chunk")) is not int
        ):
            _fail(f"data selection query {query_index} is malformed")
        row_id = query["row_id"]
        expected_chunk = query["expected_chunk"]
        if row_to_chunk.get(row_id) != expected_chunk:
            _fail(f"query row {row_id} is not in its expected chunk")
        row_entry = row_entries[row_id]
        if row_entry["used_token_count"] != row_entry["context_token_count"]:
            _fail(f"query row {row_id} is not fully contained")
        for key in ("row_sha256", "question_sha256", "answers_sha256"):
            _require_hex64(query.get(key), f"query {query_index} {key}")
        _require_hex64(
            query.get("query_token_hash"),
            f"query {query_index} token hash",
            prefixed=True,
        )
        query_rows.append(row_id)
        required_rows.add(row_id)
    if query_rows != profile["query_row_ids"] or len(set(query_rows)) != len(
        query_rows
    ):
        _fail("data selection query rows differ from profile or repeat")

    raw_rows = _load_raw_rows(dataset_path, required_rows)
    for row_id, entry in row_entries.items():
        if raw_rows[row_id]["row_sha256"] != entry["row_sha256"]:
            _fail(f"raw dataset row digest mismatch for row {row_id}")
    for query in queries:
        row_id = query["row_id"]
        row = raw_rows[row_id]["payload"]
        answers = [str(value) for value in row.get("answers", [])]
        if (
            raw_rows[row_id]["row_sha256"] != query["row_sha256"]
            or _sha256_bytes(str(row.get("input", "")).encode("utf-8"))
            != query["question_sha256"]
            or _canonical_sha256(answers) != query["answers_sha256"]
        ):
            _fail(f"raw query payload digest mismatch for row {row_id}")
    return chunks, dataset


def _validate_prompt_manifest(
    document: dict[str, Any],
    profile: dict[str, Any],
    chunks: list[dict],
    dataset_identity: dict[str, Any],
) -> None:
    if document.get("format") != "redknot_pure_mla_prompt_multi_v1":
        _fail("prompt manifest must be the multi-query frozen format")
    claimed = _require_hex64(
        document.get("prompt_manifest_sha256"), "prompt manifest digest"
    )
    payload = dict(document)
    payload.pop("prompt_manifest_sha256", None)
    if _canonical_sha256(payload) != claimed:
        _fail("prompt manifest canonical digest mismatch")
    if claimed != profile.get("prompt_manifest_sha256"):
        _fail("profile and prompt manifest digests differ")
    if document.get("data_selection_sha256") != profile[
        "data_selection_sha256"
    ]:
        _fail("prompt and data selection digests differ")
    if document.get("dataset") != dataset_identity:
        _fail("prompt and data manifest dataset identities differ")
    if document.get("output_blind") is not True:
        _fail("prompt manifest is not output-blind")

    expected_protocol = {
        "add_default_bos_token": True,
        "document_delimiter": "\n\n",
        "drop_thinking": True,
        "encoder_path": str(
            OFFICIAL_MODEL_ROOT / "encoding" / "encoding_dsv4.py"
        ),
        "encoder_sha256": "sha256:"
        + TOKENIZER_FILES["encoding/encoding_dsv4.py"],
        "query_instruction_mode": "per_case_short_span_v1",
        "reasoning_effort": "low",
        "thinking_mode": "chat",
        "tokenizer_config_path": str(
            OFFICIAL_MODEL_ROOT / "tokenizer_config.json"
        ),
        "tokenizer_config_sha256": "sha256:"
        + TOKENIZER_FILES["tokenizer_config.json"],
        "tokenizer_path": str(OFFICIAL_MODEL_ROOT / "tokenizer.json"),
        "tokenizer_sha256": "sha256:"
        + TOKENIZER_FILES["tokenizer.json"],
    }
    if document.get("protocol") != expected_protocol:
        _fail("prompt protocol is not the pinned official Pro short-span protocol")

    geometry = document.get("geometry")
    cases = document.get("cases")
    prompt = document.get("prompt")
    source = document.get("source")
    if not all(isinstance(value, dict) for value in (geometry, prompt, source)):
        _fail("prompt manifest geometry/prompt/source objects are missing")
    if not isinstance(cases, list) or len(cases) != profile["num_queries"]:
        _fail("prompt manifest case count differs from profile")
    totals = [case.get("total_tokens") for case in cases if isinstance(case, dict)]
    expected_geometry = {
        "chunk_tokens": profile["chunk_tokens"],
        "max_total_tokens": max(totals, default=0),
        "min_total_tokens": min(totals, default=0),
        "num_chunks": profile["num_chunks"],
        "num_queries": profile["num_queries"],
        "query_start": profile["target_tokens"],
    }
    if geometry != expected_geometry or profile["max_total_tokens"] != max(
        totals, default=0
    ):
        _fail("prompt geometry differs from profile/cases")
    if source.get("chunk_order") != list(range(profile["num_chunks"])):
        _fail("prompt source chunk order is not canonical")
    if source.get("query_row_ids") != profile["query_row_ids"]:
        _fail("prompt source query rows differ from profile")
    if source.get("chunk_hashes") != [chunk["token_hash"] for chunk in chunks]:
        _fail("prompt source chunk hashes differ from data selection")
    offline_hashes = prompt.get("offline_chunk_hashes")
    if (
        not isinstance(offline_hashes, list)
        or len(offline_hashes) != profile["num_chunks"]
        or any(
            not isinstance(value, str) or PREFIXED_HEX64.fullmatch(value) is None
            for value in offline_hashes
        )
        or offline_hashes != profile["offline_chunk_hashes"]
    ):
        _fail("offline prompt chunk hashes differ from profile")
    _require_hex64(
        prompt.get("offline_prefix_hash"), "offline prefix hash", prefixed=True
    )
    if cases != profile["cases"]:
        _fail("profile cases are not byte-equivalent to prompt cases")
    for query_index, case in enumerate(cases):
        if (
            case.get("query_index") != query_index
            or case.get("query_row_id") != profile["query_row_ids"][query_index]
            or type(case.get("expected_source_chunk")) is not int
            or not 0 <= case["expected_source_chunk"] < profile["num_chunks"]
            or type(case.get("online_suffix_tokens")) is not int
            or case["online_suffix_tokens"] <= 0
            or case.get("total_tokens")
            != profile["target_tokens"] + case["online_suffix_tokens"]
        ):
            _fail(f"prompt case {query_index} has invalid geometry/identity")
        for key in (
            "text_sha256",
            "full_input_ids_sha256",
            "online_suffix_hash",
            "question_sha256",
            "answers_sha256",
        ):
            _require_hex64(
                case.get(key), f"prompt case {query_index} {key}", prefixed=True
            )


def verify_profile(
    profile_path: Path,
    *,
    expected_target_tokens: int,
    model_root: Path = OFFICIAL_MODEL_ROOT,
    data_dir: Path | None = None,
    expected_profile_sha256: str | None = None,
    include_profile_bytes: bool = False,
) -> dict[str, Any]:
    if type(include_profile_bytes) is not bool:
        _fail("include_profile_bytes must be a boolean")
    if expected_target_tokens not in TARGET_GEOMETRIES:
        _fail(f"unsupported expected target: {expected_target_tokens}")
    source = profile_path.expanduser()
    if not source.is_absolute():
        _fail("qualification profile path must be absolute")
    _require_regular_nonsymlink(source, "qualification profile")
    source = source.resolve(strict=True)
    profile_bytes = source.read_bytes()
    profile_sha256 = _sha256_bytes(profile_bytes)
    sidecar = source.with_name(source.name + ".sha256")
    _require_regular_nonsymlink(sidecar, "qualification profile digest sidecar")
    sidecar_text = sidecar.read_text(encoding="ascii")
    expected_sidecar = f"{profile_sha256}  {source.name}\n"
    if sidecar_text != expected_sidecar:
        _fail("qualification profile sidecar does not match exact profile bytes")
    if expected_profile_sha256 is not None:
        _require_hex64(expected_profile_sha256, "expected profile digest")
        if profile_sha256 != expected_profile_sha256:
            _fail("qualification profile differs from the caller-pinned digest")
    profile = _load_json_bytes(profile_bytes, "qualification profile")

    geometry = TARGET_GEOMETRIES[expected_target_tokens]
    expected_scalars = {
        "format": "redknot_multidataset_profile_v2",
        "profile_contract": "pro0813_frozen_single_corpus_v1",
        "intended_execution_profile": "full_combined_production_v1",
        "target_tokens": expected_target_tokens,
        "query_start": expected_target_tokens,
        "chunk_tokens": geometry["chunk_tokens"],
        "num_chunks": geometry["num_chunks"],
        "cohort_index": geometry["cohort_index"],
        "row_offset": 0,
        "output_blind": True,
        "long_output_target_tokens": 0,
        "answer_style": "shortest_exact_span_v1",
    }
    for key, expected in expected_scalars.items():
        if profile.get(key) != expected or type(profile.get(key)) is not type(
            expected
        ):
            _fail(
                f"profile has invalid {key}: expected={expected!r} "
                f"observed={profile.get(key)!r}"
            )
    if profile.get("diagnostic_execution_profiles") != ["zoff_only"]:
        _fail("profile must label zoff_only as diagnostic, not formal")
    dataset = profile.get("dataset")
    num_queries = profile.get("num_queries")
    query_rows = profile.get("query_row_ids")
    cases = profile.get("cases")
    if (
        not isinstance(dataset, str)
        or not dataset
        or type(num_queries) is not int
        or num_queries <= 0
        or not isinstance(query_rows, list)
        or len(query_rows) != num_queries
        or any(type(value) is not int or value < 0 for value in query_rows)
        or len(set(query_rows)) != len(query_rows)
        or not isinstance(cases, list)
        or len(cases) != num_queries
    ):
        _fail("profile query/sample cohort is malformed")
    expected_quotas = [
        {
            "dataset": dataset,
            "num_queries": num_queries,
            "query_row_ids": query_rows,
        }
    ]
    if profile.get("dataset_quotas") != expected_quotas:
        _fail("profile dataset quotas do not close over its exact samples")
    _require_hex64(profile.get("dataset_source_sha256"), "dataset source digest")
    _require_hex64(profile.get("data_selection_sha256"), "data selection digest")
    _require_hex64(profile.get("prompt_manifest_sha256"), "prompt digest")
    _validate_model_identity(profile, model_root.expanduser().resolve(strict=True))

    data_path = _resolve_co_located_artifact(
        source, profile.get("data_manifest"), "data manifest"
    )
    prompt_path = _resolve_co_located_artifact(
        source, profile.get("prompt_manifest"), "prompt manifest"
    )
    artifact_sha256 = profile.get("artifact_sha256")
    expected_artifact_names = {data_path.name, prompt_path.name}
    if not isinstance(artifact_sha256, dict) or set(artifact_sha256) != (
        expected_artifact_names
    ):
        _fail("profile artifact digest map is incomplete or has extra entries")
    for artifact in (data_path, prompt_path):
        expected = _require_hex64(
            artifact_sha256.get(artifact.name), f"{artifact.name} byte digest"
        )
        if _sha256_file(artifact) != expected:
            _fail(f"frozen artifact byte digest mismatch: {artifact.name}")

    if data_dir is None:
        data_dir = source.parents[2] / "datasets" / "LongBench" / "data"
    dataset_path = data_dir.expanduser().resolve(strict=True) / f"{dataset}.jsonl"
    data_document = _load_json_bytes(data_path.read_bytes(), "data manifest")
    chunks, dataset_identity = _validate_data_selection(
        data_document, profile, dataset_path
    )
    prompt_document = _load_json_bytes(
        prompt_path.read_bytes(), "prompt manifest"
    )
    _validate_prompt_manifest(
        prompt_document, profile, chunks, dataset_identity
    )
    result = {
        "format": "redknot_pro0813_qualification_verification_v1",
        "profile": str(source),
        "profile_sha256": profile_sha256,
        "target_tokens": expected_target_tokens,
        "dataset_quotas": expected_quotas,
        "num_cases": num_queries,
        "data_selection_sha256": profile["data_selection_sha256"],
        "prompt_manifest_sha256": profile["prompt_manifest_sha256"],
        "intended_execution_profile": profile["intended_execution_profile"],
        "pass": True,
    }
    if include_profile_bytes:
        # Internal consumer handoff: callers can parse exactly the bytes whose
        # digest and sidecar were verified, instead of reopening a mutable
        # pathname and creating a profile TOCTOU window.  The CLI leaves this
        # disabled so its JSON record remains stable and serializable.
        result["_verified_profile_bytes"] = profile_bytes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--expected-target-tokens",
        required=True,
        type=int,
        choices=tuple(TARGET_GEOMETRIES),
    )
    parser.add_argument("--model-root", type=Path, default=OFFICIAL_MODEL_ROOT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--expected-profile-sha256")
    args = parser.parse_args()
    result = verify_profile(
        args.profile,
        expected_target_tokens=args.expected_target_tokens,
        model_root=args.model_root,
        data_dir=args.data_dir,
        expected_profile_sha256=args.expected_profile_sha256,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
