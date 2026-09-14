"""RedKnot-only frozen-case conversion and Flash benchmark CPU preflight.

The original suite files contain hashes, not token IDs. This module never imports
the SGLang prompt builder, silently re-tokenizes cases, or downloads datasets.
It accepts independently exported exact token IDs and checks the original
uint32-little-endian hashes. A new custom corpus may instead use suite=custom.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.benchmark_redknot import Case, parse_cases

HERE = Path(__file__).resolve().parent
FLASH_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
CATALOG_SHA256 = "19810f8d83072505141024ad51a6f25cb1f0e6d1dd494d12f08ca147acf8cd16"
MODEL_FILES_SHA256 = "d7d9a4ae3b9916187fb424d7313ac3412df6c4de3374d72b7c52c33c0cc043db"
LENGTHS = ("64K", "128K", "256K", "440K")
# These are the pinned Flash-0731 grouped wo_a dimensions, not Pro dimensions.
FLASH_GROUPS = 8
FLASH_O_LORA_RANK = 1024


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_nonfinite,
    )


def _pinned_json(path: Path, digest: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"bundled metadata checksum mismatch: {path.name}")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)


def load_catalog(suite: str) -> list[dict[str, Any]]:
    if suite not in (*LENGTHS, "all", "custom"):
        raise ValueError("unknown Flash suite")
    if suite == "custom":
        return []
    catalog = _pinned_json(HERE / "data/flash_release_catalog.json", CATALOG_SHA256)
    if catalog["model_revision"] != FLASH_REVISION or len(catalog["cases"]) != 60:
        raise ValueError("bundled Flash catalog identity mismatch")
    return [row for row in catalog["cases"] if suite == "all" or row["length"] == suite]


def token_hash(tokens: Sequence[int]) -> str:
    """Same serialization as RedKnot's _ih_chunk_hash, without its framework."""
    digest = hashlib.sha256()
    for token in tokens:
        if type(token) is not int or not 0 <= token < 2**32:
            raise ValueError("token IDs must be uint32 integers")
        digest.update(token.to_bytes(4, "little", signed=False))
    return "sha256:" + digest.hexdigest()


def convert_cases(
    document: Any, catalog: Sequence[Mapping[str, Any]], *, custom: bool = False
) -> dict[str, Any]:
    """Preserve every input token; reject missing/extra/altered frozen cases.

    Original long-output cases are not in the reference-F1 aggregate. Their gold
    short spans remain in provenance, while references=[] prevents measuring
    prose-overlap against a short span as if it were a complete answer reference.
    """
    parsed = parse_cases(document)
    by_id = {case.id: case for case in parsed}
    if not custom and set(by_id) != {row["id"] for row in catalog}:
        raise ValueError("exact frozen suite case IDs required; no partial suite")
    converted, provenance = [], []
    rows = [{"id": case.id} for case in parsed] if custom else catalog
    for row in rows:
        case = by_id[row["id"]]
        hashes = [token_hash(chunk) for chunk in case.chunks]
        full_hash, query_hash = token_hash(case.tokens), token_hash(case.query)
        references = list(case.references)
        if not custom:
            checks = (
                (len(case.chunks), row["num_chunks"], "chunk count"),
                (
                    list(map(len, case.chunks)),
                    [row["chunk_tokens"]] * row["num_chunks"],
                    "chunk lengths",
                ),
                (len(case.tokens), row["total_tokens"], "prompt length"),
                (len(case.query), row["query_tokens"], "query length"),
                (hashes, row["offline_chunk_hashes"], "chunk token hashes"),
                (full_hash, row["full_input_ids_sha256"], "full prompt hash"),
                (query_hash, row["query_hash"], "query token hash"),
            )
            for actual, expected, label in checks:
                if actual != expected:
                    raise ValueError(f"{case.id}: frozen {label} mismatch")
            expected_refs = (
                row["answers"] if row["eligible_for_accuracy_aggregate"] else []
            )
            if references and references != row["answers"]:
                raise ValueError(
                    f"{case.id}: references differ from frozen gold answers"
                )
            references = list(expected_refs)
        converted.append(
            {
                "id": case.id,
                "chunks": [list(chunk) for chunk in case.chunks],
                "query": list(case.query),
                "references": references,
            }
        )
        provenance.append(
            {
                **dict(row),
                "full_input_ids_sha256": full_hash,
                "offline_chunk_hashes": hashes,
                "query_hash": query_hash,
                "reference_f1_eligible": bool(references),
            }
        )
    return {
        "cases": converted,
        "provenance": {
            "kind": "custom_tokenized" if custom else "frozen_redknot_token_export",
            "model_revision": FLASH_REVISION,
            "hash_encoding": "uint32_little_endian",
            "retokenized": False,
            "truncated_tokens": 0,
            "cases": provenance,
        },
    }


def validate_flash_policy(policy: Mapping[str, Any]) -> None:
    if not policy or policy.get("engine_family") != "deepseek_v4_flash":
        raise ValueError("an explicit deepseek_v4_flash policy is required")
    if not policy.get("enabled") or not policy.get("allow_approximate"):
        raise ValueError("enabled and allow_approximate must both be true")
    if policy.get("model_revision") != FLASH_REVISION:
        raise ValueError("Flash policy must use the pinned Flash-0731 revision")
    # Numeric selectors prevent duplicate name/index aliases from under-counting
    # the staged per-layer payload; the worker also resolves real native layers.
    for key, heads in policy["local_heads"].items():
        if not key.isdigit() or str(int(key)) != key or not 0 <= int(key) < 43:
            raise ValueError("Flash reproduction needs canonical layer indices 0..42")
        if not heads or max(heads) >= 64:
            raise ValueError("each selected Flash layer needs logical heads in 0..63")


def cache_plan(
    policy: Mapping[str, Any],
    *,
    cases: Sequence[Case] | None = None,
    catalog: Sequence[Mapping[str, Any]] = (),
    output_tokens: int = 128,
    max_model_len: int = 8192,
) -> dict[str, Any]:
    """Payload-only CPU budget: all unique chunks are captured before all pairs.

    z_off is BF16 [T, 8*1024] plus per-layer INT64 source positions. This is not
    native KV, GPU peak memory, Python overhead, or whole-model compute savings.
    During capture the staged incoming payload coexists with earlier entries.
    """
    validate_flash_policy(policy)
    per_token = len(policy["local_heads"]) * (2 * FLASH_GROUPS * FLASH_O_LORA_RANK + 8)
    chunks: dict[str, int] = {}
    rows, blockers = [], []
    if cases is not None:
        descriptions = [
            (case.id, len(case.tokens), [(token_hash(c), len(c)) for c in case.chunks])
            for case in cases
        ]
    else:
        descriptions = [
            (
                row["id"],
                row["total_tokens"],
                [(key, row["chunk_tokens"]) for key in row["offline_chunk_hashes"]],
            )
            for row in catalog
        ]
    for identifier, prompt_tokens, entries in descriptions:
        chunks.update(entries)
        needed = sum(dict(entries).values()) * per_token
        if prompt_tokens + output_tokens > max_model_len:
            blockers.append(f"{identifier}: prompt plus output exceeds max_model_len")
        if needed > policy["max_cache_bytes"]:
            blockers.append(
                f"{identifier}: required chunk set exceeds CPU cache budget"
            )
        if all(length <= policy["boundary_tokens"] for _, length in entries):
            blockers.append(f"{identifier}: no clean rows beyond the dirty boundary")
        rows.append(
            {
                "id": identifier,
                "prompt_tokens": prompt_tokens,
                "required_case_payload_bytes": needed,
            }
        )
    required = sum(chunks.values()) * per_token
    if required > policy["max_cache_bytes"]:
        blockers.append(
            "all unique captures exceed budget; early cases would be evicted"
        )
    return {
        "case_count": len(rows),
        "cases": rows,
        "unique_chunks": len(chunks),
        "cached_tokens": sum(chunks.values()),
        "payload_bytes_per_cached_token": per_token,
        "required_all_capture_payload_bytes": required,
        "largest_capture_staging_bytes": max(chunks.values(), default=0) * per_token,
        "configured_cache_bytes": policy["max_cache_bytes"],
        "budget_token_capacity": policy["max_cache_bytes"] // per_token,
        "cpu_peak_memory_verified": False,
        "gpu_memory_verified": False,
        "blockers": blockers,
    }


def validate_gpu_environment(environ: Mapping[str, str]) -> None:
    uuid = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    if (
        re.fullmatch(
            rf"(?:0|[1-9][0-9]*|{uuid})", environ.get("CUDA_VISIBLE_DEVICES", "")
        )
        is None
    ):
        raise ValueError(
            "set CUDA_VISIBLE_DEVICES to one checked-idle GPU UUID/ordinal"
        )
    for name, value in (
        ("WORLD_SIZE", "1"),
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("VLLM_DP_SIZE", "1"),
        ("VLLM_DP_RANK", "0"),
        ("VLLM_DP_RANK_LOCAL", "0"),
    ):
        if name in environ and environ[name] != value:
            raise ValueError(f"unsupported distributed environment: {name}")


def model_file_inventory() -> dict[str, Any]:
    manifest = _pinned_json(HERE / "data/flash_model_files.json", MODEL_FILES_SHA256)
    if (
        manifest["hf_revision"] != FLASH_REVISION
        or manifest["file_count"] != 74
        or len(manifest["files"]) != 74
    ):
        raise ValueError("pinned model inventory identity mismatch")
    return manifest


def verify_local_files(
    root: Path, files: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Read-only complete SHA verification; never trust progress or .part files.

    Reject symlinks and non-regular files. Callers must prevent concurrent model
    mutation: these hashes certify the read snapshot, not future writes.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError("model must be a real local directory, not a symlink")
    records = []
    # Inventory pass rejects an unfinished checkpoint before hashing large shards.
    for spec in files:
        relative = Path(spec["path"])
        if relative.is_absolute() or any(
            part in {".", ".."} for part in relative.parts
        ):
            raise ValueError("unsafe model manifest path")
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"model symlink is unsupported: {relative}")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != spec["size"]:
            raise ValueError(
                f"model file missing, non-regular or incomplete: {relative}"
            )
        records.append({"path": path, "spec": spec, "stat": info})
    verified = []
    for record in records:
        digest = hashlib.sha256()
        with record["path"].open("rb") as handle:
            before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")

        def identity(value):
            return tuple(getattr(value, key) for key in keys)

        if (
            identity(record["stat"]) != identity(before)
            or identity(before) != identity(after)
            or identity(after) != identity(record["path"].stat())
        ):
            raise ValueError("model file changed during independent verification")
        if digest.hexdigest() != record["spec"]["sha256"]:
            raise ValueError(f"model SHA mismatch: {record['spec']['path']}")
        verified.append(dict(record["spec"]))
    return verified
