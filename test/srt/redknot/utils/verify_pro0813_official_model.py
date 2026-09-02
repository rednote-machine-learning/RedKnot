#!/usr/bin/env python3
"""Fail-closed, CPU-only gate for the official DeepSeek-V4-Pro-0813 tree.

The downloader publishes ``complete`` only after checking every available LFS
SHA-256.  This gate deliberately does not hash the roughly 893 GB of weights a
second time.  It instead binds the exact official manifest, checks every target
is a safe regular file with its exact size, hashes the small config and index,
and verifies the downloader's final status record.

The production entry point has no command-line or environment overrides.  The
explicit :class:`GatePolicy` argument on ``validate_model_layout`` exists only
so the filesystem checks can be exercised with small unit-test fixtures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any


@dataclasses.dataclass(frozen=True)
class GatePolicy:
    model_link: Path
    model_target: Path
    status_path: Path
    repo_id: str
    revision: str
    manifest_sha256: str
    expected_file_count: int
    expected_shard_count: int
    expected_total_bytes: int
    config_sha256: str
    index_sha256: str


PRODUCTION_POLICY = GatePolicy(
    model_link=Path("/workspace/Models/DeepSeek-V4-Pro-0813"),
    model_target=Path("/data/temp/Models/DeepSeek-V4-Pro-0813"),
    status_path=Path("/data/temp/Models/pro0813_download_status.json"),
    repo_id="deepseek-ai/DeepSeek-V4-Pro-0813",
    revision="72e1d3230f6c080a530b0a1d46f8eb4602340597",
    manifest_sha256=(
        "27c3ef953c3baeb1a69f8ba0c6fd8c55814b1c89a4aa768ff28486c27bebb1f5"
    ),
    expected_file_count=92,
    expected_shard_count=66,
    expected_total_bytes=892_762_497_859,
    config_sha256=(
        "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0"
    ),
    index_sha256=(
        "2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a"
    ),
)

MANIFEST_NAME = ".redknot_official_manifest.json"
CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_STATUS_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class OfficialModelGateError(RuntimeError):
    """Base class whose subclasses are safe to expose by type name."""


class ModelLinkError(OfficialModelGateError):
    pass


class ManifestFileError(OfficialModelGateError):
    pass


class ManifestHashError(OfficialModelGateError):
    pass


class ManifestSchemaError(OfficialModelGateError):
    pass


class UnsafeManifestPathError(OfficialModelGateError):
    pass


class ModelFileError(OfficialModelGateError):
    pass


class ModelFileSizeError(OfficialModelGateError):
    pass


class ConfigHashError(OfficialModelGateError):
    pass


class IndexHashError(OfficialModelGateError):
    pass


class StatusFileError(OfficialModelGateError):
    pass


class StatusSchemaError(OfficialModelGateError):
    pass


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ModelFileError("a pinned small file is not a regular file")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened_metadata = os.fstat(handle.fileno())
            if (
                opened_metadata.st_dev != metadata.st_dev
                or opened_metadata.st_ino != metadata.st_ino
            ):
                raise ModelFileError("a pinned small file changed before reading")
            while chunk := handle.read(4 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ModelFileError("a pinned small file cannot be read") from error
    return digest.hexdigest()


def _read_small_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    error_type: type[OfficialModelGateError],
) -> bytes:
    """Read a bounded regular file without following its final symlink."""

    try:
        path_metadata = path.lstat()
    except OSError as error:
        raise error_type("required metadata file is unavailable") from error
    if not stat.S_ISREG(path_metadata.st_mode) or stat.S_ISLNK(
        path_metadata.st_mode
    ):
        raise error_type("required metadata path is not a regular file")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise error_type("required metadata file is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise error_type("required metadata path is not a regular file")
        if (
            metadata.st_dev != path_metadata.st_dev
            or metadata.st_ino != path_metadata.st_ino
        ):
            raise error_type("required metadata file changed before reading")
        if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
            raise error_type("required metadata file size is invalid")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise error_type("required metadata file changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise error_type("required metadata file changed while reading")
        return b"".join(chunks)
    except OSError as error:
        raise error_type("required metadata file cannot be read") from error
    finally:
        os.close(descriptor)


def _decode_json_object(
    data: bytes,
    *,
    error_type: type[OfficialModelGateError],
) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise error_type("metadata JSON cannot be decoded") from error
    if not isinstance(payload, dict):
        raise error_type("metadata JSON root must be an object")
    return payload


def _validate_model_link(policy: GatePolicy) -> None:
    try:
        link_metadata = policy.model_link.lstat()
    except OSError as error:
        raise ModelLinkError("pinned model link is unavailable") from error
    if not stat.S_ISLNK(link_metadata.st_mode):
        raise ModelLinkError("pinned model path must itself be a symlink")

    try:
        resolved = policy.model_link.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ModelLinkError("pinned model link cannot be resolved") from error
    if resolved != policy.model_target:
        raise ModelLinkError("pinned model link resolves to a different target")

    try:
        target_metadata = policy.model_target.lstat()
    except OSError as error:
        raise ModelLinkError("pinned model target is unavailable") from error
    if not stat.S_ISDIR(target_metadata.st_mode) or stat.S_ISLNK(target_metadata.st_mode):
        raise ModelLinkError("pinned model target must be a real directory")


def _safe_relative_parts(name: object) -> tuple[str, ...]:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise UnsafeManifestPathError("manifest contains an unsafe path")
    relative = PurePosixPath(name)
    parts = relative.parts
    if (
        relative.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or relative.as_posix() != name
    ):
        raise UnsafeManifestPathError("manifest contains an unsafe path")
    return parts


def _regular_target_size(model_target: Path, parts: tuple[str, ...]) -> int:
    cursor = model_target
    for component in parts[:-1]:
        cursor = cursor / component
        try:
            metadata = cursor.lstat()
        except OSError as error:
            raise ModelFileError("a manifest target parent is unavailable") from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ModelFileError("a manifest target parent is not a real directory")

    target = cursor / parts[-1]
    try:
        metadata = target.lstat()
    except OSError as error:
        raise ModelFileError("a manifest target is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ModelFileError("a manifest target is not a regular file")
    return metadata.st_size


def _validate_manifest(policy: GatePolicy) -> dict[str, dict[str, object]]:
    manifest_path = policy.model_target / MANIFEST_NAME
    raw = _read_small_regular_file(
        manifest_path,
        maximum_bytes=MAX_MANIFEST_BYTES,
        error_type=ManifestFileError,
    )
    if _sha256_bytes(raw) != policy.manifest_sha256:
        raise ManifestHashError("official manifest hash differs from the pin")
    payload = _decode_json_object(raw, error_type=ManifestSchemaError)

    if payload.get("repo_id") != policy.repo_id:
        raise ManifestSchemaError("official manifest repository differs from the pin")
    if payload.get("revision") != policy.revision:
        raise ManifestSchemaError("official manifest revision differs from the pin")
    file_count = payload.get("file_count")
    if (
        not _is_plain_int(file_count)
        or file_count != policy.expected_file_count
    ):
        raise ManifestSchemaError("official manifest file count differs from the pin")
    total_bytes = payload.get("total_bytes")
    if (
        not _is_plain_int(total_bytes)
        or total_bytes != policy.expected_total_bytes
    ):
        raise ManifestSchemaError("official manifest total size differs from the pin")

    files = payload.get("files")
    if not isinstance(files, dict) or len(files) != policy.expected_file_count:
        raise ManifestSchemaError("official manifest files differ from the pin")

    observed_total = 0
    observed_shards: set[str] = set()
    validated: dict[str, dict[str, object]] = {}
    for name, raw_metadata in files.items():
        parts = _safe_relative_parts(name)
        if not isinstance(raw_metadata, dict):
            raise ManifestSchemaError("official manifest has malformed file metadata")
        size = raw_metadata.get("size")
        digest = raw_metadata.get("sha256")
        if not _is_plain_int(size) or size < 0:
            raise ManifestSchemaError("official manifest has an invalid file size")
        if digest is not None and (
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
        ):
            raise ManifestSchemaError("official manifest has an invalid file hash")

        actual_size = _regular_target_size(policy.model_target, parts)
        if actual_size != size:
            raise ModelFileSizeError("a manifest target has a different size")
        observed_total += size
        if name.endswith(".safetensors"):
            observed_shards.add(name)
        validated[name] = raw_metadata

    expected_shards = {
        f"model-{index:05d}-of-{policy.expected_shard_count:05d}.safetensors"
        for index in range(1, policy.expected_shard_count + 1)
    }
    if observed_shards != expected_shards:
        raise ManifestSchemaError("official manifest shard set differs from the pin")
    if observed_total != policy.expected_total_bytes:
        raise ManifestSchemaError("official manifest file sizes differ from the pin")
    if CONFIG_NAME not in validated or INDEX_NAME not in validated:
        raise ManifestSchemaError("official manifest lacks a pinned metadata file")
    return validated


def _validate_small_hashes(policy: GatePolicy) -> None:
    if _sha256_file(policy.model_target / CONFIG_NAME) != policy.config_sha256:
        raise ConfigHashError("official config hash differs from the pin")
    if _sha256_file(policy.model_target / INDEX_NAME) != policy.index_sha256:
        raise IndexHashError("official index hash differs from the pin")


def _validate_status(policy: GatePolicy) -> None:
    raw = _read_small_regular_file(
        policy.status_path,
        maximum_bytes=MAX_STATUS_BYTES,
        error_type=StatusFileError,
    )
    payload = _decode_json_object(raw, error_type=StatusSchemaError)
    if (
        payload.get("status") != "complete"
        or payload.get("revision") != policy.revision
        or not _is_plain_int(payload.get("file_count"))
        or payload.get("file_count") != policy.expected_file_count
        or not _is_plain_int(payload.get("total_bytes"))
        or payload.get("total_bytes") != policy.expected_total_bytes
        or payload.get("path") != str(policy.model_target)
    ):
        raise StatusSchemaError("download status differs from the completed pin")


def validate_model_layout(policy: GatePolicy) -> None:
    """Validate a model tree against one explicit immutable policy."""

    _validate_model_link(policy)
    _validate_manifest(policy)
    _validate_small_hashes(policy)
    _validate_status(policy)


def main() -> int:
    """Run the non-configurable production gate and emit one safe JSON line."""

    try:
        validate_model_layout(PRODUCTION_POLICY)
    except Exception as error:  # The message may contain paths; expose only its type.
        print(
            json.dumps(
                {"error_type": type(error).__name__, "status": "fail"},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 1
    print('{"status":"pass"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
