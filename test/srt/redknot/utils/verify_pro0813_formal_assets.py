#!/usr/bin/env python3
"""Fail-closed CPU preflight for every formal Pro-0813 sweep asset.

The short targets already publish a canonical ``selection_sha256`` in the
Pro HTTP entry point.  This verifier reuses that identity and proves that the
selected manifest still names the exact on-disk MuSiQue JSONL bytes.  The two
long targets delegate to ``verify_pro0813_qualification_profile.py``; that
existing verifier closes over the profile, data/prompt manifests, HotpotQA
JSONL, official tokenizer files, and tokenizer runtime.

No GPU or holder process is inspected by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REDKNOT_ROOT = HERE.parent
HTTP_ENTRY = HERE / "benchmark_dsv4_pro0813_redknot_http.py"
PROFILE_VERIFIER = HERE / "verify_pro0813_qualification_profile.py"
MODEL_ROOT = Path("/workspace/Models/DeepSeek-V4-Pro-0813")
DATA_DIR = REDKNOT_ROOT / "datasets/LongBench/data"
SHORT_TARGETS = (65536, 131072, 262144)
LONG_TARGETS = (450560, 524288)
ALL_TARGETS = SHORT_TARGETS + LONG_TARGETS
CANONICAL_SHORT_TARGETS = {
    65536: {
        "manifest": HERE / "musique_pure_prompt_selection_v1.json",
        "selection_sha256": (
            "586fd683bfe043e1a6aaa1d07c7236ea9d956d99be739be743c4a2ec1728bcd8"
        ),
        "chunk_tokens": 8192,
        "num_chunks": 8,
        "query_row_id": 68,
    },
    131072: {
        "manifest": HERE / "musique_pure_prompt_selection_128k_v1.json",
        "selection_sha256": (
            "caf99890880e0de190f845d0a38e600d760d2153cd1961888bd7776a2044f040"
        ),
        "chunk_tokens": 8192,
        "num_chunks": 16,
        "query_row_id": 68,
    },
    262144: {
        "manifest": HERE / "musique_pure_prompt_selection_256k_32k_v1.json",
        "selection_sha256": (
            "a2524b87a6ff0a91e7f5aef104d3b8eb14b9aa55e2b8c6b5db34ef0dbe1477cc"
        ),
        "chunk_tokens": 32768,
        "num_chunks": 8,
        "query_row_id": 0,
    },
}
CANONICAL_LONG_PROFILES = {
    450560: (
        REDKNOT_ROOT
        / "qualification_profiles/pro0813_440k_hotpotqa_10q/profile.json",
        "52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b",
    ),
    524288: (
        REDKNOT_ROOT
        / "qualification_profiles/pro0813_512k_hotpotqa_10q/profile.json",
        "e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d",
    ),
}
HEX64 = re.compile(r"[0-9a-f]{64}")


class FormalAssetVerificationError(ValueError):
    """A formal input is absent, mutable by indirection, or has drifted."""


def _fail(message: str) -> None:
    raise FormalAssetVerificationError(message)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load formal asset dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        _fail(f"cannot read {label}: {path}: {error}")
    try:
        document = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates
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
    except OSError as error:
        _fail(f"cannot inspect {label}: {path}: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a non-symlink regular file: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        _fail(f"cannot hash formal asset {path}: {error}")
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _verify_short_target_asset(
    target_tokens: int,
    target_profile: dict[str, Any],
    *,
    data_dir: Path,
) -> dict[str, Any]:
    """Verify one digest-frozen selection and its whole raw JSONL source."""

    if target_tokens not in SHORT_TARGETS:
        _fail(f"not a formal short target: {target_tokens}")
    expected_digest = target_profile.get("selection_sha256")
    if not isinstance(expected_digest, str) or HEX64.fullmatch(expected_digest) is None:
        _fail(f"short target {target_tokens} has no pinned selection digest")
    manifest_path = target_profile.get("data_manifest")
    if not isinstance(manifest_path, Path) or not manifest_path.is_absolute():
        _fail(f"short target {target_tokens} has an invalid manifest path")
    _require_regular_nonsymlink(manifest_path, "selection manifest")
    document = _load_json_object(manifest_path, "selection manifest")
    if document.get("format") != "redknot_indexer_hot_data_selection_v1":
        _fail(f"short target {target_tokens} selection format differs")
    claimed_digest = document.get("selection_sha256")
    if claimed_digest != expected_digest:
        _fail(
            f"short target {target_tokens} selection identity differs: "
            f"expected={expected_digest} observed={claimed_digest}"
        )
    canonical = dict(document)
    canonical.pop("selection_sha256", None)
    observed_digest = _canonical_sha256(canonical)
    if observed_digest != expected_digest:
        _fail(
            f"short target {target_tokens} selection canonical digest mismatch: "
            f"expected={expected_digest} observed={observed_digest}"
        )

    expected_geometry = {
        "chunk_tokens": int(target_profile["chunk_tokens"]),
        "num_chunks": int(target_profile["num_chunks"]),
        "num_queries": 1,
    }
    if document.get("geometry") != expected_geometry:
        _fail(f"short target {target_tokens} selection geometry differs")
    dataset = document.get("dataset")
    if (
        not isinstance(dataset, dict)
        or set(dataset) != {"name", "bytes", "sha256", "row_id_base"}
        or dataset.get("name") != "musique"
        or dataset.get("row_id_base") != 0
        or type(dataset.get("bytes")) is not int
        or dataset["bytes"] <= 0
        or not isinstance(dataset.get("sha256"), str)
        or HEX64.fullmatch(dataset["sha256"]) is None
    ):
        _fail(f"short target {target_tokens} dataset identity is malformed")
    selection = document.get("selection")
    queries = selection.get("queries") if isinstance(selection, dict) else None
    if (
        not isinstance(queries, list)
        or len(queries) != 1
        or not isinstance(queries[0], dict)
        or queries[0].get("row_id") != int(target_profile["query_row_id"])
    ):
        _fail(f"short target {target_tokens} frozen query identity differs")

    dataset_path = data_dir / "musique.jsonl"
    _require_regular_nonsymlink(dataset_path, "MuSiQue LongBench JSONL")
    observed_bytes = dataset_path.stat().st_size
    observed_dataset_sha256 = _sha256_file(dataset_path)
    if (
        observed_bytes != dataset["bytes"]
        or observed_dataset_sha256 != dataset["sha256"]
    ):
        _fail(
            "MuSiQue LongBench JSONL byte identity differs: "
            f"target={target_tokens} expected_bytes={dataset['bytes']} "
            f"observed_bytes={observed_bytes} expected_sha256={dataset['sha256']} "
            f"observed_sha256={observed_dataset_sha256}"
        )
    return {
        "kind": "builtin_selection",
        "target_tokens": target_tokens,
        "manifest": str(manifest_path),
        "selection_sha256": expected_digest,
        "dataset": "musique",
        "dataset_path": str(dataset_path),
        "dataset_bytes": observed_bytes,
        "dataset_sha256": observed_dataset_sha256,
        "pass": True,
    }


def verify_formal_assets(
    targets: Iterable[int] = ALL_TARGETS,
    *,
    data_dir: Path = DATA_DIR,
    model_root: Path = MODEL_ROOT,
) -> dict[str, Any]:
    """Verify the requested formal targets without importing torch or CUDA."""

    requested = tuple(targets)
    if not requested or len(set(requested)) != len(requested):
        _fail("formal target list must be non-empty and unique")
    unknown = set(requested).difference(ALL_TARGETS)
    if unknown:
        _fail(f"unsupported formal targets: {sorted(unknown)}")
    http_entry = _load_module(
        HTTP_ENTRY, "_redknot_pro0813_formal_asset_http_entry"
    )
    profile_verifier = _load_module(
        PROFILE_VERIFIER, "_redknot_pro0813_formal_asset_profile_verifier"
    )
    records: list[dict[str, Any]] = []
    for target_tokens in requested:
        if target_tokens in SHORT_TARGETS:
            target_profile = http_entry.REDKNOT_TARGET_PROFILES.get(target_tokens)
            if not isinstance(target_profile, dict):
                _fail(f"HTTP entry omits formal target {target_tokens}")
            canonical = CANONICAL_SHORT_TARGETS[target_tokens]
            observed_identity = {
                "manifest": target_profile.get("data_manifest"),
                "selection_sha256": target_profile.get("selection_sha256"),
                "chunk_tokens": target_profile.get("chunk_tokens"),
                "num_chunks": target_profile.get("num_chunks"),
                "query_row_id": target_profile.get("query_row_id"),
            }
            if observed_identity != canonical:
                _fail(
                    f"HTTP entry short-target identity drifted for {target_tokens}"
                )
            records.append(
                _verify_short_target_asset(
                    target_tokens, target_profile, data_dir=data_dir
                )
            )
            continue
        profile_path, expected_profile_sha256 = CANONICAL_LONG_PROFILES[
            target_tokens
        ]
        verification = profile_verifier.verify_profile(
            profile_path,
            expected_target_tokens=target_tokens,
            model_root=model_root,
            data_dir=data_dir,
            expected_profile_sha256=expected_profile_sha256,
        )
        records.append(
            {
                "kind": "qualification_profile",
                **verification,
            }
        )
    return {
        "format": "redknot_pro0813_formal_asset_preflight_v1",
        "targets": list(requested),
        "assets": records,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-tokens",
        action="append",
        type=int,
        choices=ALL_TARGETS,
        dest="targets",
        help="Verify only this target; repeat as needed. Default verifies all five.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    args = parser.parse_args()
    result = verify_formal_assets(
        args.targets or ALL_TARGETS,
        data_dir=args.data_dir,
        model_root=args.model_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
