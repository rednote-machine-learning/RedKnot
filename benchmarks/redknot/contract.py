"""CPU-only validation for immutable benchmark inputs.

The input IDs in a sealed manifest are authoritative. A runtime must never
decode and re-tokenize them, or silently reuse IDs from another tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA = "redknot-deepinfer-benchmark-v1"
BOUNDARY = "native_engine_enqueue_to_native_token_observation"
DATASETS = frozenset({"hotpotqa", "2wikimqa", "musique", "multifieldqa_en", "triviaqa"})


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def file_digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seal(value: dict) -> dict:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    return {**payload, "sha256": digest(payload)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), "manifest contains non-finite numbers")
    elif isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)


def validate_manifest(manifest: dict) -> dict:
    _require(isinstance(manifest, dict), "manifest must be an object")
    _reject_nonfinite(manifest)
    _require(manifest.get("schema") == SCHEMA, "unsupported benchmark manifest schema")
    _require(manifest.get("sha256") == seal(manifest)["sha256"], "manifest SHA-256 mismatch")
    model = manifest.get("model")
    _require(isinstance(model, dict), "manifest requires model provenance")
    _require(bool(model.get("profile")) and bool(model.get("model_id")), "model profile and model_id are required")
    tokenizer = model.get("tokenizer")
    _require(isinstance(tokenizer, dict) and bool(tokenizer.get("fingerprint")), "tokenizer fingerprint is required")
    _require(isinstance(manifest.get("suite"), dict), "suite provenance is required")
    _require(isinstance(manifest.get("cases"), list) and bool(manifest["cases"]), "cases must be a nonempty list")
    _require(isinstance(manifest.get("warmup"), list), "warmup must be an explicit list")

    ids: set[str] = set()
    input_sets: dict[str, set[str]] = {"warmup": set(), "cases": set()}
    for role in ("warmup", "cases"):
        for case in manifest[role]:
            _require(isinstance(case, dict), "each case must be an object")
            sample_id = case.get("sample_id")
            _require(isinstance(sample_id, str) and bool(sample_id), "sample_id must be a nonempty string")
            _require(sample_id not in ids, f"duplicate sample_id: {sample_id}")
            ids.add(sample_id)
            _require(case.get("dataset") in DATASETS, f"unsupported dataset for English QA metrics: {case.get('dataset')}")
            tokens = case.get("input_ids")
            _require(isinstance(tokens, list) and bool(tokens), f"{sample_id}: input_ids must be nonempty")
            _require(all(type(token) is int and 0 <= token < 2**31 for token in tokens), f"{sample_id}: invalid token IDs")
            token_hash = digest(tokens)
            _require(case.get("input_ids_sha256") == token_hash, f"{sample_id}: input token hash mismatch")
            input_sets[role].add(token_hash)
            if "input_tokens" in case:
                _require(case["input_tokens"] == len(tokens), f"{sample_id}: input length mismatch")
            references = case.get("references")
            _require(
                isinstance(references, list) and bool(references) and all(isinstance(text, str) and text.strip() for text in references),
                f"{sample_id}: nonempty reference answers required",
            )
            _require(case.get("references_sha256") == digest(references), f"{sample_id}: reference hash mismatch")
            _require(type(case.get("max_new_tokens")) is int and case["max_new_tokens"] > 0, f"{sample_id}: invalid output budget")
            _require(type(case.get("scored")) is bool, f"{sample_id}: scored must be boolean")
            _require(type(case.get("ignore_eos")) is bool, f"{sample_id}: ignore_eos must be boolean")
            _require(not (case["scored"] and case["ignore_eos"]), f"{sample_id}: fixed-length showcase must not enter QA accuracy")
            if "prompt_sha256" in case:
                _require(isinstance(case.get("prompt"), str), f"{sample_id}: prompt text missing")
                actual = hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest()
                _require(case["prompt_sha256"] == actual, f"{sample_id}: prompt hash mismatch")
    _require(not input_sets["warmup"].intersection(input_sets["cases"]), "warmup inputs overlap measured inputs")
    return manifest


def read_manifest(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return validate_manifest(json.load(stream))


def write_json(path: str | Path, value: dict) -> Path:
    """Never overwrite a frozen input or a previous result."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return destination


def tokenizer_identity(path: str | Path, *, revision: str | None = None) -> dict:
    """Fingerprint local tokenizer assets, never checkpoint weights or secrets."""
    root = Path(path).expanduser().resolve()
    _require(root.is_dir(), "tokenizer must be a local checkpoint/tokenizer directory")
    files = {
        item.name: file_digest(item)
        for item in sorted(root.iterdir())
        if item.is_file()
        and (
            item.name.startswith(("tokenizer", "tokenization_"))
            or item.name in {"special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "vocab.txt", "chat_template.jinja"}
        )
    }
    files.update(
        {str(item.relative_to(root)): file_digest(item) for item in sorted((root / "chat_templates").glob("*.jinja")) if item.is_file()}
    )
    _require(bool(files), f"no tokenizer assets found in {root}")
    if revision is not None:
        _require(bool(re.fullmatch(r"[0-9a-fA-F]{7,64}", revision)), "revision must be an immutable commit identifier")
    return {"name_or_path": str(root), "revision": revision, "files": files, "fingerprint": digest(files)}
