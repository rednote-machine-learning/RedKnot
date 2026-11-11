"""Clone six frozen 8x32K pilot inputs for an independent sensitivity run.

No retokenization, sample selection, model import, GPU query, or launch occurs.
The old manifest is retained byte-for-byte as parent-manifest.json; only the
new manifest gains the parent and sensitivity protocol bindings. Other source
files retain their relative paths and exact bytes, including the numeric pilot
protocol. Existing source path fields remain provenance, without rewriting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from gpu_pilot import launch_pilot as launch
from gpu_pilot import pilot_protocol as pp
from gpu_pilot.run_pilot import validate_inputs
from gpu_pilot.sensitivity_protocol import sensitivity_protocol


SCHEMA = "kv-cache-engram-sensitivity-input-preparation-v1"
RESERVED = {"parent-manifest.json", "sensitivity-protocol.json", "preparation.json"}


class _ManifestChecks:
    """Standard-library-only input contract used by the native CPU preflight.

    Keeping this small adapter local avoids importing executable code from an
    input directory. validate_inputs additionally enforces the original six
    cases, source exclusions, split separation, and exact 8x32768 geometry.
    """

    @staticmethod
    def validate_manifest(manifest):
        pp._sealed(manifest)
        launch.require(manifest.get("schema") == "redknot-deepinfer-real-rag-manifest-v1",
                       "unsupported independent pilot manifest schema")
        launch.require(type(manifest.get("warmup")) is list and type(manifest.get("cohort")) is list,
                       "explicit calibration and evaluation lists required")
        seen = set()
        for row in manifest["warmup"] + manifest["cohort"]:
            launch.require(type(row) is dict, "input row must be a mapping")
            sid, ids = row.get("sample_id"), row.get("input_ids")
            launch.require(type(sid) is str and bool(sid) and sid not in seen,
                           "duplicate or missing sample identity")
            seen.add(sid)
            launch.require(type(ids) is list and bool(ids)
                and all(type(token) is int and 0 <= token < 2**31 for token in ids)
                and len(ids) == row.get("input_tokens") and launch.digest(ids) == row.get("input_ids_sha256"),
                "input token/hash mismatch")
            prompt, references = row.get("prompt"), row.get("references")
            launch.require(type(prompt) is str and hashlib.sha256(prompt.encode()).hexdigest() == row.get("prompt_sha256"),
                           "prompt/hash mismatch")
            launch.require(type(references) is list and bool(references)
                and all(type(value) is str for value in references)
                and launch.digest(references) == row.get("references_sha256"), "gold/hash mismatch")

    @staticmethod
    def document_specs(row):
        docs = row.get("documents")
        launch.require(type(docs) is list and bool(docs), "frozen document spans required")
        result, previous = [], 0
        for doc in docs:
            start, end = doc.get("input_start"), doc.get("input_end")
            launch.require(type(start) is int and type(end) is int
                and previous <= start < end <= len(row["input_ids"])
                and doc.get("offline_position") == 0 and doc.get("online_position") == start
                and doc.get("tokens") == end-start, "invalid frozen document span")
            ids = row["input_ids"][start:end]
            digest = launch.digest(ids)
            launch.require(doc.get("token_ids_sha256") == digest and doc.get("document_id") == digest,
                           "document token/hash mismatch")
            result.append(dict(token_ids=ids, input_start=start, token_ids_sha256=digest))
            previous = end
        return result


def _inventory(directory):
    """No symlink traversal, special files, or silent file changes."""
    files, directories = {}, []
    for path in sorted(directory.rglob("*")):
        launch.absolute_no_symlink(path)
        relative = str(path.relative_to(directory))
        if path.is_dir():
            directories.append(relative)
        else:
            launch.require(path.is_file(), "input tree contains a non-regular file: " + relative)
            files[relative] = launch.file_digest(path)
    return files, directories


def _copy_exclusive(source, target, expected_digest):
    checksum = hashlib.sha256()
    launch.absolute_no_symlink(source)
    launch.absolute_no_symlink(target)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1 << 20), b""):
            checksum.update(block)
            outgoing.write(block)
    launch.require(checksum.hexdigest() == expected_digest,
                   "source changed while copying; partial output retained: " + str(source))


def _write_exclusive(path, value):
    with path.open("xb") as stream:
        stream.write(launch.canonical(value) + b"\n")


def prepare(source_inputs, output):
    source, output = launch.absolute_no_symlink(source_inputs), launch.absolute_no_symlink(output)
    launch.require(source.is_dir(), "source input directory does not exist")
    launch.require(source != output and source not in output.parents and output not in source.parents,
                   "source and output directories must be independent, not nested")
    launch.require(not output.exists(), "output already exists; never overwrite or retry in place")
    launch.require(output.parent.is_dir(), "output parent must already exist")
    files, directories = _inventory(source)
    launch.require({"manifest.json", "protocol.json", "source-plan.json"} <= set(files),
                   "source requires manifest.json, protocol.json and source-plan.json")
    launch.require(not (RESERVED & set(files)) and not (RESERVED & set(directories)),
                   "source already contains sensitivity/provenance output names")
    manifest = launch.read_sealed(source / "manifest.json")
    protocol = launch.read_sealed(source / "protocol.json", launch.PROTOCOL_SCHEMA)
    plan = launch.read_sealed(source / "source-plan.json")
    launch.require("sensitivity_protocol_sha256" not in manifest and "parent_manifest_sha256" not in manifest,
                   "source must be the original, unextended six-case pilot manifest")
    launch.require(manifest.get("selection_plan") == plan, "source plan and manifest differ")
    rows = validate_inputs(_ManifestChecks, manifest, protocol)
    extension = sensitivity_protocol()
    new_manifest = pp.seal_record(dict(
        {key: value for key, value in manifest.items() if key != "sha256"},
        parent_manifest_sha256=manifest["sha256"], sensitivity_protocol_sha256=extension["sha256"]))
    validate_inputs(_ManifestChecks, new_manifest, protocol)
    launch.require(_inventory(source) == (files, directories), "source changed during validation")

    # Never delete evidence or reuse a partial destination on failure. A finished
    # preparation.json is published last; its absence means incomplete output.
    output.mkdir(exist_ok=False)
    for relative in directories:
        (output / relative).mkdir()
    for relative, checksum in files.items():
        target = output / ("parent-manifest.json" if relative == "manifest.json" else relative)
        _copy_exclusive(source / relative, target, checksum)
    launch.require(_inventory(source) == (files, directories), "source changed while cloning")
    _write_exclusive(output / "sensitivity-protocol.json", extension)
    _write_exclusive(output / "manifest.json", new_manifest)
    output_files, _ = _inventory(output)
    report = pp.seal_record(dict(schema=SCHEMA, complete=True, gpu_initialized=False,
        model_imported=False, tokenizer_called=False, samples_reselected=False,
        source_inputs=str(source), output=str(output), source_files_sha256=files,
        output_files_sha256=output_files, parent_manifest_sha256=manifest["sha256"],
        parent_manifest_file_sha256=files["manifest.json"], manifest_sha256=new_manifest["sha256"],
        protocol_sha256=protocol["sha256"], sensitivity_protocol_sha256=extension["sha256"],
        source_paths_unchanged=True, calibration_cases=2, evaluation_cases=4,
        context_tokens=262144, num_chunks=8, chunk_tokens=32768,
        rows=[dict(index=row["index"], dataset=row["dataset"], split=row["split"],
                   input_tokens=row["input_tokens"], input_ids_sha256=row["input_ids_sha256"])
              for row in rows]))
    _write_exclusive(output / "preparation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare(args.source_inputs, args.output)
    print(json.dumps(dict(status="prepared", gpu_initialized=False, output=report["output"],
        parent_manifest_sha256=report["parent_manifest_sha256"],
        manifest_sha256=report["manifest_sha256"], protocol_sha256=report["protocol_sha256"],
        sensitivity_protocol_sha256=report["sensitivity_protocol_sha256"],
        preparation_sha256=report["sha256"], calibration_cases=2, evaluation_cases=4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
