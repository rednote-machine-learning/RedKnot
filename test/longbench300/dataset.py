"""Read the frozen LongBench 300-task data package for the Full / S1.6 comparison.

prepare_data() returns the data directory; its sibling sources/ contains the original
contexts used to construct the prompts. verify_dataset() returns a JSON-ready audit.
The cache is content-addressed; use a user-writable local directory on POSIX.
"""
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path


ARCHIVE_SHA256 = "ea528890ab935dbc6a225ab4a7c1cf54b8654cc6d74f0cb79982eabe7401b0d0"
ARCHIVE_SIZE = 50_461_689
DATASETS = ("hotpotqa", "2wikimqa", "musique")
_DATA_DIGEST = "25bfffc6da5c71212383a14b1132b7a10d33b5c1d018a10c79480bfb6c9b08af"
_PACKAGE_DIGEST = "4eb1b4d4aede503a18577605d0682be8d6178419fc7411b5fdd88670c4f99db4"
_LOGICAL_DIGEST = "e7fe572b19eb2428121267b5f8852029a45823eccaf271dbc88064da459115ec"

_CASE_IDS = tuple(n * 1000 + slot for n in range(1, 4) for slot in range(100))
_GROUPS = tuple(f"lb-{n:04d}" for n in range(75))
_DATA_FILES = frozenset(
    ["complete.json"] + [f"groups/{g}.json" for g in _GROUPS]
    + [f"index/{i}.json" for i in _CASE_IDS] + [f"records/{i}.json.gz" for i in _CASE_IDS]
)
_PACKAGE_FILES = frozenset(
    {f"data/{name}" for name in _DATA_FILES}
    | {f"sources/{ds}.jsonl" for ds in DATASETS}
    | {"NOTICE.txt", "provenance.json", "sources/prompts.json", "licenses/LongBench-MIT.txt",
       "licenses/Apache-2.0.txt", "licenses/CC-BY-4.0.txt", "licenses/CC-BY-SA-4.0.txt"}
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _verify_tree(root, expected_names, expected_digest):
    root = Path(root)
    _require(stat.S_ISDIR(root.lstat().st_mode), f"Expected a plain directory: {root}")
    expected_dirs = {str(p) for name in expected_names for p in Path(name).parents if str(p) != "."}
    inventory = {}
    for current, directories, files in os.walk(root):
        folder = Path(current)
        for name in directories:
            path = folder / name
            _require(path.relative_to(root).as_posix() in expected_dirs and stat.S_ISDIR(path.lstat().st_mode),
                     f"Unexpected cache directory: {path}")
        for name in files:
            path = folder / name
            relative = path.relative_to(root).as_posix()
            _require(relative in expected_names and stat.S_ISREG(path.lstat().st_mode),
                     f"Unexpected cached file: {path}")
            with path.open("rb") as stream:
                inventory[relative] = _digest_stream(stream)
    _require(inventory.keys() == expected_names, f"Incomplete cached dataset: {root}")
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    _require(hashlib.sha256(encoded).hexdigest() == expected_digest, f"Cached file contents changed: {root}")


def _read(path):
    return json.loads(path.read_bytes())


def verify_dataset(data):
    """Check the exact published file bytes and summarize the original B4 requests."""
    data = Path(data)
    _verify_tree(data, _DATA_FILES, _DATA_DIGEST)
    terminal = _read(data / "complete.json")
    _require(terminal["group_count"] == 75 and terminal["target_count"] == 300, "Unexpected cohort size")
    by_dataset = {}
    rows = {}
    bands = Counter()
    modes = Counter()
    tokens = []
    chunks = []
    for ordinal, ds in enumerate(DATASETS, 1):
        ds_rows = [_read(data / f"index/{i}.json") for i in range(ordinal * 1000, ordinal * 1000 + 100)]
        _require(all(row["dataset"] == ds for row in ds_rows), f"Wrong source dataset: {ds}")
        rows.update((row["index"], row) for row in ds_rows)
        lengths = [row["input_tokens"] for row in ds_rows]
        sizes = [row["actual_document_count"] for row in ds_rows]
        band_counts = Counter(row["length_band"] for row in ds_rows)
        mode_counts = Counter(row["composition_mode"] for row in ds_rows)
        bands.update(band_counts)
        modes.update(mode_counts)
        tokens.extend(lengths)
        chunks.extend(sizes)
        by_dataset[ds] = {
            "targets": len(ds_rows), "total_input_tokens": sum(lengths),
            "input_tokens_range": [min(lengths), max(lengths)],
            "chunk_count_range": [min(sizes), max(sizes)], "length_bands": dict(band_counts),
            "composition_modes": {k: mode_counts[k] for k in ("native", "extended")},
        }
    seen = set()
    mixed = []
    for gid in _GROUPS:
        group = _read(data / f"groups/{gid}.json")
        indices = group["indices"]
        _require(group["group_id"] == gid and len(indices) == len(set(indices)) == 4 and
                 indices == group["target_indices"] == [case["index"] for case in group["cases"]] and
                 not seen.intersection(indices), f"Invalid B4 membership: {gid}")
        _require(group["native_chunk_counts"] == [rows[i]["frames"] for i in indices] and
                 len(set(group["native_chunk_counts"])) == 1, f"Invalid native frame counts: {gid}")
        seen.update(indices)
        if len({rows[i]["dataset"] for i in indices}) > 1:
            mixed.append(gid)
    _require(seen == set(_CASE_IDS), "Wrong target IDs")
    return {
        "target_count": len(seen), "group_count": len(_GROUPS), "batch_size": 4,
        "dataset_counts": {ds: by_dataset[ds]["targets"] for ds in DATASETS}, "by_dataset": by_dataset,
        "total_input_tokens": sum(tokens), "input_tokens_range": [min(tokens), max(tokens)],
        "chunk_count": sum(chunks), "chunk_count_range": [min(chunks), max(chunks)],
        "composition_modes": dict(modes), "length_bands": dict(bands), "mixed_dataset_groups": mixed,
        "logical_sha256": _LOGICAL_DIGEST, "physical_manifest_sha256": _DATA_DIGEST,
    }


def _unpack(stream, destination):
    with zipfile.ZipFile(stream) as bundle:
        members = bundle.infolist()
        _require(len(members) == len(_PACKAGE_FILES) and {m.filename for m in members} == _PACKAGE_FILES,
                 "Unexpected or duplicate dataset archive member")
        _require(sum(m.file_size for m in members) == 70_156_169, "Unexpected uncompressed dataset size")
        for member in members:
            _require(member.orig_filename == member.filename and stat.S_ISREG(member.external_attr >> 16),
                     f"Unexpected archive member type: {member.filename}")
            path = destination / member.filename  # Every name belongs to the fixed allowlist above.
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst)


def prepare_data(archive: Path, cache: Path) -> Path:
    """Verify and atomically cache the bundle, or revalidate an already extracted copy."""
    archive, cache = Path(archive), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    name = "longbench-300-" + ARCHIVE_SHA256[:16]
    ready = cache / name
    # Callers sharing a cache wait on the same inode. The lock lives outside the
    # published directory and the temporary build is on the same filesystem.
    lock_fd = os.open(cache / ("." + name + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with archive.open("rb") as stream:
            _require(os.fstat(stream.fileno()).st_size == ARCHIVE_SIZE and _digest_stream(stream) == ARCHIVE_SHA256,
                     f"Dataset archive size or SHA-256 differs from the frozen release: {archive}")
            if os.path.lexists(ready):
                _verify_tree(ready, _PACKAGE_FILES, _PACKAGE_DIGEST)
            else:
                stream.seek(0)
                with tempfile.TemporaryDirectory(dir=cache, prefix="." + name + ".build-") as staging:
                    unpacked = Path(staging) / "release"
                    unpacked.mkdir()
                    _unpack(stream, unpacked)
                    _verify_tree(unpacked, _PACKAGE_FILES, _PACKAGE_DIGEST)
                    unpacked.rename(ready)
    return ready / "data"
