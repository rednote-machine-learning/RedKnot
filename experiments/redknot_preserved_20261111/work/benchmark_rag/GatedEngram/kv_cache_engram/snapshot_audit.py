"""Bounded snapshot-to-paired-audit bridge; never a runtime reuse admission.

Snapshot seals establish integrity, not an authenticated producer or proof of
model execution. A native source declaration stays a declaration in the audit.
This module loads complete CPU matrices within an explicit value-count budget;
it never silently samples rows, widens the budget, trains or exports a gate.
"""
from __future__ import annotations

from .core import digest, integer, require
from .offline import CAPTURE_POINT, SCHEMA, Snapshot
from .reuse_audit import PairedKV, audit_pair


DEFAULT_MAX_VALUES = 1_048_576
_SOURCES = {
    "python-float64-rmsnorm-reference": (False, False),
    "torch-cpu-float32-rmsnorm-reference": (False, False),
    "deepinfer-norm-dependency-fixture": (False, False),
    "deepinfer-native-fused-rmsnorm-only": (True, True),
}


def _metadata(snapshot):
    require(type(snapshot) is Snapshot, "opened Snapshot required")
    # Snapshot has no public is_open property. Do not enter its context manager:
    # this bridge must not take ownership of, or close, a caller's handles.
    require(snapshot._dir_fd is not None, "snapshot is closed")
    manifest = snapshot.manifest
    require(type(manifest) is dict and set(manifest) == {
        "schema", "storage", "spec", "provenance", "files", "seal"},
        "unexpected current snapshot manifest")
    require(manifest["schema"] == SCHEMA and manifest["storage"] == "little-endian-float64",
            "unsupported snapshot format")
    require(manifest["seal"] == digest({key: value for key, value in manifest.items() if key != "seal"}),
            "snapshot manifest changed after validation")
    require(manifest["spec"] == snapshot.spec.metadata(), "snapshot spec and manifest disagree")
    provenance = manifest["provenance"]
    source = (provenance["operator"], provenance["native_operator"], provenance["accelerator_capture"])
    require(source[0] in _SOURCES and source[1:] == _SOURCES[source[0]]
            and type(source[1]) is bool and type(source[2]) is bool,
            "unknown or inconsistent snapshot capture operator")
    require(provenance["adapter"] == "deepinfer-v4-swa-capture-v1"
            and provenance["capture_point"] == CAPTURE_POINT,
            "unsupported capture adapter or point")
    return manifest, source


def snapshot_pair_provenance(offline_snapshot: Snapshot, online_snapshot: Snapshot,
                             layer: int) -> dict:
    """Validate metadata and return a detached source summary, without KV reads.

    The complete manifests were validated by Snapshot on open; this additionally
    checks pairing, current in-memory seals and source compatibility. Payloads
    are revalidated on read by pair_from_snapshots, not by this metadata helper.
    """
    integer(layer, "layer")
    offline, offline_source = _metadata(offline_snapshot)
    online, online_source = _metadata(online_snapshot)
    first, second = offline_snapshot.spec, online_snapshot.spec
    require(first.kind == "independent_document", "offline snapshot must be independent_document")
    require(second.kind == "full_context_reference", "online snapshot must be full_context_reference")
    require(first.request_id != second.request_id, "independent captures require distinct request IDs")
    require(layer in first.layers and layer in second.layers, "requested layer missing from a snapshot")
    require(first.document_ids == second.document_ids, "snapshot document token IDs differ")
    require(first.identities[layer] == second.identities[layer], "snapshot model/config/layer identity differs")
    require(offline_source == online_source, "snapshot capture operators or origins differ")
    provenance = {"schema": "kv-cache-engram-snapshot-pair-source-v1", "layer": layer,
        "adapter": "deepinfer-v4-swa-capture-v1", "capture_point": CAPTURE_POINT,
        "identity_sha256": first.identities[layer].sha256,
        "operator": offline_source[0], "native_operator_declared": offline_source[1],
        "accelerator_capture_declared": offline_source[2],
        "origin_authenticated": False, "task_accuracy_certified": False,
        "offline": {"seal": offline["seal"], "kind": first.kind,
            "request_id": first.request_id, "phase": first.phase,
            "input_ids_sha256": first.request_sha256,
            "document_start": first.document_start, "document_end": first.document_end},
        "online": {"seal": online["seal"], "kind": second.kind,
            "request_id": second.request_id, "phase": second.phase,
            "input_ids_sha256": second.request_sha256,
            "document_start": second.document_start, "document_end": second.document_end}}
    return provenance


def pair_from_snapshots(offline_snapshot: Snapshot, online_snapshot: Snapshot,
                        layer: int, *, max_values: int = DEFAULT_MAX_VALUES) -> PairedKV:
    """Load a complete independent/full-context pair after an upfront budget check.

    ``max_values`` bounds the *sum* of both matrices' scalar values, not bytes,
    Python heap size, the manifest's token IDs or later audit-report memory.
    There are no reduced-coverage or automatic-budget-increase paths. Each
    selected payload is validated in full while reading the complete document.
    Native model_capture is declared only when both compatible manifests mark
    the native operator and accelerator_capture=True; manifests do not prove it.
    """
    integer(max_values, "max_values", 1)
    source = snapshot_pair_provenance(offline_snapshot, online_snapshot, layer)
    count = len(offline_snapshot.spec.document_ids)
    values = 2 * count * offline_snapshot.spec.width
    require(values <= max_values, "complete paired matrices exceed CPU value budget")
    # Avoid as_cache() allocating one side before the combined bound is known.
    cache = offline_snapshot.as_cache(layer, max_values=max_values)
    online = online_snapshot.read_rows(layer, 0, count, max_values=max_values)
    origin = ("model_capture" if source["native_operator_declared"]
              and source["accelerator_capture_declared"] else "synthetic")
    spec = online_snapshot.spec
    return PairedKV(cache, spec.identities[layer], spec.document_ids, spec.input_ids,
        spec.document_start, online, spec.request_id, origin, digest(source))


def audit_snapshots(offline_snapshot: Snapshot, online_snapshot: Snapshot,
                    layer: int, policy, *, oracle_relative_tolerance: float,
                    oracle_absolute_tolerance: float,
                    max_values: int = DEFAULT_MAX_VALUES) -> dict:
    """Run the fixed offline gate diagnostic and retain both capture sources.

    This report neither creates a CalibrationRecord nor enables runtime reuse.
    A full CPU reference read is part of offline auditing, not an online gate's
    speed path. No GPU or model operation is performed here.
    """
    pair = pair_from_snapshots(offline_snapshot, online_snapshot, layer, max_values=max_values)
    source = snapshot_pair_provenance(offline_snapshot, online_snapshot, layer)
    require(pair.capture_sha256 == digest(source), "snapshot source changed during pairing")
    result = audit_pair(pair, policy,
        oracle_relative_tolerance=oracle_relative_tolerance,
        oracle_absolute_tolerance=oracle_absolute_tolerance)
    result["snapshot_sources"] = source
    result["runtime_admission_created"] = False
    return result
