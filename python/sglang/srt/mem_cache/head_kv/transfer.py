"""Checked immutable snapshot transfer between independently owned KV managers.

HTTP copies into owned host buffers; this is not an RDMA performance claim. The
source read lease covers GPU-to-host staging. A destination root is published
only after all object digests, layout/coverage metadata, and device copies pass.
Object IDs authenticate the bytes/layout; a source StatePageID alone does not.
"""
from __future__ import annotations

import logging
import math
import re
import sys
import threading
import time

import torch

from .distributed import ShareError, content_id, validate_manifest_bounds
from .manager import ReadLease, Segment, WriteTransaction
from .pool import completion_event

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
_HEX = re.compile(r"^[0-9a-f]{64}$")
_LOG = logging.getLogger(__name__)
# Bound expanded descriptor metadata even when arithmetic runs are very small.
_MAX_VIEW_PAGES = 16384
_MAX_LOGICAL_POSITIONS = 4 * 1024 * 1024


def _text(value, label, *, empty=False):
    if not isinstance(value, str) or (not empty and not value) or len(value) > 4096:
        raise ValueError(f"invalid snapshot {label}")
    return value


def _retries(value):
    if type(value) is not int or not 0 <= value <= 8:
        raise ValueError("max_retries must be in 0..8")
    return value


def _retry(operation, retries, check=lambda: None):
    """Retry only transient, idempotent HTTP operations with bounded backoff."""
    for attempt in range(retries + 1):
        check()
        try:
            return operation()
        except ShareError as exc:
            if exc.code not in ("UNAVAILABLE", "BUSY") or attempt == retries:
                raise
            time.sleep(min(0.025 * (2 ** attempt), 0.25))


def _page_bytes(pool, ref):
    k, v = pool.read_page(ref)
    # .cpu() is a synchronizing staging boundary, including on a caller CUDA stream.
    return torch.stack((k, v)).detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _encode_positions(positions):
    """Use arithmetic runs for normal contiguous/strided positions, lists otherwise."""
    runs, cursor = [], 0
    while cursor < len(positions):
        end = cursor + 1
        step = positions[end] - positions[cursor] if end < len(positions) else 1
        if end < len(positions):
            end += 1
            while end < len(positions) and positions[end] - positions[end - 1] == step:
                end += 1
        runs.append([positions[cursor], end - cursor, step])
        cursor = end
    return {"runs": runs} if len(runs) * 3 < len(positions) else list(positions)


def _decode_positions(encoded, max_count):
    if isinstance(encoded, list):
        if len(encoded) > max_count or any(type(p) is not int or not 0 <= p < 2**63 for p in encoded):
            raise ValueError("invalid or oversized imported logical positions")
        return encoded
    if not isinstance(encoded, dict) or set(encoded) != {"runs"} or not isinstance(encoded["runs"], list):
        raise ValueError("invalid logical position encoding")
    result = []
    for run in encoded["runs"]:
        if not isinstance(run, list) or len(run) != 3 or any(type(x) is not int for x in run):
            raise ValueError("invalid logical position run")
        start, count, step = run
        if (count <= 0 or count > max_count - len(result) or not 0 <= start < 2**63 or
                step == 0 or not 0 <= start + (count - 1) * step < 2**63):
            raise ValueError("invalid or oversized logical position run")
        result.extend(range(start, start + count * step, step))
    return result


class _GrantKeeper:
    """Renew a batch grant while a slow page transfer is blocked in another thread.

    Renewal never substitutes for a source transfer pin. Once bytes have arrived
    they belong to the destination; source grant expiration cannot invalidate them.
    No new fetch starts after an unrecoverable renewal failure.
    """

    def __init__(self, client, grant_id, ttl_s):
        self.client, self.grant_id, self.ttl_s = client, grant_id, ttl_s
        self._stop = threading.Event()
        self._error = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="head-kv-grant-renewal")

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        interval = max(0.005, self.ttl_s / 3)
        while not self._stop.wait(interval):
            try:
                _retry(lambda: self.client.renew_grant(self.grant_id, self.ttl_s), 2)
            except Exception as exc:
                with self._lock:
                    self._error = exc
                return

    def check(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError("snapshot retention renewal failed") from self._error

    def close(self):
        self._stop.set()
        self._thread.join(timeout=max(1, self.client.timeout + 1))
        # A delayed renewal cannot resurrect a released grant: owner transitions
        # serialize and renewal rejects the durable released flag.
        if self._thread.is_alive():
            _LOG.warning("grant renewal still draining; release/owner TTL remains authoritative")


def export_snapshot(manager, request_id, client, *, operation_id, max_retries=2):
    max_retries = _retries(max_retries)
    with manager.bind(request_id) as lease:
        version = lease.version
        if client.namespace != version.namespace:
            raise ValueError("transport namespace mismatch")
        if not lease.refs:
            raise ValueError("cannot export an empty snapshot")
        dtype = str(manager.pool.dtype).split(".")[-1]
        records = {}
        backing_groups = {}
        for segment in version.segments.values():
            group = segment.key[:2]
            for ref in segment.pages:
                if ref in backing_groups and backing_groups[ref] != group:
                    raise ValueError("one physical page cannot alias different layer/KV groups")
                backing_groups[ref] = group
        # Stable record order makes publication request-ID retries deterministic.
        for index, ref in enumerate(sorted(lease.refs, key=lambda p: p.content_id)):
            k, _ = manager.pool.read_page(ref)
            layer, kv_head = backing_groups[ref]
            layout = dict(schema="head-kv-page-v2", dtype=dtype, shape=[2, len(k), manager.pool.head_dim],
                          byteorder=sys.byteorder, head_dim=manager.pool.head_dim,
                          page_size=manager.pool.page_size, extent=len(k), state_page_id=ref.content_id,
                          contract=version.contract, layer=layer, kv_head=kv_head)
            # Every real content ID is 64 ASCII characters; placeholders let us
            # check the exact metadata/RPC byte budget before the first upload.
            records[ref] = dict(object_id=f"{index:064x}", layout=layout)
        segments = []
        for key in sorted(version.segments):
            s = version.segments[key]
            segments.append(dict(key=list(s.key), pages=[records[p]["object_id"] for p in s.pages],
                                 positions=_encode_positions(s.positions), provenance=s.provenance,
                                 position_basis=s.position_basis, reuse_kind=s.reuse_kind,
                                 validity_certificate=s.validity_certificate))
        metadata = dict(schema="head-kv-snapshot-v2", context_id=version.context_id,
                        contract=version.contract, namespace=version.namespace,
                        dtype=dtype, head_dim=manager.pool.head_dim, page_size=manager.pool.page_size,
                        source_generation=version.generation, source_epoch=version.epoch,
                        pages=list(records.values()), segments=segments)
        objects = [r["object_id"] for r in records.values()]
        validate_manifest_bounds(objects, metadata, operation_id)
        _validate_snapshot({"namespace": version.namespace, "objects": objects, "metadata": metadata},
                           version, manager.pool, client.namespace)
        actual_ids = {}
        for ref, record in records.items():
            raw = _page_bytes(manager.pool, ref)
            oid = _retry(lambda: client.put_object(raw, record["layout"]), max_retries)
            actual_ids[record["object_id"]] = oid
            record["object_id"] = oid
        for record in metadata["segments"]:
            record["pages"] = [actual_ids[x] for x in record["pages"]]
        return _retry(lambda: client.publish_manifest(
            [r["object_id"] for r in records.values()], metadata, request_id=operation_id), max_retries)


def _validate_snapshot(manifest, old, pool, namespace):
    """Validate the entire plan before reserving device pages or fetching payloads."""
    meta = manifest["metadata"]
    if not isinstance(meta, dict) or meta.get("schema") != "head-kv-snapshot-v2":
        raise ValueError("unknown snapshot schema")
    if namespace != old.namespace or manifest.get("namespace") != old.namespace:
        raise ValueError("incompatible transport namespace")
    for name, expected in (("namespace", old.namespace), ("contract", old.contract),
                           ("context_id", old.context_id), ("dtype", str(pool.dtype).split(".")[-1]),
                           ("head_dim", pool.head_dim), ("page_size", pool.page_size)):
        if type(meta.get(name)) is not type(expected) or meta[name] != expected:
            raise ValueError(f"incompatible snapshot {name}")
    _text(meta.get("source_generation"), "source generation")
    if type(meta.get("source_epoch")) is not int or meta["source_epoch"] < 0:
        raise ValueError("invalid source epoch")
    records = meta.get("pages")
    if not isinstance(records, list) or not records or len(records) > pool.capacity_pages:
        raise ValueError("snapshot exceeds local page capacity or has no pages")
    by_oid, by_state = {}, {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid page record")
        oid, layout = record.get("object_id"), record.get("layout")
        if not isinstance(oid, str) or not _HEX.fullmatch(oid) or oid in by_oid or not isinstance(layout, dict):
            raise ValueError("invalid/duplicate page object identity")
        n = layout.get("extent")
        if type(n) is not int or not 0 < n <= pool.page_size:
            raise ValueError("page extent mismatch")
        expected = dict(schema="head-kv-page-v2", dtype=meta["dtype"], shape=[2, n, pool.head_dim],
                        byteorder=sys.byteorder, head_dim=pool.head_dim, page_size=pool.page_size,
                        contract=old.contract)
        if any(type(layout.get(k)) is not type(v) or layout[k] != v for k, v in expected.items()):
            raise ValueError("page layout mismatch")
        if any(type(layout.get(k)) is not int or layout[k] < 0 for k in ("layer", "kv_head")):
            raise ValueError("page layer/KV group identity missing")
        state = _text(layout.get("state_page_id"), "state page identity")
        if state in by_state:
            raise ValueError("conflicting page identities")
        by_oid[oid], by_state[state] = record, record
    if (not isinstance(manifest.get("objects"), list) or
            len(manifest["objects"]) != len(by_oid) or set(by_oid) != set(manifest["objects"])):
        raise ValueError("snapshot object coverage mismatch")
    records = list(by_oid.values())
    raw_segments = meta.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("invalid snapshot segments")
    segments, used, positions_per_head = {}, set(), {}
    total_positions, total_view_pages = 0, 0
    for record in raw_segments:
        if not isinstance(record, dict) or not isinstance(record.get("key"), list):
            raise ValueError("invalid segment record")
        key = tuple(record["key"])
        if (len(key) != 3 or any(type(x) is not int or x < 0 for x in key[:2]) or
                not isinstance(key[2], str) or not key[2] or key in segments):
            raise ValueError("invalid/duplicate segment key")
        pages = record.get("pages")
        if not isinstance(pages, list):
            raise ValueError("invalid segment page coverage")
        total_view_pages += len(pages)
        if total_view_pages > _MAX_VIEW_PAGES:
            raise ValueError("snapshot exceeds descriptor page-reference budget")
        pos = _decode_positions(record.get("positions"), min(len(pages) * pool.page_size,
                                                              _MAX_LOGICAL_POSITIONS - total_positions))
        total_positions += len(pos)
        if len(set(pos)) != len(pos):
            raise ValueError("invalid imported logical positions")
        seen = positions_per_head.setdefault(key[:2], set())
        if seen.intersection(pos):
            raise ValueError("overlapping logical positions within one KV head")
        seen.update(pos)
        if not isinstance(pages, list) or len(pages) != math.ceil(len(pos) / pool.page_size):
            raise ValueError("segment coverage mismatch")
        for i, oid in enumerate(pages):
            if not isinstance(oid, str) or oid not in by_oid:
                raise ValueError("unknown segment page object")
            layout = by_oid[oid]["layout"]
            if (layout["layer"], layout["kv_head"]) != key[:2]:
                raise ValueError("page cannot be referenced by a different layer/KV group")
            if by_oid[oid]["layout"]["extent"] < min(pool.page_size, len(pos) - i * pool.page_size):
                raise ValueError("short segment page")
            used.add(oid)
        _text(record.get("provenance"), "provenance")
        _text(record.get("position_basis"), "position basis")
        kind = record.get("reuse_kind")
        if kind not in ("exact_context", "certified_transform", "policy_approximate"):
            raise ValueError("invalid imported reuse kind")
        certificate = _text(record.get("validity_certificate", ""), "validity certificate", empty=True)
        if kind != "exact_context" and not certificate:
            raise ValueError("non-exact reuse requires an adapter validity certificate")
        segments[key] = {**record, "positions": pos}
    if used != set(by_oid):
        raise ValueError("manifest contains unreferenced page payload")
    return meta, records, segments, by_state


def import_snapshot(manager, request_id, client, manifest_id, *, holder, ttl_s=60, max_retries=2):
    """Import into an empty request with the same context/contract.

    For non-prefix reuse, import the certified source snapshot, then bind its
    segments into the target request using explicit ReuseProof objects. Transient
    reads retry at most max_retries times; corrupt/expired objects fail closed.
    A failed attempt leaves the request empty and can be retried by the caller.
    """
    max_retries = _retries(max_retries)
    # Acquisition uses an explicit ID for safe retry after a lost response.
    import uuid
    grant_request = uuid.uuid4().hex
    grant = _retry(lambda: client.acquire_grant(manifest_id, holder, ttl_s, request_id=grant_request), max_retries)
    grant_id = grant["grant_id"]
    reserved, lease, tx, keeper = (), None, None, None
    try:
        keeper = _GrantKeeper(client, grant_id, ttl_s).start()
        manifest = _retry(lambda: client.get_manifest(manifest_id), max_retries, keeper.check)
        pool = manager.pool
        with manager._lock:
            old = manager.version(request_id)
            if old.segments or request_id in manager._pending:
                raise ValueError("import requires an empty idle request")
            meta, records, raw_segments, by_state = _validate_snapshot(manifest, old, pool, client.namespace)
            # Source IDs are provenance, whereas object_id is the checked content
            # identity. Reject conflicting bytes for a currently resident source
            # identity. Pin resident replicas across comparison and network work.
            with pool._lock:
                published = {}
                for version in list(manager._requests.values()) + list(manager._cache.values()):
                    for segment in version.segments.values():
                        group = (version.namespace, version.contract, *segment.key[:2])
                        for ref in segment.pages:
                            published.setdefault(ref, set()).add(group)
                existing = tuple(p.ref for p in pool._pages.values() if p.ref.content_id in by_state)
                if any(not pool._get(p).sealed or p not in published for p in existing):
                    raise ValueError("same state identity has no completed local snapshot")
                for ref in existing:
                    layout = by_state[ref.content_id]["layout"]
                    expected_group = (old.namespace, old.contract, layout["layer"], layout["kv_head"])
                    if published[ref] != {expected_group}:
                        raise ValueError("same state identity belongs to a different contract/layer/KV group")
                lease = ReadLease(pool, old)
                combined_refs = tuple(set(lease.refs).union(existing))
                pool.pin(existing)
                lease.refs = combined_refs
            reserved = pool.reserve(len(records), content_ids=[r["layout"]["state_page_id"] for r in records])
            tx = WriteTransaction(manager, old, {}, reserved, lease, completion_event(pool.device))
            tx.staging = True
            manager._pending[request_id] = tx
        refs = dict(zip((r["object_id"] for r in records), reserved))

        def check():
            if tx.cancel_requested:
                raise RuntimeError("snapshot import cancelled")
            keeper.check()

        for local_ref in existing:
            check()
            record = by_state[local_ref.content_id]
            raw = _page_bytes(pool, local_ref)
            if content_id(client.namespace, record["layout"], raw) != record["object_id"]:
                raise ValueError("same state page identity claims conflicting payload/layout")
        for record, ref in zip(records, reserved):
            payload = _retry(lambda: client.fetch_object(record["object_id"], grant_id,
                                                        expected_layout=record["layout"]), max_retries, check)
            if tx.cancel_requested:
                raise RuntimeError("snapshot import cancelled")
            layout = record["layout"]
            n = layout["extent"]
            required = 2 * n * pool.head_dim * pool.k.element_size()
            # Recheck at the manager boundary, including custom transport adapters.
            if len(payload) != required or content_id(client.namespace, layout, payload) != record["object_id"]:
                raise ValueError("page payload size/content identity mismatch")
            # copy_ uses the pool's current stream; blocking H2D staging owns its
            # bytearray until the copy completes. Device retirement uses an event.
            data = torch.frombuffer(bytearray(payload), dtype=_DTYPES[meta["dtype"]]).reshape(2, n, pool.head_dim)
            pool.write_page(ref, data[0], data[1])
        segments = {
            key: Segment(key, tuple(refs[x] for x in r["pages"]), tuple(r["positions"]),
                         r["provenance"], r["position_basis"], r["reuse_kind"], r.get("validity_certificate", ""))
            for key, r in raw_segments.items()
        }
        with manager._lock:
            pool.seal(reserved)
            tx.segments = segments
            tx.event = completion_event(pool.device)
            tx.staging = False
            return tx.commit(wait=True)
    except BaseException:
        if tx is not None and tx.status == "PREPARED":
            try:
                tx.event = completion_event(manager.pool.device)
                tx.staging = False
                tx.abort()
            except BaseException:
                # Unknown GPU completion must quarantine rather than recycle slots.
                manager.pool.poisoned = True
        elif tx is None:
            try:
                event = completion_event(manager.pool.device)
                if reserved:
                    manager.pool.release(reserved, event)
                if lease is not None:
                    lease.complete(event)
            except BaseException:
                manager.pool.poisoned = True
        raise
    finally:
        if keeper is not None:
            keeper.close()
        try:
            _retry(lambda: client.release_grant(grant_id), max_retries)
        except Exception:
            _LOG.warning("retention grant release failed; owner TTL remains in force")
