"""Bounded immutable KV sharing over HTTP and a durable single authority.

This transport deliberately has no torch/CUDA dependency. A model adapter must
finish its device writes before exporting bytes and validate the full reuse
contract before importing them. It must finish importing *all* required shards
before acknowledging migration readiness. HTTP copies bytes into owned host
buffers; this module does not export GPU addresses or claim RDMA completion.

SQLite provides a single authority's serialized, durable decisions. This is not
a replicated consensus service: authority loss pauses control-plane operations.
An authenticated service is a trusted cluster member, not a tenant sandbox.
Use TLS termination or an authenticated private tunnel outside loopback.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import hmac
import http.client
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_HEX = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON = 1024 * 1024
MAX_MANIFEST_METADATA_BYTES = 768 * 1024
_MAX_LAYOUT = 16 * 1024
_DTYPE_BYTES = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
    "bool": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
}


class ShareError(RuntimeError):
    """A typed protocol failure; callers can turn MISS into recomputation."""

    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code, self.status = code, status


def _json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ShareError("INVALID", "non-canonical JSON", 400) from exc


def _name(value: Any, label: str = "name") -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ShareError("INVALID", f"invalid {label}", 400)
    return value


def _oid(value: Any) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ShareError("INVALID", "invalid object/manifest ID", 400)
    return value


def _bounded_dict(value: Any, limit: int) -> dict:
    if not isinstance(value, dict) or len(_json(value)) > limit:
        raise ShareError("INVALID", "invalid or oversized metadata", 400)
    # Bound nesting independently of the JSON parser's recursion limit.
    def depth(v: Any, n: int = 0) -> None:
        if n > 16:
            raise ShareError("INVALID", "metadata nesting exceeds 16", 400)
        if isinstance(v, dict):
            for k, child in v.items():
                if not isinstance(k, str):
                    raise ShareError("INVALID", "metadata keys must be strings", 400)
                depth(child, n + 1)
        elif isinstance(v, list):
            for child in v:
                depth(child, n + 1)
    depth(value)
    return value


def _layout(value: Any, size: int) -> dict:
    value = _bounded_dict(value, _MAX_LAYOUT)
    if not value:
        raise ShareError("INVALID", "layout identity must be explicit", 400)
    if "shape" in value or "dtype" in value:
        shape, dtype = value.get("shape"), value.get("dtype")
        if not isinstance(dtype, str):
            raise ShareError("INVALID", "dtype must be a string", 400)
        dtype = dtype.removeprefix("torch.")
        if (not isinstance(shape, list) or len(shape) > 8 or
                any(type(n) is not int or n < 0 for n in shape) or
                dtype not in _DTYPE_BYTES):
            raise ShareError("INVALID", "invalid tensor shape/dtype", 400)
        if math.prod(shape) * _DTYPE_BYTES[dtype] != size:
            raise ShareError("INVALID", "shape/dtype does not match payload bytes", 400)
    return value


def content_id(namespace: str, layout: dict, payload: bytes) -> str:
    """Bind payload identity to namespace and canonical physical layout."""
    header = _json({"namespace": namespace, "layout": layout})
    return hashlib.sha256(len(header).to_bytes(8, "big") + header + payload).hexdigest()


def validate_manifest_bounds(objects: list[str], metadata: dict, request_id: str) -> None:
    """Check both metadata and complete RPC budgets before uploading a snapshot.

    The 768 KiB metadata cap does not override the 1 MiB total request cap: up to
    4096 object IDs and the RPC envelope must also fit. This function is side-effect
    free so adapters can reject a large export before uploading any payloads.
    """
    _name(request_id, "publication request ID")
    if not isinstance(objects, list) or not objects or len(objects) > 4096:
        raise ShareError("INVALID", "manifest requires 1..4096 objects", 400)
    for oid in objects:
        _oid(oid)
    _bounded_dict(metadata, MAX_MANIFEST_METADATA_BYTES)
    envelope = {"method": "prepare_manifest", "args": {
        "objects": objects, "metadata": metadata, "request_id": request_id}}
    if len(_json(envelope)) > _MAX_JSON:
        raise ShareError("TOO_LARGE", "complete manifest RPC exceeds 1 MiB", 413)


class KVShareStore:
    """One durable, capacity-bounded owner; all GC/grant transitions serialize.

    Objects are stored as SQLite BLOBs to make capacity accounting and metadata
    updates atomic across process crashes. Capacity bounds logical live bytes;
    SQLite/WAL disk overhead and HTTP staging buffers are separate budgets.
    PREPARED publication/migration records never expire independently.
    """

    def __init__(self, root: str | Path, namespace: str, capacity_bytes: int,
                 max_object_bytes: int = 64 * 1024 * 1024,
                 max_inflight_bytes: int | None = None):
        self.namespace = _name(namespace, "namespace")
        if capacity_bytes <= 0 or max_object_bytes <= 0:
            raise ValueError("capacities must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self.max_object_bytes = min(int(max_object_bytes), self.capacity_bytes)
        self.max_inflight_bytes = int(max_inflight_bytes or self.max_object_bytes * 4)
        self.boot_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._pins: dict[str, int] = {}
        self._inflight = 0
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        # Process-local transfer pins are valid only with one live owner process.
        # Never let another service GC this owner's BLOBs while a send is active.
        self._owner_lock = open(root / "owner.lock", "a+b")
        try:
            fcntl.flock(self._owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._owner_lock.close()
            raise ShareError("BUSY", "store already has a live owner process") from exc
        self._db = sqlite3.connect(root / "kv-share.sqlite3", check_same_thread=False,
                                   isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS objects(
                id TEXT PRIMARY KEY, layout TEXT NOT NULL, size INTEGER NOT NULL,
                payload BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS manifests(
                id TEXT PRIMARY KEY, body TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS manifest_objects(
                manifest TEXT NOT NULL, object TEXT NOT NULL,
                PRIMARY KEY(manifest, object));
            CREATE TABLE IF NOT EXISTS publications(
                id TEXT PRIMARY KEY, manifest TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS grants(
                id TEXT PRIMARY KEY, manifest TEXT NOT NULL, holder TEXT NOT NULL,
                expires REAL NOT NULL, released INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS requests(
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, epoch INTEGER NOT NULL,
                checkpoint TEXT, frontier INTEGER NOT NULL, pending TEXT);
            CREATE TABLE IF NOT EXISTS migrations(
                id TEXT PRIMARY KEY, request TEXT NOT NULL, source TEXT NOT NULL,
                epoch INTEGER NOT NULL, target TEXT NOT NULL, checkpoint TEXT NOT NULL,
                frontier INTEGER NOT NULL, state TEXT NOT NULL, ready INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS outputs(
                request TEXT NOT NULL, token_index INTEGER NOT NULL, digest TEXT NOT NULL,
                PRIMARY KEY(request, token_index));
        """)
        with self._transaction() as db:
            row = db.execute("SELECT value FROM settings WHERE key='namespace'").fetchone()
            if row and row[0] != self.namespace:
                raise ShareError("NAMESPACE", "store belongs to a different namespace", 403)
            db.execute("INSERT OR IGNORE INTO settings VALUES('namespace', ?)", (self.namespace,))
            if self._used(db) > self.capacity_bytes:
                raise ShareError("CAPACITY", "persisted objects exceed configured capacity", 507)

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    @staticmethod
    def _used(db: sqlite3.Connection) -> int:
        return int(db.execute("SELECT COALESCE(SUM(size), 0) FROM objects").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            if self._pins:
                raise ShareError("BUSY", "cannot close while transfers are pinned")
            self._db.close()
            self._owner_lock.close()

    def put_object(self, payload: bytes, layout: dict) -> dict:
        if not isinstance(payload, bytes) or len(payload) > self.max_object_bytes:
            raise ShareError("TOO_LARGE", "object exceeds payload bound", 413)
        layout = _layout(layout, len(payload))
        object_id = content_id(self.namespace, layout, payload)
        with self._transaction() as db:
            existing = db.execute("SELECT size FROM objects WHERE id=?", (object_id,)).fetchone()
            if not existing:
                if self._used(db) + len(payload) > self.capacity_bytes:
                    raise ShareError("CAPACITY", "object pool is full; retire/GC or recompute", 507)
                db.execute("INSERT INTO objects VALUES(?, ?, ?, ?)",
                           (object_id, _json(layout).decode(), len(payload), payload))
        return {"object_id": object_id, "size": len(payload), "layout": layout}

    def prepare_manifest(self, objects: list[str], metadata: dict, request_id: str) -> dict:
        validate_manifest_bounds(objects, metadata, request_id)
        objects = sorted(set(_oid(x) for x in objects))
        body = {"namespace": self.namespace, "schema_version": 1,
                "objects": objects, "metadata": metadata}
        encoded = _json(body)
        manifest_id = hashlib.sha256(encoded).hexdigest()
        with self._transaction() as db:
            old = db.execute("SELECT * FROM publications WHERE id=?", (request_id,)).fetchone()
            if old:
                if old["manifest"] != manifest_id:
                    raise ShareError("CONFLICT", "publication request ID was used for other content")
                return dict(old)
            for object_id in objects:
                if not db.execute("SELECT 1 FROM objects WHERE id=?", (object_id,)).fetchone():
                    raise ShareError("MISS", f"object unavailable: {object_id}", 404)
            old_manifest = db.execute("SELECT state FROM manifests WHERE id=?", (manifest_id,)).fetchone()
            if old_manifest and old_manifest["state"] == "RETIRED":
                raise ShareError("RETIRED", "retired manifest cannot be resurrected; use new metadata/version")
            db.execute("INSERT OR IGNORE INTO manifests VALUES(?, ?, 'PREPARED')",
                       (manifest_id, encoded.decode()))
            db.executemany("INSERT OR IGNORE INTO manifest_objects VALUES(?, ?)",
                           [(manifest_id, oid) for oid in objects])
            db.execute("INSERT INTO publications VALUES(?, ?, 'PREPARED')", (request_id, manifest_id))
        return {"id": request_id, "manifest": manifest_id, "state": "PREPARED"}

    def decide_publication(self, request_id: str, commit: bool) -> dict:
        _name(request_id)
        if type(commit) is not bool:
            raise ShareError("INVALID", "commit must be boolean", 400)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM publications WHERE id=?", (request_id,)).fetchone()
            if not row:
                raise ShareError("MISS", "unknown publication", 404)
            if row["state"] == "PREPARED":
                state = "COMMITTED" if commit else "ABORTED"
                db.execute("UPDATE publications SET state=? WHERE id=?", (state, request_id))
                if commit:
                    # PREPARED intent retains every object until this atomic decision.
                    db.execute("UPDATE manifests SET state='COMMITTED' WHERE id=?", (row["manifest"],))
                return {"id": request_id, "manifest": row["manifest"], "state": state}
            return dict(row)  # Commit/abort retries observe the same terminal decision.

    def get_manifest(self, manifest_id: str) -> dict:
        _oid(manifest_id)
        with self._lock:
            row = self._db.execute("SELECT * FROM manifests WHERE id=?", (manifest_id,)).fetchone()
            if not row or row["state"] != "COMMITTED":
                raise ShareError("MISS", "manifest is not committed/available", 404)
            return {"manifest_id": manifest_id, **json.loads(row["body"])}

    @staticmethod
    def _ttl(ttl_s: Any) -> float:
        if type(ttl_s) not in (int, float) or not math.isfinite(ttl_s) or not 0 < ttl_s <= 3600:
            raise ShareError("INVALID", "grant TTL must be in (0, 3600] seconds", 400)
        return float(ttl_s)

    def acquire_grant(self, manifest_id: str, holder: str, ttl_s: float,
                      request_id: str | None = None) -> dict:
        _oid(manifest_id)
        _name(holder, "holder")
        ttl_s = self._ttl(ttl_s)
        grant_id = _name(request_id or uuid.uuid4().hex, "grant request ID")
        with self._transaction() as db:
            old = db.execute("SELECT * FROM grants WHERE id=?", (grant_id,)).fetchone()
            if old:
                if old["manifest"] != manifest_id or old["holder"] != holder:
                    raise ShareError("CONFLICT", "grant request ID reused")
                if old["released"] or old["expires"] <= time.time():
                    raise ShareError("EXPIRED", "grant request ID has already expired/released", 410)
                return {"grant_id": grant_id, **dict(old)}
            row = db.execute("SELECT state FROM manifests WHERE id=?", (manifest_id,)).fetchone()
            if not row or row[0] != "COMMITTED":
                raise ShareError("MISS", "manifest unavailable", 404)
            expires = time.time() + ttl_s
            db.execute("INSERT INTO grants(id, manifest, holder, expires) VALUES(?, ?, ?, ?)",
                       (grant_id, manifest_id, holder, expires))
        return {"grant_id": grant_id, "manifest": manifest_id, "holder": holder, "expires": expires}

    def renew_grant(self, grant_id: str, ttl_s: float) -> dict:
        _name(grant_id)
        ttl_s = self._ttl(ttl_s)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM grants WHERE id=?", (grant_id,)).fetchone()
            if not row or row["released"] or row["expires"] <= time.time():
                raise ShareError("EXPIRED", "grant is expired/released", 410)
            expires = max(row["expires"], time.time() + ttl_s)
            db.execute("UPDATE grants SET expires=? WHERE id=?", (expires, grant_id))
        return {"grant_id": grant_id, "expires": expires}

    def release_grant(self, grant_id: str) -> dict:
        with self._transaction() as db:
            db.execute("UPDATE grants SET released=1 WHERE id=?", (_name(grant_id),))
        return {"grant_id": grant_id, "released": True}

    @contextlib.contextmanager
    def pin_object(self, object_id: str, grant_id: str) -> Iterator[tuple[bytes, dict]]:
        """An admitted host transfer outlives its retention grant expiration."""
        _oid(object_id)
        _name(grant_id)
        with self._lock:
            row = self._db.execute("""
                SELECT o.layout, o.size FROM grants g
                JOIN manifest_objects mo ON mo.manifest=g.manifest
                JOIN objects o ON o.id=mo.object
                WHERE g.id=? AND mo.object=? AND g.released=0 AND g.expires>?
            """, (grant_id, object_id, time.time())).fetchone()
            if not row:
                raise ShareError("MISS", "object is missing or retention grant expired", 404)
            size = row["size"]
            if self._inflight + size > self.max_inflight_bytes:
                raise ShareError("BUSY", "transfer staging budget exhausted", 429)
            self._pins[object_id] = self._pins.get(object_id, 0) + 1
            self._inflight += size
            try:
                payload = self._db.execute("SELECT payload FROM objects WHERE id=?", (object_id,)).fetchone()[0]
            except BaseException:
                self._pins[object_id] -= 1
                if not self._pins[object_id]:
                    del self._pins[object_id]
                self._inflight -= size
                raise
        try:
            layout = json.loads(row["layout"])
            if content_id(self.namespace, layout, payload) != object_id:
                raise ShareError("CORRUPT", "stored object checksum mismatch", 500)
            yield payload, layout
        finally:
            with self._lock:
                self._pins[object_id] -= 1
                if not self._pins[object_id]:
                    del self._pins[object_id]
                self._inflight -= size

    def retire_manifest(self, manifest_id: str) -> dict:
        with self._transaction() as db:
            _oid(manifest_id)
            # An outstanding publication decision must not resurrect retirement.
            if db.execute("SELECT 1 FROM publications WHERE manifest=? AND state='PREPARED'",
                          (manifest_id,)).fetchone():
                raise ShareError("BUSY", "manifest has an undecided publication")
            db.execute("UPDATE manifests SET state='RETIRED' WHERE id=?", (manifest_id,))
        return {"manifest_id": manifest_id, "retired": True}

    def gc(self, limit: int = 1024) -> dict:
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ShareError("INVALID", "GC batch limit must be 1..4096", 400)
        with self._transaction() as db:
            candidates = db.execute("""
                SELECT id, size FROM objects WHERE id NOT IN (
                  SELECT mo.object FROM manifest_objects mo WHERE
                    mo.manifest IN (SELECT id FROM manifests WHERE state='COMMITTED') OR
                    mo.manifest IN (SELECT manifest FROM publications WHERE state='PREPARED') OR
                    mo.manifest IN (SELECT manifest FROM grants WHERE released=0 AND expires>?) OR
                    mo.manifest IN (SELECT checkpoint FROM requests WHERE checkpoint IS NOT NULL) OR
                    mo.manifest IN (SELECT checkpoint FROM migrations WHERE state='PREPARED')
                ) LIMIT ?
            """, (time.time(), limit)).fetchall()
            freed = 0
            removed = []
            for row in candidates:
                if self._pins.get(row["id"], 0):
                    continue
                db.execute("DELETE FROM objects WHERE id=?", (row["id"],))
                freed += row["size"]
                removed.append(row["id"])
        return {"freed_bytes": freed, "objects": removed}

    def stats(self) -> dict:
        with self._lock:
            return {"namespace": self.namespace, "boot_id": self.boot_id,
                    "used_bytes": self._used(self._db), "capacity_bytes": self.capacity_bytes,
                    "inflight_bytes": self._inflight, "pinned_objects": len(self._pins),
                    "object_count": self._db.execute("SELECT COUNT(*) FROM objects").fetchone()[0],
                    "authority": "single_sqlite_no_ha"}

    @staticmethod
    def _request(db: sqlite3.Connection, request_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM requests WHERE id=?", (_name(request_id),)).fetchone()
        if not row:
            raise ShareError("MISS", "unknown request", 404)
        return row

    @staticmethod
    def _checkpoint(db: sqlite3.Connection, checkpoint: str) -> None:
        row = db.execute("SELECT state FROM manifests WHERE id=?", (_oid(checkpoint),)).fetchone()
        if not row or row[0] != "COMMITTED":
            raise ShareError("MISS", "checkpoint manifest is not committed", 404)

    def create_request(self, request_id: str, owner: str, checkpoint: str | None = None) -> dict:
        _name(request_id)
        _name(owner, "owner")
        with self._transaction() as db:
            old = db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
            if old:
                return dict(old)
            if checkpoint is not None:
                self._checkpoint(db, checkpoint)
            db.execute("INSERT INTO requests VALUES(?, ?, 1, ?, 0, NULL)",
                       (request_id, owner, checkpoint))
            return dict(self._request(db, request_id))

    def get_request(self, request_id: str) -> dict:
        with self._lock:
            return dict(self._request(self._db, request_id))

    def finish_request(self, request_id: str, owner: str, epoch: int) -> dict:
        """Drop checkpoint retention but preserve a fencing tombstone forever.

        Request IDs cannot be reused: resetting an epoch would admit stale owners.
        Output digests remain available to the enclosing router's retention policy.
        """
        _name(owner)
        if type(epoch) is not int:
            raise ShareError("INVALID", "epoch must be an integer", 400)
        with self._transaction() as db:
            row = self._request(db, request_id)
            if not row["owner"]:
                return dict(row)
            if row["owner"] != owner or row["epoch"] != epoch:
                raise ShareError("FENCED", "stale execution owner/epoch")
            if row["pending"]:
                raise ShareError("BUSY", "resolve the pending migration before finishing")
            db.execute("UPDATE requests SET owner='', epoch=epoch+1, checkpoint=NULL WHERE id=?",
                       (request_id,))
            return dict(self._request(db, request_id))

    def prepare_migration(self, request_id: str, owner: str, epoch: int,
                          target: str, checkpoint: str, decision_id: str) -> dict:
        """Caller must quiesce source before prepare; this freezes output commits."""
        for value in (request_id, owner, target, decision_id):
            _name(value)
        if type(epoch) is not int or epoch < 1 or owner == target:
            raise ShareError("INVALID", "invalid migration epoch or target", 400)
        _oid(checkpoint)
        with self._transaction() as db:
            old = db.execute("SELECT * FROM migrations WHERE id=?", (decision_id,)).fetchone()
            if old:
                if any(old[k] != v for k, v in (("request", request_id), ("source", owner),
                       ("epoch", epoch), ("target", target), ("checkpoint", checkpoint))):
                    raise ShareError("CONFLICT", "migration ID reused with different arguments")
                return dict(old)
            row = self._request(db, request_id)
            if row["owner"] != owner or row["epoch"] != epoch:
                raise ShareError("FENCED", "stale execution owner/epoch")
            if row["pending"]:
                raise ShareError("BUSY", "request already has a prepared migration")
            self._checkpoint(db, checkpoint)
            db.execute("INSERT INTO migrations VALUES(?, ?, ?, ?, ?, ?, ?, 'PREPARED', 0)",
                       (decision_id, request_id, owner, epoch, target, checkpoint, row["frontier"]))
            db.execute("UPDATE requests SET pending=? WHERE id=?", (decision_id, request_id))
            return dict(db.execute("SELECT * FROM migrations WHERE id=?", (decision_id,)).fetchone())

    def get_migration(self, decision_id: str) -> dict:
        with self._lock:
            row = self._db.execute("SELECT * FROM migrations WHERE id=?", (_name(decision_id),)).fetchone()
            if not row:
                raise ShareError("MISS", "unknown migration decision", 404)
            return dict(row)

    def mark_migration_ready(self, decision_id: str, target: str, checkpoint: str) -> dict:
        """Adapter attests that every required destination shard was imported."""
        with self._transaction() as db:
            row = db.execute("SELECT * FROM migrations WHERE id=?", (_name(decision_id),)).fetchone()
            if not row:
                raise ShareError("MISS", "unknown migration", 404)
            if row["target"] != target or row["checkpoint"] != checkpoint:
                raise ShareError("CONFLICT", "readiness does not match destination/checkpoint")
            if row["state"] == "PREPARED":
                db.execute("UPDATE migrations SET ready=1 WHERE id=?", (decision_id,))
            return dict(db.execute("SELECT * FROM migrations WHERE id=?", (decision_id,)).fetchone())

    def decide_migration(self, decision_id: str, commit: bool) -> dict:
        if type(commit) is not bool:
            raise ShareError("INVALID", "commit must be boolean", 400)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM migrations WHERE id=?", (_name(decision_id),)).fetchone()
            if not row:
                raise ShareError("MISS", "unknown migration", 404)
            if row["state"] != "PREPARED":
                return dict(row)
            req = self._request(db, row["request"])
            if (req["pending"] != decision_id or req["owner"] != row["source"] or
                    req["epoch"] != row["epoch"]):
                raise ShareError("FENCED", "authority owner changed before decision")
            if commit and not row["ready"]:
                raise ShareError("NOT_READY", "all destination shards must be ready")
            state = "COMMITTED" if commit else "ABORTED"
            if commit:
                db.execute("UPDATE requests SET owner=?, epoch=epoch+1, checkpoint=?, pending=NULL WHERE id=?",
                           (row["target"], row["checkpoint"], row["request"]))
            else:
                db.execute("UPDATE requests SET pending=NULL WHERE id=?", (row["request"],))
            db.execute("UPDATE migrations SET state=? WHERE id=?", (state, decision_id))
            return dict(db.execute("SELECT * FROM migrations WHERE id=?", (decision_id,)).fetchone())

    def commit_output(self, request_id: str, owner: str, epoch: int,
                      token_index: int, token_digest: str) -> dict:
        """Router gate: atomic durable frontier/dedup; not an external delivery ACK.

        All output delivery must pass this gate and consumers must acknowledge/
        deduplicate (request_id, token_index). Direct worker output bypasses fencing.
        """
        _name(owner)
        _oid(token_digest)
        if type(epoch) is not int or type(token_index) is not int or token_index < 0:
            raise ShareError("INVALID", "invalid epoch/token index", 400)
        with self._transaction() as db:
            row = self._request(db, request_id)
            if row["owner"] != owner or row["epoch"] != epoch:
                raise ShareError("FENCED", "stale execution owner/epoch")
            if row["pending"]:
                raise ShareError("QUIESCED", "output frozen by prepared migration")
            old = db.execute("SELECT digest FROM outputs WHERE request=? AND token_index=?",
                             (request_id, token_index)).fetchone()
            if old:
                if old[0] != token_digest:
                    raise ShareError("CONFLICT", "token index already committed with different content")
                return {"status": "DUPLICATE", "frontier": row["frontier"]}
            if token_index != row["frontier"]:
                raise ShareError("GAP", "token index must equal committed frontier")
            db.execute("INSERT INTO outputs VALUES(?, ?, ?)", (request_id, token_index, token_digest))
            db.execute("UPDATE requests SET frontier=frontier+1 WHERE id=?", (request_id,))
            return {"status": "COMMITTED", "frontier": token_index + 1}


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], store: KVShareStore, token: str,
                 max_connections: int, io_timeout: float):
        self.store, self.token, self.io_timeout = store, token, io_timeout
        self._slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, _Handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            request.settimeout(self.io_timeout)
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # One bounded request per connection.

    def log_message(self, *args: Any) -> None:
        pass

    def _authorized(self) -> None:
        if self.headers.get("X-KV-Namespace") != self.server.store.namespace:
            raise ShareError("NAMESPACE", "wrong namespace", 403)
        expected = f"Bearer {self.server.token}"
        if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
            raise ShareError("AUTH", "authentication failed", 401)

    def _read(self, limit: int) -> bytes:
        if self.headers.get("Transfer-Encoding"):
            raise ShareError("INVALID", "chunked requests are unsupported", 400)
        values = self.headers.get_all("Content-Length", [])
        if len(values) != 1 or not values[0].isdigit():
            raise ShareError("INVALID", "one Content-Length is required", 400)
        length = int(values[0])
        if length > limit:
            raise ShareError("TOO_LARGE", "request body exceeds bound", 413)
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ShareError("TRUNCATED", "incomplete request body", 400)
        return payload

    def _send_json(self, value: dict, status: int = 200) -> None:
        body = _json(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            self._authorized()
            store = self.server.store
            if self.path == "/v1/objects":
                raw_layout = self.headers.get("X-KV-Layout", "")
                if len(raw_layout) > _MAX_LAYOUT * 2:
                    raise ShareError("INVALID", "layout header too large", 400)
                layout = json.loads(base64.b64decode(raw_layout, validate=True))
                result = store.put_object(self._read(store.max_object_bytes), layout)
            elif self.path == "/v1/rpc":
                body = json.loads(self._read(_MAX_JSON))
                if not isinstance(body, dict) or not isinstance(body.get("args"), dict):
                    raise ShareError("INVALID", "invalid RPC envelope", 400)
                allowed = {
                    "prepare_manifest", "decide_publication", "get_manifest", "acquire_grant",
                    "renew_grant", "release_grant", "retire_manifest", "gc", "stats",
                    "create_request", "get_request", "finish_request", "prepare_migration", "get_migration",
                    "mark_migration_ready", "decide_migration", "commit_output",
                }
                if body.get("method") not in allowed:
                    raise ShareError("INVALID", "unknown RPC", 400)
                result = getattr(store, body["method"])(**body["args"])
            else:
                raise ShareError("MISS", "unknown endpoint", 404)
            self._send_json(result)
        except ShareError as exc:
            self._send_json({"error": exc.code, "message": str(exc)}, exc.status)
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            self._send_json({"error": "INVALID", "message": "malformed request"}, 400)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_GET(self) -> None:
        started = False
        try:
            self._authorized()
            if not self.path.startswith("/v1/objects/"):
                raise ShareError("MISS", "unknown endpoint", 404)
            object_id = _oid(self.path.removeprefix("/v1/objects/"))
            with self.server.store.pin_object(object_id, self.headers.get("X-KV-Grant", "")) as (payload, layout):
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-KV-Object", object_id)
                self.send_header("X-KV-Layout", base64.b64encode(_json(layout)).decode())
                self.end_headers()
                started = True
                self.wfile.write(payload)
                self.wfile.flush()
        except ShareError as exc:
            if not started:
                self._send_json({"error": exc.code, "message": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass


class KVShareServer:
    """Context-managed HTTP owner; shutdown waits for admitted transfers."""

    def __init__(self, store: KVShareStore, host: str = "127.0.0.1", port: int = 0,
                 auth_token: str = "", max_connections: int = 16, io_timeout: float = 10):
        if host not in ("127.0.0.1", "localhost", "::1") and not auth_token:
            raise ValueError("non-loopback service requires an authentication token")
        if max_connections < 1 or io_timeout <= 0:
            raise ValueError("connection bounds must be positive")
        self._server = _HTTPServer((host, port), store, auth_token, max_connections, io_timeout)
        self._thread: threading.Thread | None = None
        address, actual_port = self._server.server_address[:2]
        self.url = f"http://{address}:{actual_port}"

    def start(self) -> "KVShareServer":
        if self._thread:
            raise RuntimeError("server already started")
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._thread:
            self._server.shutdown()
            self._thread.join()
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> "KVShareServer":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.close()


class KVShareClient:
    """No automatic mutation retries: retry using the same explicit request ID."""

    def __init__(self, url: str, namespace: str, auth_token: str = "", timeout: float = 10,
                 max_object_bytes: int = 64 * 1024 * 1024):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/"):
            raise ValueError("expected an HTTP(S) service origin")
        self.url, self.namespace = url.rstrip("/"), _name(namespace)
        self.token, self.timeout, self.max_object_bytes = auth_token, timeout, max_object_bytes

    def _request(self, path: str, data: bytes | None = None,
                 headers: dict | None = None, limit: int = _MAX_JSON) -> tuple[bytes, Any]:
        all_headers = {"X-KV-Namespace": self.namespace,
                       "Authorization": f"Bearer {self.token}", **(headers or {})}
        request = Request(self.url + path, data=data, headers=all_headers,
                          method="POST" if data is not None else "GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                length = response.headers.get("Content-Length", "")
                if not length.isdigit() or int(length) > limit:
                    raise ShareError("TOO_LARGE", "response exceeds declared bound", 413)
                body = response.read(int(length) + 1)
                if len(body) != int(length):
                    raise ShareError("TRUNCATED", "incomplete response", 502)
                return body, response.headers
        except HTTPError as exc:
            try:
                error = json.loads(exc.read(64 * 1024))
                raise ShareError(error["error"], error["message"], exc.code) from exc
            except (ValueError, KeyError, TypeError):
                raise ShareError("REMOTE", f"remote HTTP {exc.code}", exc.code) from exc
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            raise ShareError("UNAVAILABLE", "source unavailable or transfer incomplete", 503) from exc

    def rpc(self, method: str, **args: Any) -> dict:
        raw, _ = self._request("/v1/rpc", _json({"method": method, "args": args}),
                               {"Content-Type": "application/json"})
        try:
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("not an object")
            return result
        except ValueError as exc:
            raise ShareError("CORRUPT", "malformed remote response", 502) from exc

    def put_object(self, payload: bytes, layout: dict) -> str:
        if not isinstance(payload, bytes) or len(payload) > self.max_object_bytes:
            raise ShareError("TOO_LARGE", "object exceeds client bound", 413)
        _layout(layout, len(payload))
        raw, _ = self._request("/v1/objects", payload,
                               {"X-KV-Layout": base64.b64encode(_json(layout)).decode(),
                                "Content-Type": "application/octet-stream"})
        result = json.loads(raw)
        expected = content_id(self.namespace, layout, payload)
        if result.get("object_id") != expected or result.get("size") != len(payload):
            raise ShareError("CORRUPT", "published object identity mismatch", 502)
        return expected

    def fetch_object(self, object_id: str, grant_id: str,
                     expected_layout: dict | None = None) -> bytes:
        _oid(object_id)
        raw, headers = self._request(f"/v1/objects/{object_id}", headers={"X-KV-Grant": _name(grant_id)},
                                     limit=self.max_object_bytes)
        try:
            layout = json.loads(base64.b64decode(headers.get("X-KV-Layout", ""), validate=True))
            _layout(layout, len(raw))
        except (ValueError, TypeError) as exc:
            raise ShareError("CORRUPT", "invalid remote layout", 502) from exc
        if (headers.get("X-KV-Object") != object_id or
                content_id(self.namespace, layout, raw) != object_id):
            raise ShareError("CORRUPT", "object checksum/identity mismatch", 502)
        if expected_layout is not None and _json(layout) != _json(expected_layout):
            raise ShareError("LAYOUT", "physical layout does not match target adapter", 409)
        return raw

    def publish_manifest(self, objects: list[str], metadata: dict, request_id: str) -> str:
        validate_manifest_bounds(objects, metadata, request_id)
        result = self.rpc("prepare_manifest", objects=objects, metadata=metadata, request_id=request_id)
        result = self.rpc("decide_publication", request_id=request_id, commit=True)
        if result["state"] != "COMMITTED":
            raise ShareError("ABORTED", "publication was already aborted")
        return result["manifest"]

    def get_manifest(self, manifest_id: str) -> dict:
        result = self.rpc("get_manifest", manifest_id=manifest_id)
        body = {k: v for k, v in result.items() if k != "manifest_id"}
        if result.get("namespace") != self.namespace or hashlib.sha256(_json(body)).hexdigest() != manifest_id:
            raise ShareError("CORRUPT", "manifest namespace/digest mismatch", 502)
        return result

    def acquire_grant(self, manifest_id: str, holder: str, ttl_s: float,
                      request_id: str | None = None) -> dict:
        return self.rpc("acquire_grant", manifest_id=manifest_id, holder=holder,
                        ttl_s=ttl_s, request_id=request_id)

    def renew_grant(self, grant_id: str, ttl_s: float) -> dict:
        return self.rpc("renew_grant", grant_id=grant_id, ttl_s=ttl_s)

    def release_grant(self, grant_id: str) -> dict:
        return self.rpc("release_grant", grant_id=grant_id)

    def retire_manifest(self, manifest_id: str) -> dict:
        return self.rpc("retire_manifest", manifest_id=manifest_id)

    def gc(self, limit: int = 1024) -> dict:
        return self.rpc("gc", limit=limit)

    def stats(self) -> dict:
        return self.rpc("stats")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--capacity-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-object-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--token-env", default="REDKNOT_KV_SHARE_TOKEN")
    args = parser.parse_args()
    store = KVShareStore(args.root, args.namespace, args.capacity_bytes, args.max_object_bytes)
    try:
        with KVShareServer(store, args.host, args.port, os.environ.get(args.token_env, "")) as server:
            print(json.dumps({"url": server.url, "namespace": args.namespace,
                              "authority": "single_sqlite_no_ha"}), flush=True)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


if __name__ == "__main__":
    _main()
