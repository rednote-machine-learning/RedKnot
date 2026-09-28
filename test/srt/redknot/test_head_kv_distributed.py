"""CPU-only integration/fault tests; starts actual independent HTTP processes.

Run directly with ``python test/srt/redknot/test_head_kv_distributed.py``.
No sglang, torch, model weights, network account, or external database is needed.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest


MODULE = Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache/head_kv/distributed.py"
spec = importlib.util.spec_from_file_location("head_kv_distributed", MODULE)
dist = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dist
spec.loader.exec_module(dist)


class OwnerProcess:
    def __init__(self, root, capacity=1024 * 1024):
        self.root, self.capacity, self.process = root, capacity, None
        self.start()

    def start(self):
        env = dict(os.environ, REDKNOT_KV_SHARE_TOKEN="test-cluster-secret")
        self.process = subprocess.Popen(
            [sys.executable, str(MODULE), "--root", str(self.root),
             "--namespace", "model-contract-v1", "--capacity-bytes", str(self.capacity),
             "--max-object-bytes", str(self.capacity)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        ready, _, _ = select.select([self.process.stdout], [], [], 10)
        if not ready:
            self.stop()
            raise RuntimeError("owner process startup timed out")
        line = self.process.stdout.readline()
        if not line:
            err = self.process.stderr.read()
            self.stop()
            raise RuntimeError(f"owner process failed: {err}")
        self.url = json.loads(line)["url"]
        self.client = dist.KVShareClient(self.url, "model-contract-v1", "test-cluster-secret", timeout=2)

    def stop(self):
        if self.process:
            self.process.kill()  # Exercise abrupt process crash, not graceful shutdown.
            self.process.wait(timeout=10)
            self.process.stdout.close()
            self.process.stderr.close()
            self.process = None

    def restart(self):
        self.stop()
        self.start()


class DistributedNetworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.a = OwnerProcess(Path(self.temp.name) / "a")
        self.b = OwnerProcess(Path(self.temp.name) / "b")
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.a.stop)
        self.addCleanup(self.b.stop)

    @staticmethod
    def layout(size, head=2):
        return {"shape": [size], "dtype": "uint8", "layer": 12, "kv_head": head,
                "position_basis": "canonical", "model": "test-model"}

    def bundle(self, client, payload=b"abcdefghijklmnop", request_id="publish"):
        layout = self.layout(len(payload))
        oid = client.put_object(payload, layout)
        mid = client.publish_manifest([oid], {"contract": "exact-source-v1"}, request_id)
        grant = client.acquire_grant(mid, "reader", 60, "grant-" + request_id)
        return oid, mid, grant["grant_id"], layout

    def assertCode(self, code, function, *args, **kwargs):
        with self.assertRaises(dist.ShareError) as ctx:
            function(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_two_process_nonprefix_copy_on_write_source_unchanged(self):
        a, b = self.a.client, self.b.client
        source = [bytes(range(i, i + 16)) for i in (0, 16, 32)]
        layout = self.layout(16)
        objects = [a.put_object(page, layout) for page in source]
        mid = a.publish_manifest(objects, {
            "occurrence_order": ["P", "D", "E"], "segment_D": objects,
            "provenance": "source-context", "repair_policy": "test-explicit",
        }, "source-bundle")
        grant = a.acquire_grant(mid, "node-b", 60)["grant_id"]
        self.assertEqual(a.get_manifest(mid)["metadata"]["segment_D"], objects)
        # B places D after E; occurrence/position metadata belongs to its own root.
        imported = [bytearray(a.fetch_object(oid, grant, layout)) for oid in objects]
        imported[0][0:2] = b"XY"
        imported[1][1] = 99
        modified = [b.put_object(bytes(page), layout) for page in imported]
        dest = b.publish_manifest(modified, {
            "occurrence_order": ["R", "E", "D"], "segment_D": modified,
            "provenance": "target-context", "repair_policy": "test-explicit",
        }, "target-bundle")
        self.assertNotEqual(modified[0], objects[0])
        self.assertNotEqual(modified[1], objects[1])
        self.assertEqual(modified[2], objects[2])
        self.assertNotEqual(dest, mid)
        for oid, expected in zip(objects, source):
            self.assertEqual(a.fetch_object(oid, grant, layout), expected)
        self.assertEqual(a.stats()["used_bytes"], 48)
        self.assertEqual(b.stats()["used_bytes"], 48)

    def test_prepare_survives_crash_retains_objects_and_hides_manifest(self):
        client = self.a.client
        oid = client.put_object(b"payload", self.layout(7))
        prepared = client.rpc("prepare_manifest", objects=[oid], metadata={"version": 1}, request_id="p")
        self.assertCode("MISS", client.get_manifest, prepared["manifest"])
        self.assertEqual(client.gc()["freed_bytes"], 0)
        self.a.restart()
        client = self.a.client
        self.assertEqual(client.gc()["freed_bytes"], 0)
        committed = client.rpc("decide_publication", request_id="p", commit=True)
        self.assertEqual(committed["state"], "COMMITTED")
        self.assertEqual(client.rpc("decide_publication", request_id="p", commit=False)["state"], "COMMITTED")
        grant = client.acquire_grant(committed["manifest"], "reader", 60)
        self.assertEqual(client.fetch_object(oid, grant["grant_id"]), b"payload")

    def test_missing_shard_and_abort_never_publish_partial_manifest(self):
        client = self.a.client
        oid = client.put_object(b"payload", self.layout(7))
        self.assertCode("MISS", client.rpc, "prepare_manifest", objects=[oid, "f" * 64],
                        metadata={}, request_id="missing")
        prepared = client.rpc("prepare_manifest", objects=[oid], metadata={}, request_id="abort")
        self.assertEqual(client.rpc("decide_publication", request_id="abort", commit=False)["state"], "ABORTED")
        self.assertEqual(client.rpc("decide_publication", request_id="abort", commit=True)["state"], "ABORTED")
        self.assertCode("MISS", client.get_manifest, prepared["manifest"])
        self.assertEqual(client.gc()["freed_bytes"], 7)

    def test_grants_retirement_and_stale_references(self):
        client = self.a.client
        oid, mid, gid, _ = self.bundle(client)
        client.retire_manifest(mid)
        self.assertCode("MISS", client.acquire_grant, mid, "new-reader", 1)
        self.assertEqual(client.gc()["freed_bytes"], 0)
        self.assertEqual(client.fetch_object(oid, gid), b"abcdefghijklmnop")
        client.release_grant(gid)
        self.assertEqual(client.gc()["freed_bytes"], 16)
        self.assertCode("MISS", client.fetch_object, oid, gid)
        self.assertCode("EXPIRED", client.acquire_grant, mid, "reader", 60, "grant-publish")

    def test_capacity_is_bounded_and_duplicate_upload_does_not_allocate(self):
        client = self.a.client
        payload = b"a" * (1024 * 1024)
        oid = client.put_object(payload, self.layout(len(payload)))
        self.assertEqual(client.put_object(payload, self.layout(len(payload))), oid)
        self.assertEqual(client.stats()["used_bytes"], len(payload))
        self.assertCode("CAPACITY", client.put_object, b"x", self.layout(1))
        self.assertEqual(client.stats()["object_count"], 1)
        self.assertEqual(client.gc()["freed_bytes"], len(payload))

    def test_manifest_metadata_budget_and_total_rpc_budget(self):
        client = self.a.client
        oid = client.put_object(b"x", self.layout(1))
        metadata = {"description": "x" * (80 * 1024)}
        mid = client.publish_manifest([oid], metadata, "large-but-bounded")
        self.assertEqual(client.get_manifest(mid)["metadata"], metadata)
        self.assertCode("INVALID", client.publish_manifest, [oid],
                        {"description": "x" * (800 * 1024)}, "metadata-too-large")
        self.assertCode("TOO_LARGE", dist.validate_manifest_bounds,
                        [f"{i:064x}" for i in range(4096)],
                        {"description": "x" * (768 * 1024 - 100)}, "envelope-too-large")
        self.assertEqual(client.stats()["object_count"], 1)

    def test_namespace_auth_layout_and_idempotence_validation(self):
        client = self.a.client
        self.assertCode("AUTH", dist.KVShareClient(self.a.url, "model-contract-v1", "wrong").stats)
        self.assertCode("NAMESPACE", dist.KVShareClient(self.a.url, "different", "test-cluster-secret").stats)
        self.assertCode("INVALID", client.put_object, b"a", self.layout(2))
        oid, mid, gid, layout = self.bundle(client)
        wrong = {**layout, "kv_head": 3}
        self.assertCode("LAYOUT", client.fetch_object, oid, gid, wrong)
        self.assertNotEqual(client.put_object(b"abcdefghijklmnop", wrong), oid)
        self.assertCode("CONFLICT", client.publish_manifest, [oid], {"different": True}, "publish")
        self.assertEqual(client.get_manifest(mid)["objects"], [oid])

    def test_source_loss_returns_clear_unavailable(self):
        client = self.a.client
        oid, _, gid, _ = self.bundle(client)
        self.a.stop()
        self.assertCode("UNAVAILABLE", client.fetch_object, oid, gid)

    def test_migration_restart_fencing_frontier_and_unknown_commit_outcome(self):
        client = self.a.client
        _, mid, _, _ = self.bundle(client)
        request = client.rpc("create_request", request_id="r", owner="rank-group-a", checkpoint=mid)
        self.assertEqual(request["epoch"], 1)
        token = hashlib.sha256(b"token-0").hexdigest()
        self.assertEqual(client.rpc("commit_output", request_id="r", owner="rank-group-a", epoch=1,
                                    token_index=0, token_digest=token)["frontier"], 1)
        prepared = client.rpc("prepare_migration", request_id="r", owner="rank-group-a", epoch=1,
                              target="rank-group-b", checkpoint=mid, decision_id="move")
        self.assertEqual(prepared["frontier"], 1)
        self.assertCode("QUIESCED", client.rpc, "commit_output", request_id="r", owner="rank-group-a", epoch=1,
                        token_index=1, token_digest=token)
        self.assertCode("NOT_READY", client.rpc, "decide_migration", decision_id="move", commit=True)
        self.a.restart()
        client = self.a.client
        self.assertEqual(client.rpc("get_migration", decision_id="move")["state"], "PREPARED")
        client.rpc("mark_migration_ready", decision_id="move", target="rank-group-b", checkpoint=mid)
        self.assertEqual(client.rpc("decide_migration", decision_id="move", commit=True)["state"], "COMMITTED")
        self.a.restart()  # A coordinator that lost the ACK must read the committed decision.
        client = self.a.client
        self.assertEqual(client.rpc("decide_migration", decision_id="move", commit=False)["state"], "COMMITTED")
        request = client.rpc("get_request", request_id="r")
        self.assertEqual((request["owner"], request["epoch"], request["frontier"]), ("rank-group-b", 2, 1))
        self.assertCode("FENCED", client.rpc, "commit_output", request_id="r", owner="rank-group-a", epoch=1,
                        token_index=1, token_digest=token)
        self.assertEqual(client.rpc("commit_output", request_id="r", owner="rank-group-b", epoch=2,
                                    token_index=0, token_digest=token)["status"], "DUPLICATE")
        self.assertCode("CONFLICT", client.rpc, "commit_output", request_id="r", owner="rank-group-b", epoch=2,
                        token_index=0, token_digest="b" * 64)
        self.assertCode("GAP", client.rpc, "commit_output", request_id="r", owner="rank-group-b", epoch=2,
                        token_index=2, token_digest=token)

    def test_abort_and_commit_race_has_one_durable_decision(self):
        client = self.a.client
        _, mid, _, _ = self.bundle(client)
        client.rpc("create_request", request_id="r", owner="a", checkpoint=mid)
        client.rpc("prepare_migration", request_id="r", owner="a", epoch=1, target="b", checkpoint=mid, decision_id="m")
        client.rpc("mark_migration_ready", decision_id="m", target="b", checkpoint=mid)
        barrier = threading.Barrier(2)
        def decide(commit):
            barrier.wait()
            return client.rpc("decide_migration", decision_id="m", commit=commit)["state"]
        with ThreadPoolExecutor(2) as pool:
            decisions = list(pool.map(decide, [True, False]))
        self.assertEqual(decisions[0], decisions[1])
        self.a.restart()
        self.assertEqual(self.a.client.rpc("get_migration", decision_id="m")["state"], decisions[0])

    def test_finished_request_releases_checkpoint_without_epoch_reuse(self):
        client = self.a.client
        oid, mid, gid, _ = self.bundle(client)
        client.rpc("create_request", request_id="r", owner="a", checkpoint=mid)
        client.release_grant(gid)
        client.retire_manifest(mid)
        self.assertEqual(client.gc()["freed_bytes"], 0)  # Running checkpoint is a root.
        result = client.rpc("finish_request", request_id="r", owner="a", epoch=1)
        self.assertEqual((result["owner"], result["epoch"]), ("", 2))
        self.assertEqual(client.gc()["freed_bytes"], 16)
        self.a.restart()
        client = self.a.client
        result = client.rpc("create_request", request_id="r", owner="a", checkpoint=mid)
        self.assertEqual((result["owner"], result["epoch"]), ("", 2))
        self.assertCode("FENCED", client.rpc, "commit_output", request_id="r", owner="a", epoch=1,
                        token_index=0, token_digest="a" * 64)


class LocalLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = dist.KVShareStore(self.temp.name, "n", 64, max_inflight_bytes=8)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.store.close)

    def bundle(self):
        oid = self.store.put_object(b"12345678", {"format": "raw-test"})["object_id"]
        prepared = self.store.prepare_manifest([oid], {}, "p")
        self.store.decide_publication("p", True)
        grant = self.store.acquire_grant(prepared["manifest"], "r", 0.1)
        return oid, prepared["manifest"], grant["grant_id"]

    def test_expiry_and_gc_cannot_release_admitted_transfer(self):
        oid, mid, gid = self.bundle()
        with self.store.pin_object(oid, gid) as (payload, _):
            self.store.retire_manifest(mid)
            time.sleep(0.15)
            self.assertEqual(self.store.gc()["freed_bytes"], 0)
            self.assertEqual(payload, b"12345678")
            self.assertEqual(self.store.stats()["pinned_objects"], 1)
        self.assertEqual(self.store.gc()["freed_bytes"], 8)

    def test_staging_backpressure_and_single_process_ownership(self):
        oid, _, gid = self.bundle()
        with self.store.pin_object(oid, gid):
            with self.assertRaises(dist.ShareError) as ctx:
                with self.store.pin_object(oid, gid):
                    pass
            self.assertEqual(ctx.exception.code, "BUSY")
        with self.assertRaises(dist.ShareError) as ctx:
            dist.KVShareStore(self.temp.name, "n", 64)
        self.assertEqual(ctx.exception.code, "BUSY")

    def test_corrupt_persisted_bytes_do_not_escape_checksum(self):
        oid, _, gid = self.bundle()
        self.store._db.execute("UPDATE objects SET payload=? WHERE id=?", (b"87654321", oid))
        with self.assertRaises(dist.ShareError) as ctx:
            with self.store.pin_object(oid, gid):
                pass
        self.assertEqual(ctx.exception.code, "CORRUPT")
        self.assertEqual(self.store.stats()["pinned_objects"], 0)


class ClientFaultTests(unittest.TestCase):
    def test_corrupt_truncated_and_timeout_transfers(self):
        layout = {"format": "raw"}
        oid = dist.content_id("n", layout, b"good")
        mode = ["corrupt"]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                if mode[0] == "timeout":
                    time.sleep(0.2)
                    return
                self.send_response(200)
                self.send_header("Content-Length", "4")
                self.send_header("X-KV-Object", oid)
                self.send_header("X-KV-Layout", base64.b64encode(json.dumps(layout).encode()).decode())
                self.end_headers()
                self.wfile.write(b"evil" if mode[0] == "corrupt" else b"go")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = dist.KVShareClient(f"http://127.0.0.1:{server.server_port}", "n", timeout=0.05)
        try:
            for fault, expected in (("corrupt", "CORRUPT"), ("truncated", "TRUNCATED"), ("timeout", "UNAVAILABLE")):
                mode[0] = fault
                with self.subTest(fault=fault), self.assertRaises(dist.ShareError) as ctx:
                    client.fetch_object(oid, "g")
                self.assertEqual(ctx.exception.code, expected)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main(verbosity=2)
