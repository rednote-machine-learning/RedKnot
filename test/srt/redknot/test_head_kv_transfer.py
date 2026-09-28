"""Real HTTP snapshot import into independent head KV pools."""
import sys
import threading
import copy
import json
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"python/sglang/srt/mem_cache"))
from head_kv import HeadPagePool, HeadKVManager, ReuseProof, SegmentPatch, SegmentWrite
from head_kv.distributed import KVShareClient, KVShareServer, KVShareStore, ShareError
from head_kv.transfer import export_snapshot, import_snapshot
import head_kv.transfer as transfer


def make_manager(name,capacity=16):
    m=HeadKVManager(HeadPagePool(capacity,4,8))
    m.create_request(name,context_id="P,D",contract="test-model")
    return m


def test_remote_import_cow_and_content_identity(tmp_path):
    a,b=make_manager("A"),make_manager("imported")
    k,v=torch.randn(10,8),torch.randn(10,8)
    a.update("A",writes=[SegmentWrite((0,0,"D"),k,v,tuple(range(10)),"source-prov")])
    store=KVShareStore(str(tmp_path),namespace="default",capacity_bytes=100000)
    with KVShareServer(store) as server:
        client=KVShareClient(server.url,"default")
        mid=export_snapshot(a,"A",client,operation_id="publish-a")
        import_snapshot(b,"imported",client,mid,holder="node-b")
        source=a.version("A").segments[(0,0,"D")]
        target=b.version("imported").segments[(0,0,"D")]
        assert [p.content_id for p in source.pages]==[p.content_id for p in target.pages]
        b.create_request("B",context_id="R,E,D",contract="test-model")
        b.share_segment("imported","B",(0,0,"D"),(0,0,"D"),positions=range(30,40),
                        proof=ReuseProof("policy_approximate","source-prov","R,E,D","explicit-policy"))
        b.update("B",patches=[SegmentPatch((0,0,"D"),(1,6),torch.zeros(2,8),torch.ones(2,8),"B-repair")])
        for manager,rid in [(a,"A"),(b,"imported")]:
            with manager.bind(rid) as lease:
                actual=lease.gather((0,0,"D"))
                torch.testing.assert_close(actual[0],k)
                torch.testing.assert_close(actual[1],v)
        with b.bind("B") as lease:
            changed=lease.gather((0,0,"D"))
            expected=k.clone(); expected[[1,6]]=0
            torch.testing.assert_close(changed[0],expected)
    store.close()
    for manager,names in [(a,["A"]),(b,["B","imported"])]:
        for name in names: manager.release_request(name)
        assert manager.stats()["free_pages"]==16


def test_import_failure_does_not_publish_partial_root(tmp_path):
    a,b=make_manager("A"),make_manager("B")
    a.update("A",writes=[SegmentWrite((0,0,"D"),torch.randn(10,8),torch.randn(10,8),tuple(range(10)),"prov")])
    store=KVShareStore(str(tmp_path),namespace="default",capacity_bytes=100000)
    with KVShareServer(store) as server:
        client=KVShareClient(server.url,"default")
        mid=export_snapshot(a,"A",client,operation_id="pub")
        original=client.fetch_object
        calls=[]
        def broken(*args,**kwargs):
            calls.append(1)
            if len(calls)==2: raise RuntimeError("injected source loss")
            return original(*args,**kwargs)
        client.fetch_object=broken
        with pytest.raises(RuntimeError,match="source loss"):
            import_snapshot(b,"B",client,mid,holder="b")
        assert not b.version("B").segments
        assert b.stats()["free_pages"]==16
        assert b.stats()["pending_transactions"]==0
    store.close()


def test_cancel_during_import_holds_destinations_until_staging_finishes(tmp_path):
    a,b=make_manager("A"),make_manager("B")
    a.update("A",writes=[SegmentWrite((0,0,"D"),torch.randn(10,8),torch.randn(10,8),tuple(range(10)),"prov")])
    store=KVShareStore(str(tmp_path),namespace="default",capacity_bytes=100000)
    with KVShareServer(store) as server:
        client=KVShareClient(server.url,"default")
        mid=export_snapshot(a,"A",client,operation_id="pub")
        original=client.fetch_object
        entered,finish=threading.Event(),threading.Event()
        def paused(*args,**kwargs):
            data=original(*args,**kwargs)
            entered.set()
            assert finish.wait(10)
            return data
        client.fetch_object=paused
        failures=[]
        def importing():
            try: import_snapshot(b,"B",client,mid,holder="b")
            except Exception as exc: failures.append(exc)
        thread=threading.Thread(target=importing)
        thread.start()
        assert entered.wait(10)
        b.release_request("B")
        assert b.stats()["reserved_pages"]==3
        finish.set()
        thread.join(10)
        assert not thread.is_alive()
        assert len(failures)==1
        assert b.stats()["free_pages"]==16
    store.close()


@pytest.fixture
def snapshot(tmp_path):
    a, b = make_manager("A"), make_manager("B")
    k, v = torch.randn(10, 8), torch.randn(10, 8)
    a.update("A", writes=[SegmentWrite((0, 0, "D"), k, v, tuple(range(10)), "prov")])
    store = KVShareStore(str(tmp_path), namespace="default", capacity_bytes=100000)
    try:
        with KVShareServer(store) as server:
            client = KVShareClient(server.url, "default")
            mid = export_snapshot(a, "A", client, operation_id="pub")
            yield a, b, client, mid, store
    finally:
        store.close()


def test_slow_transfer_renews_grant_even_after_manifest_retirement(snapshot):
    _, b, client, mid, store = snapshot
    original_fetch, original_renew = client.fetch_object, client.renew_grant
    fetched, renewals = [], []

    def renew(*args, **kwargs):
        renewals.append(1)
        return original_renew(*args, **kwargs)

    def slow_fetch(*args, **kwargs):
        payload = original_fetch(*args, **kwargs)
        if not fetched:
            fetched.append(1)
            client.retire_manifest(mid)
            time.sleep(0.4)  # Several TTL intervals; payload source is unchanged.
            assert client.gc()["freed_bytes"] == 0
        return payload

    client.fetch_object, client.renew_grant = slow_fetch, renew
    import_snapshot(b, "B", client, mid, holder="slow-reader", ttl_s=0.15)
    assert len(renewals) >= 2
    assert b.version("B").segments
    assert client.gc()["freed_bytes"] == 640  # Import released its last source grant.


def test_transient_read_retry_and_lost_publication_ack_are_idempotent(snapshot):
    a, b, client, mid, _ = snapshot
    original_fetch, original_publish = client.fetch_object, client.publish_manifest
    reads, publications = [], []

    def intermittent_fetch(*args, **kwargs):
        reads.append(1)
        if len(reads) == 1:
            raise ShareError("UNAVAILABLE", "transient source connection loss", 503)
        return original_fetch(*args, **kwargs)

    def lost_ack(*args, **kwargs):
        result = original_publish(*args, **kwargs)
        publications.append(result)
        if len(publications) == 1:
            raise ShareError("UNAVAILABLE", "commit ACK was lost", 503)
        return result

    client.fetch_object = intermittent_fetch
    import_snapshot(b, "B", client, mid, holder="retrying-reader", max_retries=1)
    assert len(reads) == 4
    client.publish_manifest = lost_ack
    assert export_snapshot(a, "A", client, operation_id="pub-again") == mid
    assert publications == [mid, mid]
    assert client.stats()["object_count"] == 3


@pytest.mark.parametrize("fault", ["positions", "shape", "overlap", "certificate", "namespace", "unreferenced"])
def test_invalid_manifest_plan_rejected_before_page_reservation(snapshot, fault):
    _, b, client, mid, _ = snapshot
    manifest = client.get_manifest(mid)
    meta = copy.deepcopy(manifest["metadata"])
    if fault == "positions":
        meta["segments"][0]["positions"] = [True] + list(range(1, 10))
    elif fault == "shape":
        meta["pages"][0]["layout"]["shape"][0] = 1
    elif fault == "overlap":
        duplicate = copy.deepcopy(meta["segments"][0])
        duplicate["key"][2] = "repeated-D"
        meta["segments"].append(duplicate)
    elif fault == "certificate":
        meta["segments"][0]["reuse_kind"] = "policy_approximate"
        meta["segments"][0]["validity_certificate"] = ""
    elif fault == "namespace":
        meta["namespace"] = "wrong-namespace"
    else:
        meta["segments"] = []
    broken = client.publish_manifest(manifest["objects"], meta, "malformed-" + fault)
    with pytest.raises(ValueError):
        import_snapshot(b, "B", client, broken, holder="b")
    assert not b.version("B").segments
    assert b.stats()["free_pages"] == 16
    assert b.stats()["high_water_pages"] == 0
    assert b.stats()["pending_transactions"] == 0


def test_conflicting_resident_state_identity_is_rejected(snapshot):
    _, b, client, mid, _ = snapshot
    import_snapshot(b, "B", client, mid, holder="b")
    b.create_request("C", context_id="P,D", contract="test-model")
    manifest = client.get_manifest(mid)
    meta = copy.deepcopy(manifest["metadata"])
    record = meta["pages"][0]
    gid = client.acquire_grant(mid, "tamper-test", 60)["grant_id"]
    original_oid = record["object_id"]
    payload = bytearray(client.fetch_object(original_oid, gid))
    payload[0] ^= 1
    client.release_grant(gid)
    # New content hash is valid, but it falsely claims the same resident StatePageID.
    record["object_id"] = client.put_object(bytes(payload), record["layout"])
    for segment in meta["segments"]:
        segment["pages"] = [record["object_id"] if oid == original_oid else oid for oid in segment["pages"]]
    broken = client.publish_manifest([r["object_id"] for r in meta["pages"]], meta, "conflicting-state")
    with pytest.raises(ValueError, match="conflicting payload/layout"):
        import_snapshot(b, "C", client, broken, holder="b")
    assert not b.version("C").segments
    assert b.stats()["live_pages"] == 3
    assert b.stats()["free_pages"] == 13
    assert b.stats()["pinned_pages"] == 0


def test_custom_transport_cannot_bypass_manager_content_check(snapshot):
    _, b, client, mid, _ = snapshot
    original = client.fetch_object

    def corrupt(*args, **kwargs):
        raw = bytearray(original(*args, **kwargs))
        raw[0] ^= 1
        return bytes(raw)

    client.fetch_object = corrupt
    with pytest.raises(ValueError, match="content identity"):
        import_snapshot(b, "B", client, mid, holder="b")
    assert not b.version("B").segments
    assert b.stats()["free_pages"] == 16


def test_oom_releases_identity_comparison_pins(snapshot):
    _, _, client, mid, _ = snapshot
    b = make_manager("B", capacity=3)
    import_snapshot(b, "B", client, mid, holder="b")
    b.create_request("C", context_id="P,D", contract="test-model")
    with pytest.raises(RuntimeError, match="available 0"):
        import_snapshot(b, "C", client, mid, holder="b")
    assert b.stats()["pinned_pages"] == 0
    assert b.stats()["live_pages"] == 3
    b.release_request("B")
    assert b.stats()["free_pages"] == 3


def test_interrupt_cleanup_and_deferred_device_retirement(snapshot, monkeypatch):
    _, b, client, mid, _ = snapshot
    class PendingDeviceEvent:
        ready = False
        def query(self):
            return self.ready
        def synchronize(self):
            raise AssertionError("aborted import must not claim device completion")

    event = PendingDeviceEvent()
    monkeypatch.setattr(transfer, "completion_event", lambda _: event)
    original, calls = client.fetch_object, []

    def interrupt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt("injected user cancellation")
        return original(*args, **kwargs)

    client.fetch_object = interrupt
    with pytest.raises(KeyboardInterrupt):
        import_snapshot(b, "B", client, mid, holder="b")
    assert not b.version("B").segments
    assert b.stats()["pending_transactions"] == 0
    assert b.stats()["retiring_pages"] == 3
    assert b.stats()["free_pages"] == 13
    event.ready = True
    assert b.stats()["free_pages"] == 16


def test_failure_before_transaction_construction_releases_reservations(snapshot, monkeypatch):
    _, b, client, mid, _ = snapshot
    original, calls = transfer.completion_event, []
    def fail_once(device):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("injected event allocation failure")
        return original(device)
    monkeypatch.setattr(transfer, "completion_event", fail_once)
    with pytest.raises(RuntimeError, match="event allocation"):
        import_snapshot(b, "B", client, mid, holder="b")
    assert b.stats()["free_pages"] == 16
    assert b.stats()["pinned_pages"] == 0
    assert b.stats()["pending_transactions"] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_bytes_roundtrip(tmp_path, dtype):
    a = HeadKVManager(HeadPagePool(8, 4, 8, dtype=dtype))
    b = HeadKVManager(HeadPagePool(8, 4, 8, dtype=dtype))
    for manager, name in ((a, "A"), (b, "B")):
        manager.create_request(name, context_id="ctx", contract="model")
    k, v = torch.randn(5, 8).to(dtype), torch.randn(5, 8).to(dtype)
    a.update("A", writes=[SegmentWrite((0, 0, "d"), k, v, tuple(range(5)), "prov")])
    store = KVShareStore(tmp_path, "default", 10000)
    try:
        with KVShareServer(store) as server:
            client = KVShareClient(server.url, "default")
            mid = export_snapshot(a, "A", client, operation_id="p")
            import_snapshot(b, "B", client, mid, holder="b")
            with b.bind("B") as lease:
                actual = lease.gather((0, 0, "d"))
                assert torch.equal(actual[0], k)
                assert torch.equal(actual[1], v)
    finally:
        store.close()


def test_large_multilayer_snapshot_and_compact_positions(tmp_path):
    a = HeadKVManager(HeadPagePool(256, 4, 2))
    b = HeadKVManager(HeadPagePool(256, 4, 2))
    for manager, name in ((a, "A"), (b, "B")):
        manager.create_request(name, context_id="ctx", contract="32-layer-8-head-test")
    k, v = torch.randn(4, 2), torch.randn(4, 2)
    a.update("A", writes=[SegmentWrite((i // 8, i % 8, "d"), k, v, tuple(range(4)), "provenance")
                           for i in range(256)])
    store = KVShareStore(tmp_path, "default", 1024 * 1024)
    try:
        with KVShareServer(store) as server:
            client = KVShareClient(server.url, "default")
            mid = export_snapshot(a, "A", client, operation_id="full-model-shape")
            meta = client.get_manifest(mid)["metadata"]
            assert len(json.dumps(meta).encode()) > 64 * 1024
            assert all(s["positions"] == {"runs": [[0, 4, 1]]} for s in meta["segments"])
            import_snapshot(b, "B", client, mid, holder="b")
            assert len(b.version("B").segments) == 256
            with b.bind("B") as lease:
                actual = lease.gather((31, 7, "d"))
                assert torch.equal(actual[0], k)
    finally:
        store.close()


def test_oversized_export_fails_before_any_upload_and_releases_source_pin(tmp_path):
    a = make_manager("A")
    a.update("A", writes=[SegmentWrite((0, 0, "D"), torch.zeros(4, 8), torch.ones(4, 8),
                                       tuple(range(4)), "p" * (800 * 1024))])
    store = KVShareStore(tmp_path, "default", 1024 * 1024)
    try:
        with KVShareServer(store) as server:
            client = KVShareClient(server.url, "default")
            with pytest.raises(ShareError, match="oversized metadata"):
                export_snapshot(a, "A", client, operation_id="too-large")
            assert client.stats()["object_count"] == 0
            assert client.stats()["used_bytes"] == 0
            assert a.stats()["pinned_pages"] == 0
    finally:
        store.close()


def test_compressed_position_expansion_is_bounded_before_reservation(snapshot):
    _, b, client, mid, _ = snapshot
    manifest = client.get_manifest(mid)
    meta = copy.deepcopy(manifest["metadata"])
    meta["segments"][0]["positions"] = {"runs": [[0, 2**60, 1]]}
    broken = client.publish_manifest(manifest["objects"], meta, "run-expansion-bomb")
    with pytest.raises(ValueError, match="oversized logical position run"):
        import_snapshot(b, "B", client, broken, holder="b")
    assert b.stats()["high_water_pages"] == 0


@pytest.mark.parametrize("axis", ["layer", "kv_head"])
def test_manifest_cannot_relabel_page_into_another_backing_group(snapshot, axis):
    _, b, client, mid, _ = snapshot
    manifest = client.get_manifest(mid)
    meta = copy.deepcopy(manifest["metadata"])
    meta["segments"][0]["key"][0 if axis == "layer" else 1] += 1
    broken = client.publish_manifest(manifest["objects"], meta, "wrong-" + axis)
    with pytest.raises(ValueError, match="different layer/KV group"):
        import_snapshot(b, "B", client, broken, holder="b")
    assert b.stats()["high_water_pages"] == 0
    assert not b.version("B").segments


def test_resident_state_identity_cannot_claim_a_new_group_even_with_same_bytes(snapshot):
    _, b, client, mid, _ = snapshot
    import_snapshot(b, "B", client, mid, holder="b")
    b.create_request("C", context_id="P,D", contract="test-model")
    meta = copy.deepcopy(client.get_manifest(mid)["metadata"])
    gid = client.acquire_grant(mid, "forgery-test", 60)["grant_id"]
    replacements = {}
    for record in meta["pages"]:
        old_oid = record["object_id"]
        payload = client.fetch_object(old_oid, gid)
        record["layout"]["layer"] = 1
        record["object_id"] = client.put_object(payload, record["layout"])
        replacements[old_oid] = record["object_id"]
    client.release_grant(gid)
    for segment in meta["segments"]:
        segment["key"][0] = 1
        segment["pages"] = [replacements[oid] for oid in segment["pages"]]
    broken = client.publish_manifest(list(replacements.values()), meta, "forged-resident-group")
    with pytest.raises(ValueError, match="different contract/layer/KV group"):
        import_snapshot(b, "C", client, broken, holder="b")
    assert not b.version("C").segments
    assert b.stats()["pinned_pages"] == 0
    assert b.stats()["free_pages"] == 13
