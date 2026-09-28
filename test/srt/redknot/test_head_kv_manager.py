"""Lifecycle/COW regression tests; independent of the full SGLang server."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache"))
from head_kv import (CapacityError, HeadKVManager, HeadPagePool, ReuseProof,
                     SegmentPatch, SegmentWrite, StaleReference)
from head_kv.attention import paged_attention


class Gate:
    def __init__(self):
        self.ready = False
    def query(self):
        return self.ready
    def synchronize(self):
        if not self.ready:
            raise RuntimeError("test gate not ready")


def setup(capacity=32, heads=2, length=10):
    torch.manual_seed(19)
    pool = HeadPagePool(capacity, 4, 8)
    manager = HeadKVManager(pool)
    manager.create_request("source", context_id="P,D", contract="test-weights-v1")
    data = [(torch.randn(length,8), torch.randn(length,8)) for _ in range(heads)]
    manager.update("source", writes=[SegmentWrite((0,h,"D"),k,v,tuple(range(length)),"source-D")
                                     for h,(k,v) in enumerate(data)])
    return manager, data


def assert_data(manager, req, key, k, v):
    with manager.bind(req) as lease:
        actual = lease.gather(key)
        torch.testing.assert_close(actual[0], k)
        torch.testing.assert_close(actual[1], v)


def test_nonprefix_reordered_occurrence_and_partial_head_cow():
    m, data = setup()
    m.create_request("target", context_id="R,E,D", contract="test-weights-v1")
    proof = ReuseProof("policy_approximate", "source-D", "R,E,D", "test-explicit-policy")
    for h in range(2):
        m.share_segment("source","target",(0,h,"D"),(0,h,"moved-D"), proof=proof,positions=range(20,30))
    before = m.stats()
    src = m.version("source")
    patch_k,patch_v = torch.ones(3,8), -torch.ones(3,8)
    m.update("target",patches=[SegmentPatch((0,1,"moved-D"),(0,1,5),patch_k,patch_v,"target-repair")])
    tgt = m.version("target")
    assert tgt.segments[(0,0,"moved-D")].pages == src.segments[(0,0,"D")].pages
    assert tgt.segments[(0,1,"moved-D")].pages[2] == src.segments[(0,1,"D")].pages[2]
    assert m.stats()["live_pages"] - before["live_pages"] == 2
    assert m.stats()["copied_bytes"] - before["copied_bytes"] == 2*5*8*4
    expected_k,expected_v = (x.clone() for x in data[1])
    expected_k[[0,1,5]],expected_v[[0,1,5]] = patch_k,patch_v
    assert_data(m,"target",(0,1,"moved-D"),expected_k,expected_v)
    assert_data(m,"source",(0,1,"D"),*data[1])
    m.release_request("source")
    m.release_request("target")
    assert m.stats()["free_pages"] == 32


def test_entire_page_rewrite_copies_zero_old_payload():
    m,_ = setup(heads=1)
    m.fork("source","child")
    old = m.pool.copied_bytes
    m.update("child",patches=[SegmentPatch((0,0,"D"),(0,1,2,3),torch.ones(4,8),torch.zeros(4,8),"repair")])
    assert m.pool.copied_bytes == old
    assert m.stats()["live_pages"] == 4


def test_oom_is_atomic_across_multiple_heads():
    m,data = setup(capacity=7)
    m.fork("source","child")
    old = m.version("child")
    with pytest.raises(CapacityError):
        m.begin_update("child",patches=[SegmentPatch((0,h,"D"),(0,),torch.ones(1,8),torch.zeros(1,8),"repair") for h in range(2)])
    assert m.version("child") is old
    assert m.stats()["reserved_pages"] == 0
    assert m.stats()["free_pages"] == 1
    for h in range(2):
        assert_data(m,"child",(0,h,"D"),*data[h])


def test_cancel_waits_for_consumer_event_and_slot_generation_changes():
    m,_ = setup(capacity=3,heads=1)
    refs = m.version("source").segments[(0,0,"D")].pages
    read = m.bind("source")
    gate = Gate()
    read.complete(gate)
    m.release_request("source")
    assert m.stats()["retiring_pages"] == 3
    with pytest.raises(CapacityError):
        m.pool.reserve(1)
    gate.ready = True
    assert m.pool.collect() == 3
    new = m.pool.reserve(3)
    assert {r.slot for r in refs} == {r.slot for r in new}
    for r in refs:
        with pytest.raises(StaleReference):
            m.pool.read_page(r)
    m.pool.release(new)


def test_pending_commit_cancel_and_reused_request_id():
    m,_ = setup(heads=1)
    tx = m.begin_update("source",patches=[SegmentPatch((0,0,"D"),(0,),torch.ones(1,8),torch.zeros(1,8),"repair")])
    gate = Gate()
    tx.event = gate
    assert tx.commit() is None
    with pytest.raises(RuntimeError):
        m.begin_update("source")
    m.release_request("source")
    m.create_request("source",context_id="new")
    with pytest.raises(RuntimeError):
        tx.commit()
    gate.ready = True
    m.pool.collect()
    assert m.stats()["live_pages"] == 0


def test_terminal_event_failure_quarantines_once_and_rejects_existing_reader():
    class BrokenEvent:
        def query(self): raise RuntimeError("device lost")
    m,_=setup(heads=1)
    read=m.bind("source")
    tx=m.begin_update("source",patches=[SegmentPatch((0,0,"D"),(0,),torch.ones(1,8),torch.zeros(1,8),"repair")])
    tx.event=BrokenEvent()
    with pytest.raises(RuntimeError,match="device lost"):
        tx.abort()
    assert tx.status=="QUARANTINED"
    tx.abort()  # Must not decrement owners twice after partial retirement.
    with pytest.raises(RuntimeError,match="quarantined"):
        read.descriptor(0,1)
    with pytest.raises(RuntimeError,match="quarantined"):
        read.gather((0,0,"D"))


def test_shared_ancestor_and_repeated_occurrence_protect_source():
    m,data = setup(heads=1)
    m.cache("source","cache")
    m.fork("source","child")
    m.share_segment("source","child",(0,0,"D"),(0,0,"D-again"),positions=range(20,30),
                    proof=ReuseProof("policy_approximate","source-D","P,D","explicit-repeat"))
    m.release_request("source")
    m.update("child",patches=[SegmentPatch((0,0,"D-again"),(0,),torch.ones(1,8),torch.zeros(1,8),"repair")])
    assert_data(m,"child",(0,0,"D"),*data[0])
    m.release_request("child")
    assert m.stats()["live_pages"] == 3
    m.evict("cache")
    assert m.stats()["free_pages"] == 32


def test_append_partial_shared_tail_and_truncate():
    m,data = setup(heads=1)
    m.fork("source","child")
    k,v = torch.ones(3,8),torch.zeros(3,8)
    m.append("child",(0,0,"D"),k,v,(10,11,12),provenance="append")
    assert_data(m,"source",(0,0,"D"),*data[0])
    assert_data(m,"child",(0,0,"D"),torch.cat((data[0][0],k)),torch.cat((data[0][1],v)))
    m.truncate_segment("child",(0,0,"D"),9)
    m.append("child",(0,0,"D"),k[:1],v[:1],(9,),provenance="post-truncate")
    assert_data(m,"child",(0,0,"D"),torch.cat((data[0][0][:9],k[:1])),torch.cat((data[0][1][:9],v[:1])))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_append_many_initial_heads_shared_tail_and_new_head_commit_once(monkeypatch, device):
    pool = HeadPagePool(24, 4, 8, device=device)
    m = HeadKVManager(pool)
    m.create_request("source", context_id="ctx")
    data = [(torch.randn(n, 8, device=device), torch.randn(n, 8, device=device)) for n in (6, 8)]
    initial = [SegmentWrite((0, h, "D"), k, v, tuple(range(len(k))), "initial",
                            "rope-v1", "policy_approximate", "adapter-cert")
               for h, (k, v) in enumerate(data)]
    version = m.append_many("source", appends=initial, expected_epoch=0)
    assert version.epoch == 1
    m.fork("source", "child")
    before = m.stats()
    payloads = [(torch.randn(n, 8, device=device), torch.randn(n, 8, device=device)) for n in (3, 1, 2)]
    appends = [SegmentWrite((0, h, "D"), k, v,
                            tuple(range((6, 8, 0)[h], (6, 8, 0)[h] + len(k))),
                            "append", "rope-v1")
               for h, (k, v) in enumerate(payloads)]
    cat_rows, original_cat = [], torch.cat

    def tracked_cat(tensors, *args, **kwargs):
        cat_rows.append(tuple(len(tensor) for tensor in tensors))
        return original_cat(tensors, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "cat", tracked_cat)
        child = m.append_many("child", appends=appends, expected_epoch=0)
    assert child.epoch == 1  # All heads publish as one root, not one per head.
    assert cat_rows == [(2, 3), (2, 3)]  # K/V copy only the partial tail.
    assert m.stats()["live_pages"] - before["live_pages"] == 4
    assert m.stats()["copied_bytes"] - before["copied_bytes"] == 2 * 2 * 8 * 4
    for h in range(2):
        old = version.segments[(0, h, "D")]
        new = child.segments[(0, h, "D")]
        full = len(data[h][0]) // 4
        assert new.pages[:full] == old.pages[:full]
        if len(data[h][0]) % 4:
            assert new.pages[full] != old.pages[full]
        assert new.reuse_kind == "policy_approximate"
        assert new.validity_certificate == "adapter-cert"
        assert_data(m, "source", (0, h, "D"), *data[h])
        assert_data(m, "child", (0, h, "D"),
                    torch.cat((data[h][0], payloads[h][0])),
                    torch.cat((data[h][1], payloads[h][1])))
    assert_data(m, "child", (0, 2, "D"), *payloads[2])
    m.release_request("source")
    m.release_request("child")
    if device == "cuda":
        torch.cuda.synchronize(pool.device)
    assert m.stats()["free_pages"] == 24


@pytest.mark.parametrize("initial", [True, False])
def test_append_many_oom_is_atomic_before_first_head_write(initial):
    if initial:
        m = HeadKVManager(HeadPagePool(3, 4, 8))
        m.create_request("source", context_id="ctx")
        n, start = 6, 0
    else:
        m, _ = setup(capacity=7, heads=2, length=6)
        n, start = 5, 6
    before, version = m.stats(), m.version("source")
    appends = [SegmentWrite((0, h, "D"), torch.ones(n, 8), torch.zeros(n, 8),
                            tuple(range(start, start + n)), "append") for h in range(2)]
    with pytest.raises(CapacityError):
        m.append_many("source", appends=appends)
    assert m.version("source") is version
    after = m.stats()
    assert after["written_bytes"] == before["written_bytes"]
    assert after["free_pages"] == before["free_pages"]
    assert after["reserved_pages"] == after["pinned_pages"] == after["pending_transactions"] == 0


@pytest.mark.parametrize("invalid", ["duplicate_key", "duplicate_position", "negative_position",
                                     "position_basis", "new_certificate", "stale_epoch"])
def test_append_many_validates_entire_batch_before_reserving(invalid):
    from dataclasses import replace

    m, _ = setup(heads=2, length=6)
    before, old = m.stats(), m.version("source")
    appends = [SegmentWrite((0, h, "D"), torch.ones(1, 8), torch.zeros(1, 8), (6,), "append")
               for h in range(2)]
    changes = {
        "duplicate_key": {"key": (0, 0, "D")},
        "duplicate_position": {"positions": (5,)},
        "negative_position": {"positions": (-1,)},
        "position_basis": {"position_basis": "untransformed-rope"},
        "new_certificate": {"key": (0, 2, "D"), "reuse_kind": "policy_approximate"},
    }
    if invalid != "stale_epoch":
        appends[1] = replace(appends[1], **changes[invalid])
    with pytest.raises((ValueError, RuntimeError)):
        m.append_many("source", appends=appends,
                      expected_epoch=old.epoch + int(invalid == "stale_epoch"))
    assert m.version("source") is old
    assert m.stats()["written_bytes"] == before["written_bytes"]
    assert m.stats()["high_water_pages"] == before["high_water_pages"]
    assert m.stats()["free_pages"] == before["free_pages"]


def test_append_many_interrupted_second_head_waits_for_event_before_reclamation(monkeypatch):
    import head_kv.manager as manager_module

    m, data = setup(capacity=12, heads=2, length=6)
    before, old = m.stats(), m.version("source")
    gate = Gate()
    writes, original = [], m.pool.write_page

    def interrupted(ref, k, v):
        writes.append(ref)
        if len(writes) == 2:
            raise KeyboardInterrupt("second head interrupted")
        return original(ref, k, v)

    with monkeypatch.context() as patch:
        patch.setattr(m.pool, "write_page", interrupted)
        patch.setattr(manager_module, "completion_event", lambda device: gate)
        with pytest.raises(KeyboardInterrupt, match="second head"):
            m.append_many("source", appends=[SegmentWrite(
                (0, h, "D"), torch.ones(1, 8), torch.zeros(1, 8), (6,), "append")
                for h in range(2)])
    assert m.version("source") is old
    assert m.stats()["free_pages"] == before["free_pages"] - 2
    assert m.stats()["retiring_pages"] == 2
    # Pending device dependencies protect both old read sources and destinations;
    # the host read-lease pins themselves have already been released.
    assert m.stats()["pinned_pages"] == 6
    assert all(page.pins == 0 for page in m.pool._pages.values())
    assert m.stats()["pending_transactions"] == 0
    for h in range(2):
        assert_data(m, "source", (0, h, "D"), *data[h])
    gate.ready = True
    m.pool.collect()
    assert m.stats()["free_pages"] == before["free_pages"]
    assert m.stats()["pinned_pages"] == 0
    assert not m.pool.poisoned


def test_append_many_empty_existing_segment_preserves_version_and_scalar_missing_key():
    m, _ = setup(heads=1)
    old = m.version("source")
    empty = torch.empty(0, 8)
    append = SegmentWrite((0, 0, "D"), empty, empty, (), "unused")
    assert m.append_many("source", appends=[append]) is old
    assert m.append_many("source", appends=[]) is old
    assert m.append("source", append.key, empty, empty, (), provenance="unused") is old
    with pytest.raises(KeyError):
        m.append("source", (0, 1, "missing"), empty, empty, (), provenance="unused")


def test_exact_context_and_rope_relocation_rejected():
    m,_ = setup(heads=1)
    m.create_request("target",context_id="new",contract="test-weights-v1")
    with pytest.raises(ValueError,match="identical context"):
        m.share_segment("source","target",(0,0,"D"),(0,0,"D"),proof=ReuseProof("exact_context","source-D","new"))
    m.update("source",writes=[SegmentWrite((0,0,"R"),torch.ones(3,8),torch.zeros(3,8),(0,1,2),"rope-source","rope-v1")])
    with pytest.raises(ValueError,match="transform"):
        m.share_segment("source","target",(0,0,"R"),(0,0,"R"),positions=(20,21,22),
                        proof=ReuseProof("certified_transform","rope-source","new","adapter-cert"))


def test_exact_copy_preserves_approximate_lineage_and_certificate():
    m, _ = setup(heads=1)
    for name in ("approx", "copy"):
        m.create_request(name, context_id="changed-context", contract="test-weights-v1")
    m.share_segment("source", "approx", (0, 0, "D"), (0, 0, "D"),
                    proof=ReuseProof("policy_approximate", "source-D", "changed-context", "policy-cert"))
    m.share_segment("approx", "copy", (0, 0, "D"), (0, 0, "D"),
                    proof=ReuseProof("exact_context", "source-D", "changed-context"))
    copied = m.version("copy").segments[(0, 0, "D")]
    assert copied.reuse_kind == "policy_approximate"
    assert copied.validity_certificate == "policy-cert"


def test_transform_of_approximate_state_stays_approximate_with_bounded_lineage():
    m, _ = setup(heads=1)
    previous = "source"
    for i in range(30):
        target, context = f"lineage-{i}", f"context-{i}"
        m.create_request(target, context_id=context, contract="test-weights-v1")
        kind = "policy_approximate" if i == 0 else "certified_transform"
        m.share_segment(previous, target, (0, 0, "D"), (0, 0, "D"),
                        proof=ReuseProof(kind, "source-D", context, f"adapter-cert-{i}"))
        segment = m.version(target).segments[(0, 0, "D")]
        assert segment.reuse_kind == "policy_approximate"
        assert len(segment.validity_certificate) <= 80
        previous = target
    assert m.stats()["live_pages"] == 3


def test_nonexact_fresh_write_requires_explicit_attestation():
    pool = HeadPagePool(4, 4, 8)
    manager = HeadKVManager(pool)
    manager.create_request("r", context_id="ctx")
    args = ((0, 0, "D"), torch.ones(4, 8), torch.zeros(4, 8), (0, 1, 2, 3), "prov")
    with pytest.raises(ValueError, match="validity certificate"):
        manager.update("r", writes=[SegmentWrite(*args, reuse_kind="policy_approximate")])
    assert manager.stats()["high_water_pages"] == 0
    manager.update("r", writes=[SegmentWrite(*args, reuse_kind="policy_approximate",
                                            validity_certificate="adapter-attestation")])
    assert manager.version("r").segments[(0, 0, "D")].validity_certificate == "adapter-attestation"


@pytest.mark.parametrize("operation", ["write", "patch", "append"])
def test_interrupt_during_staging_releases_reservations_and_source_pins(monkeypatch, operation):
    m, _ = setup(heads=1)
    old = m.version("source")
    before = m.stats()

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("injected staging interruption")

    method = "clone_patch" if operation == "patch" else "write_page"
    monkeypatch.setattr(m.pool, method, interrupted)
    k, v = torch.ones(1, 8), torch.zeros(1, 8)
    with pytest.raises(KeyboardInterrupt):
        if operation == "write":
            m.begin_update("source", writes=[SegmentWrite((0, 1, "other"), k, v, (0,), "prov")])
        elif operation == "patch":
            m.begin_update("source", patches=[SegmentPatch((0, 0, "D"), (0,), k, v, "repair")])
        else:
            m.append("source", (0, 0, "D"), k, v, (10,), provenance="append")
    assert m.version("source") is old
    assert m.stats()["free_pages"] == before["free_pages"]
    assert m.stats()["pinned_pages"] == 0
    assert m.stats()["pending_transactions"] == 0


def test_interrupted_post_publication_cleanup_quarantines_unknown_outcome(monkeypatch):
    m, _ = setup(heads=1)
    tx = m.begin_update("source", patches=[SegmentPatch((0, 0, "D"), (0,),
                                                       torch.ones(1, 8), torch.zeros(1, 8), "repair")])
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("injected post-publication interruption")
    monkeypatch.setattr(m.pool, "release", interrupted)
    with pytest.raises(KeyboardInterrupt):
        tx.commit(wait=True)
    assert tx.status == "QUARANTINED"
    assert m.pool.poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        m.bind("source")


@pytest.mark.parametrize("failure", [KeyboardInterrupt, MemoryError])
def test_partial_reservation_metadata_failure_restores_all_slots(monkeypatch, failure):
    import head_kv.pool as pool_module
    pool = HeadPagePool(4, 4, 8)
    original, calls = pool_module._Page, []

    def fail_second(ref):
        calls.append(ref)
        if len(calls) == 2:
            raise failure("injected metadata allocation failure")
        return original(ref)

    monkeypatch.setattr(pool_module, "_Page", fail_second)
    with pytest.raises(failure):
        pool.reserve(3)
    assert pool.stats()["free_pages"] == 4
    assert pool.stats()["reserved_pages"] == 0
    monkeypatch.setattr(pool_module, "_Page", original)
    refs = pool.reserve(4)
    for failed_ref in calls:
        new = next(ref for ref in refs if ref.slot == failed_ref.slot)
        assert new.generation > failed_ref.generation
        assert new.content_id != failed_ref.content_id
    pool.release(refs)
    assert pool.stats()["free_pages"] == 4


def test_committed_repair_attention_matches_independent_dense_reference():
    m,data = setup(heads=2)
    m.fork("source","child")
    k,v = torch.randn(2,8),torch.randn(2,8)
    m.update("child",patches=[SegmentPatch((0,1,"D"),(2,8),k,v,"new")])
    q = torch.randn(4,2,8)
    pos = torch.tensor([5,9])
    with m.bind("child") as lease:
        out = paged_attention(q,**lease.descriptor(0,2),query_positions=pos,num_q_per_kv=2,backend="torch")
        densek = torch.stack([lease.gather((0,h,"D"))[0] for h in range(2)]).repeat_interleave(2,0)
        densev = torch.stack([lease.gather((0,h,"D"))[1] for h in range(2)]).repeat_interleave(2,0)
        mask = torch.arange(10)[None,:] <= pos[:,None]
        expected = torch.nn.functional.scaled_dot_product_attention(q,densek,densev,attn_mask=mask)
    torch.testing.assert_close(out,expected,atol=1e-6,rtol=1e-5)


def test_randomized_branch_repairs_against_private_tensor_oracle():
    import random
    rng = random.Random(47)
    m,data = setup(capacity=128,heads=2,length=13)
    oracle = {"source": [(k.clone(),v.clone()) for k,v in data]}
    for step in range(90):
        if len(oracle)<8 and rng.random()<.3:
            src = rng.choice(list(oracle))
            dst = f"b{step}"
            m.fork(src,dst)
            oracle[dst] = [(k.clone(),v.clone()) for k,v in oracle[src]]
        else:
            req = rng.choice(list(oracle))
            h = rng.randrange(2)
            indices = rng.sample(range(13),rng.randint(1,5))
            k,v = torch.randn(len(indices),8),torch.randn(len(indices),8)
            m.update(req,patches=[SegmentPatch((0,h,"D"),tuple(indices),k,v,f"repair-{step}")])
            oracle[req][h][0][indices] = k
            oracle[req][h][1][indices] = v
        for req,heads in oracle.items():
            for h,kv in enumerate(heads):
                assert_data(m,req,(0,h,"D"),*kv)
    for req in oracle:
        m.release_request(req)
    assert m.stats()["free_pages"] == 128


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_cuda_copy_repair_and_cross_stream_retirement():
    pool=HeadPagePool(12,16,32,dtype=torch.bfloat16,device="cuda")
    m=HeadKVManager(pool)
    m.create_request("a",context_id="ctx")
    k,v=torch.randn(35,32,device="cuda",dtype=torch.bfloat16),torch.randn(35,32,device="cuda",dtype=torch.bfloat16)
    m.update("a",writes=[SegmentWrite((0,0,"D"),k,v,tuple(range(35)),"origin")])
    m.fork("a","b")
    pk,pv=torch.zeros(2,32,device="cuda",dtype=k.dtype),torch.ones(2,32,device="cuda",dtype=k.dtype)
    m.update("b",patches=[SegmentPatch((0,0,"D"),(1,19),pk,pv,"repair")])
    assert_data(m,"a",(0,0,"D"),k,v)
    expected=k.clone();expected[[1,19]]=0
    with m.bind("b") as lease:
        actual=lease.gather((0,0,"D"))[0]
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    reader=m.bind("a")
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if hasattr(torch.cuda,"_sleep"):
            torch.cuda._sleep(50_000_000)
        snapshot=reader.gather((0,0,"D"))[0]
        event=torch.cuda.Event();event.record()
    reader.complete(event)
    m.release_request("a");m.release_request("b")
    if not event.query():
        assert pool.stats()["retiring_pages"]>0
    event.synchronize()
    torch.testing.assert_close(snapshot,k,atol=0,rtol=0)
    assert pool.stats()["free_pages"]==12


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_pool_events_and_descriptors_keep_resolved_device_after_current_device_switch():
    with torch.cuda.device(0):
        pool = HeadPagePool(8, 4, 8, device="cuda")
        manager = HeadKVManager(pool)
        manager.create_request("r", context_id="ctx")
    assert pool.device == torch.device("cuda:0")
    with torch.cuda.device(1):
        # H2D copies and event recording occur while a different device is current.
        k, v = torch.randn(7, 8), torch.randn(7, 8)
        tx = manager.begin_update("r", writes=[SegmentWrite((0, 0, "D"), k, v,
                                                            tuple(range(7)), "prov")])
        assert tx.event.device == pool.device
        tx.commit(wait=True)
        with manager.bind("r") as lease:
            descriptor = lease.descriptor(0, 1)
            assert all(t.device == pool.device for t in descriptor.values())
            actual = lease.gather((0, 0, "D"))
            torch.testing.assert_close(actual[0].cpu(), k)
            torch.testing.assert_close(actual[1].cpu(), v)
        manager.release_request("r")
        torch.cuda.synchronize(pool.device)
        assert pool.stats()["free_pages"] == 8


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_batch_patch_mixed_heads_partial_tails_and_full_pages(device, dtype, monkeypatch):
    pool = HeadPagePool(24, 4, 8, device=device, dtype=dtype)
    manager = HeadKVManager(pool)
    manager.create_request("source", context_id="ctx")
    data = [(torch.randn(n, 8, device=device, dtype=dtype), torch.randn(n, 8, device=device, dtype=dtype))
            for n in (11, 9, 8)]
    manager.update("source", writes=[SegmentWrite((0, h, "D"), k, v, tuple(range(len(k))), "origin")
                                     for h, (k, v) in enumerate(data)])
    manager.fork("source", "child")
    indices = ((0, 1, 2, 3, 5, 10), (8, 3, 4))  # Intentionally non-sorted second head.
    patches = [SegmentPatch((0, h, "D"), row_ids,
                            torch.randn(len(row_ids), 8, device=device, dtype=dtype),
                            torch.randn(len(row_ids), 8, device=device, dtype=dtype), "repair")
               for h, row_ids in enumerate(indices)]
    expected = [(k.clone(), v.clone()) for k, v in data]
    for h, patch in enumerate(patches):
        expected[h][0][list(patch.indices)] = patch.k
        expected[h][1][list(patch.indices)] = patch.v
    before = manager.stats()
    calls, original = [], pool.clone_patch_batch

    def batch(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pool, "clone_patch_batch", batch)
    monkeypatch.setattr(pool, "clone_patch", lambda *args: pytest.fail("multi-page repair used scalar path"))
    manager.update("child", patches=patches)
    assert calls == [6]
    assert manager.stats()["copied_bytes"] - before["copied_bytes"] == 2 * 11 * 8 * pool.k.element_size()
    assert manager.stats()["written_bytes"] - before["written_bytes"] == 2 * 9 * 8 * pool.k.element_size()
    assert manager.version("source").segments[(0, 2, "D")].pages == manager.version("child").segments[(0, 2, "D")].pages
    for h in range(3):
        assert_data(manager, "source", (0, h, "D"), *data[h])
        assert_data(manager, "child", (0, h, "D"), *expected[h])
    manager.release_request("source")
    manager.release_request("child")
    if pool.device.type == "cuda":
        torch.cuda.synchronize(pool.device)
    assert pool.stats()["free_pages"] == 24


def test_batch_full_rewrite_never_gathers_old_rows(monkeypatch):
    manager, _ = setup(heads=1, length=5)
    manager.fork("source", "child")
    indices = (4, 2, 0, 3, 1)
    k, v = torch.randn(5, 8), torch.randn(5, 8)
    before = manager.stats()["copied_bytes"]
    monkeypatch.setattr(torch.Tensor, "index_select", lambda *args: pytest.fail("full rewrite read old rows"))
    manager.update("child", patches=[SegmentPatch((0, 0, "D"), indices, k, v, "rewrite")])
    assert manager.stats()["copied_bytes"] == before
    expected_k, expected_v = torch.empty_like(k), torch.empty_like(v)
    expected_k[list(indices)], expected_v[list(indices)] = k, v
    assert_data(manager, "child", (0, 0, "D"), expected_k, expected_v)


def test_batch_plan_validation_happens_before_any_destination_write():
    manager, _ = setup(heads=1, length=8)
    source = manager.version("source").segments[(0, 0, "D")].pages
    reserved = manager.pool.reserve(2)
    before = manager.pool.written_bytes
    with manager.bind("source"):
        # Two destinations try consuming one repair row, which would be ambiguous.
        plans = [(source[0], reserved[0], 4, (0,), 0, (0,)),
                 (source[1], reserved[1], 4, (0,), 0, (0,))]
        with pytest.raises(ValueError, match="used twice"):
            manager.pool.clone_patch_batch(plans, [(torch.ones(1, 8), torch.zeros(1, 8))])
    assert manager.pool.written_bytes == before
    assert all(manager.pool._get(ref).extent == 0 for ref in reserved)
    manager.pool.release(reserved)


def test_batch_staging_interrupt_retires_all_reserved_pages(monkeypatch):
    manager, _ = setup(heads=2)
    before = manager.stats()
    old = manager.version("source")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("batch interrupted")

    monkeypatch.setattr(manager.pool, "clone_patch_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        manager.begin_update("source", patches=[SegmentPatch((0, h, "D"), (0, 5),
                                                              torch.ones(2, 8), torch.zeros(2, 8), "repair")
                                                for h in range(2)])
    assert manager.version("source") is old
    assert manager.stats()["free_pages"] == before["free_pages"]
    assert manager.stats()["pinned_pages"] == 0
    assert manager.stats()["pending_transactions"] == 0


def test_collect_queries_shared_event_once_for_pages_and_descriptors_per_pass():
    import weakref

    class CountedEvent:
        # Event identity, rather than hashing/equality, determines sharing.
        __hash__ = None
        ready, calls = False, 0
        def query(self):
            self.calls += 1
            return self.ready

    pool = HeadPagePool(4, 4, 8)
    refs = pool.reserve(4)
    event = CountedEvent()
    descriptor_refs = []
    for _ in range(2):
        descriptor = torch.ones(1)
        descriptor_refs.append(weakref.ref(descriptor))
        pool.defer_buffers([descriptor], event)
    del descriptor
    pool.release(refs, event)  # release calls collect once.
    assert event.calls == 1
    assert len(pool._pages) == 4 and len(pool._free) == 0
    assert all(ref() is not None for ref in descriptor_refs)
    assert pool.collect() == 0
    assert event.calls == 2  # A new pass must not reuse cached False.
    event.ready = True
    assert pool.collect() == 4
    assert event.calls == 3
    assert all(ref() is None for ref in descriptor_refs)
    assert not pool._retained_buffers
    assert pool.collect() == 0
    assert event.calls == 3  # Nothing retains the completed event now.


def test_collect_pending_result_is_conservative_if_event_completes_mid_pass():
    class CompletingEvent:
        calls = 0
        def query(self):
            self.calls += 1
            return self.calls > 1

    pool = HeadPagePool(3, 4, 8)
    refs = pool.reserve(3)
    event = CompletingEvent()
    pool.release(refs, event)
    assert event.calls == 1
    assert len(pool._pages) == 3  # One cached False must protect every page.
    assert pool.collect() == 3
    assert event.calls == 2


def test_collect_event_query_failure_quarantines_without_repeated_queries():
    class BrokenEvent:
        calls = 0
        def query(self):
            self.calls += 1
            raise RuntimeError("injected CUDA event failure")

    pool = HeadPagePool(3, 4, 8)
    refs = pool.reserve(3)
    event = BrokenEvent()
    pool.defer_buffers([torch.ones(1)], event)
    with pytest.raises(RuntimeError, match="event failure"):
        pool.release(refs, event)
    assert pool.poisoned and event.calls == 1
    assert len(pool._pages) == 3 and len(pool._free) == 0
    assert len(pool._retained_buffers) == 1
    assert pool.collect() == 0
    assert event.calls == 1
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.reserve(1)
