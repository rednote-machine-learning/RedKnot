"""Main-backend shared KV integration, without loading optional server packages."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python/sglang/srt/mem_cache"))
from head_kv import CapacityError, HeadKVManager, HeadPagePool, SegmentPatch
from head_kv.manager import ReadLease
import head_kv.manager
import head_kv.segpaged

_name = "_redknot_shared_kv_adapter_test"
_spec = importlib.util.spec_from_file_location(
    _name, ROOT / "python/sglang/srt/layers/attention/redknot/shared_kv.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _module
with patch.dict(sys.modules, {
    "sglang.srt.mem_cache.head_kv.manager": head_kv.manager,
    "sglang.srt.mem_cache.head_kv.segpaged": head_kv.segpaged,
}):
    _spec.loader.exec_module(_module)
SharedKVBackend = _module.SharedKVBackend
SharedKVRequestHandle = _module.SharedKVRequestHandle
LIVE_SEGMENT = _module.LIVE_SEGMENT


def make_backend(capacity=64, device="cpu"):
    manager = HeadKVManager(HeadPagePool(capacity, 4, 8, device=device))
    backend = SharedKVBackend(manager)
    handle = backend.create_request("parent", context_id="prompt-v1", contract="weights-v1:mha", namespace="test")
    return backend, manager, handle


def layer(**kwargs):
    fields = dict(tp_q_head_num=4, tp_k_head_num=2, tp_v_head_num=2,
                  qk_head_dim=8, v_head_dim=8, layer_id=0, scaling=0.31,
                  sliding_window_size=-1, is_cross_attention=False,
                  logit_cap=0, pos_encoding_mode="NONE", attn_type="decoder")
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def batch(handles, lengths, prefixes, device="cpu", decode=False):
    positions = [p for length, prefix in zip(lengths, prefixes) for p in range(prefix, prefix + length)]
    return SimpleNamespace(
        redknot_shared_kv_handles=list(handles), batch_size=len(handles),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
        seq_lens=torch.tensor([a+b for a, b in zip(lengths, prefixes)], dtype=torch.int32, device=device),
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        extend_prefix_lens=torch.tensor(prefixes, dtype=torch.int32, device=device),
        forward_mode=SimpleNamespace(name="DECODE" if decode else "EXTEND"),
    )


def tensors(count, device="cpu"):
    return (torch.randn(count, 4, 8, device=device),
            torch.randn(count, 2, 8, device=device),
            torch.randn(count, 2, 8, device=device))


def reference(q, k, v, prefix=0, scale=0.31, windows=(0, 0), sinks=(0, 0)):
    result = torch.empty_like(q)
    keys = torch.arange(len(k), device=q.device)
    queries = torch.arange(prefix, prefix + len(q), device=q.device)
    for head in range(q.shape[1]):
        kh = head // 2
        mask = keys[None] <= queries[:, None]
        if windows[kh]:
            mask &= ((keys[None] >= queries[:, None] - windows[kh] + 1)
                     | (keys[None] < sinks[kh]))
        score = q[:, head].float() @ k[:, kh].float().T * scale
        result[:, head] = score.masked_fill(~mask, -torch.inf).softmax(-1) @ v[:, kh].float()
    return result.reshape(len(q), -1)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_chunk_prefill_fork_batched_decode_no_dense_gather(device, monkeypatch):
    torch.manual_seed(812)
    backend, manager, parent = make_backend(device=device)
    q, k, v = tensors(5, device)
    monkeypatch.setattr(ReadLease, "gather", lambda *args: pytest.fail("forward gathered historical KV"))
    output = backend.forward(q[:3].reshape(3, -1), k[:3], v[:3].reshape(3, -1), layer(),
                             batch([parent], [3], [0], device), decode=False, save_kv_cache=True)
    torch.testing.assert_close(output, reference(q[:3], k[:3], v[:3]), rtol=2e-5, atol=2e-6)
    output = backend.forward(q[3:], k[3:], v[3:], layer(), batch([parent], [2], [3], device),
                             decode=False, save_kv_cache=True)
    torch.testing.assert_close(output, reference(q[3:], k, v, 3), rtol=2e-5, atol=2e-6)
    source = manager.version(parent.request_id)
    child = backend.fork_request(parent, "child")
    sibling = backend.fork_request(parent, "sibling")
    assert manager.stats()["live_pages"] == 4
    qd, kd, vd = tensors(2, device)
    output = backend.forward(qd, kd, vd, layer(), batch([child, sibling], [1, 1], [5, 5], device, True),
                             decode=True, save_kv_cache=True)
    for index, handle in enumerate((child, sibling)):
        expected = reference(qd[index:index+1], torch.cat((k, kd[index:index+1])),
                             torch.cat((v, vd[index:index+1])), 5)
        torch.testing.assert_close(output[index:index+1], expected, rtol=2e-5, atol=2e-6)
        version = manager.version(handle.request_id)
        for head in range(2):
            key = (0, head, LIVE_SEGMENT)
            assert version.segments[key].pages[0] == source.segments[key].pages[0]
            assert version.segments[key].pages[1] != source.segments[key].pages[1]
    assert manager.version(parent.request_id) is source
    for head in range(2):
        segment = source.segments[(0, head, LIVE_SEGMENT)]
        got_k = torch.cat([manager.pool.read_page(page, 4 if i == 0 else 1)[0]
                           for i, page in enumerate(segment.pages)])
        torch.testing.assert_close(got_k, k[:, head])
    for handle in (parent, child, sibling):
        backend.release_request(handle)
    if device == "cuda":
        torch.cuda.synchronize()
    manager.pool.collect()
    assert manager.stats()["free_pages"] == 64
    assert manager.stats()["pending_transactions"] == 0
    assert not backend._views


def test_explicit_request_generation_fences_slot_reuse_and_release():
    backend, manager, stale = make_backend()
    backend.release_request(stale)
    current = backend.create_request("parent", context_id="new", contract="weights-v2", namespace="test")
    assert stale.generation != current.generation
    for operation in (lambda: backend.release_request(stale),
                      lambda: backend.fork_request(stale, "bad"),
                      lambda: backend.repair_request(stale, []),
                      lambda: backend.forward(*tensors(1), layer(), batch([stale], [1], [0]),
                                              decode=False, save_kv_cache=True)):
        with pytest.raises(RuntimeError, match="stale"):
            operation()
    assert manager.version("parent").generation == current.generation
    assert manager.stats()["live_pages"] == 0


def test_batch_preflight_rejects_late_invalid_prefix_without_advancing_first_request():
    backend, manager, first = make_backend()
    second = backend.create_request("second", context_id="other", contract="weights-v1:mha", namespace="test")
    before = manager.version("parent")
    with pytest.raises(ValueError, match="prefix"):
        backend.forward(*tensors(2), layer(), batch([first, second], [1, 1], [0, 2]),
                        decode=False, save_kv_cache=True)
    assert manager.version("parent") is before
    assert manager.stats()["live_pages"] == 0


def test_oom_does_not_publish_partial_head_append_and_release_reclaims_shared_pages():
    backend, manager, parent = make_backend(capacity=3)
    backend.forward(*tensors(3), layer(), batch([parent], [3], [0]), decode=False, save_kv_cache=True)
    child = backend.fork_request(parent, "child")
    before = manager.version("child")
    with pytest.raises(CapacityError):
        backend.forward(*tensors(1), layer(), batch([child], [1], [3], decode=True),
                        decode=True, save_kv_cache=True)
    assert manager.version("child") is before
    assert manager.stats()["live_pages"] == 2
    assert manager.stats()["reserved_pages"] == 0
    backend.release_request(parent)
    assert manager.stats()["live_pages"] == 2
    backend.release_request(child)
    assert manager.stats()["free_pages"] == 3


def test_nonprefix_repair_cows_only_changed_head_page():
    backend, manager, parent = make_backend()
    q, k, v = tensors(7)
    backend.forward(q, k, v, layer(), batch([parent], [7], [0]), decode=False, save_kv_cache=True)
    child = backend.fork_request(parent, "child")
    original = manager.version("parent")
    key = (0, 1, LIVE_SEGMENT)
    backend.repair_request(child, [SegmentPatch(key, (5,), torch.ones(1, 8), torch.zeros(1, 8), "repair-proof")])
    modified = manager.version("child")
    assert modified.segments[key].pages[0] == original.segments[key].pages[0]
    assert modified.segments[key].pages[1] != original.segments[key].pages[1]
    assert modified.segments[(0, 0, LIVE_SEGMENT)].pages == original.segments[(0, 0, LIVE_SEGMENT)].pages
    with manager.bind("parent") as lease:
        torch.testing.assert_close(lease.gather(key)[0], k[:, 1])
    qd, kd, vd = tensors(1)
    output = backend.forward(qd, kd, vd, layer(), batch([child], [1], [7], decode=True),
                             decode=True, save_kv_cache=True)
    repaired_k, repaired_v = k.clone(), v.clone()
    repaired_k[5, 1], repaired_v[5, 1] = 1, 0
    expected = reference(qd, torch.cat((repaired_k, kd)), torch.cat((repaired_v, vd)), 7)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)


def test_layers_keep_separate_payloads_and_fork_shares_both():
    backend, manager, parent = make_backend()
    for layer_id in (0, 2):
        q, k, v = tensors(3)
        output = backend.forward(q, k, v, layer(layer_id=layer_id), batch([parent], [3], [0]),
                                 decode=False, save_kv_cache=True)
        torch.testing.assert_close(output, reference(q, k, v), rtol=2e-5, atol=2e-6)
    before = manager.version("parent")
    child = backend.fork_request(parent, "child")
    assert manager.stats()["live_pages"] == 4
    assert manager.version("child").segments == before.segments
    q, k, v = tensors(1)
    backend.forward(q, k, v, layer(layer_id=2), batch([child], [1], [3], decode=True),
                    decode=True, save_kv_cache=True)
    after = manager.version("child")
    for head in range(2):
        assert before.segments[(0, head, LIVE_SEGMENT)] == after.segments[(0, head, LIVE_SEGMENT)]
        assert before.segments[(2, head, LIVE_SEGMENT)].pages != after.segments[(2, head, LIVE_SEGMENT)].pages


def test_duplicate_handles_fail_before_any_append():
    backend, manager, parent = make_backend()
    with pytest.raises(ValueError, match="duplicate"):
        backend.forward(*tensors(2), layer(), batch([parent, parent], [1, 1], [0, 0]),
                        decode=False, save_kv_cache=True)
    assert not manager.version("parent").segments


def test_head_windows_and_model_window_preserve_attention_semantics():
    backend, manager, parent = make_backend()
    q, k, v = tensors(7)
    output = backend.forward(q, k, v, layer(sliding_window_size=2), batch([parent], [7], [0]),
                             decode=False, save_kv_cache=True, windows=[0, 2], sinks=[0, 0])
    torch.testing.assert_close(output, reference(q, k, v, windows=(3, 2)), rtol=2e-5, atol=2e-6)
    second = backend.create_request("sink", context_id="prompt-v1", contract="weights-v1:mha", namespace="test")
    output = backend.forward(q, k, v, layer(), batch([second], [7], [0]),
                             decode=False, save_kv_cache=True, windows=[3, 0], sinks=[1, 0])
    torch.testing.assert_close(output, reference(q, k, v, windows=(3, 0), sinks=(1, 0)), rtol=2e-5, atol=2e-6)
    third = backend.create_request("bad-sink", context_id="prompt-v1", contract="weights-v1:mha", namespace="test")
    with pytest.raises(ValueError, match="sink"):
        backend.forward(q, k, v, layer(sliding_window_size=2), batch([third], [7], [0]),
                        decode=False, save_kv_cache=True, windows=[3, 0], sinks=[1, 0])
    assert not manager.version("bad-sink").segments


@pytest.mark.parametrize("changed_layer,changed_batch,kwargs", [
    ({"is_cross_attention": True}, {}, {}),
    ({"attn_type": "decoder_bidirectional"}, {}, {}),
    ({"v_head_dim": 4}, {}, {}),
    ({"logit_cap": 2.0}, {}, {}),
    ({"k_scale": torch.tensor(1.0)}, {}, {}),
    ({}, {"is_cuda_graph_capture": True}, {}),
    ({}, {"spec_info": object()}, {}),
    ({}, {"encoder_lens": torch.tensor([1])}, {}),
    ({}, {"positions": torch.tensor([1])}, {}),
    ({}, {}, {"save_kv_cache": False}),
    ({}, {}, {"windows": [0]}),
])
def test_unsupported_semantics_fail_before_cache_mutation(changed_layer, changed_batch, kwargs):
    backend, manager, handle = make_backend()
    fb = batch([handle], [1], [0])
    for name, value in changed_batch.items():
        setattr(fb, name, value)
    options = dict(decode=False, save_kv_cache=True)
    options.update(kwargs)
    with pytest.raises(ValueError):
        backend.forward(*tensors(1), layer(**changed_layer), fb, **options)
    assert not manager.version(handle.request_id).segments
    assert manager.stats()["free_pages"] == 64
