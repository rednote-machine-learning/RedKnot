"""Exercise the real RedKnot/SegPaged backend and registry entry points.

Only heavyweight SGLang import dependencies and legacy kernel availability are
stubbed. The backend methods, shared-request adapter, manager, page slabs and
attention implementation are production code. CUDA cases use that same route.
The independent dense oracle is confined to this test file.
"""

from contextlib import contextmanager
from enum import Enum
import importlib
from pathlib import Path
import sys
import types
from unittest.mock import Mock, patch

import pytest
import torch


_SOURCE = Path(__file__).resolve().parents[3] / "python"


@contextmanager
def _isolated_backend_imports():
    """Load actual backend files without importing the SGLang server runtime."""
    saved = {name: module for name, module in sys.modules.items()
             if name == "sglang" or name.startswith("sglang.")}
    for name in saved:
        del sys.modules[name]

    def module(name, **attributes):
        result = types.ModuleType(name)
        result.__dict__.update(attributes)
        sys.modules[name] = result
        return result

    try:
        for name in (
            "sglang", "sglang.srt", "sglang.srt.layers",
            "sglang.srt.layers.attention", "sglang.srt.layers.attention.redknot",
            "sglang.srt.mem_cache", "sglang.srt.configs",
            "sglang.srt.model_executor", "sglang.srt.utils",
        ):
            module(name, __path__=[str(_SOURCE.joinpath(*name.split(".")))])
        module("sglang.kernel_api_logging", debug_kernel_api=lambda fn: fn)
        module("sglang.srt.utils.common", is_npu=lambda: False)
        sys.modules["sglang.srt.utils"].is_musa = lambda: False
        sys.modules["sglang.srt.utils"].get_device_capability = lambda: (0, 0)
        module("sglang.srt.configs.linear_attn_model_registry",
               get_linear_attn_config=lambda *a, **kw: None,
               import_backend_class=lambda *a, **kw: None)
        module("sglang.srt.layers.dp_attention", get_attention_tp_rank=lambda: 0)

        class AttentionType(Enum):
            DECODER = "decoder"
            ENCODER = "encoder"

        module("sglang.srt.layers.radix_attention", AttentionType=AttentionType)
        module("sglang.srt.model_executor.forward_batch_info", ForwardBatch=object)
        backend = importlib.import_module("sglang.srt.layers.attention.redknot_backend")
        segpaged = importlib.import_module("sglang.srt.layers.attention.segpaged_backend")
        registry = importlib.import_module("sglang.srt.layers.attention.attention_registry")
        head_kv = importlib.import_module("sglang.srt.mem_cache.head_kv")
        config = importlib.import_module("sglang.srt.layers.attention.redknot.head_config")
        yield types.SimpleNamespace(
            backend=backend, registry=registry, config=config,
            head_kv=head_kv,
            classes={"redknot": backend.RedKnotAttnBackend,
                     "segpaged": segpaged.SegPagedAttnBackend},
        )
    finally:
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang."):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(scope="module")
def runtime():
    with _isolated_backend_imports() as modules:
        yield modules


class _ForbiddenDensePool:
    def __getattr__(self, name):
        raise AssertionError(f"managed backend accessed dense KV pool: {name}")


class _ForbiddenRequestPool:
    @property
    def req_to_token(self):
        raise AssertionError("managed backend accessed dense request-token mapping")


class _DensePool:
    def __init__(self, heads=2, dim=16):
        self.k = torch.zeros(32, heads, dim)
        self.v = torch.zeros_like(self.k)
        self.writes = []
        self.reads = []

    def set_kv_buffer(self, layer, locations, k, v):
        self.writes.append((layer, locations.clone(), k.clone(), v.clone()))
        self.k[locations] = k.reshape(-1, 2, 16)
        self.v[locations] = v.reshape(-1, 2, 16)

    def get_key_buffer(self, layer_id):
        self.reads.append(("k", layer_id))
        return self.k

    def get_value_buffer(self, layer_id):
        self.reads.append(("v", layer_id))
        return self.v


def _runner(device="cpu", manager=None, dense=None):
    result = types.SimpleNamespace(
        device=torch.device(device),
        token_to_kv_pool=dense if dense is not None else _ForbiddenDensePool(),
        req_to_token_pool=_ForbiddenRequestPool(),
        server_args=types.SimpleNamespace(),
    )
    if manager is not None:
        result.redknot_shared_kv_manager = manager
    return result


def _manager(runtime, device="cpu"):
    return runtime.head_kv.HeadKVManager(runtime.head_kv.HeadPagePool(
        capacity_pages=128, page_size=4, head_dim=16, device=device,
        dtype=torch.float32,
    ))


def _layer():
    return types.SimpleNamespace(
        layer_id=0, tp_q_head_num=4, tp_k_head_num=2, tp_v_head_num=2,
        qk_head_dim=16, v_head_dim=16, scaling=16 ** -0.5,
        is_cross_attention=False, attn_type="decoder", sliding_window_size=-1,
    )


def _batch(handles, lengths, prefixes, *, device="cpu", decode=False):
    positions = [position for length, prefix in zip(lengths, prefixes)
                 for position in range(prefix, prefix + length)]
    # Deliberately unrelated slot IDs: stable handles must be the only identity.
    return types.SimpleNamespace(
        redknot_shared_kv_handles=handles,
        batch_size=len(handles),
        req_pool_indices=torch.tensor([90 - 23 * i for i in range(len(handles))]),
        seq_lens=torch.tensor([a + b for a, b in zip(lengths, prefixes)]),
        extend_seq_lens=torch.tensor(lengths),
        extend_prefix_lens=torch.tensor(prefixes),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
        forward_mode=types.SimpleNamespace(name="DECODE" if decode else "EXTEND"),
    )


def _qkv(length, *, device="cpu", seed=11):
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(length, heads, 16, generator=generator).to(device)
                 for heads in (4, 2, 2))


def _dense_attention(q, k, v, query_positions, *, windows=(0, 0), sinks=(0, 0)):
    """Independent float64 masked attention over test-only concatenated KV."""
    output = torch.empty_like(q)
    positions = torch.arange(len(k), device=q.device)
    targets = torch.as_tensor(query_positions, device=q.device)
    for head in range(4):
        kv_head = head // 2
        visible = positions[None, :] <= targets[:, None]
        if windows[kv_head]:
            visible &= ((positions[None, :] >= targets[:, None] - windows[kv_head] + 1)
                        | (positions[None, :] < sinks[kv_head]))
        scores = q[:, head].double() @ k[:, kv_head].double().T * (16 ** -0.5)
        scores.masked_fill_(~visible, -torch.inf)
        output[:, head] = (scores.softmax(-1) @ v[:, kv_head].double()).to(q.dtype)
    return output.reshape(len(q), -1)


def _new_request(backend, name):
    return backend.shared_kv.create_request(
        name, context_id=f"context:{name}", contract="test-model:gqa:rope-v1",
        namespace="test-tenant",
    )


def _run_batch(backend, qkv, handles, lengths, prefixes, *, decode=False):
    batch = _batch(handles, lengths, prefixes, device=qkv[0].device, decode=decode)
    backend.init_forward_metadata(batch)
    method = backend.forward_decode if decode else backend.forward_extend
    # Exercise the flattened query convention used by model projections.
    q, k, v = qkv
    return method(q.reshape(len(q), -1), k, v, _layer(), batch)


def _exercise_prefill_fork_decode(runtime, backend_name, device):
    manager = _manager(runtime, device)
    backend = runtime.classes[backend_name](_runner(device), shared_kv_manager=manager)
    first, second = (_new_request(backend, name) for name in ("first", "second"))
    histories = {first: [], second: []}
    for step, (lengths, prefixes) in enumerate((((3, 2), (0, 0)), ((2, 3), (3, 2)))):
        qkv = _qkv(sum(lengths), device=device, seed=17 + step)
        actual = _run_batch(backend, qkv, [first, second], lengths, prefixes)
        expected, offset = [], 0
        for handle, length, prefix in zip((first, second), lengths, prefixes):
            chunk = tuple(t[offset:offset + length] for t in qkv)
            histories[handle].append(chunk)
            k = torch.cat([part[1] for part in histories[handle]])
            v = torch.cat([part[2] for part in histories[handle]])
            expected.append(_dense_attention(chunk[0], k, v, range(prefix, prefix + length)))
            offset += length
        torch.testing.assert_close(actual, torch.cat(expected), atol=2e-4, rtol=2e-4)
    parent = manager.version(first.request_id)
    parent_payload = {key: tuple(value.clone() for value in manager.pool.read_page(ref, 1))
                      for key, segment in parent.segments.items()
                      for ref in segment.pages[-1:]}
    child = backend.shared_kv.fork_request(first, "child")
    fork_version = manager.version(child.request_id)
    assert fork_version.segments == parent.segments
    assert child.generation != first.generation

    # Reorder the requests and reuse arbitrary slot numbers; identities persist.
    qkv = _qkv(2, device=device, seed=33)
    actual = _run_batch(backend, qkv, [second, child], (1, 1), (5, 5), decode=True)
    expected = []
    for index, source in enumerate((second, first)):
        k = torch.cat([part[1] for part in histories[source]] + [qkv[1][index:index + 1]])
        v = torch.cat([part[2] for part in histories[source]] + [qkv[2][index:index + 1]])
        expected.append(_dense_attention(qkv[0][index:index + 1], k, v, [5]))
    torch.testing.assert_close(actual, torch.cat(expected), atol=2e-4, rtol=2e-4)
    assert manager.version(first.request_id) is parent
    for key, original in parent.segments.items():
        changed = manager.version(child.request_id).segments[key]
        assert changed.pages[0] == original.pages[0]
        assert changed.pages[-1] != original.pages[-1]
        for before, after in zip(parent_payload[key], manager.pool.read_page(original.pages[-1], 1)):
            torch.testing.assert_close(after, before, atol=0, rtol=0)
    for handle in (child, second, first):
        backend.shared_kv.release_request(handle)
    if device != "cpu":
        torch.cuda.synchronize()
    manager.pool.collect()
    assert manager.pool.stats()["live_pages"] == 0
    assert manager.pool.stats()["pinned_pages"] == 0


@pytest.mark.parametrize("backend_name", ["redknot", "segpaged"])
def test_chunked_two_request_prefill_and_fork_decode(runtime, backend_name):
    _exercise_prefill_fork_decode(runtime, backend_name, "cpu")


@pytest.mark.parametrize("backend_name", ["redknot", "segpaged"])
def test_registry_forwards_shared_manager_into_real_backend(runtime, backend_name):
    manager = _manager(runtime)
    backend = runtime.registry.ATTENTION_BACKENDS[backend_name](_runner(manager=manager))
    assert isinstance(backend, runtime.classes[backend_name])
    assert backend.shared_kv.manager is manager
    handle = _new_request(backend, "registry")
    qkv = _qkv(3)
    actual = _run_batch(backend, qkv, [handle], (3,), (0,))
    torch.testing.assert_close(actual, _dense_attention(*qkv, range(3)))
    backend.shared_kv.release_request(handle)


@pytest.mark.parametrize("backend_name", ["redknot", "segpaged"])
def test_head_config_window_and_sink_reach_shared_attention(runtime, backend_name):
    manager = _manager(runtime)
    config = runtime.config.HeadClassConfig(
        head_class=[["local", "global"]], head_max_distance=[[2, -1]],
        head_sink_size=[[1, 0]], num_layers=1, num_kv_heads=2,
    )
    backend = runtime.classes[backend_name](_runner(), shared_kv_manager=manager)
    handle = _new_request(backend, "head-policy")
    qkv = _qkv(7, seed=54)
    batch = _batch([handle], (7,), (0,))
    batch.redknot_head_config = config
    backend.init_forward_metadata(batch)
    actual = backend.forward_extend(*qkv, _layer(), batch)
    expected = _dense_attention(*qkv, range(7), windows=(2, 0), sinks=(1, 0))
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, _dense_attention(*qkv, range(7)))
    backend.shared_kv.release_request(handle)


@pytest.mark.parametrize("backend_name", ["redknot", "segpaged"])
@pytest.mark.parametrize("decode", [False, True])
def test_retrieval_policy_rejected_before_managed_write(runtime, backend_name, decode):
    manager = _manager(runtime)
    backend = runtime.classes[backend_name](_runner(), shared_kv_manager=manager)
    handle = _new_request(backend, "retrieval-policy")
    prefix = 0
    if decode:
        _run_batch(backend, _qkv(3), [handle], (3,), (0,))
        prefix = 3
    original = manager.version(handle.request_id)
    stats = manager.pool.stats()
    config = runtime.config.HeadClassConfig(
        head_class=[["retrieval", "global"]], head_max_distance=[[-1, -1]],
        head_sink_size=[[0, 0]], num_layers=1, num_kv_heads=2,
    )
    batch = _batch([handle], (1,), (prefix,), decode=decode)
    batch.redknot_head_config = config
    backend.init_forward_metadata(batch)
    method = backend.forward_decode if decode else backend.forward_extend
    with pytest.raises(ValueError, match="retrieval"):
        method(*_qkv(1), _layer(), batch)
    assert manager.version(handle.request_id) is original
    assert manager.pool.stats() == stats
    backend.shared_kv.release_request(handle)


@pytest.mark.parametrize("argument", ["custom_mask", "k_rope", "attn_sink"])
@pytest.mark.parametrize("decode", [False, True])
def test_model_specific_attention_arguments_rejected_before_write(runtime, argument, decode):
    manager = _manager(runtime)
    backend = runtime.classes["redknot"](_runner(), shared_kv_manager=manager)
    handle = _new_request(backend, "extra-arguments")
    original = manager.version(handle.request_id)
    batch = _batch([handle], (1,), (0,), decode=decode)
    method = backend.forward_decode if decode else backend.forward_extend
    with pytest.raises(ValueError, match="unsupported shared KV attention arguments"):
        method(*_qkv(1), _layer(), batch, **{argument: torch.ones(1)})
    assert manager.version(handle.request_id) is original
    assert manager.pool.stats()["live_pages"] == 0
    backend.shared_kv.release_request(handle)


@pytest.mark.parametrize("decode", [False, True])
def test_none_attention_arguments_preserve_managed_dispatch(runtime, decode):
    manager = _manager(runtime)
    backend = runtime.classes["redknot"](_runner(), shared_kv_manager=manager)
    handle = _new_request(backend, "none-arguments")
    batch = _batch([handle], (1,), (0,), decode=decode)
    method = backend.forward_decode if decode else backend.forward_extend
    qkv = _qkv(1)
    actual = method(*qkv, _layer(), batch, custom_mask=None, k_rope=None, attn_sink=None)
    torch.testing.assert_close(actual, _dense_attention(*qkv, [0]))
    backend.shared_kv.release_request(handle)


@pytest.mark.parametrize("decode", [False, True])
def test_handles_without_configured_manager_fail_before_dense_access(runtime, decode):
    with patch.object(runtime.backend, "is_flash_attn_available", return_value=True):
        backend = runtime.classes["redknot"](_runner())
    batch = _batch([object()], (1,), (0,), decode=decode)
    method = backend.forward_decode if decode else backend.forward_extend
    with pytest.raises(ValueError, match="shared_kv_manager"):
        method(*_qkv(1), _layer(), batch)


@pytest.mark.parametrize("backend_name", ["redknot", "segpaged"])
def test_legacy_offline_splice_rejected_before_managed_write(runtime, backend_name):
    manager = _manager(runtime)
    backend = runtime.classes[backend_name](_runner(), shared_kv_manager=manager)
    handle = _new_request(backend, "offline")
    original = manager.version(handle.request_id)
    batch = _batch([handle], (2,), (0,))
    batch.redknot_offline_segments = [["precomputed-document"]]
    backend.init_forward_metadata(batch)
    with pytest.raises(ValueError, match="offline splice"):
        backend.forward_extend(*_qkv(2), _layer(), batch)
    assert manager.version(handle.request_id) is original
    assert manager.pool.stats()["live_pages"] == 0
    backend.shared_kv.release_request(handle)


def test_default_legacy_extend_still_writes_dense_and_calls_fallback(runtime):
    dense = _DensePool()
    with patch.object(runtime.backend, "is_flash_attn_available", return_value=True):
        backend = runtime.classes["redknot"](_runner(dense=dense))
    batch = _batch([], (2,), (0,))
    del batch.redknot_shared_kv_handles
    batch.out_cache_loc = torch.tensor([3, 4])
    qkv = _qkv(2)
    sentinel = torch.randn(2, 64)
    backend._sdpa_fallback_extend = Mock(return_value=sentinel)
    layer = _layer()
    actual = backend.forward_extend(*qkv, layer, batch)
    assert actual is sentinel
    backend._sdpa_fallback_extend.assert_called_once_with(qkv[0], layer, batch)
    assert len(dense.writes) == 1
    torch.testing.assert_close(dense.k[batch.out_cache_loc], qkv[1])
    torch.testing.assert_close(dense.v[batch.out_cache_loc], qkv[2])


def test_default_legacy_decode_still_reads_dense_pool(runtime):
    dense = _DensePool()
    runner = _runner(dense=dense)
    runner.req_to_token_pool = types.SimpleNamespace(req_to_token=torch.arange(32)[None, :])
    with patch.object(runtime.backend, "is_flash_attn_available", return_value=True):
        backend = runtime.classes["redknot"](runner)
    q, k, v = _qkv(3)
    dense.k[:2], dense.v[:2] = k[:2], v[:2]
    batch = _batch([], (1,), (2,), decode=True)
    del batch.redknot_shared_kv_handles
    batch.req_pool_indices = torch.tensor([0])
    batch.out_cache_loc = torch.tensor([2])
    actual = backend.forward_decode(q[2:], k[2:], v[2:], _layer(), batch)
    torch.testing.assert_close(actual, _dense_attention(q[2:], k, v, [2]))
    assert len(dense.writes) == 1
    assert dense.reads == [("k", 0), ("v", 0)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("backend_name", ["redknot", "segpaged"])
def test_cuda_real_backend_prefill_fork_decode(runtime, backend_name):
    pytest.importorskip("triton")
    _exercise_prefill_fork_decode(runtime, backend_name, "cuda:0")
