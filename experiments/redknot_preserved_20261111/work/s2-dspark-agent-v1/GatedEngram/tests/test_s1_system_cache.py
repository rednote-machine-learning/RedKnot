"""CPU assembly parity, with real Torch coverage required at remote admission.

The NumPy backend is a local tensor-operation oracle, not native CUDA numerical
qualification. Real Torch uses BF16 outputs and FP32 gate/latent and state.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from kv_cache_engram.core import digest
from kv_cache_engram.s1_cache import GLOBAL_SOURCES, NativeCacheOps, S1Cache, fixed_selection
from kv_cache_engram.s1_system_cache import S1SystemCache

try:
    import torch
except ImportError:
    torch = None


class ArrayTensor:
    """Small NumPy adapter solely for exercising actual assembly control flow."""
    def __init__(self, array, dtype="float32"):
        self.array, self.dtype, self.device, self._version = np.asarray(array), dtype, "cpu", 0
    @property
    def shape(self):
        return self.array.shape
    def __getitem__(self, key):
        return ArrayTensor(self.array[key], self.dtype)
    def clone(self):
        return ArrayTensor(self.array.copy(), self.dtype)
    def copy_(self, other):
        self.array[...] = other.array
        self._version += 1
        return self
    def index_select(self, dim, indices):
        return ArrayTensor(np.take(self.array, indices.array, axis=dim), self.dtype)
    def index_copy_(self, dim, indices, other):
        assert dim == 0
        self.array[indices.array] = other.array
        self._version += 1
        return self
    def contiguous(self):
        return ArrayTensor(np.ascontiguousarray(self.array), self.dtype)
    def unsqueeze(self, dim):
        return ArrayTensor(np.expand_dims(self.array, dim), self.dtype)
    def new_zeros(self, shape, dtype=None):
        return ArrayBackend.zeros(shape, dtype=dtype or self.dtype, device=self.device)
    def numel(self):
        return self.array.size
    def element_size(self):
        return 2 if self.dtype == "bfloat16" else 4


class ArrayBackend:
    float32, bfloat16, long, int32 = "float32", "bfloat16", "long", "int32"
    @staticmethod
    def tensor(value, *, dtype, device="cpu"):
        a = np.asarray(value, dtype=np.int64 if dtype in ("long", "int32") else np.float32)
        return ArrayTensor(a.copy(), dtype)
    @staticmethod
    def zeros(shape, *, dtype, device="cpu"):
        return ArrayTensor(np.zeros(shape, dtype=np.float32), dtype)
    @staticmethod
    def empty(shape, *, dtype, device="cpu"):
        return ArrayTensor(np.empty(shape, dtype=np.float32), dtype)
    @staticmethod
    def cat(values, dim=0):
        return ArrayTensor(np.concatenate([v.array for v in values], axis=dim), values[0].dtype)
    @staticmethod
    def equal(a, b):
        return a.dtype == b.dtype and np.array_equal(a.array, b.array)


class CPUOps:
    """Same FP32 per-pair reduction for both implementations; records store ABI."""
    def __init__(self, backend):
        self.t = backend
        self.compressions, self.carry_calls = [], []
        # Exercise the unchanged NativeCacheOps tail slicing/absolute metadata.
        self.native = object.__new__(NativeCacheOps)
        self.native.t, self.native.allow_sync = backend, None
    def compress_canonical(self, mod, raw):
        self.compressions.append(raw.clone())
        if mod.kv_compress_ratio == 1:
            return raw.clone()
        if self.t is torch:
            x = raw.reshape(-1, 2, 1024)
            return (torch.softmax(x[:, :, :512], dim=1)*x[:, :, 512:]).sum(dim=1).bfloat16()
        x = raw.array.reshape(-1, 2, 1024)
        logits = x[:, :, :512]
        weights = np.exp(logits-logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        values = (weights*x[:, :, 512:]).sum(axis=1)
        # NumPy emulates round-to-nearest-even BF16 output for the local oracle.
        bits = values.astype(np.float32).view(np.uint32)
        values = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)
        return ArrayTensor(values, "bfloat16")
    def store_main_and_index(self, mod, canonical, cache, meta):
        cache.main, cache.index = canonical.clone(), canonical.clone()
    def verify_native_store(self, mod, raw, cache, meta):
        cache.verified_raw = raw.clone()
    def update_compressor_carry(self, mod, raw, cache, meta, end):
        if mod.kv_compress_ratio == 1:
            return
        # A CPU stand-in for forward_pre_norm observes the exact raw/meta ABI
        # passed by the real NativeCacheOps method, including ring slot writes.
        def forward(value, state, carry_meta):
            self.carry_calls.append((value.clone(), carry_meta.cu_seqlens_q.clone(),
                                     carry_meta.cu_seqlens_k.clone()))
            begin = end-value.shape[0]
            for row in range(value.shape[0]):
                slot = (begin+row) % state.shape[1]
                state[0, slot, 0].copy_(value[row, :512])
                state[0, slot, 1].copy_(value[row, 512:])
            return None
        mod.PackedWkv = NS(forward_pre_norm=forward)
        self.native.update_compressor_carry(mod, raw, cache, meta, end)


def fixture(backend, *, optimized, docs=None, total=None, enabled=True):
    docs = docs or [dict(start=3, end=303, key="a"*64), dict(start=303, end=606, key="b"*64)]
    total = total if total is not None else 609
    layers = [NS(self_attn=NS(layer_idx=i, kv_compress_ratio=1 if i == 20 else 2,
                             ratio_key="1" if i == 20 else "2")) for i in range(40)]
    ops = CPUOps(backend)
    cls = S1SystemCache if optimized else S1Cache
    extra = dict(enabled=enabled) if optimized else {}
    cache = cls(layers=layers, scope={"fixture": "cpu-only"}, torch_module=backend, ops=ops, **extra)
    cache.docs, cache.total = docs, total
    cache.selected = fixed_selection(docs, total)
    cache.selected_set = set(cache.selected)
    cache.device = torch.device("cpu") if backend is torch else NS(type="cpu")
    for ordinal, doc in enumerate(docs):
        n = doc["end"]-doc["start"]
        for layer in GLOBAL_SOURCES:
            width, dtype = (512, backend.bfloat16) if layer == 20 else (1024, backend.float32)
            p = np.arange(n, dtype=np.float32)[:, None]
            columns = np.arange(width, dtype=np.float32)[None, :]
            values = np.sin((p + 7*ordinal + 3*layer)*0.071 + columns*0.013).astype(np.float32)
            value = backend.tensor(values, dtype=dtype, device="cpu")
            cache.bank[doc["key"], "global_raw", layer] = dict(tensor=value, complete=True,
                version=value._version, scope=digest(cache.scope), length=n, local_start=0)
    if backend is not torch:
        # begin_phase only uses .type; actual array devices remain the same
        # object so production's online-device check is also exercised.
        cache.device = DeviceCPU()
    cache.begin_phase("cpu-parity")
    return cache, layers, ops


class DeviceCPU:
    type = "cpu"
    def __eq__(self, other):
        return other == "cpu" if isinstance(other, str) else isinstance(other, DeviceCPU)


class CacheParityMixin:
    backend = None
    def assertTensorEqual(self, left, right):
        self.assertTrue(self.backend.equal(left, right), "CPU tensor parity failed")
    def setUp(self):
        if torch is not None:
            self.assertFalse(torch.cuda.is_initialized())
    def tearDown(self):
        if torch is not None:
            self.assertFalse(torch.cuda.is_initialized())
    def run_pair(self, *, boundaries, layer=8, docs=None, total=None, verify=False, enabled=True):
        base, layers, baseops = fixture(self.backend, optimized=False, docs=docs, total=total)
        fast, fastlayers, fastops = fixture(self.backend, optimized=True, docs=docs, total=total, enabled=enabled)
        old_bank = {key: entry["tensor"].clone() for key, entry in fast.bank.items()}
        states = [NS(ca_state=self.backend.zeros((1, 2, 2, 512), dtype=self.backend.float32, device="cpu"))
                  for _ in range(2)]
        for start, end in zip(boundaries, boundaries[1:]):
            selected = [p for p in base.selected if start <= p < end]
            width, dtype = (512, self.backend.bfloat16) if layer == 20 else (1024, self.backend.float32)
            values = np.cos(np.asarray(selected, dtype=np.float32)[:, None]*0.053 +
                            np.arange(width, dtype=np.float32)[None, :]*0.019).astype(np.float32)
            raw = self.backend.tensor(values, dtype=dtype, device="cpu")
            ratio = 1 if layer == 20 else 2
            rows = max(1, end//ratio-start//ratio)
            meta = NS(positions_c={str(ratio): self.backend.tensor(list(range(rows)), dtype=self.backend.int32, device="cpu")},
                      state_block_idx=self.backend.tensor([0], dtype=self.backend.int32, device="cpu"))
            ctx = NS(start=start, end=end, positions_cpu=selected, mode="sensitive", verify=verify)
            expected = base.global_update(layers[layer].self_attn, raw, states[0], meta, ctx)
            actual = fast.global_update(fastlayers[layer].self_attn, raw, states[1], meta, ctx)
            self.assertTensorEqual(expected, actual)
            self.assertTensorEqual(states[0].main, states[1].main)
            self.assertTensorEqual(states[0].index, states[1].index)
            self.assertTensorEqual(states[0].ca_state, states[1].ca_state)
            self.assertEqual(base.global_events[-1], fast.global_events[-1])
            self.assertEqual(set(base.global_carry), set(fast.global_carry))
            if layer in base.global_carry:
                self.assertEqual(base.global_carry[layer][0], fast.global_carry[layer][0])
                self.assertTensorEqual(base.global_carry[layer][1], fast.global_carry[layer][1])
            if verify:
                self.assertTensorEqual(states[0].verified_raw, states[1].verified_raw)
        for key, value in old_bank.items():
            self.assertTensorEqual(value, fast.entry(*key))
        self.assertEqual(len(baseops.carry_calls), len(fastops.carry_calls))
        for before, after in zip(baseops.carry_calls, fastops.carry_calls):
            for x, y in zip(before, after):
                self.assertTensorEqual(x, y)
        return base, fast, states, baseops, fastops
    def test_cross_document_pair_and_odd_chunks_exact(self):
        for layer in (8, 14):
            with self.subTest(layer=layer):
                self.run_pair(boundaries=[0, 1, 2, 175, 303, 304, 605, 609], layer=layer)
    def test_full_current_qualification_keeps_original_checks(self):
        _, fast, _, _, _ = self.run_pair(boundaries=[0, 175, 303, 609], verify=True)
        self.assertTrue(all(e["checks"] == 2 for e in fast.global_events))
        self.assertEqual(sum(e["qualification_full_raw_rows"] for e in fast.system_global_update_events), 609)
    def test_empty_selected_chunks_padding_and_decode_carry(self):
        docs = [dict(start=0, end=1001, key="a"*64)]
        _, fast, states, _, _ = self.run_pair(boundaries=[0, 1, 2, 127, 128, 873, 1001], docs=docs, total=1001)
        self.assertEqual(fast.global_events[0]["online_projection_rows"], 0)
        self.assertEqual(fast.global_events[0]["complete_compressed_ids"], [])
        self.assertEqual(fast.global_carry[8][0], 1000)
        # The unpaired prompt row is really present in the native state ring;
        # the next decode row can complete this exact pair without old raw data.
        carry = fast.global_carry[8][1]
        self.assertTensorEqual(states[1].ca_state[0, 0, 0], carry[:512])
        self.assertTensorEqual(states[1].ca_state[0, 0, 1], carry[512:])
        # Read the ring's FP32 gate/latent back in original column order and
        # complete the odd pair with a synthetic next decode projection. This
        # tests state continuity, not the remote CUDA decode kernel itself.
        ring_raw = self.backend.cat((states[1].ca_state[0, 0, 0], states[1].ca_state[0, 0, 1]), dim=0)
        following = self.backend.tensor(np.linspace(-2, 2, 1024, dtype=np.float32),
                                       dtype=self.backend.float32, device="cpu")
        actual_pair = self.backend.cat((ring_raw.unsqueeze(0), following.unsqueeze(0)), dim=0)
        expected_pair = self.backend.cat((carry.unsqueeze(0), following.unsqueeze(0)), dim=0)
        self.assertTensorEqual(fast.ops.compress_canonical(fast.layers[8].self_attn, actual_pair),
                               fast.ops.compress_canonical(fast.layers[8].self_attn, expected_pair))
    def test_source20_direct_overlay_has_no_current_raw_gather(self):
        _, fast, _, _, _ = self.run_pair(boundaries=[0, 175, 303, 609], layer=20)
        for event in fast.system_global_update_events:
            self.assertEqual(event["affected_pair_member_rows"], 0)
            self.assertEqual(event["offline_raw_rows_gathered"], 0)
            self.assertEqual(event["compressor_tail_rows_gathered"], 0)
            self.assertEqual(event["full_current_raw_rows_materialized"], 0)
    def test_source20_qualification_preserves_both_checks(self):
        self.run_pair(boundaries=[0, 175, 609], layer=20, verify=True)
    def test_nondocument_gap_pair_members(self):
        docs = [dict(start=3, end=303, key="a"*64), dict(start=310, end=615, key="b"*64)]
        self.run_pair(boundaries=[0, 303, 309, 311, 617], docs=docs, total=617)
    def test_disabled_branch_is_original_implementation(self):
        _, fast, _, _, _ = self.run_pair(boundaries=[0, 175, 609], enabled=False)
        self.assertEqual(fast.system_global_update_events, [])
        self.assertTrue(fast.report()["system_global_update_diagnostics"]["baseline_global_update_unchanged"])
    def test_formal_fast_path_never_calls_full_raw_builder(self):
        fast, layers, _ = fixture(self.backend, optimized=True)
        selected = fast.selected
        raw = self.backend.zeros((len(selected), 1024), dtype=self.backend.float32, device="cpu")
        native = NS(ca_state=self.backend.zeros((1, 2, 2, 512), dtype=self.backend.float32, device="cpu"))
        meta = NS(positions_c={"2": self.backend.tensor(list(range(304)), dtype=self.backend.int32, device="cpu")},
                  state_block_idx=self.backend.tensor([0], dtype=self.backend.int32, device="cpu"))
        with patch.object(fast, "_offline_raw", side_effect=AssertionError("full raw assembly forbidden")):
            fast.global_update(layers[8].self_attn, raw, native, meta,
                NS(start=0, end=609, positions_cpu=selected, verify=False, mode="sensitive"))
    def test_configure_takes_effect_only_at_next_phase(self):
        fast, _, _ = fixture(self.backend, optimized=True)
        fast.configure(enabled=False)
        self.assertTrue(fast.report()["system_global_update_diagnostics"]["enabled"])
        fast.begin_phase("baseline-next")
        self.assertFalse(fast.report()["system_global_update_diagnostics"]["enabled"])
        self.assertEqual(fast.system_global_update_events, [])
        with self.assertRaisesRegex(ValueError, "bool"):
            fast.configure(enabled=1)
    def test_missing_carry_and_nondocument_projection_fail_closed(self):
        fast, _, _ = fixture(self.backend, optimized=True)
        counts = dict(offline_raw_rows_gathered=0, online_raw_rows_gathered=0,
                      previous_carry_rows_gathered=0)
        raw = self.backend.zeros((0, 1024), dtype=self.backend.float32, device="cpu")
        with self.assertRaisesRegex(ValueError, "missing C2"):
            fast._current_rows(8, [174], raw, {}, NS(start=175, end=303), counts)
        with self.assertRaisesRegex(ValueError, "non-document"):
            fast._current_rows(8, [0], raw, {}, NS(start=0, end=3), counts)


class NumpyAssemblyParity(CacheParityMixin, unittest.TestCase):
    backend = ArrayBackend


@unittest.skipIf(torch is None, "real Torch CPU is required at remote admission")
class TorchAssemblyParity(CacheParityMixin, unittest.TestCase):
    backend = torch


if __name__ == "__main__":
    unittest.main()
