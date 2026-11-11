"""Original/optimized CPU tensor parity; no native CUDA execution is claimed."""
from __future__ import annotations

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from kv_cache_engram.s1_attention import S1Attention
from kv_cache_engram.s1_cache import fixed_selection
from kv_cache_engram.s1_system_attention import S1SystemAttention

try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None


if np is not None:
    class Array(np.ndarray):
        """A small CPU tensor adapter, not a model/native-kernel mock."""
        @property
        def device(self):
            return "cpu"
        def numel(self): return self.size
        def element_size(self): return self.itemsize
        def new_empty(self, shape): return np.empty(shape, dtype=self.dtype).view(Array)
        def contiguous(self): return np.ascontiguousarray(self).view(Array)
        def clone(self): return self.copy().view(Array)
        def to(self, dtype): return self.astype(dtype).view(Array)
        def long(self): return self.to(np.int64)
        def clamp(self, min=None, max=None): return np.clip(self, min, max).view(Array)
        def index_select(self, dim, ids): return np.take(self, ids, axis=dim).view(Array)
        def index_copy_(self, dim, ids, value):
            if dim != 0: raise ValueError("fixture only supports row copies")
            self[ids] = value
            return self

    class NumpyTorch:
        long, int32, float32, float64 = np.int64, np.int32, np.float32, np.float64
        @staticmethod
        def tensor(values, *, device=None, dtype=None):
            return np.array(values, dtype=dtype).view(Array)
        @staticmethod
        def arange(*args, device=None, dtype=None): return np.arange(*args, dtype=dtype).view(Array)
        @staticmethod
        def empty(shape, *, device=None, dtype=None): return np.empty(shape, dtype=dtype).view(Array)
        @staticmethod
        def clamp(value, min=None, max=None): return value.clamp(min, max)
        @staticmethod
        def cat(values, dim=0): return np.concatenate(values, axis=dim).view(Array)
        @staticmethod
        def searchsorted(values, wanted): return np.searchsorted(values, wanted).view(Array)
        @staticmethod
        def where(condition, yes, no): return np.where(condition, yes, no).view(Array)
        @staticmethod
        def _assert_async(condition, message):
            if not bool(condition): raise RuntimeError(message)


def doc(start, end, key="a"):
    return dict(start=start, end=end, key=key*64)


class SystemAttentionControlTests(unittest.TestCase):
    def make(self, enabled=False):
        return S1SystemAttention(NS(t=NS()), enabled=enabled)

    def test_switch_is_locked_until_next_begin_phase(self):
        value = self.make()
        value.configure(enabled=True)
        self.assertFalse(value.enabled)
        value.begin_phase()
        self.assertTrue(value.enabled)
        value.configure(enabled=False)
        self.assertTrue(value.enabled)
        value.begin_phase()
        self.assertFalse(value.enabled)

    def test_flag_requires_real_bool(self):
        for bad in (0, 1, "yes", None):
            with self.assertRaises(ValueError): self.make(bad)
            with self.assertRaises(ValueError): self.make().configure(enabled=bad)

    def test_disabled_memory_and_indices_call_exact_original(self):
        value = self.make()
        with patch.object(S1Attention, "_memory", return_value="original") as original:
            self.assertEqual(value._memory(1, 2, 3, 4), "original")
            original.assert_called_once_with(1, 2, 3, 4)
        with patch.object(S1Attention, "_indices", return_value="original") as original:
            self.assertEqual(value._indices([7], "cpu", "dtype"), "original")
            original.assert_called_once_with([7], "cpu", "dtype")

    def test_joint_projection_ring_forward_remain_original(self):
        for name in ("_joint", "_project", "_projection", "_store_ring", "_forward"):
            self.assertIs(getattr(S1SystemAttention, name), getattr(S1Attention, name))

    def test_stream_key_prevents_cross_stream_scratch_alias(self):
        value = self.make(True)
        current = [17]
        value.t = NS(cuda=NS(current_stream=lambda device: NS(cuda_stream=current[0])))
        device = NS(type="cuda", index=0)
        one = value._storage_key(device, "bf16")
        current[0] = 22
        self.assertNotEqual(one, value._storage_key(device, "bf16"))


class TensorParityMixin:
    def tensor_bytes(self, value):
        if self.t is torch:
            return value.contiguous().view(torch.uint8).numpy().tobytes()
        return value.tobytes()

    def values(self, positions, layer, dtype=None):
        # Different online/offline and layer values expose stale scratch rows.
        rows = [[float(layer*100000+p)+j/1024 for j in range(512)] for p in positions]
        return self.t.tensor(rows, dtype=dtype or self.t.float32).reshape(len(positions), 1, 512)

    def cache(self, docs, total):
        owner = self
        def halo(layer):
            positions = [p for d in docs for p in range(d["end"]-256, d["end"]-128)]
            return positions, owner.values(positions, layer+100)
        return NS(t=self.t, docs=docs, total=total, halo=halo,
                  ops=NS(prepare_swa=lambda mod, values, positions: values.clone()))

    def pair(self, docs, total):
        cache = self.cache(docs, total)
        return S1Attention(cache), S1SystemAttention(cache, enabled=True)

    def compare(self, base, optimized, start, end, layer, dtype=None, position_dtype=None):
        selected = [p for p in fixed_selection(base.cache.docs, base.cache.total) if start <= p < end]
        dtype = dtype or self.t.float32
        positions = self.t.tensor(selected, dtype=position_dtype or self.t.int32)
        prepared = self.values(selected, layer, dtype)
        # Halo transform normally has the same native BF16 output dtype.
        original_halo = base.cache.halo
        base.cache.halo = lambda l: (lambda p, v: (p, v.to(dtype)))(*original_halo(l))
        try:
            args = NS(layer_idx=layer), prepared, NS(positions=positions), NS(start=start, end=end, positions_cpu=selected)
            expected = base._memory(*args)
            actual = optimized._memory(*args)
        finally:
            base.cache.halo = original_halo
        for x, y in zip(expected[:3], actual[:3]):
            self.assertEqual(tuple(x.shape), tuple(y.shape))
            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(self.tensor_bytes(x), self.tensor_bytes(y))
        self.assertEqual(expected[3], actual[3])
        self.assertEqual(base.previous[layer][0], optimized.previous[layer][0])
        self.assertEqual(self.tensor_bytes(base.previous[layer][1]), self.tensor_bytes(optimized.previous[layer][1]))
        return actual

    def test_layers_reuse_only_geometry_and_scratch_not_kv_values(self):
        base, opt = self.pair([doc(0,300), doc(303,603,"b")], 608)
        previous_memory = None
        for layer in (3, 4, 8, 20, 39):
            result = self.compare(base, opt, 0, 608, layer)
            if previous_memory is not None: self.assertIs(result[0], previous_memory)
            previous_memory = result[0]
        counters = opt.report()["system_optimization"]["counters"]
        self.assertEqual(counters["device_geometry_builds"], 1)
        self.assertEqual(counters["device_geometry_reuses"], 4)
        self.assertEqual(counters["local_workspace_allocations"], 1)
        self.assertEqual(counters["local_workspace_reuses"], 4)

    def test_chunk_boundary_previous_values_do_not_alias_workspace(self):
        base, opt = self.pair([doc(0,8300)], 8304)
        for layer in (3, 4): self.compare(base, opt, 0, 8192, layer)
        saved = self.tensor_bytes(opt.previous[3][1])
        # The other layer reuses scratch and has different KV values.
        self.compare(base, opt, 8192, 8304, 4)
        self.assertEqual(self.tensor_bytes(opt.previous[3][1]), saved)
        self.compare(base, opt, 8192, 8304, 3)

    def test_empty_chunk_and_later_nonempty_chunk(self):
        base, opt = self.pair([doc(0,17000)], 17000)
        for start, end in ((0,8192),(8192,16384),(16384,17000)):
            for layer in (3, 20): self.compare(base, opt, start, end, layer)
        self.assertEqual(opt.report()["system_optimization"]["counters"]["optimized_memory_calls"], 6)

    def test_causal_first_rows_nondoc_gap_and_odd_cut(self):
        base, opt = self.pair([doc(3,303),doc(310,610,"b")], 615)
        for start, end in ((0,201),(201,491),(491,615)):
            for layer in (3,14,39): self.compare(base, opt, start, end, layer)

    def test_device_query_mutation_is_not_hidden_by_cached_mask(self):
        base, opt = self.pair([doc(0,300)], 300)
        self.compare(base,opt,0,300,3)
        selected = fixed_selection(base.cache.docs,300)
        positions = self.t.tensor(selected,dtype=self.t.int32)
        positions[0] = 0
        with self.assertRaises(Exception):
            opt._memory(NS(layer_idx=4),self.values(selected,4),NS(positions=positions),
                        NS(start=0,end=300,positions_cpu=selected))

    def test_new_phase_clears_all_device_and_previous_state(self):
        base, opt = self.pair([doc(0,300)],300)
        self.compare(base,opt,0,300,3)
        old = list(opt._workspaces.values())[0]
        generation = opt.report()["system_optimization"]["phase_generation"]
        opt.begin_phase();base.begin_phase()
        self.assertFalse(opt._device_geometry or opt._device_indices or opt._workspaces or opt.previous)
        self.compare(base,opt,0,300,3)
        self.assertIsNot(old,list(opt._workspaces.values())[0])
        self.assertEqual(opt.report()["system_optimization"]["phase_generation"],generation+1)

    def test_input_scope_change_requires_new_phase(self):
        base,opt=self.pair([doc(0,300)],300)
        self.compare(base,opt,0,300,3)
        opt.cache.total=301
        with self.assertRaisesRegex(ValueError,"input changed"):
            opt._bind_geometry_scope()

    def test_dtype_partitions_indices_and_workspace(self):
        base,opt=self.pair([doc(0,300)],300)
        self.compare(base,opt,0,300,3,dtype=self.t.float32,position_dtype=self.t.int32)
        self.compare(base,opt,0,300,4,dtype=self.t.float64,position_dtype=self.t.long)
        self.assertEqual(len(opt._workspaces),2)
        self.assertEqual(len(opt._device_geometry),2)


@unittest.skipIf(np is None, "NumPy CPU adapter required")
class SystemAttentionArrayTests(TensorParityMixin, unittest.TestCase):
    def setUp(self): self.t = NumpyTorch


@unittest.skipIf(torch is None, "Torch CPU required; remote admission forbids skipped tests")
class SystemAttentionTorchTests(TensorParityMixin, unittest.TestCase):
    def setUp(self): self.t = torch

    def test_real_bf16_byte_parity_across_layers_and_chunk_cut(self):
        base,opt=self.pair([doc(0,8300)],8304)
        for start,end in ((0,8192),(8192,8304)):
            for layer in (3,8,14,20,39):
                self.compare(base,opt,start,end,layer,dtype=torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
