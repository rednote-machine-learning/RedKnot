"""Independent numerical/descriptor tests; no full sglang installation needed.

Run with ``python -m unittest discover -s test/srt/redknot -p
test_head_kv_attention.py``. CUDA tests skip when CUDA/Triton are unavailable.
The dense concatenation below is deliberately confined to the independent test
oracle; neither production backend gathers the KV sequence.
"""

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

import torch


_SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache/head_kv"
_PACKAGE_NAME = "_head_kv_attention_testpkg"
if _PACKAGE_NAME not in sys.modules:
    _package = types.ModuleType(_PACKAGE_NAME)
    _package.__path__ = [str(_SOURCE)]
    sys.modules[_PACKAGE_NAME] = _package
_spec = importlib.util.spec_from_file_location(f"{_PACKAGE_NAME}.attention", _SOURCE / "attention.py")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
paged_attention = _module.paged_attention


def _dense_oracle(query, k, v, slots, lengths, positions, qpos, group, windows=None, sinks=None, causal=True, scale=None):
    """Dense float64 softmax oracle independent of the online recurrence."""
    out = torch.zeros_like(query)
    scale = scale if scale is not None else query.shape[-1] ** -0.5
    for qh in range(query.shape[0]):
        kh = qh // group
        keys, values, pos = [], [], []
        for page in range(slots.shape[1]):
            n = int(lengths[kh, page])
            if n:
                slot = int(slots[kh, page])
                keys.append(k[slot, :n])
                values.append(v[slot, :n])
                pos.append(positions[kh, page, :n])
        if not keys:
            continue
        all_k = torch.cat(keys).double()
        all_v = torch.cat(values).double()
        all_pos = torch.cat(pos)
        target = qpos if qpos.ndim == 1 else qpos[qh]
        scores = query[qh].double() @ all_k.T * scale
        mask = torch.ones_like(scores, dtype=torch.bool)
        if causal:
            mask &= all_pos[None, :] <= target[:, None]
        window = int(windows[kh]) if windows is not None else 0
        sink = int(sinks[kh]) if sinks is not None else 0
        if window > 0:
            mask &= (all_pos[None, :] >= target[:, None] - window + 1) | (all_pos[None, :] < sink)
        scores.masked_fill_(~mask, -torch.inf)
        nonempty = mask.any(dim=-1)
        if nonempty.any():
            out[qh, nonempty] = (torch.softmax(scores[nonempty], dim=-1) @ all_v).to(query.dtype)
    return out


def _case(device="cpu", dtype=torch.float32):
    # Two KV heads, three Q heads per group, irregular/reordered segments.
    torch.manual_seed(174)
    query = torch.randn(6, 7, 5, dtype=dtype, device=device).transpose(1, 2)
    k = torch.randn(8, 7, 4, dtype=dtype, device=device).transpose(1, 2)
    v = torch.randn(8, 7, 4, dtype=dtype, device=device).transpose(1, 2)
    slots = torch.tensor([[4, 1, 6, -1], [5, 0, 3, 2]], dtype=torch.int64, device=device)
    lengths = torch.tensor([[3, 4, 1, 0], [4, 2, 3, 0]], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [[[16, 17, 18, -1], [0, 1, 2, 3], [9, -1, -1, -1], [-1, -1, -1, -1]],
         [[20, 21, 22, 23], [4, 5, -1, -1], [11, 12, 13, -1], [-1, -1, -1, -1]]],
        dtype=torch.int64, device=device,
    )
    # Exercise descriptor strides as well as non-contiguous pool/query strides.
    slots = torch.stack([slots, slots], dim=-1)[..., 0]
    positions = torch.stack([positions, positions], dim=-1)[..., 0]
    qpos = torch.tensor([0, 3, 8, 18, 30], dtype=torch.int64, device=device)
    return query, k, v, slots, lengths, positions, qpos


class HeadKVAttentionCPUTest(unittest.TestCase):
    def _assert_matches(self, *, causal=True, windows=None, sinks=None, dtype=torch.float64, per_head_positions=False):
        q, k, v, slots, lengths, positions, qpos = _case(dtype=dtype)
        if per_head_positions:
            qpos = qpos[None, :].expand(6, 5).clone()
            qpos[1] += 7
        expected = _dense_oracle(q, k, v, slots, lengths, positions, qpos, 3, windows, sinks, causal)
        actual = paged_attention(q, k, v, slots, lengths, positions, query_positions=qpos,
                                 num_q_per_kv=3, windows=windows, sinks=sinks, causal=causal)
        tolerance = 1e-11 if dtype == torch.float64 else 3e-5
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_irregular_reordered_segments_gqa_causal(self):
        self._assert_matches()

    def test_head_specific_window_sink_union(self):
        self._assert_matches(windows=[5, 0], sinks=[2, 0])

    def test_noncausal_negative_window_and_head_positions(self):
        self._assert_matches(causal=False, windows=[-1, 4], sinks=[0, 2], per_head_positions=True)

    def test_float32_online_softmax(self):
        self._assert_matches(dtype=torch.float32, windows=[5, 8], sinks=[2, 3])

    def test_all_masked_and_empty_head_output_zero(self):
        q, k, v, slots, lengths, positions, qpos = _case()
        slots[0] = -1
        lengths[0] = 0
        qpos[:] = 0
        result = paged_attention(q, k, v, slots, lengths, positions, query_positions=qpos, num_q_per_kv=3)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_zero_pages_and_zero_queries(self):
        q = torch.randn(2, 3, 7)
        k, v = torch.empty(0, 4, 7), torch.empty(0, 4, 7)
        slots = torch.empty(2, 0, dtype=torch.int64)
        lengths = torch.empty(2, 0, dtype=torch.int64)
        pos = torch.empty(2, 0, 4, dtype=torch.int64)
        result = paged_attention(q, k, v, slots, lengths, pos, query_positions=torch.arange(3))
        self.assertTrue(torch.equal(result, torch.zeros_like(q)))
        result = paged_attention(q[:, :0], k, v, slots, lengths, pos, query_positions=torch.empty(0, dtype=torch.int64))
        self.assertEqual(result.shape, (2, 0, 7))

    def test_repeated_physical_page_distinct_occurrence_positions(self):
        torch.manual_seed(18)
        q, k, v = torch.randn(1, 2, 5), torch.randn(1, 3, 5), torch.randn(1, 3, 5)
        slots = torch.tensor([[0, 0]])
        lengths = torch.tensor([[3, 3]])
        positions = torch.tensor([[[0, 1, 2], [10, 11, 12]]])
        qpos = torch.tensor([2, 12])
        actual = paged_attention(q, k, v, slots, lengths, positions, query_positions=qpos, windows=[4])
        expected = _dense_oracle(q, k, v, slots, lengths, positions, qpos, 1, [4])
        torch.testing.assert_close(actual, expected)

    def test_large_scores_stable_softmax(self):
        q, k, v, slots, lengths, positions, qpos = _case()
        q *= 100
        k *= 100
        actual = paged_attention(q, k, v, slots, lengths, positions, query_positions=qpos, num_q_per_kv=3)
        expected = _dense_oracle(q, k, v, slots, lengths, positions, qpos, 3)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_invalid_descriptors_rejected_before_access(self):
        mutations = (
            ("bad slot", lambda s, n, p: s.__setitem__((0, 0), 8)),
            ("negative slot", lambda s, n, p: s.__setitem__((0, 0), -2)),
            ("oversized length", lambda s, n, p: n.__setitem__((0, 0), 5)),
            ("negative length", lambda s, n, p: n.__setitem__((0, 0), -1)),
            ("absent page nonzero length", lambda s, n, p: s.__setitem__((0, 0), -1)),
            ("missing active position", lambda s, n, p: p.__setitem__((0, 0, 0), -1)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                q, k, v, s, n, p, qp = _case()
                mutate(s, n, p)
                with self.assertRaises(ValueError):
                    paged_attention(q, k, v, s, n, p, query_positions=qp, num_q_per_kv=3)

    def test_invalid_metadata_and_dtype(self):
        q, k, v, s, n, p, qp = _case()
        for kwargs in ({"num_q_per_kv": 4}, {"num_q_per_kv": True}, {"windows": [1]},
                       {"sinks": [-1, 0]}, {"scale": math.inf}, {"backend": "bogus"},
                       {"backend": "triton"}):
            options = {"num_q_per_kv": 3, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                paged_attention(q, k, v, s, n, p, query_positions=qp, **options)
        with self.assertRaises(TypeError):
            paged_attention(q, k, v, s.float(), n, p, query_positions=qp, num_q_per_kv=3)
        with self.assertRaises(ValueError):
            paged_attention(q, k, v, s, n, p, query_positions=qp[:-1], num_q_per_kv=3)
        with self.assertRaises(ValueError):
            paged_attention(q.requires_grad_(), k, v, s, n, p, query_positions=qp, num_q_per_kv=3)


class ManagedSegPagedBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib import import_module
        cls.HeadPagePool = import_module(f"{_PACKAGE_NAME}.pool").HeadPagePool
        cls.HeadKVManager = import_module(f"{_PACKAGE_NAME}.manager").HeadKVManager
        cls.ManagedCache = import_module(f"{_PACKAGE_NAME}.segpaged").ManagedSegPagedKVCache
        legacy_path = _SOURCE.parents[1] / "layers/attention/redknot/segpaged.py"
        spec = importlib.util.spec_from_file_location("_redknot_legacy_segpaged_test", legacy_path)
        cls.legacy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.legacy
        spec.loader.exec_module(cls.legacy)

    def _cache(self):
        pool = self.HeadPagePool(32, 4, 8)
        manager = self.HeadKVManager(pool)
        manager.create_request("parent", context_id="same-context")
        cache = self.ManagedCache(manager, "parent", num_layers=1, num_kv_heads=2)
        return pool, manager, cache

    def test_existing_entry_dispatch_and_cow_parent_child_outputs(self):
        torch.manual_seed(412)
        pool, manager, parent = self._cache()
        k, v = torch.randn(2, 7, 8), torch.randn(2, 7, 8)
        for h in range(2):
            parent.add_head_segment(layer=0, head=h, segment="document", policy="global" if h == 0 else "local",
                                    k=k[h], v=v[h], positions=range(7), provenance=f"head{h}",
                                    window=None if h == 0 else 3, sink=0 if h == 0 else 1)
        child = parent.fork("child")
        original_refs = manager.version("parent").segments[(0, 0, "document")].pages
        replacement_k, replacement_v = torch.randn(2, 8), torch.randn(2, 8)
        child.repair_head_segment(layer=0, head=0, segment="document", indices=(1, 5),
                                  k=replacement_k, v=replacement_v, provenance="repaired-head0")
        self.assertEqual(manager.version("parent").segments[(0, 0, "document")].pages, original_refs)
        self.assertNotEqual(manager.version("child").segments[(0, 0, "document")].pages, original_refs)
        self.assertEqual(manager.version("parent").segments[(0, 1, "document")].pages,
                         manager.version("child").segments[(0, 1, "document")].pages)
        query = torch.randn(4, 2, 8)
        qpos = torch.tensor([5, 6])
        outputs = []
        for cache in (parent, child):
            output = self.legacy.segpaged_attention(query, cache, layer=0, query_positions=qpos,
                                                    num_q_per_kv=2, sm_scale=8**-0.5, use_fused=False)
            with manager.bind(cache.request_id) as lease:
                desc = lease.descriptor(0, 2)
                expected = _dense_oracle(query, desc["k_pool"], desc["v_pool"], desc["page_slots"],
                                         desc["page_lengths"], desc["key_positions"], qpos, 2, [0, 3], [0, 1])
            torch.testing.assert_close(output, expected)
            outputs.append(output)
        self.assertFalse(torch.allclose(outputs[0][:2], outputs[1][:2]))
        torch.testing.assert_close(outputs[0][2:], outputs[1][2:])
        parent_metadata = parent._descriptor_cache[0][1]
        repeat = self.legacy.segpaged_attention(query, parent, layer=0, query_positions=qpos,
                                                num_q_per_kv=2, sm_scale=8**-0.5, use_fused=False)
        self.assertIs(parent._descriptor_cache[0][1], parent_metadata)
        torch.testing.assert_close(repeat, outputs[0])
        manager.release_request("parent")
        manager.release_request("child")
        self.assertEqual(pool.stats()["free_pages"], 32)

    def test_explicit_positions_required_and_causal_default(self):
        _, manager, cache = self._cache()
        k, v = torch.randn(3, 8), torch.randn(3, 8)
        cache.add_head_segment(layer=0, head=0, segment=0, policy="global", k=k, v=v,
                               positions=(0, 1, 2), provenance="kv")
        q = torch.randn(2, 1, 8)
        with self.assertRaises(ValueError):
            self.legacy.segpaged_attention(q, cache, layer=0, num_q_per_kv=1, sm_scale=1.0)
        out = self.legacy.segpaged_attention(q, cache, layer=0, num_q_per_kv=1, sm_scale=1.0,
                                             query_positions=torch.tensor([0]), use_fused=False)
        torch.testing.assert_close(out[0, 0], v[0])
        self.assertTrue(torch.equal(out[1], torch.zeros_like(out[1])))
        manager.release_request("parent")

    def test_legacy_behavior_preserved(self):
        cache = self.legacy.SegPagedKVCache(num_layers=1, num_kv_heads=1, head_dim=8,
                                           page_size=4, dtype=torch.float32)
        k, v, q = torch.randn(3, 8), torch.randn(3, 8), torch.randn(1, 2, 8)
        cache.add_head_segment(layer=0, head=0, segment=0, policy="global", k=k, v=v)
        actual = self.legacy.segpaged_attention(q, cache, layer=0, num_q_per_kv=1, sm_scale=8**-0.5, use_fused=False)
        expected = torch.softmax(q @ k.T * 8**-0.5, dim=-1) @ v
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            self.legacy.segpaged_attention(q, cache, layer=0, num_q_per_kv=1, sm_scale=1.0,
                                           query_positions=torch.tensor([1, 2]), use_fused=False)

    def test_nonexact_adapter_certificate_forwarded_and_preserved(self):
        for kind in ("certified_transform", "policy_approximate"):
            with self.subTest(reuse_kind=kind):
                pool, manager, cache = self._cache()
                payload = torch.ones(3, 8)
                args = dict(layer=0, head=0, segment="D", policy="global", k=payload, v=payload,
                            positions=(0, 1, 2), provenance="adapter-payload", reuse_kind=kind)
                with self.assertRaisesRegex(ValueError, "validity certificate"):
                    cache.add_head_segment(**args)
                segment = cache.add_head_segment(**args, validity_certificate="trusted-adapter-attestation")
                self.assertEqual(segment.reuse_kind, kind)
                self.assertEqual(segment.validity_certificate, "trusted-adapter-attestation")
                child = cache.fork("child")
                repaired = child.repair_head_segment(layer=0, head=0, segment="D", indices=(1,),
                                                       k=payload[:1]*2, v=payload[:1]*2, provenance="repair")
                self.assertEqual(repaired.reuse_kind, kind)
                self.assertEqual(repaired.validity_certificate, "trusted-adapter-attestation")
                manager.release_request("parent")
                manager.release_request("child")
                self.assertEqual(pool.stats()["free_pages"], 32)


_HAS_CUDA_TRITON = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


@unittest.skipUnless(_HAS_CUDA_TRITON, "CUDA and Triton are required")
class HeadKVAttentionGPUTest(unittest.TestCase):
    def test_irregular_gqa_strides_masks_and_dtypes(self):
        for dtype, tolerance in ((torch.float32, 2e-4), (torch.float16, 3e-3), (torch.bfloat16, 2e-2)):
            for causal in (False, True):
                with self.subTest(dtype=dtype, causal=causal):
                    q, k, v, s, n, p, qp = _case("cuda", dtype)
                    expected = _dense_oracle(q, k, v, s, n, p, qp, 3, [5, 0], [2, 0], causal)
                    actual = paged_attention(q, k, v, s, n, p, query_positions=qp, num_q_per_kv=3,
                                             windows=[5, 0], sinks=[2, 0], causal=causal, backend="triton")
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_multiple_token_tiles_decode_and_repair(self):
        for page_size, dim, nq in ((17, 32, 1), (64, 128, 1), (97, 80, 7)):
            with self.subTest(page_size=page_size, dim=dim, nq=nq):
                torch.manual_seed(5)
                q = torch.randn(4, nq, dim, device="cuda", dtype=torch.float16)
                k = torch.randn(6, page_size, dim, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                s = torch.tensor([[2, 0, 4], [5, 1, 3]], device="cuda", dtype=torch.int32)
                n = torch.tensor([[page_size, page_size - 3, 1], [page_size - 2, 0, page_size]], device="cuda", dtype=torch.int32)
                p = torch.arange(3 * page_size, device="cuda").reshape(1, 3, page_size).expand(2, 3, page_size)
                qp = torch.arange(nq, device="cuda") + page_size * 2
                actual = paged_attention(q, k, v, s, n, p, query_positions=qp, num_q_per_kv=2, windows=[35, 0], sinks=[2, 0])
                expected = _dense_oracle(q, k, v, s, n, p, qp, 2, [35, 0], [2, 0])
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)

    def test_empty_and_all_masked_gpu(self):
        q, k, v, s, n, p, qp = _case("cuda", torch.float16)
        s[0], n[0] = -1, 0
        qp[:] = 0
        actual = paged_attention(q, k, v, s, n, p, query_positions=qp, num_q_per_kv=3)
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
        actual = paged_attention(q, k, v, s[:, :0], n[:, :0], p[:, :0], query_positions=qp, num_q_per_kv=3)
        self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_split_page_softmax_with_empty_partitions(self):
        torch.manual_seed(15)
        q = torch.randn(4, 2, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(34, 32, 64, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        s = torch.arange(34, device="cuda", dtype=torch.int32).reshape(2, 17)
        n = torch.full((2, 17), 32, device="cuda", dtype=torch.int32)
        n[1, :12] = 0
        p = torch.arange(17 * 32, device="cuda").reshape(1, 17, 32).expand(2, 17, 32)
        qp = torch.tensor([10, 500], device="cuda", dtype=torch.int64)
        actual = paged_attention(q, k, v, s, n, p, query_positions=qp, num_q_per_kv=2, windows=[64, 0], sinks=[3, 0])
        expected = _dense_oracle(q, k, v, s, n, p, qp, 2, [64, 0], [3, 0])
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    unittest.main()
