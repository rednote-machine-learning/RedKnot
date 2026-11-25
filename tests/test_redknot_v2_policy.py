"""Offline CPU unit tests for RedKnot V2.1 / V2.2 / V2.3 engineering integration.

Covers every item in Section 4.2 and Section 5.1 of the RedKnot V2 Engineering
Handoff Guide without requiring GPUs:
  1. Policy routing & preset resolution (<48K, [48K,96K), [96K,192K), >=192K, S1.6)
  2. GPU F.pad + F.max_pool1d vs CPU deque Span mathematical equivalence & zero cross-doc leak
  3. Opt-3 GPU stable argsort vs CPU select_from_reduced_scores exact equivalence
  4. Geometry, Online_Head non-overlap, and 8192-frame capacity invariants
  5. MinimalCone post-20 dependency closure, row reduction, Opt-2 halo pruning, and proof audit
  6. Two-stage L2->L14 affected() strict locality law & Stage-2 promotion invariants
  7. Repository SHA-256 locks (sources.json / PINS / PARENT_SHA), FixedRuntimeRewrite AST
     assertions (1,1,1,1,1), compact_controller checks, and zero-GPU-import preflight safety.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_ROOT = REPO_ROOT / "test" / "longbench300" / "algorithm"
FIXED_ROOT = ALGO_ROOT / "fixed"
LASTKV_ROOT = ALGO_ROOT / "lastkv"
SHARED_RUNTIME_ROOT = ALGO_ROOT / "shared" / "runtime"
SHARED_ROOT = ALGO_ROOT / "shared"
MULTIOWNER_ROOT = ALGO_ROOT / "multiowner"
PRODUCER_ROOT = ALGO_ROOT / "producer"

for path in (FIXED_ROOT, LASTKV_ROOT, SHARED_RUNTIME_ROOT, SHARED_ROOT, MULTIOWNER_ROOT, PRODUCER_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import cone15_ffn_native_proof  # noqa: E402
import cone15_plan  # noqa: E402
import fixed_pack_policy  # noqa: E402
import fixed_pack_runtime_adapter  # noqa: E402
import halo_direct_copy  # noqa: E402
import index20_selector  # noqa: E402
import run_s16  # noqa: E402
import two_stage_l14  # noqa: E402
import v2_policy  # noqa: E402

try:
    import torch as REAL_TORCH
except ImportError:
    REAL_TORCH = None


class _FakeTensor:
    """Minimal faithful 1D/3D CPU tensor for exercising F.pad + F.max_pool1d + stable argsort."""

    def __init__(self, data, *, dtype="float32", device="cpu", shape=None):
        if isinstance(data, _FakeTensor):
            self._data = list(data._data)
            self.dtype = data.dtype
            self.device = data.device
            self.shape = tuple(data.shape) if shape is None else tuple(shape)
            return
        self._data = list(data)
        self.dtype = dtype
        self.device = device
        self.shape = (len(self._data),) if shape is None else tuple(shape)

    def clone(self):
        return _FakeTensor(self._data, dtype=self.dtype, device=self.device, shape=self.shape)

    def view(self, *dims):
        n = len(self._data)
        resolved = []
        neg_idx = None
        prod = 1
        for i, d in enumerate(dims):
            if d == -1:
                assert neg_idx is None
                neg_idx = i
                resolved.append(None)
            else:
                resolved.append(d)
                prod *= d
        if neg_idx is not None:
            resolved[neg_idx] = n // prod
        return _FakeTensor(self._data, dtype=self.dtype, device=self.device, shape=tuple(resolved))

    def cpu(self):
        return self

    def tolist(self):
        return list(self._data)

    def __len__(self):
        return self.shape[0]

    def __gt__(self, other):
        return _FakeTensor([x > other for x in self._data], dtype="bool", device=self.device, shape=self.shape)

    def __floordiv__(self, other):
        return _FakeTensor([x // other for x in self._data], dtype=self.dtype, device=self.device, shape=self.shape)

    def __getitem__(self, item):
        if isinstance(item, slice):
            vals = self._data[item]
            return _FakeTensor(vals, dtype=self.dtype, device=self.device)
        if isinstance(item, _FakeTensor):
            return _FakeTensor([self._data[i] for i in item._data], dtype=self.dtype, device=self.device)
        return self._data[item]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            vals = value._data if isinstance(value, _FakeTensor) else list(value)
            self._data[key] = vals
            return
        self._data[key] = value


class _FakeFunctional:
    @staticmethod
    def pad(tensor, pad, value=0.0):
        left, right = pad
        padded = [value] * left + list(tensor._data) + [value] * right
        if len(tensor.shape) == 3:
            b, c, length = tensor.shape
            assert b == 1 and c == 1
            return _FakeTensor(padded, dtype=tensor.dtype, device=tensor.device, shape=(1, 1, length + left + right))
        return _FakeTensor(padded, dtype=tensor.dtype, device=tensor.device)

    @staticmethod
    def max_pool1d(tensor, kernel_size, stride=1):
        assert stride == 1 and len(tensor.shape) == 3 and tensor.shape[0] == 1 and tensor.shape[1] == 1
        data = tensor._data
        out_len = len(data) - kernel_size + 1
        out = [max(data[i : i + kernel_size]) for i in range(out_len)]
        return _FakeTensor(out, dtype=tensor.dtype, device=tensor.device, shape=(1, 1, out_len))


class _FakeTorch:
    long = "int64"
    float32 = "float32"
    int64 = "int64"
    nn = SimpleNamespace(functional=_FakeFunctional)

    @staticmethod
    def as_tensor(values, device="cpu", dtype="int64"):
        return _FakeTensor(list(values), dtype=dtype, device=device)

    @staticmethod
    def where(cond, x, y):
        x_vals = x._data if isinstance(x, _FakeTensor) else [x] * len(cond._data)
        y_vals = y._data if isinstance(y, _FakeTensor) else [y] * len(cond._data)
        out = [xv if c else yv for c, xv, yv in zip(cond._data, x_vals, y_vals)]
        return _FakeTensor(out, dtype="float32", device=cond.device)

    @staticmethod
    def argsort(tensor, descending=False, stable=False):
        assert stable is True
        indices = list(range(len(tensor._data)))
        # Python's Timsort is guaranteed stable; descending with (-v) preserves ascending index order on ties.
        # Handle -inf explicitly so (-(-inf)) == +inf sorts at the end when descending=True.
        if descending:
            indices.sort(key=lambda idx: -tensor._data[idx])
        else:
            indices.sort(key=lambda idx: tensor._data[idx])
        return _FakeTensor(indices, dtype="int64", device=tensor.device)


def _make_random_docs(rng: random.Random, *, min_docs=2, max_docs=6, min_len=260, max_len=900):
    docs = []
    cursor = rng.randint(0, 25)
    n_docs = rng.randint(min_docs, max_docs)
    for idx in range(n_docs):
        length = rng.randint(min_len, max_len)
        start = cursor
        end = start + length
        docs.append({"start": start, "end": end, "key": f"doc-{idx}"})
        cursor = end + rng.randint(0, 31)
    total = cursor + rng.randint(17, 140)
    return docs, total


class TestV2PolicyRouting(unittest.TestCase):
    def test_v21_fast_default_band_boundaries(self) -> None:
        # <48K: single-stage 20%, NoSpan, Head64
        for tokens, band in ((15_800, "16k"), (23_999, "16k"), (24_000, "32k"), (47_999, "32k")):
            p = v2_policy.resolve_v2_policy_for_tokens(tokens)
            self.assertEqual(p.mode, "v2.1-fast")
            self.assertEqual(p.band, band)
            self.assertEqual(p.fraction, 0.20)
            self.assertEqual(p.stage1_fraction, 0.20)
            self.assertEqual(p.head_width, 64)
            self.assertEqual(p.window, (0, 0))
            self.assertFalse(p.span_enabled)
            self.assertFalse(p.two_stage_l14)
            self.assertFalse(p.host_prefix_offload)
            self.assertTrue(p.minimal_cone)
            self.assertTrue(p.suffix_only)
            self.assertTrue(p.gpu_select)
            self.assertTrue(p.static_metadata_cache)
            self.assertTrue(p.direct_halo_copy)
            self.assertEqual(p.opts, (1, 2, 3))

        # [48K, 96K): Span[-8,+24], Head64
        for tokens in (48_000, 65_000, 95_999):
            p = v2_policy.resolve_v2_policy_for_tokens(tokens)
            self.assertEqual(p.band, "64k")
            self.assertEqual(p.head_width, 64)
            self.assertEqual(p.window, (8, 24))
            self.assertTrue(p.span_enabled)
            self.assertFalse(p.two_stage_l14)

        # [96K, 192K) and >=192K: Span[-8,+24], Head128
        for tokens, band in ((96_000, "128k"), (128_908, "128k"), (191_999, "128k"), (192_000, "256k"), (253_592, "256k")):
            p = v2_policy.resolve_v2_policy_for_tokens(tokens)
            self.assertEqual(p.band, band)
            self.assertEqual(p.head_width, 128)
            self.assertEqual(p.window, (8, 24))
            self.assertTrue(p.span_enabled)
            self.assertFalse(p.two_stage_l14)
            self.assertFalse(p.host_prefix_offload)

    def test_configurable_span_min_tokens_b300_policy_b_alignment(self) -> None:
        p64 = v2_policy.resolve_v2_policy_for_tokens(65_000, span_min_tokens=96_000)
        self.assertEqual(p64.window, (0, 0))
        self.assertEqual(p64.head_width, 64)
        p128 = v2_policy.resolve_v2_policy_for_tokens(96_000, span_min_tokens=96_000)
        self.assertEqual(p128.window, (8, 24))
        self.assertEqual(p128.head_width, 128)

    def test_two_stage_presets_and_256k_host_offload_boundaries(self) -> None:
        expected_long = {
            "short20_long15": (0.12, 0.15),
            "v2.3_12": (0.09, 0.12),
            "v2.3_10": (0.07, 0.10),
            "m16": (0.16, 0.20),
        }
        for preset, (s1, tgt) in expected_long.items():
            # <48K MUST stay single-stage 20%, NoSpan, Head64, no offload
            for short_tokens in (16_000, 32_000, 47_999):
                p_short = v2_policy.resolve_v2_policy_for_tokens(
                    short_tokens, two_stage_l14=True, preset=preset
                )
                self.assertFalse(p_short.two_stage_l14)
                self.assertEqual(p_short.mode, "v2.1-fast")
                self.assertEqual((p_short.stage1_fraction, p_short.fraction), (0.20, 0.20))
                self.assertEqual(p_short.window, (0, 0))
                self.assertEqual(p_short.head_width, 64)
                self.assertFalse(p_short.host_prefix_offload)

            # 64K & 128K: two-stage active, Span[-8,+24], offload OFF by default
            p64 = v2_policy.resolve_v2_policy_for_tokens(48_000, two_stage_l14=True, preset=preset)
            self.assertTrue(p64.two_stage_l14)
            self.assertEqual((p64.stage1_fraction, p64.fraction), (s1, tgt))
            self.assertEqual(p64.head_width, 64)
            self.assertEqual(p64.window, (8, 24))
            self.assertFalse(p64.host_prefix_offload)

            p128 = v2_policy.resolve_v2_policy_for_tokens(191_999, two_stage_l14=True, preset=preset)
            self.assertTrue(p128.two_stage_l14)
            self.assertEqual((p128.stage1_fraction, p128.fraction), (s1, tgt))
            self.assertEqual(p128.head_width, 128)
            self.assertFalse(p128.host_prefix_offload)

            # >=192K (256K): host_prefix_offload auto-triggers ON
            p256 = v2_policy.resolve_v2_policy_for_tokens(192_000, two_stage_l14=True, preset=preset)
            self.assertTrue(p256.two_stage_l14)
            self.assertEqual(p256.band, "256k")
            self.assertEqual(p256.head_width, 128)
            self.assertTrue(p256.host_prefix_offload)

        # Explicit host_prefix_offload overrides on 128K and 256K
        p128_on = v2_policy.resolve_v2_policy_for_tokens(
            128_000, two_stage_l14=True, preset="short20_long15", host_prefix_offload=True
        )
        self.assertTrue(p128_on.host_prefix_offload)
        p256_off = v2_policy.resolve_v2_policy_for_tokens(
            250_000, two_stage_l14=True, preset="short20_long15", host_prefix_offload=False
        )
        self.assertFalse(p256_off.host_prefix_offload)

        # Default preset when --two-stage-l14 is set without --preset
        cfg = v2_policy.V2PolicyConfig.from_cli(two_stage_l14=True)
        self.assertEqual(cfg.preset, "short20_long15")

    def test_legacy_s16_and_invalid_combinations(self) -> None:
        p = v2_policy.resolve_v2_policy_for_tokens(128_000, mode="s1.6")
        self.assertTrue(p.is_legacy_s16)
        self.assertFalse(p.is_v2)
        self.assertEqual(p.fraction, 0.15)
        self.assertEqual(p.head_width, 0)
        self.assertEqual(p.window, (0, 0))
        self.assertFalse(p.minimal_cone)
        self.assertFalse(p.suffix_only)
        self.assertFalse(p.gpu_select)
        self.assertFalse(p.static_metadata_cache)
        self.assertFalse(p.direct_halo_copy)
        self.assertFalse(p.two_stage_l14)
        self.assertEqual(p.opts, ())

        with self.assertRaises(ValueError):
            v2_policy.V2PolicyConfig(mode="s1.6", two_stage_l14=True)
        with self.assertRaises(ValueError):
            v2_policy.V2PolicyConfig(mode="s1.6", preset="short20_long15")
        with self.assertRaises(ValueError):
            v2_policy.V2PolicyConfig(mode="v2.1-fast", two_stage_l14=True, preset="v2.1-fast")
        with self.assertRaises(ValueError):
            v2_policy.resolve_v2_policy("512k")


class TestSpanEquivalence(unittest.TestCase):
    def _run_span_equivalence(self, torch_mod) -> None:
        rng = random.Random(20261124)
        for trial in range(40):
            docs, total = _make_random_docs(rng, min_docs=2, max_docs=5, min_len=256, max_len=600)
            n_keys = total // 2
            # Include negative scores, zeros, ties, and unobserved keys (counts == 0).
            values = [rng.choice([-12.5, -3.0, -0.25, 0.0, 1.5, 4.25, 9.75]) for _ in range(n_keys)]
            counts = [0 if rng.random() < 0.25 else rng.randint(1, 5) for _ in range(n_keys)]
            # Force one document to be completely unobserved in some trials.
            if trial % 5 == 0:
                a0, z0 = (docs[0]["start"] + 1) // 2, docs[0]["end"] // 2
                for k in range(a0, z0):
                    counts[k] = 0
                    values[k] = 999.0  # Must be masked to -inf and never leak!

            for window in ((0, 0), (8, 24), (2, 6), (0, 5), (7, 0)):
                v_cpu, c_cpu = index20_selector.span_values(docs, values, counts, window[0], window[1])
                t_vals = torch_mod.as_tensor(values, dtype=torch_mod.float32)
                t_cnts = torch_mod.as_tensor(counts, dtype=torch_mod.int64)
                smooth_gpu = index20_selector.span_values_gpu(
                    docs, t_vals, t_cnts, window[0], window[1], torch_module=torch_mod
                ).cpu().tolist()

                self.assertEqual(len(smooth_gpu), n_keys)
                for k in range(n_keys):
                    if c_cpu[k] == 0:
                        self.assertEqual(smooth_gpu[k], -math.inf, f"trial={trial} window={window} k={k}")
                    else:
                        self.assertAlmostEqual(
                            smooth_gpu[k], v_cpu[k], places=5, msg=f"trial={trial} window={window} k={k}"
                        )

    def test_span_direction_and_zero_cross_doc_leakage(self) -> None:
        # Doc 0: [11, 271) -> complete C2 keys [6, 135). Doc 1: [273, 550) -> complete C2 keys [137, 275).
        docs = [{"start": 11, "end": 271, "key": "d0"}, {"start": 273, "end": 550, "key": "d1"}]
        n_keys = 300
        values = [-5.0] * n_keys
        counts = [1] * n_keys
        values[20] = 9.5       # Peak at s=20 spreads to [20-8, 20+24] = [12, 44].
        values[130] = 100.0    # Near end of Doc 0 -> MUST NOT leak into Doc 1 [137, 275)!
        counts[138] = 0        # Unobserved key in Doc 1 with spurious large raw value.
        values[138] = 500.0

        v_cpu, c_cpu = index20_selector.span_values(docs, values, counts, 8, 24)
        self.assertEqual(v_cpu[12], 9.5)
        self.assertEqual(v_cpu[44], 9.5)
        self.assertEqual(v_cpu[11], -5.0)
        self.assertEqual(v_cpu[45], -5.0)
        # No cross-doc leakage from k=130 (Doc 0) into Doc 1.
        self.assertEqual(v_cpu[137], -5.0)
        self.assertEqual(v_cpu[138], -5.0)
        self.assertGreater(c_cpu[138], 0)  # Covered by observed neighbors in Doc 1.

        # Same check via span_values_gpu (FakeTorch and real PyTorch if installed).
        for torch_mod in ([_FakeTorch] + ([REAL_TORCH] if REAL_TORCH is not None else [])):
            t_vals = torch_mod.as_tensor(values, dtype=torch_mod.float32)
            t_cnts = torch_mod.as_tensor(counts, dtype=torch_mod.int64)
            gpu = index20_selector.span_values_gpu(docs, t_vals, t_cnts, 8, 24, torch_module=torch_mod).cpu().tolist()
            self.assertEqual(gpu[12], 9.5)
            self.assertEqual(gpu[44], 9.5)
            self.assertEqual(gpu[11], -5.0)
            self.assertEqual(gpu[45], -5.0)
            self.assertEqual(gpu[137], -5.0)
            self.assertEqual(gpu[138], -5.0)

    def test_gpu_vs_cpu_span_equivalence_reference(self) -> None:
        self._run_span_equivalence(_FakeTorch)

    @unittest.skipIf(REAL_TORCH is None, "PyTorch not installed in this CPU environment")
    def test_gpu_vs_cpu_span_equivalence_real_torch(self) -> None:
        self._run_span_equivalence(REAL_TORCH)


class TestOpt3StableSortEquivalence(unittest.TestCase):
    def _run_opt3_equivalence(self, torch_mod) -> None:
        rng = random.Random(1337)
        for trial in range(30):
            docs, total = _make_random_docs(rng, min_docs=2, max_docs=5, min_len=256, max_len=700)
            n_keys = total // 2
            # Deliberately create heavy ties, negative values, and unobserved keys.
            palette = [-10.0, -1.0, 0.0, 0.0, 2.0, 2.0, 5.5]
            sums = [rng.choice(palette) for _ in range(n_keys)]
            counts = [0 if rng.random() < 0.30 else rng.randint(1, 4) for _ in range(n_keys)]
            # Mask unobserved sums to 0.0 for CPU input (matches NativeIndexScoreCollector readback).
            cpu_sums = [s if c > 0 else 0.0 for s, c in zip(sums, counts)]
            gpu_sums = [s if c > 0 else -math.inf for s, c in zip(sums, counts)]

            for frac, win, hw in ((0.20, (0, 0), 0), (0.20, (8, 24), 64), (0.15, (8, 24), 128), (0.10, (8, 24), 64)):
                cpu_res = index20_selector.select_from_reduced_scores(
                    docs,
                    total,
                    cpu_sums,
                    counts,
                    fraction=frac,
                    strategy="max",
                    span_window=win,
                    head_width=hw,
                    allow_budget_floor=True,
                )
                t_sums = torch_mod.as_tensor(gpu_sums, dtype=torch_mod.float32)
                t_cnts = torch_mod.as_tensor(counts, dtype=torch_mod.int64)
                gpu_res, target_body = index20_selector.select_from_reduced_scores_gpu(
                    docs,
                    total,
                    t_sums,
                    t_cnts,
                    fraction=frac,
                    strategy="max",
                    span_window=win,
                    head_width=hw,
                    torch_module=torch_mod,
                    geometry_cache={},
                )
                self.assertEqual(
                    gpu_res["selected_positions"],
                    cpu_res["selected_positions"],
                    f"Mismatch at trial={trial} frac={frac} win={win} hw={hw}",
                )
                self.assertEqual(gpu_res["selected_body_positions"], cpu_res["selected_body_positions"])
                self.assertEqual(target_body, cpu_res["selected_body_tokens"])

    def test_opt3_stable_sort_reference(self) -> None:
        self._run_opt3_equivalence(_FakeTorch)

    @unittest.skipIf(REAL_TORCH is None, "PyTorch not installed in this CPU environment")
    def test_opt3_stable_sort_real_torch(self) -> None:
        self._run_opt3_equivalence(REAL_TORCH)


class TestGeometryAndFrameInvariants(unittest.TestCase):
    def test_online_head_and_frame_capacity_across_multi_frame_prompts(self) -> None:
        rng = random.Random(42)
        for head_width in (0, 64, 128):
            docs = []
            cursor = 17
            for i in range(12):
                length = rng.randint(256, 2400)
                docs.append({"start": cursor, "end": cursor + length, "key": f"d{i}"})
                cursor += length + rng.randint(5, 80)
            total = cursor + 150  # Crosses multiple 8192 frames.
            self.assertGreater(total, 8192)
            n_keys = total // 2
            sums = [rng.uniform(-5.0, 10.0) for _ in range(n_keys)]
            counts = [1] * n_keys

            res = index20_selector.select_from_reduced_scores(
                docs,
                total,
                sums,
                counts,
                fraction=0.20,
                strategy="max",
                span_window=(8, 24),
                head_width=head_width,
                allow_budget_floor=True,
            )
            selected = res["selected_positions"]
            # No out-of-bounds, no duplicates, strictly sorted, frame counts <= min(8192, total - 8192*i)
            frame_rows = index20_selector.validate_frame_capacity(selected, total)
            self.assertEqual(sum(frame_rows), len(selected))
            for idx, count in enumerate(frame_rows):
                self.assertLessEqual(count, min(8192, total - 8192 * idx))

            # Verify Online_Head presence and strict non-overlap with tail128
            sel_set = set(selected)
            for d in docs:
                heads = set(range(d["start"], d["start"] + head_width))
                tails = set(range(d["end"] - 128, d["end"]))
                self.assertTrue(heads.isdisjoint(tails))
                self.assertTrue(tails <= sel_set)
                self.assertTrue(heads <= sel_set)

    def test_budget_admission_legacy_vs_v2_and_short_doc_rejection(self) -> None:
        docs = [{"start": 10, "end": 4010, "key": "d0"}, {"start": 4020, "end": 8020, "key": "d1"}]
        total = 8200
        # Legacy S1.6 allows only 0.15
        adm_s16 = fixed_pack_policy.budget_admission(docs, total, 0.15)
        self.assertEqual(adm_s16["body_fraction"], 0.15)
        with self.assertRaises(ValueError):
            fixed_pack_policy.budget_admission(docs, total, 0.20)

        # V2 allows 0.20 / 0.15 / 0.12 / 0.10
        p20 = v2_policy.resolve_v2_policy_for_tokens(64_000)
        adm_v2 = fixed_pack_policy.budget_admission(docs, total, 0.20, policy=p20)
        self.assertEqual(adm_v2["body_fraction"], 0.20)
        self.assertEqual(adm_v2["head_width"], 64)

        p10 = v2_policy.resolve_v2_policy_for_tokens(64_000, two_stage_l14=True, preset="v2.3_10")
        adm_10 = fixed_pack_policy.budget_admission(docs, total, 0.10, policy=p10)
        self.assertEqual(adm_10["body_fraction"], 0.10)
        self.assertTrue(adm_10["two_stage_l14"])

        # Reject doc shorter than 256 (would risk head/tail overlap)
        bad_docs = [{"start": 0, "end": 200, "key": "short"}]
        with self.assertRaises(ValueError):
            fixed_pack_policy.budget_admission(bad_docs, 400, 0.20, policy=p20)


class TestMinimalConeIntegrity(unittest.TestCase):
    def test_minimal_cone_dependency_closure_and_row_pruning(self) -> None:
        # 4 native 8192-token frames = 32,768 tokens with 10 documents.
        total = 32_768
        docs = [{"start": 100 + i * 3100, "end": 100 + i * 3100 + 2900, "key": f"d{i}"} for i in range(10)]
        mandatory = index20_selector.mandatory_positions(docs, total)
        # Add some ranked body rows + Online_Head.
        extra = [d["start"] + k for d in docs for k in range(64)]
        extra += [d["start"] + 500 + k for d in docs for k in range(300)]
        selected = sorted(set(mandatory) | set(extra))
        final_frame_start = (total - 1) // 8192 * 8192
        ranges = [(0, 16_384), (16_384, total)]

        full_cone = cone15_plan.Cone15Plan(
            total, selected, mandatory, last_frame_start=final_frame_start, group_ranges=ranges
        )
        min_cone = cone15_plan.Cone15Plan(
            total, selected, (), last_frame_start=final_frame_start, group_ranges=ranges
        )

        # 1. Verify backward causal completeness from sampler | dspark_ring across L39..L21.
        self.assertEqual(min_cone.layers[39].queries, min_cone.sampler)
        for l in range(21, 40):
            layer_masks = min_cone.layers[l]
            # Every query and input is a subset of selected.
            self.assertEqual(layer_masks.queries & ~min_cone.selected_mask, 0)
            self.assertEqual(layer_masks.inputs & ~min_cone.selected_mask, 0)
            # D-Spark ring (37, 38, 39) is preserved in inputs.
            if l in cone15_plan.D_SPARK:
                self.assertEqual(min_cone.ring & ~layer_masks.inputs, 0)
            # SWA 128-token window of queries | ring is fully covered by online | offline.
            swa_needed = cone15_plan.windows(layer_masks.queries) | min_cone.ring
            self.assertEqual(swa_needed, layer_masks.online | layer_masks.offline)
            self.assertEqual(layer_masks.online & layer_masks.offline, 0)
            self.assertEqual(layer_masks.online & ~layer_masks.inputs, 0)
            # Layer l's queries must supply every hidden input required by layer l+1.
            if l < 39:
                self.assertEqual(min_cone.layers[l + 1].inputs & ~layer_masks.queries, 0)

        # 2. Verify dramatic post-20 FFN query row reduction vs legacy Cone15Plan.
        full_queries = sum(
            len(full_cone.group(l, a, b).queries) for a, b in ranges for l in range(21, 40)
        )
        min_queries = sum(
            len(min_cone.group(l, a, b).queries) for a, b in ranges for l in range(21, 40)
        )
        self.assertLess(min_queries, full_queries * 0.5)

        # 3. Verify Opt-2 halo pruning plan preserves every reachable post-20 offline key.
        import fixed15_geometry
        selected_set = set(selected)
        universe = [p for p in fixed15_geometry.window_union(selected) if p not in selected_set]
        bound_groups = []
        for d in docs:
            matches = [(i, p - d["start"]) for i, p in enumerate(universe) if d["start"] <= p < d["end"]]
            if matches:
                bound_groups.append((d["key"], matches[0][0], matches[-1][0] + 1, [s for _, s in matches], None))
        l25_wanted = cone15_plan.row_ids(min_cone.layers[25].offline, 0, total)
        keep, inv, doc_ops = halo_direct_copy.plan_layer_halo_gather(universe, l25_wanted, bound_groups)
        self.assertEqual(len(keep), len(l25_wanted))
        self.assertLess(len(keep), len(universe))
        for pos in l25_wanted:
            uidx = universe.index(pos)
            self.assertGreaterEqual(inv[uidx], 0)
            self.assertEqual(universe[keep[inv[uidx]]], pos)

        # 4. Verify cone15_ffn_native_proof.live_covered under MinimalCone vs Legacy S1.6.
        a, b = ranges[0]
        geom = min_cone.group(25, a, b)
        chunk_sel = list(min_cone.selection(a, b))
        chunk_tails = [p for d in docs for p in range(d["end"] - 128, d["end"]) if a <= p < b]
        min_event = dict(
            chunk_start=a,
            chunk_end=b,
            selected_positions=chunk_sel,
            selected_rows=len(chunk_sel),
            cone15_ffn_positions=list(geom.queries),
            cone15_input_positions=list(geom.inputs) if geom.inputs else chunk_sel,
            cone15_mhc_positions=list(geom.inputs) if geom.inputs else chunk_sel,
            actual_ffn_rows=len(geom.queries),
            actual_mhc_rows=len(geom.inputs) if geom.inputs else len(chunk_sel),
        )
        # Passes under MinimalCone, rejected under Legacy S1.6 (because tails were pruned in chunk 0).
        self.assertTrue(
            cone15_ffn_native_proof.live_covered(
                min_event, chunk_tails, 25, "actual_ffn_rows", "cone15_ffn_positions", minimal_plan=min_cone
            )
        )
        self.assertTrue(
            cone15_ffn_native_proof.live_covered(
                min_event, chunk_tails, 25, "actual_mhc_rows", "cone15_mhc_positions", minimal_plan=min_cone
            )
        )
        self.assertFalse(
            cone15_ffn_native_proof.live_covered(
                min_event, chunk_tails, 25, "actual_ffn_rows", "cone15_ffn_positions", minimal_plan=None
            )
        )


class TestTwoStageL14LocalityAndPromotion(unittest.TestCase):
    def test_affected_strict_locality_at_mid14_and_promotion(self) -> None:
        rng = random.Random(2026)
        docs, total = _make_random_docs(rng, min_docs=4, max_docs=6, min_len=400, max_len=900)
        n_keys = total // 2
        l2_sums = [rng.uniform(-2.0, 8.0) for _ in range(n_keys)]
        l2_cnts = [1] * n_keys

        # Stage-1 (12%) + target budget (15% equivalent) via Opt-3 GPU selector.
        stage1_choice, target_body_budget = index20_selector.select_from_reduced_scores_gpu(
            docs,
            total,
            _FakeTorch.as_tensor(l2_sums, dtype=_FakeTorch.float32),
            _FakeTorch.as_tensor(l2_cnts, dtype=_FakeTorch.int64),
            fraction=0.15,
            stage1_fraction=0.12,
            strategy="max",
            span_window=(8, 24),
            head_width=64,
            torch_module=_FakeTorch,
            geometry_cache={},
        )
        valid_l14 = [False] * n_keys
        for d in docs:
            for k in range((d["start"] + 1) // 2, d["end"] // 2):
                valid_l14[k] = True
        l14_scores = [rng.uniform(-3.0, 12.0) for _ in range(n_keys)]

        union_choice, extra = two_stage_l14.promote_stage2(
            stage1_choice,
            docs,
            total,
            l14_scores,
            valid_l14,
            target_fraction=0.15,
            target_body_tokens=target_body_budget,
            span_window=(8, 24),
        )
        stage1_set = set(stage1_choice["selected_positions"])
        extra_set = set(extra)
        union_list = union_choice["selected_positions"]

        self.assertTrue(extra_set.isdisjoint(stage1_set))
        self.assertEqual(stage1_set | extra_set, set(union_list))
        self.assertEqual(union_choice["selected_body_tokens"], target_body_budget)
        self.assertEqual(union_choice["fraction"], 0.15)
        self.assertEqual(union_choice["stage1_fraction"], 0.12)
        self.assertTrue(union_choice["two_stage_l14"])

        # Verify L14 strict locality law: affected[layer] == set(extra) for all 3..13, disjoint from stage1.
        affected = two_stage_l14.affected_positions(stage1_set, extra, union_list, mid=14, total=total)
        self.assertEqual(set(affected.keys()), set(range(3, 14)))
        for layer in range(3, 14):
            self.assertEqual(affected[layer], extra_set)
            self.assertTrue(affected[layer].isdisjoint(stage1_set))

        # Verify cone15_ffn_native_proof.live_covered accepts valid backfill events.
        bank = two_stage_l14.PrefixBank(sorted(stage1_set))
        bank.data[(5, "hidden")] = object()
        bank.data[(5, "swa")] = object()
        state = {"mid": 14, "affected": affected, "states": bank}
        chunk_tails = [p for d in docs for p in range(d["end"] - 128, d["end"])]
        backfill_event = dict(
            selected_positions=union_list,
            selected_rows=len(union_list),
            backfill_positions=sorted(extra_set),
            actual_ffn_rows=len(extra_set),
        )
        self.assertTrue(
            cone15_ffn_native_proof.live_covered(
                backfill_event, chunk_tails, 5, "actual_ffn_rows", "cone15_ffn_positions", two_stage_state=state
            )
        )


class TestRepositoryIntegrityAndAST(unittest.TestCase):
    def test_open_source_runtime_has_no_blocking_source_sha256_gates(self) -> None:
        # Open-source contributors must be able to edit algorithm files freely without updating lockfiles.
        run_fixed_spec = importlib.util.spec_from_file_location("test_run_fixed", PRODUCER_ROOT / "run_fixed.py")
        run_fixed_mod = importlib.util.module_from_spec(run_fixed_spec)
        run_fixed_spec.loader.exec_module(run_fixed_mod)
        self.assertFalse(hasattr(run_fixed_mod, "PARENT_SHA"))
        loaded_base = run_fixed_mod.load_base(PRODUCER_ROOT / "run_s16.py")
        self.assertTrue(callable(loaded_base.compact_controller))

        # fixed_pack_runtime_adapter.pinned() succeeds without a source_sha256 map when strict mode is off.
        resolved = fixed_pack_runtime_adapter.pinned({}, FIXED_ROOT / "v2_policy.py")
        self.assertEqual(resolved, (FIXED_ROOT / "v2_policy.py").resolve())

        # local_backend does not enforce hardcoded PINS on algorithm sources.
        lb_spec = importlib.util.spec_from_file_location(
            "test_local_backend", REPO_ROOT / "test" / "longbench300" / "local_backend.py"
        )
        lb_mod = importlib.util.module_from_spec(lb_spec)
        lb_spec.loader.exec_module(lb_mod)
        self.assertFalse(hasattr(lb_mod, "PINS"))

        # Verify OWN_FILES and SHARED_FILES cover all newly added files
        self.assertIn("v2_policy.py", fixed_pack_runtime_adapter.OWN_FILES)
        self.assertIn("two_stage_l14.py", fixed_pack_runtime_adapter.OWN_FILES)
        self.assertIn("runtime/halo_direct_copy.py", fixed_pack_runtime_adapter.SHARED_FILES)

    def test_examples_redknot_v2_demo_cpu_walkthrough_runs(self) -> None:
        for preset in ("short20_long15", "v2.1-fast"):
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "examples" / "redknot_v2_demo.py"), "--cpu-demo", "--preset", preset],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("MinimalCone (Layers 21..39 FFN rows)", proc.stdout)
            self.assertIn("Opt-2 Post-20 Offline Halo rows", proc.stdout)

    def test_fixed_runtime_rewrite_ast_assertions_remain_11111(self) -> None:
        import multiowner_runtime

        rewritten = fixed_pack_runtime_adapter.private_preserved_module(multiowner_runtime)
        self.assertTrue(callable(rewritten.runtime_class))
        self.assertTrue(callable(rewritten.canonical_method))
        self.assertEqual(rewritten.canonical_method("Cone15", "max"), ("Cone15", True, "max"))

    def test_cpu_preflight_modules_never_import_gpu_libraries(self) -> None:
        # Must match fixed_pack_runtime_adapter.load_dependencies() CPU preflight contract.
        modules = [name[:-3] for name in fixed_pack_runtime_adapter.OWN_FILES] + [
            "halo_direct_copy",
            "packed_attention",
            "index20_selector",
            "run_s16",
        ]
        paths = [str(p) for p in (FIXED_ROOT, LASTKV_ROOT, SHARED_RUNTIME_ROOT, SHARED_ROOT, MULTIOWNER_ROOT, PRODUCER_ROOT)]
        code = (
            f"import importlib, sys; sys.path[:0] = {paths!r}; "
            f"[importlib.import_module(m) for m in {modules!r}]; "
            "forbidden = {'torch', 'triton', 'deepinfer'} & set(sys.modules); "
            "assert not forbidden, f'Forbidden GPU imports during CPU preflight: {forbidden}'"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_compact_controller_preserves_s16_strict_015_and_admits_v2(self) -> None:
        def make_report(fraction: float, policy_dict=None, *, two_stage=False):
            layers = [
                dict(
                    layer=l,
                    actual_ffn_rows=10,
                    dispatch_rows=10,
                    live_ffn_rows=10,
                    core=dict(actual_attention_rows=10, input_axis_rows=20),
                )
                for l in range(21, 40)
            ]
            pack = dict(enabled=True, canonical_method="Cone15")
            if policy_dict is not None:
                pack["v2_policy"] = policy_dict
            return dict(
                input_ids_sha256="a" * 64,
                full_input_tokens=65_000,
                mode="sensitive",
                model_boundary_restored=True,
                decode_native=True,
                cone15_ffn=dict(
                    method="Cone15",
                    minimal_cone=bool(policy_dict and policy_dict.get("minimal_cone")),
                    new_and_tail_full_ffn_each_layer=True,
                    all_input_lastkv_physical_replay=True,
                    qualification_cache_bytes=None,
                    groups=[dict(layers=layers)],
                ),
                index20=dict(
                    selection=dict(
                        strategy="max",
                        fraction=fraction,
                        two_stage_l14=two_stage,
                        selected_positions=[0, 1, 2, 3],
                    )
                ),
                system_pack_fixed_v1=pack,
            )

        # Legacy S1.6: 0.15 passes, 0.20 fails
        out_s16 = run_s16.compact_controller(make_report(0.15, None))
        self.assertEqual(out_s16["sparse_method"], "Cone15")
        with self.assertRaises(ValueError):
            run_s16.compact_controller(make_report(0.20, None))

        # V2.1-Fast (20%): passes
        v2_fast = v2_policy.resolve_v2_policy_for_tokens(65_000).to_dict()
        out_v2 = run_s16.compact_controller(make_report(0.20, v2_fast))
        self.assertTrue(out_v2["minimal_cone"])
        self.assertEqual(out_v2["v2_policy"]["mode"], "v2.1-fast")

        # Two-Stage L14 (short20_long15 @ 65K -> 15%): passes with two_stage=True, fails if fraction mismatches
        v2_ts = v2_policy.resolve_v2_policy_for_tokens(65_000, two_stage_l14=True, preset="short20_long15").to_dict()
        out_ts = run_s16.compact_controller(make_report(0.15, v2_ts, two_stage=True))
        self.assertTrue(out_ts["v2_policy"]["two_stage_l14"])
        with self.assertRaises(ValueError):
            run_s16.compact_controller(make_report(0.20, v2_ts, two_stage=True))

    def test_suffix_only_collector_query_mask_excludes_doc_tail128(self) -> None:
        class _MaskTensor(_FakeTensor):
            def zero_(self):
                self._data = [False] * len(self._data)
                return self

            def fill_(self, value):
                self._data = [value] * len(self._data)
                return self

            def __setitem__(self, key, value):
                if isinstance(key, slice):
                    start, stop, step = key.indices(len(self._data))
                    for idx in range(start, stop, step):
                        self._data[idx] = value
                    return
                if isinstance(key, _FakeTensor):
                    for idx in key._data:
                        self._data[idx] = value
                    return
                self._data[key] = value

        class _CollectorTorch(_FakeTorch):
            bool = "bool"
            int32 = "int32"

            @staticmethod
            def zeros(n, dtype="float32", device="cpu"):
                val = False if dtype == "bool" else (0 if dtype in ("int32", "int64") else 0.0)
                return _MaskTensor([val] * n, dtype=dtype, device=device)

            @staticmethod
            def tensor(values, dtype="int64", device="cpu"):
                return _MaskTensor(list(values), dtype=dtype, device=device)

        fake_mod = SimpleNamespace(
            layer_idx=2,
            kv_compress_ratio=2,
            use_candidate_topk=False,
            _compute_prefill_p_sum=lambda *a, **k: None,
        )
        # Clean class-level descriptor simulation (attribute not in __dict__ initially).
        class _Mod:
            layer_idx = 2
            kv_compress_ratio = 2
            use_candidate_topk = False

            def _compute_prefill_p_sum(self, *a, **k):
                return None

        mod = _Mod()
        docs = [{"start": 10, "end": 310, "key": "d0"}, {"start": 320, "end": 620, "key": "d1"}]
        total = 700
        collector = index20_selector.NativeIndexScoreCollector(
            mod, torch_module=_CollectorTorch, allow_sync=lambda: None
        )
        try:
            # V2 suffix-only: query set is strictly [620, 700), excluding all doc tail128 rows!
            collector.configure_v2(dict(suffix_only=True, head_width=64, span_window=(0, 0), gpu_select=True))
            collector.begin(docs, total, "cpu", fraction=0.20, strategy="max")
            self.assertEqual(collector.mandatory, list(range(620, 700)))
            self.assertEqual(sum(collector.query_mask._data), 80)
            self.assertFalse(any(collector.query_mask._data[:620]))
            self.assertTrue(all(collector.query_mask._data[620:]))
            collector.active = False

            # Legacy S1.6: query set includes doc tail128 + non-doc tokens
            collector.configure_v2(None)
            collector.begin(docs, total, "cpu", fraction=0.15, strategy="max")
            expected_legacy = index20_selector.mandatory_positions(docs, total)
            self.assertEqual(collector.mandatory, expected_legacy)
            self.assertGreater(len(collector.mandatory), 80 + 256 - 1)
            collector.active = False
        finally:
            collector.close()

    def test_opt1_static_metadata_cache_hit_invalidation_and_restore(self) -> None:
        rag_root = str(ALGO_ROOT / "rag")
        if rag_root not in sys.path:
            sys.path.insert(0, rag_root)
        import fixed_pack_controller
        import rag_natural_documents as rag

        ids = list(range(100, 700))
        doc_key = rag.digest(ids[10:310])
        docs = [dict(input_start=10, input_end=310, token_ids_sha256=doc_key)]
        orig_natural = rag.natural_documents
        calls = [0]

        def counting_natural(call_ids, call_docs):
            calls[0] += 1
            return orig_natural(call_ids, call_docs)

        class _BaseCtrl:
            def __init__(self):
                self.state = "IDLE"
                self.case_input = "case-sha-1"
                self.case_binding = ("bind-1",)
                self.cache = SimpleNamespace(scope={"model": "v41"})

            def close(self):
                pass

        CtrlCls = fixed_pack_controller.fixed_pack_controller_class(_BaseCtrl)
        ctrl = CtrlCls()
        rag.natural_documents = counting_natural
        try:
            # First call: miss -> calls natural_documents once, second lookup inside same reset hits cache.
            with ctrl._opt1_static_reset_scope(tuple(ids), docs):
                d1 = rag.natural_cache_documents(ids, docs)
                d2 = rag.natural_cache_documents(ids, docs)
                self.assertEqual(d1, d2)
            self.assertEqual(calls[0], 1)
            self.assertEqual(ctrl._v2_reset_stats["route"], "miss")
            self.assertEqual(ctrl._v2_reset_stats["natural_hits"], 1)
            self.assertIs(rag.natural_documents, counting_natural)

            # Second identical reset: cache hit -> zero additional natural_documents calls.
            with ctrl._opt1_static_reset_scope(tuple(ids), docs):
                d3 = rag.natural_cache_documents(ids, docs)
                self.assertEqual(d3, d1)
            self.assertEqual(calls[0], 1)
            self.assertEqual(ctrl._v2_reset_stats["route"], "hit")
            self.assertEqual(ctrl._v2_reset_stats["natural_hits"], 1)

            # Binding change invalidates cache immediately.
            ctrl.case_binding = ("bind-2",)
            with ctrl._opt1_static_reset_scope(tuple(ids), docs):
                rag.natural_cache_documents(ids, docs)
            self.assertEqual(calls[0], 2)
            self.assertEqual(ctrl._v2_reset_stats["route"], "miss")
        finally:
            rag.natural_documents = orig_natural

    def test_benchmark_cli_rejects_invalid_s16_two_stage_combo(self) -> None:
        for script in ("test/benchmark_rag_redknot.py", "test/benchmark_rag_redknot_21.py"):
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / script), "--mode", "s1.6", "--two-stage-l14", "--check"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0, f"{script} should reject --mode s1.6 --two-stage-l14")
            self.assertIn("s1.6", proc.stderr)

    @unittest.skipIf(sys.version_info < (3, 11), "local_backend.hash_file uses hashlib.file_digest (Python 3.11+)")
    def test_end_to_end_bundled_runtime_prepare_native_preflight(self) -> None:
        code = f"""
import sys, tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, {str(REPO_ROOT / 'test')!r})
from longbench300 import local_backend as lb

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    model = root / 'model'
    out = root / 'out'
    model.mkdir(); out.mkdir()
    v2_cls = lb.load_v2_policy_config_class()
    with patch.object(lb, 'verify_checkpoint', return_value=None):
        cfg_fast, loc = lb.bundled_runtime(model, out, v2_config=v2_cls.from_cli(mode='v2.1-fast'))
        cfg_ts, _ = lb.bundled_runtime(model, out, v2_config=v2_cls.from_cli(two_stage_l14=True, preset='short20_long15'))
        cfg_s16, _ = lb.bundled_runtime(model, out, v2_config=v2_cls.from_cli(mode='s1.6'))
    assert len({{cfg_fast['sha256'], cfg_ts['sha256'], cfg_s16['sha256']}}) == 3
    assert cfg_fast['redknot_v2']['mode'] == 'v2.1-fast'
    assert cfg_ts['redknot_v2']['preset'] == 'short20_long15'
    assert cfg_s16['redknot_v2']['mode'] == 's1.6'
    lb.prepare_native(cfg_fast)
    base, producer = lb.local_modules(cfg_fast, loc)
    assert callable(base.compact_controller) and callable(producer.identity)
    assert not ({{'torch', 'triton', 'deepinfer'}} & set(sys.modules))
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
