"""Offline CPU unit tests for RedKnot V2 (`two-stage` and `single-stage`).

Covers policy routing, GPU vs CPU Span equivalence, Opt-3 stable sort, geometry
and 8192-frame capacity, MinimalCone dependency closure + Opt-2 halo pruning +
native proof audit, L2->L14 promotion and strict locality, suffix-only query
masking, Opt-1 metadata cache, FixedRuntimeRewrite AST assertions, zero-GPU
preflight, open-source non-blocking source hashing, and CLI entrypoints.
"""
from __future__ import annotations

import importlib.util
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

for _p in (FIXED_ROOT, LASTKV_ROOT, SHARED_RUNTIME_ROOT, SHARED_ROOT, MULTIOWNER_ROOT, PRODUCER_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cone15_ffn_native_proof  # noqa: E402
import cone15_plan  # noqa: E402
import fixed15_geometry  # noqa: E402
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
    """Minimal 1D/3D CPU tensor used only when PyTorch is absent."""

    def __init__(self, data, *, dtype="float32", device="cpu", shape=None):
        self._data = list(data._data) if isinstance(data, _FakeTensor) else list(data)
        self.dtype, self.device = dtype, device
        self.shape = (len(self._data),) if shape is None else tuple(shape)

    def clone(self): return _FakeTensor(self._data, dtype=self.dtype, device=self.device, shape=self.shape)
    def cpu(self): return self
    def tolist(self): return list(self._data)
    def zero_(self): self._data = [False] * len(self._data); return self
    def fill_(self, v): self._data = [v] * len(self._data); return self
    def __len__(self): return self.shape[0]
    def __gt__(self, o): return _FakeTensor([x > o for x in self._data], dtype="bool", device=self.device, shape=self.shape)
    def __floordiv__(self, o): return _FakeTensor([x // o for x in self._data], dtype=self.dtype, device=self.device, shape=self.shape)

    def view(self, *dims):
        known = [d for d in dims if d != -1]
        prod = 1
        for d in known:
            prod *= d
        resolved = tuple(len(self._data) // prod if d == -1 else d for d in dims)
        return _FakeTensor(self._data, dtype=self.dtype, device=self.device, shape=resolved)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _FakeTensor(self._data[item], dtype=self.dtype, device=self.device)
        if isinstance(item, _FakeTensor):
            return _FakeTensor([self._data[i] for i in item._data], dtype=self.dtype, device=self.device)
        return self._data[item]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            if isinstance(value, bool):
                for i in range(*key.indices(len(self._data))):
                    self._data[i] = value
            else:
                self._data[key] = value._data if isinstance(value, _FakeTensor) else list(value)
        elif isinstance(key, _FakeTensor):
            for i in key._data:
                self._data[i] = value
        else:
            self._data[key] = value


class _FakeTorch:
    bool = "bool"
    int32 = "int32"
    int64 = long = "int64"
    float32 = "float32"

    class _F:
        @staticmethod
        def pad(t, pad, value=0.0):
            l, r = pad
            out = [value] * l + list(t._data) + [value] * r
            shape = (1, 1, t.shape[2] + l + r) if len(t.shape) == 3 else (len(out),)
            return _FakeTensor(out, dtype=t.dtype, device=t.device, shape=shape)

        @staticmethod
        def max_pool1d(t, k, stride=1):
            assert stride == 1 and t.shape[:2] == (1, 1)
            d = t._data
            out = [max(d[i : i + k]) for i in range(len(d) - k + 1)]
            return _FakeTensor(out, dtype=t.dtype, device=t.device, shape=(1, 1, len(out)))

    nn = SimpleNamespace(functional=_F)

    @staticmethod
    def as_tensor(values, device="cpu", dtype="int64"):
        return _FakeTensor(values, dtype=dtype, device=device)

    tensor = as_tensor

    @staticmethod
    def zeros(n, dtype="float32", device="cpu"):
        fill = False if dtype == "bool" else (0 if dtype in ("int32", "int64") else 0.0)
        return _FakeTensor([fill] * n, dtype=dtype, device=device)

    @staticmethod
    def where(cond, x, y):
        xv = x._data if isinstance(x, _FakeTensor) else [x] * len(cond._data)
        yv = y._data if isinstance(y, _FakeTensor) else [y] * len(cond._data)
        return _FakeTensor([a if c else b for c, a, b in zip(cond._data, xv, yv)], dtype="float32", device=cond.device)

    @staticmethod
    def argsort(t, descending=False, stable=False):
        assert stable is True
        idx = list(range(len(t._data)))
        idx.sort(key=(lambda i: -t._data[i]) if descending else (lambda i: t._data[i]))
        return _FakeTensor(idx, dtype="int64", device=t.device)


def _torch_mod():
    return REAL_TORCH if REAL_TORCH is not None else _FakeTorch


def _make_random_docs(rng: random.Random, *, min_docs=2, max_docs=6, min_len=260, max_len=900):
    docs, cursor = [], rng.randint(0, 25)
    for idx in range(rng.randint(min_docs, max_docs)):
        length = rng.randint(min_len, max_len)
        docs.append({"start": cursor, "end": cursor + length, "key": f"doc-{idx}"})
        cursor += length + rng.randint(0, 31)
    return docs, cursor + rng.randint(17, 140)


class TestRedKnotV2Core(unittest.TestCase):
    def test_policy_routing_keep_pct_two_stage_single_stage_and_legacy(self) -> None:
        cfg_default = v2_policy.V2PolicyConfig.default()
        cfg_single = v2_policy.V2PolicyConfig.from_cli(keep_pct="20+0")
        cfg_shorthand = v2_policy.V2PolicyConfig.from_cli(keep_pct="20")
        cfg_12p3 = v2_policy.V2PolicyConfig.from_cli(keep_pct="12+3")
        self.assertEqual(cfg_default.keep_pct, "9+3")
        self.assertTrue(cfg_default.two_stage_l14 and cfg_12p3.two_stage_l14)
        self.assertEqual(cfg_single, cfg_shorthand)
        self.assertFalse(cfg_single.two_stage_l14)

        # Default 9+3: <48K -> single-stage 20% NoSpan Head64; >=48K -> 9%+3%=12% Span Head64/128; >=192K -> offload
        expected_9p3 = [
            (15_800, "16k", "single-stage", 0.20, 0.20, 64, (0, 0), False, False),
            (47_999, "32k", "single-stage", 0.20, 0.20, 64, (0, 0), False, False),
            (48_000, "64k", "two-stage", 0.09, 0.12, 64, (8, 24), True, False),
            (96_000, "128k", "two-stage", 0.09, 0.12, 128, (8, 24), True, False),
            (191_999, "128k", "two-stage", 0.09, 0.12, 128, (8, 24), True, False),
            (192_000, "256k", "two-stage", 0.09, 0.12, 128, (8, 24), True, True),
        ]
        for toks, band, mode, s1, frac, hw, win, ts, offload in expected_9p3:
            p = cfg_default.resolve(toks)
            self.assertEqual((p.band, p.mode, p.stage1_fraction, p.fraction, p.head_width, p.window, p.two_stage_l14, p.host_prefix_offload),
                             (band, mode, s1, frac, hw, win, ts, offload))
            self.assertTrue(p.is_v2 and p.minimal_cone and p.suffix_only and p.gpu_select and p.static_metadata_cache and p.direct_halo_copy)

        # Custom 12+3 works with zero code edits (<48K stays 20% single-stage; >=48K -> 0.12 / 0.15)
        self.assertEqual((cfg_12p3.resolve(32_000).stage1_fraction, cfg_12p3.resolve(32_000).fraction), (0.20, 0.20))
        p_12p3 = cfg_12p3.resolve(128_000)
        self.assertEqual((p_12p3.stage1_fraction, p_12p3.fraction, p_12p3.two_stage_l14), (0.12, 0.15, True))

        # Single-stage 20+0 (and custom 15+0): stage2==0 -> single-stage everywhere, 0 extra cache (never offload)
        for toks, hw, win in ((32_000, 64, (0, 0)), (64_000, 64, (8, 24)), (128_000, 128, (8, 24)), (250_000, 128, (8, 24))):
            p = cfg_single.resolve(toks)
            self.assertEqual((p.mode, p.stage1_fraction, p.fraction, p.head_width, p.window), ("single-stage", 0.20, 0.20, hw, win))
            self.assertFalse(p.two_stage_l14 or p.host_prefix_offload)
        p_15 = v2_policy.V2PolicyConfig(keep_pct="15+0").resolve(128_000)
        self.assertEqual((p_15.mode, p_15.stage1_fraction, p_15.fraction, p_15.two_stage_l14), ("single-stage", 0.15, 0.15, False))

        # Explicit host_prefix_offload overrides and configurable span_min_tokens
        self.assertTrue(v2_policy.V2PolicyConfig.from_cli(keep_pct="9+3", host_prefix_offload="on").resolve(128_000).host_prefix_offload)
        self.assertFalse(v2_policy.V2PolicyConfig.from_cli(keep_pct="9+3", host_prefix_offload="off").resolve(250_000).host_prefix_offload)
        self.assertEqual(v2_policy.V2PolicyConfig(keep_pct="20+0", span_min_tokens=96_000).resolve(65_000).window, (0, 0))
        self.assertEqual(v2_policy.V2PolicyConfig(keep_pct="20+0", span_min_tokens=96_000).resolve(96_000).window, (8, 24))

        # Internal legacy S1.6 compatibility & invalid keep_pct rejection
        p_s16 = v2_policy.V2PolicyConfig.legacy_s16().resolve(128_000)
        self.assertTrue(p_s16.is_legacy_s16 and not p_s16.is_v2 and p_s16.fraction == 0.15 and p_s16.head_width == 0)
        self.assertFalse(p_s16.minimal_cone or p_s16.suffix_only or p_s16.gpu_select or p_s16.two_stage_l14)
        for bad in ("", "0+3", "-5+0", "80+30", "9+3+1", "unknown"):
            with self.assertRaises(ValueError):
                v2_policy.V2PolicyConfig(keep_pct=bad)

    def test_span_gpu_vs_cpu_equivalence_and_doc_isolation(self) -> None:
        torch_mod = _torch_mod()
        # 1. Explicit asymmetric [-8, +24] direction & zero cross-doc leakage
        docs = [{"start": 11, "end": 271, "key": "d0"}, {"start": 273, "end": 550, "key": "d1"}]
        values, counts = [-5.0] * 300, [1] * 300
        values[20], values[130], counts[138], values[138] = 9.5, 100.0, 0, 500.0
        v_cpu, c_cpu = index20_selector.span_values(docs, values, counts, 8, 24)
        v_gpu = index20_selector.span_values_gpu(
            docs, torch_mod.as_tensor(values, dtype=torch_mod.float32),
            torch_mod.as_tensor(counts, dtype=torch_mod.int64), 8, 24, torch_module=torch_mod,
        ).cpu().tolist()
        for arr in (v_cpu, v_gpu):
            self.assertEqual((arr[11], arr[12], arr[44], arr[45]), (-5.0, 9.5, 9.5, -5.0))
            self.assertEqual((arr[137], arr[138]), (-5.0, -5.0))  # No leakage from k=130 into Doc 1
        self.assertGreater(c_cpu[138], 0)

        # 2. Randomized equivalence with negative scores, odd boundaries, and unobserved docs
        rng = random.Random(20261124)
        for trial in range(25):
            r_docs, total = _make_random_docs(rng, min_docs=2, max_docs=5, min_len=256, max_len=600)
            n_keys = total // 2
            vals = [rng.choice([-12.5, -3.0, -0.25, 0.0, 1.5, 4.25, 9.75]) for _ in range(n_keys)]
            cnts = [0 if rng.random() < 0.25 else rng.randint(1, 5) for _ in range(n_keys)]
            if trial % 5 == 0:
                for k in range((r_docs[0]["start"] + 1) // 2, r_docs[0]["end"] // 2):
                    cnts[k], vals[k] = 0, 999.0
            for win in ((8, 24), (4, 12)):
                vc, cc = index20_selector.span_values(r_docs, vals, cnts, win[0], win[1])
                vg = index20_selector.span_values_gpu(
                    r_docs, torch_mod.as_tensor(vals, dtype=torch_mod.float32),
                    torch_mod.as_tensor(cnts, dtype=torch_mod.int64), win[0], win[1], torch_module=torch_mod,
                ).cpu().tolist()
                for k in range(n_keys):
                    if cc[k] == 0:
                        self.assertEqual(vg[k], -math.inf)
                    else:
                        self.assertAlmostEqual(vg[k], vc[k], places=5)

    def test_opt3_gpu_stable_sort_matches_cpu_reference(self) -> None:
        torch_mod = _torch_mod()
        rng = random.Random(1337)
        for _ in range(20):
            docs, total = _make_random_docs(rng, min_docs=2, max_docs=5, min_len=256, max_len=700)
            n_keys = total // 2
            sums = [rng.choice([-10.0, -1.0, 0.0, 0.0, 2.0, 2.0, 5.5]) for _ in range(n_keys)]
            counts = [0 if rng.random() < 0.30 else rng.randint(1, 4) for _ in range(n_keys)]
            cpu_sums = [s if c > 0 else 0.0 for s, c in zip(sums, counts)]
            gpu_sums = [s if c > 0 else -math.inf for s, c in zip(sums, counts)]
            for frac, win, hw in ((0.20, (0, 0), 64), (0.20, (8, 24), 128), (0.09, (8, 24), 128), (0.12, (8, 24), 64)):
                cpu_res = index20_selector.select_from_reduced_scores(
                    docs, total, cpu_sums, counts, fraction=frac, strategy="max",
                    span_window=win, head_width=hw, allow_budget_floor=True,
                )
                gpu_res, target_body = index20_selector.select_from_reduced_scores_gpu(
                    docs, total, torch_mod.as_tensor(gpu_sums, dtype=torch_mod.float32),
                    torch_mod.as_tensor(counts, dtype=torch_mod.int64), fraction=frac,
                    strategy="max", span_window=win, head_width=hw, torch_module=torch_mod, geometry_cache={},
                )
                self.assertEqual(gpu_res["selected_positions"], cpu_res["selected_positions"])
                self.assertEqual(target_body, cpu_res["selected_body_tokens"])

    def test_geometry_frame_capacity_and_budget_admission(self) -> None:
        rng = random.Random(42)
        for hw in (0, 64, 128):
            docs, cursor = [], 17
            for i in range(12):
                length = rng.randint(256, 2400)
                docs.append({"start": cursor, "end": cursor + length, "key": f"d{i}"})
                cursor += length + rng.randint(5, 80)
            total = cursor + 150
            self.assertGreater(total, 8192)
            res = index20_selector.select_from_reduced_scores(
                docs, total, [rng.uniform(-5.0, 10.0) for _ in range(total // 2)], [1] * (total // 2),
                fraction=0.20, strategy="max", span_window=(8, 24), head_width=hw, allow_budget_floor=True,
            )
            frame_rows = index20_selector.validate_frame_capacity(res["selected_positions"], total)
            self.assertEqual(sum(frame_rows), len(res["selected_positions"]))
            sel_set = set(res["selected_positions"])
            for d in docs:
                heads, tails = set(range(d["start"], d["start"] + hw)), set(range(d["end"] - 128, d["end"]))
                self.assertTrue(heads.isdisjoint(tails) and heads <= sel_set and tails <= sel_set)

        # Budget admission: legacy 0.15 only; V2 admits exact policy stage1/target (e.g. 20+0, 9+3, 12+3)
        b_docs = [{"start": 10, "end": 4010, "key": "d0"}, {"start": 4020, "end": 8020, "key": "d1"}]
        self.assertEqual(fixed_pack_policy.budget_admission(b_docs, 8200, 0.15)["body_fraction"], 0.15)
        with self.assertRaises(ValueError):
            fixed_pack_policy.budget_admission(b_docs, 8200, 0.20)
        p_single = v2_policy.V2PolicyConfig(keep_pct="20+0").resolve(64_000)
        p_two = v2_policy.V2PolicyConfig(keep_pct="9+3").resolve(64_000)
        p_12p3 = v2_policy.V2PolicyConfig(keep_pct="12+3").resolve(64_000)
        self.assertEqual(fixed_pack_policy.budget_admission(b_docs, 8200, 0.20, policy=p_single)["head_width"], 64)
        self.assertTrue(fixed_pack_policy.budget_admission(b_docs, 8200, 0.09, policy=p_two)["two_stage_l14"])
        self.assertTrue(fixed_pack_policy.budget_admission(b_docs, 8200, 0.12, policy=p_two)["two_stage_l14"])
        self.assertTrue(fixed_pack_policy.budget_admission(b_docs, 8200, 0.15, policy=p_12p3)["two_stage_l14"])
        with self.assertRaises(ValueError):
            fixed_pack_policy.budget_admission(b_docs, 8200, 0.10, policy=p_two)
        with self.assertRaises(ValueError):
            fixed_pack_policy.budget_admission([{"start": 0, "end": 200, "key": "s"}], 400, 0.20, policy=p_single)

    def test_minimal_cone_closure_opt2_halo_and_proof_audit(self) -> None:
        total = 32_768
        docs = [{"start": 100 + i * 3100, "end": 100 + i * 3100 + 2900, "key": f"d{i}"} for i in range(10)]
        mandatory = index20_selector.mandatory_positions(docs, total)
        extra = [d["start"] + k for d in docs for k in range(64)] + [d["start"] + 500 + k for d in docs for k in range(300)]
        selected = sorted(set(mandatory) | set(extra))
        final_frame, ranges = (total - 1) // 8192 * 8192, [(0, 16_384), (16_384, total)]
        full_cone = cone15_plan.Cone15Plan(total, selected, mandatory, last_frame_start=final_frame, group_ranges=ranges)
        min_cone = cone15_plan.Cone15Plan(total, selected, (), last_frame_start=final_frame, group_ranges=ranges)

        # Reverse causal completeness across L21..L39
        self.assertEqual(min_cone.layers[39].queries, min_cone.sampler)
        for l in range(21, 40):
            lm = min_cone.layers[l]
            self.assertEqual(lm.queries & ~min_cone.selected_mask, 0)
            self.assertEqual(lm.inputs & ~min_cone.selected_mask, 0)
            if l in cone15_plan.D_SPARK:
                self.assertEqual(min_cone.ring & ~lm.inputs, 0)
            self.assertEqual(cone15_plan.windows(lm.queries) | min_cone.ring, lm.online | lm.offline)
            self.assertEqual(lm.online & lm.offline, 0)
            if l < 39:
                self.assertEqual(min_cone.layers[l + 1].inputs & ~lm.queries, 0)

        full_q = sum(len(full_cone.group(l, a, b).queries) for a, b in ranges for l in range(21, 40))
        min_q = sum(len(min_cone.group(l, a, b).queries) for a, b in ranges for l in range(21, 40))
        self.assertLess(min_q, full_q * 0.5)

        # Opt-2 halo pruning plan preserves every reachable offline key
        sel_set = set(selected)
        universe = [p for p in fixed15_geometry.window_union(selected) if p not in sel_set]
        bound = [
            (d["key"], m[0][0], m[-1][0] + 1, [s for _, s in m], None)
            for d in docs if (m := [(i, p - d["start"]) for i, p in enumerate(universe) if d["start"] <= p < d["end"]])
        ]
        l25_wanted = cone15_plan.row_ids(min_cone.layers[25].offline, 0, total)
        keep, inv, _ = halo_direct_copy.plan_layer_halo_gather(universe, l25_wanted, bound)
        self.assertEqual(len(keep), len(l25_wanted))
        self.assertLess(len(keep), len(universe))
        for pos in l25_wanted:
            self.assertEqual(universe[keep[inv[universe.index(pos)]]], pos)

        # Proof audit accepts MinimalCone and rejects legacy without minimal_plan
        a, b = ranges[0]
        geom = min_cone.group(25, a, b)
        chunk_sel = list(min_cone.selection(a, b))
        chunk_tails = [p for d in docs for p in range(d["end"] - 128, d["end"]) if a <= p < b]
        ev = dict(
            chunk_start=a, chunk_end=b, selected_positions=chunk_sel, selected_rows=len(chunk_sel),
            cone15_ffn_positions=list(geom.queries), cone15_input_positions=list(geom.inputs) or chunk_sel,
            cone15_mhc_positions=list(geom.inputs) or chunk_sel, actual_ffn_rows=len(geom.queries),
            actual_mhc_rows=len(geom.inputs) if geom.inputs else len(chunk_sel),
        )
        self.assertTrue(cone15_ffn_native_proof.live_covered(ev, chunk_tails, 25, "actual_ffn_rows", "cone15_ffn_positions", minimal_plan=min_cone))
        self.assertFalse(cone15_ffn_native_proof.live_covered(ev, chunk_tails, 25, "actual_ffn_rows", "cone15_ffn_positions", minimal_plan=None))

    def test_both_recipes_single_stage_and_two_stage_l14_locality(self) -> None:
        # Reproduce both optimal recipes on a 128K 10-doc multi-hop RAG prompt
        torch_mod = _torch_mod()
        rng = random.Random(2026)
        total = 128_000
        docs = [{"start": 16 + i * 12_754, "end": 16 + i * 12_754 + 12_750, "key": f"doc-{i}"} for i in range(10)]
        n_keys = total // 2
        l2_scores = [rng.uniform(-1.5, 1.5) for _ in range(n_keys)]
        for rel in (docs[2], docs[7]):
            for k in range((rel["start"] + 4_000) // 2, (rel["start"] + 4_080) // 2):
                l2_scores[k] = rng.uniform(6.0, 10.0)
        counts = [1] * n_keys

        # Recipe 1: single-stage (20+0)
        p_single = v2_policy.V2PolicyConfig(keep_pct="20+0").resolve(total)
        sel_single, _ = index20_selector.select_from_reduced_scores_gpu(
            docs, total, torch_mod.as_tensor(l2_scores, dtype=torch_mod.float32),
            torch_mod.as_tensor(counts, dtype=torch_mod.int64),
            fraction=p_single.fraction, strategy="max", span_window=p_single.window,
            head_width=p_single.head_width, torch_module=torch_mod, geometry_cache={},
        )
        index20_selector.validate_frame_capacity(sel_single["selected_positions"], total)

        # Recipe 2: two-stage (9+3 = 9% Stage-1 + 3% L14 bridge promotion = 12% target)
        p_two = v2_policy.V2PolicyConfig(keep_pct="9+3").resolve(total)
        sel_s1, target_body = index20_selector.select_from_reduced_scores_gpu(
            docs, total, torch_mod.as_tensor(l2_scores, dtype=torch_mod.float32),
            torch_mod.as_tensor(counts, dtype=torch_mod.int64),
            fraction=p_two.fraction, stage1_fraction=p_two.stage1_fraction, strategy="max",
            span_window=p_two.window, head_width=p_two.head_width, torch_module=torch_mod, geometry_cache={},
        )
        valid_l14 = [False] * n_keys
        for d in docs:
            for k in range((d["start"] + 1) // 2, d["end"] // 2):
                valid_l14[k] = True
        l14_scores = [rng.uniform(-1.0, 2.0) for _ in range(n_keys)]
        bridge_start = docs[5]["start"] + 5_000
        for k in range(bridge_start // 2, (bridge_start + 160) // 2):
            l14_scores[k] = rng.uniform(9.0, 14.0)
        sel_two, extra = two_stage_l14.promote_stage2(
            sel_s1, docs, total, l14_scores, valid_l14,
            target_fraction=p_two.fraction, target_body_tokens=target_body, span_window=p_two.window,
        )
        index20_selector.validate_frame_capacity(sel_two["selected_positions"], total)

        s1_set, extra_set, two_set = set(sel_s1["selected_positions"]), set(extra), set(sel_two["selected_positions"])
        self.assertLess(len(two_set), len(sel_single["selected_positions"]))  # 12% < 20%
        self.assertTrue(extra_set.isdisjoint(s1_set) and (s1_set | extra_set) == two_set)
        self.assertEqual(sel_two["selected_body_tokens"], target_body)
        self.assertTrue(all(p in two_set for p in range(bridge_start, bridge_start + 160)))

        # L14 strict locality law + backfill proof audit
        affected = two_stage_l14.affected_positions(s1_set, extra, sel_two["selected_positions"], mid=14, total=total)
        self.assertEqual(set(affected.keys()), set(range(3, 14)))
        for layer in range(3, 14):
            self.assertEqual(affected[layer], extra_set)
            self.assertTrue(affected[layer].isdisjoint(s1_set))
        bank = two_stage_l14.PrefixBank(sorted(s1_set))
        bank.data[(5, "hidden")], bank.data[(5, "swa")] = object(), object()
        backfill_ev = dict(
            selected_positions=sel_two["selected_positions"], selected_rows=len(two_set),
            backfill_positions=sorted(extra_set), actual_ffn_rows=len(extra_set),
        )
        self.assertTrue(cone15_ffn_native_proof.live_covered(
            backfill_ev, [], 5, "actual_ffn_rows", "cone15_ffn_positions",
            two_stage_state={"mid": 14, "affected": affected, "states": bank},
        ))


class TestRepositoryContractsAndCLI(unittest.TestCase):
    def test_suffix_only_collector_and_opt1_metadata_cache(self) -> None:
        # 1. Suffix-only collector mask excludes doc tail128 in V2 and includes them in legacy S1.6
        class _Mod:
            layer_idx, kv_compress_ratio, use_candidate_topk = 2, 2, False
            def _compute_prefill_p_sum(self, *a, **k): return None

        docs = [{"start": 10, "end": 310, "key": "d0"}, {"start": 320, "end": 620, "key": "d1"}]
        collector = index20_selector.NativeIndexScoreCollector(_Mod(), torch_module=_FakeTorch, allow_sync=lambda: None)
        try:
            collector.configure_v2(dict(suffix_only=True, head_width=64, span_window=(0, 0), gpu_select=True))
            collector.begin(docs, 700, "cpu", fraction=0.20, strategy="max")
            self.assertEqual(collector.mandatory, list(range(620, 700)))
            self.assertTrue(not any(collector.query_mask._data[:620]) and all(collector.query_mask._data[620:]))
            collector.active = False

            collector.configure_v2(None)
            collector.begin(docs, 700, "cpu", fraction=0.15, strategy="max")
            self.assertEqual(collector.mandatory, index20_selector.mandatory_positions(docs, 700))
            collector.active = False
        finally:
            collector.close()

        # 2. Opt-1 static metadata cache hits on repeated reset and invalidates on binding change
        rag_root = str(ALGO_ROOT / "rag")
        if rag_root not in sys.path:
            sys.path.insert(0, rag_root)
        import fixed_pack_controller, rag_natural_documents as rag

        ids = list(range(100, 700))
        r_docs = [dict(input_start=10, input_end=310, token_ids_sha256=rag.digest(ids[10:310]))]
        orig_natural, calls = rag.natural_documents, [0]

        def counting_natural(call_ids, call_docs):
            calls[0] += 1
            return orig_natural(call_ids, call_docs)

        class _BaseCtrl:
            def __init__(self):
                self.state, self.case_input, self.case_binding = "IDLE", "case-1", ("bind-1",)
                self.cache = SimpleNamespace(scope={"model": "v41"})
            def close(self): pass

        ctrl = fixed_pack_controller.fixed_pack_controller_class(_BaseCtrl)()
        rag.natural_documents = counting_natural
        try:
            with ctrl._opt1_static_reset_scope(tuple(ids), r_docs):
                d1 = rag.natural_cache_documents(ids, r_docs)
                self.assertEqual(rag.natural_cache_documents(ids, r_docs), d1)
            self.assertEqual((calls[0], ctrl._v2_reset_stats["route"]), (1, "miss"))
            with ctrl._opt1_static_reset_scope(tuple(ids), r_docs):
                self.assertEqual(rag.natural_cache_documents(ids, r_docs), d1)
            self.assertEqual((calls[0], ctrl._v2_reset_stats["route"]), (1, "hit"))
            ctrl.case_binding = ("bind-2",)
            with ctrl._opt1_static_reset_scope(tuple(ids), r_docs):
                rag.natural_cache_documents(ids, r_docs)
            self.assertEqual((calls[0], ctrl._v2_reset_stats["route"]), (2, "miss"))
        finally:
            rag.natural_documents = orig_natural

    def test_ast_rewrite_zero_gpu_preflight_and_no_blocking_sha256_gates(self) -> None:
        # 1. FixedRuntimeRewrite AST (1,1,1,1,1) assertions intact
        import multiowner_runtime
        rewritten = fixed_pack_runtime_adapter.private_preserved_module(multiowner_runtime)
        self.assertEqual(rewritten.canonical_method("Cone15", "max"), ("Cone15", True, "max"))

        # 2. Open-source non-blocking SHA-256 gates & file registration
        spec = importlib.util.spec_from_file_location("t_run_fixed", PRODUCER_ROOT / "run_fixed.py")
        rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
        self.assertFalse(hasattr(rf, "PARENT_SHA"))
        self.assertTrue(callable(rf.load_base(PRODUCER_ROOT / "run_s16.py").compact_controller))
        self.assertEqual(fixed_pack_runtime_adapter.pinned({}, FIXED_ROOT / "v2_policy.py"), (FIXED_ROOT / "v2_policy.py").resolve())
        self.assertTrue({"v2_policy.py", "two_stage_l14.py"} <= set(fixed_pack_runtime_adapter.OWN_FILES))
        self.assertIn("runtime/halo_direct_copy.py", fixed_pack_runtime_adapter.SHARED_FILES)

        # 3. CPU preflight modules never import torch / triton / deepinfer
        mods = [n[:-3] for n in fixed_pack_runtime_adapter.OWN_FILES] + ["halo_direct_copy", "packed_attention", "index20_selector", "run_s16"]
        paths = [str(p) for p in (FIXED_ROOT, LASTKV_ROOT, SHARED_RUNTIME_ROOT, SHARED_ROOT, MULTIOWNER_ROOT, PRODUCER_ROOT)]
        code = (
            f"import importlib, sys; sys.path[:0] = {paths!r}; [importlib.import_module(m) for m in {mods!r}]; "
            "assert not ({'torch', 'triton', 'deepinfer'} & set(sys.modules))"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_compact_controller_and_cli_preflight(self) -> None:
        def make_report(fraction: float, policy_dict=None, *, two_stage=False):
            layers = [dict(layer=l, actual_ffn_rows=10, dispatch_rows=10, live_ffn_rows=10,
                           core=dict(actual_attention_rows=10, input_axis_rows=20)) for l in range(21, 40)]
            pack = dict(enabled=True, canonical_method="Cone15")
            if policy_dict is not None:
                pack["v2_policy"] = policy_dict
            return dict(
                input_ids_sha256="a" * 64, full_input_tokens=65_000, mode="sensitive",
                model_boundary_restored=True, decode_native=True,
                cone15_ffn=dict(method="Cone15", minimal_cone=bool(policy_dict and policy_dict.get("minimal_cone")),
                                new_and_tail_full_ffn_each_layer=True, all_input_lastkv_physical_replay=True,
                                qualification_cache_bytes=None, groups=[dict(layers=layers)]),
                index20=dict(selection=dict(strategy="max", fraction=fraction, two_stage_l14=two_stage,
                                            selected_positions=[0, 1, 2, 3])),
                system_pack_fixed_v1=pack,
            )

        self.assertEqual(run_s16.compact_controller(make_report(0.15, None))["sparse_method"], "Cone15")
        with self.assertRaises(ValueError):
            run_s16.compact_controller(make_report(0.20, None))
        p_single = v2_policy.V2PolicyConfig(keep_pct="20+0").resolve(65_000).to_dict()
        self.assertEqual(run_s16.compact_controller(make_report(0.20, p_single))["v2_policy"]["mode"], "single-stage")
        p_two = v2_policy.V2PolicyConfig(keep_pct="9+3").resolve(65_000).to_dict()
        self.assertTrue(run_s16.compact_controller(make_report(0.12, p_two, two_stage=True))["v2_policy"]["two_stage_l14"])
        with self.assertRaises(ValueError):
            run_s16.compact_controller(make_report(0.20, p_two, two_stage=True))

        # CLI checks across benchmark_rag_redknot.py, benchmark_rag_redknot_21.py, and examples/redknot_v2_demo.py
        for script in ("test/benchmark_rag_redknot.py", "test/benchmark_rag_redknot_21.py", "examples/redknot_v2_demo.py"):
            help_proc = subprocess.run([sys.executable, str(REPO_ROOT / script), "--help"], capture_output=True, text=True)
            self.assertEqual(help_proc.returncode, 0, help_proc.stderr)
            self.assertIn("--keep-pct", help_proc.stdout)
            bad_proc = subprocess.run([sys.executable, str(REPO_ROOT / script), "--keep-pct", "0+3"], capture_output=True, text=True)
            self.assertNotEqual(bad_proc.returncode, 0)

    @unittest.skipIf(sys.version_info < (3, 11), "local_backend.hash_file uses hashlib.file_digest (Python 3.11+)")
    def test_bundled_runtime_prepare_native_preflight(self) -> None:
        code = f"""
import sys, tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, {str(REPO_ROOT / 'test')!r})
from longbench300 import local_backend as lb
assert not hasattr(lb, 'PINS')
with tempfile.TemporaryDirectory() as tmp:
    model, out = Path(tmp) / 'model', Path(tmp) / 'out'
    model.mkdir(); out.mkdir()
    v2_cls = lb.load_v2_policy_config_class()
    with patch.object(lb, 'verify_checkpoint', return_value=None):
        cfg_single, loc = lb.bundled_runtime(model, out, v2_config=v2_cls.from_cli(keep_pct='20+0'))
        cfg_two, _ = lb.bundled_runtime(model, out, v2_config=v2_cls.from_cli(keep_pct='9+3'))
        cfg_12p3, _ = lb.bundled_runtime(model, out, v2_config=v2_cls.from_cli(keep_pct='12+3'))
        cfg_s16, _ = lb.bundled_runtime(model, out, v2_config=v2_cls.legacy_s16())
    assert len({{cfg_single['sha256'], cfg_two['sha256'], cfg_12p3['sha256'], cfg_s16['sha256']}}) == 4
    assert (cfg_single['redknot_v2']['keep_pct'], cfg_two['redknot_v2']['keep_pct'], cfg_12p3['redknot_v2']['keep_pct'], cfg_s16['redknot_v2']['keep_pct']) == ('20+0', '9+3', '12+3', 's1.6')
    lb.prepare_native(cfg_two)
    base, producer = lb.local_modules(cfg_two, loc)
    assert callable(base.compact_controller) and callable(producer.identity)
    assert not ({{'torch', 'triton', 'deepinfer'}} & set(sys.modules))
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
