"""End-to-end CPU fixtures and explicit limitations, never model claims."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import unittest

from kv_cache_engram import (Binding, CacheArtifact, CacheIdentity, CalibrationRecord,
                             GatePolicy, OnlineRows, BlockTransaction, make_plan)
from kv_cache_engram.demo import run_demo


class IntegrationTests(unittest.TestCase):
    def test_serialized_cache_to_gate_and_commit_demo(self):
        result = run_demo()
        self.assertTrue(result["synthetic_cpu_fixture"])
        self.assertFalse(result["gpu_initialized"])
        self.assertFalse(result["model_accuracy_measured"])
        default, passing, rejected = result["scenarios"]
        self.assertEqual(default["mode"], "full_recompute")
        self.assertEqual(default["reused_rows"], 0)
        self.assertEqual(passing["reused_rows"], 12)
        self.assertEqual(passing["online_rows_in_result"], 4)
        self.assertEqual(rejected["mode"], "full_recompute")
        self.assertEqual(rejected["fallback_reason"], "threshold_exceeded")
        # Include failed probes in cost accounting; they are not free.
        self.assertEqual(rejected["online_callback_row_counts"], [2, 16])
        for scenario in result["scenarios"]:
            self.assertEqual(scenario["submit_calls"], 1)
            self.assertEqual(scenario["state"], "COMMITTED")

    def test_probe_acceptance_does_not_certify_unsampled_rows(self):
        identity = CacheIdentity("a" * 64, "b" * 64, "c" * 64, "d" * 64, 3, 16, 2)
        bound = Binding("r", "p", identity, 0, 0, 16, "e" * 64)
        policy = GatePolicy(enabled=True, block_size=16, probe_count=2, head_guard=0,
                            tail_guard=0, relative_tolerance=0.0, absolute_tolerance=0.0,
                            calibration_id="synthetic-limit-demonstration")
        plan = make_plan(bound, policy)
        hidden_bad = next(p for p in bound.positions if p not in plan.probe_positions)
        cache = CacheArtifact.build(identity, ((1.0, 2.0),) * 16)

        def compute(binding, positions):
            return OnlineRows(binding.sha256, positions,
                              tuple((999.0, 999.0) if p == hidden_bad else (1.0, 2.0) for p in positions))

        tx = BlockTransaction(bound, cache, policy, compute,
                              calibration=CalibrationRecord.bind(identity, policy, "f" * 64))
        prepared = tx.prepare()
        self.assertTrue(prepared.gate_decision.allow_reuse)
        self.assertIn(hidden_bad, prepared.cached_positions)
        self.assertNotEqual(prepared.rows[hidden_bad], compute(bound, (hidden_bad,)).rows[0])
        # This is deliberately a passing limitation test: sampled gates are not
        # equivalence certificates, even with zero sample error and a bound record.

    def test_no_heavy_runtime_imports_or_launches(self):
        package_root = str(Path(__file__).resolve().parents[1])
        code = '''
import importlib.abc, sys
blocked = {"torch", "triton", "numpy", "deepinfer", "cupy"}
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("unexpected heavy import: " + fullname)
sys.meta_path.insert(0, Reject())
from kv_cache_engram.demo import run_demo
assert run_demo()["synthetic_cpu_fixture"]
assert not blocked.intersection(sys.modules)
'''
        env = dict(os.environ, PYTHONPATH=package_root, PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                             capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_each_request_context_requires_its_own_online_binding(self):
        identity = CacheIdentity("a" * 64, "b" * 64, "c" * 64, "d" * 64, 3, 2, 1)
        first = Binding("same-id", "p", identity, 0, 0, 2, "e" * 64)
        second = replace(first, request_sha256="f" * 64)
        writes = []
        for bound in (first, second):
            tx = BlockTransaction(bound, None, GatePolicy(),
                                  lambda b, ps: OnlineRows(b.sha256, ps, ((1.0,),) * len(ps)))
            tx.prepare()
            tx.commit(lambda b, rows: writes.append(rows.binding_sha256), bound)
        self.assertNotEqual(writes[0], writes[1])


if __name__ == "__main__":
    unittest.main()
