"""Tiny synthetic CPU workflow. No model accuracy or latency measurements."""
from dataclasses import asdict
import json
import tempfile
from pathlib import Path

from .cache import CacheArtifact, load_cache, save_cache
from .calibration import CalibrationRecord
from .core import Binding, CacheIdentity, OnlineRows, token_digest
from .gate import GatePolicy
from .runtime import BlockTransaction


def run_demo() -> dict:
    identity = CacheIdentity("a" * 64, "b" * 64, "c" * 64,
                             token_digest(range(32)), 3, 32, 2)
    binding = Binding("synthetic-request", "cpu-reference", identity, 100, 0, 16, "e" * 64)
    rows = tuple((float(i + 1), float(i + 2)) for i in range(32))
    policy = GatePolicy(enabled=True, block_size=16, probe_count=2,
                        head_guard=2, tail_guard=2,
                        relative_tolerance=0.0, absolute_tolerance=0.0,
                        calibration_id="synthetic-fixture-NOT-model-calibration")
    record = CalibrationRecord.bind(identity, policy, "f" * 64)
    results = []
    with tempfile.TemporaryDirectory(prefix="kv-cache-engram-cpu-") as folder:
        path = Path(folder) / "fixture.json"
        save_cache(path, CacheArtifact.build(identity, rows))
        cache = load_cache(path)
        for name, config, offset, qualification in (
            ("default_full_recompute", GatePolicy(block_size=16), 0.0, None),
            ("matching_probes_reuse", policy, 0.0, record),
            ("probe_mismatch_fallback", policy, 10.0, record),
        ):
            calls, submitted = [], []

            def compute_online(bound, positions):
                calls.append(len(positions))
                return OnlineRows(bound.sha256, positions,
                                  tuple(tuple(v + offset for v in rows[p]) for p in positions))

            tx = BlockTransaction(binding, cache, config, compute_online, calibration=qualification)
            prepared = tx.prepare()
            tx.commit(lambda bound, data: submitted.append((bound.sha256, len(data.rows))), binding)
            results.append({
                "scenario": name, "mode": prepared.mode,
                "fallback_reason": prepared.fallback_reason,
                "reused_rows": len(prepared.cached_positions),
                "online_rows_in_result": len(prepared.online_positions),
                "online_callback_row_counts": calls,
                "submit_calls": len(submitted), "state": tx.state,
                "gate": asdict(prepared.gate_decision) if prepared.gate_decision else None,
            })
    return {"module": "KV Cache Engram", "synthetic_cpu_fixture": True,
            "gpu_initialized": False, "model_accuracy_measured": False,
            "model_speedup_measured": False, "scenarios": results}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
