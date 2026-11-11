import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kv_cache_engram.experiments import SCENARIOS, run_synthetic_suite, synthetic_pair


class ExperimentTests(unittest.TestCase):
    def test_reproducible_pair(self):
        a, p, changed = synthetic_pair("fixed_sparse_spikes", token_count=2048)
        b, q, repeated = synthetic_pair("fixed_sparse_spikes", token_count=2048)
        self.assertEqual(a, b)
        self.assertEqual(p, q)
        self.assertEqual(changed, repeated)
        self.assertEqual(len(a.request_token_ids), 8*2048)

    def test_full_suite_preserves_all_scenarios_and_exposes_miss(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new"
            report = run_synthetic_suite(path, token_count=2048)
            self.assertEqual([r["scenario"] for r in report["scenarios"]], list(SCENARIOS))
            rows = {r["scenario"]: r for r in report["scenarios"]}
            self.assertEqual(rows["identical"]["summary"]["bad_reused_rows"], 0)
            self.assertEqual(rows["head127_only"]["summary"]["bad_reused_rows"], 0)
            self.assertGreater(rows["adversarial_unprobed_spike"]["summary"]["false_accept_blocks"], 0)
            self.assertGreater(rows["propagated_boundary1024"]["summary"]["fallback_probe_overhead_rows"], 0)
            for row in rows.values():
                artifact = path / row["full_report"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), row["file_sha256"])
                raw = json.loads(gzip.decompress(artifact.read_bytes()))
                self.assertEqual(len(raw["tokens"]), 2048)
                self.assertEqual(raw["data_origin_declared"], "synthetic")
            self.assertIsNone(report["actual_model_em_f1"])
            self.assertIsNone(report["actual_model_ttft"])
            with self.assertRaises(FileExistsError):
                run_synthetic_suite(path, token_count=2048)


if __name__ == "__main__":
    unittest.main()
