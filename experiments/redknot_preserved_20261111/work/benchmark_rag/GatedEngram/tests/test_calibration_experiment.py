import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kv_cache_engram.calibration_experiment import run_calibration_experiment


class CalibrationExperimentTests(unittest.TestCase):
    def test_all_scenarios_fail_closed_and_preserve_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new"
            result = run_calibration_experiment(target)
            self.assertTrue(result["clean_heldout"]["kv_validation_pass"])
            self.assertFalse(result["adversarial_heldout"]["kv_validation_pass"])
            self.assertEqual(result["adversarial_heldout"]["summary"]["bad_reused_rows"], 1)
            self.assertFalse(result["adversarial_heldout"]["candidate_reselection_performed"])
            self.assertEqual(result["sparse_calibration"]["mode"], "full_recompute")
            self.assertFalse(result["sparse_calibration"]["heldout_pass"])
            self.assertEqual(len(result["sparse_calibration"]["candidates"]), 4)
            self.assertFalse(any(c["eligible"] for c in result["sparse_calibration"]["candidates"]))
            self.assertEqual(result["uniform_single_bad_row_budget"]["probes_for_one_percent_conditional_miss"], 507)
            self.assertEqual(result["actual_model_captures"], 0)
            self.assertFalse(result["runtime_admission_created"])
            for item in result["artifacts"].values():
                path = target / item["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
                with gzip.open(path, "rt") as stream:
                    self.assertIsInstance(json.load(stream), dict)
            with self.assertRaises(FileExistsError):
                run_calibration_experiment(target)


if __name__ == "__main__":
    unittest.main()
