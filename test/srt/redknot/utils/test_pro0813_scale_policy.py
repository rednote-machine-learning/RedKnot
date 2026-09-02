"""CPU-only checks for component-specific Pro-0813/Flash sizing."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
POLICY_PATH = (
    ROOT
    / "python/sglang/srt/layers/attention/redknot/pro0813/scale_policy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_redknot_pro0813_scale_policy_test", POLICY_PATH
)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)

SELECTOR_PATH = (
    ROOT
    / "python/sglang/srt/layers/attention/redknot/v4/request_selector.py"
)
SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "_redknot_request_selector_capacity_test", SELECTOR_PATH
)
assert SELECTOR_SPEC is not None and SELECTOR_SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SELECTOR_SPEC)
sys.modules[SELECTOR_SPEC.name] = SELECTOR
SELECTOR_SPEC.loader.exec_module(SELECTOR)


class TestPro0813ScalePolicy(unittest.TestCase):
    def test_model_size_is_not_used_as_one_uniform_multiplier(self) -> None:
        audit = POLICY.scale_policy_audit()
        ratios = audit["component_ratios_pro_over_flash"]
        self.assertAlmostEqual(ratios["logical_weight_bytes"], 5.349565018656392)
        self.assertAlmostEqual(ratios["reusable_layers"], 55 / 37)
        self.assertEqual(ratios["attention_heads"], 2.0)
        self.assertEqual(ratios["index_topk"], 2.0)
        self.assertEqual(ratios["q_lora_rank"], 1.5)
        self.assertAlmostEqual(ratios["zoff_bytes_per_token"], 110 / 37)
        self.assertEqual(
            audit["component_geometry"]["physical_mla"]["row_width_ratio"],
            1.0,
        )

    def test_zoff_and_host_reservations_are_exact_for_every_target(self) -> None:
        expected_zoff = {
            65_536: 14_763_950_080,
            131_072: 29_527_900_160,
            262_144: 59_055_800_320,
            450_560: 101_502_156_800,
            524_288: 118_111_600_640,
        }
        audit = POLICY.scale_policy_audit()
        for target, zoff in expected_zoff.items():
            with self.subTest(target=target):
                record = audit["targets"][str(target)]
                self.assertEqual(POLICY.pro0813_zoff_bytes_per_rank(target), zoff)
                self.assertEqual(record["zoff_bytes_per_rank"], zoff)
                self.assertGreater(
                    record["cpu_reservation_bytes_per_rank"], zoff
                )
                self.assertGreaterEqual(
                    record["cpu_cap_bytes_per_rank"],
                    record["cpu_reservation_bytes_per_rank"],
                )

    def test_b300_target_memory_policy_retains_runtime_reserve(self) -> None:
        self.assertEqual(
            dict(POLICY.PRO0813_MEM_FRACTION_STATIC),
            {
                65_536: 0.80,
                131_072: 0.77,
                262_144: 0.67,
                450_560: 0.80,
                524_288: 0.80,
            },
        )
        for target, fraction in POLICY.PRO0813_MEM_FRACTION_STATIC.items():
            resident_mib = (
                POLICY.pro0813_zoff_bytes_per_rank(target) // 1024**2
                if POLICY.PRO0813_DEVICE_ZOFF_CAP_MIB[target]
                else 0
            )
            used = math.ceil(POLICY.B300_TOTAL_MEMORY_MIB * fraction) + resident_mib
            self.assertLessEqual(
                used + POLICY.B300_RUNTIME_RESERVE_MIB,
                POLICY.B300_TOTAL_MEMORY_MIB,
            )

    def test_only_layer_position_controls_are_depth_scaled(self) -> None:
        self.assertEqual(POLICY.PRO0813_TOKEN_SPARSE_DEEP_START, 34)
        self.assertEqual(
            POLICY.PRO0813_PROGRESSIVE_TOPK_SCHEDULE,
            "0-15:6,16-39:5,40-60:4",
        )
        self.assertEqual(POLICY.PRO0813_STANDARD_ACTIVE_RATIO, 0.20)
        self.assertEqual(POLICY.PRO0813_QUERY_PROTECTION_TOKENS, 8192)
        audit = POLICY.scale_policy_audit()
        self.assertEqual(
            audit["row_policy"]["calibration_status"],
            "candidate_grid_requires_pro_measurement",
        )
        self.assertFalse(audit["moe_policy"]["adaptive_formal_enabled"])

    def test_policy_digest_is_deterministic(self) -> None:
        self.assertEqual(
            POLICY.PRO0813_SCALE_POLICY_DIGEST,
            POLICY.scale_policy_digest(),
        )
        self.assertRegex(POLICY.PRO0813_SCALE_POLICY_DIGEST, r"^sha256:[0-9a-f]{64}$")

    def test_checkpoint_capacity_is_target_specific_and_covers_diffuse_arm(
        self,
    ) -> None:
        expected = {
            65_536: 64,
            131_072: 64,
            262_144: 128,
            450_560: 256,
            524_288: 256,
        }
        self.assertEqual(dict(POLICY.PRO0813_CHECKPOINT_MAX_ISLANDS), expected)
        for target, capacity in expected.items():
            with self.subTest(target=target):
                diffuse_required = POLICY.pro0813_required_checkpoint_islands(
                    target, POLICY.PRO0813_DIFFUSE_ACTIVE_RATIO
                )
                self.assertLessEqual(diffuse_required, capacity)
                self.assertLessEqual(
                    capacity, POLICY.PRO0813_CHECKPOINT_DESCRIPTOR_LIMIT
                )
        for target in (262_144, 450_560, 524_288):
            with self.subTest(old_flash_cap_target=target):
                self.assertGreater(
                    POLICY.pro0813_required_checkpoint_islands(
                        target, POLICY.PRO0813_STANDARD_ACTIVE_RATIO
                    ),
                    64,
                )

    def test_realized_ratio_gate_fails_closed_on_missing_or_low_evidence(
        self,
    ) -> None:
        requested = POLICY.PRO0813_STANDARD_ACTIVE_RATIO
        minimum = POLICY.pro0813_min_realized_active_ratio(524_288, requested)
        missing = POLICY.pro0813_realized_active_ratio_gate(
            requested, minimum, None, required=True
        )
        low = POLICY.pro0813_realized_active_ratio_gate(
            requested, minimum, 1.0 - minimum + 0.001, required=True
        )
        enough = POLICY.pro0813_realized_active_ratio_gate(
            requested, minimum, 1.0 - minimum, required=True
        )
        self.assertFalse(missing["realized_active_ratio_pass"])
        self.assertFalse(low["realized_active_ratio_pass"])
        self.assertTrue(enough["realized_active_ratio_pass"])
        self.assertAlmostEqual(
            enough["actual_realized_active_ratio"], minimum
        )

    def test_fast_allocator_and_runtime_contract_accept_more_than_64_islands(
        self,
    ) -> None:
        cells = [
            SELECTOR.CheckpointCellCandidates(0, index, (1.0,))
            for index in range(100)
        ]
        selected = SELECTOR.allocate_checkpoint_cell_islands_fast(
            cells,
            token_budget_tokens=100 * 128,
            max_islands=128,
        )
        self.assertEqual(len(selected), 100)
        self.assertEqual(SELECTOR.MAX_CHECKPOINT_ISLANDS, 256)
        with self.assertRaisesRegex(ValueError, r"\[1, 256\]"):
            SELECTOR.allocate_checkpoint_cell_islands_fast(
                cells,
                token_budget_tokens=128,
                max_islands=257,
            )


if __name__ == "__main__":
    unittest.main()
