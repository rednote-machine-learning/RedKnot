"""CPU-only contract tests for the pure-MLA transfer evidence audit."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_torch_stub_if_missing():
    """Let this focused parser test run in a stdlib-only review workspace."""

    try:
        import torch  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    torch = types.ModuleType("torch")

    def no_grad(function=None):
        if function is None:
            return lambda wrapped: wrapped
        return function

    torch.no_grad = no_grad
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        synchronize=lambda: None,
        empty_cache=lambda: None,
    )
    torch.version = types.SimpleNamespace(cuda=None)
    torch.bfloat16 = object()
    torch.float16 = object()
    torch.float32 = object()
    torch.__file__ = __file__
    torch.__version__ = "test-stub"
    sys.modules["torch"] = torch


def _load_benchmark():
    _install_torch_stub_if_missing()
    path = Path(__file__).with_name(
        "benchmark_RedKnot_DeepSeekV4_Flash_RAG.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_redknot_transfer_audit_contract_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPureMLATransferAuditContract(unittest.TestCase):
    TP_SIZE = 8

    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_benchmark()

    def _controller_delta(self, *, legacy=False, dirty_rows=0):
        delta = {
            field: 0
            for field in self.benchmark._IH_MLA_OFF_TRANSFER_COUNTER_FIELDS
        }
        if legacy:
            delta.update(
                {
                    "device_restore_calls": 1,
                    "device_rows_restored": 128,
                    "online_device_gather_index_h2d_calls": 1,
                    "online_device_gather_index_h2d_rows": 128,
                    "online_device_gather_index_h2d_bytes": 1024,
                    "online_device_scatter_index_h2d_calls": 1,
                    "online_device_scatter_index_h2d_rows": 128,
                    "online_device_scatter_index_h2d_bytes": 1024,
                }
            )
        if dirty_rows:
            delta.update(
                {
                    "online_dirty_index_h2d_calls": 1,
                    "online_dirty_index_h2d_rows": dirty_rows,
                    "online_dirty_index_h2d_bytes": dirty_rows * 8,
                }
            )
        delta["online_index_h2d_bytes"] = (
            delta["online_device_gather_index_h2d_bytes"]
            + delta["online_device_scatter_index_h2d_bytes"]
            + delta["online_dirty_index_h2d_bytes"]
        )
        delta["online_total_h2d_bytes"] = (
            delta["online_artifact_h2d_bytes"]
            + delta["online_index_h2d_bytes"]
        )
        return delta

    def _shared_delta(self, values):
        return dict(
            zip(self.benchmark._IH_SHARED_RESTORE_COUNTER_FIELDS, values)
        )

    def _event(self, request_id, forward_id, rank, controller, shared):
        zero_controller = {
            field: 0
            for field in self.benchmark._IH_MLA_OFF_TRANSFER_COUNTER_FIELDS
        }
        zero_shared = {
            field: 0
            for field in self.benchmark._IH_SHARED_RESTORE_COUNTER_FIELDS
        }
        return {
            "path_rank": rank,
            "payload": {
                "schema": self.benchmark._IH_MLA_OFF_TRANSFER_AUDIT_SCHEMA,
                "byte_semantics": (
                    self.benchmark._IH_MLA_OFF_TRANSFER_BYTE_SEMANTICS
                ),
                "request_id": request_id,
                "forward_id": forward_id,
                "forward_mode": "extend",
                "q_rows": 8192 if forward_id == "prefix" else 49,
                "tp_rank": rank,
                "tp_size": self.TP_SIZE,
                "counter_start": zero_controller,
                "counter_end": dict(controller),
                "counter_delta": dict(controller),
                "gauge_snapshot": {
                    "device_cache_enabled": 1,
                    "reserved_device_bytes": 4096,
                    "allocated_device_bytes": 2048,
                    "max_device_cache_bytes": 8192,
                },
                "shared_restore": {
                    "schema": self.benchmark._IH_SHARED_RESTORE_AUDIT_SCHEMA,
                    "counter_start": zero_shared,
                    "counter_end": dict(shared),
                    "counter_delta": dict(shared),
                },
            },
        }

    def _runtime(self, prefix_controller, prefix_shared, *, suffix=False):
        request_id = "request"
        events = [
            self._event(
                request_id, "prefix", rank, prefix_controller, prefix_shared
            )
            for rank in range(self.TP_SIZE)
        ]
        manifests = {
            "prefix": {"forward_mode": "extend", "q_rows": 8192}
        }
        metrics = {
            "prefix": {
                "3": {"reused_local_head_rows": 57344},
            }
        }
        if suffix:
            zero_controller = self._controller_delta()
            zero_shared = self._shared_delta((0, 0, 0))
            events.extend(
                self._event(
                    request_id, "suffix", rank, zero_controller, zero_shared
                )
                for rank in range(self.TP_SIZE)
            )
            manifests["suffix"] = {"forward_mode": "extend", "q_rows": 49}
            metrics["suffix"] = {
                "3": {"reused_local_head_rows": 0},
            }
        return {
            "mla_off_transfer_audit": {
                "events": events,
                "parse_errors": [],
            },
            "mla_forward_manifest": {request_id: manifests},
            "mla_forward_metric_rows": {request_id: metrics},
        }

    def _validate(self, runtime):
        return self.benchmark._ih_validate_mla_transfer_audit(
            runtime,
            expected_request_ids={"request"},
            expected_tp_size=self.TP_SIZE,
            expected_device_cache_enabled=True,
            expected_device_max_bytes=8192,
        )

    def test_shared_only_positive_is_composite_restore_and_suffix_zero_passes(self):
        runtime = self._runtime(
            self._controller_delta(dirty_rows=128),
            self._shared_delta((1, 147, 378103)),
            suffix=True,
        )

        result = self._validate(runtime)

        self.assertTrue(result["pass"], result["evidence_errors"])
        evidence = result["positive_evidence"]
        self.assertTrue(evidence["composite_device_restore_positive_all_ranks"])
        self.assertTrue(evidence["shared_device_restore_positive_all_ranks"])
        self.assertFalse(evidence["legacy_device_restore_positive_all_ranks"])
        self.assertEqual(evidence["restore_evidence_modes"], ["shared_composite"])
        self.assertEqual(
            result["by_request"]["request"]["suffix"][
                "restore_evidence_mode"
            ],
            "none",
        )

    def test_partial_or_zero_shared_without_legacy_fails_closed(self):
        cases = {
            "partial": (1, 0, 378103),
            "zero": (0, 0, 0),
        }
        for name, shared_values in cases.items():
            with self.subTest(name=name):
                result = self._validate(
                    self._runtime(
                        self._controller_delta(),
                        self._shared_delta(shared_values),
                    )
                )
                self.assertFalse(result["pass"])
                details = "\n".join(result["evidence_errors"])
                if name == "partial":
                    self.assertIn("partial_shared_device_restore", details)
                else:
                    self.assertIn("missing_device_restore", details)
                    self.assertIn(
                        "device_gather_scatter_index_mismatch", details
                    )

    def test_legacy_positive_passes_when_shared_is_inactive(self):
        result = self._validate(
            self._runtime(
                self._controller_delta(legacy=True),
                self._shared_delta((0, 0, 0)),
            )
        )

        self.assertTrue(result["pass"], result["evidence_errors"])
        evidence = result["positive_evidence"]
        self.assertTrue(evidence["composite_device_restore_positive_all_ranks"])
        self.assertFalse(evidence["shared_device_restore_positive_all_ranks"])
        self.assertTrue(evidence["legacy_device_restore_positive_all_ranks"])
        self.assertEqual(evidence["restore_evidence_modes"], ["legacy_device"])

    def test_mixed_shared_and_legacy_evidence_fails_closed(self):
        result = self._validate(
            self._runtime(
                self._controller_delta(legacy=True),
                self._shared_delta((1, 147, 378103)),
            )
        )

        self.assertFalse(result["pass"])
        self.assertIn(
            "mixed_shared_legacy_device_restore",
            "\n".join(result["evidence_errors"]),
        )

    def test_pure_failure_label_does_not_claim_indexer_hot(self):
        with patch.multiple(
            self.benchmark,
            ENGINE_MODE="indexer_hot",
            DRY_RUN=False,
            DRIFT_PROFILE=False,
            PROFILE=False,
            IH_MLA_OFFLOAD=True,
        ), patch.object(
            self.benchmark, "_run_indexer_hot_benchmark", return_value=False
        ):
            with self.assertRaisesRegex(
                SystemExit,
                "pure MLA offline/online merge quality/runtime/performance",
            ):
                self.benchmark.main()


if __name__ == "__main__":
    unittest.main()
