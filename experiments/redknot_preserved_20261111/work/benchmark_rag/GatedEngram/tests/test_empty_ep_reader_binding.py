"""CPU evidence integrity tests, not native GPU qualification."""
import copy
import unittest

from gpu_pilot import read_ffn_only_results as reader
from kv_cache_engram.core import ValidationError
from test_ffn_only_pilot import fixture


class EmptyEPBindingTests(unittest.TestCase):
    def test_exact_phase_and_controller_binding(self):
        proof = "a" * 64
        phase = dict(empty_ep_qualification_sha256=proof,
                     controller=dict(empty_ep_qualification_sha256=proof))
        reader.empty_ep_binding(phase, dict(empty_ep_qualification_sha256=proof))

    def test_missing_or_foreign_phase_and_controller_rejected(self):
        proof = "a" * 64
        for producer in (None, "b" * 64):
            for field in ("phase", "controller"):
                phase = dict(empty_ep_qualification_sha256=proof,
                             controller=dict(empty_ep_qualification_sha256=proof))
                target = phase if field == "phase" else phase["controller"]
                target["empty_ep_qualification_sha256"] = producer
                with self.subTest(producer=producer, field=field), self.assertRaises(ValidationError):
                    reader.empty_ep_binding(phase, dict(empty_ep_qualification_sha256=proof))

    def test_unsealed_qualification_identifier_rejected(self):
        with self.assertRaises(ValidationError):
            reader.empty_ep_binding({}, dict(empty_ep_qualification_sha256=None))

    def test_invented_empty_call_count_rejected(self):
        control, row, phase, spans = fixture()
        reader.audit_controller(control, row, phase, spans)
        control["adapter"]["counts"]["empty_native_mlp_calls"] += 1
        with self.assertRaisesRegex(ValidationError, "empty native EP"):
            reader.audit_controller(control, row, phase, spans)


if __name__ == "__main__":
    unittest.main()
