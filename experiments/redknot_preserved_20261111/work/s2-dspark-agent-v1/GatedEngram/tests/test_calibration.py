"""Synthetic CPU contract fixtures, NOT real model calibration or accuracy data.

The repeated hexadecimal evidence digest is a test placeholder, not an artifact
asserting any actual model passed a calibration or downstream accuracy check.
"""
from dataclasses import FrozenInstanceError, asdict, replace
import subprocess
import sys
import unittest

from kv_cache_engram.calibration import CalibrationRecord
from kv_cache_engram.core import CacheIdentity, ValidationError, digest
from kv_cache_engram.gate import GatePolicy


def identity(**changes):
    fields = dict(model_sha256="a" * 64, tokenizer_sha256="b" * 64,
                  execution_sha256="c" * 64, tokens_sha256="d" * 64,
                  layer=3, token_count=1024, width=2)
    fields.update(changes)
    return CacheIdentity(**fields)


def policy(**changes):
    fields = dict(enabled=True, block_size=512, probe_count=8, head_guard=128,
                  tail_guard=128, relative_tolerance=0.01,
                  absolute_tolerance=0.001, norm_floor=1e-8,
                  calibration_id="synthetic-fixture-not-real-calibration",
                  seed="synthetic-plan-seed")
    fields.update(changes)
    return GatePolicy(**fields)


class CalibrationRecordTests(unittest.TestCase):
    def setUp(self):
        self.identity = identity()
        self.policy = policy()
        self.record = CalibrationRecord.bind(self.identity, self.policy, "e" * 64)

    def test_binding_validates_and_is_deterministic(self):
        self.assertIsNone(self.record.validate(self.identity, self.policy))
        self.assertEqual(self.record.record_id, self.policy.calibration_id)
        self.assertEqual(self.record, CalibrationRecord.bind(
            self.identity, self.policy, "e" * 64))

    def test_scope_excludes_only_document_digest_and_token_count(self):
        expected = asdict(self.identity)
        del expected["tokens_sha256"]
        del expected["token_count"]
        self.assertEqual(self.record.scope_sha256, digest(expected))
        another_document = identity(tokens_sha256="f" * 64, token_count=4097)
        self.record.validate(another_document, self.policy)
        self.assertEqual(self.record, CalibrationRecord.bind(
            another_document, self.policy, "e" * 64))

    def test_all_model_execution_layer_shape_identity_changes_rejected(self):
        for field, value in (
            ("model_sha256", "f" * 64), ("tokenizer_sha256", "f" * 64),
            ("execution_sha256", "f" * 64), ("layer", 4), ("width", 3),
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.record.validate(replace(self.identity, **{field: value}), self.policy)

    def test_representation_and_storage_remain_bound_and_revalidated(self):
        scope_fields = asdict(self.identity)
        scope_fields.pop("tokens_sha256")
        scope_fields.pop("token_count")
        for field in ("representation", "storage_dtype"):
            altered_scope = dict(scope_fields, **{field: "unsupported"})
            self.assertNotEqual(self.record.scope_sha256, digest(altered_scope))
            # Simulate accidental corruption bypassing the frozen constructor;
            # validation must not trust the original construction-time check.
            damaged = replace(self.identity)
            object.__setattr__(damaged, field, "unsupported")
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.record.validate(damaged, self.policy)

    def test_policy_fingerprint_includes_every_field(self):
        self.assertEqual(self.record.policy_sha256, digest(asdict(self.policy)))
        changes = {
            "enabled": False,
            "block_size": 256,
            "probe_count": 9,
            "head_guard": 0,
            "tail_guard": 0,
            "relative_tolerance": 0.02,
            "absolute_tolerance": 0.002,
            "norm_floor": 1e-6,
            "calibration_id": "another-synthetic-fixture",
            "seed": "another-plan-seed",
        }
        self.assertEqual(set(changes), set(asdict(self.policy)))
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.record.validate(self.identity, replace(self.policy, **{field: value}))

    def test_changed_policy_cannot_reuse_even_when_record_id_is_unchanged(self):
        widened = replace(self.policy, absolute_tolerance=1e100,
                          relative_tolerance=1e100, probe_count=1,
                          head_guard=0, tail_guard=0)
        self.assertEqual(widened.calibration_id, self.record.record_id)
        with self.assertRaises(ValidationError):
            self.record.validate(self.identity, widened)

    def test_every_record_field_is_protected_by_checksum(self):
        for field in asdict(self.record):
            value = "another-id" if field == "record_id" else "f" * 64
            damaged = replace(self.record, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValidationError):
                damaged.validate(self.identity, self.policy)

    def test_evidence_changes_record_checksum(self):
        other = CalibrationRecord.bind(self.identity, self.policy, "f" * 64)
        self.assertNotEqual(other.checksum, self.record.checksum)
        self.assertEqual(other.scope_sha256, self.record.scope_sha256)
        self.assertEqual(other.policy_sha256, self.record.policy_sha256)
        # Both are syntactically valid declarations; neither proves evidence.
        other.validate(self.identity, self.policy)

    def test_record_and_digests_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.record.evidence_sha256 = "f" * 64

    def test_bind_requires_typed_contracts_and_calibration_id(self):
        for bad_identity, bad_policy in ((None, self.policy),
                                         (asdict(self.identity), self.policy),
                                         (self.identity, None),
                                         (self.identity, asdict(self.policy)),
                                         (self.identity, GatePolicy())):
            with self.subTest(identity=bad_identity, policy=bad_policy):
                with self.assertRaises(ValidationError):
                    CalibrationRecord.bind(bad_identity, bad_policy, "e" * 64)

    def test_invalid_record_ids_and_digests_rejected(self):
        for bad in ("", " ", None, 3):
            with self.subTest(record_id=bad), self.assertRaises(ValidationError):
                replace(self.record, record_id=bad)
        for field in ("scope_sha256", "policy_sha256", "evidence_sha256", "checksum"):
            for bad in ("", "F" * 64, "e" * 63, "g" * 64, None, True):
                with self.subTest(field=field, value=bad), self.assertRaises(ValidationError):
                    replace(self.record, **{field: bad})
        for bad in ("", None, "NOT-A-DIGEST"):
            with self.subTest(evidence=bad), self.assertRaises(ValidationError):
                CalibrationRecord.bind(self.identity, self.policy, bad)

    def test_validation_rechecks_corrupted_policy_and_record(self):
        corrupted_policy = replace(self.policy)
        object.__setattr__(corrupted_policy, "absolute_tolerance", float("nan"))
        with self.assertRaises(ValidationError):
            self.record.validate(self.identity, corrupted_policy)
        corrupted_record = replace(self.record)
        object.__setattr__(corrupted_record, "evidence_sha256", None)
        with self.assertRaises(ValidationError):
            corrupted_record.validate(self.identity, self.policy)

    def test_bind_does_not_claim_evidence_authenticity_or_accuracy(self):
        # Even a zero digest is syntactically valid: binding is not provenance.
        record = CalibrationRecord.bind(self.identity, self.policy, "0" * 64)
        record.validate(self.identity, self.policy)
        self.assertFalse(hasattr(record, "accuracy_certified"))
        self.assertFalse(hasattr(record, "evidence_authenticated"))

    def test_clean_import_has_no_gpu_or_array_dependencies(self):
        code = ("import sys; import kv_cache_engram.calibration; "
                "assert not any(n.split('.')[0] in ('torch','numpy','triton','deepinfer') "
                "for n in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
