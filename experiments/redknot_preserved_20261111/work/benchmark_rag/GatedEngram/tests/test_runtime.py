"""Standard-library CPU tests for the transaction's side-effect boundary."""
import dataclasses
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest import mock

from kv_cache_engram.cache import CacheArtifact
from kv_cache_engram.calibration import CalibrationRecord
from kv_cache_engram.core import Binding, CacheIdentity, OnlineRows, ValidationError
from kv_cache_engram.gate import GatePolicy, make_plan
from kv_cache_engram.runtime import (
    BlockTransaction, PreparedBlock, ProbeFailure, TransactionStateError,
)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.identity = CacheIdentity(
            model_sha256="a" * 64, tokenizer_sha256="b" * 64,
            execution_sha256="c" * 64, tokens_sha256="d" * 64,
            layer=1, token_count=16, width=2,
        )
        self.binding = Binding(
            request_id="request-A", phase="quality", identity=self.identity,
            document_start=100, block_start=0, block_end=8,
            request_sha256="e" * 64,
        )
        self.rows = tuple((float(i), float(i + 1)) for i in range(16))
        self.cache = CacheArtifact.build(self.identity, self.rows)
        self.policy = GatePolicy(
            enabled=True, block_size=8, probe_count=2, head_guard=1,
            tail_guard=1, relative_tolerance=0.0, absolute_tolerance=0.0,
            calibration_id="unit-test-declaration-not-accuracy-certification",
        )
        self.calibration = CalibrationRecord.bind(self.identity, self.policy, "f" * 64)
        self.calls = []

    def compute(self, binding, positions):
        self.calls.append((binding, positions))
        return OnlineRows(binding.sha256, positions,
                          tuple(self.rows[p] for p in positions))

    def transaction(self, cache="default", policy=None, compute=None, calibration="default"):
        selected_policy = self.policy if policy is None else policy
        if calibration == "default":
            # Tests explicitly bind their synthetic evidence to the selected
            # policy; production runtime never creates calibration records.
            calibration = (CalibrationRecord.bind(self.identity, selected_policy, "f" * 64)
                           if selected_policy.enabled else None)
        return BlockTransaction(
            self.binding, self.cache if cache == "default" else cache,
            selected_policy,
            self.compute if compute is None else compute,
            calibration=calibration,
        )

    def assert_aborted(self, tx):
        self.assertEqual(tx.state, "ABORTED")
        with self.assertRaises(TransactionStateError):
            tx.prepare()
        with self.assertRaises(TransactionStateError):
            tx.commit(lambda *_: self.fail("must not submit"), self.binding)

    def test_prepare_reuses_but_never_submits_and_preserves_cache(self):
        original = (self.cache.rows, self.cache.checksum)
        tx = self.transaction()
        self.assertEqual(tx.state, "NEW")
        prepared = tx.prepare()
        self.assertEqual(tx.state, "PREPARED")
        self.assertEqual(prepared.mode, "gated_reuse")
        self.assertEqual(prepared.rows, self.rows[:8])
        self.assertTrue(prepared.cached_positions)
        self.assertEqual((self.cache.rows, self.cache.checksum), original)
        self.assertEqual(set(prepared.online_positions) | set(prepared.cached_positions),
                         set(self.binding.positions))
        self.assertFalse(set(prepared.online_positions) & set(prepared.cached_positions))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            prepared.mode = "forged"
        with self.assertRaises(TypeError):
            prepared.rows[0][0] = 999.0

    def test_default_policy_fully_recomputes(self):
        tx = self.transaction(policy=GatePolicy())
        prepared = tx.prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertEqual(prepared.fallback_reason, "policy_disabled")
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])
        self.assertFalse(prepared.cached_positions)

    def test_enabled_uncalibrated_policy_is_rejected(self):
        with self.assertRaises(ValidationError):
            GatePolicy(enabled=True, relative_tolerance=0.0, absolute_tolerance=0.0)

    def test_enabled_with_only_calibration_id_fully_recomputes(self):
        tx = self.transaction(calibration=None)
        prepared = tx.prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertEqual(prepared.fallback_reason, "calibration_missing")
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])

    def test_untyped_calibration_fully_recomputes_before_probe(self):
        prepared = self.transaction(calibration={"calibration_id": self.policy.calibration_id}).prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertTrue(prepared.fallback_reason.startswith("calibration_invalid:"))
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])

    def test_calibration_identity_mismatch_fully_recomputes_before_probe(self):
        different = dataclasses.replace(self.identity, layer=2)
        calibration = CalibrationRecord.bind(different, self.policy, "f" * 64)
        prepared = self.transaction(calibration=calibration).prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertTrue(prepared.fallback_reason.startswith("calibration_invalid:"))
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])

    def test_calibration_policy_mismatch_fully_recomputes_before_probe(self):
        policy = dataclasses.replace(self.policy, probe_count=self.policy.probe_count + 1)
        prepared = self.transaction(policy=policy, calibration=self.calibration).prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertTrue(prepared.fallback_reason.startswith("calibration_invalid:"))
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])

    def test_calibration_validator_exception_fully_recomputes_before_probe(self):
        with mock.patch.object(CalibrationRecord, "validate", side_effect=ValidationError("bad evidence")):
            prepared = self.transaction(calibration=self.calibration).prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertTrue(prepared.fallback_reason.startswith("calibration_invalid:"))
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])

    def test_disabled_policy_does_not_read_calibration(self):
        with mock.patch.object(CalibrationRecord, "validate", side_effect=AssertionError("must not validate")):
            prepared = self.transaction(policy=GatePolicy(), calibration=self.calibration).prepare()
        self.assertEqual(prepared.fallback_reason, "policy_disabled")

    def test_missing_cache_fully_recomputes(self):
        prepared = self.transaction(cache=None).prepare()
        self.assertEqual(prepared.fallback_reason, "cache_missing")
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])

    def test_wrong_cache_identity_falls_back_without_reuse(self):
        other = dataclasses.replace(self.identity, layer=2)
        prepared = self.transaction(cache=CacheArtifact.build(other, self.rows)).prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertTrue(prepared.fallback_reason.startswith("cache_invalid:"))
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])

    def test_corrupt_cache_falls_back(self):
        with mock.patch.object(CacheArtifact, "validate", side_effect=ValidationError("checksum")):
            prepared = self.transaction().prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])

    def test_cache_take_failure_falls_back(self):
        with mock.patch.object(CacheArtifact, "take", side_effect=ValueError("broken rows")):
            prepared = self.transaction().prepare()
        self.assertEqual(prepared.mode, "full_recompute")

    def test_malformed_cache_read_falls_back(self):
        with mock.patch.object(CacheArtifact, "take", return_value=((1.0,),) * 8):
            prepared = self.transaction().prepare()
        self.assertEqual(prepared.mode, "full_recompute")

    def test_numeric_probe_rejection_fully_recomputes(self):
        def changed(binding, positions):
            self.calls.append((binding, positions))
            return OnlineRows(binding.sha256, positions,
                              tuple((self.rows[p][0] + 10.0, self.rows[p][1]) for p in positions))
        prepared = self.transaction(compute=changed).prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertEqual(prepared.fallback_reason, "threshold_exceeded")
        self.assertEqual(self.calls[-1], (self.binding, self.binding.positions))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(prepared.online_positions, self.binding.positions)

    def test_explicit_probe_failure_allows_full_recompute(self):
        def unavailable(binding, positions):
            self.calls.append((binding, positions))
            if len(self.calls) == 1:
                raise ProbeFailure("read-only probe unavailable")
            return OnlineRows(binding.sha256, positions, tuple(self.rows[p] for p in positions))
        prepared = self.transaction(compute=unavailable).prepare()
        self.assertEqual(prepared.fallback_reason, "probe_unavailable")
        self.assertEqual(self.calls[-1][1], self.binding.positions)

    def test_ordinary_probe_exception_aborts_without_second_callback(self):
        def broken(binding, positions):
            self.calls.append((binding, positions))
            raise RuntimeError("unknown backend fault")
        tx = self.transaction(compute=broken)
        with self.assertRaises(RuntimeError):
            tx.prepare()
        self.assertEqual(len(self.calls), 1)
        self.assert_aborted(tx)

    def test_wrong_probe_binding_aborts_not_fallback(self):
        def wrong(binding, positions):
            self.calls.append((binding, positions))
            return OnlineRows("e" * 64, positions, tuple(self.rows[p] for p in positions))
        tx = self.transaction(compute=wrong)
        with self.assertRaises(ValidationError):
            tx.prepare()
        self.assertEqual(len(self.calls), 1)
        self.assert_aborted(tx)

    def test_missing_probe_position_aborts(self):
        def missing(binding, positions):
            return OnlineRows(binding.sha256, positions[:-1],
                              tuple(self.rows[p] for p in positions[:-1]))
        tx = self.transaction(compute=missing)
        with self.assertRaises(ValidationError):
            tx.prepare()
        self.assert_aborted(tx)

    def test_wrong_width_aborts(self):
        def wrong_width(binding, positions):
            return OnlineRows(binding.sha256, positions, tuple((1.0,) for _ in positions))
        tx = self.transaction(compute=wrong_width)
        with self.assertRaises(ValidationError):
            tx.prepare()
        self.assert_aborted(tx)

    def test_nonfinite_probe_aborts(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                def nonfinite(binding, positions):
                    return OnlineRows(binding.sha256, positions,
                                      tuple((value, 1.0) for _ in positions))
                tx = self.transaction(compute=nonfinite)
                with self.assertRaises(ValidationError):
                    tx.prepare()
                self.assert_aborted(tx)

    def test_invalid_full_recompute_aborts(self):
        for mode in ("wrong_binding", "wrong_width", "missing", "nonfinite", "exception"):
            with self.subTest(mode=mode):
                def broken(binding, positions):
                    if mode == "exception":
                        raise RuntimeError("full backend failed")
                    selected = positions[:-1] if mode == "missing" else positions
                    rows = tuple(self.rows[p] for p in selected)
                    if mode == "wrong_width":
                        rows = tuple((1.0,) for _ in selected)
                    if mode == "nonfinite":
                        rows = tuple((float("nan"), 1.0) for _ in selected)
                    return OnlineRows("e" * 64 if mode == "wrong_binding" else binding.sha256,
                                      selected, rows)
                tx = self.transaction(cache=None, compute=broken)
                with self.assertRaises((ValidationError, RuntimeError)):
                    tx.prepare()
                self.assert_aborted(tx)

    def test_mandatory_rows_are_fresh_online_even_when_probes_pass(self):
        plan = make_plan(self.binding, self.policy)
        def current(binding, positions):
            self.calls.append((binding, positions))
            return OnlineRows(binding.sha256, positions,
                tuple((900.0, 901.0) if p in plan.mandatory_positions else self.rows[p]
                      for p in positions))
        prepared = self.transaction(compute=current).prepare()
        self.assertEqual(prepared.mode, "gated_reuse")
        by_position = dict(zip(prepared.positions, prepared.rows))
        for p in plan.mandatory_positions:
            self.assertEqual(by_position[p], (900.0, 901.0))
            self.assertIn(p, prepared.online_positions)
            self.assertNotIn(p, prepared.cached_positions)
        self.assertEqual(self.cache.rows, self.rows)

    def test_accepted_probe_rows_override_cache(self):
        policy = dataclasses.replace(self.policy, relative_tolerance=1.0, absolute_tolerance=1.0)
        plan = make_plan(self.binding, policy)
        def current(binding, positions):
            return OnlineRows(binding.sha256, positions,
                              tuple((self.rows[p][0] + 0.25, self.rows[p][1]) for p in positions))
        prepared = self.transaction(policy=policy, compute=current).prepare()
        self.assertEqual(prepared.mode, "gated_reuse")
        by_position = dict(zip(prepared.positions, prepared.rows))
        for p in plan.probe_positions:
            self.assertEqual(by_position[p][0], self.rows[p][0] + 0.25)
            self.assertIn(p, prepared.online_positions)
        for p in prepared.cached_positions:
            self.assertEqual(by_position[p], self.rows[p])

    def test_mandatory_callback_failure_aborts_without_submit(self):
        def broken(binding, positions):
            self.calls.append((binding, positions))
            if len(self.calls) == 2:
                raise ProbeFailure("mandatory is not a probe availability fallback")
            return OnlineRows(binding.sha256, positions, tuple(self.rows[p] for p in positions))
        tx = self.transaction(compute=broken)
        with self.assertRaises(ProbeFailure):
            tx.prepare()
        self.assert_aborted(tx)

    def test_all_mandatory_block_fully_recomputes(self):
        policy = dataclasses.replace(self.policy, head_guard=8, tail_guard=8)
        prepared = self.transaction(policy=policy).prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertEqual(prepared.fallback_reason, "no_reusable_positions")
        self.assertEqual(self.calls, [(self.binding, self.binding.positions)])

    def test_probes_covering_all_nonmandatory_rows_are_not_claimed_as_reuse(self):
        policy = dataclasses.replace(self.policy, probe_count=32)
        plan = make_plan(self.binding, policy)
        prepared = self.transaction(policy=policy).prepare()
        self.assertEqual(prepared.mode, "full_recompute")
        self.assertEqual(prepared.fallback_reason, "all_positions_online")
        self.assertFalse(prepared.cached_positions)
        self.assertEqual(prepared.online_positions, self.binding.positions)
        self.assertEqual(prepared.rows, self.rows[:8])
        # Preserve already-computed online values without a redundant third
        # callback for the full block.
        self.assertEqual(self.calls, [(self.binding, plan.probe_positions),
                                     (self.binding, plan.mandatory_positions)])

    def test_prepare_is_single_use(self):
        tx = self.transaction()
        tx.prepare()
        count = len(self.calls)
        with self.assertRaises(TransactionStateError):
            tx.prepare()
        self.assertEqual(len(self.calls), count)
        self.assertEqual(tx.state, "PREPARED")

    def test_commit_before_prepare_does_not_submit(self):
        tx = self.transaction()
        with self.assertRaises(TransactionStateError):
            tx.commit(lambda *_: self.fail("must not submit"), self.binding)
        self.assertEqual(tx.state, "NEW")

    def test_successful_commit_is_exactly_once_and_exposes_committing_state(self):
        tx = self.transaction()
        prepared = tx.prepare()
        calls = []
        def submit(binding, data):
            self.assertEqual(tx.state, "COMMITTING")
            self.assertEqual(binding, self.binding)
            self.assertEqual(data.binding_sha256, prepared.binding_sha256)
            self.assertEqual(data.positions, prepared.positions)
            self.assertEqual(data.rows, prepared.rows)
            calls.append(data)
            return "submitted"
        self.assertEqual(tx.commit(submit, self.binding), "submitted")
        self.assertEqual(tx.state, "COMMITTED")
        with self.assertRaises(TransactionStateError):
            tx.commit(submit, self.binding)
        with self.assertRaises(TransactionStateError):
            tx.prepare()
        self.assertEqual(len(calls), 1)

    def test_submit_exception_poisoned_even_after_partial_side_effect(self):
        for exception in (RuntimeError("write failed"), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                tx = self.transaction()
                tx.prepare()
                writes = []
                def submit(binding, data):
                    writes.append(data)
                    raise exception
                with self.assertRaises(type(exception)):
                    tx.commit(submit, self.binding)
                self.assertEqual(tx.state, "POISONED")
                with self.assertRaises(TransactionStateError):
                    tx.commit(submit, self.binding)
                with self.assertRaises(TransactionStateError):
                    tx.prepare()
                self.assertEqual(len(writes), 1)

    def test_commit_rejects_cross_request_phase_layer_document_and_block(self):
        wrong = [
            dataclasses.replace(self.binding, request_id="request-B"),
            dataclasses.replace(self.binding, request_sha256="f" * 64),
            dataclasses.replace(self.binding, phase="different-phase"),
            dataclasses.replace(self.binding, identity=dataclasses.replace(self.identity, layer=2)),
            dataclasses.replace(self.binding, document_start=200),
            dataclasses.replace(self.binding, block_start=8, block_end=16),
        ]
        for binding in wrong:
            with self.subTest(binding=binding):
                tx = self.transaction()
                tx.prepare()
                with self.assertRaises(ValidationError):
                    tx.commit(lambda *_: self.fail("must not submit"), binding)
                self.assert_aborted(tx)

    def test_external_prepared_block_cannot_be_submitted_as_binding(self):
        tx = self.transaction()
        prepared = tx.prepare()
        forged = dataclasses.replace(prepared, binding_sha256="e" * 64)
        self.assertIsInstance(forged, PreparedBlock)
        with self.assertRaises(ValidationError):
            tx.commit(lambda *_: self.fail("must not submit"), forged)
        self.assert_aborted(tx)

    def test_reentrant_commit_does_not_double_submit(self):
        tx = self.transaction()
        tx.prepare()
        calls = []
        def submit(binding, data):
            calls.append(data)
            with self.assertRaises(TransactionStateError):
                tx.commit(submit, binding)
        tx.commit(submit, self.binding)
        self.assertEqual(tx.state, "COMMITTED")
        self.assertEqual(len(calls), 1)

    def test_invalid_submitter_aborts_before_side_effect(self):
        tx = self.transaction()
        tx.prepare()
        with self.assertRaises(ValidationError):
            tx.commit(None, self.binding)
        self.assert_aborted(tx)

    def test_concurrent_commit_attempt_cannot_double_submit(self):
        tx = self.transaction()
        tx.prepare()
        entered, release = threading.Event(), threading.Event()
        calls = []
        def submit(binding, data):
            calls.append(data)
            entered.set()
            if not release.wait(2):
                raise AssertionError("test release missing")
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(tx.commit, submit, self.binding)
            try:
                self.assertTrue(entered.wait(2))
                second = pool.submit(tx.commit, submit, self.binding)
                with self.assertRaises(TransactionStateError):
                    second.result(timeout=2)
            finally:
                release.set()
            first.result(timeout=2)
        self.assertEqual(tx.state, "COMMITTED")
        self.assertEqual(len(calls), 1)

    def test_concurrent_prepare_attempt_cannot_duplicate_backend_work(self):
        entered, release = threading.Event(), threading.Event()
        def compute(binding, positions):
            self.calls.append((binding, positions))
            entered.set()
            if not release.wait(2):
                raise AssertionError("test release missing")
            return OnlineRows(binding.sha256, positions, tuple(self.rows[p] for p in positions))
        tx = self.transaction(cache=None, compute=compute)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(tx.prepare)
            try:
                self.assertTrue(entered.wait(2))
                second = pool.submit(tx.prepare)
                with self.assertRaises(TransactionStateError):
                    second.result(timeout=2)
            finally:
                release.set()
            first.result(timeout=2)
        self.assertEqual(tx.state, "PREPARED")
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
