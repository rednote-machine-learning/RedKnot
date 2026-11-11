"""Scalar CPU fixtures for the hook contract, NOT native numerical/quality tests."""
from array import array
from dataclasses import replace
import importlib.util
import math
from types import SimpleNamespace
import unittest

from kv_cache_engram.core import ValidationError, token_digest
from kv_cache_engram.gate import GatePolicy
from kv_cache_engram.native_reuse import (
    CHUNK, DOC, LAYERS, NativeCacheIdentity, NativeCalibrationRecord,
    NativeReuseOps, KVCacheEngramHook, document_plan, frame_parts, probe_plan,
)


class Tensor:
    """A scalar-per-row stand-in, intentionally NOT a width-512 model tensor."""
    def __init__(self, values=None, *, storage=None, start=0, end=None):
        self.storage = storage if storage is not None else [array("d", values), 0]
        self.start = start
        self.end = len(self.storage[0]) if end is None else end

    def __len__(self):
        return self.end-self.start

    def __iter__(self):
        return iter(self.storage[0][self.start:self.end])

    def __getitem__(self, index):
        return self.storage[0][self.start+index]


class FakeOps:
    backend = "scalar-cpu-contract-fixture-not-model"
    native_execution = False

    def __init__(self):
        self.sync_calls = 0
        self.fail_barrier = False
        self.split_delta = 0
        self.gather_delta = 0

    def make_projections(self, module):
        return (lambda x: Tensor(v+100 for v in self._raw(x)),
                lambda x: Tensor(v+self.split_delta for v in self._raw(x)))

    def _raw(self, value):
        return value[0] if type(value) is tuple else value

    def rows(self, hidden):
        return len(self._raw(hidden))

    def validate(self, module, metadata, cross, marker):
        assert metadata.is_prompt and metadata.num_reqs == 1
        assert metadata.canonical_request_binding == marker
        assert metadata.total_q == marker["end"]-marker["start"]
        assert cross.consumer_full_metadata is None

    def validate_full_view(self, metadata, marker):
        if not metadata.normal_full_view:
            raise ValidationError("masked full view")

    def take_hidden(self, hidden, positions):
        if type(hidden) is tuple:
            return tuple(self.take(value, positions) for value in hidden)
        return Tensor(hidden[p]+self.gather_delta for p in positions)

    def validate_tensor(self, value, rows=None):
        if not isinstance(value, Tensor) or (rows is not None and len(value) != rows):
            raise ValidationError("fixture row shape")
        if not all(math.isfinite(v) for v in value):
            raise ValidationError("fixture nonfinite")

    def normalize(self, module, value):
        self.validate_tensor(value)
        return Tensor(v/4 for v in value)

    def allocate(self, rows, like):
        return Tensor([float("nan")]*rows)

    def copy(self, destination, value):
        assert len(destination) == len(value)
        destination.storage[0][destination.start:destination.end] = array("d", value)
        destination.storage[1] += 1

    def take(self, value, positions):
        return Tensor(value[p] for p in positions)

    def scatter(self, destination, positions, values):
        assert len(positions) == len(values)
        for p, v in zip(positions, values):
            destination.storage[0][destination.start+p] = v
        destination.storage[1] += 1

    def slice(self, value, start, end):
        return Tensor(storage=value.storage, start=value.start+start, end=value.start+end)

    def version(self, value):
        return value.storage[1]

    def postprocess(self, module, canonical, metadata):
        self.validate_tensor(canonical, metadata.total_q)
        return Tensor(canonical), None

    def errors(self, current, cached, floor):
        self.validate_tensor(current)
        self.validate_tensor(cached, len(current))
        return tuple((abs(a-b)/max(abs(a), floor), abs(a-b)) for a, b in zip(current, cached))

    def identity_error(self, actual, expected):
        if len(actual) != len(expected):
            raise ValidationError("identity shape")
        error = max(abs(a-b) for a, b in zip(actual, expected))
        if error != 0 or not math.isfinite(error):
            raise ValidationError("split/gather differs")
        return error

    def prepared_tensor(self, value):
        assert value[1] is None
        return value[0]

    def synchronize(self):
        self.sync_calls += 1
        if self.fail_barrier:
            raise RuntimeError("asynchronous tail failure")


class Module:
    def __init__(self, layer, ops):
        self.layer_idx, self.ops = layer, ops
        self.last = None
        self.attention_rows = self.moe_fixture_rows = 0

    def _project_q_and_swa(self, hidden_states, shared_hidden_bf16):
        raw = self.ops._raw(hidden_states)
        return Tensor(v+100 for v in raw), Tensor(raw)

    def _prepare_swa_kv(self, kv_swa, swa_cache, input_metadata, use_cp):
        return self.ops.postprocess(self, self.ops.normalize(self, kv_swa), input_metadata)

    def forward(self, hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime):
        q, kv = self._project_q_and_swa(hidden_states, None)
        prepared = self._prepare_swa_kv(kv, None, input_metadata, False)
        self.last = q, prepared[0]
        self.attention_rows += len(q)
        # This fixture count only tests that the full forward is reached.
        self.moe_fixture_rows += len(q)
        return self.last


def policy(**changes):
    return GatePolicy(enabled=True, probe_count=32, relative_tolerance=.001,
        absolute_tolerance=.005, calibration_id="explicit-test-only", **changes)


def inputs(prefix=27):
    ids = [999]*prefix
    docs = []
    for index in range(8):
        values = [index+1]*DOC
        start = len(ids)
        ids.extend(values)
        sha = token_digest(values)
        docs.append(dict(input_start=start, input_end=len(ids), token_ids_sha256=sha,
                         document_id=sha, offline_position=0))
    ids.extend([998]*3)
    return ids, docs


class NativeReuseTests(unittest.TestCase):
    def setUp(self):
        self.ops = FakeOps()
        self.modules = [Module(layer, self.ops) for layer in LAYERS]
        self.hook = KVCacheEngramHook(self.modules, model_sha256="a"*64,
            tokenizer_sha256="b"*64, execution_sha256="c"*64,
            ops=self.ops, enabled=True).install()
        self.addCleanup(self.hook.close)

    def call(self, start, end, *, values=None, layer=None, mutate=None):
        metadata = SimpleNamespace(canonical_request_binding=self.hook.chunk_marker(start, end),
            total_q=end-start, num_reqs=1, is_prompt=True, normal_full_view=True)
        if mutate:
            mutate(metadata)
        hidden = Tensor(self.hook.ids[start:end] if values is None else values)
        cross = SimpleNamespace(consumer_full_metadata=None)
        for module in self.modules:
            if layer is None or module.layer_idx == layer:
                module.forward(hidden, None, metadata, hidden, cross)

    def capture(self, value=1):
        self.hook.reset("capture-"+str(value), "capture_independent", input_ids=[value]*DOC, phase="offline")
        for start in range(0, DOC, CHUNK):
            self.call(start, start+CHUNK)
        self.hook.finish()

    def seed_bank(self, docs, *, layers=LAYERS, delta=0):
        # Explicitly fabricated CPU evidence, never NativeReuseOps or GPU labels.
        for index, doc in enumerate(docs):
            for layer in layers:
                tensor = Tensor([(index+1)/4+delta]*DOC)
                self.hook.bank[(doc["document_id"], layer)] = dict(tensor=tensor, cursor=DOC,
                    complete=True, version=self.ops.version(tensor),
                    identity=self.hook._identity(doc["document_id"], layer))

    def start_gated(self, *, delta=0, calibrated=True, prefix=27):
        ids, docs = inputs(prefix)
        self.seed_bank(docs, layers=(3,), delta=delta)
        p = policy()
        identity = self.hook._identity(docs[0]["document_id"], 3)
        records = {3: NativeCalibrationRecord.bind(identity, p, "d"*64)} if calibrated else {}
        # CPU scalar fixtures do NOT qualify native numerics. This deliberate
        # fixture flag isolates the runtime-gate tests from the full pilot pass.
        self.hook.identity_qualified = True
        self.hook.reset("gated", "gated", input_ids=ids, documents=docs, phase="test",
                        policies={3: p}, calibrations=records)
        return p, docs

    def test_capture_full_coverage_and_labels(self):
        self.capture()
        report = self.hook.report()
        self.assertTrue(report["complete"])
        self.assertFalse(report["native_execution_declared"])
        self.assertEqual(report["backend"], FakeOps.backend)
        self.assertFalse(report["identity_exact"])
        self.assertEqual(len(list(self.hook.iter_cache_entries())), 14)
        self.assertEqual(set(self.hook.cache_identities()), set(LAYERS))
        self.assertTrue(all(x.storage_dtype == "bfloat16-native-canonical"
                            for x in self.hook.cache_identities().values()))
        self.assertEqual(self.ops.sync_calls, 1)
        self.hook.report()
        self.assertEqual(self.ops.sync_calls, 1, "report must not repeat device synchronization")

    def test_capture_may_not_overwrite(self):
        self.capture()
        with self.assertRaises(ValidationError):
            self.hook.reset("again", "capture_independent", input_ids=[1]*DOC, phase="offline")

    def test_capture_sink_failure_poisons_no_publish(self):
        def fail(*args):
            raise OSError("sink failed")
        self.hook.sink = fail
        self.hook.reset("bad", "capture_independent", input_ids=[1]*DOC, phase="offline")
        with self.assertRaises(OSError):
            self.call(0, CHUNK, layer=3)
        self.assertEqual(self.hook.state, "ABORTED")
        with self.assertRaises(ValidationError):
            self.hook.report()
        with self.assertRaises(ValidationError):
            list(self.hook.iter_cache_entries())

    def test_tail_barrier_failure_never_publishes(self):
        self.ops.fail_barrier = True
        with self.assertRaises(RuntimeError):
            self.capture()
        self.assertEqual(self.hook.state, "ABORTED")
        self.assertTrue(all(not v["complete"] for v in self.hook.bank.values()))

    def test_incomplete_coverage_fails(self):
        self.hook.reset("short", "capture_independent", input_ids=[1]*DOC, phase="offline")
        self.call(0, CHUNK)
        with self.assertRaises(ValidationError):
            self.hook.finish()
        self.assertEqual(self.hook.state, "ABORTED")

    def test_successful_gating_keeps_full_q_and_forward(self):
        self.start_gated(delta=.0001)
        self.call(0, CHUNK)
        counts = self.hook.counts[3]
        self.assertGreater(counts["reused_kv_rows"], 0)
        self.assertEqual(counts["online_kv_rows"]+counts["reused_kv_rows"], CHUNK)
        self.assertEqual(counts["q_rows"], CHUNK)
        self.assertEqual(counts["split_q_rows"], CHUNK)
        module = self.modules[0]
        self.assertEqual(module.attention_rows, CHUNK)
        self.assertEqual(module.moe_fixture_rows, CHUNK)
        q, kv = module.last
        self.assertEqual(tuple(q), tuple(v+100 for v in self.hook.ids[:CHUNK]))
        # Prefix, first 128 document rows, and the trailing cross-chunk fragment
        # are always online. Accepted candidates alone contain the cached delta.
        self.assertTrue(all(kv[i] == self.hook.ids[i]/4 for i in range(27+128)))
        self.assertTrue(all(kv[i] == self.hook.ids[i]/4 for i in range(27+15*512, CHUNK)))
        self.assertGreater(sum(kv[i] != self.hook.ids[i]/4 for i in range(CHUNK)), 0)
        self.assertTrue(all(e["current_hybrid_hidden_probes"] for e in self.hook.events))

    def test_deny_supplements_all_missing_kv(self):
        self.start_gated(delta=1)
        self.call(0, CHUNK, layer=3)
        self.assertEqual(self.hook.counts[3]["reused_kv_rows"], 0)
        self.assertGreater(self.hook.counts[3]["fallback_rows"], 0)
        self.assertEqual(self.hook.counts[3]["online_kv_rows"], CHUNK)
        self.assertEqual(tuple(self.modules[0].last[1]), tuple(v/4 for v in self.hook.ids[:CHUNK]))

    def test_no_calibration_stays_native(self):
        self.start_gated(calibrated=False)
        self.call(0, CHUNK, layer=3)
        counts = self.hook.counts[3]
        self.assertEqual(counts["native_fused_rows"], CHUNK)
        self.assertEqual(counts["split_q_rows"], 0)
        self.assertEqual(counts["reused_kv_rows"], 0)

    def test_stale_calibration_poisoned_before_reuse(self):
        self.start_gated()
        self.hook.policies[3] = replace(self.hook.policies[3], seed="changed")
        with self.assertRaises(ValidationError):
            self.call(0, CHUNK, layer=3)
        self.assertEqual(self.hook.state, "ABORTED")

    def test_cache_mutation_detected_before_projection(self):
        self.start_gated()
        entry = next(iter(self.hook.bank.values()))
        entry["tensor"].storage[1] += 1
        with self.assertRaises(ValidationError):
            self.call(0, CHUNK, layer=3)
        self.assertEqual(self.hook.counts[3]["split_q_rows"], 0)

    def test_probe_measured_from_current_hidden_not_reference(self):
        self.start_gated(delta=0)
        values = [v+1 for v in self.hook.ids[:CHUNK]]
        self.call(0, CHUNK, values=values, layer=3)
        self.assertEqual(self.hook.counts[3]["reused_kv_rows"], 0)
        self.assertEqual(tuple(self.modules[0].last[1]), tuple(v/4 for v in values))

    def test_split_identity_includes_sparse_gathers(self):
        ids, docs = inputs()
        self.hook.reset("identity", "split_identity", input_ids=ids, documents=docs, phase="qualification")
        self.call(0, CHUNK, layer=3)
        errors = self.hook.qualifications[0]["errors"]
        self.assertEqual(set(errors), {"q", "kv_raw", "kv_canonical", "gather32", "gather128", "post_rope_cast",
            "post_rope_gather32", "post_rope_gather128", "post_rope_gather384", "post_rope_gather512"})
        self.assertEqual(set(errors.values()), {0})

    def test_bad_sparse_gather_aborts_identity(self):
        ids, docs = inputs()
        self.ops.gather_delta = .125
        self.hook.reset("identity", "split_identity", input_ids=ids, documents=docs, phase="qualification")
        with self.assertRaises(ValidationError):
            self.call(0, CHUNK, layer=3)
        self.assertEqual(self.hook.state, "ABORTED")

    def test_full_fixture_qualification_latch_requires_all_layers_and_barrier(self):
        ids, docs = inputs()
        self.hook.reset("full-identity", "split_identity", input_ids=ids, documents=docs, phase="qualification")
        self.assertFalse(self.hook.identity_qualified)
        for start in range(0, len(ids), CHUNK):
            self.call(start, min(start+CHUNK, len(ids)))
        self.assertFalse(self.hook.identity_qualified, "Python forward completion alone is not a qualification")
        self.hook.finish()
        self.assertTrue(self.hook.identity_qualified)
        report = self.hook.report()
        self.assertTrue(report["identity_exact"])
        self.assertFalse(report["native_execution_declared"], "fixture success must not claim native execution")
        self.assertEqual(report["coverage"], {str(layer): len(ids) for layer in LAYERS})
        self.assertEqual(len(report["qualification"]), len(LAYERS)*((len(ids)+CHUNK-1)//CHUNK))
        self.hook.clear_cache()
        self.assertTrue(self.hook.identity_qualified)

    def test_foreign_request_binding_and_mask_poison(self):
        self.start_gated()
        with self.assertRaises(ValidationError):
            self.call(0, CHUNK, layer=3, mutate=lambda m: setattr(m, "normal_full_view", False))
        self.assertEqual(self.hook.state, "ABORTED")

    def test_foreign_request_marker_poison(self):
        self.start_gated()
        def mutate(m):
            m.canonical_request_binding["request_id"] = "foreign"
        with self.assertRaises(ValidationError):
            self.call(0, CHUNK, layer=3, mutate=mutate)

    def test_default_disabled_reuse(self):
        self.hook.enabled = False
        ids, docs = inputs()
        with self.assertRaises(ValidationError):
            self.hook.reset("no", "gated", input_ids=ids, documents=docs, phase="no")
        self.hook.reset("yes", "native", input_ids=ids, documents=docs, phase="baseline")
        self.call(0, CHUNK, layer=3)
        self.assertEqual(self.hook.counts[3]["native_fused_rows"], CHUNK)

    def test_unqualified_enabled_policy_is_rejected(self):
        ids, docs = inputs()
        with self.assertRaisesRegex(ValidationError, "qualification"):
            self.hook.reset("unqualified", "gated", input_ids=ids, documents=docs,
                            phase="test", policies={3: policy()})

    def test_native_baseline_does_not_call_engram_gpu_checks(self):
        ids, docs = inputs()
        self.hook.reset("standard", "native", input_ids=ids, documents=docs, phase="baseline")
        def fail(*args):
            raise AssertionError("Engram GPU validation must not inflate baseline")
        self.ops.validate = fail
        self.ops.validate_full_view = fail
        # Original fixture preparation uses the same FakeOps.validate_tensor;
        # count its native call only. Engram must not add a third validation.
        calls, original = [], self.ops.validate_tensor
        def observe(*args):
            calls.append(args)
            return original(*args)
        self.ops.validate_tensor = observe
        self.call(0, CHUNK, layer=3)
        self.assertEqual(len(calls), 2)

    def test_shadow_oracle_excludes_online_guards_and_probes(self):
        ids, docs = inputs(prefix=0)
        self.seed_bank(docs)
        p = policy()
        entry = self.hook.bank[(docs[0]["document_id"], 3)]
        # Corrupt only a guaranteed online guard. Full-row oracle sees the error,
        # but candidate-reuse oracle correctly excludes the online-overwritten row.
        entry["tensor"].storage[0][0] = 10
        entry["version"] = self.ops.version(entry["tensor"])
        self.hook.reset("shadow", "shadow_full", input_ids=ids, documents=docs,
                        phase="calibration", candidate_policies=[p])
        self.call(0, CHUNK, layer=3)
        event = self.hook.events[0]
        self.assertTrue(event["accepted"])
        self.assertEqual(event["oracle_bad_rows"], 1)
        self.assertEqual(event["bad_candidate_rows"], 0)
        self.assertEqual(event["actual_reused_rows"], 0)
        self.assertGreater(event["would_reuse_rows"], 0)
        self.assertEqual(tuple(self.modules[0].last[1]), tuple(v/4 for v in ids[:CHUNK]))

    def test_shadow_false_accept_detects_unprobed_bad_row(self):
        ids, docs = inputs(prefix=0)
        self.seed_bank(docs)
        p = policy()
        entry = self.hook.bank[(docs[0]["document_id"], 3)]
        probes, guards = probe_plan(entry["identity"], 0, 512, p)
        bad = next(i for i in range(512) if i not in set(probes+guards))
        entry["tensor"].storage[0][bad] = 10
        self.hook.reset("shadow", "shadow_full", input_ids=ids, documents=docs,
                        phase="calibration", candidate_policies=[p])
        self.call(0, CHUNK, layer=3)
        event = self.hook.events[0]
        self.assertTrue(event["accepted"])
        self.assertEqual(event["oracle_bad_reused_rows"], 1)
        self.assertEqual(event["actual_reused_rows"], 0)

    def test_full_fixture_shadow_partitions_every_document_layer(self):
        ids, docs = inputs()
        self.seed_bank(docs)
        self.hook.reset("full-shadow", "shadow_full", input_ids=ids, documents=docs,
                        phase="calibration", candidate_policies=[policy()])
        for start in range(0, len(ids), CHUNK):
            self.call(start, min(start+CHUNK, len(ids)))
        self.hook.finish()
        report = self.hook.report()
        self.assertFalse(report["native_execution_declared"])
        groups = {(layer, doc["document_id"]): [] for layer in LAYERS for doc in docs}
        full_digest = token_digest(ids)
        for event in report["events"]:
            groups[(event["layer"], event["document_sha256"])].append(event)
            self.assertEqual(event["actual_reused_rows"], 0)
            self.assertEqual(event["input_sha256"], full_digest)
            self.assertEqual(event["oracle_bad_reused_rows"], 0)
        for events in groups.values():
            cursor = 0
            self.assertTrue(any(event["mode"] == "full_recompute" for event in events))
            for event in events:
                self.assertEqual(event["document_local_start"], cursor)
                cursor = event["document_local_end"]
                if not event["eligible_for_reuse"]:
                    self.assertFalse(event["accepted"])
                    self.assertEqual(event["candidate_reusable_rows"], 0)
            self.assertEqual(cursor, DOC)

    def test_foreign_patch_refuses_install(self):
        other = Module(3, self.ops)
        other.forward = lambda *a: None
        replacement = [other]+self.modules[1:]
        h = KVCacheEngramHook(replacement, model_sha256="a"*64, tokenizer_sha256="b"*64,
                             execution_sha256="c"*64, ops=self.ops)
        with self.assertRaises(ValidationError):
            h.install()

    def test_clear_only_between_complete_requests(self):
        self.capture()
        self.hook.clear_cache()
        self.assertEqual(len(self.hook.bank), 0)
        self.hook.reset("active", "capture_independent", input_ids=[2]*DOC, phase="offline")
        with self.assertRaises(ValidationError):
            self.hook.clear_cache()


class PlannerTests(unittest.TestCase):
    def test_native_unweighted_rmsnorm_passes_none_without_rope(self):
        # Pure dependency-injection fixture; no torch, model, or native kernel.
        ops = NativeReuseOps()
        ops.validate_tensor = lambda *args: None
        ops.NormArgs = lambda weight, epsilon: (weight, epsilon)
        calls = []
        def transform(value, **kwargs):
            calls.append(kwargs)
            return value
        ops.transform = transform
        raw = SimpleNamespace(shape=(4, 512))
        module = SimpleNamespace(kv_swa_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6, out_scale=1.0))
        self.assertIs(ops.normalize(module, raw), raw)
        self.assertEqual(calls, [dict(norm_args=(None, 1e-6), rope_args=None, cast_args=None, out_args=None)])
        module.kv_swa_norm.variance_epsilon = float("nan")
        with self.assertRaises(ValidationError):
            ops.normalize(module, raw)

    def test_fragments_partition_document_without_promoting_half_blocks(self):
        ids, docs = inputs()
        plan = document_plan(ids, docs)
        pieces = {p["key"]: [] for p in plan}
        for start in range(0, len(ids), CHUNK):
            for part in frame_parts(plan, start, min(start+CHUNK, len(ids))):
                pieces[part["document"]["key"]].append(part)
                if part["eligible"]:
                    self.assertEqual(part["start"] % 512, 0)
                    self.assertEqual(part["end"]-part["start"], 512)
        for parts in pieces.values():
            cursor = 0
            self.assertTrue(any(not part["eligible"] for part in parts))
            for part in parts:
                self.assertEqual(part["start"], cursor)
                cursor = part["end"]
            self.assertEqual(cursor, DOC)

    def test_probe_plan_is_deterministic_and_guards_disjoint(self):
        identity = NativeCacheIdentity("a"*64, "b"*64, "c"*64, "d"*64, 3)
        p = policy()
        first = probe_plan(identity, 0, 512, p)
        self.assertEqual(first, probe_plan(identity, 0, 512, p))
        self.assertEqual(first[1], tuple(range(128)))
        self.assertTrue(set(first[0]).isdisjoint(first[1]))
        self.assertEqual(len(first[0]), 32)
        self.assertNotEqual(first[0], probe_plan(identity, 0, 512, replace(p, seed="different"))[0])

    def test_native_identity_never_accepts_cpu_reference_storage(self):
        with self.assertRaises(ValidationError):
            NativeCacheIdentity("a"*64, "b"*64, "c"*64, "d"*64, 3,
                                storage_dtype="float64-cpu-reference")

    def test_calibration_scope_excludes_document_but_not_layer(self):
        identity = NativeCacheIdentity("a"*64, "b"*64, "c"*64, "d"*64, 3)
        p = policy()
        record = NativeCalibrationRecord.bind(identity, p, "e"*64)
        record.validate(replace(identity, tokens_sha256="f"*64), p)
        with self.assertRaises(ValidationError):
            record.validate(replace(identity, layer=4), p)
        with self.assertRaises(ValidationError):
            replace(record, checksum="0"*64).validate(identity, p)

    def test_changed_document_hash_rejected(self):
        ids, docs = inputs()
        ids[100] += 1
        with self.assertRaises(ValidationError):
            document_plan(ids, docs)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU torch unavailable")
class TorchGatherTests(unittest.TestCase):
    def test_packed_sf_token_rows_preserved_for_noncontiguous_gather(self):
        import torch
        # Direct dependency-fixture construction; never from_deepinfer/GPU.
        ops = NativeReuseOps()
        ops.torch = torch
        x = torch.arange(257*8, dtype=torch.float32).reshape(257, 8)
        sf = torch.arange(257*4, dtype=torch.int32).reshape(4, 257).t()
        positions = (0, 2, 5, 17, 256)
        out = ops.take_hidden((x, sf), positions)
        self.assertTrue(torch.equal(out[0], x[list(positions)]))
        self.assertTrue(torch.equal(out[1], sf[list(positions)]))
        self.assertEqual(tuple(out[1].shape), (5, 4))


if __name__ == "__main__":
    unittest.main()
