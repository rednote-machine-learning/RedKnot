"""Independent controller contract fixtures, never a native/GPU qualification.

The lifecycle cases need only stdlib. Torch cases explicitly use CPU tensors,
dummy native ops and owned fake adapter hooks; they cannot initialize CUDA.
"""
from contextlib import contextmanager
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import inspect
import math
import sys
import tempfile
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from kv_cache_engram.core import ValidationError, digest, token_digest
from kv_cache_engram.early_mhc import EarlyMHCController, tensor_fingerprint, verify_dependencies
from kv_cache_engram.early_sensitivity import plan_from_scores
from gpu_pilot.early_mhc_protocol import CAPTURE_LAYERS, SPARSE_LAYERS


@dataclass
class DummyPlan:
    full_rows: int
    selected_rows: object
    selected_count: int
    request_id: str
    phase: str
    chunk_start: int
    chunk_end: int
    plan_sha256: str
    request_sha256: str
    active_positions: tuple
    host_layer_plan: object


class DummyAdapter:
    def __init__(self, *args, **kwargs):
        self.pending, self.poisoned = None, False
        self.events = []
        self.counts = dict(full_rows=0, selected_rows=0)
        self.closed = False

    def install(self):
        return self

    def report(self):
        if self.pending is not None or self.poisoned:
            raise ValidationError("incomplete dummy adapter")
        return dict(events=list(self.events), counts=dict(self.counts))

    def close(self, allow_incomplete=False):
        self.closed = True


def controller(root="/unused-cpu-fixture"):
    value = EarlyMHCController([SimpleNamespace(layer_idx=i) for i in range(40)],
        model_sha256="a"*64, execution_sha256="b"*64,
        tokenizer_sha256="c"*64, dependency_root=root)
    value.adapter = DummyAdapter()
    value.state = "IDLE"
    return value


class LifecycleTests(unittest.TestCase):
    def test_constructor_geometry_and_identity_fail_closed(self):
        for layers, model in (([SimpleNamespace(layer_idx=0)]*40, "a"*64),
                              ([SimpleNamespace(layer_idx=i) for i in range(40)], "bad")):
            with self.assertRaises(ValidationError):
                EarlyMHCController(layers, model_sha256=model, execution_sha256="b"*64,
                    tokenizer_sha256="c"*64, dependency_root="/unused")

    def test_reset_qualification_immutable_reference_and_request_lifecycle(self):
        c = controller()
        c.reset("r", "qualified_native", input_ids=[1]*16384, phase="qual")
        self.assertEqual(c.state, "RUNNING")
        self.assertEqual(c.reference_input, token_digest([1]*16384))
        with self.assertRaises(ValidationError):
            c.reset("r2", "qualified_native", input_ids=[1]*16384, phase="qual")
        with self.assertRaises(ValidationError):
            c.finish()
        c.coverage = 16384
        c.identity_seen = {(chunk, chunk+8192, layer) for chunk in (0, 8192) for layer in range(40)}
        report = c.finish()
        self.assertEqual(report["identity_check_count"], 80)
        self.assertFalse(c.qualified)
        with self.assertRaises(ValidationError):
            c.reset("r2", "qualified_native", input_ids=[1]*16384, phase="qual")

    def test_reset_rejects_unqualified_sensitive_fraction_and_dirty_adapter(self):
        for mode, fraction in (("sensitive", .2), ("native", .2), ("other", None)):
            with self.assertRaises(ValidationError):
                controller().reset("r", mode, input_ids=[1]*16384, phase="p", fraction=fraction)
        c = controller()
        c.adapter.pending = object()
        with self.assertRaises(ValidationError):
            c.reset("r", "qualified_native", input_ids=[1]*16384, phase="p")
        c.adapter.pending, c.adapter.poisoned = None, True
        with self.assertRaises(ValidationError):
            c.reset("r", "qualified_native", input_ids=[1]*16384, phase="p")

    def test_marker_binds_request_phase_bounds_and_exact_schema(self):
        c = controller()
        c.reset("r", "qualified_native", input_ids=[1]*16384, phase="qual")
        marker = dict(request_id="r", phase="qual", start=0, end=8192)
        c._marker(marker)
        for changes in (dict(request_id="other"), dict(phase="other"), dict(start=True),
                        dict(start=-1), dict(end=16385), dict(extra=1)):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                c._marker(dict(marker, **changes))

    def test_close_owns_hooks_and_is_idempotent(self):
        c = controller()
        obj = SimpleNamespace()
        c._patch(obj, "method", lambda owner: 9)
        self.assertEqual(obj.method(), 9)
        c.close()
        self.assertFalse(hasattr(obj, "method"))
        self.assertTrue(c.adapter.closed)
        c.close()
        self.assertEqual(c.state, "CLOSED")
        c2 = controller()
        c2._patch(obj, "method", lambda owner: 10)
        obj.method = lambda: 11
        with self.assertRaises(ValidationError):
            c2.close()

    def test_cache_destination_exclusive_and_active_clear_forbidden(self):
        c = controller()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"cache"
            c.set_cache_output_dir(target)
            self.assertTrue(target.is_dir())
            with self.assertRaises(FileExistsError):
                c.set_cache_output_dir(target)
            c.state = "RUNNING"
            with self.assertRaises(ValidationError):
                c.clear_cache()
            c.state = "COMPLETE"
            c.clear_cache()
            self.assertIsNone(c.cache_output_dir)

    def test_dependency_bytes_are_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"example.py"
            path.write_bytes(b"x = 1\n")
            expected = {"example.py": hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch("kv_cache_engram.early_mhc.DEPENDENCIES", expected):
                self.assertEqual(verify_dependencies(root), expected)
                path.write_bytes(b"x = 2\n")
                with self.assertRaises(ValidationError):
                    verify_dependencies(root)


class IndependentReaderTests(unittest.TestCase):
    def test_mask_reconstruction_matches_reference_with_shifted_document_boundary(self):
        from gpu_pilot.read_early_mhc_results import mask_reference
        from kv_cache_engram.early_sensitivity import EarlySensitivityConfig, METRIC, REPRESENTATION
        cfg = EarlySensitivityConfig(metric=METRIC, optional_fractions=(.2, .5, 1.), representation=REPRESENTATION)
        scores = [float(i % 11) for i in range(8192)]
        spans = [(3, 32771, "a"*64), (32771, 65539, "b"*64)]
        for fraction in (.2, .5):
            actual, mandatory, optional = mask_reference(scores, 32768, 40960, spans, fraction)
            reference = plan_from_scores(scores, start=32768, end=40960, input_tokens=65539,
                document_spans=[(a,b) for a,b,_ in spans], config=cfg)
            target = next(r for r in reference["selections"] if r["optional_fraction"] == fraction)
            self.assertEqual([i+32768 for i in actual], target["retained_positions"])
            self.assertEqual((mandatory,optional), (target["mandatory_rows"],target["optional_rows"]))

    def test_reader_mask_rejects_nonfinite_or_missing_scores(self):
        from gpu_pilot.read_early_mhc_results import mask_reference
        for scores in ([0.]*8191, [float("nan")]*8192, [-1.]*8192):
            with self.assertRaises(ValidationError):
                mask_reference(scores, 0, 8192, [(0,32768,"a"*64)], .2)

    def test_fixed_gold_scoring_keeps_zero_and_partial_credit(self):
        from gpu_pilot.read_early_mhc_results import score
        self.assertEqual(score("November 23, 2002", ["November"]),
                         dict(answer_em=0., answer_token_f1=.5))
        self.assertEqual(score("Coca-Cola FEMSA", ["Fomento Económico Mexicano"]),
                         dict(answer_em=0., answer_token_f1=0.))

    def test_reader_rejects_changed_seal_and_duplicate_keys(self):
        from gpu_pilot.read_early_mhc_results import read
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"raw.json"
            body = dict(x=1)
            path.write_text(json.dumps(dict(body,sha256=digest(body))))
            self.assertEqual(read(path,{})["x"], 1)
            path.write_text(json.dumps(dict(x=2,sha256=digest(body))))
            with self.assertRaises(ValidationError):
                read(path,{})
            path.write_text('{"x":1,"x":1,"sha256":"'+digest(body)+'"}')
            with self.assertRaises(ValidationError):
                read(path,{})

    def test_raw_document_bindings_require_eight_distinct_exact_32k_spans(self):
        from gpu_pilot.read_early_mhc_results import documents
        docs = [dict(input_start=3+i*32768, input_end=3+(i+1)*32768,
                     token_ids_sha256=digest([i])) for i in range(8)]
        row = dict(input_tokens=262147,documents=docs)
        self.assertEqual(len(documents(row)),8)
        docs[7]["token_ids_sha256"] = docs[0]["token_ids_sha256"]
        with self.assertRaises(ValidationError):
            documents(row)

    def test_no_qualification_is_incomplete_not_success(self):
        from gpu_pilot.read_early_mhc_results import audit, RUN_SCHEMA
        from gpu_pilot.early_mhc_protocol import early_mhc_protocol
        docs = [dict(input_start=3+i*32768,input_end=3+(i+1)*32768,
                     token_ids_sha256=digest([i])) for i in range(8)]
        rows = [dict(index=index,dataset=dataset,input_tokens=262147,input_ids_sha256=digest([index]),
                     documents=docs) for index,dataset in ((2,"hotpotqa"),(3,"2wikimqa"),
                                                          (4,"hotpotqa"),(5,"2wikimqa"))]
        body = dict(schema=RUN_SCHEMA,protocol=early_mhc_protocol(),evaluation_case_indices=[2,3,4,5],
            audit_cases=rows,manifest_sha256="a"*64,identity=dict(execution_configuration_sha256="b"*64))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"preflight.json").write_text(json.dumps(dict(body,sha256=digest(body))))
            result = audit(directory)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["complete_cases"],0)
            self.assertFalse(result["qualification_passed"])


TORCH = importlib.util.find_spec("torch") is not None


class HostTransferGuardTests(unittest.TestCase):
    """Stdlib device-operation stub: forward host transfers are hard errors."""

    class Vector:
        def __init__(self, values, *, dtype="int64", device="fixture-device"):
            self.values, self.dtype, self.device = tuple(values), dtype, device
            self.shape = (len(self.values),)

        def index_select(self, dim, ids):
            assert dim == 0 and ids.device == self.device
            return type(self)((self.values[i] for i in ids.values), dtype=self.dtype, device=self.device)

        def __getitem__(self, index):
            if isinstance(index, slice):
                return type(self)(self.values[index], dtype=self.dtype, device=self.device)
            return self.values[index]

        def sort(self):
            return SimpleNamespace(values=type(self)(sorted(self.values), dtype=self.dtype, device=self.device))

        def contiguous(self):
            return self

    class DeviceOps:
        long = "int64"
        def __init__(self, vector):
            self.vector, self.preparing, self.host_transfers = vector, True, 0

        def device(self, value):
            return value

        def tensor(self, values, *, device, dtype):
            if not self.preparing:
                raise AssertionError("host-to-device tensor construction inside get_plan")
            self.host_transfers += 1
            return self.vector(values, device=device, dtype=dtype)

        def arange(self, count, *, device, dtype):
            if not self.preparing:
                raise AssertionError("index allocation inside get_plan")
            return self.vector(range(count), device=device, dtype=dtype)

        def stack(self, values):
            vector = self.vector
            return SimpleNamespace(mean=lambda dim: vector(
                (sum(items)/len(values) for items in zip(*(v.values for v in values))),
                device=values[0].device, dtype="float64"))

        def isfinite(self, value):
            return SimpleNamespace(all=lambda: all(math.isfinite(v) for v in value.values))

        def _assert_async(self, condition, message):
            if not condition:
                raise ValidationError(message)

        def argsort(self, value, *, descending, stable):
            assert descending and stable
            return self.vector(sorted(range(len(value.values)), key=lambda i: (-value.values[i], i)),
                               device=value.device)

        def cat(self, values):
            return self.vector((x for v in values for x in v.values), device=values[0].device)

    def prepared(self, mode="sensitive", fraction=.2):
        c = controller()
        c.t, c.DevicePlan = self.DeviceOps(self.Vector), DummyPlan
        c.qualified = True
        if mode == "all_selected":
            ids, docs = (1,)*16384, None
            c.reference_input = token_digest(ids)
            c.reference = {(start,start+8192,layer): {} for start in (0,8192) for layer in range(40)}
            fraction = None
        else:
            ids = (99,)*3 + tuple(value for i in range(8) for value in (i+1,)*32768) + (98,)*4
            docs = [dict(input_start=3+i*32768, input_end=3+(i+1)*32768,
                         token_ids_sha256=token_digest((i+1,)*32768)) for i in range(8)]
            c._entry = lambda key, layer: dict(tensor=SimpleNamespace(device="fixture-device"))
        c.reset("r", mode, input_ids=ids, documents=docs, phase="p", fraction=fraction,
                index_device="fixture-device")
        c.t.preparing = False
        return c

    def test_sensitive_forward_uses_only_pretransferred_indices(self):
        for fraction in (.2,.5):
            c = self.prepared(fraction=fraction)
            self.assertEqual(c.t.host_transfers, 2*math.ceil(len(c.ids)/8192))
            values = [float(i % 7) for i in range(8192)]
            c.scores[(8192,16384)] = {layer:self.Vector(values, dtype="float64") for layer in range(6)}
            meta = SimpleNamespace(total_q=8192, positions=self.Vector(range(8192,16384)),
                canonical_request_binding=dict(request_id="r",phase="p",start=8192,end=16384))
            attn = SimpleNamespace(layer_idx=6,is_kv_source=False,has_local_index=False,
                                   is_yoco_hidden_consumer_layer=False)
            with patch("kv_cache_engram.early_mhc.plan_from_scores", side_effect=AssertionError("CPU geometry in forward")):
                plan = c.get_plan(attn, meta)
                self.assertIs(c.get_plan(attn, meta), plan)
            oracle = plan_from_scores(values,start=8192,end=16384,input_tokens=len(c.ids),
                document_spans=[(d["start"],d["end"]) for d in c.docs],config=c.config)
            expected = next(r for r in oracle["selections"] if r["optional_fraction"] == fraction)
            self.assertEqual([8192+i for i in plan.selected_rows.values],expected["retained_positions"])

    def test_all_selected_forward_cannot_allocate_or_transfer_indices(self):
        c = self.prepared("all_selected")
        meta = SimpleNamespace(total_q=8192,positions=self.Vector(range(8192)),
            canonical_request_binding=dict(request_id="r",phase="p",start=0,end=8192))
        attn = SimpleNamespace(layer_idx=6,is_kv_source=False,has_local_index=False,
                               is_yoco_hidden_consumer_layer=False)
        self.assertEqual(c.get_plan(attn,meta).selected_rows.values,tuple(range(8192)))
        self.assertEqual(c.t.host_transfers,0)

    def test_final_short_chunk_with_no_optional_rows_needs_no_transfer(self):
        c = self.prepared()
        start, end = 262144, len(c.ids)
        count = end-start
        c.scores[(start,end)] = {layer:self.Vector([0.]*count,dtype="float64") for layer in range(6)}
        meta = SimpleNamespace(total_q=count,positions=self.Vector(range(start,end)),
            canonical_request_binding=dict(request_id="r",phase="p",start=start,end=end))
        attn = SimpleNamespace(layer_idx=6,is_kv_source=False,has_local_index=False,
                               is_yoco_hidden_consumer_layer=False)
        plan = c.get_plan(attn,meta)
        self.assertEqual(plan.selected_rows.values,tuple(range(count)))
        self.assertEqual(c.plans[(start,end)]["optional_rows"],0)

    def test_prepared_scope_and_unexpected_chunk_fail_closed(self):
        c = self.prepared("all_selected")
        meta = SimpleNamespace(total_q=8192,positions=self.Vector(range(8192)),
            canonical_request_binding=dict(request_id="r",phase="p",start=0,end=8192))
        attn = SimpleNamespace(layer_idx=6,is_kv_source=False,has_local_index=False,
                               is_yoco_hidden_consumer_layer=False)
        c.prepared_index_scope = ("old", "p", c.input_sha, c.mode, c.fraction)
        with self.assertRaises(ValidationError):
            c.get_plan(attn,meta)
        with self.assertRaises(ValidationError):
            c._prepare_indices(c.mode,c.ids,c.docs,"fixture-device")

    def test_forward_methods_contain_no_host_transfer_or_sync_escape(self):
        forbidden = {"tensor","as_tensor","from_numpy","item","cpu","tolist","numpy","allow_sync"}
        for name in ("get_plan","_score","_canonical"):
            tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(EarlyMHCController,name))))
            calls = {node.func.attr for node in ast.walk(tree)
                     if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
            self.assertFalse(calls & forbidden,(name,calls & forbidden))


@unittest.skipUnless(TORCH, "optional Torch CPU fixtures unavailable")
class TorchControllerTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        self.t.set_num_threads(self.old_threads)

    def ready(self, mode="sensitive", fraction=.2, start=0):
        c = controller()
        c.t, c.DevicePlan = self.t, DummyPlan
        c.state, c.mode, c.rid, c.phase, c.fraction = "RUNNING", mode, "r", "quality", fraction
        c.ids, c.input_sha = (1,)*32768, "d"*64
        c.docs = (dict(start=0, end=32768, key="e"*64),)
        c.scores, c.plans, c.identity_seen = {}, {}, set()
        c.coverage = 0
        c.capture_coverage = {layer: 0 for layer in CAPTURE_LAYERS}
        c.state = "IDLE"
        c.prepared_indices = c._prepare_indices(mode, c.ids, c.docs, "cpu")
        c.prepared_index_scope = (c.rid,c.phase,c.input_sha,c.mode,c.fraction)
        c.state = "RUNNING"
        meta = SimpleNamespace(positions=self.t.arange(start, start+8192), total_q=8192,
            canonical_request_binding=dict(request_id="r", phase="quality", start=start, end=start+8192))
        attn = SimpleNamespace(layer_idx=6, is_kv_source=False, has_local_index=False,
                               is_yoco_hidden_consumer_layer=False)
        return c, attn, meta

    def put_scores(self, c, marker, values):
        c.scores[(marker["start"], marker["end"])] = {
            layer: self.t.tensor(values, dtype=self.t.float64) for layer in range(6)}

    def test_plan_matches_exact_cpu_reference_ties_and_nonzero_chunk_offset(self):
        for fraction in (.2, .5):
            c, attn, meta = self.ready(fraction=fraction, start=8192)
            values = [float(i % 7) for i in range(8192)]
            self.put_scores(c, meta.canonical_request_binding, values)
            plan = c.get_plan(attn, meta)
            oracle = plan_from_scores(values, start=8192, end=16384, input_tokens=32768,
                document_spans=[(0, 32768)], config=c.config)
            expected = next(item for item in oracle["selections"] if item["optional_fraction"] == fraction)
            self.assertEqual((plan.selected_rows+8192).tolist(), expected["retained_positions"])
            self.assertEqual(plan.selected_count, expected["retained_rows"])
            self.assertIs(c.get_plan(attn, meta), plan)

    def test_all_selected_does_not_need_scores_and_selects_every_row(self):
        c, attn, meta = self.ready(mode="all_selected", fraction=None)
        plan = c.get_plan(attn, meta)
        self.assertEqual(plan.selected_rows.tolist(), list(range(8192)))
        self.assertEqual(plan.selected_count, 8192)

    def test_device_plan_uses_no_torch_host_constructor_inside_forward(self):
        for mode,fraction in (("sensitive",.2),("sensitive",.5),("all_selected",None)):
            c,attn,meta = self.ready(mode=mode,fraction=fraction)
            if mode == "sensitive":
                self.put_scores(c,meta.canonical_request_binding,[float(i % 11) for i in range(8192)])
            with patch.object(self.t,"tensor",side_effect=AssertionError("forward host tensor")), \
                    patch.object(self.t,"as_tensor",side_effect=AssertionError("forward host tensor")), \
                    patch.object(self.t,"arange",side_effect=AssertionError("forward index construction")):
                plan = c.get_plan(attn,meta)
                self.assertIs(c.get_plan(attn,meta),plan)
            self.assertGreater(plan.selected_count,0)

    def test_prepared_index_device_mismatch_rejected_before_ranking(self):
        c,attn,meta = self.ready()
        self.put_scores(c,meta.canonical_request_binding,[0.]*8192)
        meta.positions = SimpleNamespace(device="not-cpu")
        with self.assertRaises(ValidationError):
            c.get_plan(attn,meta)
        self.assertFalse(c.plans)

    def test_missing_layer_and_nan_scores_fail_without_plan(self):
        c, attn, meta = self.ready()
        with self.assertRaises(ValidationError):
            c.get_plan(attn, meta)
        self.put_scores(c, meta.canonical_request_binding, [0.]*8192)
        c.scores[(0, 8192)][3][100] = float("nan")
        with self.assertRaises((RuntimeError, ValidationError)):
            c.get_plan(attn, meta)
        self.assertFalse(c.plans)

    def test_protected_layers_and_metadata_scope_never_reuse_stale_plan(self):
        c, attn, meta = self.ready()
        self.put_scores(c, meta.canonical_request_binding, [1.]*8192)
        c.get_plan(attn, meta)
        for layer in (0, 5, 8, 14, 19, 20, 39):
            self.assertIsNone(c.get_plan(SimpleNamespace(layer_idx=layer), meta))
        meta.canonical_request_binding["request_id"] = "foreign"
        with self.assertRaises(ValidationError):
            c.get_plan(attn, meta)
        meta.canonical_request_binding["request_id"] = "r"
        meta.total_q = 8191
        with self.assertRaises(ValidationError):
            c.get_plan(attn, meta)

    def test_source_index_or_compact_flags_rejected_on_whitelisted_layer(self):
        for flag in ("is_kv_source", "has_local_index", "is_yoco_hidden_consumer_layer"):
            c, attn, meta = self.ready(mode="all_selected", fraction=None)
            setattr(attn, flag, True)
            with self.subTest(flag=flag), self.assertRaises(ValidationError):
                c.get_plan(attn, meta)

    def test_score_is_same_layer_float64_and_requires_layer_order(self):
        c, _, _ = self.ready()
        marker = dict(start=0, end=8)
        canonical = self.t.full((8, 512), 2., dtype=self.t.bfloat16)
        c._canonical = lambda layer, start, end, device: self.t.ones((8, 512), dtype=self.t.bfloat16)
        with self.assertRaises(ValidationError):
            c._score(1, marker, canonical)
        for layer in range(6):
            c._score(layer, marker, canonical)
        self.assertEqual(c.scores[(0, 8)][0].dtype, self.t.float64)
        self.assertTrue(self.t.equal(c.scores[(0, 8)][0], self.t.full((8,), .5, dtype=self.t.float64)))
        with self.assertRaises(ValidationError):
            c._score(0, marker, canonical)

    def bank_entry(self, c, key="e"*64, layer=6):
        tensor = self.t.zeros((32768, 512), dtype=self.t.bfloat16)
        entry = dict(tensor=tensor, complete=True, version=tensor._version,
            scope=dict(c.scope, document_sha256=key, layer=layer,
                       representation="swa-postnorm-prerope-bf16"))
        c.bank[(key, layer)] = entry
        return entry

    def test_cache_version_complete_and_execution_identity_fail_closed(self):
        c, _, _ = self.ready()
        entry = self.bank_entry(c)
        self.assertIs(c._entry("e"*64, 6), entry)
        entry["tensor"][0, 0] = 1
        with self.assertRaises(ValidationError):
            c._entry("e"*64, 6)
        entry["version"] = entry["tensor"]._version
        entry["complete"] = False
        with self.assertRaises(ValidationError):
            c._entry("e"*64, 6)
        entry["complete"] = True
        entry["scope"]["execution_sha256"] = "f"*64
        with self.assertRaises(ValidationError):
            c._entry("e"*64, 6)

    def test_cache_document_layer_and_representation_identity_bound(self):
        c, _, _ = self.ready()
        entry = self.bank_entry(c)
        for name, bad in (("document_sha256", "f"*64), ("layer", 7),
                          ("representation", "post-rope")):
            old = entry["scope"][name]
            entry["scope"][name] = bad
            with self.subTest(name=name), self.assertRaises(ValidationError):
                c._entry("e"*64, 6)
            entry["scope"][name] = old

    def test_fingerprint_uses_exact_bytes_and_lazy_none_distinction(self):
        a = self.t.tensor([[0., -0.], [1., 2.]], dtype=self.t.bfloat16)
        first = tensor_fingerprint(a, self.t)
        self.assertEqual(first, tensor_fingerprint(a.clone(), self.t))
        self.assertEqual(first["bytes"], 8)
        self.assertIsNone(tensor_fingerprint(None, self.t))
        a[0, 1] = 0.
        self.assertNotEqual(first["sha256"], tensor_fingerprint(a, self.t)["sha256"])
        with self.assertRaises(ValidationError):
            tensor_fingerprint(self.t.tensor([float("nan")]), self.t)

    @contextmanager
    def installed_fake(self):
        t = self.t
        class Attn:
            def __init__(self, layer):
                self.layer_idx = layer
            def _prepare_swa_kv(self, raw, cache, metadata, use_cp):
                return raw
        class Layer:
            def __init__(self, index):
                self.layer_idx, self.self_attn = index, Attn(index)
                self.change_lazy, self.change_hidden = False, False
            def forward(self, hidden, cache, input_metadata):
                output = hidden.clone()
                lazy = SimpleNamespace(residual=t.zeros(2, dtype=t.bfloat16),
                    post_layer_mix=None, comb_res_mix=t.ones(2, dtype=t.bfloat16),
                    pre_layer_mix=None)
                if self.change_hidden:
                    output[0] += 1
                if self.change_lazy:
                    lazy.comb_res_mix[0] += 1
                return output, lazy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sa, sr = ModuleType("sparse_attention"), ModuleType("sparse_residual")
            sa.__file__ = str(root/"sparse_query_20261030_v8/sparse_attention.py")
            sr.__file__ = str(root/"sparse_query_20261030_v8/sparse_residual.py")
            sa.sparse_attention = sa.make_active_metadata = lambda *a, **k: None
            sr.DevicePlan = DummyPlan
            layers = [Layer(i) for i in range(40)]
            c = EarlyMHCController(layers, model_sha256="a"*64, execution_sha256="b"*64,
                tokenizer_sha256="c"*64, dependency_root=root)
            original_path = list(sys.path)
            with patch.dict(sys.modules, sparse_attention=sa, sparse_residual=sr), \
                    patch("kv_cache_engram.early_mhc.verify_dependencies", return_value={}), \
                    patch("kv_cache_engram.early_mhc.NativeReuseOps.from_deepinfer", return_value=SimpleNamespace(torch=t)), \
                    patch("kv_cache_engram.early_mhc.EarlyMHCAdapter", DummyAdapter):
                try:
                    c.install()
                    yield c, layers
                finally:
                    c.close()
                    sys.path[:] = original_path

    def reference_pass(self, c, layers):
        c.reset("native", "qualified_native", input_ids=[1]*16384, phase="parity")
        for start in (0, 8192):
            metadata = SimpleNamespace(canonical_request_binding=dict(
                request_id="native", phase="parity", start=start, end=start+8192))
            for layer in layers:
                layer.forward(self.t.zeros(2, dtype=self.t.bfloat16), None, metadata)
        c.finish()

    def test_real_wrapper_guards_every_lazy_fingerprint_not_only_hidden(self):
        for field in ("change_hidden", "change_lazy"):
            with self.subTest(field=field), self.installed_fake() as (c, layers):
                self.reference_pass(c, layers)
                self.assertEqual(len(c.reference), 80)
                c.reset("selected", "all_selected", input_ids=[1]*16384, phase="parity2", index_device="cpu")
                metadata = SimpleNamespace(canonical_request_binding=dict(
                    request_id="selected", phase="parity2", start=0, end=8192))
                setattr(layers[0], field, True)
                with self.assertRaises(ValidationError):
                    layers[0].forward(self.t.zeros(2, dtype=self.t.bfloat16), None, metadata)
                self.assertEqual(c.state, "FAILED")
                self.assertFalse(c.qualified)

    def test_unchanged_wrapper_checks_all_40_layers_two_chunks(self):
        with self.installed_fake() as (c, layers):
            self.reference_pass(c, layers)
            c.reset("selected", "all_selected", input_ids=[1]*16384, phase="parity2", index_device="cpu")
            for start in (0, 8192):
                metadata = SimpleNamespace(canonical_request_binding=dict(
                    request_id="selected", phase="parity2", start=start, end=start+8192))
                for layer in layers:
                    layer.forward(self.t.zeros(2, dtype=self.t.bfloat16), None, metadata)
            self.assertEqual(c.identity_seen, set(c.reference))
            self.assertEqual(len(c.identity_seen), 80)


if __name__ == "__main__":
    unittest.main()
