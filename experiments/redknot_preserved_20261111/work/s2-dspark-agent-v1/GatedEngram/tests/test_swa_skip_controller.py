"""CPU contracts for the independent global-only ablation, not GPU proof."""
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kv_cache_engram.core import ValidationError, digest
from kv_cache_engram.swa_skip import SwaSkipController, expected_zos_decode_steps, global_buffer_view, global_scheduler_key


class Tensor:
    def __init__(self, *shape, values=None):
        self.shape = shape
        self.values = list(range(shape[0])) if values is None else list(values)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise AssertionError("unexpected scalar tensor access")
        return Tensor(len(self.values[item]), *self.shape[1:], values=self.values[item])

    def clone(self):
        return Tensor(*self.shape, values=self.values)

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Adapter:
    poisoned = in_call = False

    def __init__(self):
        self.events, self.counts = [], {}

    def report(self):
        return dict(events=[], counts={})

    def close(self, **kwargs):
        pass


def make_controller():
    layers = tuple(SimpleNamespace(layer_idx=i) for i in range(40))
    c = SwaSkipController(layers, model_sha256="a" * 64, execution_sha256="b" * 64,
                          tokenizer_sha256="c" * 64,
                          native_model=SimpleNamespace(layers=layers, yoco_boundary_layer_idx=20))
    c.original_yoco_boundary, c.adapter, c.state = 20, Adapter(), "IDLE"
    c.t = SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(is_current_stream_capturing=lambda: False))
    c.ops = SimpleNamespace(rows=lambda v: v.shape[0],
                            PG=SimpleNamespace(context_parallel_group=lambda: SimpleNamespace(size=lambda: 1)))
    return c


def sequence(*, rid="r", seq_id=7, prompt=True, q=8, k=8, placeholder=False):
    return SimpleNamespace(request_id=rid, seq_id=seq_id, is_prompt=prompt,
                           seqlen_q=q, seqlen_k=k, is_placeholder=placeholder)


def completed_schedule(c, seq, *, decode_step=None):
    c.placeholder_request_id = "placeholder"
    c.schedule_identity_installed = True
    context = c._schedule_context([seq])
    context.update(layer_indices=list(range(40)), layer_rows=[seq.seqlen_q] * 40,
                   real_decode_step=decode_step, complete=True)
    c.scheduler_invocations.append(context)
    c.current_schedule = context
    return context


def metadata(prompt=True, start=0, end=8):
    return SimpleNamespace(is_prompt=prompt, total_q=end-start if prompt else 1,
                           positions=Tensor(end-start if prompt else 1),
                           canonical_request_binding=dict(request_id="r", phase="p", start=start, end=end) if prompt else None)


class Attention:
    layer_idx, has_packed_kv, use_swa = 6, True, True
    kv_compress_ratio, layer_cfg = 2, {"index_topk": 512}
    sched_meta_key = "2_True_True_512"
    use_shared_kv, is_kv_source = True, False

    def __init__(self):
        self.calls = []
        self.fail = False

    def forward(self, hidden, cache, meta, bf16, cross_layer_runtime=None):
        q, raw = self._project_q_and_swa(hidden, None)
        self._prepare_swa_kv(raw, cache, meta, False)
        if self.fail:
            raise RuntimeError("CPU synthetic failure")
        if meta.is_prompt:
            offset = meta.total_q + 128 if self.use_swa else 0
            kv = self._prepare_prefill_kv(offset + 10, offset, None, cross_layer_runtime)
            return self._prefill_attention(q, kv, "global-index", "swa-index" if self.use_swa else None,
                                           128 if self.use_swa else 0, meta.total_q if self.use_swa else 0, meta)
        return self._decode_attention(q, "global-index", "swa-index" if self.use_swa else None, cache, meta)

    def _project_q_and_swa(self, hidden, shared):
        self.calls.append("native-project")
        return Tensor(hidden.shape[0], 1280), Tensor(hidden.shape[0], 512)

    def _prepare_swa_kv(self, raw, cache, meta, use_cp):
        self.calls.append("native-swa-prepare")
        assert raw is not None
        return Tensor(meta.total_q, 1, 512), None

    def _prepare_prefill_kv(self, total, offset, device, runtime):
        self.calls.append(("native-buffer", total, offset))
        return Tensor(total, 1, 512)

    def _prefill_attention(self, q, kv, global_index, swa_index, past, current, meta):
        self.calls.append(("core", self.use_swa, kv.shape[0], swa_index, past, current))
        return Tensor(meta.total_q, 512)

    def _decode_attention(self, q, global_index, swa_index, cache, meta):
        self.calls.append(("decode", self.use_swa, self.sched_meta_key, swa_index))
        return Tensor(meta.total_q, 512)


def activate(c, variant="global_only", mode="all_selected"):
    c.state, c.mode, c.attention_variant = "RUNNING", mode, variant
    c.rid, c.phase, c.ids = "r", "p", (1,) * 8
    c.projections[6] = (lambda x: Tensor(x.shape[0], 1280), Mock(side_effect=AssertionError("SWA projection called")))
    c.current_schedule = dict(kind="request", invocation_index=0)


class BufferTests(unittest.TestCase):
    def test_mixed_source_buffer_keeps_only_global_tail_without_copy(self):
        result = global_buffer_view(Tensor(146, 1, 512), 136, 10)
        self.assertEqual(result.shape, (10, 1, 512))
        self.assertEqual(result.values, list(range(136, 146)))

    def test_source_already_global_only_accepts_zero_offset(self):
        self.assertEqual(global_buffer_view(Tensor(10, 1, 512), 0, 10).shape, (10, 1, 512))

    def test_empty_global_shape_keeps_valid_geometry(self):
        self.assertEqual(global_buffer_view(Tensor(0, 1, 512), 0, 0).shape, (0, 1, 512))

    def test_invalid_offsets_and_geometry_fail_closed(self):
        for loaded, offset, rows in ((None, 0, 0), (Tensor(10, 1, 512), -1, 11),
                                     (Tensor(10, 1, 512), True, 9), (Tensor(10, 1, 512), 0, 11),
                                     (Tensor(10, 2, 512), 0, 10), (Tensor(10, 1, 256), 0, 10)):
            with self.subTest(offset=offset, rows=rows), self.assertRaises(ValidationError):
                global_buffer_view(loaded, offset, rows)


class OwnershipTests(unittest.TestCase):
    def test_import_does_not_import_torch_or_native(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import kv_cache_engram.swa_skip; assert 'torch' not in sys.modules; assert not any(k.startswith('deepinfer') for k in sys.modules)"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_owned_flags_restore_after_error(self):
        c, attn = make_controller(), Attention()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with c._without_swa(attn):
                self.assertFalse(attn.use_swa)
                self.assertEqual(attn.sched_meta_key, "2_True_False_512")
                raise RuntimeError("failure")
        self.assertTrue(attn.use_swa)
        self.assertEqual(attn.sched_meta_key, "2_True_True_512")

    def test_foreign_flag_change_is_not_overwritten(self):
        c, attn = make_controller(), Attention()
        with self.assertRaises(ValidationError):
            with c._without_swa(attn):
                attn.sched_meta_key = "foreign"
        self.assertEqual(attn.sched_meta_key, "foreign")

    def test_foreign_instance_hook_cannot_be_replaced(self):
        c, attn = make_controller(), Attention()
        c._install_attention(attn)
        attn.forward = lambda *a: None
        with self.assertRaises(ValidationError):
            c._replace_owned_hook(attn, "forward", lambda *a: None)

    def test_unknown_variant_rejected_before_state_change(self):
        c = make_controller()
        with self.assertRaises(ValidationError):
            c.reset("r", "native", input_ids=[1], phase="p", attention_variant="bad")
        self.assertEqual(c.state, "IDLE")

    def test_global_qualification_requires_native_qualification(self):
        c = make_controller()
        with self.assertRaises(ValidationError):
            c.reset("r", "qualified_native", input_ids=[1] * 16384, phase="p", attention_variant="global_reference")

    def test_global_sensitive_requires_complete_global_reference(self):
        c = make_controller()
        with self.assertRaises(ValidationError):
            c.reset("r", "sensitive", input_ids=[1] * 16384, phase="p", fraction=.15, attention_variant="global_only")

    def test_global_reference_resets_native_identity_independently(self):
        c = make_controller()
        c.native_qualified, c.qualified = True, True
        c.reference = {(s, s+8192, layer): {} for s in (0, 8192) for layer in range(40)}
        c.reference_input = "d" * 64
        c.reset("r", "qualified_native", input_ids=[1] * 16384, phase="p", attention_variant="global_reference")
        self.assertEqual(len(c.native_reference_saved[0]), 80)
        self.assertEqual(c.native_reference_saved[1], "d" * 64)
        self.assertEqual(c.reference, {})
        self.assertTrue(c.global_reference_started)
        self.assertEqual(c.attention_variant, "global_reference")


class ForwardTests(unittest.TestCase):
    def test_optimized_prefill_skips_swa_and_handles_mixed_global_buffer(self):
        c, attn = make_controller(), Attention()
        c._install_attention(attn)
        activate(c)
        runtime = SimpleNamespace(consumer_full_metadata=None, loaded_prefill_kv=Tensor(146, 1, 512), loaded_prefill_kv_main_offset=136)
        out = attn.forward(Tensor(8, 5120), object(), metadata(), Tensor(8, 5120), cross_layer_runtime=runtime)
        self.assertEqual(out.shape, (8, 512))
        self.assertEqual(attn.calls, [("core", False, 10, None, 0, 0)])
        self.assertTrue(attn.use_swa)
        self.assertIsNone(c.frame)
        self.assertEqual(len(c.global_skip_events), 1)
        self.assertEqual(c.global_skip_events[0]["swa_projection_rows"], 0)
        self.assertEqual(c.global_skip_events[0]["offline_kv_attention_reads"], 0)
        self.assertEqual(runtime.loaded_prefill_kv_main_offset, 136)
        c.close()
        self.assertNotIn("forward", attn.__dict__)

    def test_reference_executes_native_projection_but_only_global_core(self):
        c, attn = make_controller(), Attention()
        c._install_attention(attn)
        activate(c, "global_reference", "qualified_native")
        out = attn.forward(Tensor(8, 5120), object(), metadata(), Tensor(8, 5120), cross_layer_runtime=SimpleNamespace())
        self.assertEqual(out.shape, (8, 512))
        self.assertIn("native-project", attn.calls)
        self.assertIn("native-swa-prepare", attn.calls)
        self.assertIn(("core", False, 10, None, 0, 0), attn.calls)
        self.assertEqual(c.global_skip_events[0]["swa_projection_rows"], 8)
        self.assertTrue(attn.use_swa)

    def test_optimized_decode_never_reads_uninitialized_swa_cache(self):
        c, attn = make_controller(), Attention()
        c._install_attention(attn)
        activate(c)
        c.decode_step = 0
        out = attn.forward(Tensor(1, 5120), object(), metadata(False), Tensor(1, 5120), cross_layer_runtime=SimpleNamespace())
        self.assertEqual(out.shape, (1, 512))
        self.assertEqual(attn.calls, [("decode", False, "2_True_False_512", None)])
        self.assertEqual(c.decode_skip_events[0]["decode_step"], 0)
        self.assertTrue(attn.use_swa)

    def test_reference_decode_has_native_projection_but_distinct_global_scheduler(self):
        c, attn = make_controller(), Attention()
        c._install_attention(attn)
        activate(c, "global_reference", "qualified_native")
        c.decode_step = 0
        attn.forward(Tensor(1, 5120), object(), metadata(False), Tensor(1, 5120), cross_layer_runtime=SimpleNamespace())
        self.assertIn("native-project", attn.calls)
        self.assertIn(("decode", False, "2_True_False_512", None), attn.calls)
        self.assertEqual(c.decode_skip_events[0]["swa_projection_rows"], 1)

    def test_projection_failure_poison_restores_flags_and_frame(self):
        c, attn = make_controller(), Attention()
        c._install_attention(attn)
        activate(c)
        attn.fail = True
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            attn.forward(Tensor(8, 5120), object(), metadata(), Tensor(8, 5120), cross_layer_runtime=SimpleNamespace())
        self.assertEqual(c.state, "FAILED")
        self.assertIsNone(c.frame)
        self.assertTrue(attn.use_swa)

    def test_first_six_layers_not_disabled(self):
        c, attn = make_controller(), Attention()
        attn.layer_idx = 0
        c._install_attention(attn)
        activate(c, mode="qualified_native")
        attn.forward(Tensor(8, 5120), object(), metadata(), Tensor(8, 5120), cross_layer_runtime=SimpleNamespace())
        self.assertIn(("core", True, 146, "swa-index", 128, 8), attn.calls)
        self.assertEqual(c.global_skip_events, [])

    def test_foreign_intervention_rejected_before_projection(self):
        c, attn = make_controller(), Attention()
        c._install_attention(attn)
        activate(c)
        meta = metadata()
        meta.sparse_phase = object()
        with self.assertRaises(ValidationError):
            attn.forward(Tensor(8, 5120), object(), meta, Tensor(8, 5120), cross_layer_runtime=SimpleNamespace())
        self.assertEqual(attn.calls, [])

    def test_no_cached_attention_output_or_fabricated_canonical_restore(self):
        source = inspect.getsource(SwaSkipController._install_attention)
        self.assertNotIn("self._canonical(", source)
        self.assertNotIn("self.ops.postprocess(", source)
        self.assertIn("return q, None", source)

    def test_source_buffer_still_uses_native_global_allocation(self):
        c, attn = make_controller(), Attention()
        attn.is_kv_source = True
        c._install_attention(attn)
        activate(c)
        attn.forward(Tensor(8, 5120), object(), metadata(), Tensor(8, 5120), cross_layer_runtime=SimpleNamespace())
        self.assertIn(("native-buffer", 10, 0), attn.calls)


def finished_qualification(optimized):
    c = make_controller()
    activate(c, "global_only" if optimized else "global_reference", "all_selected" if optimized else "qualified_native")
    c.ids, c.coverage, c.input_sha = (1,) * 16384, 16384, "e" * 64
    c.native_qualified, c.reference_input = True, "e" * 64
    c.decode_step, c.plans, c.ffn_layouts = 1, {}, []
    fp = dict(shape=[1, 512], dtype="torch.bfloat16", bytes=1024, sha256="d" * 64)
    value = dict(output=fp, lazy={k: None for k in ("residual", "post_layer_mix", "comb_res_mix", "pre_layer_mix")})
    c.observed = {(s, s+8192, layer): value for s in (0,8192) for layer in range(40)}
    c.reference = dict(c.observed)
    c.decode_observed = {(step, layer): value for step in range(2) for layer in range(40)}
    c.global_decode_reference = dict(c.decode_observed)
    for start in (0,8192):
        completed_schedule(c, sequence(q=8192, k=start+8192))
        for layer in range(6,40):
            meta = metadata(start=start, end=start+8192)
            meta.positions = Tensor(8192, values=range(start,start+8192))
            c._record_attention(SimpleNamespace(layer_idx=layer), meta, decode=False)
            if optimized:
                c.ffn_layouts.append(dict(layer=layer, chunk_start=start, chunk_end=start+8192,
                                          current_positions=meta.positions))
    for step in range(2):
        completed_schedule(c, sequence(prompt=False, q=1, k=16385+step), decode_step=step)
        c.decode_step = step
        for layer in range(6,40):
            meta = metadata(False)
            meta.positions = Tensor(1, values=[16384+step])
            c._record_attention(SimpleNamespace(layer_idx=layer), meta, decode=True)
    c.current_schedule = None
    return c


class FinishIntegrationTests(unittest.TestCase):
    def test_actual_finish_reports_pass_independent_global_pair_reader(self):
        from gpu_pilot.read_swa_skip_results import PHASE_SCHEMA, validate_global_pair
        phases = []
        for optimized in (False, True):
            c = finished_qualification(optimized)
            c.phase = "qualification." + c.attention_variant
            c.rid = "swa-skip:0:" + c.phase + ":fixture"
            for call in c.scheduler_invocations:
                call["phase"], call["request_id"] = c.phase, c.rid
            report = c.finish(output_tokens=2)
            self.assertEqual(report["decode_identity_fingerprints_sha256"], digest(report["decode_identity_fingerprints"]))
            self.assertEqual(report["identity_fingerprints_sha256"], digest(report["identity_fingerprints"]))
            phase = dict(schema=PHASE_SCHEMA, mode=c.mode, attention_variant=c.attention_variant,
                         name=c.phase, repeat=None, request_id=c.rid,
                         arm="qualification", index=0, timing_valid=False, quality_eligible=False,
                         controller=report, record=dict(success=True, rank=0, input_tokens=16384,
                                                       phase=c.phase, request_id=c.rid, input_ids_sha256=c.input_sha,
                                                       output_ids=[10,11], output_ids_sha256=digest([10,11]), natural_EOS=False))
            for field in ("input_ids_sha256", "execution_configuration_sha256", "swa_skip_protocol_sha256",
                          "manifest_sha256", "empty_ep_qualification_sha256"):
                phase[field] = "c" * 64
            phase["input_ids_sha256"] = c.input_sha
            phase["sha256"] = digest(phase)
            phases.append(phase)
        result = validate_global_pair(*phases)
        self.assertEqual(result["decode_identity_check_count"], 80)
        self.assertFalse(result["native_full_model_equivalence_claim"])

    def test_missing_prefill_event_rejected_and_poisoned(self):
        c = finished_qualification(True)
        c.global_skip_events.pop()
        with self.assertRaises(ValidationError):
            c.finish(output_tokens=2)
        self.assertEqual(c.state, "FAILED")

    def test_missing_decode_event_rejected_and_poisoned(self):
        c = finished_qualification(True)
        c.decode_skip_events.pop()
        with self.assertRaises(ValidationError):
            c.finish(output_tokens=2)
        self.assertEqual(c.state, "FAILED")

    def test_partial_decode_fingerprints_do_not_qualify(self):
        c = finished_qualification(True)
        c.decode_observed.pop((0,39))
        with self.assertRaises(ValidationError):
            c.finish(output_tokens=2)
        self.assertFalse(c.global_qualified)

    def test_global_optimized_fingerprints_must_equal_reference(self):
        c = finished_qualification(True)
        c.decode_observed[(0,39)] = dict(output=None, lazy={})
        with self.assertRaises(ValidationError):
            c.finish(output_tokens=2)
        self.assertFalse(c.global_qualified)

    def test_nonzero_swa_consumption_rejected(self):
        c = finished_qualification(True)
        c.global_skip_events[0]["offline_kv_attention_reads"] = 1
        with self.assertRaises(ValidationError):
            c.finish(output_tokens=2)

    def test_report_does_not_claim_unused_bank_is_effective_reuse(self):
        c = finished_qualification(True)
        c.bank = {("doc", 6): object()}
        report = c.finish(output_tokens=2)
        self.assertEqual(report["retained_offline_bank_entries"], 1)
        self.assertEqual(report["effective_swa_kv_reuse_rate"], 0.0)
        self.assertEqual(report["retained_offline_bank_attention_reads"], 0)
        self.assertTrue(report["skip_is_not_effective_swa_kv_reuse"])
        self.assertFalse(report["identity_exact"])
        self.assertTrue(report["global_identity_exact"])
        self.assertEqual(report, json.loads(json.dumps(report)))

    def test_native_zos_count_distinguishes_prompt_only_and_decode_tail(self):
        self.assertEqual(expected_zos_decode_steps(1), 0)
        self.assertEqual(expected_zos_decode_steps(2), 2)
        self.assertEqual(expected_zos_decode_steps(32), 32)
        for value in (0, -1, True, 2.0, None):
            with self.assertRaises(ValidationError):
                expected_zos_decode_steps(value)

    def test_terminal_tail_missing_is_not_accepted_as_full_qualification(self):
        c = finished_qualification(True)
        c.decode_step = 0
        with self.assertRaises(ValidationError):
            c.finish(output_tokens=2)


class ScheduleIdentityTests(unittest.TestCase):
    def bound(self):
        c = make_controller()
        activate(c)
        c.ids = (1,) * 16384
        c.current_schedule = None

        class Runner:
            fail_at = None

            def execute_main_model(runner, sequences, *args, **kwargs):
                seq = sequences[0]
                meta = metadata(seq.is_prompt, seq.seqlen_k-seq.seqlen_q, seq.seqlen_k)
                if seq.is_placeholder:
                    meta.canonical_request_binding = None
                for layer in range(40):
                    c._record_scheduled_layer(layer, meta)
                    if runner.fail_at == layer:
                        raise RuntimeError("CPU scheduled failure")
                return "native-result"

        runner = Runner()
        c.install_schedule_identity(runner, placeholder_request_id="placeholder")
        return c, runner

    def test_two_chunks_with_intervening_dp_placeholder_do_not_create_decode(self):
        c, runner = self.bound()
        runner.execute_main_model([sequence(q=8192, k=8192)])
        runner.execute_main_model([sequence(rid="placeholder", seq_id=0, prompt=False, q=1, k=1, placeholder=True)])
        runner.execute_main_model([sequence(q=8192, k=16384)])
        self.assertEqual(c.decode_step, -1)
        self.assertEqual(c._scheduler_counts(), dict(total=3, real_prefill=2, real_decode=0,
                                                    placeholder_prefill=0, placeholder_decode=1))
        self.assertTrue(all(e["complete"] and len(e["layer_indices"]) == 40 for e in c.scheduler_invocations))
        self.assertIsNone(c.current_schedule)

    def test_terminal_cleanup_placeholder_is_not_extra_real_decode(self):
        c, runner = self.bound()
        for step in range(2):
            runner.execute_main_model([sequence(prompt=False, q=1, k=16385+step)])
        runner.execute_main_model([sequence(rid="placeholder", seq_id=0, prompt=False, q=1, k=1, placeholder=True)])
        self.assertEqual(c.decode_step, 1)
        self.assertEqual([e["real_decode_step"] for e in c.scheduler_invocations], [0, 1, None])
        evidence = c.failure_diagnostics()
        self.assertEqual(evidence["real_decode_steps"], 2)
        self.assertEqual(evidence["scheduler_counts"]["placeholder_decode"], 1)
        self.assertEqual(evidence, json.loads(json.dumps(evidence)))

    def test_placeholder_prompt_is_also_native_not_canonical_prefill(self):
        c, runner = self.bound()
        runner.execute_main_model([sequence(rid="placeholder", seq_id=0, q=1, k=1, placeholder=True)])
        self.assertEqual(c._scheduler_counts()["placeholder_prefill"], 1)
        self.assertEqual(c._scheduler_counts()["real_prefill"], 0)

    def test_fake_request_ids_and_inconsistent_placeholder_flags_fail_closed(self):
        cases = [sequence(rid="foreign"), sequence(rid="placeholder"),
                 sequence(rid="r", seq_id=0, q=1, k=1, placeholder=True),
                 sequence(rid="placeholder", seq_id=1, q=1, k=1, placeholder=True),
                 sequence(rid="placeholder", seq_id=0, q=1, k=2, placeholder=True)]
        for seq in cases:
            c, runner = self.bound()
            with self.subTest(seq=seq), self.assertRaises(ValidationError):
                runner.execute_main_model([seq])
            self.assertEqual(c.state, "FAILED")
            self.assertEqual(c.scheduler_invocations, [])
            self.assertEqual(c.failure_diagnostics()["rejected_schedule_identity"]["sequences"][0]["request_id"], seq.request_id)

    def test_real_sequence_id_cannot_change_inside_phase(self):
        c, runner = self.bound()
        runner.execute_main_model([sequence(q=8192, k=8192)])
        with self.assertRaisesRegex(ValidationError, "sequence identity changed"):
            runner.execute_main_model([sequence(seq_id=8, q=8192, k=16384)])

    def test_mixed_or_empty_native_batch_rejected(self):
        for seqs in ([], [sequence(), sequence()]):
            c, runner = self.bound()
            with self.assertRaises(ValidationError):
                runner.execute_main_model(seqs)

    def test_interrupted_layer_trace_retains_actual_calls_and_identity(self):
        c, runner = self.bound()
        runner.fail_at = 12
        with self.assertRaisesRegex(RuntimeError, "scheduled failure"):
            runner.execute_main_model([sequence(q=8192, k=8192)])
        evidence = c.failure_diagnostics()
        self.assertEqual(evidence["scheduler_invocations"][0]["layer_indices"], list(range(13)))
        self.assertFalse(evidence["scheduler_invocations"][0]["complete"])
        self.assertIsNone(c.current_schedule)
        self.assertEqual(c.state, "FAILED")

    def test_placeholder_uses_original_swa_and_never_global_skip_hooks(self):
        c, attn = make_controller(), Attention()
        c._install_attention(attn)
        activate(c)
        c.current_schedule = dict(kind="placeholder", invocation_index=0)
        meta = metadata(False)
        attn.forward(Tensor(1,5120), object(), meta, Tensor(1,5120), cross_layer_runtime=SimpleNamespace())
        self.assertIn("native-project", attn.calls)
        self.assertIn("native-swa-prepare", attn.calls)
        self.assertIn(("decode", True, "2_True_True_512", "swa-index"), attn.calls)
        self.assertEqual(c.decode_skip_events, [])

    def test_schedule_hook_is_restored_by_owned_close(self):
        c, runner = self.bound()
        self.assertIn("execute_main_model", runner.__dict__)
        c.close()
        self.assertNotIn("execute_main_model", runner.__dict__)

    def test_layer_without_native_cpu_identity_rejected(self):
        c = make_controller()
        activate(c)
        c.current_schedule = None
        with self.assertRaisesRegex(ValidationError, "authenticated native"):
            c._record_scheduled_layer(0, metadata())


if __name__ == "__main__":
    unittest.main()
