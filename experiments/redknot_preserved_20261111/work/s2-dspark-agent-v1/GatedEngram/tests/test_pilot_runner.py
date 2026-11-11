"""Pure CPU protocol and timing guards for the isolated native runner."""
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


FILE = Path(__file__).resolve().parents[1] / 'gpu_pilot/run_pilot.py'
spec = importlib.util.spec_from_file_location('pilot_runner_under_test', FILE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def valid_args(**changes):
    value = dict(dp=8, ep=8, cp=1, concurrency=1, max_num_seqs=1,
        mode='quality', arm='recomputed', max_new_tokens=64, max_num_batched_tokens=8192,
        max_model_len=270336, gpu_memory_utilization=.60, skip_warmup=True, repeat=1,
        evaluate_calibration=False, profile_limit=None, limit=None, sample_id=None,
        head_profile=None, intervention_policy=None, cache_dir=None, head_cache_representation='expanded',
        allow_gpu_init=False, batch_invariant=1, sampler_force_topk=1024, prefill_attempt_interval=1,
        seed=0, diagnostic_progress=False, diagnostic_ep_sync=False, diagnostic_finite=False,
        diagnostic_stacks_seconds=0, diagnostic_prefill_breakdown=False, check_only=True,
        cpu_ready_file=None, release_file=None, experiment_root=Path('/cpu-fixture'),
        pilot_protocol=Path('/cpu-fixture/protocol.json'), repo=Path('/cpu-fixture/repo'))
    value.update(changes)
    return SimpleNamespace(**value)


def phases(index=0, *, baseline=8.0, gated=6.0):
    output = []
    for repeat, arm in runner.timing_schedule(index):
        ttft = baseline if arm == 'recomputed' else gated
        output.append(dict(repeat=repeat, arm=arm, timing_valid=True,
            online_global_window_s=ttft + 2,
            record=dict(success=True, output_ids=list(range(32)), online_ttft_s=ttft,
                        online_prepare_to_submit_s=.2, online_elapsed_s=ttft + 1)))
    return output


class PilotRunnerTests(unittest.TestCase):
    def test_fixed_schedule_three_paired_repeats_alternates(self):
        self.assertEqual(runner.timing_schedule(0), [(0, 'recomputed'), (0, 'gated'),
            (1, 'gated'), (1, 'recomputed'), (2, 'recomputed'), (2, 'gated')])
        self.assertEqual(runner.timing_schedule(1), [(0, 'gated'), (0, 'recomputed'),
            (1, 'recomputed'), (1, 'gated'), (2, 'gated'), (2, 'recomputed')])

    def test_paired_online_ttft_units_and_scope(self):
        value = runner.summarize_timing(phases(), 0)
        self.assertEqual(value['recomputed']['online_ttft_p50_s'], 8)
        self.assertEqual(value['gated']['online_ttft_p50_s'], 6)
        self.assertAlmostEqual(value['ttft_speedup'], 8/6)
        self.assertEqual(value['ttft_reduction_s'], 2)
        self.assertEqual(value['ttft_reduction_percent'], 25)
        self.assertTrue(value['online_gate_preparation_included'])
        self.assertFalse(value['offline_build_included'])
        self.assertFalse(value['model_load_included'])
        self.assertFalse(value['saturated_capacity_claim'])

    def test_each_dataset_has_balanced_initial_arm(self):
        for indices in ((2, 4), (3, 5)):
            self.assertEqual({runner.timing_schedule(i)[0][1] for i in indices},
                             {'recomputed', 'gated'})

    def test_slowdown_is_retained_not_clipped_or_rejected(self):
        value = runner.summarize_timing(phases(baseline=6, gated=8), 0)
        self.assertEqual(value['ttft_speedup'], .75)
        self.assertEqual(value['ttft_reduction_s'], -2)
        self.assertLess(value['ttft_reduction_percent'], 0)

    def test_missing_duplicate_or_wrong_order_fails(self):
        bad = [phases()[:-1], phases() + phases()[:1], list(reversed(phases()))]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.summarize_timing(value, 0)
        with self.assertRaises(ValueError):
            runner.summarize_timing(phases(1), 0)

    def test_invalid_measurement_or_fixed_tokens_fail(self):
        mutations = [lambda p: p.update(timing_valid=False),
                     lambda p: p.update(timing_valid=1),
                     lambda p: p['record'].update(success=False),
                     lambda p: p['record'].update(output_ids=list(range(31))),
                     lambda p: p['record'].update(output_ids=list(range(33))),
                     lambda p: p['record'].update(online_ttft_s=0),
                     lambda p: p['record'].update(online_ttft_s=-1),
                     lambda p: p['record'].update(online_prepare_to_submit_s=-.1),
                     lambda p: p['record'].update(online_elapsed_s=1),
                     lambda p: p.update(online_global_window_s=1)]
        for mutate in mutations:
            value = phases()
            mutate(value[0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                runner.summarize_timing(value, 0)

    def test_all_nonfinite_timing_fields_fail(self):
        for key in ('online_ttft_s', 'online_prepare_to_submit_s', 'online_elapsed_s', 'online_global_window_s'):
            for number in (float('nan'), float('inf'), -float('inf')):
                value = phases()
                (value[0] if key == 'online_global_window_s' else value[0]['record'])[key] = number
                with self.subTest(key=key, number=number), self.assertRaises(ValueError):
                    runner.summarize_timing(value, 0)

    def test_cpu_check_only_does_not_require_release(self):
        runner.validate_args(valid_args())

    def test_actual_runner_requires_both_barrier_paths(self):
        for changes in ({}, dict(cpu_ready_file=Path('/ready')), dict(release_file=Path('/release'))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.validate_args(valid_args(check_only=False, **changes))
        runner.validate_args(valid_args(check_only=False, cpu_ready_file=Path('/ready'), release_file=Path('/release')))

    def test_gpu_bypass_and_experiment_mutations_fail_closed(self):
        changes = (dict(allow_gpu_init=True), dict(cp=2), dict(dp=4), dict(ep=4),
            dict(concurrency=4), dict(max_num_seqs=2), dict(max_new_tokens=32), dict(repeat=2),
            dict(mode='throughput'), dict(arm='redknot'), dict(max_model_len=32768),
            dict(max_num_batched_tokens=4096), dict(gpu_memory_utilization=.9),
            dict(skip_warmup=False), dict(evaluate_calibration=True), dict(profile_limit=1),
            dict(limit=1), dict(sample_id='one'), dict(head_profile=Path('/x')),
            dict(intervention_policy=Path('/x')), dict(cache_dir=Path('/x')),
            dict(head_cache_representation='projected'), dict(seed=2), dict(batch_invariant=0),
            dict(sampler_force_topk=5), dict(prefill_attempt_interval=2),
            dict(diagnostic_progress=True), dict(diagnostic_ep_sync=True), dict(diagnostic_finite=True),
            dict(diagnostic_stacks_seconds=1), dict(diagnostic_prefill_breakdown=True))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_args(valid_args(**change))

    def test_check_only_never_imports_accelerator_or_process_group(self):
        args = valid_args()
        identity = dict(fixed_env={})
        manifest = dict(sha256='1' * 64)
        protocol = dict(sha256='2' * 64)
        rows = [dict(index=0, dataset='hotpotqa', split='calibration', input_tokens=262200)]
        profile = SimpleNamespace(identity=lambda args: {'frozen': True})
        base = SimpleNamespace(preflight=lambda args: (manifest, identity),
            read_json=lambda path: protocol, source_identity=lambda repo: {},
            digest=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        native = SimpleNamespace(gpu_runtime_imported=lambda: False)
        scoped = lambda *a, **kw: contextlib.nullcontext()
        original_import = builtins.__import__
        def guarded_import(name, *a, **kw):
            if name.split('.')[0] in ('torch', 'deepinfer', 'quality_process_group', '_config'):
                raise AssertionError('CPU check-only imported ' + name)
            return original_import(name, *a, **kw)
        with mock.patch.object(runner, 'parse_args', return_value=args), \
             mock.patch.object(runner, 'load_runtime', return_value=(base, native, lambda **kw: profile, scoped)), \
             mock.patch.object(runner, 'validate_inputs', return_value=rows), \
             mock.patch.object(runner, 'source_hashes', return_value={}), \
             mock.patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
             mock.patch('builtins.__import__', side_effect=guarded_import), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(runner.main([]), 0)
        result = json.loads(out.getvalue())
        self.assertFalse(result['gpu_initialized'])
        self.assertEqual(result['cases'], rows)


class PilotRawPairTests(unittest.TestCase):
    def setUp(self):
        from gpu_pilot.raw_pairs import RawPairWriter
        self.writer_class = RawPairWriter
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve() / 'new-pair'
        self.docs = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(8)]

    def writer(self, **changes):
        kwargs = dict(documents=self.docs, layers=[3], identity={'fixture': 'CPU bytes, not native model'}, max_bytes=4096)
        kwargs.update(changes)
        value = self.writer_class(self.directory, **kwargs)
        self.addCleanup(value.close)
        return value

    def test_raw_pair_is_exclusive_and_partial_never_publishes(self):
        writer = self.writer()
        writer.append_bytes('capture_independent', 3, self.docs[0], 0, 1, bytes(1024))
        with self.assertRaises(RuntimeError):
            writer.finish()
        self.assertFalse((self.directory / 'pair.json').exists())
        with self.assertRaises(FileExistsError):
            self.writer()

    def test_raw_pair_byte_nan_inf_poison_before_write(self):
        for word in (b'\x80\x7f', b'\xc0\x7f', b'\x80\xff'):
            writer = self.writer_class(self.directory / word.hex(), documents=self.docs,
                layers=[3], identity={'fixture': True}, max_bytes=4096)
            self.addCleanup(writer.close)
            with self.subTest(word=word), self.assertRaises(ValueError):
                writer.append_bytes('capture_independent', 3, self.docs[0], 0, 1, word + bytes(1022))
            self.assertEqual(writer.bytes, 0)
            with self.assertRaises(RuntimeError):
                writer.finish()
            self.assertFalse((writer.directory / 'pair.json').exists())

    def test_raw_pair_rejects_budget(self):
        writer = self.writer(max_bytes=1024)
        writer.append_bytes('capture_independent', 3, self.docs[0], 0, 1, bytes(1024))
        with self.assertRaises(ValueError):
            writer.append_bytes('capture_independent', 3, self.docs[0], 1, 1, bytes(1024))
        self.assertEqual(writer.bytes, 1024)
        with self.assertRaises(RuntimeError):
            writer.finish()

    def test_raw_pair_rejects_gap_and_duplicate(self):
        for start in (0, 2):
            writer = self.writer_class(self.directory / str(start), documents=self.docs,
                layers=[3], identity={'fixture': True}, max_bytes=4096)
            self.addCleanup(writer.close)
            writer.append_bytes('capture_independent', 3, self.docs[0], 0, 1, bytes(1024))
            with self.subTest(start=start), self.assertRaises(ValueError):
                writer.append_bytes('capture_independent', 3, self.docs[0], start, 1, bytes(1024))
            with self.assertRaises(RuntimeError):
                writer.finish()
            self.assertFalse((writer.directory / 'pair.json').exists())

    def test_raw_pair_rejects_foreign_or_noninteger_layer(self):
        for layers in ([True], [0], [8], [19], ['3'], [3, 3]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                self.writer(layers=layers)
        self.assertFalse(self.directory.exists())

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'optional Torch CPU dependency absent')
    def test_bf16_cpu_tensor_serialization_without_gpu(self):
        import torch
        writer = self.writer()
        tensor = torch.tensor([[1.0] * 512], device='cpu', dtype=torch.bfloat16)
        writer('capture_independent', 3, self.docs[0], 0, tensor)
        writer.close()
        path = self.directory / ('capture_independent.layer3.' + self.docs[0] + '.bf16')
        self.assertEqual(path.read_bytes(), b'\x80\x3f' * 512)
        self.assertFalse(torch.cuda.is_initialized())

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'optional Torch CPU dependency absent')
    def test_tensor_path_no_implicit_cast_and_nonfinite_rejected(self):
        import torch
        for index, (number, dtype, shape) in enumerate(((0, torch.float32, (1, 512)),
                                     (float('nan'), torch.bfloat16, (1, 512)),
                                     (0, torch.bfloat16, (1, 511)))):
            writer = self.writer_class(self.directory / str(index),
                documents=self.docs, layers=[3], identity={'fixture': True}, max_bytes=4096)
            self.addCleanup(writer.close)
            with self.subTest(number=number, dtype=dtype, shape=shape), self.assertRaises(ValueError):
                writer('capture_independent', 3, self.docs[0], 0, torch.full(shape, number, dtype=dtype, device='cpu'))
            with self.assertRaises(RuntimeError):
                writer.finish()
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
