"""CPU-only launch-contract tests: no subprocess or GPU may actually launch."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


MODULE = Path(__file__).resolve().parents[1] / 'gpu_pilot/launch_pilot.py'
spec = importlib.util.spec_from_file_location('pilot_launcher_under_test', MODULE)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class PilotLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # /var and /tmp aliases may be symlinks on macOS; production refuses them.
        self.root = Path(temporary.name).resolve()
        self.stage = self.root / 'gated_engram_cpu_fixture/GatedEngram'
        (self.stage / 'gpu_pilot').mkdir(parents=True)
        (self.stage / 'kv_cache_engram').mkdir()
        (self.stage / 'tests').mkdir()
        (self.root / 'results').mkdir()
        for name in ('gpu_pilot/run_pilot.py', 'gpu_pilot/launch_pilot.py',
                     'kv_cache_engram/core.py', 'tests/test_cpu.py'):
            (self.stage / name).write_text('# frozen CPU fixture\n')
        self.manifest = self.stage / 'pilot.manifest.json'
        self.protocol = self.stage / 'pilot.protocol.json'
        self.write_sealed(self.protocol, dict(schema=launch.PROTOCOL_SCHEMA, cases=[1, 2]))
        self.write_sealed(self.manifest, dict(schema='independent-native-pilot-input-v1', samples=[1, 2],
                          artifact_kind='kv_cache_engram_native_pilot',
                          pilot_protocol_sha256=launch.read_json(self.protocol)['sha256']))

    @staticmethod
    def write_sealed(path, data):
        path.write_bytes(launch.canonical(launch.seal(data)))

    def plan(self, **changes):
        args = dict(root=self.root, stage_dir=self.stage, name='kce-cpu-fixture-01',
                    manifest=self.manifest, protocol=self.protocol)
        args.update(changes)
        return launch.build_plan(**args)

    def physical(self):
        metadata = dict(pid=999, pgid=999, worker_pids=list(range(1000, 1008)), gpu_ids=list(range(8)))
        gpus = [dict(index=i, uuid='GPU-fixture-' + str(i), memory_mib=1009, utilization=80) for i in range(8)]
        contexts = [dict(uuid='GPU-fixture-' + str(i), pid=1000 + i, memory_mib=1000) for i in range(8)]
        return metadata, gpus, contexts

    def audit(self):
        metadata, gpus, contexts = self.physical()
        return launch.seal(dict(schema='kv-cache-engram-pilot-holder-audit-v1', read_only=True,
                                metadata=metadata, gpus=gpus, contexts=contexts))

    def successful_supervisor(self, plan):
        return dict(schema='redknot-glm53-gpu-lifecycle-v1', holder_restored=True, gpu_ids=list(range(8)),
                    worker_exit=0, finished_unix=1234.5, supervisor_sha256=launch.SUPERVISOR_SHA,
                    holder_helper_sha256=launch.HELPER_SHA, child_environment_sha256=launch.ENV_SHA,
                    runner_sha256=plan['stage_sources']['gpu_pilot/run_pilot.py'])

    def fake_child(self, plan, exit_code=0, result_mutator=None):
        def wait():
            output = Path(plan['paths']['result_dir'])
            output.mkdir()
            value = self.successful_supervisor(plan)
            if result_mutator:
                result_mutator(value)
            (output / 'supervisor.json').write_text(json.dumps(value))
            return exit_code
        return SimpleNamespace(pid=31415, wait=wait)

    def run_fake(self, plan=None, *, child=None, audits=None):
        plan = plan or self.plan()
        child = child or self.fake_child(plan)
        with mock.patch.object(launch, 'verify_lifecycle') as lifecycle, \
             mock.patch.object(launch, 'holder_snapshot', side_effect=audits or [self.audit(), self.audit()]) as holder, \
             mock.patch.object(launch.subprocess, 'Popen', return_value=child) as process, \
             contextlib.redirect_stdout(io.StringIO()):
            code = launch.run_plan(plan)
        return code, plan, process, holder, lifecycle

    def test_fixed_command_and_full_stage_seal(self):
        plan = self.plan()
        self.assertEqual(plan['schema'], launch.PLAN_SCHEMA)
        self.assertEqual(plan['sha256'], launch.digest({k: v for k, v in plan.items() if k != 'sha256'}))
        self.assertEqual(len(plan['stage_sources']), 4)
        self.assertIn('tests/test_cpu.py', plan['stage_sources'])
        args = plan['argv']
        self.assertEqual(args[:2], ['/root/miniconda3/bin/python', str(launch.SUPERVISOR)])
        self.assertEqual(args[args.index('--dp') + 1], '8')
        self.assertEqual(args[args.index('--cp') + 1], '1')
        self.assertEqual(args[args.index('--gpu-memory-utilization') + 1], '0.60')
        self.assertEqual(args[args.index('--experiment-root') + 1], str(self.root))
        self.assertEqual(args[args.index('--max-model-len') + 1], '270336')
        self.assertNotIn('--allow-gpu-init', args)
        for key in ('--ready-file', '--release-file', '--output-dir'):
            self.assertNotIn(key, args)
        self.assertFalse(Path(plan['paths']['plan']).exists())

    def test_check_only_no_process_no_nvidia_no_files(self):
        with mock.patch.object(launch, 'holder_snapshot', side_effect=AssertionError('no GPU query')) as holder, \
             mock.patch.object(launch.subprocess, 'Popen', side_effect=AssertionError('no launch')) as process, \
             mock.patch.object(launch.subprocess, 'check_output', side_effect=AssertionError('no subprocess')) as query, \
             contextlib.redirect_stdout(io.StringIO()) as out:
            status = launch.main(['--name', 'kce-check-only', '--root', str(self.root),
                                  '--stage-dir', str(self.stage), '--manifest', str(self.manifest),
                                  '--pilot-protocol', str(self.protocol), '--check-only'])
        self.assertEqual(status, 0)
        self.assertFalse(json.loads(out.getvalue())['gpu_initialized'])
        self.assertFalse(json.loads(out.getvalue())['lifecycle_identity_checked'])
        holder.assert_not_called(); process.assert_not_called(); query.assert_not_called()
        self.assertEqual(list((self.root / 'results').iterdir()), [])

    def test_arbitrary_runner_flags_not_accepted(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launch.main(['--name', 'kce-test', '--manifest', str(self.manifest),
                         '--pilot-protocol', str(self.protocol), '--allow-gpu-init'])

    def test_invalid_timeout_name_scope(self):
        for kwargs in (dict(timeout=0), dict(timeout=21601), dict(timeout=True),
                       dict(prepare_timeout=0), dict(prepare_timeout=1801),
                       dict(name='../escape'), dict(name='sparse-query20-rag50-v8'),
                       dict(stage_dir=self.root / 'elsewhere')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(**kwargs)

    def test_manifest_and_protocol_are_distinct_and_local(self):
        with self.assertRaises(ValueError):
            self.plan(manifest=self.protocol)
        outside = self.root / 'old-v8.json'
        outside.write_bytes(self.manifest.read_bytes())
        with self.assertRaises(ValueError):
            self.plan(manifest=outside)

    def test_invalid_protocol_seal_and_schema(self):
        value = launch.read_json(self.protocol)
        value['cases'].append(3)
        self.protocol.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.plan()
        self.write_sealed(self.protocol, dict(schema='old-sparse-protocol', cases=[1, 2]))
        with self.assertRaises(ValueError):
            self.plan()

    def test_independent_manifest_binds_exact_protocol(self):
        for changes in (dict(artifact_kind='old_experiment'), dict(pilot_protocol_sha256='0' * 64)):
            value = launch.read_json(self.manifest)
            value.pop('sha256')
            value.update(changes)
            self.write_sealed(self.manifest, value)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan()

    def test_duplicate_json_keys_and_nonfinite_fail(self):
        for payload in ('{"schema":"x","schema":"y"}', '{"a":NaN}'):
            self.protocol.write_text(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.plan()

    def test_stage_symlinks_fail_even_when_not_python(self):
        (self.stage / 'linked-data').symlink_to(self.manifest)
        with self.assertRaises(ValueError):
            self.plan()

    def test_existing_any_evidence_blocks_duplicate(self):
        plan = self.plan()
        for name, item in plan['paths'].items():
            path = Path(item)
            path.write_text('preserve me')
            with self.subTest(path=name), self.assertRaises(ValueError):
                self.plan()
            self.assertEqual(path.read_text(), 'preserve me')
            path.unlink()

    def test_dangling_evidence_symlink_blocks_duplicate(self):
        plan = self.plan()
        Path(plan['paths']['log']).symlink_to(self.root / 'nonexistent')
        with self.assertRaises(ValueError):
            self.plan()

    def test_source_mutation_detected_before_launch(self):
        plan = self.plan()
        (self.stage / 'kv_cache_engram/core.py').write_text('# changed\n')
        with mock.patch.object(launch.subprocess, 'Popen') as process, self.assertRaises(ValueError):
            launch.run_plan(plan)
        process.assert_not_called()

    def test_manifest_mutation_detected(self):
        plan = self.plan()
        self.write_sealed(self.manifest, dict(schema='independent-native-pilot-input-v1', samples=[3]))
        with self.assertRaises(ValueError):
            launch.verify_plan_inputs(plan)

    def test_plan_mutation_detected(self):
        plan = self.plan()
        plan['argv'].append('--allow-gpu-init')
        with self.assertRaises(ValueError):
            launch.verify_plan_inputs(plan)

    def test_actual_launch_disallowed_on_local_root(self):
        with self.assertRaises(ValueError):
            launch.verify_lifecycle(self.root)

    def test_native_repository_cannot_be_actual_stage(self):
        plan = self.plan()
        native = self.root / 'DeepInfer-PR148/GatedEngram'
        native.parent.mkdir()
        self.stage.rename(native)
        self.stage = native
        self.manifest = native / 'pilot.manifest.json'
        self.protocol = native / 'pilot.protocol.json'
        plan = self.plan()
        with mock.patch.object(launch, 'verify_lifecycle'), \
             mock.patch.object(launch.subprocess, 'Popen') as process, self.assertRaises(ValueError):
            launch.run_plan(plan)
        process.assert_not_called()

    def test_all_eight_physical_identities_required(self):
        metadata, gpus, contexts = self.physical()
        launch.validate_physical_holder(metadata, gpus, contexts)
        mutations = (
            lambda m, g, c: c.pop(),
            lambda m, g, c: c.append(dict(uuid='GPU-foreign', pid=4321, memory_mib=1)),
            lambda m, g, c: c[0].update(pid=99999),
            lambda m, g, c: c[0].update(uuid=c[1]['uuid']),
            lambda m, g, c: c[0].update(memory_mib=0),
            lambda m, g, c: m['worker_pids'].__setitem__(0, True),
            lambda m, g, c: g[0].update(index=8),
            lambda m, g, c: m.update(gpu_ids=list(range(7))),
        )
        for mutation in mutations:
            values = copy.deepcopy((metadata, gpus, contexts))
            mutation(*values)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                launch.validate_physical_holder(*values)

    def mock_identity_audit(self, *, foreign=False, changing_pid=False, bad_session=False):
        metadata, gpus, contexts = self.physical()
        calls = {}
        def identity(pid):
            calls[pid] = calls.get(pid, 0) + 1
            return dict(pid=pid, ppid=1 if pid == 999 else 999, pgid=999,
                        sid=123 if bad_session and pid == 1000 else 999,
                        start_ticks=100 + (1 if changing_pid and calls[pid] > 1 else 0), argv=['fixture'])
        lifecycle = SimpleNamespace(read_holder=mock.Mock(return_value=(metadata, {8888} if foreign else set())))
        helper = SimpleNamespace()
        modules = [lifecycle, helper]
        module_specs = [SimpleNamespace(loader=SimpleNamespace(exec_module=mock.Mock())) for _ in modules]
        gpu_csv = '\n'.join(f"{r['index']}, {r['uuid']}, 80, 1009" for r in gpus)
        context_csv = '\n'.join(f"{r['uuid']}, {r['pid']}, 1000" for r in contexts)
        with mock.patch.object(launch, 'verify_lifecycle') as verify, \
             mock.patch.object(launch, 'proc_identity', side_effect=identity), \
             mock.patch.object(launch.importlib.util, 'spec_from_file_location', side_effect=module_specs), \
             mock.patch.object(launch.importlib.util, 'module_from_spec', side_effect=modules), \
             mock.patch.object(launch.subprocess, 'check_output', side_effect=[gpu_csv, context_csv]) as query:
            value = launch.holder_snapshot(self.root)
        return value, lifecycle, query, verify

    def test_original_identity_checked_around_actual_telemetry(self):
        value, lifecycle, query, verify = self.mock_identity_audit()
        self.assertTrue(value['read_only'])
        self.assertEqual(lifecycle.read_holder.call_count, 2)
        self.assertEqual(query.call_count, 2)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(set(value['process_identities']), {str(i) for i in [999, *range(1000, 1008)]})

    def test_pid_reuse_foreign_and_session_mismatch_fail(self):
        for options in (dict(changing_pid=True), dict(foreign=True), dict(bad_session=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.mock_identity_audit(**options)

    def test_atomic_evidence_never_overwrites(self):
        path = self.root / 'evidence.json'
        launch.write_exclusive(path, {'first': True})
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            launch.write_exclusive(path, {'second': True})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob('evidence.json.*.tmp')), [])

    def test_success_is_one_launch_and_two_physical_audits(self):
        code, plan, process, holder, lifecycle = self.run_fake()
        self.assertEqual(code, 0)
        process.assert_called_once()
        self.assertEqual(holder.call_count, 2)
        self.assertEqual(lifecycle.call_count, 1)
        options = process.call_args.kwargs
        self.assertTrue(options['start_new_session'])
        self.assertEqual(options['stdin'], launch.subprocess.DEVNULL)
        outcome = launch.read_sealed(plan['paths']['outcome'])
        self.assertTrue(outcome['successful'])
        self.assertTrue(outcome['holder_physically_verified'])
        self.assertEqual(outcome['supervisor_pid'], 31415)
        for name in ('plan', 'before', 'started', 'after', 'outcome', 'log'):
            self.assertTrue(Path(plan['paths'][name]).is_file())
        # The same name can never silently retry.
        with self.assertRaises(ValueError):
            launch.run_plan(plan)

    def test_prelaunch_foreign_holder_failure_launches_nothing(self):
        code, plan, process, holder, _ = self.run_fake(audits=[ValueError('foreign contexts'), ValueError('foreign contexts')])
        self.assertEqual(code, 1)
        process.assert_not_called()
        self.assertEqual(holder.call_count, 2)
        outcome = launch.read_sealed(plan['paths']['outcome'])
        self.assertFalse(outcome['supervisor_started'])
        self.assertFalse(outcome['holder_physically_verified'])

    def test_supervisor_failure_still_physically_audited(self):
        plan = self.plan()
        code, _, process, holder, _ = self.run_fake(plan, child=self.fake_child(plan, exit_code=3))
        self.assertEqual(code, 1)
        self.assertEqual(holder.call_count, 2)
        process.assert_called_once()
        outcome = launch.read_sealed(plan['paths']['outcome'])
        self.assertTrue(outcome['holder_physically_verified'])
        self.assertFalse(outcome['successful'])
        self.assertEqual(outcome['supervisor_exit'], 3)

    def test_physical_failure_cannot_claim_restored(self):
        code, plan, process, _, _ = self.run_fake(audits=[self.audit(), ValueError('missing GPU 7')])
        self.assertEqual(code, 1)
        process.assert_called_once()
        outcome = launch.read_sealed(plan['paths']['outcome'])
        self.assertFalse(outcome['successful'])
        self.assertFalse(outcome['holder_physically_verified'])
        self.assertFalse(Path(plan['paths']['after']).exists())

    def test_supervisor_identity_and_warning_fail_closed(self):
        for changes in (dict(holder_restored=False), dict(worker_exit=1), dict(supervisor_sha256='0' * 64),
                        dict(holder_helper_sha256='0' * 64), dict(child_environment_sha256='0' * 64),
                        dict(runner_sha256='0' * 64), dict(cleanup_warnings=['unauthenticated child']),
                        dict(error='failed'), dict(cleanup_error='missing holder')):
            plan = self.plan(name='kce-validation-' + str(len(str(changes))))
            value = self.successful_supervisor(plan)
            value.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                launch.verify_supervisor_result(value, plan)

    def test_launch_creation_exception_still_audited_no_retry(self):
        plan = self.plan()
        with mock.patch.object(launch, 'verify_lifecycle'), \
             mock.patch.object(launch, 'holder_snapshot', side_effect=[self.audit(), self.audit()]) as holder, \
             mock.patch.object(launch.subprocess, 'Popen', side_effect=OSError('spawn failed')) as process:
            self.assertEqual(launch.run_plan(plan), 1)
        process.assert_called_once()
        self.assertEqual(holder.call_count, 2)
        self.assertFalse(launch.read_sealed(plan['paths']['outcome'])['supervisor_started'])

    def test_postrun_source_changes_fail_closed(self):
        plan = self.plan()
        child = self.fake_child(plan)
        wait = child.wait
        def changed():
            result = wait()
            (self.stage / 'kv_cache_engram/core.py').write_text('# unexpected changed source\n')
            return result
        child.wait = changed
        code, _, _, holder, _ = self.run_fake(plan, child=child)
        self.assertEqual(code, 1)
        self.assertEqual(holder.call_count, 2)


if __name__ == '__main__':
    unittest.main()
