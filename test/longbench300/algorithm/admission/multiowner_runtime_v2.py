"""Independent admission barrier/error preservation around frozen DP4 v1.

The original model, index selection, LastKV, world classifier and native
scheduler remain untouched. Real async multimodal preprocessing must finish on
every submitted owner before the first native model step may start. Its time
remains after the original received/submitted clocks.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    return dict(value, sha256=digest(value))


def write_error_once(path, value):
    """Preserve the original failure before potentially blocking shutdown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(seal(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    try:
        with path.open('x') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return False
    return True


def recording_worker_class(base_worker):
    class FailureRecordingWorker(base_worker):
        def __init__(self, *args, failure_dir, root_failure_dir=None, failure_context=None, **kwargs):
            gather = kwargs['phase_gather']
            self.last_actual_scheduler_observations = None
            self.rejected_events = []
            self.failure_dir = Path(failure_dir)
            self.root_failure_dir = None if root_failure_dir is None else Path(root_failure_dir)
            self.failure_context = failure_context

            def observe_gather(local):
                result = gather(local)
                if isinstance(local, dict) and 'ordinal' in local and 'sequences' in local:
                    # These are copied native metadata scalars, never GPU data.
                    self.last_actual_scheduler_observations = json.loads(json.dumps(result))
                return result

            kwargs['phase_gather'] = observe_gather
            super().__init__(*args, **kwargs)

        def configure(self, spec):
            self.last_actual_scheduler_observations = None
            return super().configure(spec)

        def execute_step(self, seqs, outputs):
            try:
                return super().execute_step(seqs, outputs)
            except BaseException as error:
                event = dict(schema='s13-multiowner-rejected-native-step-v2', rank=self.rank,
                    observed_unix=time.time(), error_type=type(error).__name__, error=repr(error),
                    original_traceback=traceback.format_exc(), world_plan=self.spec,
                    prefill_completed=self.prefill_completed,
                    actual_scheduler_observations=self.last_actual_scheduler_observations,
                    actual_prior_world_trace=list(self.world_trace),
                    actual_prior_prefill_trace=list(self.prefill_trace),
                    synthetic_or_reconstructed_samples=False)
                if self.failure_context is not None:
                    try:
                        event['admission_context'] = self.failure_context()
                    except BaseException as context_error:
                        event['context_error'] = repr(context_error)
                self.rejected_events.append(event)
                try:
                    detail = self.failure_dir/('requestphaseerror.rank'+str(self.rank)+'.json')
                    write_error_once(detail, event)
                    if self.root_failure_dir is not None:
                        write_error_once(self.root_failure_dir/('failure.rank'+str(self.rank)+'.json'),
                            dict(status='failed', rank=self.rank, error=repr(error),
                                error_type=type(error).__name__, recorded_before_close=True,
                                detail_path=str(detail), detail_file_sha256=hashlib.sha256(detail.read_bytes()).hexdigest()))
                except BaseException as persistence_error:
                    # Preserve the actual model/scheduler exception even if
                    # the filesystem itself is failing during recovery.
                    print('request phase evidence persistence failed: '+repr(persistence_error), file=sys.stderr)
                raise

    return FailureRecordingWorker


def runtime_class(parent, preserved):
    base = preserved.runtime_class(parent)

    class AdmissionReadyRuntime(base):
        def _error_context(self):
            port = getattr(self, 'admission_port', None)
            return dict(rank=self.rank, phase=self.phase,
                readiness=None if port is None else port.peek_local_status(),
                binding=self.binding, world_plan=self.schedule)

        def _install_multiowner(self):
            super()._install_multiowner()
            from admission_gate import ReadinessBoundPort
            from multiowner_worker import MultiOwnerWorkerSchedule
            cap = self.batch_worker.capability
            self.observer.close()
            self.batch_worker.close()
            worker = recording_worker_class(MultiOwnerWorkerSchedule)
            self.batch_worker = worker(self.model, rank=self.rank, torch_module=self.torch,
                empty_ep_capability=cap, controller=self.controller, engine=self.driver.engine,
                phase_gather=self.driver.gather, allow_native_decode_graphs=True,
                active_ranks=self.installed_owner_ranks, failure_dir=self.out,
                root_failure_dir=self.out.parent,
                failure_context=self._error_context)
            self.observer = preserved.MultiOwnerScheduleObserver(self.driver.engine, self.rank)
            self.port = self.admission_port = ReadinessBoundPort(self.port, rank=self.rank,
                gather=self.driver.gather, timeout_s=float(self.config.get('multiowner_admission_timeout_s', 120.)),
                poll_s=float(self.config.get('multiowner_admission_poll_s', .001)))
            self.driver.barrier()

        def _configure_many(self, requests, phase, mode, method, strategy, verify_sparse=False):
            result = super()._configure_many(requests, phase, mode, method, strategy, verify_sparse)
            self.admission_port.begin_phase(requests, phase)
            return dict(result, original_async_preprocessor_readiness_required=True,
                native_schedulable_flag_manually_overridden=False)

        def _finish_many(self, requests):
            result = super()._finish_many(requests)
            reports = self.driver.gather(self.admission_port.report())
            return dict(result, actual_owner_admission_readiness_reports=reports,
                original_multimodal_mapper_preserved=True, admission_wait_in_received_ttft=True)

        def _execute_many(self, *args, **kwargs):
            try:
                return super()._execute_many(*args, **kwargs)
            except BaseException as error:
                try:
                    detail = self.out/('requestphaseerror.rank'+str(self.rank)+'.json')
                    write_error_once(detail,
                        dict(schema='s13-multiowner-request-phase-error-v2', rank=self.rank,
                            error_type=type(error).__name__, error=repr(error),
                            original_traceback=traceback.format_exc(), context=self._error_context(),
                            actual_rejected_worker_events=list(getattr(self.batch_worker, 'rejected_events', [])),
                            recorded_before_close=True, synthetic_or_reconstructed_samples=False))
                    write_error_once(self.out.parent/('failure.rank'+str(self.rank)+'.json'),
                        dict(status='failed', rank=self.rank, error_type=type(error).__name__, error=repr(error),
                            recorded_before_close=True, detail_path=str(detail),
                            detail_file_sha256=hashlib.sha256(detail.read_bytes()).hexdigest()))
                except BaseException as persistence_error:
                    print('request phase evidence persistence failed: '+repr(persistence_error), file=sys.stderr)
                raise

    return AdmissionReadyRuntime


def load_preserved(cfg):
    here = Path(__file__).resolve().parent
    candidates = ([Path(cfg['multiowner_v1_root'])] if 'multiowner_v1_root' in cfg else []) + [
        here.parent/'frozen/dp4-runtime-v1', here.parent/'dp4-runtime-v1']
    matches = [p.resolve() for p in candidates if (p/'multiowner_runtime.py').is_file()]
    require(matches, 'Explicit frozen DP4 v1 source root is required')
    root = matches[0]
    strict = bool(cfg.get('strict_source_hashes', False))
    for name in ('multiowner_runtime.py', 'multiowner_transport.py', 'multiowner_worker.py',
                 'multiowner_controller.py', 'world_plan_coordinator.py'):
        path = root/name
        require(path.is_file(), 'Missing original DP4 v1 source: '+str(path))
        if strict:
            require(cfg.get('source_sha256', {}).get(str(path)) == hashlib.sha256(path.read_bytes()).hexdigest(),
                'Unpinned or changed original DP4 v1 source: '+str(path))
    for name in ('admission_gate.py', 'multiowner_runtime_v2.py'):
        path = here/name
        require(path.is_file(), 'Missing admission repair source: '+str(path))
        if strict:
            require(cfg.get('source_sha256', {}).get(str(path)) == hashlib.sha256(path.read_bytes()).hexdigest(),
                'Unpinned admission repair source: '+str(path))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(here))
    source = root/'multiowner_runtime.py'
    spec = importlib.util.spec_from_file_location('s13_preserved_dp4_runtime_v1', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def create_runtime(cfg, *, rank, output_dir=None):
    preserved = load_preserved(cfg)
    original = preserved.create_runtime.__wrapped__
    factory = lambda parent: runtime_class(parent, preserved)
    # Copy the factory binding in a private generator function, leaving the
    # preserved module globals and exact original bootstrap code unmodified.
    private = FunctionType(original.__code__, dict(original.__globals__, runtime_class=factory),
        original.__name__, original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    with contextmanager(private)(cfg, rank=rank, output_dir=output_dir) as runtime:
        yield runtime
