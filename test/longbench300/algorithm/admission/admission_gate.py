"""Wait for actual native preprocessing on every owner before first step.

Never sets Sequence.schedulable. The native multimodal processor owns that
transition; this gate only pumps its original fetch API and observes it.
"""
import hashlib
import json
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def seal(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    return dict(value, sha256=hashlib.sha256(raw).hexdigest())


class ReadinessBoundPort:
    def __init__(self, port, *, rank, gather, timeout_s=120., poll_s=.001, after_all_ready=None):
        require(type(rank) is int and 0 <= rank < 8 and callable(gather), 'Actual eight-rank readiness binding required')
        require(0 < timeout_s <= 600 and 0 <= poll_s <= .1, 'Invalid native readiness wait limits')
        require(after_all_ready is None or callable(after_all_ready), 'Invalid post-readiness callback')
        self.port, self.rank, self.gather = port, rank, gather
        self.timeout_s, self.poll_s, self.after_all_ready = timeout_s, poll_s, after_all_ready
        self.phase = None
        self.reported = True

    def __getattr__(self, name):
        return getattr(self.port, name)

    def begin_phase(self, requests_by_rank, phase):
        require(self.reported and type(phase) is str and phase, 'Previous admission phase not reported')
        require(type(requests_by_rank) is dict and requests_by_rank and all(type(r) is int and 0 <= r < 8 for r in requests_by_rank),
            'Actual owner request mapping required')
        self.requests = dict(requests_by_rank)
        self.request = self.requests.get(self.rank)
        self.phase, self.sequence, self.ready, self.reported = phase, None, False, False
        self.polls, self.first_samples, self.final_samples = 0, None, None
        self.wait_started_ns = self.wait_finished_ns = None

    def submit(self, request):
        require(self.phase is not None and self.request is not None and self.sequence is None
            and request.request_id == self.request.request_id, 'Readiness submission owner mismatch')
        sequence = self.port.submit(request)
        require(sequence is not None and str(sequence.request_id) == request.request_id, 'Native submission returned another Sequence')
        self.sequence = sequence
        return sequence

    def peek_local_status(self):
        processor = self.port.engine.multi_modal_processor
        pending = bool(processor.has_pending()) if processor is not None else False
        return dict(rank=self.rank, phase=self.phase, owner=self.request is not None,
            request_id=None if self.request is None else self.request.request_id,
            submitted=self.sequence is not None,
            actual_sequence_schedulable=None if self.sequence is None else bool(self.sequence.schedulable),
            actual_mapper_present=processor is not None, actual_mapper_has_pending=pending)

    def _pump(self):
        processor = self.port.engine.multi_modal_processor
        if processor is not None:
            # Same API and abort handling used by native LLMEngine.step().
            processor.fetch()
            aborted = processor.fetch_to_abort_requests()
            if aborted:
                self.port.engine.abort_request(aborted)
                raise RuntimeError('Native multimodal preprocessing failed: ' + repr(aborted))

    def _wait_native_ready(self):
        require(self.phase is not None and not self.reported, 'Missing active admission phase')
        self.wait_started_ns = time.perf_counter_ns()
        while True:
            error = None
            try:
                self._pump()
                status = self.peek_local_status()
                require(self.request is None or self.sequence is not None, 'Owner has not submitted its native Sequence')
                if (time.perf_counter_ns() - self.wait_started_ns) / 1e9 > self.timeout_s:
                    raise TimeoutError('Native owner preprocessing did not become ready')
            except BaseException as exc:
                error = dict(type=type(exc).__name__, message=str(exc))
                status = dict(rank=self.rank, phase=self.phase)
            samples = self.gather(dict(status, error=error,
                active_ranks=sorted(self.requests), readiness_poll=self.polls))
            require(type(samples) is list and len(samples) == 8 and [s.get('rank') for s in samples] == list(range(8)),
                'Actual eight-rank native readiness observations required')
            require(all(s['phase'] == self.phase and s['active_ranks'] == sorted(self.requests)
                and s['readiness_poll'] == self.polls for s in samples), 'Readiness world phase/owner mismatch')
            if self.first_samples is None:
                self.first_samples = samples
            self.final_samples = samples
            self.polls += 1
            errors = [s for s in samples if s.get('error') is not None]
            require(not errors, 'Native preprocessing readiness failed: ' + json.dumps(errors, ensure_ascii=False))
            for rank, sample in enumerate(samples):
                expected = self.requests.get(rank)
                require(sample['owner'] is (expected is not None)
                    and sample['request_id'] == (None if expected is None else expected.request_id),
                    'Readiness observations belong to another request')
            if all(not s['actual_mapper_has_pending'] and (not s['owner'] or
                (s['submitted'] and s['actual_sequence_schedulable'] is True)) for s in samples):
                self.wait_finished_ns = time.perf_counter_ns()
                self.ready = True
                if self.after_all_ready is not None:
                    self.after_all_ready()
                return
            if self.poll_s:
                time.sleep(self.poll_s)

    def has_unfinished(self):
        if not self.ready:
            self._wait_native_ready()
        return self.port.has_unfinished()

    def advance(self, unfinished):
        require(self.ready, 'No native model step before every owner is ready')
        return self.port.advance(unfinished)

    def report(self):
        require(self.ready and self.wait_finished_ns is not None, 'No successful actual native readiness evidence')
        self.reported = True
        return seal(dict(schema='s13-actual-native-owner-readiness-v2', rank=self.rank, phase=self.phase,
            actual_all_owner_ready=True, actual_native_mapper_preserved=True, sequence_schedulable_manually_changed=False,
            first_actual_samples=self.first_samples, final_actual_samples=self.final_samples,
            readiness_poll_count=self.polls, wait_started_ns=self.wait_started_ns, wait_finished_ns=self.wait_finished_ns,
            actual_wait_s=(self.wait_finished_ns-self.wait_started_ns)/1e9,
            waiting_in_received_ttft=True, extra_per_decode_step_host_gathers=0))
