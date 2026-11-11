import copy
import unittest

from engine_bridge import NativeEnginePort, TokenRequest, run_serial_request, serial_completed_qps


class Pointers(list):
    def tolist(self):
        return list(self)


class OutputAPI:
    pass


for _field in ("request_id", "token_id", "prompt_len", "finished", "finish_reason"):
    setattr(OutputAPI, "get_" + _field, staticmethod(lambda obj, field=_field: obj[field]))
OutputAPI.get_token_logprobs = staticmethod(lambda obj: obj["logprob"])


class Clock:
    def __init__(self):
        self.value = 1.0

    def __call__(self):
        self.value += .1
        return self.value


class FakeNativeEngine:
    multi_modal_processor = None

    def __init__(self):
        self.requests = []
        self.cursor = 0
        self.trace = []

    def add_request(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        self.cursor = 0
        self.trace.append("submit")
        return object()

    def has_unfinished_requests(self):
        self.trace.append("has_unfinished")
        return self.cursor < 2

    def output(self, finished):
        self.cursor += 1
        return Pointers([dict(request_id=self.requests[-1]["request_id"], token_id=65+self.cursor,
            prompt_len=len(self.requests[-1]["prompt_token_ids"]), logprob=-.1,
            finished=finished, finish_reason="stop" if finished else "")])

    def step(self):
        self.trace.append("step")
        return self.output(False)

    def flush_inflight_reqs(self):
        self.trace.append("flush")
        return self.output(True)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeNativeEngine()
        self.port = NativeEnginePort(self.engine, output_api=OutputAPI,
            array_factory=list, params_factory=lambda **kw: kw)
        self.request = TokenRequest("r1", (0, 10, 11), 8, "full", 1.0)
        self.clock = Clock()
        self.events = []

    def run_request(self, **kw):
        return run_serial_request(self.port, kw.pop("request", self.request), rank=0,
            barrier=lambda: self.engine.trace.append("barrier"),
            prepare=lambda r: self.engine.trace.append("prepare"),
            finish=lambda r: self.engine.trace.append("finish"),
            emit=kw.pop("emit", self.events.append), clock=self.clock, **kw)

    def test_first_token_emitted_before_later_steps_and_final_flush(self):
        def emit(event):
            if event["event"] == "raw_token" and not self.events:
                self.assertEqual(self.engine.trace.count("step"), 1)
                self.assertNotIn("flush", self.engine.trace)
            self.events.append(event)
        report = self.run_request(emit=emit)
        self.assertEqual(report["output_ids"], [66, 67, 68])
        self.assertEqual(self.engine.trace.count("flush"), 1)
        self.assertLess(report["first_raw_token_s"], report["terminal_s"])
        self.assertGreater(report["raw_ttft_s"], report["engine_raw_ttft_s"])
        self.assertTrue(report["raw_token_is_not_http_first_byte"])

    def test_persistent_engine_accepts_two_requests(self):
        first = self.run_request()
        second = self.run_request(request=TokenRequest("r2", (0, 20), 5, "s2", self.clock()))
        self.assertEqual([r["request_id"] for r in self.engine.requests], ["r1", "r2"])
        self.assertEqual(second["arm"], "s2")
        self.assertEqual(self.engine.requests[0]["sampling_params"]["ignore_eos"], False)
        qps = serial_completed_qps([first, second])
        self.assertEqual(qps["completed_requests"], 2)
        self.assertEqual(qps["concurrency"], 1)
        self.assertFalse(qps["saturated_qps_claim"])

    def test_submit_barrier_precedes_collective_work(self):
        self.run_request()
        self.assertEqual(self.engine.trace[:5], ["prepare", "barrier", "submit", "barrier", "has_unfinished"])

    def test_copies_borrowed_pointer_values(self):
        self.port.submit(self.request)
        row = self.engine.output(False)[0]
        self.engine.step = lambda: Pointers([row])
        copied = self.port.advance(True)
        row["token_id"] = 999
        self.assertEqual(copied[0]["token_id"], 66)

    def test_foreign_output_is_rejected(self):
        original = self.engine.output
        def output(finished):
            rows = original(finished)
            rows[0]["request_id"] = "foreign"
            return rows
        self.engine.output = output
        with self.assertRaisesRegex(RuntimeError, "foreign"):
            self.run_request()

    def test_no_terminal_drain_is_rejected(self):
        self.engine.flush_inflight_reqs = lambda: Pointers([])
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            self.run_request()

    def test_inactive_rank_participates_without_real_submission(self):
        self.engine.has_unfinished_requests = lambda: False
        self.engine.flush_inflight_reqs = lambda: Pointers([])
        result = run_serial_request(self.port, self.request, rank=1, barrier=lambda: None,
            prepare=lambda r: None, finish=lambda r: None, emit=self.events.append, clock=self.clock)
        self.assertIsNone(result)
        self.assertEqual(self.engine.requests, [])
        self.assertEqual(self.events, [])

    def test_callback_evidence_is_bound_to_final_request_report(self):
        report = run_serial_request(self.port, self.request, rank=0, barrier=lambda:None,
            prepare=lambda r:dict(effective_arm="full", fallback_reason="short"),
            finish=lambda r:dict(all8_receipts="actual-ref"), emit=self.events.append, clock=self.clock)
        self.assertEqual(report["preparation"]["fallback_reason"], "short")
        self.assertEqual(report["completion"]["all8_receipts"], "actual-ref")

    def test_qps_explicit_window_keeps_failed_tail_and_all_failed_window(self):
        report = self.run_request()
        end = report["drained_s"]+50
        value = serial_completed_qps([report], observation_start_s=1, observation_end_s=end, failed_requests=1)
        self.assertEqual(value["qps"],1/(end-1))
        self.assertEqual(value["failed_requests"],1)
        self.assertTrue(value["benchmark_qps_eligible"])
        empty = serial_completed_qps([], observation_start_s=1, observation_end_s=51, failed_requests=2)
        self.assertEqual(empty["qps"],0)
        self.assertFalse(serial_completed_qps([report])["benchmark_qps_eligible"])

    def test_validation_fails_before_native_entry(self):
        invalid = TokenRequest("r1", (0, 10), 270336, "full", 0)
        with self.assertRaises(ValueError):
            self.run_request(request=invalid)
        self.assertEqual(self.engine.requests, [])


if __name__ == "__main__":
    unittest.main()
