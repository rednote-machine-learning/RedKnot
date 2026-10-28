from __future__ import annotations

import copy
import unittest
from typing import Any

from benchmarks.redknot.native import BOUNDARY, run_phase


def case(sample_id: str = "sample", *, ignore_eos: bool = False, budget: int = 2) -> dict:
    return {"sample_id": sample_id, "input_ids": [1, 2, 3], "max_new_tokens": budget, "ignore_eos": ignore_eos}


class Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> float:
        self.value += 1
        return float(self.value)


class FakeEngine:
    def __init__(self, batches: list[list[tuple]], flush: list[tuple] | None = None, *, reject: bool = False) -> None:
        self.batches = batches
        self.flush = flush or []
        self.reject = reject
        self.requests: list[dict] = []
        self.index = 0
        self.flush_count = 0
        self.aborted: list[str] = []
        self.prepared = 0
        self.barriers = 0

    def prepare_input_ids(self, tokens: list[int]) -> tuple[int, ...]:
        self.prepared += 1
        return tuple(tokens)

    def before_phase(self) -> None:
        self.barriers += 1
        if self.prepared == 0:
            raise AssertionError("barrier must follow CPU preparation")

    def get_num_unfinished_requests(self) -> int:
        return 0

    def add_request(self, **kwargs: Any) -> object | None:
        self.requests.append(kwargs)
        return None if self.reject else object()

    def has_unfinished_requests(self) -> bool:
        return self.index < len(self.batches)

    def _outputs(self, batch: list[tuple]) -> list[dict]:
        return [
            {
                "request_id": self.requests[index]["request_id"] if index >= 0 else "unknown",
                "token_id": token,
                "finished": finished,
                "finish_reason": reason,
            }
            for index, token, finished, reason in batch
        ]

    def step(self) -> list[dict]:
        batch = self.batches[self.index]
        self.index += 1
        return self._outputs(batch)

    def flush_inflight_reqs(self) -> list[dict]:
        self.flush_count += 1
        return self._outputs(self.flush)

    def abort_request(self, ids: list[str]) -> None:
        self.aborted = ids


def run(engine: FakeEngine, cases: list[dict] | None = None, *, clock: Any = None) -> dict:
    return run_phase(
        engine,
        cases or [case()],
        sampling_factory=lambda item: {"max_tokens": item["max_new_tokens"]},
        output_reader=dict,
        clock=clock or Clock(),
    )


class NativePhaseTests(unittest.TestCase):
    def test_ttft_stops_at_first_output_and_flush_keeps_last_token(self) -> None:
        engine = FakeEngine([[(0, 10, False, "")]], [(0, 11, True, "length")])
        result = run(engine)
        self.assertEqual(result["measurement_boundary"], BOUNDARY)
        self.assertEqual(result["elapsed_s"], 4.0)
        self.assertEqual(result["completed_requests"], 1)
        self.assertEqual(result["requests"][0]["output_token_ids"], [10, 11])
        self.assertEqual(result["requests"][0]["ttft_s"], 1.0)
        self.assertEqual(result["requests"][0]["latency_s"], 2.0)
        self.assertEqual(engine.flush_count, 1)
        self.assertEqual(engine.barriers, 1)
        self.assertIsInstance(engine.requests[0]["prompt_token_ids"], tuple)

    def test_per_request_admission_and_output_order(self) -> None:
        engine = FakeEngine([[(1, 20, True, "stop")], [(0, 10, True, "stop")]])
        result = run(engine, [case("first"), case("second")])
        first, second = result["requests"]
        self.assertEqual([first["sample_id"], second["sample_id"]], ["first", "second"])
        self.assertEqual(first["ttft_s"], 3.0)
        self.assertEqual(second["ttft_s"], 1.0)
        self.assertEqual(first["latency_s"], first["ttft_s"])

    def test_no_unfinished_requests_still_flushes(self) -> None:
        engine = FakeEngine([], [(0, 0, True, "stop")])
        self.assertEqual(run(engine)["requests"][0]["output_token_ids"], [0])
        self.assertEqual(engine.flush_count, 1)

    def test_rejected_request_is_not_successful(self) -> None:
        engine = FakeEngine([], reject=True)
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            run(engine)
        self.assertTrue(engine.aborted)

    def test_missing_completion_is_an_error(self) -> None:
        engine = FakeEngine([[(0, 10, False, "")]])
        with self.assertRaisesRegex(RuntimeError, "without completing"):
            run(engine)
        self.assertTrue(engine.aborted)

    def test_unknown_and_duplicate_completion_are_errors(self) -> None:
        for batch, message in (
            ([(-1, 10, True, "stop")], "unknown"),
            ([(0, 10, True, "stop"), (0, 11, True, "stop")], "after completion"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                run(FakeEngine([batch]))

    def test_failed_finish_reason_is_an_error(self) -> None:
        for reason in ("abort", "ve_error", "insufficient_system_resource", "prompt_length_exceeded"):
            with self.subTest(reason=reason), self.assertRaisesRegex(RuntimeError, "unsuccessfully"):
                run(FakeEngine([[(0, 10, True, reason)]]))

    def test_fixed_length_must_produce_full_budget(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ended early"):
            run(FakeEngine([[(0, 10, True, "length")]]), [case(ignore_eos=True)])
        with self.assertRaisesRegex(RuntimeError, "stopped on EOS"):
            run(FakeEngine([[(0, 10, True, "stop")]]), [case(ignore_eos=True, budget=1)])
        result = run(FakeEngine([[(0, 10, False, ""), (0, 11, True, "length")]]), [case(ignore_eos=True)])
        self.assertTrue(result["requests"][0]["finished"])

    def test_over_budget_and_invalid_output_tokens_fail(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            run(FakeEngine([[(0, 10, False, ""), (0, 11, True, "length")]]), [case(budget=1)])
        for invalid in (-1, True, 1.2):
            with self.subTest(token=invalid), self.assertRaisesRegex(RuntimeError, "invalid token"):
                run(FakeEngine([[(0, invalid, True, "stop")]]))

    def test_manifest_shape_is_checked_before_admission(self) -> None:
        base = case()
        for mutation in ({"input_ids": [True]}, {"input_ids": []}, {"input_ids": [2**31]}, {"max_new_tokens": 0}, {"sample_id": ""}):
            bad = {**base, **mutation}
            engine = FakeEngine([])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                run(engine, [bad])
            self.assertEqual(engine.requests, [])
        with self.assertRaisesRegex(ValueError, "unique"):
            run(FakeEngine([]), [base, copy.deepcopy(base)])

    def test_bad_clocks_and_preexisting_work_fail(self) -> None:
        for values in ([2, 1], [float("nan")]):
            ticks = iter(values)
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                run(FakeEngine([]), clock=lambda: next(ticks))
        engine = FakeEngine([])
        engine.get_num_unfinished_requests = lambda: 1
        with self.assertRaisesRegex(RuntimeError, "idle"):
            run(engine)


if __name__ == "__main__":
    unittest.main()
