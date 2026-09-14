"""Cache contract, designed before implementation.

1. Purpose: bound cache-owned payload bytes while active readers retain entries.
2. I/O: put accepts nonempty string keys and nonnegative integer byte counts;
   invalid arguments raise TypeError/ValueError. Capacity/pin refusals return
   False. A lease returns a read-only mapping for all unique keys, or None.
   Payloads are caller-read-only; external references are outside this budget.
   Hits/misses count lease requests; rejected_puts counts capacity/pin refusals;
   evictions counts capacity removals, excluding replacement and clear.
3. Failures: over-budget state, partial eviction on rejected puts, pin leaks,
   duplicate-key underflow, accidental mutation, and concurrent state corruption.
4. Cheapest coverage: deterministic stdlib unit tests of public APIs, including
   coordinated reader/writer threads. No Torch, GPU, or integration fixture.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

from vllm_redknot.cache import CacheManager


class CacheManagerTest(unittest.TestCase):
    def test_lru_evicts_the_least_recently_used_unpinned_entry(self):
        cache = CacheManager(6)
        for key in ("a", "b", "c"):
            self.assertTrue(cache.put(key, key, 2))
        with cache.lease(["a"]) as values:
            self.assertEqual(values, {"a": "a"})
        self.assertTrue(cache.put("d", "d", 2))
        with cache.lease(["b"]) as values:
            self.assertIsNone(values)
        with cache.lease(["a", "c", "d"]) as values:
            self.assertEqual(values, {"a": "a", "c": "c", "d": "d"})
        stats = cache.stats()
        self.assertEqual((stats["bytes"], stats["entries"]), (6, 3))
        self.assertEqual(stats["evictions"], 1)

    def test_oversized_put_preserves_existing_entries(self):
        cache = CacheManager(2)
        cache.put("a", "original", 2)
        self.assertFalse(cache.put("a", "replacement", 3))
        self.assertFalse(cache.put("b", "new", 3))
        with cache.lease(["a"]) as values:
            self.assertEqual(values, {"a": "original"})
        self.assertEqual(cache.stats()["evictions"], 0)
        self.assertEqual(cache.stats()["rejected_puts"], 2)

    def test_pinned_victim_rejects_put_without_partial_eviction(self):
        cache = CacheManager(6)
        for key in ("a", "b", "c"):
            cache.put(key, key, 2)
        with cache.lease(["c"]):
            self.assertFalse(cache.put("large", "new", 5))
            with cache.lease(["a", "b", "c"]) as values:
                self.assertEqual(values, {"a": "a", "b": "b", "c": "c"})
            self.assertEqual(cache.stats()["bytes"], 6)
            self.assertEqual(cache.stats()["evictions"], 0)

    def test_failed_growth_preserves_the_replaced_entry(self):
        cache = CacheManager(6)
        for key in ("a", "b", "c"):
            cache.put(key, key, 2)
        with cache.lease(["b", "c"]):
            self.assertFalse(cache.put("a", "replacement", 3))
            with cache.lease(["a"]) as values:
                self.assertEqual(values, {"a": "a"})
        self.assertEqual(cache.stats()["evictions"], 0)

    def test_replacement_accounts_for_old_bytes_before_eviction(self):
        cache = CacheManager(5)
        cache.put("a", "old", 2)
        cache.put("b", "victim", 3)
        self.assertTrue(cache.put("a", "new", 4))
        with cache.lease(["a"]) as values:
            self.assertEqual(values, {"a": "new"})
        self.assertEqual(cache.stats()["bytes"], 4)
        self.assertEqual(cache.stats()["entries"], 1)
        self.assertEqual(cache.stats()["evictions"], 1)

    def test_nested_duplicate_leases_pin_once_and_release_after_exception(self):
        cache = CacheManager(2)
        cache.put("a", "original", 2)
        with self.assertRaisesRegex(RuntimeError, "reader failed"):
            with cache.lease(["a", "a"]) as values:
                self.assertEqual(values, {"a": "original"})
                self.assertEqual(cache.stats()["pinned_entries"], 1)
                with cache.lease(["a", "a"]):
                    self.assertEqual(cache.stats()["pinned_entries"], 1)
                self.assertFalse(cache.put("a", "smaller", 1))
                self.assertEqual(cache.clear(), 0)
                raise RuntimeError("reader failed")
        self.assertEqual(cache.stats()["pinned_entries"], 0)
        self.assertTrue(cache.put("a", "replacement", 2))

    def test_partial_miss_does_not_pin_or_refresh_present_keys(self):
        cache = CacheManager(2)
        cache.put("a", "a", 1)
        cache.put("b", "b", 1)
        with cache.lease(["a", "missing"]) as values:
            self.assertIsNone(values)
            self.assertEqual(cache.stats()["pinned_entries"], 0)
        cache.put("c", "c", 1)
        with cache.lease(["b", "c"]) as values:
            self.assertEqual(values, {"b": "b", "c": "c"})
        with cache.lease(["a"]) as values:
            self.assertIsNone(values)

    def test_mapping_is_read_only_and_retained_payload_is_not_budgeted(self):
        cache = CacheManager(1)
        payload = object()
        cache.put("a", payload, 1)
        with cache.lease(["a"]) as values:
            self.assertIs(values["a"], payload)
            with self.assertRaises(TypeError):
                values["a"] = object()
        self.assertEqual(cache.clear(), 1)
        self.assertEqual(cache.stats()["bytes"], 0)
        self.assertIs(values["a"], payload)

    def test_clear_removes_only_unpinned_entries(self):
        cache = CacheManager(3)
        cache.put("a", "a", 1)
        cache.put("b", "b", 2)
        with cache.lease(["a"]):
            self.assertEqual(cache.clear(), 1)
            self.assertEqual(cache.stats()["bytes"], 1)
            self.assertEqual(cache.stats()["entries"], 1)
        self.assertEqual(cache.clear(), 1)
        self.assertEqual(cache.clear(), 0)
        self.assertEqual(cache.stats()["bytes"], 0)
        self.assertEqual(cache.stats()["evictions"], 0)

    def test_zero_budget_accepts_zero_bytes_and_empty_lease(self):
        cache = CacheManager(0)
        self.assertTrue(cache.put("zero", None, 0))
        self.assertFalse(cache.put("positive", "x", 1))
        with cache.lease([]) as values:
            self.assertEqual(values, {})
            self.assertEqual(cache.stats()["pinned_entries"], 0)
        with cache.lease(["zero"]) as values:
            self.assertEqual(values, {"zero": None})
        self.assertEqual(cache.stats()["bytes"], 0)

    def test_stats_count_requests_and_are_a_detached_snapshot(self):
        cache = CacheManager(1)
        cache.put("a", "a", 1)
        with cache.lease(["a", "a"]):
            self.assertFalse(cache.put("a", "new", 0))
        with cache.lease(["a", "missing"]):
            pass
        self.assertTrue(cache.put("b", "b", 1))
        expected = {
            "entries": 1,
            "bytes": 1,
            "max_bytes": 1,
            "pinned_entries": 0,
            "hits": 1,
            "misses": 1,
            "evictions": 1,
            "rejected_puts": 1,
        }
        self.assertEqual(cache.stats(), expected)
        snapshot = cache.stats()
        snapshot["bytes"] = 999
        self.assertEqual(cache.stats(), expected)

    def test_invalid_byte_counts_raise_before_mutating_cache(self):
        cache = CacheManager(1)
        cache.put("a", "original", 1)
        before = cache.stats()
        for invalid in (-1, True, False, 1.0, "1", None):
            with self.subTest(nbytes=invalid):
                exception = ValueError if invalid == -1 else TypeError
                with self.assertRaises(exception):
                    cache.put("a", "replacement", invalid)
                self.assertEqual(cache.stats(), before)
        with cache.lease(["a"]) as values:
            self.assertEqual(values, {"a": "original"})

    def test_invalid_keys_and_budgets_are_rejected(self):
        cache = CacheManager(1)
        for invalid in ("", None, 1, False):
            with self.subTest(key=invalid):
                exception = ValueError if invalid == "" else TypeError
                with self.assertRaises(exception):
                    cache.put(invalid, "value", 0)
                with self.assertRaises(exception):
                    with cache.lease([invalid]):
                        self.fail("invalid key unexpectedly yielded a lease")
        with self.assertRaises(TypeError):
            with cache.lease("key"):
                self.fail("a string is not a sequence of cache keys")
        for invalid in (-1, True, False, 1.0, "1", None):
            with self.subTest(max_bytes=invalid):
                exception = ValueError if invalid == -1 else TypeError
                with self.assertRaises(exception):
                    CacheManager(invalid)

    def test_reader_pin_blocks_concurrent_writes_until_release(self):
        cache = CacheManager(2)
        cache.put("a", "a", 1)
        cache.put("b", "b", 1)
        entered, release = Event(), Event()

        def reader():
            with cache.lease(["a", "a"]) as values:
                entered.set()
                self.assertTrue(release.wait(5), "writer did not release reader")
                self.assertEqual(values, {"a": "a"})

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(reader)
            try:
                self.assertTrue(entered.wait(5), "reader did not acquire lease")
                self.assertFalse(cache.put("a", "replacement", 0))
                self.assertFalse(cache.put("large", "large", 2))
                self.assertEqual(cache.stats()["entries"], 2)
                self.assertEqual(cache.clear(), 1)
            finally:
                release.set()
            future.result(timeout=5)
        self.assertEqual(cache.stats()["pinned_entries"], 0)
        self.assertTrue(cache.put("a", "replacement", 2))

    def test_concurrent_put_lease_and_clear_preserve_budget_and_pins(self):
        cache = CacheManager(7)
        start = Barrier(6)

        def exercise(worker):
            start.wait(timeout=5)
            for iteration in range(60):
                key = f"{worker}:{iteration % 3}"
                cache.put(key, key, iteration % 3 + 1)
                with cache.lease([key, key]) as values:
                    if values is not None:
                        self.assertEqual(values, {key: key})
                        self.assertFalse(cache.put(key, "replacement", 0))
                    stats = cache.stats()
                    self.assertGreaterEqual(stats["bytes"], 0)
                    self.assertLessEqual(stats["bytes"], stats["max_bytes"])
                    self.assertGreaterEqual(stats["pinned_entries"], 0)
                    self.assertLessEqual(stats["pinned_entries"], stats["entries"])
                    if iteration % 7 == 0:
                        cache.clear()

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(exercise, worker) for worker in range(6)]
            for future in futures:
                future.result(timeout=10)
        self.assertEqual(cache.stats()["pinned_entries"], 0)
        cache.clear()
        self.assertEqual(cache.stats()["bytes"], 0)
        self.assertEqual(cache.stats()["entries"], 0)


if __name__ == "__main__":
    unittest.main()
