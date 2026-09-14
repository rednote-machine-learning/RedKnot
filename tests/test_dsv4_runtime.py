"""DSV4 z_off transaction failure paths independent of native GPU kernels."""

import unittest
from dataclasses import replace

try:
    import torch
except ImportError:
    torch = None

from vllm_redknot.cache import CacheManager
from vllm_redknot.dsv4_projection import CachedLocalZ
from vllm_redknot.dsv4_runtime import DSV4LayerSpec, DSV4Runtime
from vllm_redknot.runtime import RedKnotSettings


@unittest.skipUnless(torch is not None, "CPU Torch required for z_off artifacts")
class DSV4RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.settings = RedKnotSettings.from_mapping(
            {
                "local_heads": {"3": [1, 2]},
                "model_revision": "fixture",
                "allow_approximate": True,
                "boundary_tokens": 1,
            }
        )
        self.runtime = DSV4Runtime(self.settings, CacheManager(1 << 20), "fixture")
        self.specs = {
            "l3": DSV4LayerSpec((1, 2), 64, 8, 2, 4),
            "l4": DSV4LayerSpec((1, 2), 64, 8, 2, 128),
        }

    def step(self, mode, tokens=(1, 2, 3), spans=None):
        spans = [(0, len(tokens))] if spans is None else spans
        return self.runtime.step(
            extra_args={
                "redknot": {
                    "mode": mode,
                    "namespace": "test",
                    "allow_approximate": True,
                    "chunks": [{"start": a, "end": b} for a, b in spans],
                }
            },
            token_ids=tokens,
            specs=self.specs,
        )

    def stage(self, state, names=None):
        for name in self.specs if names is None else names:
            spec = self.specs[name]
            state.staged[name] = CachedLocalZ(
                torch.zeros(
                    state.prompt_length, spec.groups * spec.rank, dtype=torch.bfloat16
                ),
                torch.arange(state.prompt_length),
                spec.local_heads,
                spec.num_heads,
                spec.head_dim,
                self.runtime.policy_key(name, spec),
            )

    def test_capture_is_atomic_and_exact_byte_accounted(self):
        with self.step("capture") as state:
            self.stage(state, ["l3"])
            self.assertEqual(self.runtime.cache.stats()["bytes"], 0)
        self.assertEqual(self.runtime.counters["capture_incomplete"], 1)
        self.assertEqual(self.runtime.cache.stats()["bytes"], 0)
        with self.step("capture") as state:
            self.stage(state)
        self.assertEqual(self.runtime.cache.stats()["bytes"], 2 * 3 * (16 * 2 + 8))
        self.assertEqual(self.runtime.counters["capture_committed"], 1)

    def test_exception_and_capacity_never_publish_partial(self):
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with self.step("capture") as state:
                self.stage(state)
                raise RuntimeError("failure")
        self.assertEqual(self.runtime.cache.stats()["bytes"], 0)
        self.runtime.settings = replace(self.settings, max_cache_bytes=1)
        with self.step("capture") as state:
            self.assertEqual(state.reason, "capture_capacity")
        self.assertEqual(self.runtime.cache.stats()["bytes"], 0)

    def test_reuse_requires_sparse_and_projection_visits_and_releases_lease(self):
        with self.step("capture") as state:
            self.stage(state)
        for mark_sparse, mark_project in ((False, True), (True, False)):
            with self.subTest(mark_sparse=mark_sparse):
                with self.assertRaisesRegex(RuntimeError, "every selected layer"):
                    with self.step("reuse") as state:
                        if mark_sparse:
                            state.sparse_layers.update(self.specs)
                        if mark_project:
                            state.projected_layers.update(self.specs)
                self.assertEqual(self.runtime.cache.stats()["pinned_entries"], 0)
                self.assertEqual(self.runtime.counters["reuse_steps"], 0)

    def test_whole_request_miss_and_successful_clean_dirty_runs(self):
        with self.step("capture") as state:
            self.stage(state)
        with self.step("reuse", (9, 1, 2, 3, 4, 5, 6), [(1, 4), (4, 7)]) as state:
            self.assertEqual(state.reason, "cache_miss")
        with self.step("reuse", (9, 1, 2, 3, 8), [(1, 4)]) as state:
            self.assertEqual(state.clean_runs, ((2, 4, 0),))
            self.assertEqual(state.dirty_runs, ((0, 2), (4, 5)))
            state.sparse_layers.update(self.specs)
            state.projected_layers.update(self.specs)
        self.assertEqual(self.runtime.counters["reuse_steps"], 1)
        self.assertEqual(self.runtime.cache.stats()["pinned_entries"], 0)


if __name__ == "__main__":
    unittest.main()
