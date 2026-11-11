"""CPU position-aware fixtures for subset qualification, not native identity."""
from types import SimpleNamespace
import unittest

from kv_cache_engram.core import ValidationError
from kv_cache_engram.native_reuse import KVCacheEngramHook, LAYERS
from test_native_reuse import FakeOps, Module, Tensor, inputs


class PositionOps(FakeOps):
    def __init__(self):
        super().__init__()
        self.subsets = []
        self.corrupt_size = None

    def postprocess(self, module, canonical, metadata):
        self.validate_tensor(canonical, metadata.total_q)
        positions = metadata.positions
        if not hasattr(metadata, "canonical_request_binding"):
            self.subsets.append(tuple(positions))
        correction = .5 if (not hasattr(metadata, "canonical_request_binding")
                              and metadata.total_q == self.corrupt_size) else 0
        return Tensor(v+position*.001+correction for v, position in zip(canonical, positions)), None


class NativeSubsetQualificationTests(unittest.TestCase):
    def setUp(self):
        self.ops = PositionOps()
        self.modules = [Module(layer, self.ops) for layer in LAYERS]
        self.hook = KVCacheEngramHook(self.modules, model_sha256="a"*64,
            tokenizer_sha256="b"*64, execution_sha256="c"*64,
            ops=self.ops, enabled=True).install()
        self.addCleanup(self.hook.close)
        ids, docs = inputs()
        self.hook.reset("subset-fixture", "split_identity", input_ids=ids,
                        documents=docs, phase="qualification")

    def call(self, start, end):
        metadata = SimpleNamespace(canonical_request_binding=self.hook.chunk_marker(start, end),
            total_q=end-start, positions=Tensor(range(start, end)), num_reqs=1,
            is_prompt=True, normal_full_view=True)
        hidden = Tensor(self.hook.ids[start:end])
        self.modules[0].forward(hidden, None, metadata, hidden,
            SimpleNamespace(consumer_full_metadata=None))

    def test_noncontiguous_subsets_preserve_actual_nonzero_positions(self):
        self.call(0, 512)
        self.ops.subsets.clear()
        self.call(512, 1536)
        self.assertEqual([len(values) for values in self.ops.subsets], [32, 128, 384, 512])
        for values in self.ops.subsets:
            self.assertEqual(values[0], 512)
            self.assertEqual(values[-1], 1535)
            self.assertEqual(len(values), len(set(values)))
        errors = self.hook.qualifications[-1]["errors"]
        for count in (32, 128, 384, 512):
            self.assertEqual(errors["post_rope_gather"+str(count)], 0)

    def test_subset_only_postprocess_difference_aborts(self):
        self.ops.corrupt_size = 384
        with self.assertRaises(ValidationError):
            self.call(0, 1024)
        self.assertEqual(self.hook.state, "ABORTED")
        self.assertFalse(self.hook.identity_qualified)

    def test_single_row_tail_still_qualifies_each_declared_density(self):
        self.call(0, 1)
        self.assertEqual(self.ops.subsets, [(0.,)]*4)
        errors = self.hook.qualifications[-1]["errors"]
        self.assertTrue(all(errors["post_rope_gather"+str(count)] == 0
                            for count in (32, 128, 384, 512)))


if __name__ == "__main__":
    unittest.main()
