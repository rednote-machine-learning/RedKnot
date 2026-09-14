"""CPU oracle for the prefill-to-native-decode paged KV handoff."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_backend as _backend_fixture

from vllm_redknot.ops import relocate_rope
from vllm_redknot.runner import CONTEXT_KEY

torch = _backend_fixture.torch


@unittest.skipUnless(torch is not None, "CPU Torch is required for decode oracle")
class NativeDecodeContractTests(unittest.TestCase):
    def setUp(self):
        self.harness = _backend_fixture.BackendTests(methodName="runTest")
        self.harness.setUp()

    def test_native_decode_consumes_restored_local_kv_after_warm_reuse(self):
        """Gather real mutated cache pages; incoming decode K/V alone cannot pass."""
        h = self.harness
        offline_q, offline_k, offline_v = h.tensors(3)
        h.call(
            "capture",
            [11, 12, 13],
            (0, 3),
            offline_q,
            offline_k,
            offline_v,
            torch.tensor([4, 5, 6]),
        )
        q, k, v = h.tensors(6)
        tokens = [90, 91, 11, 12, 13, 99]
        slots = torch.tensor([8, 9, 10, 11, 0, 1])
        _, cache = h.call("reuse", tokens, (2, 5), q, k, v, slots)

        expected_k, expected_v = k.clone(), v.clone()
        expected_k[3:5, 1:2] = relocate_rope(
            offline_k[1:3, 1:2], torch.arange(1, 3), torch.arange(3, 5), 8, 10000
        )
        expected_v[3:5, 1:2] = offline_v[1:3, 1:2]

        next_q, next_k, next_v = h.tensors(1)
        # The next token occupies logical row 6 in physical block 0, offset 2.
        # This models native do_kv_cache_update running before attention.forward.
        cache[0, :, 2, :8] = next_k[0]
        cache[0, :, 2, 8:] = next_v[0]
        expected_k = torch.cat((expected_k, next_k))
        expected_v = torch.cat((expected_v, next_v))
        expected = _backend_fixture.dense(next_q, expected_k, expected_v)
        stale = _backend_fixture.dense(
            next_q, torch.cat((k, next_k)), torch.cat((v, next_v))
        )
        self.assertGreater((expected - stale).abs().max().item(), 0.001)

        native_calls = []

        def paged_native_decode(
            impl, layer, query, key, value, kv_cache, metadata, output, *unused
        ):
            native_calls.append(metadata.seq_len)
            positions = torch.arange(metadata.seq_len)
            block_ids = metadata.block_table[0, positions // kv_cache.shape[2]]
            offsets = positions % kv_cache.shape[2]
            history_k = kv_cache[block_ids, :, offsets, :8]
            history_v = kv_cache[block_ids, :, offsets, 8:]
            torch.testing.assert_close(history_k, expected_k)
            torch.testing.assert_close(history_v, expected_v)
            output.copy_(_backend_fixture.dense(query, history_k, history_v))
            return output

        metadata = SimpleNamespace(seq_len=7, block_table=torch.tensor([[2, 0]]))
        output = torch.empty_like(next_q)
        before_attention_calls = len(h.calls)
        h.context.slot_mapping[h.layer.layer_name] = torch.tensor([2])
        native_class = type(h.impl).__mro__[1]
        with h.runtime.step(
            extra_args=h.args("reuse", 2, 5),
            token_ids=tokens,
            specs={h.layer.layer_name: h.spec},
            unsupported_reason="not_full_prefill",
        ) as state:
            self.assertEqual(state.mode, "native")
            h.context.additional_kwargs[CONTEXT_KEY] = h.runtime, state
            try:
                with patch.object(native_class, "forward", paged_native_decode):
                    result = h.impl.forward(
                        h.layer, next_q, next_k, next_v, cache, metadata, output
                    )
            finally:
                h.context.additional_kwargs.clear()
        self.assertEqual(native_calls, [7])
        self.assertEqual(len(h.calls), before_attention_calls)
        self.assertEqual(h.runtime.cache.stats()["pinned_entries"], 0)
        torch.testing.assert_close(result, expected, rtol=0.002, atol=0.002)


if __name__ == "__main__":
    unittest.main()
