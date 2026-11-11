"""Dependency-free CPU ownership/row/mHC contracts (not model qualification)."""
import unittest
from types import SimpleNamespace as NS

from kv_cache_engram.early_mhc_adapter import EarlyMHCAdapter, gather_packed_token_rows

try:
    import torch
except ImportError:
    torch = None


class Rows:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values), 1)


class Movable:
    def __init__(self, value):
        self.value, self.moves = value, 0

    def move(self):
        self.moves += 1
        if self.moves != 1:
            raise ValueError("moved twice")
        return self.value


class Ops:
    native_execution = False

    def check_runtime(self):
        pass

    def unwrap(self, value):
        return value.move() if isinstance(value, Movable) else value

    def rows(self, value):
        return (value[0] if isinstance(value, tuple) else value).shape[0]

    def validate_full(self, norm, bf16, rows):
        if self.rows(norm) != rows or self.rows(bf16) != rows:
            raise ValueError("full norm rows mismatch")
        if isinstance(norm, tuple) and self.rows(norm[1]) != rows:
            raise ValueError("SF token rows mismatch")

    def validate_plan(self, plan, reference):
        if (plan.full_rows != self.rows(reference) or not 1 <= plan.selected_count <= plan.full_rows
                or len(plan.selected_rows) != plan.selected_count
                or plan.selected_rows != sorted(set(plan.selected_rows))
                or not all(0 <= i < plan.full_rows for i in plan.selected_rows)):
            raise ValueError("bad selected plan")

    def gather(self, value, ids):
        if isinstance(value, tuple):
            return tuple(self.gather(x, ids) for x in value)
        return Rows(value.values[i] for i in ids)

    def scatter_zero(self, selected, reference, plan):
        if self.rows(selected) != plan.selected_count:
            raise ValueError("bad selected output")
        out = [0.0]*plan.full_rows
        for i, value in zip(plan.selected_rows, selected.values):
            out[i] = value
        return Rows(out)


class Attn:
    def __init__(self, index):
        self.layer_idx, self.calls = index, []

    def forward(self, hidden_states, kv_cache, input_metadata, hidden_states_bf16, cross_layer_runtime=None):
        norm = Ops().unwrap(hidden_states)
        bf16 = Ops().unwrap(hidden_states_bf16)
        self.calls.append((bf16.shape[0], input_metadata))
        return Rows(4*x for x in bf16.values)


class MLP:
    use_mega_moe = False

    def __init__(self):
        self.calls, self.fail = [], False

    def forward(self, norm, bf16, metadata):
        if self.fail:
            raise RuntimeError("MLP failure")
        self.calls.append((bf16.shape[0], metadata, norm))
        return Rows(5*x for x in bf16.values)


class Layer:
    hc_enabled, hc_mult, hc_shifted = True, 4, True
    use_mega_mhc, yoco_captures_shared_hidden = False, False
    engram, dflash_input_index = None, None

    def __init__(self, index=6, fp8=False):
        self.layer_idx, self.self_attn, self.mlp = index, Attn(index), MLP()
        self.norm_calls, self.fp8 = [], fp8

    def _norm(self, hidden, residual, name):
        self.norm_calls.append((name, len(hidden.values)))
        # Scalar fixture of native mHC post then pre. The coefficients are
        # intentionally not identity: zero branch output must still mix trunk.
        trunk = Rows(2*r+3*x for r, x in zip(residual.values, hidden.values))
        bf16 = Rows(.1*x for x in trunk.values)
        norm = (Rows(bf16.values), Rows(100+i for i in range(len(bf16.values)))) if self.fp8 else bf16
        return norm, bf16, trunk

    def forward(self, hidden, cache, metadata, residual, runtime):
        norm, bf16, lazy = self._norm(hidden, residual, "attn")
        out = self.self_attn.forward(Movable(norm), cache, metadata, Movable(bf16), runtime)
        norm, bf16, lazy = self._norm(out, lazy, "ffn")
        return self.mlp.forward(norm, bf16, metadata), lazy


def fixture(index=6, selected=(0, 2), fp8=False):
    layer = Layer(index, fp8)
    metadata = NS(is_prompt=True, num_reqs=1, total_q=4, valid_token_mask=[True]*4,
                  canonical_request_binding=dict(request_id="rid", phase="p", start=8, end=12))
    plan = NS(full_rows=4, selected_rows=list(selected), selected_count=len(selected),
              request_id="rid", phase="p", chunk_start=8, chunk_end=12, plan_sha256="a"*64)
    runtime, cache = NS(consumer_full_metadata=None, consumer_full_hidden=None), object()
    observed = []

    def make_meta(attn, full, p, kv):
        return NS(total_q=p.selected_count, parent=full, valid_token_mask=[full.valid_token_mask[i] for i in p.selected_rows])

    def bridge(attn, norm, bf16, full, active, p, rt, kv, provider):
        observed.append((norm, bf16, full, active, p, rt, kv))
        return NS(output=Rows(4*x for x in bf16.values), plan=p, metadata=active, full_rows=p.full_rows)

    adapter = EarlyMHCAdapter([layer], lambda a,m:plan, bridge, make_meta, None, ops=Ops()).install()
    return NS(layer=layer, metadata=metadata, plan=plan, runtime=runtime, cache=cache, observed=observed, adapter=adapter)


class EarlyMHCAdapterTests(unittest.TestCase):
    def run_layer(self, f):
        return f.layer.forward(Rows([1,2,3,4]), f.cache, f.metadata, Rows([10]*4), f.runtime)

    def test_full_norm_and_mhc_but_selected_sublayers(self):
        f = fixture()
        output, lazy = self.run_layer(f)
        self.assertEqual(f.layer.norm_calls, [("attn",4),("ffn",4)])
        self.assertEqual(output.values[1::2], [0,0])
        self.assertEqual(lazy.values[1::2], [52,64])
        self.assertNotEqual(lazy.values[1::2], [10,10])
        self.assertEqual(f.layer.mlp.calls[0][0], 2)
        self.assertIs(f.layer.mlp.calls[0][1], f.observed[0][3])
        report = f.adapter.report()
        self.assertEqual(report["counts"]["masked_layer_calls"], 1)
        self.assertEqual(report["counts"]["inactive_ffn_zero_rows"], 2)
        self.assertFalse(report["native_execution"])
        f.adapter.close()

    def test_fp8_scale_rows_gathered_with_data(self):
        f = fixture(fp8=True)
        self.run_layer(f)
        self.assertEqual(f.observed[0][0][1].values, [100,102])
        self.assertEqual(f.layer.mlp.calls[0][2][1].values, [100,102])
        f.adapter.close()

    def test_all_selected_matches_unwrapped_fixture(self):
        f = fixture(selected=(0,1,2,3))
        reference = Layer()
        expected = reference.forward(Rows([1,2,3,4]), f.cache, f.metadata, Rows([10]*4), f.runtime)
        actual = self.run_layer(f)
        self.assertEqual([x.values for x in actual], [x.values for x in expected])
        self.assertEqual(len(f.observed), 1)  # Qualification must not native-bypass.
        f.adapter.close()

    def test_decoder_method_unchanged_and_restore_idempotent(self):
        f = fixture()
        self.assertIs(f.layer.forward.__func__, Layer.forward)
        self.run_layer(f)
        f.adapter.close(); f.adapter.close()
        self.assertNotIn("forward", f.layer.self_attn.__dict__)
        self.assertNotIn("forward", f.layer.mlp.__dict__)
        self.assertFalse(hasattr(f.layer.self_attn, "_early_mhc_owner"))

    def test_native_passthrough_and_source_publish_after_call(self):
        f = fixture(index=2)
        f.adapter.get_plan = lambda a,m:None
        published=[]
        f.adapter.publish_native_source = lambda a,m,r:published.append(len(a.calls))
        self.run_layer(f)
        self.assertEqual(published, [1])
        self.assertEqual(f.layer.mlp.calls[0][0], 4)
        self.assertEqual(f.observed, [])
        f.adapter.close()

    def test_decode_does_not_request_sparse_plan(self):
        f = fixture()
        f.metadata.is_prompt = False
        f.adapter.get_plan = lambda *a: (_ for _ in ()).throw(AssertionError("decode plan"))
        self.run_layer(f)
        self.assertEqual(f.observed, [])
        f.adapter.close()

    def test_protected_layers_rejected(self):
        for index in (0,5,8,14,19,20,39):
            f=fixture(index=index)
            with self.assertRaises(ValueError):self.run_layer(f)
            self.assertTrue(f.adapter.poisoned)
            f.adapter.close(allow_incomplete=True)

    def test_bad_row_plans_fail_closed(self):
        for selected in ((), (2,0), (1,1), (0,4)):
            f=fixture(selected=selected)
            with self.assertRaises(ValueError):self.run_layer(f)
            f.adapter.close(allow_incomplete=True)

    def test_foreign_binding(self):
        f=fixture();f.plan.phase="foreign"
        with self.assertRaisesRegex(ValueError,"foreign"):self.run_layer(f)
        self.assertEqual(f.observed,[])
        f.adapter.close(allow_incomplete=True)

    def test_mega_and_yoco_fail_closed(self):
        for kind in ("mega", "yoco"):
            f=fixture()
            if kind=="mega":f.layer.mlp.use_mega_moe=True
            else:f.runtime.consumer_full_metadata=object()
            with self.assertRaises(ValueError):self.run_layer(f)
            f.adapter.close(allow_incomplete=True)

    def test_attention_failure_poison_no_retry(self):
        f=fixture()
        f.adapter.attention_forward=lambda *a: (_ for _ in ()).throw(RuntimeError("after cache mutation"))
        with self.assertRaises(RuntimeError):self.run_layer(f)
        with self.assertRaises(ValueError):self.run_layer(f)
        with self.assertRaises(ValueError):f.adapter.report()
        f.adapter.close(allow_incomplete=True)

    def test_mlp_failure_preserves_incomplete_state(self):
        f=fixture();f.layer.mlp.fail=True
        with self.assertRaises(RuntimeError):self.run_layer(f)
        self.assertIsNotNone(f.adapter.pending)
        with self.assertRaises(ValueError):f.adapter.report()
        with self.assertRaises(ValueError):f.adapter.close()
        f.adapter.close(allow_incomplete=True)

    def test_wrong_metadata_between_attention_and_ffn(self):
        f=fixture()
        f.layer.self_attn.forward(Movable(Rows([1]*4)),f.cache,f.metadata,Movable(Rows([1]*4)),f.runtime)
        altered=NS(**vars(f.metadata))
        with self.assertRaisesRegex(ValueError,"does not match"):
            f.layer.mlp.forward(Rows([1]*4),Rows([1]*4),altered)
        f.adapter.close(allow_incomplete=True)

    def test_unfinished_attention_cannot_start_next_layer(self):
        f=fixture()
        f.layer.self_attn.forward(Movable(Rows([1]*4)),f.cache,f.metadata,Movable(Rows([1]*4)),f.runtime)
        with self.assertRaisesRegex(ValueError,"matching FFN"):
            f.layer.self_attn.forward(Movable(Rows([1]*4)),f.cache,f.metadata,Movable(Rows([1]*4)),f.runtime)
        f.adapter.close(allow_incomplete=True)

    def test_existing_interval_hook_rejected(self):
        f=fixture();f.adapter.close();f.layer._sparse_interval_owner=object()
        with self.assertRaisesRegex(ValueError,"interval"):
            EarlyMHCAdapter([f.layer],lambda *a:None,lambda *a:None,lambda *a:None,None,ops=Ops()).install()

    def test_shared_sublayer_instances_rejected(self):
        a,b=Layer(6),Layer(7);b.mlp=a.mlp
        with self.assertRaisesRegex(ValueError,"shared"):
            EarlyMHCAdapter([a,b],lambda *a:None,lambda *a:None,lambda *a:None,None,ops=Ops())

    def test_zero_branch_still_changes_next_layer_trunk(self):
        f=fixture();out,lazy=self.run_layer(f)
        next_layer=Layer(7)
        norm,bf16,next_lazy=next_layer._norm(out,lazy,"attn")
        self.assertEqual(next_lazy.values[1::2],[104,128])
        f.adapter.close()


@unittest.skipIf(torch is None, "Torch CPU unavailable; native GPU not exercised")
class PackedTokenGatherTests(unittest.TestCase):
    def check_size(self, selected):
        # uint8 is an explicit byte-data fixture, not a claim of CPU FP8 GEMM.
        count = selected+11
        data = (torch.arange(count, dtype=torch.int32)%251).to(torch.uint8)[:, None].expand(count,5120)
        sf_store = torch.zeros((40,(count+3)//4*4),dtype=torch.int32)
        sf = sf_store[:,:count].T
        sf.copy_(torch.arange(count*40,dtype=torch.int32).reshape(count,40))
        before = sf.clone()
        ids = torch.arange(selected,dtype=torch.int64)+5
        out,sf_out = gather_packed_token_rows((data,sf),ids)
        self.assertTrue(torch.equal(out,data.index_select(0,ids)))
        self.assertEqual(out.dtype,torch.uint8)
        self.assertTrue(torch.equal(sf_out,sf.index_select(0,ids)))
        self.assertEqual(tuple(sf_out.shape),(selected,40))
        self.assertEqual(sf_out.stride(),(1,(selected+3)//4*4))
        padding = sf_out.T.as_strided((40,(selected+3)//4*4),((selected+3)//4*4,1))[:,selected:]
        self.assertEqual(int(torch.count_nonzero(padding)),0)
        self.assertNotEqual(out.data_ptr(),data.data_ptr())
        self.assertNotEqual(sf_out.data_ptr(),sf.data_ptr())
        sf_out[0,0] = -1
        self.assertTrue(torch.equal(sf,before))

    def test_packed_32(self):self.check_size(32)
    def test_packed_33(self):self.check_size(33)
    def test_packed_128(self):self.check_size(128)
    def test_packed_129(self):self.check_size(129)
    def test_packed_384(self):self.check_size(384)
    def test_packed_512(self):self.check_size(512)
    def test_packed_1845(self):self.check_size(1845)
    def test_packed_8192(self):self.check_size(8192)

    def test_bf16_data_dtype_preserved(self):
        data=torch.arange(40*8,dtype=torch.float32).reshape(40,8).to(torch.bfloat16)
        sf=torch.arange(40*2,dtype=torch.int32).reshape(40,2)
        ids=torch.tensor([1,4,7,11,39],dtype=torch.int64)
        out,out_sf=gather_packed_token_rows((data,sf),ids)
        self.assertEqual(out.dtype,torch.bfloat16)
        self.assertTrue(torch.equal(out,data.index_select(0,ids)))
        self.assertTrue(torch.equal(out_sf,sf.index_select(0,ids)))

    def test_invalid_sf_and_ids_rejected(self):
        data=torch.ones((8,16));sf=torch.ones((8,2),dtype=torch.int32)
        for hidden,ids in (((data,sf.float()),torch.tensor([1])),
                           ((data,sf[:7]),torch.tensor([1])),
                           ((data,sf),torch.tensor([],dtype=torch.int64)),
                           ((data,sf),torch.tensor([1.]))):
            with self.assertRaises(ValueError):gather_packed_token_rows(hidden,ids)


if __name__ == "__main__":
    unittest.main()
