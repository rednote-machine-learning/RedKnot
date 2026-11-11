"""S1 geometry and Torch CPU cache assembly; native GPU gates run on real requests."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace as NS

from kv_cache_engram.s1_cache import S1Cache, SWA_LAYERS, GLOBAL_SOURCES, fixed_selection
from kv_cache_engram.s1_attention import S1Attention, local_memory_plan

try:
    import torch
except ImportError:
    torch = None


def doc(start, end, key="a"):
    return dict(start=start,end=end,key=key*64)


class S1MemoryGeometryTests(unittest.TestCase):
    def test_tail128_requires127_halo_rows_and_no_extra_links(self):
        docs = [doc(0,4096)]
        p = local_memory_plan(docs,4096,0,4096,list(range(3968,4096)))
        self.assertEqual(p["memory_positions"],list(range(3841,4096)))
        self.assertEqual(p["source_kinds"].count("halo"),127)
        self.assertFalse(p["extra_nonlocal_swa_links"])

    def test_nondocument_gaps_use_native_previous_tail_and_online_gap(self):
        docs = [doc(3,303),doc(310,610,"b")]
        selected = fixed_selection(docs,615)
        p = local_memory_plan(docs,615,0,615,selected)
        self.assertEqual(selected[:3],[0,1,2])
        self.assertTrue(all(x in selected for x in range(303,310)))
        self.assertTrue(all(x in p["memory_positions"] for x in range(176,304)))
        self.assertEqual(p["local_lengths"][:3],[1,2,3])

    def test_cross_chunk_tail_requires_previous_online_members(self):
        docs = [doc(0,8300)]
        with self.assertRaisesRegex(ValueError,"missing local"):
            local_memory_plan(docs,8300,8192,8300,list(range(8192,8300)))
        p = local_memory_plan(docs,8300,8192,8300,list(range(8192,8300)),list(range(8172,8192)))
        self.assertEqual(p["source_kinds"].count("previous"),20)
        self.assertEqual(p["memory_positions"][0],8065)

    def test_empty_middle_chunk_has_no_queries_or_hidden_revival(self):
        p = local_memory_plan([doc(0,17000)],17000,0,8192,[])
        self.assertEqual(p["memory_positions"],[])
        self.assertEqual(p["local_lengths"],[])

    def test_short_document_and_adaptive_extra_query_are_rejected(self):
        with self.assertRaises(Exception):
            fixed_selection([doc(0,255)],255)
        with self.assertRaises(Exception):
            local_memory_plan([doc(0,300)],300,0,300,[1]+list(range(172,300)))

    def test_halo_does_not_change_original_window_to_tail_only(self):
        docs = [doc(0,300),doc(300,600,"b")]
        p = local_memory_plan(docs,600,0,600,fixed_selection(docs,600))
        self.assertNotIn(300,p["memory_positions"])
        self.assertIn(345,p["memory_positions"])


class ReferenceOps:
    """Test-only CPU oracle; never imported by the native implementation."""
    def normalize_swa(self,mod,raw):
        return raw.clone()
    def prepare_swa(self,mod,canonical,positions):
        return canonical.reshape(-1,1,512).clone()
    def compress_canonical(self,mod,raw):
        if mod.kv_compress_ratio == 1:
            return raw.clone()
        x = raw.reshape(-1,2,1024)
        weights = torch.softmax(x[:,:,:512].float(),dim=1)
        return (weights*x[:,:,512:].float()).sum(1).bfloat16()
    def store_main_and_index(self,mod,canonical,cache,meta):
        cache.written = canonical.clone()
    def update_compressor_carry(self,mod,raw,cache,meta,end):
        cache.carry = raw[-1:].clone()
    def verify_native_store(self,mod,raw,cache,meta):
        cache.verify_called = True


@unittest.skipIf(torch is None,"Torch CPU required")
class S1TorchCacheTests(unittest.TestCase):
    def make_cache(self,length=300):
        layers = [NS(self_attn=NS(layer_idx=i,kv_compress_ratio=1 if i>=20 else 2,
                                 ratio_key="1" if i>=20 else "2")) for i in range(40)]
        cache = S1Cache(layers=layers,scope={"checkpoint":"fixture"},torch_module=torch,ops=ReferenceOps())
        cache.begin_capture("a"*64,length)
        for begin,end in ((0,100),(100,length)):
            meta = NS(canonical_request_binding=dict(start=begin,end=end))
            for layer in SWA_LAYERS:
                values = torch.arange(begin,end,dtype=torch.bfloat16)[:,None].expand(-1,512).contiguous()
                cache.capture_swa(layers[layer].self_attn,values,meta)
            for layer in GLOBAL_SOURCES:
                width,dtype = (512,torch.bfloat16) if layer==20 else (1024,torch.float32)
                values = torch.arange(begin,end,dtype=torch.float32)[:,None].expand(-1,width).contiguous().to(dtype)
                if width==1024:
                    values[:,:512] = 0
                cache.capture_global(layers[layer].self_attn,values,meta)
        cache.finish_capture()
        cache.prepare_case(documents=[doc(0,length)],input_tokens=length,index_device="cpu")
        return cache,layers

    def test_only_halo_and_three_raw_sources_are_persistent(self):
        cache,_ = self.make_cache()
        report = cache.report()
        self.assertEqual(report["offline_bank_entry_count"],40)
        self.assertEqual(report["offline_swa_halo_rows"],37*128)
        self.assertEqual(report["offline_swa_halo_bytes"],37*128*512*2)
        self.assertEqual(report["offline_global_raw_bytes"],300*9216)
        self.assertFalse(report["full_swa_bank_persisted"])
        self.assertEqual(cache.entry("a"*64,"swa_halo",3).shape,(128,512))

    def test_global_capture_rejects_bf16_gate_latent_payload(self):
        cache,layers = self.make_cache()
        cache.begin_capture("b"*64,300)
        with self.assertRaises(Exception):
            cache.capture_global(layers[8].self_attn,torch.zeros(300,1024,dtype=torch.bfloat16),
                NS(canonical_request_binding=dict(start=0,end=300)))

    def test_immutable_bank_mutation_detected(self):
        cache,_ = self.make_cache()
        cache.entry("a"*64,"global_raw",8)[0,0] = 42
        with self.assertRaises(Exception):
            cache.entry("a"*64,"global_raw",8)

    def test_phase_global_base_is_fresh_and_raw_bank_unchanged(self):
        cache,_ = self.make_cache()
        before = cache.entry("a"*64,"global_raw",8).clone()
        cache.begin_phase("one")
        cache.global_bases[8][0] = 0
        cache.begin_phase("two")
        self.assertTrue(torch.equal(cache.entry("a"*64,"global_raw",8),before))
        self.assertTrue(torch.equal(cache.global_bases[8][0],torch.full((512,),0.5,dtype=torch.bfloat16)))

    def test_mixed_global_group_updates_only_selected_members(self):
        cache,layers = self.make_cache(301)
        cache.begin_phase("fixture")
        selected = cache.selected
        online = torch.zeros((128,1024),dtype=torch.float32)
        online[:,512:] = 1000
        meta = NS(positions_c={"2":torch.arange(0,300,2,dtype=torch.int32)})
        ctx = NS(start=0,end=301,positions_cpu=selected,mode="sensitive",verify=True)
        native_cache = NS()
        cache.global_update(layers[8].self_attn,online,native_cache,meta,ctx)
        # 173 is the first tail; absolute pair172/173 mixes offline172 with online1000.
        self.assertTrue(torch.equal(native_cache.written[86],torch.full((512,),586,dtype=torch.bfloat16)))
        self.assertTrue(native_cache.verify_called)
        self.assertEqual(cache.global_events[-1]["online_projection_positions"],selected)
        self.assertEqual(cache.global_carry[8][0],300)

    def test_odd_chunk_carry_preserves_online_previous_member(self):
        cache,layers = self.make_cache(301)
        cache.begin_phase("fixture")
        state = NS()
        for begin,end in ((0,175),(175,301)):
            selected = [p for p in cache.selected if begin<=p<end]
            raw = torch.zeros((len(selected),1024),dtype=torch.float32)
            raw[:,512:] = 1000
            meta = NS(positions_c={"2":torch.arange(begin//2*2,end//2*2,2,dtype=torch.int32)})
            cache.global_update(layers[8].self_attn,raw,state,meta,
                NS(start=begin,end=end,positions_cpu=selected,mode="sensitive",verify=False))
        self.assertTrue(torch.equal(state.written[0],torch.full((512,),1000,dtype=torch.bfloat16)))

    def test_source20_c1_overlay_and_no_source2_update(self):
        cache,layers = self.make_cache()
        cache.begin_phase("fixture")
        state = NS()
        raw = torch.full((128,512),777,dtype=torch.bfloat16)
        cache.global_update(layers[20].self_attn,raw,state,
            NS(positions_c={"1":torch.arange(300,dtype=torch.int32)}),
            NS(start=0,end=300,positions_cpu=cache.selected,mode="sensitive",verify=False))
        self.assertTrue(torch.equal(state.written[172:],raw))
        self.assertEqual(cache.global_events[0]["affected_compressed_ids"],list(range(172,300)))
        with self.assertRaises(Exception):
            cache.global_update(layers[2].self_attn,raw,state,NS(),NS(start=0,end=300))

    def test_real_torch_memory_gather_matches_position_values(self):
        cache,layers = self.make_cache()
        cache.begin_phase("fixture")
        attention = S1Attention(cache,torch_module=torch)
        positions = cache.selected
        values = torch.tensor(positions,dtype=torch.bfloat16)[:,None,None].expand(-1,1,512).contiguous()
        meta = NS(positions=torch.tensor(positions,dtype=torch.int32))
        memory,absolute,indices,plan = attention._memory(layers[3].self_attn,values,meta,
            NS(start=0,end=300,positions_cpu=positions))
        self.assertTrue(torch.equal(memory[:,0,0],absolute.to(torch.bfloat16)))
        self.assertTrue(torch.equal(absolute[indices[0].long()],torch.arange(45,173,dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
