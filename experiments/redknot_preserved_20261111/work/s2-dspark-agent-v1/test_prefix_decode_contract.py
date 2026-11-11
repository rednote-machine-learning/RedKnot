import copy
import unittest
from dataclasses import dataclass
from types import SimpleNamespace as NS
from prefix_decode_contract import (validate_dflash_swa_coverage,
    validate_fresh_prefix_resume, validate_stable_graph_storage,inspect_native_decode_storage,native_graph_storage)


class MetaTensor:
    """Metadata-only fixture; accidental device reads have no implementation."""
    dtype='uint8'
    device='cuda:fixture'
    def __init__(self,ptr,shape,strides,offset=0,storage_bytes=8192):
        self.ptr,self.shape,self.strides,self.offset,self.storage_bytes=ptr,shape,strides,offset,storage_bytes
    def element_size(self):return 1
    def untyped_storage(self):return NS(data_ptr=lambda:self.ptr,nbytes=lambda:self.storage_bytes)
    def data_ptr(self):return self.ptr+self.offset
    def storage_offset(self):return self.offset
    def stride(self):return self.strides


@dataclass
class GraphInput:
    tokens:object


def actual_metadata_fixture():
    state=MetaTensor(1000,[2,4096],[4096,1])
    blocks=MetaTensor(10000,[4,512],[512,1])
    views=[NS(ica_cache=None,ca_cache=None,ica_state=None,ca_state=None,swa_cache=None) for _ in range(41)]
    views[0].ca_cache=MetaTensor(10000,[4,1,1,256],[512,256,256,1])
    views[40].swa_cache=MetaTensor(1000,[2,134,1,2],[4096,2,2,1],offset=100)
    pool=NS(nextn=5,token_ids=MetaTensor(20000,[10],[1]),logprobs=MetaTensor(30000,[10,20],[20,1]),
        confidence=None,logprobs_index=None)
    graph_input=GraphInput(MetaTensor(40000,[6],[1]))
    main=NS(graph=object(),input_metadata_source=graph_input,input_metadata_device=graph_input,output=graph_input)
    draft=NS(graph=object(),decode_input=graph_input,output=graph_input)
    ce=NS(gpu_states=state,gpu_blocks=blocks,cfg=NS(num_hidden_layers=40,attn_module_list=[object()]*42,
        window_size=[128]),gpu_cache=NS(cache_list=views))
    mr=NS(model=NS(MTP_module=NS(mtp_info_pool=pool)),cuda_graph_runner=NS(graphs={6:main}),
        dflash_cuda_graph_runner=NS(graphs={1:draft}))
    return NS(driver_worker=NS(cache_engine=ce,model_runner=mr))


def descriptors():
    before = dict(request_id='new',state_id=1,block_table=[4,5],num_cached_tokens=0,
        previous_num_tokens=0,swa_valid_start=0,block_size=128,prompt_tokens=130,
        output_tokens=0,schedule_status='WAITING',mtp_info_slot=0,mtp_has_draft_info=False,
        mtp_last_output_token=129,last_copy_state_token_idx=0)
    before.update({key:0 for key in ('mtp_accepted_count','mtp_draft_count','mtp_accept_rate_count',
        'mtp_prev_accept_rate_count','mtp_confidence_count','mtp_prev_confidence_count',
        'mtp_process_count','mtp_schedule_count','total_inflight_tokens','last_inflight_tokens')})
    after=dict(before,num_cached_tokens=128,previous_num_tokens=128)
    checkpoint=dict(is_prompt=True,is_placeholder=False,seqlen_q=128,seqlen_k=128,
        committed_num_cached_tokens=128,swa_valid_start=0,source_prompt_tokens=130)
    return checkpoint,before,after


class ContractTests(unittest.TestCase):
    def test_actual_cache_and_graph_inventory_without_payload_reads(self):
        engine=actual_metadata_fixture()
        result=inspect_native_decode_storage(engine)
        self.assertEqual(result['dflash_swa_coverage']['dflash_layer_ids'],[40])
        self.assertEqual(len(result['all_native_cache_views']),2)
        before=native_graph_storage(engine)
        engine.driver_worker.model_runner.model.MTP_module.mtp_info_pool.token_ids.ptr=999
        with self.assertRaisesRegex(ValueError,'replaced'):
            validate_stable_graph_storage(before,native_graph_storage(engine))

    def test_actual_view_outside_raw_slot_or_missing_dflash_fails(self):
        engine=actual_metadata_fixture()
        engine.driver_worker.cache_engine.gpu_cache.cache_list[40].swa_cache.ptr=777
        with self.assertRaisesRegex(ValueError,'covered'):
            inspect_native_decode_storage(engine)
        engine=actual_metadata_fixture()
        engine.driver_worker.cache_engine.gpu_cache.cache_list[40].swa_cache=None
        with self.assertRaisesRegex(ValueError,'Missing'):
            inspect_native_decode_storage(engine)

    def test_fresh_resume_rejects_stale_proposal_and_rollback_baseline(self):
        c,b,a=descriptors()
        self.assertFalse(validate_fresh_prefix_resume(list(range(130)),list(range(128)),c,b,a)['numerical_qualification'])
        for key,value in [('mtp_info_slot',1),('mtp_has_draft_info',True),
                          ('mtp_draft_count',5),('previous_num_tokens',0),('total_inflight_tokens',1)]:
            bad=dict(a);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_fresh_prefix_resume(list(range(130)),list(range(128)),c,b,bad)

    def test_prefix_boundary_requires_exact_ids_and_real_suffix(self):
        c,b,a=descriptors()
        for prompt in [list(range(129)),list(range(1,131))]:
            with self.assertRaises(ValueError):
                validate_fresh_prefix_resume(prompt,list(range(128)),c,b,a)
        bad=dict(c,source_prompt_tokens=400)
        with self.assertRaisesRegex(ValueError,'cutoff'):
            validate_fresh_prefix_resume(list(range(130)),list(range(128)),bad,b,a)
        c['is_prompt']=False
        with self.assertRaises(ValueError):
            validate_fresh_prefix_resume(list(range(130)),list(range(128)),c,b,a)

    def test_full_raw_state_must_cover_every_dflash_ring(self):
        raw=dict(storage_ptr=1000,storage_bytes=3072,shape=[3,1024],stride_bytes=[1024,1])
        views=[dict(layer_id=i,storage_ptr=1000,byte_offset=j*300,element_bytes=1,
            shape=[3,134,1,2],stride_bytes=[1024,2,2,1]) for j,i in enumerate([40,41])]
        validate_dflash_swa_coverage(raw,views,[40,41],window_tokens=128,backtrack_tokens=6)
        for bad in [views[:1],[dict(views[0],storage_ptr=2000),views[1]],
                    [views[0],dict(views[1],byte_offset=900)]]:
            with self.assertRaises(ValueError):
                validate_dflash_swa_coverage(raw,bad,[40,41],window_tokens=128,backtrack_tokens=6)

    def test_graph_addresses_cannot_be_swapped_for_bank_tensors(self):
        before={name:{'storage_ptr':i+100,'shape':[3],'stride':[1]} for i,name in enumerate(
            ['gpu_blocks','gpu_states','mtp_info_pool.token_ids','mtp_info_pool.logprobs',
             'main_graph.positions','dflash_graph.golden_tokens'])}
        validate_stable_graph_storage(before,copy.deepcopy(before))
        after=copy.deepcopy(before);after['gpu_states']['storage_ptr']=999
        with self.assertRaises(ValueError):validate_stable_graph_storage(before,after)


if __name__ == '__main__': unittest.main()
