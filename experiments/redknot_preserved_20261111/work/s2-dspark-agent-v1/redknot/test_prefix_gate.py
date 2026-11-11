"""CPU tests of evidence rejection, never synthetic GPU qualification."""
import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'GatedEngram'));sys.path.insert(0,str(ROOT))
from kv_cache_engram.core import digest,token_digest
from redknot.prefix_gate import audit_prefix_request,audit_full_request,compare_full_prefix
from prefix_decode_contract import validate_dflash_swa_coverage


def seal(body):
    return dict(body,sha256=digest({k:v for k,v in body.items() if k!='sha256'}))


def rows(ids,cached,rid='request'):
    return [seal(dict(schema='s2-native-scheduled-rows-v1',rank=r,request_id=rid,input_tokens=len(ids),
        initial_cached_tokens=cached if r==0 else 0,scheduled_prefill_rows=len(ids)-cached if r==0 else 0,
        first_prefill_q_start=cached if r==0 else None,
        prefill_ranges=[dict(q_start=cached,q_end=len(ids),rows=len(ids)-cached)] if r==0 else [],decode_steps=31 if r==0 else 0)) for r in range(8)]


def storage_contract():
    raw=dict(storage_ptr=1000,storage_bytes=1024,shape=[2,512],stride_bytes=[512,1])
    view=dict(layer_id=40,field='swa_cache',storage_ptr=1000,byte_offset=0,element_bytes=1,
              shape=[2,134,1,2],stride_bytes=[512,2,2,1])
    graph={k:i for i,k in enumerate(('gpu_blocks','gpu_states','mtp_info_pool.token_ids',
        'mtp_info_pool.logprobs','main_graph.buffer','dflash_graph.buffer'))}
    return seal(dict(schema='s2-prefix-dspark-storage-contract-v1',raw_states=raw,
        raw_blocks=dict(storage_ptr=2000,storage_bytes=2048,shape=[2,1024],stride_bytes=[1024,1]),
        all_native_cache_views=[view],dflash_swa_views=[view],
        dflash_swa_coverage=validate_dflash_swa_coverage(raw,[view],[40],window_tokens=128,backtrack_tokens=6),
        graph_storage=graph,structural_only=True,numerical_qualification=False,old_proposal_pool_copied=False))


def descriptor(ids,n=0):
    result=dict(request_id='request',state_id=1,block_table=[7,8,9],num_cached_tokens=n,
        previous_num_tokens=n,swa_valid_start=0,block_size=128,prompt_tokens=len(ids),output_tokens=0,
        schedule_status='WAITING',mtp_info_slot=0,mtp_has_draft_info=False,mtp_last_output_token=ids[-1],
        last_copy_state_token_idx=0)
    result.update({k:0 for k in ('mtp_accepted_count','mtp_draft_count','mtp_accept_rate_count',
        'mtp_prev_accept_rate_count','mtp_confidence_count','mtp_prev_confidence_count',
        'mtp_process_count','mtp_schedule_count','total_inflight_tokens','last_inflight_tokens')})
    return result


def receipt(ids,hit=True):
    n=128 if hit else 0
    fp=dict(kv='a'*64,state='b'*64)
    restore=dict(snapshot_id='snapshot1',prefix_tokens=n,prefix_ids_sha256=token_digest(ids[:n]),
        capture_request_id='source',capture_metadata_seqlen_k=n,capture_committed_num_cached_tokens=n,
        source_block_ids=[2],destination_block_ids=[7],source_state_id=1,destination_state_id=1,
        kv_bytes=1024,state_bytes=512,copy_complete=True,raw_state_complete=True,
        raw_kv_and_state_same_checkpoint=True,restored_num_cached_tokens=n,
        source_fingerprints=fp,restored_fingerprints=copy.deepcopy(fp),
        fresh_sequence_before=descriptor(ids),fresh_sequence_after=descriptor(ids,n),
        fresh_resume_admission=dict(prefix_tokens=n,native_suffix_tokens=len(ids)-n,
            source_prompt_tokens=len(ids),target_prompt_tokens=len(ids),compact_keep_tokens=140,compact_cutoff_compatible=True,
            proposal_policy='fresh_native_slot_then_native_suffix_and_decode',old_proposal_pool_copied=False,
            numerical_qualification=False),
        graph_storage_stability=dict(native_storage_unchanged=True,graph_replay_observed=False,numerical_qualification=False),
        decode_storage_contract_sha256=storage_contract()['sha256']) if hit else None
    return seal(dict(schema='s2-prefix-copy-request-v1',request_id='request',input_tokens=len(ids),
        input_ids_sha256=token_digest(ids),method='Prefix(copy)',cache_hit=hit,reused_prefix_tokens=n,
        matched_prefix_sha256=token_digest(ids[:n]) if hit else None,restore=restore,captures=[],
        bank_bytes_before=1536 if hit else 0,bank_bytes_after=1536 if hit else 0,bank_budget_bytes=4096,evictions=[],
        decode_storage_contract=storage_contract(),
        graph_storage_stability=dict(native_storage_unchanged=True,graph_replay_observed=False,numerical_qualification=False),
        proposal_policy='fresh_native_slot_then_native_suffix_and_decode',old_proposal_pool_copied=False,
        prefix_checkpoint_only=True))


def registry(ids):
    restore=receipt(ids)['restore']
    capture={k:restore[k] for k in ('snapshot_id','prefix_tokens','prefix_ids_sha256','capture_request_id',
        'source_block_ids','source_state_id','kv_bytes','state_bytes','copy_complete','raw_state_complete',
        'raw_kv_and_state_same_checkpoint')}
    capture.update(is_prompt=True,metadata_seqlen_q=128,metadata_seqlen_k=128,committed_num_cached_tokens=128,
        swa_valid_start=0,raw_fingerprints=restore['source_fingerprints'],
        source_prompt_tokens=len(ids),compact_keep_tokens=140,
        decode_storage_contract_sha256=storage_contract()['sha256'],prefix_only_intermediate_checkpoint=True,
        old_proposal_pool_copied=False,
        committed_sequence=dict(descriptor(ids,128),request_id='source',schedule_status='RUNNING',mtp_info_slot=1))
    return {'snapshot1':dict(receipt=capture,input_ids=ids)}


class PrefixGateTests(unittest.TestCase):
    def setUp(self):
        self.ids=list(range(300,600))

    def test_actual_restoration_and_full_rows_close_bounded_same_output(self):
        full=dict(request_id='full',input_ids_sha256=token_digest(self.ids),output_ids=list(range(40,72)),text='actual fixture',finish_reason='length')
        candidate=dict(full,request_id='request')
        result=compare_full_prefix(self.ids,full,candidate,receipt(self.ids),rows(self.ids,128),rows(self.ids,0,'full'),
                                   source_snapshots=registry(self.ids))
        self.assertTrue(result['passed'])
        self.assertFalse(result['complete_agent_quality_claim'])

    def test_text_hit_without_native_skipped_rows_is_rejected(self):
        with self.assertRaises(ValueError): audit_prefix_request(self.ids,receipt(self.ids),rows(self.ids,0))
        with self.assertRaises(ValueError): audit_full_request(self.ids,rows(self.ids,128))

    def test_wrong_state_boundary_or_payload_hash_is_rejected(self):
        report=receipt(self.ids);report['restore']['capture_committed_num_cached_tokens']=129
        with self.assertRaises(ValueError): audit_prefix_request(self.ids,seal(report),rows(self.ids,128))
        report=receipt(self.ids);report['restore']['restored_fingerprints']['state']='c'*64
        with self.assertRaises(ValueError): audit_prefix_request(self.ids,seal(report),rows(self.ids,128),
            require_payload_fingerprints=True,source_snapshots=registry(self.ids))

    def test_hit_must_bind_the_previously_captured_source_payload(self):
        source=registry(self.ids);source['snapshot1']['receipt']['raw_fingerprints']['state']='c'*64
        with self.assertRaises(ValueError):
            audit_prefix_request(self.ids,receipt(self.ids),rows(self.ids,128),require_payload_fingerprints=True,
                                 source_snapshots=source)
        with self.assertRaises(ValueError):
            audit_prefix_request(self.ids,receipt(self.ids),rows(self.ids,128),require_payload_fingerprints=True)

    def test_query_overlap_rank_omission_and_counter_only_claim_rejected(self):
        reports=rows(self.ids,128)
        reports[0]['prefill_ranges']=[dict(q_start=128,q_end=250,rows=122),dict(q_start=200,q_end=300,rows=100)]
        reports[0]=seal(reports[0])
        with self.assertRaises(ValueError): audit_prefix_request(self.ids,receipt(self.ids),reports)
        with self.assertRaises(ValueError): audit_prefix_request(self.ids,receipt(self.ids),rows(self.ids,128)[:7])
        report=receipt(self.ids);report['restore']=None
        with self.assertRaises(ValueError): audit_prefix_request(self.ids,seal(report),rows(self.ids,128))

    def test_miss_is_measured_with_all_full_rows(self):
        result=audit_prefix_request(self.ids,receipt(self.ids,False),rows(self.ids,0))
        self.assertEqual(result['reused_prefix_tokens'],0)
        report=receipt(self.ids,False);report['bank_bytes_after']=4097
        with self.assertRaises(ValueError): audit_prefix_request(self.ids,seal(report),rows(self.ids,0))

    def test_oversize_capture_is_explicit_skip_not_fake_snapshot(self):
        report=receipt(self.ids,False)
        report['captures']=[dict(prefix_tokens=128,skipped='snapshot_exceeds_budget',required_bytes=4097)]
        audit_prefix_request(self.ids,seal(report),rows(self.ids,0),require_payload_fingerprints=True)
        report['captures'][0]['required_bytes']=4096
        with self.assertRaises(ValueError): audit_prefix_request(self.ids,seal(report),rows(self.ids,0))

    def test_capture_requires_same_committed_prompt_boundary(self):
        report=receipt(self.ids,False)
        capture=registry(self.ids)['snapshot1']['receipt']
        capture['capture_request_id']='request';capture['committed_sequence']['request_id']='request'
        report['captures']=[capture]
        audit_prefix_request(self.ids,seal(report),rows(self.ids,0),require_payload_fingerprints=True)
        report['captures'][0]['is_prompt']=False
        with self.assertRaises(ValueError): audit_prefix_request(self.ids,seal(report),rows(self.ids,0))

    def test_different_generated_output_does_not_pass_prefix_gate(self):
        full=dict(request_id='full',input_ids_sha256=token_digest(self.ids),output_ids=[88,1],text='one',finish_reason='stop')
        candidate=dict(full,request_id='request',output_ids=[89,1],text='two')
        with self.assertRaises(ValueError):
            compare_full_prefix(self.ids,full,candidate,receipt(self.ids),rows(self.ids,128),rows(self.ids,0,'full'),
                                source_snapshots=registry(self.ids))

    def test_stale_proposal_or_missing_dflash_storage_is_rejected(self):
        report=receipt(self.ids)
        report['restore']['fresh_sequence_after']['mtp_has_draft_info']=True
        with self.assertRaises(ValueError):
            audit_prefix_request(self.ids,seal(report),rows(self.ids,128),source_snapshots=registry(self.ids))
        report=receipt(self.ids)
        report['decode_storage_contract']['dflash_swa_views']=[]
        report['decode_storage_contract']=seal(report['decode_storage_contract'])
        report['restore']['decode_storage_contract_sha256']=report['decode_storage_contract']['sha256']
        with self.assertRaises(ValueError):
            audit_prefix_request(self.ids,seal(report),rows(self.ids,128),source_snapshots=registry(self.ids))


if __name__=='__main__':unittest.main()
