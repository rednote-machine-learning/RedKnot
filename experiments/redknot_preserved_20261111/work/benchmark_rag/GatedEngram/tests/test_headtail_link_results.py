"""Independent interval/set oracle, evidence binding and mutation rejection."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from gpu_pilot import read_headtail_link_results as r
from kv_cache_engram.core import digest
from kv_cache_engram.headtail_link_plan import plan, materialize_indices, query_key_positions


def seal(body):
    return dict(body,sha256=digest(body))


def refresh(positions,spans,layer,start,end,previous=(),generation=7,qualified=True):
    boundary=r.boundary_positions(spans)
    rows=[i for i,p in enumerate(positions) if p in set(boundary)]
    current=[positions[i] for i in rows]
    available=set(previous)|set(current)
    visible={p for p in boundary if positions and p<=positions[-1]}
    checked=qualified and bool(rows)
    return dict(schema="headtail-link-dynamic-swa-v1",layer=layer,chunk_start=start,chunk_end=end,
        phase_generation=generation,source="current_layer_attention_input",input_representation="native_packed_current",
        refresh_enabled=True,current_boundary_local_rows=rows,current_boundary_positions=current,
        refreshed_local_rows=rows,refreshed_positions=current,refreshed_bank_rows=[boundary.index(p) for p in current],
        available_refreshed_positions=sorted(available),visible_refreshed_positions=sorted(available&visible),
        visible_offline_fallback_positions=sorted(visible-available),projection_identity_checked=checked,
        prepare_identity_checked=checked,checks=2 if checked else 0,max_absolute_error=0.0 if checked else None)


class IndependentMaskTests(unittest.TestCase):
    def test_random_spans_overlap_yoco_and_chunk_layout_match_set_oracle(self):
        rng=random.Random(734)
        for case in range(80):
            spans=[];cursor=rng.randrange(0,20)
            for i in range(rng.randrange(1,9)):
                end=cursor+rng.randrange(1,600)
                spans.append((cursor,end,str(i)*64));cursor=end+rng.randrange(0,12)
            total=cursor+rng.randrange(1,170)
            start=rng.randrange(total);end=min(total,start+rng.randrange(1,300))
            layer=rng.choice((3,19,20,24,39))
            positions=r.native_positions_reference(start,end,total,layer)
            produced=plan([dict(start=a,end=b,key=c) for a,b,c in spans],positions,start,end,total,layer=layer)
            expected=r.expected_plan(positions,start,end,total,layer,spans)
            self.assertEqual(produced,expected)
            for row,q in enumerate(positions):
                lo=max(0,q-127,total-140 if layer>=20 else 0)
                keys=set(range(lo,q+1))
                for a,b,_ in spans:
                    if b<=q:keys.update(range(max(a,b-128),b))
                actual=query_key_positions(produced,q)
                self.assertEqual(set(actual),keys)
                self.assertEqual(len(actual),len(keys))
                self.assertEqual(expected['union_lengths'][row],len(keys))
                self.assertTrue(all(k<=q for k in keys))
            queries=positions[::max(1,len(positions)//5)]
            self.assertTrue(r.audit_observed_indices(expected,queries,materialize_indices(produced,queries)))

    def test_full_geometry_has_all_eight_previous_tails_and_native_middle(self):
        spans=[(5+i*32768,5+(i+1)*32768,str(i)*64) for i in range(8)]
        total=spans[-1][1]+45
        e=r.expected_plan(list(range(262144,total)),262144,total,total,3,spans)
        self.assertEqual(e['index_width'],1152)
        self.assertEqual(e['completed_document_counts'][-1],8)
        local,extra=r.query_keys_reference(total-1,3,total,spans)
        self.assertEqual(len(local),128)
        self.assertEqual(len(local+extra),len(set(local+extra)))
        self.assertEqual(set(local+extra),set(local)|{p for a,b,_ in spans for p in range(b-128,b)})
        q=spans[1][0]+10
        local,_=r.query_keys_reference(q,3,total,spans)
        self.assertIn(spans[0][1]-1,local)

    def test_bad_survivor_geometry_never_revives_deleted_hidden(self):
        spans=[(0,1000,'a'*64)]
        for positions in (list(range(500,600)),[598,599],[599,599],[True]):
            with self.subTest(positions=positions[:2]),self.assertRaises(ValueError):
                r.expected_plan(positions,500,600,1200,20,spans)

    def test_raw_index_mutations_reject_future_duplicate_padding_and_global_slot(self):
        spans=[(0,400,'a'*64),(400,800,'b'*64)]
        e=r.expected_plan(list(range(400,620)),400,620,1000,3,spans)
        queries=[400,550,619]
        original=materialize_indices(e,queries)
        self.assertTrue(r.audit_observed_indices(e,queries,original))
        changes=[(0,0,e['global_index_offset']), (1,0,128+200), (1,1,original[1][0]),
                 (0,129,e['tail_extra_offset']+1), (0,383,123), (2,0,-1)]
        for row,col,value in changes:
            changed=copy.deepcopy(original);changed[row][col]=value
            with self.subTest(row=row,col=col),self.assertRaises(ValueError):
                r.audit_observed_indices(e,queries,changed)
        changed=copy.deepcopy(original);changed[0][0]=True
        with self.assertRaises(ValueError):r.audit_observed_indices(e,queries,changed)

    def test_cross_yoco_rectangle_additional_tails_do_not_change_local_mask(self):
        spans=[(0,400,'a'*64),(400,800,'b'*64)]
        e=r.expected_plan([499],400,500,1000,20,spans)
        self.assertEqual(e['native_window_lengths'],[0])
        self.assertEqual(e['extra_tail_lengths'],[128])
        self.assertEqual(e['current_positions'],[499])
        indices=materialize_indices(e,[499])
        self.assertEqual(indices[0][:128],[-1]*128)
        self.assertTrue(r.audit_observed_indices(e,[499],indices))


class RefreshTests(unittest.TestCase):
    spans=[(1,1001,'a'*64),(1001,2001,'b'*64)]

    def test_layer3_refreshes_head_and_tail_immediately(self):
        positions=list(range(0,120))
        e=refresh(positions,self.spans,3,0,120)
        seen=set()
        self.assertEqual(r.audit_refresh_event(e,positions,self.spans,seen,generation=7,qualification=True),119)
        self.assertEqual(seen,set(range(1,120)))
        for field,value in [('source','offline_layer3_seed'),('refresh_enabled',False),
                            ('source','previous_layer_kv'),('input_representation','bf16_current')]:
            bad=copy.deepcopy(e);bad[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,field):
                r.audit_refresh_event(bad,positions,self.spans,set(),generation=7,qualification=True)

    def test_cross_chunk_same_layer_accumulation_and_fallback(self):
        previous={1,2,3,998};positions=[999,1000,1001,1199]
        e=refresh(positions,self.spans,24,999,1200,previous)
        seen=set(previous)
        self.assertEqual(r.audit_refresh_event(e,positions,self.spans,seen,generation=7,qualification=True),3)
        self.assertTrue(previous<=seen)
        self.assertIn(997,e['visible_offline_fallback_positions'])
        self.assertNotIn(1001,e['visible_offline_fallback_positions'])
        for field,value in [('available_refreshed_positions',[999,1000,1001]),
                            ('refreshed_bank_rows',[0,1,2]),('visible_offline_fallback_positions',[]),
                            ('phase_generation',6),('visible_refreshed_positions',[999])]:
            bad=copy.deepcopy(e);bad[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,field):
                r.audit_refresh_event(bad,positions,self.spans,set(previous),generation=7,qualification=True)

    def test_zero_refresh_still_checks_native_input_flag_and_absolute_fallback(self):
        for layer in (20,24,39):
            e=refresh([499],self.spans,layer,400,500)
            self.assertEqual(r.audit_refresh_event(e,[499],self.spans,set(),generation=7,qualification=True),0)
            self.assertEqual(e['checks'],0)
            self.assertEqual(e['visible_offline_fallback_positions'],list(range(1,129)))
            for field,value in [('input_representation','bf16_current'),('checks',2),('max_absolute_error',0.0)]:
                bad=copy.deepcopy(e);bad[field]=value
                with self.subTest(layer=layer,field=field),self.assertRaises(ValueError):
                    r.audit_refresh_event(bad,[499],self.spans,set(),generation=7,qualification=True)

    def test_short_document_overlap_is_refreshed_once_and_repeated_update_rejected(self):
        spans=[(1,20,'a'*64)]
        positions=list(range(21));e=refresh(positions,spans,3,0,21)
        state=set()
        self.assertEqual(r.audit_refresh_event(e,positions,spans,state,generation=7,qualification=True),19)
        with self.assertRaisesRegex(ValueError,'twice'):
            r.audit_refresh_event(e,positions,spans,state,generation=7,qualification=True)

    def test_projection_proof_requires_exact_zero_on_both_paths(self):
        e=refresh([1,2,3],self.spans,3,0,4)
        for field,value in [('prepare_identity_checked',False),('projection_identity_checked',False),
                            ('checks',1),('max_absolute_error',1e-15)]:
            bad=copy.deepcopy(e);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                r.audit_refresh_event(bad,[1,2,3],self.spans,set(),generation=7,qualification=True)


class AccountingTests(unittest.TestCase):
    def test_interval_source_counts_equal_explicit_pair_enumeration(self):
        rng=random.Random(6)
        spans=[(10,410,'a'*64),(420,820,'b'*64),(820,1220,'c'*64)]
        total=1261
        for layer in (3,20,39):
            for dense in (False,True):
                for start,end in ((0,300),(300,900),(900,total)):
                    positions=r.native_positions_reference(start,end,total,layer)
                    e=r.expected_plan(positions,start,end,total,layer,spans)
                    refreshed={p for p in r.boundary_positions(spans) if p<=positions[-1] and rng.randrange(3)==0}
                    nondoc=sorted(p for p in range(end) if not any(a<=p<b for a,b,_ in spans))
                    expected=dict(consumed_local_pairs=0,consumed_extra_tail_pairs=0,consumed_cached_pairs=0,
                        consumed_refreshed_pairs=0,consumed_online_nondoc_pairs=0)
                    materialized=set()
                    queries=positions if dense else [positions[i] for i in e['selected_local_rows']]
                    for q in queries:
                        local,extra=r.query_keys_reference(q,layer,total,spans)
                        expected['consumed_local_pairs']+=len(local)
                        expected['consumed_extra_tail_pairs']+=len(extra)
                        for p in local+extra:
                            if p in refreshed:expected['consumed_refreshed_pairs']+=1
                            elif any(a<=p<b for a,b,_ in spans):
                                expected['consumed_cached_pairs']+=1;materialized.add(p)
                            else:expected['consumed_online_nondoc_pairs']+=1
                    expected['materialized_offline_rows']=len(materialized)
                    self.assertEqual(r.memory_accounting(e,spans,refreshed,nondoc,dense=dense),expected)

    def test_missing_online_nondocument_kv_is_rejected(self):
        spans=[(10,410,'a'*64)]
        e=r.expected_plan(list(range(420)),0,420,420,3,spans)
        with self.assertRaisesRegex(ValueError,'ungenerated nondocument'):
            r.memory_accounting(e,spans,set(),[],dense=False)


class PayloadTests(unittest.TestCase):
    def test_bf16_finite_byte_oracle_matches_all_65536_scalar_patterns(self):
        for word in range(65536):
            self.assertEqual(r._finite_bfloat16(word.to_bytes(2,'little')),word&0x7f80!=0x7f80)
        for nonfinite in (0x7f80,0x7fc0,0xff80,0xffff):
            self.assertFalse(r._finite_bfloat16(b'\0'*18+nonfinite.to_bytes(2,'little')+b'\0'*20))
        with self.assertRaises(ValueError):r._finite_bfloat16(b'\0')

    def test_payload_must_be_full_shape_finite_hash_bound_and_regular(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'full.bf16'
            # Sparse file on disk, but auditor reads and hashes all32MiB.
            with path.open('wb') as handle:handle.truncate(32768*512*2)
            h=hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(1024**2),b''):h.update(block)
            self.assertEqual(r.audit_bank_payload(path,h.hexdigest())['bytes'],32768*512*2)
            with self.assertRaisesRegex(ValueError,'bytes changed'):r.audit_bank_payload(path,'a'*64)
            link=Path(folder)/'link.bf16';link.symlink_to(path)
            with self.assertRaises(ValueError):r.audit_bank_payload(link,h.hexdigest())
            with path.open('r+b') as handle:handle.seek(200);handle.write(b'\x80\x7f')
            with self.assertRaisesRegex(ValueError,'nonfinite'):r.audit_bank_payload(path,h.hexdigest())
            with path.open('wb') as handle:handle.truncate(128*512*2)
            with self.assertRaisesRegex(ValueError,'tail-only'):r.audit_bank_payload(path,h.hexdigest())


class ControllerAuditTests(unittest.TestCase):
    def fixture(self):
        # Actual8x32K spans and33chunks. One consumer layer keeps this CPU fixture small;
        # the event coverage test checks that the declared layer set is exact.
        from tests.test_headtail_link_controller import controller
        c=controller();c.native_yoco_swa_share_hidden=False;c.phase_generation=7
        spans=[(5+i*32768,5+(i+1)*32768,str(i)*64) for i in range(8)]
        # Match the producer's bank identity: one (document digest, layer)
        # entry for every document/layer pair, even in this metadata fixture.
        c.bank={(key,layer):None for _,_,key in spans for layer in range(40)}
        total=262194
        row=dict(index=2,input_ids_sha256='a'*64,input_tokens=total)
        phase=dict(arm=r.ARM,mode='sensitive',name='case2.headtail128_link_quality',fraction=None,
            dynamic_document_swa=True,native_yoco_swa_share_hidden=False,verify_sparse=False,
            record=dict(request_id='headtail-link:2:quality:sample'))
        control=dict(c._report_fields(True),mode=phase['mode'],phase=phase['name'],
            input_ids_sha256=row['input_ids_sha256'],full_input_tokens=total,ffn_layers=[20],
            first_three_full=True,actual_full_row_native_mhc=True,yoco_layout_preserved=True,
            global_kv_native_online=True,decode_native=True,formal_admission_unchanged=True,
            not_model_flops_fraction=True,selected_tokens_forced_through_yoco=False,
            selection=[],ffn_layouts=[],swa_events=[],attention_events=[])
        adapter=dict(schema='kv-cache-engram-index-hit20-adapter-v1',decoder_forward_unchanged=True,
            attention_forward_unchanged=True,native_mhc_unchanged=True,native_yoco_layout_unchanged=True,
            all_selected_delegates_native=True,empty_selection_skips_ep_collective=False,
            cached_hidden_or_ffn_output=False,accuracy_guarantee=False,speedup_claim=False,events=[])
        previous=set();full=selected=0
        for start in range(0,total,8192):
            end=min(start+8192,total);positions=r.native_positions_reference(start,end,total,20)
            event=plan([dict(start=a,end=b,key=k) for a,b,k in spans],positions,start,end,total,layer=20)
            ids=event['selected_local_rows'];n=len(positions);k=len(ids);full+=n;selected+=k
            base=dict(chunk_start=start,chunk_end=end,layer=20)
            selection=dict(start=start,end=end,layer=20,current_positions=positions,selected_local_rows=ids,
                selected_positions=[positions[i] for i in ids],full_rows=n,selected_rows=k,plan_sha256=event['plan_sha256'])
            control['selection'].append(selection);control['ffn_layouts'].append(dict(selection))
            adapter['events'].append(dict(base,full_rows=n,selected_rows=k,actual_mlp_rows=k,reference_dense=False,
                plan_sha256=event['plan_sha256'],request_id=phase['record']['request_id'],phase=phase['name'],
                inactive_ffn_output='zero',empty_native_collective=k==0 and n>0))
            control['attention_events'].append(dict(base,full_rows=n,selected_rows=k,main_q_a_rows=n,index_q_rows=n,
                main_q_b_rows=k,joint_attention_rows=k,output_projection_rows=k,inactive_attention_zero_rows=n-k,
                global_compute_called=True,native_joint_attention=True,is_global_source=True,reference_dense=False,
                native_local_window_preserved=True,extra_completed_document_tail_links=True,
                plan_sha256=event['plan_sha256'],global_kv_rows=0))
            control['swa_events'].append(dict(base,current_positions=positions,online_local_rows=ids,online_rows=k,
                reused_rows=n-k,full_rows=n,compact=True,native_cache_store_rows=n,
                head_rows=len(event['document_head_local_rows']),tail_rows=len(event['document_tail_local_rows']),
                global_index_offset=event['global_index_offset']))
            event.update(query_positions=[positions[i] for i in ids],actual_joint_query_rows=k,reference_dense=False,
                native_cache_store_rows=n,global_source_is_native=True,
                tail_slots_absolute_positions_sha256=digest(event['tail_bank_positions']),
                gpu_native_window_checked=True,gpu_union_oracle_checked=False,
                native_valid_start_binding='native_indices_verified_zero_start')
            control['headtail_memory_events'].append(event)
            e=refresh(positions,spans,20,start,end,previous,qualified=False)
            previous.update(e['refreshed_positions']);control['headtail_swa_events'].append(e)
        adapter['counts']=dict(full_rows=full,selected_rows=selected,actual_mlp_rows=selected,inactive_ffn_zero_rows=full-selected)
        control['adapter']=adapter
        control['boundary_restore_identity_events']=[dict(layer=20,rows=2048,checked=False,checks=0,
            absolute_positions_sha256=digest(r.boundary_positions(spans)),max_absolute_error=None)]
        control['decode_ring_positions']=[dict(layer=20,positions=list(range(total-128,total)),
            physical_readback_exact=True,checks=128,physical_slots=list(range(128)))]
        return control,row,phase,spans

    def check(self,*args):
        with patch.object(r,'FFN_LAYERS',(20,)):return r.audit_controller(*args)

    def test_real_report_fields_and_plan_events_are_independently_counted(self):
        result=self.check(*self.fixture())
        self.assertGreater(result['consumed_cached_pairs'],0)
        self.assertGreater(result['dynamic_refreshed_rows'],0)
        self.assertGreater(result['unrefreshed_boundary_slots'],0)
        self.assertLess(result['dynamic_refreshed_boundary_fraction'],1)
        self.assertEqual(result['boundary_slot_capacity'],2048)
        self.assertEqual(result['dynamic_refresh_start_layer'],3)
        self.assertFalse(result['layer3_offline_seed'])

    def test_full_bank_scope_and_phase_generation_are_fail_closed(self):
        original=self.fixture()
        for field,value in [('offline_gpu_resident',False),('offline_bank_immutable',False),
                            ('middle_kv_immutable',False),('cross_phase_overlay_reuse',True),
                            ('phase_overlay_fresh',False),('phase_generation',6),
                            ('offline_bank_rows',320*128),('offline_bank_entry_count',296),
                            ('native_yoco_swa_share_hidden',True),('native_local_mask_preserved',False),
                            ('extra_links_cross_yoco_rectangle',False),('dynamic_refresh_start_layer',4)]:
            args=copy.deepcopy(original);args[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(*args)

    def test_native_window_refresh_sources_and_global_offset_mutations_rejected(self):
        original=self.fixture()
        for collection,field,value in [
            ('headtail_memory_events','gpu_native_window_checked',False),
            ('headtail_memory_events','gpu_union_oracle_checked',True),
            ('headtail_memory_events','native_yoco_rect_lo',0),
            ('headtail_memory_events','global_index_offset',129),
            ('headtail_memory_events','native_window_lows',[0]),
            ('headtail_memory_events','global_source_is_native',False),
            ('swa_events','native_cache_store_rows',0),
            ('attention_events','global_compute_called',False),
            ('headtail_swa_events','source','previous_layer_kv'),
            ('headtail_swa_events','input_representation','bf16_current')]:
            args=copy.deepcopy(original);args[0][collection][0][field]=value
            with self.subTest(collection=collection,field=field),self.assertRaises(ValueError):self.check(*args)

    def test_missing_duplicate_layer_chunk_or_ring_collision_rejected(self):
        original=self.fixture()
        for collection in ('selection','ffn_layouts','headtail_memory_events','headtail_swa_events',
                           'swa_events','attention_events'):
            for duplicate in (False,True):
                args=copy.deepcopy(original)
                if duplicate:args[0][collection].append(copy.deepcopy(args[0][collection][0]))
                else:args[0][collection].pop()
                with self.subTest(collection=collection,duplicate=duplicate),self.assertRaises(ValueError):self.check(*args)
        args=copy.deepcopy(original);args[0]['decode_ring_positions'][0]['physical_slots']=[0]*128
        with self.assertRaises(ValueError):self.check(*args)
        args=copy.deepcopy(original);args[0]['decode_ring_positions'][0]['positions'][-1]-=1
        with self.assertRaises(ValueError):self.check(*args)


class ClosureTests(unittest.TestCase):
    def timing(self):
        samples=[9.,3.,6.]
        value=dict(arms={r.ARM:dict(online_ttft_samples_s=samples,online_ttft_p50_s=6.,repetitions=3,fixed_output_tokens=32)},
            concurrency=1,active_dp_ranks=1,participating_ep_ranks=8,online_full_bank_gpu_resident=True,
            online_gather_scatter_and_fallback_included=True,model_load_included=False,offline_build_included=False,
            saturated_capacity_claim=False,fresh_paired_baseline=False,historical_results_used_for_admission=False)
        return value,samples

    def test_case_summary_is_reconstructed_from_three_candidate_samples(self):
        summary,samples=self.timing();r.audit_timing_summary(summary,samples)
        for field,value in [('fresh_paired_baseline',True),('historical_results_used_for_admission',True),
                            ('offline_build_included',True),('concurrency',8)]:
            bad=copy.deepcopy(summary);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):r.audit_timing_summary(bad,samples)
        for field,value in [('online_ttft_p50_s',3.),('online_ttft_samples_s',[9.,6.,3.]),('repetitions',1)]:
            bad=copy.deepcopy(summary);bad['arms'][r.ARM][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):r.audit_timing_summary(bad,samples)
        bad=copy.deepcopy(summary);bad['arms']['recomputed']=bad['arms'][r.ARM]
        with self.assertRaises(ValueError):r.audit_timing_summary(bad,samples)

    def test_final_closure_requires_four_raw_candidate_references(self):
        pre=dict(identity={'run':'new'},manifest_sha256='a'*64)
        qual=seal(dict(passed=True));smoke=seal(dict(passed=True,kind='new-smoke'))
        closures=[seal(dict(index=i)) for i in (2,3,4,5)]
        evidence={f'case{c["index"]}.candidate.json':dict(file_sha256=str(c['index'])*64,seal_sha256=c['sha256']) for c in closures}
        refs=[dict(index=c['index'],file=f'case{c["index"]}.candidate.json',**evidence[f'case{c["index"]}.candidate.json']) for c in closures]
        body=dict(schema=r.RUN_SCHEMA,status='complete',completed_case_indices=[2,3,4,5],identity=pre['identity'],
            manifest_sha256=pre['manifest_sha256'],qualification=qual,partial_smoke_sha256=smoke['sha256'],cases=refs)
        self.assertTrue(r.validate_closed_run(seal(body),pre,qual,smoke,closures,[],Path('.'),evidence))
        for field,value in [('schema','kv-cache-engram-tail-link-pilot-run-v1'),('status','running'),
                            ('completed_case_indices',[2,3,4]),('partial_smoke_sha256','b'*64),('cases',refs[:-1])]:
            bad=copy.deepcopy(body);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                r.validate_closed_run(seal(bad),pre,qual,smoke,closures,[],Path('.'),evidence)
        bad=copy.deepcopy(body);bad['cases'][0]['file']='case2.paired.json'
        with self.assertRaises(ValueError):r.validate_closed_run(seal(bad),pre,qual,smoke,closures,[],Path('.'),evidence)
        with self.assertRaises(ValueError):r.validate_closed_run(seal(body),pre,qual,smoke,closures,[{'error':'x'}],Path('.'),evidence)

    def test_common_strict_json_and_reference_boundaries_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.json'
            path.write_text('{"status":"complete","status":"failed"}')
            with self.assertRaises(ValueError):r.read(path,{})
            path.write_text(json.dumps(seal(dict(passed=True))))
            link=Path(folder)/'link.json';link.symlink_to(path)
            with self.assertRaises(ValueError):r.read(link,{})


if __name__=='__main__':unittest.main()
