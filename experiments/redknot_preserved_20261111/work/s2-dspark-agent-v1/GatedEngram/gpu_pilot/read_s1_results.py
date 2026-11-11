"""Independent CPU-only S1 audit: absolute positions, real counts and raw closure."""
from __future__ import annotations
import argparse
from bisect import bisect_left
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kv_cache_engram.core import digest,require,validate_digest
from gpu_pilot import read_tail_link_results as common
from gpu_pilot import read_natural_boundary_results as natural
from gpu_pilot.s1_protocol import s1_protocol,phase_policy,timing_schedule,ARMS,POLICY_ARMS,EVALUATION_INDICES

SCHEMA="loong-s1-independent-readout-v1"
RUN_SCHEMA="kv-cache-engram-loong-s1-pilot-run-v1"
PHASE_SCHEMA="kv-cache-engram-loong-s1-phase-v1"
CASE_INDICES=EVALUATION_INDICES
DEEP_LAYERS=tuple(range(3,40))
INPUT_COMPANIONS=("source-records.json","selection-plan.json","tokenization-evidence.json")
read,sealed,number=common.read,common.sealed,common.number
score,verify_reference=common.score,common.verify_reference
empty_ep_binding=common.empty_ep_binding
_fingerprint=common._fingerprint
_fields=natural._fields
audit_quality_completion=natural.audit_quality_completion
from gpu_pilot.token_divergence import decoder_for_identity,compare_quality_records,audit_decoded_record
from gpu_pilot.loong_metrics import score_answer,metric_policy,gold_schema,answer_agreement,summarize_scores,PRIMARY_METRIC


DATASETS=('loong',)
LENGTH_BANDS=('128k','64k')
TARGET_INPUT_TOKENS={'128k':131072,'64k':65536}
BAND_LIMITS={'128k':(112*1024,144*1024),'64k':(56*1024,72*1024)}
DATASET_ORDER=['loong']*20
CORE_SHA256={
    'kv_cache_engram/native_reuse.py':'7d7b0f2b301d46a862190142089706724eb6794bbd1dd26f15e224b31cc1ea43',
    'kv_cache_engram/s1_runtime.py':'7fcdde36fffe69da9d902bd449b7094bd91798c8904cca655599123e12d9ca0b',
    'kv_cache_engram/s1_attention.py':'3c47ef7119ee16a2764e1203d10b2af86eb00935a36af5566eb00db225cabebb',
    'kv_cache_engram/s1_cache.py':'8eab07b22e97aa7688c5d5235da0d3a0df7ef71358ad03eecb81f87074087b4f',
    'kv_cache_engram/s1_plan.py':'b7a576332933ee11bf39ffe080856d1a5c9b889b468161cc0957099d478f9fc3',
    'kv_cache_engram/tail128_global_cache_plan.py':'29225efcc150e75b5df276abb9d486d44ad6a4360896748e59436390a50887b7',
}
CORE_POLICY_SHA256='82273f5646f23e7b59fb4830ec1c3d7ac155e4326a08fdb57763b9379e4438f9'
NATIVE_SNAPSHOT_SHA256='d89dad6c23f9df603b7ba27ca3d07717c315bc9d4dcbf7471346552b300cb94f'

def verify_core_bindings(identity,protocol):
    expected=dict(core_source_sha256=CORE_SHA256,core_policy_sha256=CORE_POLICY_SHA256,
        native_snapshot_sources_sha256=NATIVE_SNAPSHOT_SHA256)
    _fields(identity,expected,'actual previous S1 algorithm/native identity')
    _fields(protocol,expected,'declared previous S1 algorithm/native identity')
    return expected



def length_identity(row):
    index=row.get('index')
    require(type(index) is int and index in range(2,22),'foreign long-context case index')
    band=LENGTH_BANDS[(index-2)%2];ordinal=(index-2)//2+1;total=row.get('input_tokens')
    require(row.get('dataset')=='loong' and row.get('length_band')==band
        and type(row.get('length_band_ordinal')) is int and row['length_band_ordinal']==ordinal
        and type(row.get('target_input_tokens')) is int and row['target_input_tokens']==TARGET_INPUT_TOKENS[band]
        and type(total) is int and BAND_LIMITS[band][0]<=total<=BAND_LIMITS[band][1],
        'actual tokenizer length/band/target/order differs')
    return band


def audit_length_inputs(rows):
    require(type(rows) is list and [r.get('index') for r in rows]==list(range(2,22)),'twenty actual long-context inputs required')
    seen=set()
    for row in rows:
        length_identity(row)
        require(type(row.get('source_id')) is str and row['source_id'] not in seen,'long-context original QA repeated')
        seen.add(row['source_id'])
        audit_source_metadata(row)
        for lo,hi,key in documents(row):
            require(digest(row['input_ids'][lo:hi])==key,'natural block actual input IDs differ')
    return True


SCOPE_FIELDS=('length_band','length_band_ordinal','target_input_tokens','source_document_count',
    'original_document_count','source_domain','source_language','source_level','raw_answer_sha256')


def audit_source_metadata(row):
    gold=row.get('raw_answer');gold_schema(gold)
    expected=gold if type(gold) is str else json.dumps(gold,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    require(row.get('raw_answer_sha256')==digest(gold) and row.get('references')==[expected],
        'original raw gold/reference transport differs')
    count=row.get('source_document_count')
    require(type(count) is int and 5<=count<=10 and row.get('original_document_count')==count,
        'original Loong document count differs from five to ten')
    require(row.get('source_domain') in ('financial','legal','paper') and row.get('source_language') in ('en','zh')
        and type(row.get('source_level')) is int and 1<=row['source_level']<=4,'Loong domain/language/level differs')
    originals=row.get('original_documents');docs=row.get('documents')
    require(type(originals) is list and len(originals)==count and type(docs) is list and len(docs)==10,
        'original source and cache-block identities missing')
    previous=0
    for ordinal,original in enumerate(originals):
        require(original.get('source_document_ordinal')==ordinal
            and type(original.get('source_document_name')) is str
            and original.get('context_char_start')==previous
            and type(original.get('context_char_end')) is int and original['context_char_end']>previous,
            'original source document order/boundaries differ')
        previous=original['context_char_end']
        blocks=[d for d in docs if d.get('source_document_ordinal')==ordinal]
        require(bool(blocks),'original document omitted from cache blocks')
        local=0
        for block in blocks:
            require(block.get('source_document_name')==original['source_document_name']
                and block.get('source_doc_char_start')==local
                and type(block.get('source_doc_char_end')) is int and block['source_doc_char_end']>local
                and block.get('context_char_start')==original['context_char_start']+local
                and block.get('context_char_end')==original['context_char_start']+block['source_doc_char_end'],
                'cache block crosses or mislabels original source document')
            local=block['source_doc_char_end']
        require(local==original['context_char_end']-original['context_char_start'],'original source text not fully partitioned')
    require([d['source_document_ordinal'] for d in docs]==sorted(d['source_document_ordinal'] for d in docs)
        and all(type(d['source_document_ordinal']) is int and 0<=d['source_document_ordinal']<count for d in docs),
        'cache blocks reorder or invent original documents')
    return True


def existing_timing_breakdown(phase):
    """Read already-recorded clocks; no profiler or synchronization is introduced."""
    record=phase['record'];cache=number(phase['controller'].get('online_cache_prepare_s'),'cache base preparation')
    before=number(record.get('online_prepare_to_submit_s'),'pre-submit preparation')
    engine=number(record.get('ttft_s'),'native request TTFT',True)
    total=number(record.get('online_ttft_s'),'whole online TTFT',True)
    require(cache<=before+1e-6 and before<=total+1e-6
        and math.isclose(before+engine,total,rel_tol=1e-8,abs_tol=1e-6),'existing timing breakdown inconsistent')
    return dict(online_cache_prepare_s=cache,online_prepare_to_submit_s=before,engine_ttft_s=engine)


def documents(row):
    if row.get('s1_empty_chunk_qualification') is True:
        total=row.get('input_tokens');docs=row.get('documents')
        require(row.get('index')==0 and total==16384 and type(docs) is list and len(docs)==1,
            'empty-chunk geometry must be original calibration16K as one document')
        doc=docs[0];key=doc.get('token_ids_sha256');validate_digest(key,'empty qualification document')
        require(doc.get('input_start')==0 and doc.get('input_end')==total
            and key==row.get('input_ids_sha256'),'empty qualification span/input differs')
        return [(0,total,key)]
    total=row.get('input_tokens');docs=row.get('documents')
    require(type(total) is int and total>0 and type(docs) is list and len(docs)==10,
        'ten original natural cache block bindings required')
    previous=0;seen=set();spans=[]
    for doc in docs:
        lo,hi,key=doc.get('input_start'),doc.get('input_end'),doc.get('token_ids_sha256')
        validate_digest(key,'MuSiQue block token SHA')
        require(type(lo) is int and type(hi) is int and previous<=lo<hi<=total
            and hi-lo>=256 and key not in seen,'MuSiQue block order/extent/size/distinctness differs')
        if 'tokens' in doc:require(type(doc['tokens']) is int and doc['tokens']==hi-lo,'MuSiQue block length differs')
        spans.append((lo,hi,key));seen.add(key);previous=hi
    return spans


def case_geometry(row):
    lengths=[end-start for start,end,_ in documents(row)];total=row['input_tokens'];count=len(lengths)
    return dict(index=row['index'],input_tokens=total,document_token_lengths=lengths,
        document_tokens=sum(lengths),nondocument_tokens=total-sum(lengths),
        native_prefill_chunks=(total+8191)//8192,layer_chunk_checks=40*((total+8191)//8192),
        offline_bank_entry_count=40*count,offline_swa_halo_rows=37*count*128,
        offline_global_raw_rows=3*sum(lengths),offline_bank_rows=37*count*128+3*sum(lengths),
        offline_bank_bytes=37*count*128*512*2+9216*sum(lengths))


def execution_scope(identity):
    return dict(model_sha256=identity['checkpoint_index_sha256'],
        tokenizer_sha256=identity['tokenizer_json_sha256'],execution_sha256=identity['execution_configuration_sha256'])


def bank_scope_sha256(spans,identity):
    return digest(dict(scope=execution_scope(identity),documents=[dict(document_sha256=key,document_tokens=end-start)
        for start,end,key in spans]))


def comparison_scope(control,phase,*,spans,expected_scope=None):
    require(type(control) is dict and control.get('schema')=='s1-controller-v1','foreign S1 controller')
    _fields(control,dict(arm=phase['arm'],cache_storage_policy='same_s1_offline_gpu_bank_resident_both_arms',
        gpu_bank_constant_across_arms=True,offline_gpu_resident=True,
        document_token_lengths=[end-start for start,end,_ in spans]),'S1 shared-bank scope')
    lengths=[end-start for start,end,_ in spans]
    expected=dict(offline_bank_entry_count=40*len(spans),offline_swa_halo_rows=37*128*len(spans),
        offline_global_raw_rows=3*sum(lengths),offline_bank_rows=37*128*len(spans)+3*sum(lengths),
        offline_bank_bytes=37*128*len(spans)*512*2+9216*sum(lengths))
    _fields(control,expected,'actual S1 halo/global storage geometry')
    require(type(control.get('bank_generation')) is int and control['bank_generation']>0,
        'S1 bank generation missing')
    validate_digest(control.get('offline_bank_scope_sha256'),'S1 bank source scope')
    if expected_scope is not None:
        require(control['offline_bank_scope_sha256']==expected_scope,'S1 bank checkpoint/source/document scope differs')
    return dict(bank_generation=control['bank_generation'],offline_bank_scope_sha256=control['offline_bank_scope_sha256'],**expected)


def binding(phase,preflight):
    _fields(phase,dict(execution_configuration_sha256=preflight['identity']['execution_configuration_sha256'],
        manifest_sha256=preflight['manifest_sha256'],s1_protocol_sha256=preflight['protocol']['sha256'],
        native_yoco_swa_share_hidden=False,natural_boundary_inputs=True),'S1 frozen phase binding')
    require(phase.get('arm') in (*ARMS,'qualification','offline'),'foreign S1 phase arm')
    _fields(phase,phase_policy(phase['arm']),'S1 phase selection declaration')


def audit_phase(phase,row,preflight,*,arm,timing,quality,fixed_tokens=32):
    sealed(phase);binding(phase,preflight)
    _fields(phase,dict(schema=PHASE_SCHEMA,index=row['index'],dataset=row['dataset'],arm=arm,
        mode='native' if arm=='recomputed' else 'sensitive',fraction=None,
        s1_policy=arm=='s1',timing_valid=timing,quality_eligible=quality,
        input_ids_sha256=row['input_ids_sha256'],diagnostic=False,
        online_full_bank_gpu_resident=True,online_clock_started_before_reset=True,
        phase_local_memory_rebuilt=True,offline_build_included=False,native_first3_source2_preserved=True),
        'S1 fresh phase eligibility')
    _fields(phase,{k:row[k] for k in SCOPE_FIELDS},'long-context length phase identity')
    rec=phase['record']
    _fields(rec,dict(success=True,rank=0,input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],
        phase=phase['name'],sample_id=row['sample_id'],dataset=row['dataset'],
        request_id=f"s1:{row['index']}:{phase['name']}:{row['sample_id']}"),'S1 complete rank0 request')
    ids=rec.get('output_ids')
    require(type(ids) is list and bool(ids) and all(type(i) is int and i>=0 for i in ids)
        and rec.get('output_ids_sha256')==digest(ids),'S1 raw output token binding differs')
    if quality:quality_record(rec,phase,row)
    else:
        require(fixed_tokens in (1,32) and (not timing or fixed_tokens==32)
            and len(ids)==fixed_tokens and rec.get('natural_EOS') is False
            and rec.get('finish_reason')=='length' and rec.get('truncated') is False,
            'S1 fixed timing/warmup/qualification output differs')
    require(number(rec.get('online_prepare_to_submit_s'),'S1 prepare')<=number(rec.get('online_ttft_s'),'S1 TTFT',True)
        <=number(rec.get('online_elapsed_s'),'S1 elapsed',True)<=number(phase.get('online_global_window_s'),'S1 global window',True),
        'S1 timing order differs')
    comparison_scope(phase['controller'],phase,spans=documents(row),
        expected_scope=bank_scope_sha256(documents(row),preflight['identity']))
    return audit_controller(phase['controller'],row,phase,documents(row))


def position_oracle(spans,total):
    """Independent set oracle; no producer planner or native runtime imports."""
    require(type(total) is int and total>0 and type(spans) in (list,tuple) and bool(spans),"S1 document geometry required")
    cursor=0;tail=[];halo=[];nondoc=[]
    for start,end,key in spans:
        require(type(start) is int and type(end) is int and cursor<=start<end<=total,"invalid ordered S1 document spans")
        validate_digest(key,"S1 document identity")
        nondoc.extend(range(cursor,start))
        tail.extend(range(max(start,end-128),end))
        halo.extend(range(max(start,end-256),max(start,end-128)))
        cursor=end
    nondoc.extend(range(cursor,total))
    selected=sorted(set(tail)|set(nondoc))
    selected_set,halo_set=set(selected),set(halo)
    consumed_halo=set();local_pairs=offline_pairs=0
    for query in selected:
        for key in range(max(0,query-127),query+1):
            require(key in selected_set or key in halo_set,
                f"S1 selected query {query} has uncovered original local key {key}")
            local_pairs+=1
            if key not in selected_set:
                consumed_halo.add(key);offline_pairs+=1
    return dict(full_input_tokens=total,selected_positions=selected,tail_positions=tail,
        mandatory_nondoc_positions=nondoc,halo_positions=halo,
        consumed_offline_halo_positions=sorted(consumed_halo),local_query_key_pairs=local_pairs,
        offline_halo_query_key_pairs=offline_pairs,selected_tokens=len(selected),
        selected_fraction=len(selected)/total,unselected_tokens=total-len(selected),
        native_yoco140_intersection_applied=False,extra_cross_document_tail_links=False)


def chunk_oracle(spans,total):
    whole=position_oracle(spans,total)
    selected=whole['selected_positions']
    return [dict(chunk_start=start,chunk_end=min(start+8192,total),
        selected_positions=selected[bisect_left(selected,start):bisect_left(selected,min(start+8192,total))],
        original_rows=min(8192,total-start)) for start in range(0,total,8192)]


def global_groups_oracle(total,selected,layer):
    require(layer in (8,14,20),"only actual deep global sources may be updated")
    require(type(selected) is list and selected==sorted(set(selected))
        and all(type(p) is int and 0<=p<total for p in selected),"invalid selected absolute positions")
    ratio=1 if layer==20 else 2;chosen=set(selected)
    groups=[]
    for key in range(total//ratio):
        members=list(range(key*ratio,(key+1)*ratio))
        online=[p for p in members if p in chosen]
        groups.append(dict(compressed_key=key,member_positions=members,online_positions=online,
            offline_positions=[p for p in members if p not in chosen],affected=bool(online)))
    return dict(source_layer=layer,compression_ratio=ratio,groups=groups,
        affected_groups=sum(g['affected'] for g in groups),
        pending_positions=list(range((total//ratio)*ratio,total)))


def local_memory_oracle(spans,total,chunk,previous):
    positions=chunk['selected_positions'];start,end=chunk['chunk_start'],chunk['chunk_end']
    halo=[p for lo,hi,_ in spans for p in range(max(lo,hi-256),max(lo,hi-128))]
    h={p:i for i,p in enumerate(halo)};current={p:i for i,p in enumerate(positions)};prev={p:i for i,p in enumerate(previous)}
    memory=sorted({p for q in positions for p in range(max(0,q-127),q+1)})
    kinds=[];rows=[]
    for p in memory:
        if p in current:kind,source='current',current[p]
        elif p in prev:kind,source='previous',prev[p]
        else:
            require(p in h,f'S1 required local key {p} is absent from halo/current/history')
            kind,source='halo',h[p]
        kinds.append(kind);rows.append(source)
    body=dict(chunk_start=start,chunk_end=end,query_positions=positions,memory_positions=memory,
        source_kinds=kinds,source_rows=rows,native_window_size=128,extra_nonlocal_swa_links=False,
        local_lengths=[min(128,p+1) for p in positions])
    return dict(body,sha256=digest(body))


def _events(items,keys,label):
    require(type(items) is list and len(items)==len(keys),label+' event count differs')
    out={}
    for event in items:
        require(type(event) is dict,label+' event must be an object')
        key=(event.get('chunk_start'),event.get('chunk_end'),event.get('layer'))
        require(key in keys and key not in out,label+' foreign/duplicate layer-chunk')
        out[key]=event
    return out


def audit_controller(control,row,phase,spans):
    _fields(control,dict(schema='s1-controller-v1',mode=phase['mode'],arm=phase['arm'],phase=phase['name'],
        expected_document_count=10,
        input_ids_sha256=row['input_ids_sha256'],full_input_tokens=row['input_tokens'],first_three_full=True,
        native_yoco_swa_share_hidden=False,decode_native=True,model_boundary_restored=True,
        extra_cross_document_tail_links=False,native_local_window=128,source2_native_cache_preserved=True,
        source2_formal_byte_hash_claim=False,capture_sampler_events=[]),'S1 decoder scope')
    cache,attention=control.get('cache_report',{}),control.get('attention_report',{})
    _fields(cache,dict(schema='s1-cache-report-v1',full_swa_bank_persisted=False,hidden_or_ffn_bank_persisted=False,
        phase_preparation_in_ttft_required=True,source2_native_prepass_unchanged=True),'S1 raw-cache contract')
    _fields(attention,dict(schema='s1-attention-report-v1',full_swa_bank_used=False,only_tail_and_nondoc_queries=True,
        native_global_attention_kernel=True,selected_queries_survive_layers3_through39=True),'S1 attention contract')
    require(type(control.get('phase_generation')) is int and control['phase_generation']>0
        and cache.get('phase_generation')==control['phase_generation'],'S1 phase generation missing')
    number(control.get('online_cache_prepare_s'),'S1 cache setup')
    require(control['online_cache_prepare_s']==cache.get('phase_preparation_s'),'S1 setup time report differs')
    total=row['input_tokens'];chunks=chunk_oracle(spans,total)
    if phase['arm']=='recomputed':
        require(phase['mode']=='native' and control.get('yoco_layout_preserved') is True
            and control.get('selected_survive_layer39') is False
            and control.get('decoder_events')==[] and control.get('source2_identity_events')==[]
            and cache.get('global_events')==[] and attention.get('attention_events')==[]
            and attention.get('decode_ring_events')==[],'native full baseline executed S1 policy')
        count=sum(len(natural.head.native_positions_reference(c['chunk_start'],c['chunk_end'],total,l)) for c in chunks for l in DEEP_LAYERS)
        return dict(ffn_full_rows=count,ffn_selected_rows=count,ffn_bypassed_rows=0,
            attention_full_rows=count,attention_selected_rows=count,attention_bypassed_rows=0,
            residual_bypassed_rows=0,mhc_bypassed_rows=0,dynamic_refreshed_rows=0,
            consumed_cached_pairs=0,actual_partial_chunks=0,actual_empty_chunks=0,
            actual_rows_instrumented=False,counts_basis='native surviving-layout oracle; baseline operators uninstrumented',
            first_three_layers_excluded=True,not_model_flops_fraction=True,phase_generation=control['phase_generation'])
    require(phase['arm']=='s1' and phase['mode'] in ('reference_sparse','sensitive'),'unknown S1 controller policy')
    reference=phase['mode']=='reference_sparse';qualified=reference or phase.get('verify_sparse') is True
    _fields(control,dict(selected_survive_layer39=True,yoco_layout_preserved=False,
        unselected_deep_residual_mhc=reference,dense_reference_only=reference),'S1 physical compaction scope')
    allkeys={(c['chunk_start'],c['chunk_end'],l) for c in chunks for l in range(40)}
    events=_events(control.get('decoder_events'),allkeys,'S1 decoder')
    deepkeys={k for k in allkeys if k[2]>=3}
    attn=_events(attention.get('attention_events'),deepkeys,'S1 attention')
    globalkeys={k for k in allkeys if k[2] in (8,14,20)}
    globalevents=_events(cache.get('global_events'),globalkeys,'S1 global cache')
    engram=_events(control.get('engram_events'),{k for k in allkeys if k[2]==14},'S1 native Engram')
    selected_all=position_oracle(spans,total)['selected_positions'];selected_set=set(selected_all)
    previous=[];offline_pairs=0;materialized=0;projectionchecks=0;actual_ffn=0
    for c in chunks:
        start,end=c['chunk_start'],c['chunk_end'];positions=c['selected_positions'];n=len(positions)
        local=local_memory_oracle(spans,total,c,previous)
        _fields(engram[start,end,14],dict(layer=14,chunk_start=start,chunk_end=end,
            original_hash_rows=end-start,selected_positions=positions,
            actual_engram_rows=end-start if reference else n,hook_consumed_once=True,
            original_ngram_hashes_preserved=True,empty_hook_consumed_without_fake_rows=not reference and n==0),
            'S1 selected Engram rows and distributed hook consumption')
        halo_positions={p for p,k in zip(local['memory_positions'],local['source_kinds']) if k=='halo'}
        offline_pairs+=sum(sum(p in halo_positions for p in range(max(0,q-127),q+1)) for q in positions)*37
        materialized+=len(halo_positions)*37
        for layer in range(40):
            current=list(range(start,end)) if layer<3 else positions
            actual=end-start if layer<3 or reference else n
            e=events[start,end,layer]
            _fields(e,dict(layer=layer,chunk_start=start,chunk_end=end,full_rows=end-start,
                selected_positions=current,selected_rows=len(current),actual_attention_rows=len(current),
                actual_ffn_rows=actual,actual_residual_rows=actual,actual_mhc_rows=actual,
                dense_reference_only=reference and layer>=3,original_absolute_positions=True,
                empty_native_ep_call=layer>=3 and actual==0),'S1 actual per-operator rows')
            if layer<3:continue
            actual_ffn+=actual
            e=attn[start,end,layer]
            _fields(e,dict(layer=layer,chunk_start=start,chunk_end=end,selected_positions=positions,
                selected_rows=n,full_rows=end-start,memory_positions=local['memory_positions'],
                offline_halo_rows=len(halo_positions),actual_attention_rows=n,
                gpu_local_window_oracle_checked=True,actual_native_swa_indices_checked=True,global_causal_checked=True,
                no_unselected_hidden_fallback=True,no_extra_swa_links=True,plan_sha256=local['sha256']),
                'S1 exact native local-window memory')
            checks=e.get('projection_identity_checks')
            require(type(checks) is int and checks>=0 and (not reference or checks>=2)
                and e.get('max_absolute_error')==(0.0 if checks else None),'S1 native projection qualification absent')
            projectionchecks+=checks
            if layer in (8,14,20):
                ratio=1 if layer==20 else 2;begin_id,end_id=start//ratio,end//ratio
                affected=sorted({p//ratio for p in selected_all if begin_id*ratio<=p<end_id*ratio})
                _fields(globalevents[start,end,layer],dict(layer=layer,chunk_start=start,chunk_end=end,
                    phase_generation=control['phase_generation'],compression_ratio=ratio,
                    online_projection_rows=n,online_projection_positions=positions,full_raw_rows=end-start,
                    complete_compressed_ids=list(range(begin_id,end_id)),affected_compressed_ids=affected,
                    offline_base_rows=end_id-begin_id-len(affected),checks=2 if qualified else 0,
                    max_absolute_error=0.0 if qualified else None,source2_untouched=True,
                    main_and_index_k_native_store=True,global_base_representation='compressed_prenorm_bf16',
                    packed_main_and_index_store_checked=qualified,native_compressor_carry_updated=ratio==2),
                    'S1 selected/offline compression and native main/index storage')
        previous=[p for p in previous+positions if p>=end-127]
    rings=attention.get('decode_ring_events')
    wanted=list(range(max(0,total-128),total))
    require(type(rings) is list and len(rings)==37 and {e.get('layer') for e in rings}==set(DEEP_LAYERS),
        'S1 final physical decode ring coverage')
    for e in rings:_fields(e,dict(positions=wanted,physical_store=True,physical_readback_exact=qualified,
        checks=len(wanted) if qualified else 0),'S1 physical decode ring identity')
    source_events=control.get('source2_identity_events')
    if qualified:
        require(type(source_events) is list and len(source_events)==len(chunks),'S1 source2 cache fingerprint coverage')
        for e,c in zip(source_events,chunks):
            _fields(e,dict(chunk_start=c['chunk_start'],chunk_end=c['chunk_end'],exact=True),'S1 native source2 chunk')
            require(e.get('before')==e.get('after'),'S1 source2 main/index data changed')
            _fields(e['before'],dict(schema='s1-source2-cache-fingerprint-v1',rows=c['chunk_end']//2),'S1 valid source2 rows')
            for k in ('main_sha256','index_k_sha256'):validate_digest(e['before'].get(k),'source2 '+k)
    else:require(source_events==[],'formal S1 unexpectedly synchronized qualification byte fingerprints')
    full=37*total;selected=37*len(selected_all)
    return dict(ffn_full_rows=full,ffn_selected_rows=selected,ffn_actual_computed_rows=actual_ffn,
        ffn_bypassed_rows=full-actual_ffn,attention_full_rows=full,attention_selected_rows=selected,
        attention_bypassed_rows=full-selected,residual_bypassed_rows=full-actual_ffn,mhc_bypassed_rows=full-actual_ffn,
        dynamic_refreshed_rows=selected,consumed_cached_pairs=offline_pairs,
        materialized_offline_halo_rows=materialized,projection_identity_checks=projectionchecks,
        actual_partial_chunks=37*sum(len(c['selected_positions'])<c['original_rows'] for c in chunks),
        actual_empty_chunks=37*sum(not c['selected_positions'] for c in chunks),
        actual_rows_instrumented=True,first_three_layers_excluded=True,not_model_flops_fraction=True,
        counts_basis='original full chunk domain; S1 keeps all selected rows through layer39',
        phase_generation=control['phase_generation'])


def quality_record(record,phase,row):
    audit_quality_completion(record,phase,row,s1_protocol()['quality'])
    expected=score(record.get('text'),row['references'])
    require(all(type(record.get(k)) in (int,float) and math.isfinite(record[k])
        and abs(record[k]-value)<=1e-12 for k,value in expected.items()),"S1 gold score is not reproducible")
    require(type(record.get('output_ids')) is list and record.get('output_ids_sha256')==digest(record['output_ids']),
        'quality output IDs/hash must survive compact audit')
    result={k:record[k] for k in ('text','answer_em','answer_token_f1','truncated','finish_reason','output_ids','output_ids_sha256','natural_EOS')}
    result['loong_score']=score_answer(record['text'],row['raw_answer'])
    result['metric_policy']=metric_policy(row['dataset'])
    result['generated_tokens']=len(record['output_ids'])
    result.update(audit_quality_completion(record,phase,row,s1_protocol()['quality']))
    return result


def expected_timing_summary(samples):
    require(set(samples)==set(ARMS) and all(type(v) is list and len(v)==3 for v in samples.values()),
        "exactly three fresh timing samples required for each S1 arm")
    arms={}
    for arm,values in samples.items():
        for value in values:number(value,"S1 TTFT",True)
        arms[arm]=dict(online_ttft_samples_s=values,online_ttft_p50_s=statistics.median(values),
            repetitions=3,fixed_output_tokens=32)
    return dict(arms=arms,comparisons={'s1':dict(ttft_speedup=arms['recomputed']['online_ttft_p50_s']/arms['s1']['online_ttft_p50_s'],
        paired_repeat_speedups=[samples['recomputed'][i]/samples['s1'][i] for i in range(3)])},
        concurrency=1,active_dp_ranks=1,participating_ep_ranks=8,
        online_full_bank_gpu_resident=True,online_gather_scatter_and_fallback_included=True,
        model_load_included=False,offline_build_included=False,saturated_capacity_claim=False,
        fresh_paired_baseline=True,historical_results_used_for_admission=False)


def native_identity(reference,candidate):
    """Identity mode retains native query layout and uses online native KV."""
    expected={(start,start+8192,layer) for start in (0,8192) for layer in range(40)}
    fingerprints=[]
    for phase,mode in ((reference,'qualified_native'),(candidate,'all_selected')):
        sealed(phase)
        _fields(phase,dict(schema=PHASE_SCHEMA,mode=mode,index=0,arm='qualification',
            timing_valid=False,quality_eligible=False),'S1 native identity phase')
        record,control=phase['record'],phase['controller']
        require(control.get('expected_document_count')==10 and control.get('capture_sampler_events')==[],
            'S1 native identity must not rebind independent capture outputs')
        require(record.get('success') is True and record.get('rank')==0
            and record.get('input_tokens')==16384 and record.get('natural_EOS') is False
            and type(record.get('output_ids')) is list and len(record['output_ids'])==1
            and digest(record['output_ids'])==record.get('output_ids_sha256'),"S1 native identity output binding")
        rows=control.get('identity_fingerprints')
        require(type(rows) is list and len(rows)==80 and control.get('identity_check_count')==80,
            "all80 native layer/chunk fingerprints required")
        keys=[]
        for item in rows:
            keys.append(tuple(item.get('key',[])))
            value=item.get('value',{})
            require(set(value)=={'output','lazy'} and value['output'] is not None
                and set(value['lazy'])=={'residual','post_layer_mix','comb_res_mix','pre_layer_mix'},
                "native identity must bind output and all four mHC fields")
            _fingerprint(value['output'])
            expected_rows=len(natural.head.native_positions_reference(keys[-1][0],keys[-1][1],16384,keys[-1][2]))
            require(value['output']['shape'][0]==expected_rows,'native identity output row layout differs')
            for lazy in value['lazy'].values():_fingerprint(lazy)
        require(set(keys)==expected and len(set(keys))==80 and digest(rows)==control.get('identity_fingerprints_sha256'),
            "native identity fingerprints missing/duplicated/changed")
        require(control.get('identity_native_query_layout_preserved') is True
            and control.get('identity_all_kv_online') is True,"identity mode must preserve native layout and online KV")
        fingerprints.append(rows)
    for key in ('input_ids_sha256','execution_configuration_sha256','s1_protocol_sha256','manifest_sha256'):
        require(reference.get(key)==candidate.get(key),"native identity input/source binding differs")
    require(reference['record']['output_ids']==candidate['record']['output_ids']
        and fingerprints[0]==fingerprints[1] and candidate['controller'].get('identity_exact') is True,
        "all-selected identity route differs from native")
    return True


RUNTIME_CHECK_NAMES=('decoder_hidden_gather','four_mhc_fields_axis1','query_metadata_preserves_key_axes',
    'empty_packed_sf_abi','noncontiguous_absolute_positions','sampler_final_row_rebind',
    'engram_embedding_gather_and_empty_hook_consumption')


def validate_runtime_qualification(value,identity,empty_ep_sha256):
    sealed(value)
    _fields(value,dict(schema='s1-runtime-all-ranks-qualification-v1',passed=True,
        execution_configuration_sha256=identity['execution_configuration_sha256'],
        model_sha256=identity['checkpoint_index_sha256'],empty_ep_qualification_sha256=empty_ep_sha256,
        quality_eligible=False,timing_valid=False),'actual eight-rank S1 runtime qualification')
    reports=value.get('reports')
    require(type(reports) is list and len(reports)==8,'all eight actual GPU runtime reports required')
    for rank,report in enumerate(reports):
        sealed(report)
        _fields(report,dict(schema='s1-runtime-rank-qualification-v1',rank=rank,world_size=8,passed=True,
            execution_configuration_sha256=identity['execution_configuration_sha256'],model_sha256=identity['checkpoint_index_sha256'],
            cuda_exercised=True,quality_eligible=False,timing_valid=False,native_empty_ep_requires_separate_all_ranks_proof=True),
            'S1 rank-specific GPU ABI qualification')
        checks=report.get('checks')
        require(type(checks) is list and [e.get('name') for e in checks]==list(RUNTIME_CHECK_NAMES),
            'S1 runtime qualification coverage differs')
        for e in checks:
            require(e.get('passed') is True and e.get('actual_rows')==(0 if e['name']=='empty_packed_sf_abi' else 3)
                and type(e.get('cuda_tensor_device')) is str and e['cuda_tensor_device'].startswith('cuda'),
                'S1 runtime check did not exercise actual expected CUDA rows')
        validate_digest(report.get('output_sha256'),'S1 runtime gathered GPU output')
    return value


def sparse_identity(reference,candidate,row,qualification_sha256,*,controller_results=None):
    validate_digest(qualification_sha256,'S1 native qualification seal')
    banks=[];chunks=chunk_oracle(documents(row),row['input_tokens'])
    expected={(c['chunk_start'],c['chunk_end'],l):c for c in chunks for l in range(40)}
    for phase,mode in ((reference,'reference_sparse'),(candidate,'sensitive')):
        sealed(phase)
        _fields(phase,dict(schema=PHASE_SCHEMA,mode=mode,arm='s1',index=row['index'],fraction=None,
            input_ids_sha256=row['input_ids_sha256'],timing_valid=False,quality_eligible=False,diagnostic=False,
            natural_boundary_inputs=True,s1_policy=True,native_yoco_swa_share_hidden=False,**phase_policy('s1')),
            'S1 same-policy dense/compact qualification')
        control=phase['controller'];result=audit_controller(control,row,phase,documents(row))
        if controller_results is not None:controller_results[phase['sha256']]=result
        banks.append(comparison_scope(control,phase,spans=documents(row)))
        record=phase['record'];ids=record.get('output_ids')
        _fields(record,dict(success=True,rank=0,input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],
            natural_EOS=False),'S1 qualification raw request')
        require(type(ids) is list and len(ids)==1 and type(ids[0]) is int and ids[0]>=0
            and record.get('output_ids_sha256')==digest(ids),'S1 qualification output token hash differs')
        values=control.get('sparse_identity_fingerprints')
        require(type(values) is list and len(values)==len(expected)
            and control.get('sparse_identity_check_count')==len(expected)
            and control.get('sparse_identity_fingerprints_sha256')==digest(values),'S1 full-layer sparse fingerprints missing')
        seen=set()
        for item in values:
            key=tuple(item.get('key',[]));value=item.get('value',{})
            require(key in expected and key not in seen and set(value)=={'output','lazy'}
                and set(value['lazy'])=={'residual','post_layer_mix','comb_res_mix','pre_layer_mix'},
                'S1 sparse output/all-four-mHC fingerprint coverage differs')
            seen.add(key);require(value['output'] is not None,'S1 sparse decoder output absent')
            _fingerprint(value['output'])
            n=key[1]-key[0] if key[2]<3 else len(expected[key]['selected_positions'])
            require(value['output']['shape'][0]==n,'S1 fingerprint includes missing/extra decoder rows')
            for field in value['lazy'].values():
                _fingerprint(field)
                require(field is None or len(field['shape'])>=2 and field['shape'][:2]==[1,n],
                    'S1 mHC fingerprint token axis/rows differs')
    for key in ('manifest_sha256','execution_configuration_sha256','s1_protocol_sha256',
                'empty_ep_qualification_sha256','runtime_qualification_sha256'):
        require(reference.get(key)==candidate.get(key),'S1 dense/compact execution binding differs')
    require(banks[0]==banks[1] and reference['controller']['phase_generation']!=candidate['controller']['phase_generation'],
        'S1 qualification reused a phase or changed GPU bank')
    require(candidate.get('verify_sparse') is True and reference.get('verify_sparse') is False
        and candidate['controller'].get('sparse_identity_exact') is True
        and reference['record']['output_ids']==candidate['record']['output_ids']
        and reference['controller']['sparse_identity_fingerprints']==candidate['controller']['sparse_identity_fingerprints'],
        'S1 dense-policy/compact per-layer output/lazy byte parity failed')
    body=dict(schema='loong-s1-sparse-qualification-v1',passed=True,arm='s1',index=row['index'],
        input_tokens=row['input_tokens'],input_ids_sha256=row['input_ids_sha256'],qualification_sha256=qualification_sha256,
        reference_sha256=reference['sha256'],candidate_sha256=candidate['sha256'],
        identity_check_count=candidate['controller']['sparse_identity_check_count'],
        identity_fingerprints_sha256=candidate['controller']['sparse_identity_fingerprints_sha256'],
        quality_eligible=False,timing_valid=False,equivalence_to_full_model_claim=False,exact_same_policy_reference=True,
        empty_chunk_exercised=any(not c['selected_positions'] for c in chunks))
    return dict(body,sha256=digest(body))


def frozen_sources(preflight):
    """Recheck every bound Python/dependency/native source, without importing it."""
    from gpu_pilot.s1_protocol import DEPENDENCIES, REQUIRED_NEW_DEPENDENCIES
    require(DEPENDENCIES and all(k in DEPENDENCIES for k in REQUIRED_NEW_DEPENDENCIES), "S1 reader dependency pins have not been frozen")
    stage = Path(__file__).resolve().parents[1]
    identity = preflight["identity"]
    verify_core_bindings(identity,preflight["protocol"])
    root = Path(identity["experiment_root"])
    native = Path(identity["isolated_native_snapshot"])
    require(root.is_absolute() and native.is_absolute() and native.resolve() == stage.parent/"NativeSnapshot",
            "source roots differ from independent frozen deployment")
    recorded = identity.get("pilot_source_sha256")
    require(type(recorded) is dict and all(recorded.get(k) == v for k,v in DEPENDENCIES.items()),
            "preflight dependency pins differ from reader")
    paths = {"GatedEngram/"+str(p.relative_to(stage)):p for p in stage.rglob("*.py")
             if "__pycache__" not in p.parts}
    expected_stage = {k for k in recorded if k.startswith("GatedEngram/")}
    require(set(paths) == expected_stage, "frozen stage Python inventory changed")
    for name, expected in recorded.items():
        validate_digest(expected, "frozen source SHA")
        path = paths[name] if name in paths else root/name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "frozen source changed: "+name)
    native_files = identity["runtime_source_files_sha256"]
    require(digest(native_files) == identity["runtime_source_tree_sha256"], "native source aggregate differs")
    require(native.is_dir() and not native.is_symlink(), "native snapshot must be a real directory")
    git = ["git", "-c", "safe.directory="+str(native.resolve()), "-C", str(native)]
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], env=git_env).decode().strip()
    diff = subprocess.check_output([*git, "diff", "--no-ext-diff", "HEAD"], env=git_env)
    names = subprocess.check_output([*git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"], env=git_env)
    current_names = set()
    for raw in set(names.split(b"\0")):
        if raw:
            name = os.fsdecode(raw)
            path = native/name
            if path.is_file() and path.suffix in (".py", ".pyi", ".cpp", ".hpp", ".cu", ".h", ".toml"):
                require(not path.is_symlink(), "native runtime source symlink rejected")
                current_names.add(name)
    require(current_names == set(native_files) and head == identity["commit"]
            and hashlib.sha256(diff).hexdigest() == identity["source_diff_sha256"]
            and bool(diff) is identity["source_is_modified"], "native source inventory/commit/diff changed")
    for name, expected in native_files.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "native source path escapes snapshot")
        path = native/name
        require(path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected, "private native source changed: "+name)
    require(all(recorded.get('GatedEngram/'+name)==sha for name,sha in CORE_SHA256.items()),
        'S1 core algorithm differs from completed predecessor')
    return dict(pilot_source_sha256=digest(recorded), native_source_sha256=digest(native_files))


def audit_payload(path,sha256,byte_count,dtype):
    import array
    path=Path(path);validate_digest(sha256,'S1 raw payload SHA')
    require(path.is_file() and not path.is_symlink() and not path.parent.is_symlink()
        and path.stat().st_size==byte_count,'S1 raw payload file/extent differs')
    require(dtype in ('float32','bfloat16'),'foreign S1 payload dtype')
    h=hashlib.sha256();count=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            h.update(block);count+=len(block)
            values=array.array('f' if dtype=='float32' else 'H');values.frombytes(block)
            if sys.byteorder!='little':values.byteswap()
            require(all(math.isfinite(v) for v in values) if dtype=='float32'
                else all((v&0x7f80)!=0x7f80 for v in values),'nonfinite S1 offline payload')
    require(count==byte_count and h.hexdigest()==sha256,'S1 offline raw bytes changed')
    return dict(file_sha256=sha256,bytes=count,dtype=dtype,finite=True)


def audit_capture_sampler(control,full_input_tokens):
    """Audit output-only ABI rebinding after full40 independent capture.

    The original engine consumes the native YOCO suffix/placeholder layout.
    Capture computes all rows; only its final output is sliced, after banks and
    the native head/norm have completed. This asserts coordinates, not equality
    between an independent full-row document and native YOCO inference.
    """
    require(type(full_input_tokens) is int and full_input_tokens>=256,
        'S1 independent capture token extent differs')
    expected=[]
    for start in range(0,full_input_tokens,8192):
        end=min(start+8192,full_input_tokens);rows=end-start
        first=max(start,min(full_input_tokens-140,end-1))
        expected.append(dict(chunk_start=start,chunk_end=end,full_output_rows=rows,
            native_sampler_rows=end-first,output_local_start=first-start,
            output_last_absolute_position=end-1,source='full40_post_head_norm',
            rebinding_after_capture=True))
    require(control.get('capture_sampler_events')==expected,
        'S1 capture sampler output coordinates/coverage differ from native engine ABI')
    return expected


def audit_offline_phase(phase,row,ordinal,preflight,qualification,directory,evidence,*,prefix=None):
    spans=documents(row);lo,hi,key=spans[ordinal];length=hi-lo
    prefix=prefix or f"case{row['index']}"
    name=f'{prefix}.offline_doc{ordinal}'
    binding(phase,preflight);empty_ep_binding(phase,qualification)
    _fields(phase,dict(schema=PHASE_SCHEMA,name=name,index=None,arm='offline',mode='capture_independent',
        input_ids_sha256=key,diagnostic=False,partial_smoke=False,verify_sparse=False,s1_policy=False,
        timing_valid=False,quality_eligible=False,repeat=None,fraction=None,online_full_bank_gpu_resident=False),
        'S1 independent document capture phase')
    rec=phase['record'];ids=rec.get('output_ids')
    _fields(rec,dict(success=True,rank=0,input_ids_sha256=key,input_tokens=length,natural_EOS=False,
        phase=name,sample_id=key,dataset=row['dataset'],request_id=f's1:doc:{name}:{key}',
        finish_reason='length',truncated=False),'S1 offline source request')
    require(type(ids) is list and len(ids)==1 and type(ids[0]) is int and ids[0]>=0
        and rec.get('output_ids_sha256')==digest(ids),'S1 offline output token hash')
    control=phase['controller'];cache=control.get('cache_report',{})
    _fields(control,dict(schema='s1-controller-v1',mode='capture_independent',arm='offline',phase=name,
        expected_document_count=10,
        input_ids_sha256=key,full_input_tokens=length,model_boundary_restored=True,first_three_full=True,
        decoder_events=[],source2_identity_events=[],cache_scope=execution_scope(preflight['identity'])),
        'S1 offline native execution scope')
    audit_capture_sampler(control,length)
    previous=spans[:ordinal+1];lengths=[b-a for a,b,_ in previous]
    _fields(control,dict(offline_bank_entry_count=40*len(previous),offline_swa_halo_rows=37*128*len(previous),
        offline_global_raw_rows=3*sum(lengths),offline_bank_rows=37*128*len(previous)+3*sum(lengths),
        offline_bank_bytes=37*128*len(previous)*512*2+9216*sum(lengths),
        offline_bank_scope_sha256=bank_scope_sha256(previous,preflight['identity'])),
        'progressive S1 offline bank extent/source')
    captures=cache.get('capture_reports')
    require(type(captures) is list and len(captures)==ordinal+1,'S1 independent document capture coverage')
    capture=captures[-1]
    _fields(capture,dict(schema='s1-offline-capture-v1',document_sha256=key,document_tokens=length,
        swa_halo_bytes=37*128*512*2,global_raw_bytes=length*9216,
        full_swa_bank_persisted=False,hidden_or_ffn_bank_persisted=False),'S1 halo and precompression source')
    payloads=capture.get('payloads')
    require(type(payloads) is list and len(payloads)==40 and payloads==control.get('offline_payloads'),
        'S1 37 halo and three global payloads required')
    expected={('swa_halo',l) for l in DEEP_LAYERS}|{('global_raw',l) for l in (8,14,20)};seen=set()
    for item in payloads:
        kind,layer=item.get('kind'),item.get('layer');identity=(kind,layer)
        require(identity in expected and identity not in seen,'S1 repeated/foreign source payload');seen.add(identity)
        halo=kind=='swa_halo';rows=128 if halo else length
        width=512 if halo or layer==20 else 1024;dtype='bfloat16' if halo or layer==20 else 'float32'
        size=rows*width*(2 if dtype=='bfloat16' else 4);filename=f'{key}.{kind}.layer{layer}.bin'
        _fields(item,dict(kind=kind,layer=layer,rows=rows,width=width,dtype=dtype,
            local_start=length-256 if halo else 0,bytes=size,document_sha256=key,source_document_tokens=length,
            scope_sha256=digest(execution_scope(preflight['identity'])),file=filename),'S1 raw payload scope/layout')
        relative=f'{prefix}.offline-kv/{filename}'
        require(relative not in evidence,'S1 payload must be read exactly once per document')
        evidence[relative]=audit_payload(Path(directory)/relative,item.get('file_sha256'),size,dtype)
    return control


def qualification(directory,preflight,evidence):
    mask=common.cuda_mask_qualification(directory,preflight,evidence)
    from gpu_pilot.index_hit20_empty_ep_qualification import validate_all_ranks_report
    empty=read(directory/'empty_ep_qualification.json',evidence)
    validate_all_ranks_report(empty,execution_sha256=preflight['identity']['execution_configuration_sha256'],
        model_sha256=preflight['identity']['checkpoint_index_sha256'])
    runtime=read(directory/'s1_runtime_qualification.json',evidence)
    validate_runtime_qualification(runtime,preflight['identity'],empty['sha256'])
    reference=read(directory/'qualification.native.json',evidence)
    candidate=read(directory/'qualification.all_selected.json',evidence)
    closure=read(directory/'qualification.json',evidence)
    _fields(closure,dict(schema='loong-s1-qualification-v1',passed=True,cuda_mask_qualification_sha256=mask['sha256'],
        empty_ep_qualification_sha256=empty['sha256'],runtime_qualification_sha256=runtime['sha256'],
        reference_sha256=reference['sha256'],candidate_sha256=candidate['sha256'],identity_check_count=80,
        qualification_input_tokens=16384,quality_claim=False,full_input_numerical_equivalence_claim=False,
        old_qualification_reused=False),'S1 fresh native/runtime qualification closure')
    for phase,name in ((reference,'qualification.native'),(candidate,'qualification.all_selected')):
        binding(phase,preflight);empty_ep_binding(phase,closure)
        _fields(phase,dict(name=name,runtime_qualification_sha256=runtime['sha256'],
            input_ids_sha256=preflight['qualification_input_ids_sha256']),'S1 calibration input binding')
        require(phase['controller']['cache_report'].get('offline_bank_entry_count')==0,
            'native all-selected identity used an offline bank')
    native_identity(reference,candidate)
    return closure


def empty_qualification(directory,preflight,qual,row,evidence):
    prefix='qualification.empty_s1'
    reference=read(directory/(prefix+'.reference.json'),evidence)
    candidate=read(directory/(prefix+'.compact.json'),evidence)
    closure=read(directory/'s1_empty_chunk_qualification.json',evidence)
    for phase in (reference,candidate):binding(phase,preflight);empty_ep_binding(phase,qual)
    expected=sparse_identity(reference,candidate,row,qual['sha256'])
    require(closure.get('sparse_qualification')==expected and expected['empty_chunk_exercised'] is True,
        'actual empty-chunk dense/compact qualification is absent')
    _fields(closure,dict(schema='loong-s1-empty-chunk-qualification-v1',passed=True,
        qualification_sha256=qual['sha256'],input_ids_sha256=row['input_ids_sha256'],input_tokens=16384,
        first_chunk_selected_rows=0,second_chunk_selected_rows=128,deep_empty_layer_calls=37,
        quality_eligible=False,timing_valid=False),'actual empty first chunk contract')
    capture=read(directory/(prefix+'.offline_doc0.json'),evidence)
    audit_offline_phase(capture,row,0,preflight,qual,directory,evidence,prefix=prefix)
    require(closure.get('offline_phase_sha256')==capture['sha256'],'empty qualification offline source detached')
    return closure


def partial_smoke(directory,preflight,qual,evidence):
    row=preflight['audit_cases'][0]
    reference=read(directory/'case2.s1_sparse_reference.json',evidence)
    candidate=read(directory/'case2.s1_partial_smoke.json',evidence)
    sparse=read(directory/'s1.sparse_qualification.json',evidence)
    closure=read(directory/'s1.partial_smoke.json',evidence)
    for phase,name in ((reference,'case2.s1_sparse_reference'),(candidate,'case2.s1_partial_smoke')):
        binding(phase,preflight);empty_ep_binding(phase,qual)
        require(phase.get('name')==name and phase.get('repeat') is None,'S1 smoke filename binding')
        comparison_scope(phase['controller'],phase,spans=documents(row),
            expected_scope=bank_scope_sha256(documents(row),preflight['identity']))
    require(sparse==sparse_identity(reference,candidate,row,qual['sha256']),'S1 same-policy sparse closure differs')
    _fields(closure,dict(schema='loong-s1-partial-smoke-v1',passed=True,arm='s1',index=2,
        qualification_sha256=qual['sha256'],sparse_qualification_sha256=sparse['sha256'],
        phase_sha256={'s1':candidate['sha256']},input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],
        quality_eligible=False,timing_valid=False,accuracy_guarantee=False,full_input_original_ten_documents=True,
        independent_swa_and_ffn_paths_exercised=True,output_tokens_per_arm=1),'S1 full-input smoke closure')
    require(candidate.get('partial_smoke') is True and reference.get('partial_smoke') is False,
        'S1 smoke eligibility swapped')
    actual=audit_phase(candidate,row,preflight,arm='s1',timing=False,quality=False,fixed_tokens=1)
    require(all(actual.get(k,0)>0 for k in ('ffn_bypassed_rows','residual_bypassed_rows','mhc_bypassed_rows',
        'consumed_cached_pairs','dynamic_refreshed_rows')),'S1 smoke did not exercise complete bypass and halo reuse')
    return closure,dict(passed=True,actual_rows=actual,closure_sha256=closure['sha256'],
        sparse_qualification_sha256=sparse['sha256'],same_policy_byteexact=True,equivalence_to_full_model_claim=False,
        shared_bank_identity=comparison_scope(candidate['controller'],candidate,spans=documents(row)))


def audit_case(directory,row,preflight,qual,smoke,empty,evidence,*,decoder,source_row):
    index=row['index'];pair=read(directory/f'case{index}.paired.json',evidence)
    _fields(pair,dict(schema='loong-s1-case-paired-v1',qualification_sha256=qual['sha256'],
        partial_smoke_sha256={'s1':smoke['sha256']},empty_chunk_qualification_sha256=empty['sha256'],
        ffn_layers=list(DEEP_LAYERS),offline_swa_layers=list(DEEP_LAYERS),offline_global_source_layers=[8,14,20],
        all_negative_results_retained=True,natural_boundary_inputs=True,input_geometry=case_geometry(row),
        calibration_claim=False,fresh_paired_baseline=True),'S1 paired formal closure')
    _fields(pair,{k:row[k] for k in ('index','dataset','sample_id','references','question','input_tokens','input_ids_sha256')},
        'S1 exact question/gold/input identity')
    _fields(pair,{k:row[k] for k in SCOPE_FIELDS},'long-context length closure identity')
    require(set(pair.get('quality',{}))==set(ARMS),'both fresh S1/full quality answers required')
    samples={arm:[] for arm in ARMS};quality={};actual={};generations=set();bank=None
    clocks={arm:{k:[] for k in ('online_cache_prepare_s','online_prepare_to_submit_s','engine_ttft_s')} for arm in ARMS}
    def bind(phase):
        nonlocal bank
        empty_ep_binding(phase,qual)
        require(phase.get('runtime_qualification_sha256')==qual['runtime_qualification_sha256'],
            'S1 formal phase detached from actual runtime qualification')
        observed=comparison_scope(phase['controller'],phase,spans=documents(row),
            expected_scope=bank_scope_sha256(documents(row),preflight['identity']))
        require(bank is None or observed==bank,'S1 bank changed between formal arms/repeats')
        bank=observed
        generation=phase['controller']['phase_generation']
        require(generation not in generations,'S1 formal overlay generation reused')
        generations.add(generation)
        require(phase.get('partial_smoke') is False and phase.get('verify_sparse') is False,
            'S1 formal phase ran qualification synchronization')
    for arm in ARMS:
        name=f'case{index}.{arm}_quality';phase=read(directory/(name+'.json'),evidence)
        require(phase.get('name')==name and phase.get('repeat') is None,'S1 quality filename identity')
        verify_reference(pair['quality'][arm],directory/(name+'.json'),phase,evidence)
        actual[arm]=audit_phase(phase,row,preflight,arm=arm,timing=False,quality=True);bind(phase)
        quality[arm]=quality_record(phase['record'],phase,row)
        quality[arm]['output_tokenization']=audit_decoded_record(phase['record'],decoder)
        name=f'case{index}.{arm}_warmup';phase=read(directory/(name+'.json'),evidence)
        require(phase.get('name')==name and phase.get('repeat') is None,'S1 warmup filename identity')
        audit_phase(phase,row,preflight,arm=arm,timing=False,quality=False);bind(phase)
    refs=pair.get('timing_phases');schedule=timing_schedule(index)
    require(type(refs) is list and [(r.get('repeat'),r.get('arm')) for r in refs]==schedule,
        'S1 requires six rotated fresh measurements')
    for ref,(repeat,arm) in zip(refs,schedule):
        name=f'case{index}.{arm}_measure{repeat}';phase=read(directory/(name+'.json'),evidence)
        verify_reference(ref,directory/(name+'.json'),phase,evidence,extras=('repeat','arm'))
        require(phase.get('name')==name and phase.get('repeat')==repeat,'S1 timing repeat/filename identity')
        audit_phase(phase,row,preflight,arm=arm,timing=True,quality=False);bind(phase)
        samples[arm].append(phase['record']['online_ttft_s'])
        for key,value in existing_timing_breakdown(phase).items():clocks[arm][key].append(value)
    summary=expected_timing_summary(samples)
    require(pair.get('timing_summary')==summary and len(generations)==10
        and pair.get('shared_bank_identity')==bank,'S1 ten-phase bank/timing closure differs')
    builds=[]
    for ordinal in range(len(documents(row))):
        phase=read(directory/f'case{index}.offline_doc{ordinal}.json',evidence)
        control=audit_offline_phase(phase,row,ordinal,preflight,qual,directory,evidence)
        require(control.get('bank_generation')==bank['bank_generation'],'S1 offline/formal bank generation differs')
        builds.append(phase['sha256'])
    require(pair.get('offline_phase_sha256')==builds,'S1 all ten document builds are not bound')
    result=dict(index=index,dataset=row['dataset'],question=row['question'],instruction=source_row['instruction'],
        prompt_template=source_row['prompt_template'],references=row['references'],
        input_tokens=row['input_tokens'],input_geometry=case_geometry(row),quality=quality,
        online_ttft_samples_s=samples,online_ttft_p50_s={arm:statistics.median(v) for arm,v in samples.items()},
        comparisons=summary['comparisons'],actual_rows_and_swa_reuse=actual,shared_bank_identity=bank,
        offline_build_s=number(pair.get('offline_build_s'),'S1 offline build'),offline_storage_bytes=bank['offline_bank_bytes'],
        closure_sha256=pair['sha256'],fresh_paired_baseline=True,historical_results_used_for_admission=False,
        token_divergence=compare_quality_records(quality['recomputed'],quality['s1'],decoder),
        metric_policy=metric_policy(row['dataset']),
        length_band=row['length_band'],length_band_ordinal=row['length_band_ordinal'],target_input_tokens=row['target_input_tokens'],
        timing_breakdown_samples_s=clocks,
        raw_answer=source_row['raw_answer'],raw_answer_sha256=source_row['raw_answer_sha256'],
        source_domain=row['source_domain'],source_language=row['source_language'],source_level=row['source_level'],
        source_document_count=row['source_document_count'],original_document_count=row['original_document_count'],
        original_documents=source_row['original_documents'],cache_blocks=source_row['documents'],
        answer_agreement=answer_agreement(quality['recomputed']['text'],quality['s1']['text'],source_row['raw_answer']),
        selected_prompt_tokens=len(position_oracle(documents(row),row['input_tokens'])['selected_positions']),
        source_dataset=row['dataset'],source_row_sha256=source_row['source_row_sha256'],
        context_sha256=source_row['context_sha256'],
        document_content_sha256=[d['token_ids_sha256'] for d in source_row['documents']],
        document_context_char_spans=[[d['context_char_start'],d['context_char_end']] for d in source_row['documents']])
    return pair,result


def summarize_cases(cases):
    if not cases:return None
    arms={}
    for arm in ARMS:
        samples=[s for case in cases for s in case['online_ttft_samples_s'][arm]]
        arms[arm]=dict(n=len(cases),samples=len(samples),online_ttft_samples_s=samples,
            online_ttft_p50_s=statistics.median(samples),
            em_percent=100*sum(c['quality'][arm]['answer_em'] for c in cases)/len(cases),
            f1_percent=100*sum(c['quality'][arm]['answer_token_f1'] for c in cases)/len(cases),
            truncated_count=sum(c['quality'][arm]['truncated'] for c in cases),
            loong_metrics=summarize_scores([c['quality'][arm]['loong_score'] for c in cases]),
            generated_tokens=[c['quality'][arm]['generated_tokens'] for c in cases],
            non_special_generated_tokens=[c['quality'][arm]['output_tokenization']['non_special_generated_tokens'] for c in cases],
            non_special_at_least50_count=sum(c['quality'][arm]['output_tokenization']['non_special_at_least50'] for c in cases),
            non_special_between50_and200_count=sum(c['quality'][arm]['output_tokenization']['non_special_between50_and200'] for c in cases))
    return dict(n=len(cases),arms=arms,comparisons={'s1':dict(ttft_speedup=arms['recomputed']['online_ttft_p50_s']/arms['s1']['online_ttft_p50_s'])},
        offline_build_total_s=sum(c['offline_build_s'] for c in cases),
        offline_storage_total_bytes=sum(c['offline_storage_bytes'] for c in cases),
        historical_results_used_for_admission=False,fresh_paired_baseline=True,statistical_significance_claim=False,
        cold_cache_ttft_claim=False,counts_are_not_flops_or_measured_time_fraction=True,
        identical_output_ids_count=sum(c['token_divergence']['identical_output_ids'] for c in cases),
        first_difference_positions_1based=[c['token_divergence']['first_difference_position_1based'] for c in cases],
        both_outputs_at_least50_non_special_count=sum(c['token_divergence']['both_outputs_at_least50_non_special_tokens'] for c in cases),
        quality_metrics_must_be_interpreted_per_dataset=True,
        answer_agreement=dict(available=sum(c['answer_agreement']['structural_comparison_available'] for c in cases),
            equal=sum(c['answer_agreement']['structural_equal'] is True for c in cases),
            agreement_is_gold_correctness=False),
        timing_breakdown_samples_s={arm:{k:[v for c in cases for v in c['timing_breakdown_samples_s'][arm][k]]
            for k in ('online_cache_prepare_s','online_prepare_to_submit_s','engine_ttft_s')} for arm in ARMS})


def validate_closed_run(run,preflight,qual,smoke,empty,closures,errors,directory,evidence):
    sealed(run)
    require(not errors and len(closures)==20 and run.get('schema')==RUN_SCHEMA and run.get('status')=='complete'
        and run.get('completed_case_indices')==list(CASE_INDICES) and [c.get('index') for c in closures]==list(CASE_INDICES)
        and run.get('identity')==preflight['identity'] and run.get('manifest_sha256')==preflight['manifest_sha256']
        and run.get('qualification')==qual and run.get('partial_smoke_sha256')=={'s1':smoke['sha256']}
        and run.get('empty_chunk_qualification_sha256')==empty['sha256']
        and type(run.get('cases')) is list and len(run['cases'])==20,'S1 run does not close all twenty fresh paired cases')
    generations=[]
    for ref,closure in zip(run['cases'],closures):
        index=closure['index'];require(ref.get('index')==index,'S1 final case order differs')
        verify_reference(ref,Path(directory)/f'case{index}.paired.json',closure,evidence,extras=('index',))
        generations.append(closure['shared_bank_identity']['bank_generation'])
    require(generations==sorted(set(generations)) and len(generations)==20,'S1 bank generations reused across cases')
    return True


def completion_state(results,errors,*,complete,case_index=None,allow_partial=False,partial_reason=None):
    """Explicit partial reporting never promotes case closure to whole-run closure."""
    require(not (case_index is not None and allow_partial),'choose one-case or partial-run audit, not both')
    require((allow_partial and partial_reason in ('hard_deadline','runner_failure','bounded_window'))
        or (not allow_partial and partial_reason is None),
        'partial audit requires explicit termination reason')
    indices=[c['index'] for c in results]
    require(indices==sorted(set(indices)) and all(i in CASE_INDICES for i in indices),
        'MuSiQue audited case indices repeated, foreign or reordered')
    generations=[c['shared_bank_identity']['bank_generation'] for c in results]
    require(all(type(g) is int and g>0 for g in generations) and generations==sorted(set(generations)),
        'MuSiQue partial/formal cases reused or reordered bank generations')
    if complete:
        require(case_index is None and not errors and indices==list(CASE_INDICES),
            'whole MuSiQue run cannot be complete without all20 audited cases')
    partial=bool(allow_partial and not errors and not complete)
    if partial:
        require(len(indices)<20 and indices==list(CASE_INDICES)[:len(indices)],
            'partial audit requires the uninterrupted completed case prefix')
    subset=case_index is not None and indices==[case_index] and not errors
    status='invalid' if errors else 'complete' if complete else 'partial_complete' if partial else 'subset_complete' if subset else 'incomplete'
    return dict(status=status,completed_case_indices=indices,complete_cases=len(indices),planned_cases=20,
        missing_case_indices=[i for i in CASE_INDICES if i not in indices],
        partial_audit_requested=allow_partial,partial_results_only=partial or subset,
        partial_completion_reason=partial_reason if partial else None,
        all_completed_cases_audited=case_index is None and not errors,
        final_run_audited=complete,overall_run_complete=complete,all_registered_cases_audited=complete)


def audit(directory,case_index=None,*,allow_partial=False,partial_reason=None):
    directory=Path(directory);evidence={};errors=[];results=[];closures=[]
    require(case_index is None or type(case_index) is int and case_index in CASE_INDICES,'unknown S1 subset index')
    preflight=read(directory/'preflight.json',evidence)
    _fields(preflight,dict(schema=RUN_SCHEMA,protocol=s1_protocol(),evaluation_case_indices=list(CASE_INDICES),
        natural_boundary_inputs=True,native_yoco_swa_share_hidden=False,qualification_input_tokens=16384),
        'S1 actual execution preflight')
    common.audit_native_swa_configuration(preflight);source_before=frozen_sources(preflight)
    manifest=read(directory/'input_manifest.json',evidence);inputs=read(directory/'input-protocol.json',evidence)
    for name in INPUT_COMPANIONS:read(directory/name,evidence)
    sys.path.insert(0,str(Path(preflight['identity']['experiment_root'])/'benchmark'))
    import pr148_native_runner as base
    from gpu_pilot.loong_inputs import validate_inputs
    validate_inputs(base,manifest,inputs,input_dir=directory)
    require(manifest['sha256']==preflight['manifest_sha256']==preflight['protocol']['input_manifest_sha256'],
        'S1 source manifest differs from original natural cohort')
    decoder=decoder_for_identity(preflight['identity'])
    calibration=manifest['warmup'][0];ids=calibration['input_ids'][:16384]
    require(calibration['index']==0 and len(ids)==16384 and digest(ids)==preflight.get('qualification_input_ids_sha256'),
        'S1 native qualification did not use original calibration16K')
    rows=preflight.get('audit_cases');by_index={row['index']:row for row in manifest['cohort']}
    require(type(rows) is list and [row.get('index') for row in rows]==list(CASE_INDICES),'S1 twenty-case cohort incomplete')
    audit_length_inputs(manifest['cohort'])
    require([row['dataset'] for row in rows]==DATASET_ORDER, 'Long-context paired dataset order differs')
    for row in rows:
        source=by_index[row['index']]
        _fields(row,{k:source[k] for k in SCOPE_FIELDS},'preflight long-context source binding')
        require(all(source.get(k)==v for k,v in row.items()),'S1 preflight input/gold/source drift')
        require(len(source['input_ids'])==row['input_tokens'] and digest(source['input_ids'])==row['input_ids_sha256'],
            'S1 full input token identity differs')
        for lo,hi,key in documents(row):require(hi-lo>=256 and digest(source['input_ids'][lo:hi])==key,'S1 document span identity')
        row['raw_answer']=source['raw_answer']
    geometries=[case_geometry(row) for row in rows]
    _fields(preflight.get('input_geometry',{}),dict(schema='loong-s1-input-geometry-v1',cases=geometries,
        sparse_qualification=dict(index=2,input_tokens=rows[0]['input_tokens'],layer_chunk_checks=geometries[0]['layer_chunk_checks']),
        offline_payload_bytes_all_cases=sum(g['offline_bank_bytes'] for g in geometries)),
        'S1 actual halo/raw-global geometry')
    qual=qualification(directory,preflight,evidence)
    key=digest(ids)
    empty_row=dict(index=0,dataset=calibration['dataset'],input_tokens=16384,input_ids_sha256=key,
        sample_id='s1-empty-'+key,references=['__same_policy_qualification_not_quality__'],s1_empty_chunk_qualification=True,
        documents=[dict(input_start=0,input_end=16384,token_ids_sha256=key)])
    empty=empty_qualification(directory,preflight,qual,empty_row,evidence)
    smoke,smoke_summary=partial_smoke(directory,preflight,qual,evidence)
    requested=rows if case_index is None else [r for r in rows if r['index']==case_index]
    for row in requested:
        if not (directory/f"case{row['index']}.paired.json").exists():continue
        try:
            closure,result=audit_case(directory,row,preflight,qual,smoke,empty,evidence,decoder=decoder,source_row=by_index[row['index']])
            if row['index']==2:require(result['shared_bank_identity']==smoke_summary['shared_bank_identity'],'S1 first formal bank changed after smoke')
            closures.append(closure);results.append(result)
        except (ValueError,KeyError,TypeError,OSError) as exc:errors.append(dict(index=row['index'],error=str(exc)))
    complete=False
    if case_index is None and (directory/'run.json').exists():
        complete=validate_closed_run(read(directory/'run.json',evidence),preflight,qual,smoke,empty,closures,errors,directory,evidence)
    completion=completion_state(results,errors,complete=complete,case_index=case_index,
        allow_partial=allow_partial,partial_reason=partial_reason)
    require(frozen_sources(preflight)==source_before,'S1 source changed during independent CPU audit')
    result=dict(schema=SCHEMA,**completion,arms=list(ARMS),errors=errors,cases=results,evidence=evidence,
        frozen_sources_verified=source_before,cuda_initialized=False,
        qualification_passed=True,empty_chunk_qualification_sha256=empty['sha256'],partial_smoke=smoke_summary,
        manifest_sha256=manifest['sha256'],input_protocol_sha256=inputs['sha256'],protocol_sha256=preflight['protocol']['sha256'],
        execution_configuration_sha256=preflight['identity']['execution_configuration_sha256'],
        completed_by_length_band={band:sum(c['length_band']==band for c in results) for band in LENGTH_BANDS},
        by_length_band={band:summarize_cases([c for c in results if c['length_band']==band]) for band in LENGTH_BANDS},
        by_language={v:summarize_cases([c for c in results if c['source_language']==v]) for v in sorted({c['source_language'] for c in results})},
        by_domain={v:summarize_cases([c for c in results if c['source_domain']==v]) for v in sorted({c['source_domain'] for c in results})},
        length_trend_is_fixed_overhead_causal_proof=False,new_profiling_added=False,
        core_algorithm_unchanged=True,core_algorithm_files_sha256=CORE_SHA256,
        **verify_core_bindings(preflight["identity"],preflight["protocol"]),
        summary=summarize_cases(results),by_dataset={d:summarize_cases([c for c in results if c['dataset']==d]) for d in sorted({c['dataset'] for c in results})},
        quality_texts_preserved_in_full=True,quality_output_ids_preserved_in_full=True,quality_output_policy=preflight['protocol']['quality'],
        actual_output_tokenizer=decoder.identity,all_output_texts_redecoded=True,
        first_token_divergence_is_causal_error_claim=False,outputs_minimum_length_forced=False,
        output_length_eligibility_is_posthoc_observation=True,
        historical_results_used_for_admission=False,fresh_all_twenty_paired_baselines=complete,
        cold_start_claim=False,no_flops_fraction_claim=True,independent_holdout_claim=False,
        cache_storage_policy='same_s1_offline_gpu_bank_resident_both_arms',
        offline_build_in_online_ttft=False,offline_build_cost_reported=True,
        input_data_contract=dict(datasets=list(DATASETS),evaluation_cases=20,
            length_bands=list(LENGTH_BANDS),target_input_tokens=TARGET_INPUT_TOKENS,
            actual_tokenizer_band_limits={band:list(v) for band,v in BAND_LIMITS.items()},cases_per_length_band=10,
            cohort_order='128k then 64k, interleaved ten times',original_qa_contexts_per_case=1,original_document_count_range=[5,10],
            cache_blocks_are_original_documents=False,cache_blocks_cross_original_document_boundaries=False,
            natural_cache_blocks_per_case=10,minimum_cache_block_tokens=256,
            calibration_source=manifest['calibration_source'],calibration_in_quality_metrics=False,
            gold_length_selection=False,selection_uses_generated_output_length=False,
            actual_prompt_length_selection=True,selection_uses_model_outcomes=False,
            full_original_context_preserved=True,filler_added=False,content_truncated=False,
            original_documents_question_instruction_and_gold_preserved=True,primary_metric=PRIMARY_METRIC,
            official_loong_gpt4_score=False,external_judge_called=False,
            official_random_shuffle_applied=False,official_length_based_document_omission_applied=False,
            generation_matches_official_benchmark_settings=False))
    return dict(result,sha256=digest(result))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--case-index',type=int,choices=CASE_INDICES)
    parser.add_argument('--allow-partial',action='store_true',
        help='audit the fully closed case prefix without claiming completion of all20')
    parser.add_argument('--partial-reason',choices=('hard_deadline','runner_failure','bounded_window'))
    args=parser.parse_args(argv);result=audit(args.result_dir,args.case_index,
        allow_partial=args.allow_partial,partial_reason=args.partial_reason)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8') as handle:handle.write(payload)
    else:print(payload,end='')
    return 0 if result['status'] in ('complete','subset_complete','partial_complete') and not result['errors'] else 1


if __name__=='__main__':raise SystemExit(main())
