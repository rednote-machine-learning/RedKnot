"""Independent fresh20 three-arm audit, including shared full GPU bank evidence."""
from __future__ import annotations
import argparse
import base64
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kv_cache_engram.core import digest,require,validate_digest
from gpu_pilot import read_headtail_link_results as head
from gpu_pilot import read_tail_link_results as common
from gpu_pilot.headtail_scale20_protocol import headtail_scale20_protocol,phase_policy,timing_schedule
from gpu_pilot.headtail_scale20_inputs import validate_inputs
SCHEMA='headtail-scale20-independent-readout-v1'
PHASE_SCHEMA='kv-cache-engram-headtail-scale20-phase-v1'
RUN_SCHEMA='kv-cache-engram-headtail-scale20-pilot-run-v1'
ARMS=('recomputed','tail128_link','headtail128_link')
POLICY_ARMS=ARMS[1:]
CASE_INDICES=tuple(range(2,22))
FFN_LAYERS=tuple(range(3,40))
CAPTURE_LAYERS=tuple(range(40))
DOCUMENT,WIDTH,WINDOW=32768,512,128
read,sealed,number=common.read,common.sealed,common.number
documents,score,verify_reference=common.documents,common.score,common.verify_reference
empty_ep_binding=common.empty_ep_binding
_event_map,_fingerprint=common._event_map,common._fingerprint
_fields=head._fields
audit_native_swa_configuration=common.audit_native_swa_configuration
audit_bank_payload=head.audit_bank_payload
audit_native_no_cache=common.audit_native_no_cache
projection_identity=common.projection_identity
cache_restore_identity=common.cache_restore_identity
audit_joint_attention=common.audit_joint_attention
cuda_mask_qualification=common.cuda_mask_qualification
DIAGNOSTIC_LAYERS=(0,1,2,3,7,13,19,20,39)
DIAGNOSTIC_EDGES=(0,128,256,384,512,1024,4096,8192,16384,32640,32768)


def compare_statistics(actual,expected,path='diagnostic statistics'):
    """Only roundoff in derived CPU aggregates; native qualification stays exact."""
    if isinstance(expected,dict):
        require(type(actual) is dict and set(actual)==set(expected),path+' fields differ')
        for key,value in expected.items():compare_statistics(actual[key],value,path+'.'+key)
    elif isinstance(expected,list):
        require(type(actual) is list and len(actual)==len(expected),path+' count differs')
        for index,value in enumerate(expected):compare_statistics(actual[index],value,path+f'[{index}]')
    elif type(expected) is float:
        require(type(actual) in (int,float) and math.isfinite(actual)
            and math.isclose(actual,expected,rel_tol=2e-12,abs_tol=1e-12),path+' recomputation differs')
    else:require(type(actual) is type(expected) and actual==expected,path+' differs')


def unpack_diagnostic_scalars(block,expected_positions):
    import numpy as np
    n=len(expected_positions)
    _fields(block,dict(encoding='zlib-base64',position_dtype='<u4',value_dtype='<f4',
        layout='positions_then_row_major_values',columns=['error_l2_squared','reference_l2_squared','cosine_distance'],
        rows=n,raw_bytes=n*16),'diagnostic scalar encoding')
    require(0<n<=32768 and type(block.get('data')) is str and len(block['data'])<=n*32+4096,
            'diagnostic scalar geometry or compressed payload limit exceeded')
    validate_digest(block.get('raw_sha256'),'raw diagnostic scalar SHA')
    try:
        packed=base64.b64decode(block['data'],validate=True);inflater=zlib.decompressobj()
        raw=inflater.decompress(packed,n*16+1)
    except (ValueError,zlib.error) as exc:raise ValueError('invalid compressed diagnostic scalar evidence') from exc
    require(inflater.eof and not inflater.unused_data and not inflater.unconsumed_tail and len(raw)==n*16
        and hashlib.sha256(raw).hexdigest()==block['raw_sha256'],'diagnostic scalar SHA/length/stream differs')
    positions=np.frombuffer(raw,dtype='<u4',count=n)
    values=np.frombuffer(raw,dtype='<f4',offset=n*4).reshape(n,3)
    require(np.array_equal(positions,np.asarray(expected_positions,dtype='<u4')),
            'diagnostic positions differ from actual native surviving rows')
    require(np.isfinite(values).all() and (values>=0).all() and (values[:,2]<=2).all(),
            'diagnostic scalar nonfinite, negative or invalid cosine distance')
    return positions,values


def diagnostic_summary(positions,values):
    import numpy as np
    p=np.asarray(positions);v=np.asarray(values,dtype=np.float64)
    energy,reference,cosine=v[:,0],v[:,1],v[:,2]
    relative=np.sqrt(energy/np.maximum(reference,1e-12));total=float(energy.sum())
    def group(mask):
        rows=int(np.count_nonzero(mask));error=float(energy[mask].sum())
        return dict(rows=rows,error_energy=error,mean_relative_l2=float(relative[mask].mean()) if rows else None,
            mean_cosine_distance=float(cosine[mask].mean()) if rows else None,
            error_energy_fraction=error/total if total else None)
    bins=[dict(local_start=lo,local_end=hi,**group((p>=lo)&(p<hi)))
          for lo,hi in zip(DIAGNOSTIC_EDGES,DIAGNOSTIC_EDGES[1:])]
    boundaries={}
    for width in (128,256,384,512):
        boundaries[f'head{width}']=group(p<width)
        boundaries[f'head{width}_tail128']=group((p<width)|(p>=32640))
    boundaries['tail128']=group(p>=32640)
    order=np.lexsort((p,-energy));budgets=[]
    for width in (128,256,384,512):
        k=min(width,len(p));budgets.append(dict(budget=k,
            top_error_energy_fraction=float(energy[order[:k]].sum()/total) if total else None,
            uniform_random_expected_energy_fraction=k/len(p)))
    return dict(observed_rows=len(p),possible_rows=32768,coverage=len(p)/32768,error_energy=total,
        reference_energy=float(reference.sum()),relative_frobenius=float(np.sqrt(total/max(float(reference.sum()),1e-12))),
        relative_l2_percentiles={str(q):float(np.percentile(relative,q)) for q in (50,90,95,99,100)},
        bins=bins,boundaries=boundaries,oracle_budgets=budgets)


def diagnostic_row_groups(positions,values,document_start,layer,total):
    import numpy as np
    absolute=np.asarray(positions)+document_start;v=np.asarray(values,dtype=np.float64)
    rect=max(0,total-140);masks={
        'full_layout':np.full(len(absolute),layer<20,dtype=bool),
        'yoco_placeholder':(absolute<rect)&(layer>=20),
        'yoco_final_rectangle':(absolute>=rect)&(layer>=20)}
    groups={}
    for name,mask in masks.items():
        rows=int(np.count_nonzero(mask));error=float(v[mask,0].sum());reference=float(v[mask,1].sum())
        groups[name]=dict(rows=rows,error_energy=error,reference_energy=reference,
            relative_frobenius=float(np.sqrt(error/max(reference,1e-12))) if rows else None,
            mean_relative_l2=float(np.sqrt(v[mask,0]/np.maximum(v[mask,1],1e-12)).mean()) if rows else None,
            mean_cosine_distance=float(v[mask,2].mean()) if rows else None)
    return groups


def diagnostic_rank_comparison(a_positions,a_values,b_positions,b_values):
    import numpy as np
    positions,ia,ib=np.intersect1d(a_positions,b_positions,return_indices=True)
    count=len(positions)
    if count<2:return dict(common_rows=count,spearman=None,top256_overlap_fraction=None)
    a=np.sqrt(a_values[ia,0]/np.maximum(a_values[ia,1],1e-12))
    b=np.sqrt(b_values[ib,0]/np.maximum(b_values[ib,1],1e-12))
    def rank(v):
        order=np.argsort(v,kind='stable');sorted_values=v[order]
        starts=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1]
        ends=np.r_[starts[1:],len(v)];result=np.empty(len(v),dtype=np.float64)
        result[order]=np.repeat((starts+ends-1)/2,ends-starts)
        return result
    ra,rb=rank(a),rank(b);budget=min(256,count)
    rho=float(np.corrcoef(ra,rb)[0,1]) if ra.std() and rb.std() else None
    a_top=set(np.lexsort((positions,-a))[:budget].tolist());b_top=set(np.lexsort((positions,-b))[:budget].tolist())
    return dict(common_rows=count,spearman=rho,top256_overlap_fraction=len(a_top&b_top)/budget,top_budget=budget)


def audit_observer_report(report,row,phase,preflight):
    sealed(report);spans=documents(row);total=row['input_tokens']
    _fields(report,dict(schema='early-token-diagnostics-v1',complete=True,phase=phase['name'],
        input_ids_sha256=row['input_ids_sha256'],mode='native',arm='recomputed',included_in_formal_timing=False,
        representation='swa-postnorm-prerope-bf16',layers=list(DIAGNOSTIC_LAYERS),
        metrics=['error_l2_squared','reference_l2_squared','cosine_distance'],
        full_kv_vectors_exported=False,attention_weights_collected=False,only_actual_native_surviving_rows=True,
        offline_bank_mutated=False,bin_edges=list(DIAGNOSTIC_EDGES),native_position_coverage_checked=True,
        prompt_chunk_tokens=8192,native_yoco_keep_tokens=140),'independent diagnostic report scope')
    records=report.get('records')
    require(type(records) is list and len(records)==72,'all9layers x8documents diagnostic records required')
    by_key={};compact=[]
    for record in records:
        layer=record.get('layer');ordinal=record.get('document_ordinal');key=layer,ordinal
        require(type(layer) is int and layer in DIAGNOSTIC_LAYERS and type(ordinal) is int and 0<=ordinal<8
                and key not in by_key,'diagnostic layer/document duplicate or foreign')
        lo,hi,doc_key=spans[ordinal]
        expected=[p-lo for start in range(0,total,8192)
            for p in head.native_positions_reference(start,min(start+8192,total),total,layer) if lo<=p<hi]
        _fields(record,dict(document_sha256=doc_key,document_start=lo,document_end=hi,
            native_yoco_rectangle_lo=None if layer<20 else max(0,total-140)),'diagnostic original document binding')
        positions,values=unpack_diagnostic_scalars(record['scalars'],expected)
        compare_statistics(record.get('summary'),diagnostic_summary(positions,values))
        compare_statistics(record.get('native_row_groups'),diagnostic_row_groups(positions,values,lo,layer,total),
                           'diagnostic native placeholder/final-rectangle groups')
        by_key[key]=(positions,values)
        compact.append({**{k:v for k,v in record.items() if k!='scalars'},
            'scalar_evidence':{k:v for k,v in record['scalars'].items() if k!='data'}})
    expected_correlations=[dict(early_layer=early,late_layer=late,document_ordinal=ordinal,
        **diagnostic_rank_comparison(*by_key[early,ordinal],*by_key[late,ordinal]))
        for early in (1,2,3) for late in (7,13,19,20,39) for ordinal in range(8)]
    compare_statistics(report.get('early_late_rank_comparisons'),expected_correlations,'early-late rank evidence')
    audit_question_routing(report,row,preflight)
    return dict(schema=report['schema'],sha256=report['sha256'],original_observer_sha256=report['sha256'],records=compact,
        early_late_rank_comparisons=report['early_late_rank_comparisons'],question_routing_events=report['question_routing_events'],
        question_routing_layers=report['question_routing_layers'],
        question_routing_compress_ratios=report['question_routing_compress_ratios'],
        question_routing_selection_kinds=report['question_routing_selection_kinds'],
        all_scalar_hashes_and_statistics_recomputed=True,actual_native_survivor_positions_verified=True,
        scalar_payloads_validated=True,summary_recomputed_from_lossless_scalars=True,
        scalar_payloads_retained_in_original_phase_file=True,limitations=report.get('limitations'))


def audit_question_routing(report,row,preflight):
    pin=preflight['protocol']['diagnostics'];identity=preflight['identity']
    require(identity.get('checkpoint_config_sha256')==pin['checkpoint_config_sha256']
        and all(identity.get('runtime_source_files_sha256',{}).get(name)==sha
                for name,sha in pin['native_source_files_sha256'].items()),
        'diagnostic native prefix topology source/config binding differs')
    layers=pin['native_prefix_routing_layers'];ratios=pin['native_prefix_routing_compress_ratios']
    kinds=pin['native_prefix_routing_selection_kinds']
    _fields(report,dict(question_routing_layers=layers,question_routing_compress_ratios=ratios,
        question_routing_selection_kinds=kinds),'actual prefix routing topology')
    start_suffix=max(end for _,end,_ in documents(row));total=row['input_tokens']
    expected={(start,min(start+8192,total),layer) for start in range(0,total,8192)
              if min(start+8192,total)>start_suffix for layer in layers}
    events=_event_map(report.get('question_routing_events'),expected,'question-suffix native routing')
    for (start,end,layer),event in events.items():
        queries=list(range(max(start,start_suffix),end));ratio=ratios[str(layer)]
        ends=[(q+1)//ratio for q in queries]
        _fields(event,dict(query_absolute_positions=queries,question_suffix_start=start_suffix,
            kv_compress_ratio=ratio,selection_kind=kinds[str(layer)],
            coordinate_system='loaded_compressed_cache_rows_single_request',invalid_index=-1,
            token_interval='[index*ratio,(index+1)*ratio)',context_parallel_size=1,num_reqs=1,
            native_compressed_key_starts=[0]*len(queries),native_compressed_key_ends=ends,
            interpretation='native_compressed_block_routing_proxy_not_attention_weights',
            includes_question_suffix_template_tokens=True),'native compressed index absolute coordinate binding')
        matrix=event.get('native_global_indices')
        require(type(matrix) is list and len(matrix)==len(queries)
            and all(type(values) is list and len(values)==pin['native_index_topk']
                and all(type(j) is int and (j==-1 or 0<=j<limit) for j in values)
                for values,limit in zip(matrix,ends)),'native routing indices include future/out-of-range/foreign-width rows')
    return True


def binding(phase,preflight):
    audit_native_swa_configuration(preflight)
    _fields(phase,dict(manifest_sha256=preflight['manifest_sha256'],
        headtail_scale20_protocol_sha256=preflight['protocol']['sha256'],
        execution_configuration_sha256=preflight['identity']['execution_configuration_sha256'],
        native_yoco_swa_share_hidden=False,**phase_policy(phase.get('arm'))),'three-arm phase binding')
    require(phase.get('controller',{}).get('native_yoco_swa_share_hidden') is False,
            'controller native input configuration must remain False')


def bank_scope_sha256(spans,identity):
    records=[]
    for _,_,key in spans:
        for layer in CAPTURE_LAYERS:
            scope=dict(model_sha256=identity['checkpoint_index_sha256'],
                tokenizer_sha256=identity['tokenizer_json_sha256'],
                execution_sha256=identity['execution_configuration_sha256'],document_sha256=key,
                layer=layer,representation='swa-postnorm-prerope-bf16')
            records.append(dict(document=key,layer=layer,scope=scope))
    return digest(sorted(records,key=lambda item:(item['document'],item['layer'])))


def comparison_scope(control,phase,*,expected_scope=None):
    arm=phase['arm'];formal=arm in ARMS
    _fields(control,dict(comparison_schema='headtail-three-arm-comparison-v1',comparison_arm=arm,
        single_installed_hook_owner=True,independent_policy_qualification=True,
        qualification_policy=arm if phase['mode'] in ('reference_sparse','sensitive') else None,
        cache_storage_policy='shared_gpu_full_canonical_bank_all_arms',gpu_bank_constant_across_arms=True,
        offline_gpu_resident=True,offline_bank_representation='swa-postnorm-prerope-bf16',
        offline_bank_immutable=True,cross_phase_overlay_reuse=False,phase_overlay_fresh=True,
        cached_document_kv_consumed=phase['mode'] in ('reference_sparse','sensitive'),
        cold_cache_ttft_claim=False,historical_static_cpu_staging_comparable=False), 'single-owner comparison scope')
    require(type(control.get('phase_generation')) is int and control['phase_generation']>0,
            'positive unique phase generation required')
    validate_digest(control.get('offline_bank_scope_sha256'),'full GPU bank scope seal')
    if formal:
        _fields(control,dict(offline_bank_entry_count=320,offline_bank_rows=320*DOCUMENT,
            offline_bank_bytes=320*DOCUMENT*WIDTH*2),'equal full GPU bank memory load')
        require(type(control.get('bank_generation')) is int and control['bank_generation']>0,
                'positive case bank generation required')
    if expected_scope is not None:
        require(control['offline_bank_scope_sha256']==expected_scope,'full bank document/model/layer scope differs')
    return {key:control[key] for key in ('bank_generation','offline_bank_scope_sha256',
        'offline_bank_entry_count','offline_bank_rows','offline_bank_bytes')}


def static_memory_plan(positions,start,end,total,spans,available):
    """Independent static-tail oracle: sorted absolute sources and bisect causality."""
    ids,non_doc,tail_ids=common.mask_reference(positions,start,end,spans)
    cutoff=positions[-1]
    tails={p:(i,p-lo,p-(hi-128)) for i,(lo,hi,_) in enumerate(spans)
           for p in range(hi-128,min(hi,cutoff+1))}
    memory=sorted(set(tails)|{p for p in available if p<=cutoff})
    owner={p:i for i,p in enumerate(positions)};selected_set=set(ids)
    lengths=[bisect_right(memory,q) for q in positions]
    plan=dict(schema='kce-tail-link-plan-v1',full_input_tokens=total,chunk_start=start,chunk_end=end,
        tail_tokens=128,documents=[dict(occurrence=i,key=key,start=lo,end=hi,tail_start=hi-128,
            tail_end=hi,tail_rows=128) for i,(lo,hi,key) in enumerate(spans)],
        current_positions=positions,selected_local_rows=ids,selected_positions=[positions[i] for i in ids],
        document_tail_local_rows=tail_ids,mandatory_nondoc_local_rows=non_doc,
        inactive_local_rows=[i for i in range(len(positions)) if i not in selected_set],
        selected_rows=len(ids),full_rows=len(positions),memory_positions=memory,memory_count=len(memory),
        memory_kinds=['document_tail' if p in tails else 'online_nondocument' for p in memory],
        memory_doc_indices=[tails[p][0] if p in tails else -1 for p in memory],
        memory_doc_token_offsets=[tails[p][1] if p in tails else -1 for p in memory],
        memory_tail_bank_rows=[tails[p][2] if p in tails else -1 for p in memory],
        memory_current_local_rows=[owner.get(p,-1) for p in memory],
        selected_memory_lengths=[lengths[i] for i in ids],reference_memory_lengths=lengths,
        available_nondoc_positions=sorted(available),nondocument_availability='explicit_generated_only',
        compact_slots_are_absolute_positions=False,global_index_offset=len(memory),
        native_yoco_surviving_positions_preserved=True)
    return dict(plan,plan_sha256=digest(plan))


def audit_static_restores(control,expected,spans,qualified):
    tail_sha=digest([p for _,hi,_ in spans for p in range(hi-128,hi)])
    events=control.get('gpu_tail_restore_events')
    require(type(events) is list and len(events)==len(FFN_LAYERS)
        and {e.get('layer') for e in events}==set(FFN_LAYERS),'all static GPU tail restores required')
    names=['gpu_full_bank_tail_slice','absolute_rope_nativecast']
    for event in events:
        _fields(event,dict(rows=1024,source='shared_gpu_full_canonical_bank',source_rows_per_document=DOCUMENT,
            source_local_start=DOCUMENT-128,checked=qualified,checks=2 if qualified else 0,
            checks_named=names if qualified else [],absolute_positions_sha256=tail_sha,
            phase_generation=control['phase_generation'],host_to_device_copy=False),'static GPU full-bank restore')
        require((type(event.get('max_absolute_error')) in (int,float) and event['max_absolute_error']==0)
                if qualified else event.get('max_absolute_error') is None,'GPU tail restore must be exact zero')
    proofs=control.get('tail_restore_identity_events')
    assembly=control.get('memory_assembly_identity_events')
    if not qualified:
        require(proofs==[] and assembly==[],'formal static phase contains qualification-only checks')
        return
    require(type(proofs) is list and len(proofs)==len(FFN_LAYERS)
        and {e.get('layer') for e in proofs}==set(FFN_LAYERS),'static split restore proof coverage missing')
    for event in proofs:
        _fields(event,dict(rows=1024,checks=2,checks_named=names,absolute_positions_sha256=tail_sha),
                'independent GPU tail slice and absolute-RoPE qualification')
        require(type(event.get('max_absolute_error')) in (int,float) and event['max_absolute_error']==0,
                'static qualification numeric mismatch')
    assemblies=_event_map(assembly,expected,'static memory assembly')
    memories=_event_map(control.get('tail_memory_events'),expected,'static causal memory')
    for key,event in assemblies.items():
        _fields(event,dict(schema='tail-link-memory-assembly-identity-v1',checks=1,
            rows=memories[key]['memory_count']),'static assembled KV identity')
        require(type(event.get('max_absolute_error')) in (int,float) and event['max_absolute_error']==0,
                'static assembled KV is not byte-exact')


def audit_static_controller(control,row,phase,spans):
    _fields(control,dict(schema='tail-link-gpu-bank-controller-v1',mode=phase['mode'],phase=phase['name'],
        input_ids_sha256=row['input_ids_sha256'],full_input_tokens=row['input_tokens'],ffn_layers=list(FFN_LAYERS),
        first_three_full=True,actual_full_row_native_mhc=True,yoco_layout_preserved=True,
        formal_admission_unchanged=True,decode_native=True,global_kv_native_online=True,
        not_model_flops_fraction=True,prefill_only_link=True,selected_tokens_forced_through_yoco=False,
        native_yoco_swa_share_hidden=False,head_tokens=0,tail_tokens=128,dynamic_document_swa=False,
        dynamic_refresh_start_layer=None,dynamic_swa_events=[],staging=None,
        static_tail_source='shared_gpu_full_bank_tail128_slice',static_swa_projection_reused=True,
        native_local_window_preserved=False,native_local_mask_preserved=False,
        full_bank_retained_but_static_attention_tail_only=True,tail_bank_source_shape=[DOCUMENT,WIDTH]),
        'new static GPU-tail policy scope')
    require(phase['mode'] in ('reference_sparse','sensitive') and phase.get('fraction') is None,
            'unknown static comparison mode')
    adapter=control.get('adapter',{})
    _fields(adapter,dict(schema='kv-cache-engram-index-hit20-adapter-v1',decoder_forward_unchanged=True,
        attention_forward_unchanged=True,native_mhc_unchanged=True,native_yoco_layout_unchanged=True,
        all_selected_delegates_native=True,empty_selection_skips_ep_collective=False,
        cached_hidden_or_ffn_output=False,accuracy_guarantee=False,speedup_claim=False),'static native FFN adapter')
    chunks=[(s,min(s+8192,row['input_tokens'])) for s in range(0,row['input_tokens'],8192)]
    expected={(s,e,l) for s,e in chunks for l in FFN_LAYERS}
    collections={name:_event_map(control.get(name),expected,name) for name in
        ('selection','ffn_layouts','swa_events','attention_events','tail_memory_events')}
    ffns=_event_map(adapter.get('events'),expected,'static actual FFN')
    dense=phase['mode']=='reference_sparse';qualified=dense or phase.get('verify_sparse') is True
    audit_static_restores(control,expected,spans,qualified)
    totals=dict(ffn_full_rows=0,ffn_selected_rows=0,ffn_actual_computed_rows=0,attention_full_rows=0,
        attention_selected_rows=0,attention_actual_computed_rows=0,swa_online_rows=0,swa_reused_rows=0,
        consumed_cached_pairs=0,consumed_online_nondoc_pairs=0,materialized_offline_rows=0,
        dynamic_refreshed_rows=0,actual_partial_chunks=0,encoder_rows=0,encoder_selected_rows=0,
        decoder_rows=0,decoder_selected_rows=0,main_q_a_rows=0,index_q_rows=0,global_compute_calls=0)
    available={l:set() for l in FFN_LAYERS};templates={}
    for start,end,layer in sorted(expected):
        key=start,end,layer;layout_key=start,end,layer>=20
        if layout_key not in templates:
            positions=head.native_positions_reference(start,end,row['input_tokens'],layer)
            _,non,_=common.mask_reference(positions,start,end,spans)
            available[layer].update(positions[i] for i in non)
            templates[layout_key]=static_memory_plan(positions,start,end,row['input_tokens'],spans,available[layer])
        plan=templates[layout_key];positions=plan['current_positions'];ids=plan['selected_local_rows']
        non=plan['mandatory_nondoc_local_rows'];tails=plan['document_tail_local_rows']
        available[layer].update(positions[i] for i in non)
        require(sorted(available[layer])==plan['available_nondoc_positions'],'static cross-layer nondoc availability differs')
        m=collections['selection'][key];n,k=len(positions),len(ids);actual=n if dense else k
        _fields(m,dict(current_positions=positions,selected_local_rows=ids,selected_positions=[positions[i] for i in ids],
            full_rows=n,selected_rows=k,plan_sha256=plan['plan_sha256']),'static survivor selection')
        _fields(collections['ffn_layouts'][key],{field:m[field] for field in
            ('current_positions','selected_local_rows','full_rows','selected_rows','plan_sha256')},'static FFN selection')
        _fields(ffns[key],dict(full_rows=n,selected_rows=k,actual_mlp_rows=actual,reference_dense=dense,
            plan_sha256=plan['plan_sha256'],request_id=phase['record']['request_id'],phase=phase['name'],
            inactive_ffn_output='zero',empty_native_collective=k==0 and n>0),'static executed FFN')
        local_index=layer in (8,14,20,24,28,32,36)
        ae=collections['attention_events'][key]
        _fields(ae,dict(full_rows=n,selected_rows=k,main_q_a_rows=n if local_index or dense else k,
            index_q_rows=n if local_index else 0,main_q_b_rows=actual,joint_attention_rows=actual,
            output_projection_rows=actual,global_compute_called=True,native_joint_attention=True,
            reference_dense=dense,is_global_source=layer in (8,14,20),plan_sha256=plan['plan_sha256']),
            'static native global/Q/attention path')
        require(type(ae.get('global_kv_rows')) is int and ae['global_kv_rows']>=0,'static global geometry absent')
        _fields(collections['swa_events'][key],dict(current_positions=positions,full_rows=n,document_online_rows=0,
            document_tail_rows=len(tails),non_document_online_rows=len(non),reused_rows=len(tails),
            online_local_rows=non,online_rows=len(non),compact=layer>=20),'static SWA reused row counters')
        memory=collections['tail_memory_events'][key]
        _fields(memory,plan,'independent static causal memory')
        lengths=plan['reference_memory_lengths'] if dense else plan['selected_memory_lengths']
        _fields(memory,dict(query_positions=positions if dense else [positions[i] for i in ids],
            query_memory_lengths=lengths,actual_joint_query_rows=actual,gpu_causal_oracle_checked=qualified),
            'static actual causal queries')
        prefix=[0]
        for kind in plan['memory_kinds']:prefix.append(prefix[-1]+(kind=='document_tail'))
        cached_pairs=sum(prefix[length] for length in lengths)
        for field,value in dict(ffn_full_rows=n,ffn_selected_rows=k,ffn_actual_computed_rows=actual,
            attention_full_rows=n,attention_selected_rows=k,attention_actual_computed_rows=actual,
            swa_online_rows=len(non),swa_reused_rows=len(tails),consumed_cached_pairs=cached_pairs,
            consumed_online_nondoc_pairs=sum(lengths)-cached_pairs,materialized_offline_rows=prefix[-1],
            actual_partial_chunks=int(0<k<n),main_q_a_rows=ae['main_q_a_rows'],index_q_rows=ae['index_q_rows'],
            global_compute_calls=1).items():totals[field]+=value
        category='encoder' if layer<20 else 'decoder';totals[category+'_rows']+=n;totals[category+'_selected_rows']+=k
    _fields(adapter.get('counts',{}),dict(full_rows=totals['ffn_full_rows'],selected_rows=totals['ffn_selected_rows'],
        actual_mlp_rows=totals['ffn_actual_computed_rows'],inactive_ffn_zero_rows=totals['ffn_full_rows']-totals['ffn_selected_rows']),
        'static FFN aggregate')
    audit_decode_ring(control,row)
    totals.update(ffn_bypassed_rows=totals['ffn_full_rows']-totals['ffn_selected_rows'],
        attention_bypassed_rows=totals['attention_full_rows']-totals['attention_selected_rows'],
        phase_generation=control['phase_generation'],offline_kv_h2d_bytes=0,dynamic_refresh_start_layer=None,
        first_three_layers_excluded=True,not_model_flops_fraction=True,actual_rows_instrumented=True)
    return totals


def audit_decode_ring(control,row):
    events=control.get('decode_ring_positions')
    require(type(events) is list and len(events)==len(FFN_LAYERS)
        and {e.get('layer') for e in events}==set(FFN_LAYERS),'all physical decode rings required')
    for event in events:
        _fields(event,dict(positions=list(range(row['input_tokens']-128,row['input_tokens'])),
            physical_readback_exact=True,checks=128),'physical last128 decode ring')
        slots=event.get('physical_slots')
        require(type(slots) is list and len(slots)==len(set(slots))==128
            and all(type(i) is int and i>=0 for i in slots),'physical decode slot collision')


def audit_controller(control,row,phase,spans):
    comparison_scope(control,phase)
    _fields(control,dict(diagnostic=phase.get('diagnostic',False)),'diagnostic hook eligibility')
    if phase.get('diagnostic') is not True:
        require(control.get('early_token_diagnostics') is None,'formal request executed diagnostic observer')
    require(phase.get('native_yoco_swa_share_hidden') is False,'unverified native input representation')
    if phase['arm']=='headtail128_link':
        return dict(head.audit_controller(control,row,phase,spans),offline_kv_h2d_bytes=0,actual_rows_instrumented=True)
    if phase['arm']=='tail128_link':return audit_static_controller(control,row,phase,spans)
    require(phase['arm']=='recomputed' and phase['mode']=='native' and phase.get('fraction') is None,
            'unknown three-arm policy')
    _fields(control,dict(schema='headtail-link-controller-v1',mode='native',phase=phase['name'],
        input_ids_sha256=row['input_ids_sha256'],full_input_tokens=row['input_tokens'],
        head_tokens=0,tail_tokens=0,dynamic_refresh_start_layer=None,dynamic_document_swa=False,
        native_yoco_swa_share_hidden=False,staging=None,headtail_swa_events=[],headtail_memory_events=[],
        union_oracle_events=[],boundary_restore_identity_events=[],phase_local_boundary_overlay=False,
        boundary_overlay_rows_per_layer=0,decode_ring_positions=[]),'unchanged native baseline')
    audit_native_no_cache(control)
    _fields(control.get('adapter',{}).get('counts',{}),dict(full_rows=0,selected_rows=0,actual_mlp_rows=0),
            'native baseline cannot claim sparse intervention')
    native_rows=sum(len(head.native_positions_reference(s,min(s+8192,row['input_tokens']),row['input_tokens'],l))
                    for s in range(0,row['input_tokens'],8192) for l in FFN_LAYERS)
    return dict(ffn_full_rows=native_rows,ffn_selected_rows=native_rows,ffn_bypassed_rows=0,
        attention_full_rows=native_rows,attention_selected_rows=native_rows,attention_bypassed_rows=0,
        dynamic_refreshed_rows=0,consumed_cached_pairs=0,actual_partial_chunks=0,offline_kv_h2d_bytes=0,
        phase_generation=control['phase_generation'],first_three_layers_excluded=True,not_model_flops_fraction=True,
        actual_rows_instrumented=False,counts_basis='independent native surviving-layout count; native operators uninstrumented')


def sparse_identity(reference,candidate,row,qualification_sha256,*,controller_results=None):
    validate_digest(qualification_sha256,'fresh native qualification')
    arm=reference.get('arm');require(arm in POLICY_ARMS and candidate.get('arm')==arm,'independent policy arm mismatch')
    banks=[]
    for phase,mode in ((reference,'reference_sparse'),(candidate,'sensitive')):
        sealed(phase)
        _fields(phase,dict(schema=PHASE_SCHEMA,mode=mode,arm=arm,index=row['index'],fraction=None,
            input_ids_sha256=row['input_ids_sha256'],timing_valid=False,quality_eligible=False,diagnostic=False,
            dynamic_document_swa=arm=='headtail128_link',native_yoco_swa_share_hidden=False,**phase_policy(arm)),
            'same-policy qualification identity')
        control=phase['controller'];record=phase['record']
        require(control.get('same_policy_causal_oracle_checked') is True,'same-policy raw causal oracle missing')
        result=audit_controller(control,row,phase,documents(row))
        if controller_results is not None:controller_results[phase['sha256']]=result
        banks.append(comparison_scope(control,phase))
        _fields(record,dict(success=True,rank=0,input_ids_sha256=row['input_ids_sha256'],
            input_tokens=row['input_tokens'],natural_EOS=False),'sparse one-token request')
        ids=record.get('output_ids');require(type(ids) is list and len(ids)==1 and type(ids[0]) is int
            and ids[0]>=0 and record.get('output_ids_sha256')==digest(ids),'sparse output hash differs')
        chunks=[(s,min(s+8192,row['input_tokens'])) for s in range(0,row['input_tokens'],8192)]
        expected={(s,e,l) for s,e in chunks for l in range(40)}
        values=control.get('sparse_identity_fingerprints')
        require(type(values) is list and len(values)==len(expected)
            and control.get('sparse_identity_check_count')==len(expected)
            and control.get('sparse_identity_fingerprints_sha256')==digest(values),'full40-layer sparse fingerprints missing')
        seen=set()
        for item in values:
            key=tuple(item.get('key',[]));value=item.get('value',{})
            require(key in expected and key not in seen and set(value)=={'output','lazy'}
                and set(value['lazy'])=={'residual','post_layer_mix','comb_res_mix','pre_layer_mix'},
                'sparse decoder/four mHC fields incomplete')
            seen.add(key);require(value['output'] is not None,'sparse layer output missing')
            _fingerprint(value['output'])
            for value in value['lazy'].values():_fingerprint(value)
    for key in ('manifest_sha256','execution_configuration_sha256','headtail_scale20_protocol_sha256','empty_ep_qualification_sha256'):
        require(reference.get(key)==candidate.get(key),'same-policy reference execution differs')
    require(banks[0]==banks[1] and reference['controller']['phase_generation']!=candidate['controller']['phase_generation'],
            'reference/candidate shared bank or fresh phase generation differs')
    require(candidate.get('verify_sparse') is True and candidate['controller'].get('sparse_identity_exact') is True
        and reference['record']['output_ids']==candidate['record']['output_ids']
        and reference['controller']['sparse_identity_fingerprints']==candidate['controller']['sparse_identity_fingerprints'],
        'same-policy dense/sparse must be exactly equal')
    fields=('start','end','layer','current_positions','selected_local_rows')
    require([{k:e[k] for k in fields} for e in reference['controller']['selection']]
        ==[{k:e[k] for k in fields} for e in candidate['controller']['selection']],'sparse/reference selection differs')
    body=dict(schema='headtail-scale20-sparse-qualification-v1',passed=True,arm=arm,index=row['index'],
        input_tokens=row['input_tokens'],input_ids_sha256=row['input_ids_sha256'],qualification_sha256=qualification_sha256,
        reference_sha256=reference['sha256'],candidate_sha256=candidate['sha256'],
        identity_check_count=candidate['controller']['sparse_identity_check_count'],
        identity_fingerprints_sha256=candidate['controller']['sparse_identity_fingerprints_sha256'],
        quality_eligible=False,timing_valid=False,equivalence_to_full_model_claim=False,exact_same_policy_reference=True)
    return dict(body,sha256=digest(body))


def audit_phase(phase,row,preflight,*,arm,timing,quality,fixed_tokens=32,diagnostic=False,controller_result=None):
    sealed(phase);binding(phase,preflight)
    require(arm in ARMS,'unknown fresh comparison arm')
    smoke=not timing and not quality and fixed_tokens==1 and not diagnostic
    _fields(phase,dict(schema=PHASE_SCHEMA,index=row['index'],dataset=row['dataset'],arm=arm,
        mode='native' if arm=='recomputed' else 'sensitive',fraction=None,
        dynamic_document_swa=arm=='headtail128_link',timing_valid=timing,quality_eligible=quality,
        input_ids_sha256=row['input_ids_sha256'],diagnostic=diagnostic,
        online_full_bank_gpu_resident=True,online_clock_started_before_reset=True,phase_local_memory_rebuilt=True,
        offline_build_included=False,native_global_kv_and_index_preserved=True,partial_smoke=smoke,
        verify_sparse=smoke),'fresh three-arm phase scope')
    require(not diagnostic or arm=='recomputed' and not timing and not quality and fixed_tokens==1,
            'diagnostic cannot enter quality or formal timing')
    record=phase['record']
    _fields(record,dict(success=True,rank=0,input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],
        phase=phase['name'],sample_id=row['sample_id'],dataset=row['dataset'],
        request_id=f"headtail-scale20:{row['index']}:{phase['name']}:{row['sample_id']}"),'complete request scope')
    ids=record.get('output_ids')
    require(type(ids) is list and bool(ids) and all(type(i) is int and i>=0 for i in ids)
        and record.get('output_ids_sha256')==digest(ids),'raw output token binding differs')
    if quality:
        require(record.get('natural_EOS') is True and len(ids)<=64 and record.get('finish_reason') in ('length','stop')
            and record.get('truncated') is (record['finish_reason']=='length'),'quality EOS64 cap differs')
        expected=score(record.get('text'),row['references'])
        require(all(type(record.get(k)) in (int,float) and math.isfinite(record[k]) and abs(record[k]-v)<=1e-12
            for k,v in expected.items()),'fixed-gold score not reproducible')
    else:
        require(type(fixed_tokens) is int and fixed_tokens in (1,32) and (not timing or fixed_tokens==32)
            and len(ids)==fixed_tokens and record.get('natural_EOS') is False
            and record.get('finish_reason')=='length' and record.get('truncated') is False,'fixed output length/EOS differs')
    ttft=number(record.get('online_ttft_s'),'online TTFT',True)
    elapsed=number(record.get('online_elapsed_s'),'online elapsed',True)
    require(number(record.get('online_prepare_to_submit_s'),'prepare')<=ttft<=elapsed
        <=number(phase.get('online_global_window_s'),'common online wall',True),'phase timing ordering differs')
    comparison_scope(phase['controller'],phase,expected_scope=bank_scope_sha256(documents(row),preflight['identity']))
    return audit_controller(phase['controller'],row,phase,documents(row)) if controller_result is None else controller_result


def partial_smoke(directory,preflight,qual,evidence,arm):
    require(arm in POLICY_ARMS,'unknown sparse qualification policy')
    row=next(r for r in preflight['audit_cases'] if r['index']==2)
    closure=read(directory/f'{arm}.partial_smoke.json',evidence)
    _fields(closure,dict(schema='headtail-scale20-partial-smoke-v1',passed=True,arm=arm,index=2,
        qualification_sha256=qual['sha256'],input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],
        quality_eligible=False,timing_valid=False,accuracy_guarantee=False,full_input_original_eight_documents=True,
        independent_swa_and_ffn_paths_exercised=True,output_tokens_per_arm=1),'policy-specific full-input smoke')
    require(set(closure.get('phase_sha256',{}))=={arm},'smoke has another policy arm')
    reference=read(directory/f'case2.{arm}_sparse_reference.json',evidence)
    candidate=read(directory/f'case2.{arm}_partial_smoke.json',evidence)
    sparse=read(directory/f'{arm}.sparse_qualification.json',evidence)
    for phase,name in ((reference,f'case2.{arm}_sparse_reference'),(candidate,f'case2.{arm}_partial_smoke')):
        binding(phase,preflight);empty_ep_binding(phase,qual)
        require(phase.get('name')==name and phase.get('repeat') is None,'sparse filename/repeat differs')
        comparison_scope(phase['controller'],phase,expected_scope=bank_scope_sha256(documents(row),preflight['identity']))
    require(reference.get('partial_smoke') is False and closure['phase_sha256'][arm]==candidate['sha256'],
            'smoke/reference closure swapped')
    results={};expected=sparse_identity(reference,candidate,row,qual['sha256'],controller_results=results)
    require(sparse==expected and closure.get('sparse_qualification_sha256')==sparse['sha256'],
            'policy-specific exact numerical closure differs')
    actual=audit_phase(candidate,row,preflight,arm=arm,timing=False,quality=False,fixed_tokens=1,
        controller_result=results[candidate['sha256']])
    require(actual['ffn_bypassed_rows']>0 and actual['actual_partial_chunks']>0
        and actual['consumed_cached_pairs']>0,'smoke omitted actual partial/cache paths')
    if arm=='headtail128_link':require(actual['dynamic_refreshed_rows']>0,'smoke omitted dynamic boundary refresh')
    return closure,dict(passed=True,arm=arm,index=2,input_tokens=row['input_tokens'],actual_rows=actual,
        closure_sha256=closure['sha256'],quality_eligible=False,timing_valid=False,fixed_output_tokens=1,
        same_policy_sparse_byteexact=True,sparse_qualification_sha256=sparse['sha256'],
        shared_bank_identity=comparison_scope(candidate['controller'],candidate),
        first_three_reference_fingerprints_sha256=digest([e for e in reference['controller']['sparse_identity_fingerprints']
                                                        if e['key'][2]<3]),equivalence_to_full_model_claim=False)


def expected_timing_summary(samples):
    require(set(samples)==set(ARMS) and all(type(v) is list and len(v)==3 for v in samples.values()),
            'three fresh samples for every arm required')
    arms={}
    for arm,values in samples.items():
        for value in values:number(value,'formal TTFT',True)
        arms[arm]=dict(online_ttft_samples_s=values,online_ttft_p50_s=statistics.median(values),
                      repetitions=3,fixed_output_tokens=32)
    comparisons={arm:dict(ttft_speedup=arms['recomputed']['online_ttft_p50_s']/arms[arm]['online_ttft_p50_s'],
        paired_repeat_speedups=[samples['recomputed'][repeat]/samples[arm][repeat] for repeat in range(3)])
        for arm in POLICY_ARMS}
    return dict(arms=arms,comparisons=comparisons,concurrency=1,active_dp_ranks=1,participating_ep_ranks=8,
        online_full_bank_gpu_resident=True,online_gather_scatter_and_fallback_included=True,
        model_load_included=False,offline_build_included=False,saturated_capacity_claim=False,
        fresh_paired_baseline=True,historical_results_used_for_admission=False)


def audit_offline_phase(phase,row,ordinal,key,preflight,qual):
    empty_ep_binding(phase,qual);binding(phase,preflight);comparison_scope(phase['controller'],phase)
    name=f"case{row['index']}.offline_doc{ordinal}"
    _fields(phase,dict(schema=PHASE_SCHEMA,name=name,index=None,dataset=row['dataset'],arm='offline',
        mode='capture_independent',input_ids_sha256=key,repeat=None,fraction=None,diagnostic=False,
        timing_valid=False,quality_eligible=False,partial_smoke=False,verify_sparse=False,
        dynamic_document_swa=False,online_full_bank_gpu_resident=False,online_clock_started_before_reset=True,
        phase_local_memory_rebuilt=True,offline_build_included=False,native_global_kv_and_index_preserved=True),
        'offline source document phase')
    record=phase['record']
    _fields(record,dict(success=True,rank=0,input_ids_sha256=key,input_tokens=DOCUMENT,phase=name,
        sample_id=key,dataset=row['dataset'],request_id=f'headtail-scale20:doc:{name}:{key}',
        natural_EOS=False,finish_reason='length',truncated=False),'offline native document request')
    ids=record.get('output_ids')
    require(type(ids) is list and len(ids)==1 and type(ids[0]) is int and ids[0]>=0
        and record.get('output_ids_sha256')==digest(ids),'offline fixed output differs')
    control=phase['controller']
    _fields(control,dict(mode='capture_independent',phase=name,input_ids_sha256=key,full_input_tokens=DOCUMENT,
        offline_yoco_full_layout=True,model_boundary_restored=True,dynamic_document_swa=False,diagnostic=False,
        early_token_diagnostics=None,headtail_swa_events=[],headtail_memory_events=[],union_oracle_events=[],
        boundary_restore_identity_events=[]),'unmodified independent offline capture')
    audit_native_no_cache(control)
    return control


def audit_diagnostic(phase,row,preflight,qual,first_token,bank,used_generations):
    empty_ep_binding(phase,qual)
    require(phase.get('name')==f"case{row['index']}.diagnostic_recomputed" and phase.get('repeat') is None,
            'diagnostic filename/repeat differs')
    audit_phase(phase,row,preflight,arm='recomputed',timing=False,quality=False,fixed_tokens=1,diagnostic=True)
    require(phase['record']['output_ids']==[first_token],'diagnostic observer altered native first output token')
    control=phase['controller']
    require(comparison_scope(control,phase)==bank,'diagnostic changed resident bank identity')
    require(control['phase_generation']>max(used_generations),'diagnostic did not follow all formal phase generations')
    report=control.get('early_token_diagnostics')
    require(type(report) is dict and bool(report),'actual early-token diagnostic evidence missing')
    observer=audit_observer_report(report,row,phase,preflight)
    return dict(phase_sha256=phase['sha256'],first_token_matches_native_quality=True,
        excluded_from_formal_ttft_and_quality=True,observer=observer)


def audit_case(directory,row,preflight,qual,smokes,evidence):
    index=row['index'];pair=read(directory/f'case{index}.paired.json',evidence)
    _fields(pair,dict(schema='headtail-scale20-case-paired-v1',qualification_sha256=qual['sha256'],
        partial_smoke_sha256={arm:smokes[arm]['sha256'] for arm in POLICY_ARMS},ffn_layers=list(FFN_LAYERS),
        offline_capture_layers=list(CAPTURE_LAYERS),all_negative_results_retained=True,
        exploratory_previously_exposed_inputs=True,calibration_claim=False,fresh_paired_baseline=True,
        diagnostic_excluded_from_formal_metrics=True),'fresh three-arm case closure')
    _fields(pair,{k:row[k] for k in ('index','dataset','sample_id','references','question','input_tokens','input_ids_sha256')},
            'paired input/gold identity')
    require(set(pair.get('quality',{}))==set(ARMS),'all three actual quality arms required')
    quality={};samples={arm:[] for arm in ARMS};masks={};generations=set();bank=None;first_token=None
    expected_scope=bank_scope_sha256(documents(row),preflight['identity'])
    def bind_phase(phase):
        nonlocal bank
        observed=comparison_scope(phase['controller'],phase,expected_scope=expected_scope)
        require(bank is None or bank==observed,'full resident GPU bank changed between arms/repeats')
        bank=observed
        generation=phase['controller']['phase_generation']
        require(generation not in generations,'formal phase reused a state/overlay generation')
        generations.add(generation)
    for arm in ARMS:
        filename=f'case{index}.{arm}_quality.json';phase=read(directory/filename,evidence)
        empty_ep_binding(phase,qual);verify_reference(pair['quality'][arm],directory/filename,phase,evidence)
        require(phase.get('name')==filename[:-5] and phase.get('repeat') is None,'quality filename identity changed')
        masks[arm]=audit_phase(phase,row,preflight,arm=arm,timing=False,quality=True);bind_phase(phase)
        record=phase['record']
        quality[arm]={k:record[k] for k in ('text','answer_em','answer_token_f1','truncated','finish_reason')}
        quality[arm]['generated_tokens']=len(record['output_ids'])
        if arm=='recomputed':first_token=record['output_ids'][0]
        filename=f'case{index}.{arm}_warmup.json';phase=read(directory/filename,evidence)
        empty_ep_binding(phase,qual)
        require(phase.get('name')==filename[:-5] and phase.get('repeat') is None,'warmup filename identity changed')
        audit_phase(phase,row,preflight,arm=arm,timing=False,quality=False);bind_phase(phase)
    schedule=timing_schedule(index);references=pair.get('timing_phases')
    require(type(references) is list and [(v.get('repeat'),v.get('arm')) for v in references]==schedule,
            'exact rotated three-arm three-repeat schedule required')
    for reference,(repeat,arm) in zip(references,schedule):
        filename=f'case{index}.{arm}_measure{repeat}.json';phase=read(directory/filename,evidence)
        empty_ep_binding(phase,qual);verify_reference(reference,directory/filename,phase,evidence,extras=('repeat','arm'))
        require(phase.get('name')==filename[:-5] and phase.get('repeat')==repeat,'formal filename/repeat differs')
        audit_phase(phase,row,preflight,arm=arm,timing=True,quality=False);bind_phase(phase)
        samples[arm].append(phase['record']['online_ttft_s'])
    require(len(generations)==15 and pair.get('shared_bank_identity')==bank,'15-phase shared GPU bank closure differs')
    expected_summary=expected_timing_summary(samples)
    require(pair.get('timing_summary')==expected_summary,'three-arm summary does not reproduce raw9 measurements')
    filename=f'case{index}.diagnostic_recomputed.json';phase=read(directory/filename,evidence)
    verify_reference(pair.get('diagnostic_recomputed'),directory/filename,phase,evidence)
    diagnostic=audit_diagnostic(phase,row,preflight,qual,first_token,bank,generations)
    diagnostic_elapsed=number(pair.get('diagnostic_elapsed_s'),'separate diagnostic elapsed',True)
    require(diagnostic_elapsed>=phase['record']['online_elapsed_s'],'diagnostic wall excludes its own request')
    builds=[]
    spans=documents(row)
    for ordinal,(_,_,key) in enumerate(spans):
        phase=read(directory/f'case{index}.offline_doc{ordinal}.json',evidence)
        control=audit_offline_phase(phase,row,ordinal,key,preflight,qual)
        require(control['bank_generation']==bank['bank_generation'],'offline/formal bank generation differs')
        _fields(control,dict(offline_bank_entry_count=(ordinal+1)*40,offline_bank_rows=(ordinal+1)*40*DOCUMENT,
            offline_bank_bytes=(ordinal+1)*40*DOCUMENT*WIDTH*2,
            offline_bank_scope_sha256=bank_scope_sha256(spans[:ordinal+1],preflight['identity'])),
            'progressive canonical GPU bank source scope')
        payloads=control.get('offline_payloads')
        require(type(payloads) is list and len(payloads)==len(CAPTURE_LAYERS),'40-layer full canonical payloads required')
        by_layer={item['scope']['layer']:item for item in payloads}
        require(set(by_layer)==set(CAPTURE_LAYERS),'offline layer missing or repeated')
        for layer,item in by_layer.items():
            filename=key+f'.layer{layer}.bf16'
            scope=dict(model_sha256=preflight['identity']['checkpoint_index_sha256'],
                tokenizer_sha256=preflight['identity']['tokenizer_json_sha256'],
                execution_sha256=preflight['identity']['execution_configuration_sha256'],
                document_sha256=key,layer=layer,representation='swa-postnorm-prerope-bf16')
            _fields(item,dict(scope=scope,file=filename,shape=[DOCUMENT,WIDTH],dtype='bfloat16',bytes=DOCUMENT*WIDTH*2),
                    'full canonical GPU bank raw payload')
            relative=f'case{index}.offline-kv/'+filename;path=directory/relative
            require(path.parent.is_dir() and not path.parent.is_symlink(),'payload directory symlink rejected')
            require(relative not in evidence,'full bank must be hashed exactly once per case')
            evidence[relative]=audit_bank_payload(path,item['sha256'])
        builds.append(phase['sha256'])
    require(pair.get('offline_phase_sha256')==builds,'all eight offline capture closures differ')
    return pair,dict(index=index,dataset=row['dataset'],references=row['references'],question=row['question'],
        input_tokens=row['input_tokens'],quality=quality,online_ttft_samples_s=samples,
        online_ttft_p50_s={arm:statistics.median(samples[arm]) for arm in ARMS},
        comparisons=expected_summary['comparisons'],actual_rows_and_swa_reuse=masks,shared_bank_identity=bank,
        diagnostic=diagnostic,diagnostic_elapsed_s=diagnostic_elapsed,
        offline_build_s=number(pair.get('offline_build_s'),'offline build'),closure_sha256=pair['sha256'],
        fresh_paired_baseline=True,historical_results_used_for_admission=False)


def summarize_cases(cases):
    if not cases:return None
    summaries={};rates={}
    for arm in ARMS:
        samples=[s for case in cases for s in case['online_ttft_samples_s'][arm]]
        summaries[arm]=dict(n=len(cases),online_ttft_samples_s=samples,online_ttft_p50_s=statistics.median(samples),
            em_percent=100*sum(c['quality'][arm]['answer_em'] for c in cases)/len(cases),
            f1_percent=100*sum(c['quality'][arm]['answer_token_f1'] for c in cases)/len(cases),
            truncated_count=sum(c['quality'][arm]['truncated'] for c in cases))
        values=[c['actual_rows_and_swa_reuse'][arm] for c in cases]
        totals={key:sum(v.get(key,0) for v in values) for key in values[0]
            if type(values[0][key]) is int and key not in ('phase_generation','dynamic_refresh_start_layer')}
        totals.update(ffn_selected_fraction=totals['ffn_selected_rows']/totals['ffn_full_rows'],
            attention_selected_fraction=totals['attention_selected_rows']/totals['attention_full_rows'],
            first_three_layers_excluded=True,not_model_flops_fraction=True,
            actual_rows_instrumented=arm!='recomputed')
        if arm=='headtail128_link':
            totals['dynamic_refreshed_boundary_fraction']=totals['dynamic_refreshed_rows']/totals['boundary_slot_capacity']
        rates[arm]=totals
    baseline=summaries['recomputed']['online_ttft_p50_s']
    return dict(n=len(cases),arms=summaries,actual_rates_by_arm=rates,
        comparisons={arm:dict(ttft_speedup=baseline/summaries[arm]['online_ttft_p50_s']) for arm in POLICY_ARMS},
        headtail_vs_static=dict(ttft_speedup=summaries['tail128_link']['online_ttft_p50_s']/summaries['headtail128_link']['online_ttft_p50_s']),
        diagnostic_cases=len(cases),diagnostic_excluded_from_formal_metrics=True,
        fresh_paired_baseline=True,historical_results_used_for_admission=False,statistical_significance_claim=False,
        static_arm_cache_media='shared full canonical GPU bank; historical CPU-tail DMA absent')


def validate_closed_run(run,preflight,qual,smokes,closures,errors,directory,evidence):
    sealed(run)
    require(not errors and qual is not None and set(smokes)==set(POLICY_ARMS)
        and len(closures)==len(CASE_INDICES) and run.get('schema')==RUN_SCHEMA and run.get('status')=='complete'
        and run.get('completed_case_indices')==list(CASE_INDICES) and [c.get('index') for c in closures]==list(CASE_INDICES)
        and run.get('identity')==preflight['identity'] and run.get('manifest_sha256')==preflight['manifest_sha256']
        and run.get('qualification')==qual and run.get('partial_smoke_sha256')=={arm:smokes[arm]['sha256'] for arm in POLICY_ARMS}
        and run.get('diagnostic_excluded_from_formal_metrics') is True
        and type(run.get('cases')) is list and len(run['cases'])==20
        and type(run.get('diagnostics')) is list and len(run['diagnostics'])==20,
        'final run does not close all20 fresh paired cases and separate diagnostics')
    generations=[]
    for reference,diag,closure in zip(run['cases'],run['diagnostics'],closures):
        index=closure['index']
        require(reference.get('index')==diag.get('index')==index,'final case/diagnostic order differs')
        verify_reference(reference,Path(directory)/f'case{index}.paired.json',closure,evidence,extras=('index',))
        expected=dict(index=index,**closure['diagnostic_recomputed'])
        require(diag==expected,'final diagnostic detached from case raw evidence')
        generations.append(closure['shared_bank_identity']['bank_generation'])
    require(len(set(generations))==20 and generations==sorted(generations),'case bank generations reused or reordered')
    return True


def audit(directory,case_index=None):
    directory=Path(directory)
    require(case_index is None or type(case_index) is int and case_index in CASE_INDICES,'unknown subset case')
    evidence={};errors=[];results=[];closures=[]
    preflight=read(directory/'preflight.json',evidence)
    require(preflight.get('schema')==RUN_SCHEMA and preflight.get('protocol')==headtail_scale20_protocol()
        and preflight.get('evaluation_case_indices')==list(CASE_INDICES),'foreign twenty-case execution protocol')
    audit_native_swa_configuration(preflight);source_before=frozen_sources(preflight)
    manifest=read(directory/'input_manifest.json',evidence)
    input_protocol=read(directory/'input-protocol.json',evidence)
    for name in ('source-manifest.json','extension-manifest.json','extension-input-protocol.json','source-plan.json'):
        read(directory/name,evidence)
    sys.path.insert(0,str(Path(preflight['identity']['experiment_root'])/'benchmark'))
    import pr148_native_runner as base
    validate_inputs(base,manifest,input_protocol,input_dir=directory)
    require(manifest['sha256']==preflight['manifest_sha256']==preflight['protocol']['input_manifest_sha256'],
            'twenty-case input protocol/seal differs')
    by_index={r['index']:r for r in manifest['cohort']}
    rows=preflight.get('audit_cases')
    require(type(rows) is list and [r.get('index') for r in rows]==list(CASE_INDICES)
        and [r.get('dataset') for r in rows]==['hotpotqa','2wikimqa']*10,'fixed balanced20 cohort differs')
    for row in rows:
        require(type(row.get('input_tokens')) is int and row['input_tokens']>=262145,'all20 must contain8x32K')
        source=by_index[row['index']]
        require(all(row[k]==source[k] for k in row),'preflight gold/input/source drift')
        ids=source['input_ids']
        require(len(ids)==row['input_tokens'] and digest(ids)==row['input_ids_sha256'],'raw input IDs changed')
        for lo,hi,key in documents(row):require(digest(ids[lo:hi])==key,'exact document token hash differs')
    qual=qualification(directory,preflight,evidence) if (directory/'qualification.json').exists() else None
    smokes={};smoke_summaries={}
    for arm in POLICY_ARMS:
        if (directory/f'{arm}.partial_smoke.json').exists():
            require(qual is not None,'same-policy smoke precedes native qualification')
            smokes[arm],smoke_summaries[arm]=partial_smoke(directory,preflight,qual,evidence,arm)
    if len(smokes)==2:
        require(smoke_summaries[POLICY_ARMS[0]]['shared_bank_identity']==smoke_summaries[POLICY_ARMS[1]]['shared_bank_identity'],
                'two policies qualified with different GPU bank identities')
        require(smoke_summaries[POLICY_ARMS[0]]['first_three_reference_fingerprints_sha256']
                ==smoke_summaries[POLICY_ARMS[1]]['first_three_reference_fingerprints_sha256'],
                'first three native layers differ across independently qualified policies')
    requested=rows if case_index is None else [r for r in rows if r['index']==case_index]
    for row in requested:
        if not (directory/f"case{row['index']}.paired.json").exists():continue
        require(qual is not None and len(smokes)==2,'formal case precedes both fresh policy admissions')
        try:
            closure,result=audit_case(directory,row,preflight,qual,smokes,evidence)
            if row['index']==2:
                require(result['shared_bank_identity']==smoke_summaries[POLICY_ARMS[0]]['shared_bank_identity'],
                        'first formal case changed bank after smoke')
            closures.append(closure);results.append(result)
        except (ValueError,KeyError,TypeError,OSError) as exc:errors.append(dict(index=row['index'],error=str(exc)))
    complete=False
    if case_index is None and (directory/'run.json').exists():
        complete=validate_closed_run(read(directory/'run.json',evidence),preflight,qual,smokes,closures,errors,directory,evidence)
    subset=case_index is not None and not errors and len(results)==1
    status='invalid' if errors else 'complete' if complete else 'subset_complete' if subset else 'incomplete'
    require(frozen_sources(preflight)==source_before,'source changed during independent audit')
    output=dict(schema=SCHEMA,status=status,frozen_sources_verified=source_before,
        completed_case_indices=[r['index'] for r in results],complete_cases=len(results),planned_cases=20,
        audit_scope='all_registered_cases' if case_index is None else 'explicit_single_case_subset',
        requested_case_indices=[r['index'] for r in requested],final_run_audited=complete,overall_run_complete=complete,
        all_registered_cases_audited=complete,qualification_passed=qual is not None,partial_smokes=smoke_summaries,
        cases=results,errors=errors,evidence=evidence,summary=summarize_cases(results),
        by_dataset={d:summarize_cases([c for c in results if c['dataset']==d]) for d in ('hotpotqa','2wikimqa')},
        manifest_sha256=preflight['manifest_sha256'],input_protocol_sha256=input_protocol['sha256'],
        execution_configuration_sha256=preflight['identity']['execution_configuration_sha256'],
        protocol_sha256=preflight['protocol']['sha256'],cuda_initialized=False,
        fresh_all_twenty_paired_baselines=True,historical_results_used_for_admission=False,
        diagnostic_excluded_from_formal_metrics=True,diagnostic_cases=len(results),
        original_four_inputs_previously_exposed=True,statistical_generalization=False,cold_start_claim=False,
        cache_storage_policy='shared_gpu_full_canonical_bank_all_arms',historical_static_cpu_staging_comparable=False,
        no_flops_fraction_claim=True,c4_measured=False)
    return dict(output,sha256=digest(output))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--case-index',type=int,choices=CASE_INDICES)
    args=parser.parse_args(argv);result=audit(args.result_dir,args.case_index)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:stream.write(payload)
    else:print(payload,end='')
    return 0 if result['status'] in ('complete','subset_complete') and not result['errors'] else 1


def frozen_sources(preflight):
    """Recheck every bound Python/dependency/native source, without importing it."""
    from gpu_pilot.headtail_scale20_protocol import DEPENDENCIES
    require(DEPENDENCIES, "new reader dependency pins have not been frozen")
    stage = Path(__file__).resolve().parents[1]
    identity = preflight["identity"]
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
    return dict(pilot_source_sha256=digest(recorded), native_source_sha256=digest(native_files))

def qualification(directory, preflight, evidence):
    mask = cuda_mask_qualification(directory, preflight, evidence)
    from gpu_pilot.index_hit20_empty_ep_qualification import validate_all_ranks_report
    empty_ep = read(directory/"empty_ep_qualification.json", evidence)
    validate_all_ranks_report(empty_ep,
        execution_sha256=preflight["identity"]["execution_configuration_sha256"],
        model_sha256=preflight["identity"]["checkpoint_index_sha256"])
    raw = [read(directory/name, evidence) for name in
           ("qualification.native.json", "qualification.all_selected.json")]
    closure = read(directory/"qualification.json", evidence)
    require(closure.get("cuda_mask_qualification_sha256") == mask["sha256"],
            "strict numerical qualification must bind completed CUDA mask checks")
    require(closure.get("empty_ep_qualification_sha256") == empty_ep["sha256"],
            "strict numerical qualification must bind current native empty-EP proof")
    require(closure.get("schema") == "headtail-scale20-qualification-v1" and closure.get("passed") is True
            and closure.get("reference_sha256") == raw[0]["sha256"]
            and closure.get("candidate_sha256") == raw[1]["sha256"]
            and closure.get("qualification_input_tokens") == 16384
            and closure.get("identity_check_count") == 80
            and closure.get("full_256k_numerical_parity_claim") is False
            and closure.get("old_qualification_reused") is False, "qualification closure mismatch")
    fingerprints = []
    for phase, mode, name in zip(raw, ("qualified_native", "all_selected"),
                                 ("qualification.native","qualification.all_selected")):
        empty_ep_binding(phase, closure)
        require(phase.get("schema") == PHASE_SCHEMA and phase.get("mode") == mode
                and phase.get("index") == 0 and phase.get("arm") == "qualification"
                and phase.get("name") == name and phase.get("repeat") is None
                and phase.get("timing_valid") is False and phase.get("quality_eligible") is False,
                "foreign qualification phase")
        binding(phase, preflight)
        record, control = phase["record"], phase["controller"]
        _fields(phase,dict(diagnostic=False),'native qualification diagnostic flag')
        comparison_scope(control,phase,expected_scope=digest([]))
        _fields(control,dict(diagnostic=False,early_token_diagnostics=None,offline_bank_entry_count=0,
            offline_bank_rows=0,offline_bank_bytes=0),'qualification before shared-bank construction')
        if mode == "qualified_native":
            audit_native_no_cache(control)
        expected_input=preflight.get("qualification_input_ids_sha256")
        validate_digest(expected_input,"qualification manifest prefix")
        require(record.get("success") is True and record.get("rank") == 0
                and phase.get("input_ids_sha256")==record.get("input_ids_sha256")==control.get("input_ids_sha256")==expected_input
                and control.get("full_input_tokens")==16384
                and record.get("phase")==control.get("phase")==name and control.get("mode")==mode
                and record.get("natural_EOS") is False
                and record.get("input_tokens") == 16384 and len(record.get("output_ids", [])) == 1
                and record.get("output_ids_sha256") == digest(record["output_ids"]),
                "qualification output/length mismatch")
        rows = control.get("identity_fingerprints")
        expected = {(start, start+8192, layer) for start in (0, 8192) for layer in range(40)}
        require(type(rows) is list and len(rows) == 80 and control.get("identity_check_count") == 80,
                "all eighty raw layer/chunk fingerprints required")
        seen = set()
        for row in rows:
            key = tuple(row.get("key", []))
            require(key in expected and key not in seen, "foreign or duplicate qualification layer/chunk")
            seen.add(key)
            value = row["value"]
            require(set(value) == {"output", "lazy"}
                    and set(value["lazy"]) == {"residual", "post_layer_mix", "comb_res_mix", "pre_layer_mix"},
                    "every native lazy field must be fingerprinted")
            require(value["output"] is not None, "decoder output missing")
            _fingerprint(value["output"])
            for item in value["lazy"].values():
                _fingerprint(item)
        require(digest(rows) == control.get("identity_fingerprints_sha256"), "fingerprint aggregate mismatch")
        fingerprints.append(rows)
    require(raw[0]["input_ids_sha256"] == raw[1]["input_ids_sha256"]
            and raw[0]["record"]["output_ids"] == raw[1]["record"]["output_ids"]
            and fingerprints[0] == fingerprints[1]
            and raw[1]["controller"].get("identity_exact") is True, "all-selected differs from native")
    checks=projection_identity(raw[1]["controller"])
    require(closure.get("projection_identity_checks")==checks,"split qualification closure mismatch")
    require(closure.get("cache_restore_identity_checks")==cache_restore_identity(raw[1]["controller"]),
            "cache restoration qualification closure mismatch")
    require(audit_joint_attention(raw[1]["controller"])==0,"all-selected qualification cannot reuse cached rows")
    return closure


if __name__=='__main__':raise SystemExit(main())
