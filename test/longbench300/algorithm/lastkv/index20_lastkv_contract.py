"""CPU readers for actual last-layer coverage and paired numerical diagnostics."""
from rag_dspark_runtime import digest,require,seal

def validate_lastkv_phase(phase,row):
    from index20_lastkv_audit import audit_projection_evidence,audit_phase
    control=phase['controller'];info=control['index20_lastkv']
    require(info['sha256']==digest({k:v for k,v in info.items()if k!='sha256'}),'LastKV evidence seal differs')
    require(info['schema']=='index20-all-input-layer39-swa-v1'and info['enabled']is True
        and info['variant']=='all_input_layer39_swa_reprojection'and info['input_tokens']==row['input_tokens'],
        'Wrong all-input last-layer execution scope')
    require(info['all_input_global_layer39_recomputed']is False
        and info['global_layer39']=='reuse_source20_no_independent_projection'
        and info['missing_online_context_hidden_recomputed']is False,
        'LastKV must disclose restored offline hidden and native global source sharing')
    storage=audit_phase(phase,require_all_input=True)
    require(storage['status']=='passed','LastKV native attention/ring/source20/D-Spark evidence is incomplete')
    selected=control['index20']['selection']['selected_positions']
    coverage=audit_projection_evidence(info['attention'],row['input_tokens'],selected)
    require(coverage['status']=='passed'and coverage==info['coverage']and info['all_input_rows_reprojected']is True,
        'Actual full-input last-layer projection coverage is incomplete')
    require(info['live_selected_rows']==len(selected)and info['restored_offline_rows']==row['input_tokens']-len(selected),
        'Actual online/offline last-layer input source counts differ')
    return seal(dict(schema='index20-lastkv-case-execution-proof-v1',passed=True,index=row['index'],phase_sha256=phase['sha256'],
        live_selected_rows=len(selected),restored_offline_rows=row['input_tokens']-len(selected),input_tokens=row['input_tokens'],
        actual_coverage=coverage,native_storage_and_source_audit=storage,extra_offline_input_bank=info['extra_offline_input_bank'],
        global_source='source20_reused_no_independent_layer39_projection',missing_online_context_hidden_recomputed=False,
        all_input_projection_does_not_mean_all_rows_persist_in_swa_ring=True))

def compare_selected_states(original,candidate):
    a=original['controller']['index20'];b=candidate['controller']['index20']
    require(a['selection']['selected_positions']==b['selection']['selected_positions'],'Original/new variant selected different token sets')
    require(a['diagnostic_fingerprints_collected']is True and b['diagnostic_fingerprints_collected']is True,
        'Actual selected-state diagnostics were not collected')
    total=original['record']['input_tokens'];require(candidate['record']['input_tokens']==total,'Diagnostic input length changed')
    expected=[[start,min(start+8192,total),layer]for start in range(0,total,8192)for layer in range(40)]
    left=a['diagnostic_fingerprints'];right=b['diagnostic_fingerprints']
    require([r['key']for r in left]==expected,'Diagnostic fingerprints omit or repeat native chunk/layer coverage')
    require(left and right and [r['key']for r in left]==[r['key']for r in right],'Selected-state layer/chunk coverage differs')
    different=[dict(key=x['key'],output_equal=x['value']['output']==y['value']['output'],
        lazy_fields_equal={k:x['value']['lazy'][k]==y['value']['lazy'][k]for k in x['value']['lazy']})
        for x,y in zip(left,right)if x['value']!=y['value']]
    require(not any(x['key'][2]<39 for x in different),'LastKV changed selected hidden/mHC before layer39')
    return seal(dict(schema='index20-lastkv-selected-state-comparison-v1',original_phase_sha256=original['sha256'],
        lastkv_phase_sha256=candidate['sha256'],same_selected_positions=True,layer_chunk_comparisons=len(left),
        pre_last_layer_selected_hidden_and_all_four_mhc_exact=True,pre_last_layer_checked_ids=list(range(39)),
        selected_hidden_and_all_four_mhc_bitwise_equal=not different,differing_layer_chunks=different,
        first_differing_layer_zero_based=min(x['key'][2]for x in different)if different else None,
        diagnostic32_output_ids_equal=original['record']['output_ids']==candidate['record']['output_ids'],
        diagnostic32_output_text_equal=original['record']['text']==candidate['record']['text'],
        full_natural_output_equality_measured_here=False,numerical_tolerance_relaxed=False,
        equality_not_assumed=True,fingerprint_equality_not_full_context_hidden_equivalence=True))
