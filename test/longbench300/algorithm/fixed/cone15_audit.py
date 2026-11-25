"""Read true eight-rank phase records and compare live states separately from outputs."""
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
import textwrap


def require(ok, message):
    if not ok:
        raise ValueError(message)


def physical_last(event, original):
    ids=event.get('cone15_core_positions')
    return (isinstance(ids,list) and ids==sorted(set(ids)) and set(ids)<=set(original)
            and event['actual_attention_rows']==len(ids) and event['selected_wrapper_rows']==len(original))


def private_func(fn,old,new,extra):
    source=textwrap.dedent(inspect.getsource(fn))
    require(source.count(old)==1,'Unknown native validation seam '+fn.__name__)
    ns=dict(fn.__globals__,**extra)
    exec(compile(source.replace(old,new),'<private Cone15 '+fn.__name__+'>','exec'),ns)
    return ns[fn.__name__]


@lru_cache(maxsize=1)
def cone_lastkv():
    import index20_lastkv_audit as audit
    import index20_lastkv_contract as contract
    phase=private_func(audit.audit_phase,
                       'need(event,"actual_attention_rows")==len(expected)','physical_last(event,expected)',
                       dict(physical_last=physical_last))
    return private_func(contract.validate_lastkv_phase,
                        'from index20_lastkv_audit import audit_projection_evidence,audit_phase',
                        'audit_projection_evidence,audit_phase=original_projection,physical_phase',
                        dict(original_projection=audit.audit_projection_evidence,physical_phase=phase))


def validate_lastkv(phase,row):
    from index20_lastkv_contract import validate_lastkv_phase
    if phase['method']=='Cone15':
        return cone_lastkv()(phase,row)
    require(phase['method']=='Fixed15Opt','Unknown lastKV phase arm')
    return validate_lastkv_phase(phase,row)


def source_guard(proof):
    guards=proof['worker_guards']
    require([g['rank'] for g in guards]==list(range(8)) and all(
        g['world']==8 and g['cp']==1 and g['rearrangement_interval']==0 and g['random_routing'] is False
        and g['num_extra_experts']==0 and g['sm']==90 and g['layers']==list(range(21,40))
        and enabled_digest(g['native_empty_ep']) for g in guards)
        and len({g['native_empty_ep'] for g in guards})==1,'Invalid real eight-rank guard')


def enabled_digest(value):
    return isinstance(value, str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def tensor_receipt(value):
    return (isinstance(value,dict) and enabled_digest(value.get('sha256')) and
            isinstance(value.get('shape'),list) and all(type(v) is int and v>=0 for v in value['shape'])
            and isinstance(value.get('dtype'),str) and value['dtype'].startswith('torch.'))


def cache_receipts(cache):
    require(isinstance(cache,dict) and tensor_receipt(cache['dflash_features']),'Missing D-Spark feature bytes')
    rings=cache['rings']; keys=[r['layer'] for r in rings]
    require(keys[:19]==list(range(21,40)) and len(keys)>19 and len(set(keys))==len(keys) and
            all(tensor_receipt(r['valid_swa']) for r in rings),'Missing or duplicated main/draft native SWA bytes')


def check_input_oracle(value,enabled,count):
    fields=value['fields']
    if not enabled:
        require(value==dict(fields={},checked_fields=0,all_equal=None,different_fields=[]),
                'Unexpected input tensor readback outside masked qualification')
        return ()
    names={'fp8','sf','bf16','residual','post_layer_mix','comb_res_mix','pre_layer_mix','q_a','swa'}
    require(set(fields)==names,'Incomplete masked input tensor comparison')
    changed=[];present=0
    for name,f in fields.items():
        if not f['present']:
            require(name in {'residual','post_layer_mix','comb_res_mix','pre_layer_mix'} and
                    f['rows']==f['different_rows']==0 and f['reference_sha256'] is None and
                    f['candidate_sha256'] is None and f['byte_equal'] is True,'Invalid absent native mHC field')
            continue
        present+=1
        left,right=f['reference_sha256'],f['candidate_sha256']
        require(enabled_digest(left) and enabled_digest(right) and f['rows']==count and
                0<=f['different_rows']<=count and
                (left==right)==(f['different_rows']==0)==f['byte_equal'],'Inconsistent masked numerical evidence')
        if not f['byte_equal']:changed.append(name)
    require(present==value['checked_fields'] and sorted(changed)==sorted(value['different_fields']) and
            value['all_equal']==(not changed),'Inconsistent masked-input summary')
    return changed


def audit_cone_phase(phase,row):
    from cone15_plan import Cone15Plan,row_receipt
    from cone_query import validate_core
    c=phase['controller']; mode=phase['method']; reference=mode=='Fixed15Opt'; qual=phase['qualification_only']
    n=row['input_tokens']; proof=c['cone15_ffn']
    require(mode in ('Fixed15Opt','Cone15') and mode==proof['method'] and proof['extra_native_collectives']==0
            and phase['record']['input_ids_sha256']==row['input_ids_sha256'],'Cone15 phase binding differs')
    source_guard(proof)
    from index20_plan import mandatory_selection
    docs=[dict(start=d['input_start'],end=d['input_end'],key=d['token_ids_sha256']) for d in row['documents']
          if d['input_end']-d['input_start']>=256]
    mandatory=mandatory_selection(docs,n)
    sel=c['index20']['selection']
    chosen=sel['selected_positions']
    require(sel['mandatory_positions']==mandatory and sel['fraction']==.15 and sel['strategy']=='max',
            'Initial fixed selection or mandatory positions differ')
    groups=c['system_batch_execution']['groups']
    ranges=[(g['start'],g['end']) for g in groups]
    plan=Cone15Plan(n,chosen,mandatory,last_frame_start=groups[-1]['member_chunks'][-1][0],group_ranges=ranges)
    require(ranges==[(g['start'],g['end']) for g in proof['groups']],'Missing actual group geometry')
    decoder={(e['layer'],e['chunk_start']):e for e in c['decoder_events']}
    attn={(e['layer'],e['chunk_start']):e for e in c['attention_report']['attention_events']}
    require(len(decoder)==len(c['decoder_events'])==40*((n+8191)//8192),'Invalid decoder coverage')
    require(len(attn)==len(c['attention_report']['attention_events'])==37*len(groups),'Invalid SWA coverage')
    layers=[dict(layer=i,selected_rows=0,core_rows=0,ffn_rows=0,ffn_norm_rows=0,attn_norm_rows=0,
                 residual_rows=0,swa_projection_rows=0,query_projection_rows=0) for i in range(40)]
    qualified_cache=proof['qualification_cache_bytes']
    require((qualified_cache is not None)==qual,'Own32 cache hashes missing or computed in timing')
    if qual:cache_receipts(qualified_cache)
    for layer in range(21):
        for a in range(0,n,8192):
            e=decoder[layer,a]; count=e['selected_rows']
            require(e['actual_ffn_rows']==e['actual_attention_rows']==e['actual_residual_rows']==e['actual_mhc_rows']==count,
                    'Original first 21 layers changed')
            for k in ('selected_rows','core_rows','ffn_rows','ffn_norm_rows','residual_rows','attn_norm_rows','swa_projection_rows','query_projection_rows'):
                layers[layer][k]+=count
    for g,actual in zip(groups,proof['groups']):
        a,b=g['start'],g['end']; S=g['selected_positions']
        require(list(plan.selection(a,b))==S and [l['layer'] for l in actual['layers']]==list(range(21,40)),
                'Native group/layer sequence changed')
        for e in actual['layers']:
            i=e['layer']; geom=plan.group(i,a,b);q=list(geom.queries); wanted_input=list(geom.inputs)
            original=S if reference else q
            fallback_input=bool(not reference and S and not wanted_input)
            projected=S if (reference or fallback_input) else wanted_input
            pl=int(bool(S) and not original); mhc_rows=sorted(set(projected+([S[0]] if pl else [])))
            native=attn[i,a]; digest=row_receipt(q)['sha256']
            require(e['live_ffn_rows']==len(q) and e['live_positions_sha256']==digest
                    and e['actual_ffn_rows']==e['dispatch_rows']==len(original)
                    and e['actual_ffn_norm_rows']==e['actual_ffn_mhc_rows']==len(original)+pl and e['norm_placeholder_rows']==pl
                    and e['combine_calls']==1,'Actual norm/MLP/EP rows differ from independent plan')
            if qual:
                require(e['qualification_row_ids_sha256']==digest and enabled_digest(e['qualification_live_ffn_sha256'])
                        and e['qualification_live_ffn_shape'][0]==len(q),'Missing actual device FFN bytes/IDs')
            else:
                require(all(e[k] is None for k in ('qualification_row_ids_sha256','qualification_live_ffn_sha256','qualification_live_ffn_shape')),
                        'Unexpected FFN readback in benchmark')
            validate_core(e['core'],i,len(S),len(q),reference,qual)
            inp=e['inputs']
            require(inp['selected_axis_rows']==len(S) and inp['live_input_rows']==len(wanted_input)
                    and inp['empty_live_uses_native']==fallback_input
                    and inp['attn_norm_calls']==inp['q_a_swa_calls']==int(bool(S))
                    and inp['actual_attn_norm_rows']==inp['actual_q_a_swa_rows']==len(projected),
                    'Physical attention normalization / first Q_a+SWA projection differs')
            oracle=bool(qual and not reference and 0<len(wanted_input)<len(S))
            check_input_oracle(inp['input_equivalence'],oracle,len(wanted_input))
            require(inp['qualification_comparison_enabled']==oracle
                    and inp['norm_checked_rows']==inp['q_a_swa_checked_rows']==(len(wanted_input) if oracle else 0)
                    and inp['dense_oracle_norm_rows']==inp['dense_oracle_q_a_swa_rows']==(len(S) if oracle else 0),
                    'Qualification-only norm/Q_a/SWA oracle evidence incomplete')
            require(native['chunk_end']==b and native['selected_positions']==S and native['actual_attention_rows']==len(original)
                    and native['selected_wrapper_rows']==len(S) and native['cone15_core_positions']==original
                    and native['actual_query_projection_rows']==len(original) and native['actual_swa_projection_rows']==len(projected)
                    and native['cone15_input_positions']==projected
                    and native['gpu_local_window_oracle_checked'] is True and native['actual_native_swa_indices_checked'] is True
                    and native['global_causal_checked'] is True,'Actual attention event or native visibility differs')
            for ca,cb in g['member_chunks']:
                sub=[p for p in original if ca<=p<cb]; selected=[p for p in S if ca<=p<cb]; dec=decoder[i,ca]
                frame_input=[p for p in projected if ca<=p<cb]; frame_mhc=[p for p in mhc_rows if ca<=p<cb]
                require(dec['selected_positions']==selected and dec['actual_attention_rows']==dec['actual_ffn_rows']==len(sub)
                        and dec['actual_ffn_norm_rows']==dec['actual_ffn_mhc_rows']==len(sub)+int(bool(pl) and ca<=S[0]<cb)
                        and dec['selected_wrapper_rows']==len(selected) and dec['actual_query_projection_rows']==len(sub)
                        and dec['actual_attn_norm_rows']==dec['actual_swa_projection_rows']==len(frame_input)
                        and dec['actual_residual_rows']==dec['actual_mhc_rows']==len(frame_mhc) and dec['cone15_mhc_positions']==frame_mhc
                        and dec['cone15_input_positions']==frame_input
                        and dec['cone15_ffn_positions']==dec['cone15_core_positions']==sub,
                        'Native-frame decoder execution differs from group/owner plan')
            for k,v in dict(selected_rows=len(S),core_rows=len(original),ffn_rows=len(original),ffn_norm_rows=len(original)+pl,
                            residual_rows=len(mhc_rows),attn_norm_rows=len(projected),swa_projection_rows=len(projected),
                            query_projection_rows=len(original)).items():
                layers[i][k]+=v
    require(c['dflash_tail']['target_layer_ids']==[37,38,39] and
            c['dflash_tail']['retained_positions']==list(range(max(0,n-128),n)), 'D-Spark target tail differs')
    return dict(physical_rows=layers,actual_ffn_token_layers=sum(l['ffn_rows'] for l in layers),
                actual_attention_core_token_layers=sum(l['core_rows'] for l in layers),
                original_layer2_selection_preserved=True, full_lastkv_replay_checked_separately=True,
                scope='logical operator inputs of this owner; excludes tensor tiling, EP receives and inactive local owners')


def pair_own32(base,cand,row,reader):
    def bounded(req):
        entry=reader(req['qualification_ref']); warm=reader(entry['actual_same_owner_set_warm_ref'])
        phase=reader(warm['phase_ref'])
        require(phase['qualification_only'] is True and phase['benchmark_request'] is False and
                phase['record']['input_ids_sha256']==req['input_ids_sha256']==row['input_ids_sha256'] and
                warm['input_ids_sha256']==row['input_ids_sha256'] and 0<len(phase['record']['output_ids'])<=32 and
                req['output_ids'][:len(phase['record']['output_ids'])]==phase['record']['output_ids'] and
                warm['method']==phase['method'] and warm['output_ids']==phase['record']['output_ids'] and
                warm['owner_rank']==phase['owner_rank']==req['owner_rank'],'Invalid own32 provenance')
        return phase
    left,right=bounded(base),bounded(cand)
    require((left['method'],right['method'])==('Fixed15Opt','Cone15') and left['owner_rank']==right['owner_rank'],
            'Cross-arm own32 owner/method changed')
    a,b=left['controller']['cone15_ffn'],right['controller']['cone15_ffn']
    groups=[]; differences=[]
    cache_match=a['qualification_cache_bytes']==b['qualification_cache_bytes']
    require(a['qualification_cache_bytes'] and b['qualification_cache_bytes'], 'Missing real per-owner cache receipt')
    require(len(a['groups'])==len(b['groups']) and a['groups'],'Missing cross-arm own32 groups')
    for gl,gr in zip(a['groups'],b['groups']):
        require((gl['start'],gl['end'])==(gr['start'],gr['end']), 'Cross-arm group geometry changed')
        item=dict(start=gl['start'],end=gl['end'],layers=[])
        require([l['layer'] for l in gl['layers']]==[l['layer'] for l in gr['layers']]==list(range(21,40)),
                'Missing cross-arm own32 layers')
        for x,y in zip(gl['layers'],gr['layers']):
            require(x['layer']==y['layer'] and x['live_ffn_rows']==y['live_ffn_rows'] and
                    x['live_positions_sha256']==y['live_positions_sha256'],'Cross-arm live position geometry changed')
            ix=x['layer']; lhs=x['qualification_live_ffn_sha256']; rhs=y['qualification_live_ffn_sha256']
            core0=x['core']['retained_output'];core1=y['core']['retained_output']
            require(enabled_digest(lhs) and enabled_digest(rhs) and tensor_receipt(core0) and tensor_receipt(core1)
                    and x['qualification_live_ffn_shape']==y['qualification_live_ffn_shape'],
                    'Missing live state bytes on own32')
            fmatch=lhs==rhs; cmatch=core0==core1
            iv=y['inputs']['input_equivalence']
            input_errors=check_input_oracle(iv,y['inputs']['qualification_comparison_enabled'],y['inputs']['live_input_rows'])
            result=dict(layer=ix,ffn_live_bytes_match=fmatch,attention_live_bytes_match=cmatch,input_byte_oracle_match=iv['all_equal'],
                        differing_input_tensors={key:iv['fields'][key] for key in input_errors})
            item['layers'].append(result)
            if not (fmatch and cmatch) or input_errors:
                differences.append(dict(start=gl['start'],end=gl['end'],**result))
        final=gl['end']==row['input_tokens']
        sa,sb=gl['qualification_sampler_sha256'],gr['qualification_sampler_sha256']
        require((sa is not None)==final and (sb is not None)==final and
                (not final or (enabled_digest(sa) and enabled_digest(sb))), 'Missing terminal sampler observation')
        item['sampler_match']=(sa==sb and gl['qualification_sampler_shape']==gr['qualification_sampler_shape']) if final else None
        groups.append(item)
    left_ids,right_ids=left['record']['output_ids'],right['record']['output_ids']
    return dict(index=row['index'],reference_phase_sha256=left['sha256'],candidate_phase_sha256=right['sha256'],
                all_live_bytes_exact=not differences and cache_match and groups[-1]['sampler_match'],
                cache_bytes_match=cache_match, output_ids_match=left_ids==right_ids,
                observed_reference_ids=left_ids,observed_candidate_ids=right_ids,
                per_group=groups,observed_state_differences=differences)
