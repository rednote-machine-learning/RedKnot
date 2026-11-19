"""CPU-only audit of real selected/full-input layer39 SWA phase evidence.

Reports distinguish recorded native operations from an independent tensor
readback. Missing fields never become successful checks. No Torch is imported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def seal(value):return dict(value,sha256=digest(value))


class MissingEvidence(ValueError):pass
class ContractMismatch(ValueError):pass


def need(value,key):
    if not isinstance(value,dict) or key not in value:raise MissingEvidence("missing "+key)
    return value[key]


def require(value,message):
    if not value:raise ContractMismatch(message)


def checked(function):
    try:return dict(status="passed",evidence=function())
    except MissingEvidence as error:return dict(status="incomplete",reason=str(error))
    except (ContractMismatch,TypeError,ValueError,IndexError,KeyError,AttributeError) as error:
        return dict(status="failed",reason=str(error))


def positions(value,total,label):
    require(isinstance(value,list) and all(type(p)is int and 0<=p<total for p in value),label+" positions invalid")
    require(value==sorted(set(value)),label+" positions must be sorted/unique")
    return value


def audit_projection_evidence(evidence,total,selected):
    def verify():
        require(type(total)is int and total>0,"invalid full input length")
        positions(selected,total,"selected")
        require(need(evidence,"enabled")is True,"full-input reprojection is disabled")
        events=need(evidence,"events")
        require(isinstance(events,list) and events,"no full-input projection events")
        cursor=0;live=0;offline=0;ranges=[]
        for event in events:
            a,z=need(event,"start"),need(event,"end")
            require(type(a)is int and type(z)is int and a==cursor and a<z<=total,
                    "full-input projection gap, overlap or unordered interval")
            expected=[p for p in selected if a<=p<z]
            require(need(event,"projected_rows")==z-a,"projection row count differs from full interval")
            require(need(event,"selected_positions")==expected,"online selected source mapping differs")
            require(need(event,"live_selected_rows")==len(expected),"live selected row count differs")
            require(need(event,"restored_offline_rows")==z-a-len(expected),"offline restored row count differs")
            require(need(event,"native_swa_projection")is True and need(event,"global_projection")is False
                and need(event,"positions_preserved")is True,"wrong projection kind/position semantics")
            live+=len(expected);offline+=z-a-len(expected);cursor=z;ranges.append([a,z])
        require(cursor==total and live==len(selected) and offline==total-len(selected),
                "full-input projection coverage incomplete")
        return dict(projected_input_rows=total,coverage_ranges=ranges,live_selected_rows=live,
            restored_offline_rows=offline,native_projection_event_count=len(events),
            all_input_positions_exactly_once=True,global_layer39_projected=False,
            independent_projected_tensor_readback=False,
            proof_scope="recorded_native_projection_calls_and_source_mapping")
    result=checked(verify)
    return seal(dict(schema="index20-lastkv-projection-audit-v1",**result))


def audit_phase(phase,*,require_all_input=False,verify_seal=True):
    checks={}
    def seal_check():
        require(isinstance(phase,dict),"phase is not an object")
        require(need(phase,"sha256")==digest({k:v for k,v in phase.items()if k!="sha256"}),"phase seal differs")
        return dict(phase_sha256=phase["sha256"])
    checks["phase_seal"]=checked(seal_check)if verify_seal else dict(status="not_checked")
    control=phase.get("controller")if isinstance(phase,dict)else None
    total=control.get("full_input_tokens")if isinstance(control,dict)else None
    info=control.get("index20",{})if isinstance(control,dict)else{}
    selection=info.get("selection",{})if isinstance(info,dict)else{}
    selected=selection.get("selected_positions")if isinstance(selection,dict)else None
    attention=control.get("attention_report",{})if isinstance(control,dict)else{}
    variant=control.get("index20_lastkv",{})if isinstance(control,dict)else{}

    def identity():
        require(need(control,"schema")=="index20-controller-v1","not an Index20 controller phase")
        require(type(total)is int and total>0,"invalid input length")
        require(need(need(phase,"record"),"input_ids_sha256")==need(control,"input_ids_sha256"),"phase input identity mismatch")
        positions(need(selection,"selected_positions"),total,"selected")
        require(need(selection,"selected_tokens")==len(selected),"selected count mismatch")
        mandatory=positions(need(selection,"mandatory_positions"),total,"mandatory")
        require(set(mandatory)<=set(selected),"mandatory tokens missing from selected set")
        return dict(input_tokens=total,selected_tokens=len(selected),mandatory_tokens=len(mandatory),
            selected_set_sha256=digest(selected),all_input_selected=len(selected)==total)
    checks["identity_and_selection"]=checked(identity)

    def layer39():
        events=need(attention,"attention_events")
        require(isinstance(events,list),"attention event type")
        events=[e for e in events if e.get("layer")==39]
        require(events,"no layer39 attention events")
        cursor=0;seen=[];ranges=[]
        for event in events:
            a,z=need(event,"chunk_start"),need(event,"chunk_end")
            require(a==cursor and a<z<=total,"layer39 attention interval gap/overlap")
            expected=[p for p in selected if a<=p<z]
            require(need(event,"selected_positions")==expected,"layer39 selected positions differ")
            require(need(event,"selected_rows")==len(expected)and need(event,"actual_attention_rows")==len(expected),
                    "layer39 actual row count differs")
            for field in ("gpu_local_window_oracle_checked","actual_native_swa_indices_checked","global_causal_checked"):
                require(need(event,field)is True,"layer39 native dependency check absent: "+field)
            seen.extend(expected);cursor=z;ranges.append([a,z])
        require(cursor==total and seen==selected,"layer39 selected attention coverage incomplete")
        return dict(selected_layer39_rows=len(seen),ranges=ranges,
            projection_scope="selected_queries_use_current_layer39_attention_inputs",
            raw_swa_projection_tensor_readback=False,all_input_swa_projection_proven_by_this_check=False)
    checks["layer39_selected_attention"]=checked(layer39)

    def ring():
        events=need(attention,"decode_ring_events")
        events=[e for e in events if e.get("layer")==39]
        require(len(events)==1,"need exactly one final layer39 ring write")
        event=events[0];expected=list(range(max(0,total-128),total))
        require(need(event,"positions")==expected and need(event,"physical_store")is True,
                "layer39 native final ring not fully written")
        readback=need(event,"physical_readback_exact")
        require(type(readback)is bool,"ring readback evidence must be explicit")
        return dict(positions=expected,native_physical_store_recorded=True,
            exact_tensor_readback_verified=readback,all_input_positions_persisted_in_ring=False)
    checks["layer39_final_decode_ring"]=checked(ring)

    def global_source():
        events=need(need(control,"cache_report"),"global_events")
        require(not any(e.get("layer")==39 for e in events),"unexpected layer39 global projection")
        source=[e for e in events if e.get("layer")==20]
        require(source,"no source20 global refresh evidence")
        cursor=0;online=[]
        for event in source:
            a,z=need(event,"chunk_start"),need(event,"chunk_end")
            require(a==cursor and a<z<=total,"source20 global coverage gap/overlap")
            expected=[p for p in selected if a<=p<z]
            require(need(event,"online_projection_positions")==expected
                and need(event,"compression_ratio")==1
                and need(event,"main_and_index_k_native_store")is True,"source20 main/index refresh differs")
            online.extend(expected);cursor=z
        require(cursor==total and online==selected,"source20 refresh incomplete")
        if variant:
            require(need(variant,"all_input_global_layer39_recomputed")is False
                and need(variant,"global_layer39")=="reuse_source20_no_independent_projection",
                "variant mislabeled source20 shared global KV")
        return dict(layer39_independent_global_projection=False,global_kv_source_layer=20,
            source20_online_selected_rows=len(online),source20_main_and_index_written=True,
            proof_scope="recorded_source_topology_and_global_update_events")
    checks["layer39_global_source20"]=checked(global_source)

    def dflash():
        d=need(control,"dflash_tail")
        require(need(d,"enabled")is True and need(d,"integrated_all_rank_runtime")is True,
                "D-Spark integrated bridge not enabled")
        require(need(d,"input_ids_sha256")==need(control,"input_ids_sha256"),"D-Spark input differs")
        expected=list(range(max(0,total-128),total));targets=need(d,"target_layer_ids")
        require(isinstance(targets,list)and targets and len(set(targets))==len(targets),"D-Spark target list invalid")
        require(need(d,"window_size")==128 and need(d,"retained_positions")==expected,"D-Spark retained window differs")
        coverage=need(d,"coverage_by_target");events=need(d,"collection_events")
        for target in targets:
            require(need(coverage,str(target))==len(expected),"D-Spark target coverage count missing")
            seen=[]
            for event in events:
                if event.get("layer")!=target:continue
                a,z=need(event,"start"),need(event,"end")
                require(need(event,"rows")==z-a and a<z,"D-Spark target event shape")
                seen.extend(range(a,z))
            require(seen==expected,"D-Spark target interval gap/overlap")
        require(need(d,"native_main_cache_update_calls")==1 and need(d,"state_id_verified_on_device")is True
            and need(d,"fingerprint_before_decode")is True
            and need(d,"sampler_placeholder_features_used")is False,"D-Spark cache initialization evidence differs")
        return dict(target_layer_ids=targets,retained_rows=len(expected),all_target_features_recorded=True,
            native_main_cache_update_calls=1,features_are_attention_inputs_not_layer39_ffn_outputs=True,
            feature_tensor_fingerprints_present=d.get("qualification_fingerprints")is not None,
            same_policy_reference_exact=d.get("same_policy_reference_exact"),
            full_model_equivalence_claim=False)
    checks["dspark_tail"]=checked(dflash)

    if variant.get("enabled")is True:
        checks["all_input_layer39_reprojection"]=audit_projection_evidence(variant.get("attention"),total,selected)
    elif require_all_input:
        checks["all_input_layer39_reprojection"]=dict(status="incomplete",reason="no enabled all-input lastKV variant evidence")
    else:
        checks["all_input_layer39_reprojection"]=dict(status="not_requested",reason="selected attention coverage does not prove all-input projection")
    states=[c["status"]for c in checks.values()]
    status="failed"if"failed"in states else"incomplete"if"incomplete"in states else"passed"
    return seal(dict(schema="index20-last-layer-phase-audit-v1",status=status,
        phase_sha256=phase.get("sha256")if isinstance(phase,dict)else None,
        require_all_input=require_all_input,checks=checks,gpu_calls=0,network_calls=0,
        independent_tensor_recomputation=False,full_context_hidden_recomputed_claim=False,
        evidence_scope="sealed_runtime_phase_reports_not_new_GPU_execution"))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",type=Path,required=True)
    parser.add_argument("--require-all-input",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    raw=args.phase.read_bytes();phase=json.loads(raw)
    result=audit_phase(phase,require_all_input=args.require_all_input)
    result=seal(dict({k:v for k,v in result.items()if k!="sha256"},
        phase_path=str(args.phase.resolve()),phase_file_sha256=hashlib.sha256(raw).hexdigest()))
    text=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if args.output:
        require(args.output.resolve()!=args.phase.resolve(),"audit cannot overwrite raw phase")
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end="")
    return 0 if result["status"]=="passed"else 1


if __name__=="__main__":raise SystemExit(main())
