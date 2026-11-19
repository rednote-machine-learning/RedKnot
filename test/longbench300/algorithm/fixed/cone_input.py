"""Compact live S1 attn-norm and primary Q_a+SWA projection. Restore S-width for native consumers."""
from contextlib import contextmanager
from copy import copy
import hashlib
from types import MethodType

LAZY = ('residual','post_layer_mix','comb_res_mix','pre_layer_mix')


def require(ok,message):
    if not ok:
        raise ValueError(message)


def pad_tensor(x, ids, count, axis=0, torch=None):
    shape=list(x.shape)
    require(shape[axis]==ids.numel(),'Compact tensor does not match its source indices')
    shape[axis]=count
    out=x.new_zeros(shape)
    if torch is not None and x.dtype==torch.float8_e4m3fn:
        require(axis==0 and x.stride(-1)==1,'Unexpected FP8 data layout')
        out.view(torch.uint8).index_copy_(axis,ids,x.view(torch.uint8))
    else:
        out.index_copy_(axis,ids,x)
    return out


def pad_packed(t, packed, ids, count):
    require(isinstance(packed,tuple) and len(packed)==2,'Expected native MXFP8 attention input')
    raw,sf=packed
    require(raw.dtype==t.float8_e4m3fn and raw.ndim==2 and sf.dtype==t.int32 and
            sf.shape==(raw.shape[0],raw.shape[1]//128),'Unexpected native packed data or scales')
    full_raw=pad_tensor(raw,ids,count,torch=t)
    aligned=(count+3)//4*4
    scales=sf.new_zeros((sf.shape[1],aligned)).transpose(0,1)[:count]
    scales.index_copy_(0,ids,sf)
    return full_raw,scales


def lazy_subset(bundle, ids):
    require(bundle is not None,'Missing native mHC state')
    result=copy(bundle)
    for field in LAZY:
        value=getattr(bundle,field)
        if value is not None:
            require(value.ndim>=2 and value.shape[0]==1,'Unexpected native mHC axis')
            value=value.index_select(1,ids)
        setattr(result,field,value)
    return result


def lazy_pad(bundle, ids, count):
    out=copy(bundle)
    for name in LAZY:
        src=getattr(bundle,name)
        setattr(out,name,None if src is None else pad_tensor(src,ids,count,axis=1))
    return out


def compare_rows(t, dense, ids, compact, axis, label, evidence):
    if dense is None or compact is None:
        require(dense is None and compact is None,'Native field presence changed: '+label)
        evidence['fields'][label]=dict(present=False,rows=0,byte_equal=True,reference_sha256=None,
            candidate_sha256=None,different_rows=0)
        return
    require(dense.dtype==compact.dtype and dense.device==compact.device,'Native dtype/device changed: '+label)
    if dense.dtype==t.float8_e4m3fn:
        dense=dense.contiguous().view(t.uint8)
        compact=compact.contiguous().view(t.uint8)
    subset=dense.index_select(axis,ids)
    require(tuple(subset.shape)==tuple(compact.shape),'Native retained shape changed: '+label)
    expected=subset.contiguous().view(t.uint8)
    actual=compact.contiguous().view(t.uint8)
    equal=bool(t.equal(expected,actual))
    # Digest the exact retained byte sequence. If GPU equality has been established, one host
    # copy suffices. Only qualification enables this path; timed model calls do no host reads.
    import numpy as np
    reference=expected.detach().cpu().numpy()
    refhash=hashlib.sha256(memoryview(reference)).hexdigest()
    acthash=refhash
    different=0
    if not equal:
        new=actual.detach().cpu().numpy()
        acthash=hashlib.sha256(memoryview(new)).hexdigest()
        left=np.moveaxis(reference,axis,0)
        right=np.moveaxis(new,axis,0)
        for start in range(0,len(left),64):
            chunk=left[start:start+64]!=right[start:start+64]
            different+=int(chunk.reshape(len(chunk),-1).any(axis=1).sum())
        evidence['different_fields'].append(label)
    evidence['fields'][label]=dict(present=True,rows=ids.numel(),byte_equal=equal,
        reference_sha256=refhash,candidate_sha256=acthash,different_rows=different)
    evidence['checked_fields']+=1
    evidence['all_equal']=not evidence['different_fields']


def compare_norm(t, dense, ids, compact, evidence):
    dp, db, dl=dense
    cp, cb, cl=compact
    require(isinstance(dp,tuple) and isinstance(cp,tuple) and len(dp)==len(cp)==2,'Expected native packed scales')
    for label,d,c in [('fp8',dp[0],cp[0]),('sf',dp[1],cp[1]),('bf16',db,cb)]:
        compare_rows(t,d,ids,c,0,label,evidence)
    for field in LAZY:
        compare_rows(t,getattr(dl,field),ids,getattr(cl,field),1,field,evidence)


@contextmanager
def compact_inputs(controller,layer,group,local_input_ids,qualify,evidence):
    mod=layer.self_attn
    idx=layer.layer_idx
    mode=controller._fixed_pack_method
    local=tuple(local_input_ids)
    selected=tuple(group.positions_cpu)
    n=len(selected); k=len(local)
    require(mode in ('Cone15','Fixed15Opt') and 21<=idx<=39 and mod.layer_idx==idx and
            all(type(i)is int and 0<=i<n for i in local) and
            all(a<b for a,b in zip(local,local[1:])), 'Invalid layer or compact input mapping')
    require(layer.hc_enabled and layer.engram is None and mod.use_swa and not mod.yoco_swa_share_hidden
            and not mod.is_kv_source and not mod.is_index_k_source,
            'Post20 input compaction requires native consumer topology and row-local mHC')
    candidate=mode=='Cone15'
    # All 12 measured requests have positive I in every nonempty group. For a different shape with no
    # live row, do not synthesize an empty native mHC format: retain native work and record its actual count.
    slim=candidate and 0<k<n
    check=candidate and qualify and slim
    stats=dict(layer=idx,selected_axis_rows=n,live_input_rows=k,actual_attn_norm_rows=0,actual_q_a_swa_rows=0,
               attn_norm_calls=0,q_a_swa_calls=0,norm_checked_rows=0,q_a_swa_checked_rows=0,
               dense_oracle_norm_rows=0,dense_oracle_q_a_swa_rows=0,full_replay_bypass_calls=0,
               full_replay_bypass_rows=0,empty_live_uses_native=bool(candidate and n and not k),
               qualification_comparison_enabled=check,native_passthrough=not slim,
               input_equivalence=dict(checked_fields=0,different_fields=[],all_equal=None,fields={}))
    evidence['inputs']=stats
    # Empty owners have no call to either native primary operator.
    if n==0:
        yield
        return
    t=controller.t
    ids=controller._cone_device_ids[local] if slim else None
    if slim:
        require(ids.dtype==t.long and ids.ndim==1 and ids.numel()==k and
                ids.device==group.compact_meta.positions.device,'Input indices differ from the native owner')
    norm=layer.attn_norm_forward
    project=mod._project_q_and_swa
    state={}
    def attn_norm(_,hidden,residual,hc_post_intervention=None):
        require(stats['attn_norm_calls']==0 and hidden.ndim in (2,3) and hidden.shape[0]==n
                and hc_post_intervention is None,'Unexpected attention norm call site')
        stats['attn_norm_calls']=1
        if not slim:
            out=norm(hidden,residual,hc_post_intervention)
            state['returned']=out[0]
            stats['actual_attn_norm_rows']=n
            return out
        # Native mHC clears its input lazy bundle. Isolate the compact bundle, and preserve only the
        # dense reference bundle for qualification before releasing the original consumed fields.
        narrow=lazy_subset(residual,ids)
        oracle_input=copy(residual) if check else None
        # Native mHC consumes this object. The compact and optional oracle bundles now own
        # the required data, so release the parent references before allocating native outputs.
        residual.clear()
        compact=norm(hidden.index_select(0,ids),narrow,hc_post_intervention)
        stats['actual_attn_norm_rows']=k
        if check:
            dense=norm(hidden,oracle_input,hc_post_intervention)
            with controller.allow_sync():
                compare_norm(t,dense,ids,compact,stats['input_equivalence'])
            stats.update(norm_checked_rows=k,dense_oracle_norm_rows=n)
            state['oracle_packed']=dense[0]
            del dense
        p,b,l=compact
        padded=pad_packed(t,p,ids,n)
        state.update(compact=p,returned=padded)
        return padded,pad_tensor(b,ids,n),lazy_pad(l,ids,n)

    def projection(_,packed,shared):
        if packed is not state.get('returned'):
            # Extra last-layer full-input projection calls use fresh materializer-owned input objects.
            # A native packed object can have the same number of rows as S; identity distinguishes it.
            require(idx==39 and stats['q_a_swa_calls']==1,'Unbound primary Q_a/SWA projection')
            result=project(packed,shared)
            stats['full_replay_bypass_calls']+=1
            stats['full_replay_bypass_rows']+=packed[0].shape[0]
            return result
        require(stats['q_a_swa_calls']==0 and shared is None,'Duplicate/shared Q_a/SWA projection')
        stats['q_a_swa_calls']=1
        del state['returned']  # Other layers in this group's ExitStack must not pin consumed GPU buffers.
        if not slim:
            stats['actual_q_a_swa_rows']=n
            return project(packed,shared)
        small_q,small_k=project(state.pop('compact'),shared)
        require(small_q.shape[0]==small_k.shape[0]==k,'Native Q_a/SWA changed live axis')
        stats['actual_q_a_swa_rows']=k
        if check:
            dense_q,dense_k=project(state.pop('oracle_packed'),shared)
            with controller.allow_sync():
                compare_rows(t,dense_q,ids,small_q,0,'q_a',stats['input_equivalence'])
                compare_rows(t,dense_k,ids,small_k,0,'swa',stats['input_equivalence'])
            stats.update(q_a_swa_checked_rows=k,dense_oracle_q_a_swa_rows=n)
        return pad_tensor(small_q,ids,n),pad_tensor(small_k,ids,n)

    saved=[]
    try:
        for module,name,fn in ((layer,'attn_norm_forward',attn_norm),(mod,'_project_q_and_swa',projection)):
            bound=MethodType(fn,module)
            saved.append((module,name,name in vars(module),vars(module).get(name),bound))
            setattr(module,name,bound)
        yield
        require(stats['attn_norm_calls']==stats['q_a_swa_calls']==1 and
                'compact' not in state and 'oracle_packed' not in state,'Incomplete real native input path')
    finally:
        state.clear()
        for module,name,was_present,old,bound in reversed(saved):
            require(vars(module).get(name) is bound,'Input hook ownership changed')
            if was_present:
                setattr(module,name,old)
            else:
                delattr(module,name)
