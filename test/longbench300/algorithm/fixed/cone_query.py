"""Run the SM90 sparse attention and output projection on the live query axis.

The original input projection, index generation, global memory, causal mask and
ring writes surround these two native calls. Their absolute indices stay intact.
"""
from contextlib import contextmanager
from copy import copy
from hashlib import sha256
from types import MethodType


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fingerprint(t, tensor):
    raw=tensor.detach().contiguous().view(t.uint8).cpu().numpy().tobytes()
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=sha256(raw).hexdigest())


def query_view(mod, metadata, ids, positions):
    require(metadata.is_prompt and metadata.num_reqs==1 and
            all(getattr(metadata,k,None) is None for k in
                ('redknot_binding','attention_range_binding','head_intervention_binding')),
            'Unexpected consumer of full query metadata in live attention')
    # _prefill_attention uses only positions and (for dense global attention)
    # kv_ca_index_len. The three opt-in observer paths are excluded above.
    view=copy(metadata)
    view.total_q=ids.numel()
    view.positions=positions
    if mod.has_packed_kv and not mod.is_sparse:
        lengths=metadata.kv_ca_index_len
        require(mod.ratio_key in lengths and lengths[mod.ratio_key].shape[0]==metadata.total_q,
                'Invalid dense global query lengths')
        view.kv_ca_index_len={**lengths,mod.ratio_key:lengths[mod.ratio_key].index_select(0,ids)}
    return view


def gather_queries(tensor, ids, count):
    if tensor is None:
        return None
    require(tensor.shape[0]==count,'Mismatched native attention query axis')
    return tensor.index_select(0,ids)


@contextmanager
def compact_core(controller, layer, group, local_ids, qualify, evidence):
    mod=layer.self_attn
    attn,project,q_b=mod._prefill_attention,mod._wv_and_o_proj,mod._project_q_b
    fields=('_prefill_attention','_wv_and_o_proj','_project_q_b')
    original={k:vars(mod).get(k) for k in fields}
    had={k:k in vars(mod) for k in fields}
    local=tuple(local_ids)
    n=len(group.positions_cpu)
    nlive=len(local)
    reference=controller._fixed_pack_method=='Fixed15Opt'
    require(controller._fixed_pack_method in ('Fixed15Opt','Cone15') and
            mod.layer_idx==layer.layer_idx and 21<=mod.layer_idx<=39 and
            all(0<=v<n for v in local) and all(l<r for l,r in zip(local,local[1:])),
            'Invalid attention cone group')
    receipt=dict(layer=mod.layer_idx,selected_projection_rows=n,input_axis_rows=n,live_query_rows=nlive,actual_q_b_rows=0,q_b_calls=0,
        actual_attention_rows=0,actual_output_projection_rows=0,
        core_calls=0,output_projection_calls=0,retained_output=None)
    evidence['core']=receipt
    slim=(not reference and nlive<n)
    active=None
    seen=False
    ids=None
    positions=None
    actual=0
    t=controller.t
    if slim or qualify:
        ids=controller._cone_device_ids[local]
        require(ids.numel()==nlive and ids.device==group.compact_meta.positions.device and ids.dtype==t.long,
                'Live query indices do not belong to this group')
    seen_q=False
    def query_projection(_,q_a,full_pos):
        nonlocal seen_q,positions
        require(not seen_q and full_pos is group.compact_meta.positions and q_a.shape[0]==n,
                'Q_b projection must consume the selected query axis once')
        seen_q=True
        if not slim:
            query,index_q=q_b(q_a,full_pos)
            receipt.update(actual_q_b_rows=n,q_b_calls=int(n>0))
            return query,index_q
        positions=full_pos.index_select(0,ids)
        if not nlive:
            query=q_a.new_empty((0,mod.num_heads,mod.qk_head_dim))
            index_q=q_a.new_zeros((n,mod.index_n_head*mod.index_head_dim)) if mod.has_local_index else None
        else:
            query,index_q=q_b(q_a.index_select(0,ids),positions)
            if index_q is not None:
                # The native indexer and its inherited positions retain the S axis.
                # Every inheriting consumer below is a subset of this layer's Q.
                full_index=index_q.new_zeros((n,*index_q.shape[1:]))
                full_index.index_copy_(0,ids,index_q)
                index_q=full_index
        receipt.update(actual_q_b_rows=nlive,q_b_calls=int(nlive>0))
        return query,index_q

    def core(_,q,kv,global_index,local_index,cached,current,meta):
        nonlocal positions,active,seen,actual
        require(seen_q and not seen and meta is group.compact_meta and meta.total_q==n and
                q.shape[0]==(nlive if slim else n) and meta.positions.shape[0]==n,
                'Unbound or repeated query contraction')
        seen=True
        if slim:
            view=query_view(mod,meta,ids,positions)
            if nlive:
                active=attn(q,kv,gather_queries(global_index,ids,n),
                            gather_queries(local_index,ids,n),cached,current,view)
            else:
                active=object()
            actual=nlive
        else:
            positions=meta.positions
            active=attn(q,kv,global_index,local_index,cached,current,meta)
            actual=n
        receipt.update(actual_attention_rows=actual,core_calls=int(actual>0))
        return active

    def output(_,context,full_positions,**kw):
        nonlocal active
        require(seen and active is context and full_positions is group.compact_meta.positions,
                'Native output projection received a different query or context')
        active=None
        if actual:
            value=project(context,positions,**kw)
            require(value.ndim==2 and value.shape[0]==actual and value.dtype==t.bfloat16,
                    'Native attention output layout changed')
        else:
            value=t.empty((0,mod.out_proj.output_size),device=positions.device,dtype=t.bfloat16)
        if qualify:
            with controller.allow_sync():
                observed=value.index_select(0,ids) if reference else value
                receipt['retained_output']=fingerprint(t,observed)
        receipt.update(actual_output_projection_rows=actual,output_projection_calls=int(actual>0))
        if not slim:
            return value
        result=value.new_zeros((n,value.shape[1]))
        if nlive:
            result.index_copy_(0,ids,value)
        return result
    hooks={fields[0]:MethodType(core,mod),fields[1]:MethodType(output,mod),fields[2]:MethodType(query_projection,mod)}
    for k,v in hooks.items():setattr(mod,k,v)
    try:
        yield
        if not n and qualify:
            with controller.allow_sync():
                z=t.empty((0,mod.out_proj.output_size),device=group.compact_meta.positions.device,dtype=t.bfloat16)
                receipt['retained_output']=fingerprint(t,z)
        require(seen_q==seen==bool(n) and active is None,'Missing attention readout for group')
        validate_core(receipt,mod.layer_idx,n,nlive,reference,qualify)
    finally:
        for k in fields:
            require(vars(mod).get(k) is hooks[k],'Private attention hook ownership changed')
            if had[k]:setattr(mod,k,original[k])
            else:delattr(mod,k)


def validate_core(proof,layer,selected_count,live_count,reference,qualify):
    actual=selected_count if reference else live_count
    require(proof['layer']==layer and proof['selected_projection_rows']==selected_count and
            proof['live_query_rows']==live_count and
            proof['input_axis_rows']==selected_count and proof['actual_q_b_rows']==actual and
            proof['actual_attention_rows']==proof['actual_output_projection_rows']==actual and
            proof['q_b_calls']==proof['core_calls']==proof['output_projection_calls']==int(actual>0),
            'Physical attention or output GEMM row count differs')
    receipt=proof['retained_output']
    require((receipt is not None)==qualify,'Unexpected attention readback outside qualification')
    if qualify:
        require(receipt['shape'][0]==live_count and len(receipt['sha256'])==64,'Wrong retained attention fingerprint')
    return proof
