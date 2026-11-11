"""Structural checks for the independent Prefix(copy) + D-Spark adapter.

These checks do not install hooks, copy GPU state, or establish numerical
equivalence. Their inputs must come from actual native descriptors/layouts.
No numerical-equivalence claim follows from these host-side checks.
"""
from __future__ import annotations


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compact_prefix_compatible(prefix_tokens, source_prompt_tokens, target_prompt_tokens):
    """Native compact140 consumer masks depend on the complete prompt length.

    Before the target cutoff its consumer tail will be rebuilt by the suffix.
    Beyond that cutoff a source snapshot requires the identical cutoff.
    """
    return (type(source_prompt_tokens) is int and source_prompt_tokens >= prefix_tokens+2
        and (prefix_tokens <= target_prompt_tokens-140 or source_prompt_tokens == target_prompt_tokens))


def validate_fresh_prefix_resume(prompt_ids, prefix_ids, checkpoint, before, after):
    """Check the allocated fresh sequence immediately around in-place restore.

before/after are read-only native descriptor dictionaries. No old MtpState or
MTPInfoPool entry may be transferred. The native suffix path remains mandatory.
"""
    n = len(prefix_ids)
    require(n >= 128 and n % 128 == 0 and n <= len(prompt_ids)-2,
        'Only aligned prompt prefixes with at least two fresh suffix tokens are supported')
    require(tuple(prompt_ids[:n]) == tuple(prefix_ids), 'Actual prefix token IDs differ')
    require(compact_prefix_compatible(n,checkpoint['source_prompt_tokens'],len(prompt_ids)),
        'Native compact140 consumer cutoff differs for the restored near-tail prefix')
    require(checkpoint['is_prompt'] is True and checkpoint['is_placeholder'] is False
        and checkpoint['seqlen_q'] > 0
        and checkpoint['seqlen_k'] == checkpoint['committed_num_cached_tokens'] == n,
        'Snapshot is not a completed native prompt checkpoint')
    require(before['request_id'] == after['request_id']
        and before['state_id'] == after['state_id'] and before['state_id'] > 0
        and before['block_table'] == after['block_table'],
        'Restoration changed native allocation ownership')
    require(before['num_cached_tokens'] == before['previous_num_tokens'] == 0
        and after['num_cached_tokens'] == after['previous_num_tokens'] == n,
        'Native cache commit/rollback baseline was not established by its setter')
    require(after['swa_valid_start'] == checkpoint['swa_valid_start']
        and 0 <= after['swa_valid_start'] <= n, 'Restored SWA valid boundary differs')
    require(after['block_size'] > 0
        and len(after['block_table']) >= (n+after['block_size']-1)//after['block_size'],
        'Prefix exceeds the destination block table')
    for desc in (before, after):
        require(desc['prompt_tokens'] == len(prompt_ids) and desc['output_tokens'] == 0
            and desc['schedule_status'] == 'WAITING', 'Resume must use an unexecuted fresh request')
        require(desc['mtp_info_slot'] == 0 and desc['mtp_has_draft_info'] is False
            and desc['mtp_last_output_token'] == prompt_ids[-1],
            'Fresh native proposal state was replaced or already published')
        for key in ('mtp_accepted_count','mtp_draft_count','mtp_accept_rate_count',
                    'mtp_prev_accept_rate_count','mtp_confidence_count','mtp_prev_confidence_count',
                    'mtp_process_count','mtp_schedule_count','total_inflight_tokens','last_inflight_tokens'):
            require(desc[key] == 0, 'Unexpected prior request state: '+key)
    require(before['last_copy_state_token_idx'] == after['last_copy_state_token_idx'] == 0,
        'This minimal contract does not restore a separate KV-store copy_state checkpoint')
    return dict(prefix_tokens=n, native_suffix_tokens=len(prompt_ids)-n,
        source_prompt_tokens=checkpoint['source_prompt_tokens'],target_prompt_tokens=len(prompt_ids),
        compact_keep_tokens=140,compact_cutoff_compatible=True,
        proposal_policy='fresh_native_slot_then_native_suffix_and_decode',
        old_proposal_pool_copied=False, numerical_qualification=False)


def validate_dflash_swa_coverage(raw_states, views, expected_layer_ids, *, window_tokens, backtrack_tokens):
    """Check actual tensor metadata without reading device contents.

raw_states: storage_ptr, storage_bytes, shape [slots, bytes], stride_bytes.
views: layer_id, storage_ptr, byte_offset, element_bytes, shape, stride_bytes.
Offsets are measured from the backing storage base, not tensor.data_ptr().
"""
    slots, width = raw_states['shape']
    require(slots > 1 and width > 0 and raw_states['stride_bytes'] == [width, 1]
        and raw_states['storage_bytes'] >= slots*width, 'Expected complete contiguous native raw state slots')
    require(backtrack_tokens >= 6 and window_tokens > 0, 'MTP5 requires the native rollback ring capacity')
    expected = set(expected_layer_ids)
    require(expected and len(expected) == len(expected_layer_ids), 'Expected actual unique DFlash layer IDs')
    require(len(views) == len(expected) and {v['layer_id'] for v in views} == expected,
        'Missing, duplicate, or foreign DFlash SWA layer view')
    for view in views:
        shape, strides = view['shape'], view['stride_bytes']
        require(len(shape) >= 2 and len(shape) == len(strides) and all(x > 0 for x in shape)
            and all(x > 0 for x in strides), 'Invalid SWA view geometry')
        require(view['storage_ptr'] == raw_states['storage_ptr']
            and shape[0] == slots and strides[0] == width,
            'DFlash SWA is not in the raw slot storage being copied')
        require(shape[1] >= window_tokens+backtrack_tokens, 'DFlash ring omits rollback history')
        end = view['byte_offset']+sum((size-1)*stride for size,stride in zip(shape[1:],strides[1:]))+view['element_bytes']
        require(view['element_bytes'] > 0 and 0 <= view['byte_offset'] < end <= width,
            'A DFlash SWA view crosses the complete copied state-slot boundary')
    return dict(dflash_layer_ids=sorted(expected), whole_state_slot_bytes=width,
        structural_swa_coverage=True, device_bytes_verified=False, numerical_qualification=False)


def validate_stable_graph_storage(before, after):
    """Caller inventories *all* captured native buffers, never bank snapshots."""
    required = {'gpu_blocks','gpu_states','mtp_info_pool.token_ids','mtp_info_pool.logprobs'}
    require(required <= set(before) and set(before) == set(after), 'Native graph storage inventory differs')
    require(any(k.startswith('main_graph.') for k in before)
        and any(k.startswith('dflash_graph.') for k in before), 'Both captured graph buffer inventories are required')
    require(before == after, 'Restore replaced or resized captured native storage; only in-place copies are admitted')
    return dict(native_storage_unchanged=True, graph_replay_observed=False, numerical_qualification=False)


def tensor_metadata(tensor):
    """Host metadata only; never read a device payload or launch a kernel."""
    item = tensor.element_size()
    storage = tensor.untyped_storage()
    return dict(storage_ptr=storage.data_ptr(), storage_bytes=storage.nbytes(),
        data_ptr=tensor.data_ptr(), byte_offset=tensor.storage_offset()*item,
        element_bytes=item, shape=list(tensor.shape), stride_bytes=[s*item for s in tensor.stride()],
        dtype=str(tensor.dtype), device=str(tensor.device))


def _tensor_leaves(obj, path, result, seen):
    import dataclasses
    if hasattr(obj, 'untyped_storage') and hasattr(obj, 'stride'):
        result[path] = tensor_metadata(obj)
        return
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)) or id(obj) in seen:
        return
    seen.add(id(obj))
    if dataclasses.is_dataclass(obj):
        children = [(f.name, getattr(obj,f.name)) for f in dataclasses.fields(obj)]
    elif isinstance(obj, dict):
        children = sorted(obj.items(), key=lambda pair:str(pair[0]))
    elif isinstance(obj, (tuple,list)):
        children = enumerate(obj)
    else:
        return
    for name, value in children:
        _tensor_leaves(value, path+'.'+str(name), result, seen)


def native_graph_storage(engine):
    """Inventory actual captured inputs/outputs plus immutable buffer addresses.

    Bank tensors are deliberately absent: only live native engine storage matters.
    """
    ce = engine.driver_worker.cache_engine
    mr = engine.driver_worker.model_runner
    result = {name:tensor_metadata(getattr(ce,name)) for name in ('gpu_blocks','gpu_states')}
    mtp = mr.model.MTP_module
    pool = mtp.mtp_info_pool
    require(pool is not None and pool.nextn == 5, 'Actual native MTP5 info pool required')
    for name in ('token_ids','logprobs','confidence','logprobs_index'):
        value = getattr(pool,name)
        if value is not None:
            result['mtp_info_pool.'+name] = tensor_metadata(value)
    for prefix, attr in (('main_graph','cuda_graph_runner'),('dflash_graph','dflash_cuda_graph_runner')):
        runner = getattr(mr,attr,None)
        require(runner is not None and runner.graphs, 'Both native graph runners must have captured graphs')
        for size, data in sorted(runner.graphs.items()):
            require(data.graph is not None, 'Missing native captured graph')
            # Include graph identity as well as buffers; graph recreation is not an in-place restore.
            result[f'{prefix}.{size}.graph_identity'] = id(data.graph)
            fields = ('output','input_metadata_source','input_metadata_device') if prefix == 'main_graph' else ('output','decode_input')
            for field in fields:
                _tensor_leaves(getattr(data,field), f'{prefix}.{size}.{field}', result, set())
        require(any(k.startswith(prefix+'.') and isinstance(v,dict) for k,v in result.items()),
                'Captured graph tensor inventory is empty')
    return result


def inspect_native_decode_storage(engine):
    """Prove every native cache view fits the raw row/block copied by the adapter."""
    ce = engine.driver_worker.cache_engine
    cfg = ce.cfg
    main_layers = cfg.num_hidden_layers
    modules = cfg.attn_module_list
    expected = [i-1 for i in range(main_layers+1,len(modules)) if modules[i] is not None]
    require(main_layers == 40 and expected, 'Actual native main40 plus DFlash attention layers required')
    raw_state,raw_blocks = tensor_metadata(ce.gpu_states),tensor_metadata(ce.gpu_blocks)
    views=[]
    all_views=[]
    require(len(ce.gpu_cache.cache_list) == main_layers+len(expected), 'Native cache layer coverage differs')
    for layer_id, cache in enumerate(ce.gpu_cache.cache_list):
        for field in ('ica_cache','ca_cache','ica_state','ca_state','swa_cache'):
            tensor=getattr(cache,field)
            if tensor is None:
                continue
            meta=dict(tensor_metadata(tensor),layer_id=layer_id,field=field)
            raw = raw_blocks if field in ('ica_cache','ca_cache') else raw_state
            width=raw['shape'][1]
            end=meta['byte_offset']+sum((size-1)*stride for size,stride in
                zip(meta['shape'][1:],meta['stride_bytes'][1:]))+meta['element_bytes']
            require(meta['storage_ptr'] == raw['storage_ptr'] and meta['shape'][0] == raw['shape'][0]
                and meta['stride_bytes'][0] == width and 0 <= meta['byte_offset'] < end <= width,
                'Native cache view is not fully covered by its copied raw row: '+str((layer_id,field)))
            all_views.append(meta)
            if layer_id in expected and field == 'swa_cache':
                views.append(meta)
    coverage=validate_dflash_swa_coverage(raw_state,views,expected,window_tokens=cfg.window_size[0],backtrack_tokens=6)
    inventory=native_graph_storage(engine)
    validate_stable_graph_storage(inventory,inventory)
    return dict(schema='s2-prefix-dspark-storage-contract-v1',raw_states=raw_state,raw_blocks=raw_blocks,
        all_native_cache_views=all_views,dflash_swa_views=views,dflash_swa_coverage=coverage,
        graph_storage=inventory,structural_only=True,numerical_qualification=False,
        old_proposal_pool_copied=False)
