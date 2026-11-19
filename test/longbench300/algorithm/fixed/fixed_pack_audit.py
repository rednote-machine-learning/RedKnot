"""CPU readers for Cone15 fixed15 fusion; neither synthetic nor progressive evidence qualifies."""
import math
from fixed_pack_controller import require, digest, seal
from fixed_pack_policy import ARM_FRACTIONS


def checked(value):
    require(isinstance(value, dict) and value.get('sha256') ==
        digest({k: v for k, v in value.items() if k != 'sha256'}), 'Evidence seal differs')
    return value



def require_actual_fusion(pack):
    """Admit an optimized benchmark only after actual gather AND scatter calls.

    A one-layer tail fallback may coexist with real fused windows. A wholly
    delegated phase is unsupported for this fusion experiment, never a pass.
    This is an execution gate, not evidence of speedup or answer quality.
    """
    scatter, gather = pack['actual_fused_kernel_launches'], pack['actual_fused_bank_gather_launches']
    require(type(scatter) is int and type(gather) is int and scatter >= 0 and gather >= 0,
        'Actual fused launch counters must be nonnegative integers')
    require(scatter == sum(e['scatter_data_kernel_launches'] for e in pack['leases'])
        and gather == sum(e['gather_data_kernel_launches'] for e in pack['document_bank_gathers']),
        'Fused launch counters differ from actual recorded leases')
    require(scatter > 0 and gather > 0,
        'Unsupported optimized execution: no actual fused bank-gather and memory-scatter; '
        'all-fallback or unhooked execution cannot pass as Cone15 fusion')
    return dict(actual_fused_execution_admitted=True,
        actual_fused_bank_gather_launches=gather,actual_fused_memory_scatter_launches=scatter,
        all_fallback_execution=False,
        explicit_fallback_count=len(pack['explicit_single_layer_fallbacks']),
        quality_or_speedup_claim=False)


def validate_fixed_pack_phase(phase, row, *, expected_method):
    require(expected_method in ARM_FRACTIONS and phase.get('method') == expected_method,
        'Cone15 phase canonical identity differs; another arm qualification cannot be borrowed')
    from cone15_audit import validate_lastkv
    native = validate_lastkv(phase, row)
    control, record = phase['controller'], phase['record']
    info = checked(control['system_pack_fixed_v1'])
    require(expected_method in ARM_FRACTIONS, 'Unknown fixed ratio arm')
    fraction = ARM_FRACTIONS[expected_method]
    require(info['schema'] == 'redknot-fixed15-system-pack-v1' and
        info['algorithm_changed_from_fixed20'] is True and info['enabled'] is True and info['canonical_method'] == expected_method and
        info['reference_method'] == 'Full' and info['selector_strategy'] == 'max' and
        info['body_fraction'] == fraction and info['selector_layers'] == [2] and
        info['regroup_enabled'] is False and info['selected_ids_retained_through_layer39'] is True,
        'Phase is not the fixed15 packed arm')
    require(record['input_tokens'] == row['input_tokens'] and
        record['input_ids_sha256'] == row['input_ids_sha256'] and
        control['algorithm'] == 'index20' and 'progressive' not in control,
        'Input or algorithm differs from fixed15')
    selection = checked(control['index20']['selection'])
    selected = selection['selected_positions']
    require(selection['strategy'] == 'max' and selection['fraction'] == fraction and
        selected == sorted(set(selected)) and selection['body_budget_exact'] is True,
        'Original native fixed15 selection is missing')
    body, tails = set(), set()
    for doc in row['documents']:
        a, z = doc['input_start'], doc['input_end']
        if z-a >= 256:
            body.update(range(a, z)); tails.update(range(z-128, z))
    new = set(range(row['input_tokens'])) - body
    require(tails | new <= set(selected) and len(set(selected) & body) == math.ceil(fraction*len(body)),
        'Mandatory rows or fixed body budget differ')
    from cone15_audit import audit_cone_phase
    cone = audit_cone_phase(phase, row)
    events = control['decoder_events']
    require(len(events) == 40*((row['input_tokens']+8191)//8192), 'Missing decoder coverage')
    for layer in range(40):
        wanted = set(range(row['input_tokens'])) if layer < 3 else set(selected)
        seen, cursor = [], 0
        for event in (e for e in events if e['layer'] == layer):
            start, end = event['chunk_start'], event['chunk_end']
            expected = sorted(p for p in wanted if start <= p < end)
            require(start == cursor and end == min(start+8192, row['input_tokens']) and
                event['selected_positions'] == expected and event['selected_rows'] == len(expected), 'Fixed decoder row coverage differs')
            seen.extend(expected); cursor = end
        require(cursor == row['input_tokens'] and seen == sorted(wanted), 'Fixed layer omitted/repeated rows')
    if expected_method in ARM_FRACTIONS:
        require(info['packed_only_replay_rows'] == row['input_tokens'], 'Optimized last replay omitted input rows')
        require(all(info['geometry_calls'].get(k,0)>0 for k in ('chunk','local','offline')), 'Optimized geometry unused')
    else:
        require(info['packed_only_replay_rows'] == 0 and not info['geometry_calls'], 'Optimized code active for reference')
    pack = info['kv_pack']
    require(pack['enabled'] is True and pack['active_lease'] is False and pack['fixed_group_active'] is False
        and pack['copy_windows_are_selection_stages'] is False and pack['algorithm_selection_layers'] == [2],
        'Fixed pack copy windows or lifecycle differ')
    actual_fusion = require_actual_fusion(pack)
    online = pack.get('fixed15_online', {})
    if expected_method in ARM_FRACTIONS:
        require(online.get('kernel_launches', 0) > 0, 'Optimized candidate used no fused online-row kernel')
    else:
        require(online.get('kernel_launches', 0) == 0, 'Reference candidate ran optimized kernel')

    require(all(e['bounded_window_additional_bytes'] <= info['kv_pack_budget_bytes'] for e in pack['leases']),
        'Actual pack exceeded the configured memory budget')
    if phase['qualification_only']:
        checks = checked(info['tensor_checks'])
        require(checks['input_ids_sha256'] == row['input_ids_sha256'] and
            checks['owner_rank'] == phase['owner_rank'] and checks['qualification_only'] is True and
            checks['benchmark_request'] is False and checks['exact_comparison'] is True,
            'Fixed-pack own32 must carry its real owner/input tensor observations')
        observed = checks['kv_checks']
        for entry in observed:
            checked(entry)
            require(entry['exact'] is True and entry['reference'] == entry['actual'],
                'A fused KV byte comparison failed')
        require(sum(x['kind'] == 'normalized_gather' for x in observed) ==
            sum(len(x['layers']) for x in pack['document_bank_gathers']) and
            sum(x['kind'] == 'assembled_memory' for x in observed) ==
            sum(len(x['assembled_layers']) for x in pack['leases']),
            'Own32 byte checks do not cover every actual fused layer')
    else:
        require(info['tensor_checks'] is None, 'Tensor synchronization leaked into benchmark phase')
    groups = control['system_batch_execution']['groups']
    require([(x['start'], x['end'], x['selected_positions']) for x in groups] ==
        [(x['start'], x['end'], x['selected_positions']) for x in pack['fixed_group_events']] and
        all(x['completed'] and x['actual_memory_layers'] == (list(range(3,40)) if x['selected_positions'] else [])
            for x in pack['fixed_group_events']), 'Fixed group-major execution was changed')
    return seal(dict(schema='redknot-fixed15-phase-audit-v1', passed=True,
        canonical_method=expected_method, body_fraction=fraction, cone15=cone,
        input_ids_sha256=row['input_ids_sha256'], input_tokens=row['input_tokens'],
        fixed_selection_sha256=selection['sha256'], native_lastkv=native,
        actual_fused_bank_gather_launches=pack['actual_fused_bank_gather_launches'],
        actual_fused_memory_scatter_launches=pack['actual_fused_kernel_launches'],
        all_fallback_execution=False, actual_fusion=actual_fusion,
        quality_or_speedup_claim=False))

