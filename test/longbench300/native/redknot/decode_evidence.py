"""Read-only D-Spark phase evidence checks; flags alone cannot pass these."""
from kv_cache_engram.core import require, digest


def cache_storage_snapshot(engine):
    """Actual native view addresses, observed without a device payload read."""
    from prefix_decode_contract import tensor_metadata
    cache = engine.driver_worker.cache_engine
    views = {}
    for layer, item in enumerate(cache.gpu_cache.cache_list):
        for field in ('swa_cache','ca_cache','ica_cache','ca_state','ica_state'):
            tensor = getattr(item, field)
            if tensor is not None:
                views[f'{layer}.{field}'] = tensor_metadata(tensor)
    require(len(cache.gpu_cache.cache_list) > 40 and views,
            'actual main plus DFlash native cache views required')
    value = dict(schema='s2-dspark-native-cache-storage-v1',
        layer_count=len(cache.gpu_cache.cache_list), views=views,
        gpu_blocks=tensor_metadata(cache.gpu_blocks), gpu_states=tensor_metadata(cache.gpu_states),
        host_metadata_only=True, baseline='after_native_graph_capture_before_custom_controller')
    return dict(value,sha256=digest(value))


def validate_decode_phase(phase, *, require_real_decode=True):
    reports = phase.get('decode_graph_evidence')
    require(type(reports) is list and len(reports) == 8,
            'all eight actual D-Spark graph observations required')
    for rank, report in enumerate(reports):
        require(type(report) is dict and report.get('rank') == rank
            and report.get('profile') == 'dspark' and report.get('includes_native_idle') is True,
            'D-Spark graph evidence rank/profile differs')
        for name in ('main', 'dflash'):
            before, after = report['before'][name], report['after'][name]
            delta = report['successful_replay_deltas'][name]
            require(before.get('runner_present') is True and after.get('runner_present') is True
                and before.get('captured_graph_sizes') and
                before['captured_graph_sizes'] == after['captured_graph_sizes']
                and type(delta) is int and delta >= 0
                and delta == after['successful_replay_calls']-before['successful_replay_calls'],
                'actual captured graph/replay observations differ')
    require(phase.get('native_idle_bypass_installed') is False,
            'MTP0 idle bypass is not admitted for D-Spark')
    storage = phase.get('native_cache_storage_evidence')
    require(type(storage) is list and len(storage) == 8, 'all eight actual native cache addresses required')
    for rank, item in enumerate(storage):
        require(item.get('rank') == rank and item.get('native_cache_storage_unchanged') is True
            and item.get('baseline_sha256') == item.get('before_sha256') == item.get('after_sha256'),
            'native cache view replaced after graph capture')
    scheduled = phase.get('scheduled_rows_reports')
    require(type(scheduled) is list and len(scheduled) == 8,
            'actual all-rank native scheduled rows required')
    if require_real_decode:
        require(scheduled[0].get('decode_steps', 0) > 0
            and len(phase['record']['output_ids']) > 1
            and all(reports[0]['successful_replay_deltas'][name] > 0 for name in ('main','dflash')),
            'qualification must execute real native decode and both graph runners')
    return dict(actual_native_decode_steps=scheduled[0]['decode_steps'],
        successful_main_replays=reports[0]['successful_replay_deltas']['main'],
        successful_dflash_replays=reports[0]['successful_replay_deltas']['dflash'],
        speculative_acceptance_rate=None, per_step_acceptance_equivalence_claim=False,
        native_idle_bypass_installed=False)
