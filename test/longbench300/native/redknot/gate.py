"""Validate D-Spark final-window feature and KV fingerprints."""

from kv_cache_engram.core import digest, require, validate_digest

def _fingerprint(value):
    if value is None:
        return
    require(type(value) is dict and set(value) == {"shape", "dtype", "bytes", "sha256"},
            "raw activation fingerprint missing fields")
    shape = value["shape"]
    require(type(shape) is list and bool(shape) and all(type(n) is int and n >= 0 for n in shape)
            and type(value["dtype"]) is str and type(value["bytes"]) is int and value["bytes"] >= 0,
            "invalid activation geometry fingerprint")
    validate_digest(value["sha256"], "activation bytes")

def verify_dflash_tail_pair(reference, candidate, *, input_ids_sha256, input_tokens):
    tails = [report.get('dflash_tail') for report in (reference, candidate)]
    positions = list(range(max(0, input_tokens-128), input_tokens))
    for index, tail in enumerate(tails):
        require(type(tail) is dict and tail.get('schema') == 's2-dflash-tail-runtime-v1'
            and tail.get('enabled') is True and tail.get('integrated_all_rank_runtime') is True
            and tail.get('input_ids_sha256') == input_ids_sha256
            and tail.get('input_tokens') == input_tokens and tail.get('window_size') == 128
            and tail.get('retained_positions') == positions
            and tail.get('native_main_cache_update_calls') == 1
            and tail.get('selected_positions_unchanged') is True
            and tail.get('sampler_placeholder_features_used') is False
            and tail.get('fingerprint_before_decode') is True
            and tail.get('same_policy_reference_exact') is bool(index),
            'actual S2 DFlash final-window initialization evidence differs')
        targets = tail.get('target_layer_ids')
        require(type(targets) is list and targets and len(set(targets)) == len(targets)
            and all(type(i) is int and 0 <= i < 40 for i in targets)
            and tail.get('coverage_by_target') == {str(k):len(positions) for k in targets}
            and type(tail.get('state_id')) is int and tail['state_id'] > 0,
            'complete real-row DFlash target/state coverage required')
        fingerprints = tail.get('qualification_fingerprints')
        require(type(fingerprints) is dict and set(fingerprints) == {'target_features','valid_main_swa'},
                'actual DFlash input and main-cache fingerprints required')
        _fingerprint(fingerprints['target_features'])
        require(fingerprints['target_features']['shape'][0] == len(positions), 'DFlash tail row axis differs')
        caches = fingerprints['valid_main_swa']
        require(type(caches) is list and caches
            and [item['layer'] for item in caches] == tail.get('native_dflash_layer_ids')
            and len(set(item['layer'] for item in caches)) == len(caches), 'all native DFlash layers required')
        for item in caches:
            _fingerprint(item['value'])
            require(item['value']['shape'][0] == len(positions), 'DFlash valid SWA row axis differs')
    for key in ('target_layer_ids','native_dflash_layer_ids','retained_positions','qualification_fingerprints'):
        require(tails[0][key] == tails[1][key], 'DFlash reference/candidate byte identity failed: '+key)
    return dict(retained_rows=len(positions), target_layer_ids=tails[0]['target_layer_ids'],
        native_dflash_layer_ids=tails[0]['native_dflash_layer_ids'],
        fingerprints_sha256=digest(tails[0]['qualification_fingerprints']),
        actual_prefill_tail_exact=True, complete_future_decode_equivalence_claim=False)
