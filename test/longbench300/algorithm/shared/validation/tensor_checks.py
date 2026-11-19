"""Real tensor checks, enabled only in the existing own32 qualification phases.

No CUDA import or allocation at module import. Callers supply their actual torch
module. Reports are observations, never substitutes for native B4/graph proofs.
Fingerprinting synchronizes and transfers tensor bytes; never enable it in a
timed benchmark phase. Canonical rows are absolute input positions, not groups.
"""
import hashlib
import json

LAZY = ('residual', 'post_layer_mix', 'comb_res_mix', 'pre_layer_mix')
SOURCE_LAYERS = (2, 8, 14, 20)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    require('sha256' not in value, 'Do not reseal an existing evidence object')
    return dict(value, sha256=digest(value))


def checked(value):
    require(isinstance(value, dict) and value.get('sha256') == digest(
        {k: v for k, v in value.items() if k != 'sha256'}), 'Evidence seal differs')
    return value


def tensor_fingerprint(value, torch_module):
    """Hash logical bytes in canonical row order; ignore allocation/stride padding."""
    if value is None:
        return None
    if isinstance(value, tuple):
        require(len(value) == 2, 'Packed hidden requires data and scale')
        return {'packed': [tensor_fingerprint(x, torch_module) for x in value]}
    raw = value.detach().contiguous().view(torch_module.uint8).cpu().numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest())


def _slice(value, start, end, axis):
    if value is None:
        return None
    if isinstance(value, tuple):
        require(axis == 0 and len(value) == 2, 'Packed hidden token axis must be zero')
        return tuple(_slice(v, start, end, axis) for v in value)
    return value.narrow(axis, start, end-start)


def _cat(values, axis, torch_module):
    require(values, 'Missing actual tensor pieces')
    if all(v is None for v in values):
        return None
    require(all(v is not None for v in values), 'Inconsistent absent mHC state')
    if isinstance(values[0], tuple):
        require(axis == 0 and all(isinstance(v, tuple) and len(v) == 2 for v in values),
            'Inconsistent packed hidden state')
        return tuple(_cat([v[i] for v in values], axis, torch_module) for i in range(2))
    require(all(not isinstance(v, tuple) for v in values), 'Mixed packed and dense state')
    first = values[0]
    require(all(v.dtype == first.dtype and v.device == first.device for v in values),
        'Tensor dtype/device changed while regrouping')
    return values[0] if len(values) == 1 else torch_module.cat(values, dim=axis)


def canonical_group_state(groups, *, total, torch_module):
    """All selected hidden/mHC rows, partitioned by ORIGINAL absolute chunks.

    Each returned fingerprint represents at most 8192 rows. Large tensor states
    are released after each chunk; only hashes and exact positions are retained.
    The result is independent of the number of chronological execution groups.
    """
    groups = list(groups)
    require(groups and type(total) is int and total > 0, 'Actual groups and input required')
    positions = [p for g in groups for p in g.positions_cpu]
    require(positions == sorted(set(positions)) and positions and 0 <= positions[0]
        and positions[-1] < total, 'Absolute selected positions are invalid')
    for group in groups:
        require(all(hasattr(group.lazy, k) for k in LAZY), 'All four mHC fields required')
        hidden = group.hidden[0] if isinstance(group.hidden, tuple) else group.hidden
        require(hidden.shape[0] == len(group.positions_cpu), 'Hidden token axis mismatch')
        for field in LAZY:
            value = getattr(group.lazy, field)
            require(value is None or (value.ndim >= 2 and value.shape[0] == 1
                and value.shape[1] == len(group.positions_cpu)), 'mHC token axis mismatch')
    result = []
    # Monotonic group rows make every chunk intersection a contiguous slice.
    import bisect
    for start in range(0, total, 8192):
        end = min(total, start+8192)
        pieces, selected = [], []
        for group in groups:
            p = group.positions_cpu
            a, z = bisect.bisect_left(p, start), bisect.bisect_left(p, end)
            if a < z:
                pieces.append((group, a, z)); selected.extend(p[a:z])
        if not pieces:
            continue
        hidden = _cat([_slice(g.hidden, a, z, 0) for g, a, z in pieces], 0, torch_module)
        value = dict(hidden=tensor_fingerprint(hidden, torch_module), lazy={})
        del hidden
        for field in LAZY:
            state = _cat([_slice(getattr(g.lazy, field), a, z, 1)
                          for g, a, z in pieces], 1, torch_module)
            value['lazy'][field] = tensor_fingerprint(state, torch_module)
            del state
        result.append(dict(chunk_start=start, chunk_end=end,
            selected_positions=selected, state=value))
    return dict(full_input_tokens=total, selected_positions=positions, chunks=result)


def compare_group_state(reference, actual):
    """CPU comparison; group count and physical tensor strides are irrelevant."""
    require(reference['full_input_tokens'] == actual['full_input_tokens'], 'Input length differs')
    require(reference['selected_positions'] == actual['selected_positions'], 'Selected row identities differ')
    mismatches = []
    before = {x['chunk_start']: x for x in reference['chunks']}
    after = {x['chunk_start']: x for x in actual['chunks']}
    require(set(before) == set(after) and len(before) == len(reference['chunks'])
        and len(after) == len(actual['chunks']), 'Missing or duplicate canonical chunks')
    for start in sorted(before):
        left, right = before[start], after[start]
        require(left['chunk_end'] == right['chunk_end']
            and left['selected_positions'] == right['selected_positions'], 'Canonical chunk rows differ')
        for field in ('hidden',)+LAZY:
            a = left['state']['hidden'] if field == 'hidden' else left['state']['lazy'][field]
            b = right['state']['hidden'] if field == 'hidden' else right['state']['lazy'][field]
            if a != b:
                mismatches.append(dict(chunk_start=start, field=field, reference=a, actual=b))
    return seal(dict(schema='s13-canonical-hidden-mhc-comparison-v1',
        status='passed' if not mismatches else 'failed',
        selected_rows=len(reference['selected_positions']),
        reference_sha256=digest(reference), actual_sha256=digest(actual),
        exact_logical_byte_equality=not mismatches, mismatches=mismatches,
        group_count_equality_required=False))


class QualificationChecks:
    """Observation sink attached to a REAL own32 phase, never synthetic success.

    Hook `check_kv` immediately after fused gather/scatter with the corresponding
    original implementation's independently computed result. Hook `state` before
    and after regrouping, and at stage exits for baseline/optimized comparison.
    Calling `report` without observations remains `incomplete`.
    """
    def __init__(self, *, input_ids_sha256, owner_rank, qualification_only,
                 benchmark_request, torch_module):
        require(qualification_only is True and benchmark_request is False,
            'Tensor checks must be outside benchmark quality/timing phases')
        require(isinstance(input_ids_sha256, str) and len(input_ids_sha256) == 64
            and owner_rank in range(4), 'Exact request hash and B4 owner required')
        self.input_sha, self.owner, self.t = input_ids_sha256, owner_rank, torch_module
        self.kv_checks, self.regroup_checks, self.states = [], [], {}

    def check_kv(self, reference, actual, *, kind, layer, stage, group_key,
                 positions, source_plan_sha256):
        require(kind in ('normalized_gather', 'assembled_memory'), 'Unknown KV comparison point')
        require(type(layer) is int and 3 <= layer < 39 and stage in range(4)
            and isinstance(group_key, str) and group_key, 'Actual KV scope required')
        positions = list(positions)
        require(positions == sorted(set(positions)), 'KV positions must be sorted and exact')
        require(reference.shape == actual.shape and reference.dtype == actual.dtype
            and reference.device == actual.device and reference.shape[0] == len(positions),
            'Original and packed KV layout or row namespace differs')
        ref, got = tensor_fingerprint(reference, self.t), tensor_fingerprint(actual, self.t)
        item = seal(dict(kind=kind, layer=layer, stage=stage, group_key=group_key,
            positions_sha256=digest(positions), rows=len(positions),
            source_plan_sha256=source_plan_sha256, reference=ref, actual=got,
            exact=ref == got, reference_operator='original_per_layer_unpacked_path'))
        self.kv_checks.append(item)
        require(item['exact'], 'Actual KV pack differs from original per-layer reference')
        return item

    def state(self, groups, *, total, stage, boundary):
        require(stage in range(4) and boundary in ('regroup_before', 'regroup_after', 'stage_exit'),
            'Unknown actual stage observation')
        key = str(stage)+':'+boundary
        require(key not in self.states, 'Repeated state observation')
        snapshot = canonical_group_state(groups, total=total, torch_module=self.t)
        self.states[key] = snapshot
        if boundary == 'regroup_after':
            require(str(stage)+':regroup_before' in self.states, 'Missing before-regroup tensor state')
            comparison = compare_group_state(self.states[str(stage)+':regroup_before'], snapshot)
            self.regroup_checks.append(dict(stage=stage, comparison=comparison))
            require(comparison['status'] == 'passed', 'Regroup changed hidden or one of four mHC states')
        return snapshot

    def report(self):
        observed = bool(self.kv_checks or self.states or self.regroup_checks)
        failed = any(not x['exact'] for x in self.kv_checks) or any(
            x['comparison']['status'] != 'passed' for x in self.regroup_checks)
        return seal(dict(schema='s13-system-pack-tensor-observations-v1',
            status='failed' if failed else ('observed' if observed else 'incomplete'),
            input_ids_sha256=self.input_sha, owner_rank=self.owner,
            qualification_only=True, benchmark_request=False,
            kv_checks=list(self.kv_checks), regroup_checks=list(self.regroup_checks),
            stage_states=dict(self.states), passed_coverage_claim=False,
            gpu_timing_measurement=False, exact_comparison=True))


def compare_stage_selectors(reference_phase, actual_phase):
    """Use complete selection arrays from phases, NOT counts or snapshot digests."""
    for phase in (reference_phase, actual_phase):
        require(phase['record']['input_ids_sha256'] == reference_phase['record']['input_ids_sha256'],
            'Selector comparison input differs')
    before = reference_phase['controller']['progressive']['selections']
    after = actual_phase['controller']['progressive']['selections']
    require(len(before) == len(after) == 4, 'Four actual stage selectors required')
    checks = []
    for layer, left, right in zip(SOURCE_LAYERS, before, after):
        checked(left); checked(right)
        require(left['source_layer_zero_based'] == right['source_layer_zero_based'] == layer,
            'Stage source layer differs')
        a, b = left['selected_positions'], right['selected_positions']
        require(isinstance(a, list) and isinstance(b, list) and a == sorted(set(a))
            and b == sorted(set(b)), 'Complete ordered selector IDs required')
        checks.append(dict(layer=layer, exact=a == b, reference_rows=len(a), actual_rows=len(b),
            reference_sha256=digest(a), actual_sha256=digest(b),
            missing_positions=sorted(set(a)-set(b)), added_positions=sorted(set(b)-set(a))))
    return seal(dict(schema='s13-system-pack-selector-comparison-v1',
        input_ids_sha256=reference_phase['record']['input_ids_sha256'],
        status='passed' if all(x['exact'] for x in checks) else 'failed', stages=checks))
