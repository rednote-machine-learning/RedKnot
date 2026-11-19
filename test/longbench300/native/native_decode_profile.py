"""Independent native Full decode profiles; importing this module is CPU-only.

The graph-aware validator replaces only the validator in a private copy of the
frozen actual_runtime function. Its checkpoint/topology/layout checks remain.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import FunctionType


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    return dict(value, sha256=digest(value))


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class DecodeProfile:
    name: str

    def __post_init__(self):
        require(self.name in ('base', 'dspark'), 'only base and dspark profiles are registered')

    @property
    def mtp_tokens(self):
        return 0 if self.name == 'base' else 5

    @property
    def keep_tokens(self):
        return 140

    @property
    def cuda_graphs(self):
        return self.name == 'dspark'

    @property
    def mtp_varlen(self):
        return self.name == 'dspark'

    def environment(self, args=None):
        require(getattr(args, 'batch_invariant', 1) == 1 and
            getattr(args, 'prefill_attempt_interval', 1) == 1,
            'diagnostic fixes batch invariance and prefill intervals to one')
        return {
            'DEEPINFER_USE_NEXTN_DECODE': str(self.mtp_tokens),
            'DEEPINFER_ENABLE_MTP_VARLEN': str(int(self.mtp_varlen)),
            'DEEPINFER_MAX_BACKTRACKING_TOKENS': str(self.mtp_tokens + 1),
            'DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE': 'compact',
            'DEEPINFER_YOCO_PREFILL_KEEP_TOKENS': '140',
            'DEEPINFER_ENFORCE_EAGER': str(int(not self.cuda_graphs)),
            'DEEPINFER_DFLASH_ENFORCE_EAGER': str(int(not self.cuda_graphs)),
            'DEEPINFER_ZERO_OVERHEAD_SCHEDULING': '1',
            'DEEPINFER_BATCH_INVARIANT': '1',
            'DEEPINFER_PREFILL_ATTEMPT_INTERVAL': '1',
            'DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL': '1',
            'DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL': '0',
            'DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE': '0',
            'DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS': '0',
        }

    def llm_kwargs(self):
        return dict(lookahead_tokens=min(1, self.mtp_tokens),
                    chunked_prefill=True, decode_sequence=False)

    def identity(self, args=None):
        return seal(dict(schema='s2-native-decode-profile-v1', profile=asdict(self),
            environment=self.environment(args), llm_kwargs=self.llm_kwargs(),
            module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            topology='EP8_DP8_CP1', keep_tokens=140,
            native_checkpoint_csa2_maps_unchanged=True,
            requested_flags_are_not_replay_or_acceptance_evidence=True,
            quality_qualified=False, performance_qualified=False))


def validate_runtime(profile, observed, args=None):
    """Keep the original checks, with exactly two disclosed graph profiles."""
    expected = dict(nextn=profile.mtp_tokens, use_cuda_graph=profile.cuda_graphs,
        dflash_use_cuda_graph=profile.cuda_graphs,
        yoco_prefill_truncation_mode='compact', keep_tokens=140,
        max_backtracking_tokens=profile.mtp_tokens + 1,
        mtp_varlen=profile.mtp_varlen, batch_invariant=True,
        lookahead_tokens=min(1, profile.mtp_tokens),
        mtp_module_present=bool(profile.mtp_tokens), zero_overhead_scheduling=True,
        cp_size=1, engine_type='NORMAL')
    required = set(expected) | {'window_size', 'compute_capability', 'csa2_maps', 'expected_csa2_maps'}
    require(required <= set(observed), 'runtime evidence is incomplete')
    for key, value in expected.items():
        require(type(observed[key]) is type(value) and observed[key] == value,
                f'runtime {key}={observed[key]!r}, expected {value!r}')
    capability = observed['compute_capability']
    require(isinstance(capability, (list, tuple)) and len(capability) == 2
        and all(type(x) is int for x in capability) and tuple(capability) == (9, 0),
        'native diagnostic requires actual SM90')
    window = observed['window_size']
    require(type(window) is int and window > 0 and 140 >= window + profile.mtp_tokens + 1,
            'compact140 cannot cover actual window plus backtracking')
    maps = observed['expected_csa2_maps']
    require(isinstance(maps, (dict, list)) and bool(maps)
        and digest(maps) == digest(observed['csa2_maps']), 'loaded native CSA2 maps changed')
    return seal(dict(schema='s2-native-decode-runtime-validation-v1', profile=profile.identity(args),
        observed=json.loads(json.dumps(observed)), validated_runtime_configuration=True,
        native_csa2_maps_sha256=digest(maps), quality_qualified=False,
        performance_qualified=False))


def actual_runtime(native_module, *args, **kwargs):
    original = native_module.actual_runtime
    private = dict(original.__globals__, validate_runtime=validate_runtime)
    clone = FunctionType(original.__code__, private, original.__name__,
                         original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__
    return clone(*args, **kwargs)


class GraphReplayMeter:
    """Count successful calls to the native graph runners, without GPU reads."""
    def __init__(self, model_runner):
        self.entries = {}
        for name, attr in (('main', 'cuda_graph_runner'), ('dflash', 'dflash_cuda_graph_runner')):
            runner = getattr(model_runner, attr, None)
            if runner is None:
                self.entries[name] = dict(runner=None, calls=0, captured=[])
                continue
            original = runner.replay
            entry = dict(runner=runner, original=original, override=runner.__dict__.get('replay'),
                had_override='replay' in runner.__dict__, calls=0,
                captured=sorted((getattr(runner, 'graphs', None) or {}).keys()))
            def counted(*args, _entry=entry, **kwargs):
                result = _entry['original'](*args, **kwargs)
                _entry['calls'] += 1
                return result
            entry['hook'] = counted
            runner.replay = counted
            self.entries[name] = entry

    def snapshot(self):
        return {name:dict(runner_present=e['runner'] is not None,
            captured_graph_sizes=e['captured'], successful_replay_calls=e['calls'])
            for name, e in self.entries.items()}

    def close(self):
        for entry in self.entries.values():
            runner = entry['runner']
            if runner is None:
                continue
            require(runner.__dict__.get('replay') is entry['hook'], 'graph meter ownership changed')
            if entry['had_override']:
                runner.replay = entry['override']
            else:
                del runner.replay
