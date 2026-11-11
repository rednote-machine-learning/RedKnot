"""CPU reference protocol for an explicitly configured early-layer selector.

No model, Torch, device discovery, runtime hook or experiment is started here.
The score compares ONLINE and independent-document OFFLINE SWA KV at the SAME
layer and position, after normalization and BEFORE RoPE. It is not a comparison
between adjacent layers and is not a proof that a skipped token is harmless.

Six fully recomputed layers (0..5) supply scores. Larger scores retain online
computation. Fractions apply to the OPTIONAL eligible document interior only;
all mandatory rows are retained in addition. The production tensor adapter must
match this reference ordering and must account for its real online overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

from .core import ValidationError


EARLY_LAYERS = (0, 1, 2, 3, 4, 5)
METRIC = "mean_early_same_layer_relative_l2_online_ref"
REPRESENTATION = "swa_postnorm_prerope"
SCHEMA = "early-six-layer-token-sensitivity-selector-v1"


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _integer(value):
    return type(value) is int


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class EarlySensitivityConfig:
    """No implicit choice of a score or experiment fractions is permitted."""

    metric: str
    optional_fractions: tuple
    representation: str
    norm_floor: float = 1e-8
    guard_tokens: int = 128
    native_chunk_tokens: int = 8192
    document_tokens: int = 32768
    early_layers: tuple = EARLY_LAYERS

    def __post_init__(self):
        _require(self.metric == METRIC, "unsupported or unresolved sensitivity metric")
        _require(self.representation == REPRESENTATION,
                 "scores must compare normalized, pre-RoPE SWA KV")
        _require(type(self.early_layers) is tuple and self.early_layers == EARLY_LAYERS,
                 "all six full early layers 0..5 are mandatory")
        fractions = self.optional_fractions
        _require(type(fractions) is tuple and bool(fractions)
                 and all(_finite(f) and 0 < f <= 1 for f in fractions),
                 "explicit positive finite optional fractions required")
        _require(all(a < b for a, b in zip(fractions, fractions[1:]))
                 and fractions[-1] == 1,
                 "fractions must strictly increase and include full-retain 1.0")
        _require(_finite(self.norm_floor) and self.norm_floor > 0,
                 "positive finite norm floor required")
        _require(_integer(self.guard_tokens) and self.guard_tokens >= 128,
                 "at least 128 protected boundary tokens required")
        _require(self.native_chunk_tokens == 8192 and _integer(self.native_chunk_tokens)
                 and self.document_tokens == 32768 and _integer(self.document_tokens),
                 "this protocol requires actual native 8192 and document 32768 lengths")
        _require(self.guard_tokens * 2 <= self.native_chunk_tokens,
                 "boundary guard larger than a native chunk")

    def protocol(self):
        return dict(schema=SCHEMA, metric=self.metric, representation=self.representation,
            early_layers=list(self.early_layers), optional_fractions=list(self.optional_fractions),
            norm_floor=self.norm_floor, guard_tokens=self.guard_tokens,
            native_chunk_tokens=self.native_chunk_tokens, document_tokens=self.document_tokens,
            relative_l2="norm(online_same_layer-offline_same_layer)/max(norm(online_same_layer), norm_floor)",
            layer_aggregation="arithmetic mean of all six relative L2 values",
            rank_order="descending score, then ascending absolute input position",
            fraction_denominator="eligible optional interior rows in this native chunk",
            count_rounding="ceil(fraction * optional_interior_count)",
            mandatory_rows_additive=True, missing_scores_fail_closed=True,
            accuracy_admission=False, runtime_hook_installed=False)


def _shape_and_values(value):
    """Flatten a finite, rectangular Python reference row without tensor imports."""
    if _finite(value):
        return (), (float(value),)
    _require(isinstance(value, (list, tuple)) and bool(value),
             "KV reference rows must be nonempty rectangular finite sequences")
    parts = [_shape_and_values(item) for item in value]
    _require(all(shape == parts[0][0] for shape, _ in parts), "ragged KV reference row")
    return (len(parts),) + parts[0][0], tuple(x for _, values in parts for x in values)


def reference_layer_scores(online, offline, *, config):
    """Finite same-shaped [T, ...] plain sequences, online norm as denominator.

    This is a CPU test oracle, not an implementation intended for a timed model
    forward. Zero online norm uses the declared positive floor; overflow fails.
    """
    _require(isinstance(config, EarlySensitivityConfig), "validated explicit config required")
    _require(isinstance(online, (list, tuple)) and isinstance(offline, (list, tuple))
             and len(online) == len(offline) and len(online) > 0,
             "paired nonempty equal-length KV row sequences required")
    result, expected_shape = [], None
    for a, b in zip(online, offline):
        shape_a, va = _shape_and_values(a)
        shape_b, vb = _shape_and_values(b)
        _require(shape_a and shape_a == shape_b, "paired KV rows must have equal non-scalar shape")
        if expected_shape is None:
            expected_shape = shape_a
        _require(shape_a == expected_shape, "KV row shape changed between tokens")
        norm = math.hypot(*va)
        delta = math.hypot(*(x-y for x, y in zip(va, vb)))
        _require(math.isfinite(norm) and math.isfinite(delta), "nonfinite derived KV norm")
        score = delta / max(norm, config.norm_floor)
        _require(math.isfinite(score), "nonfinite derived relative L2")
        result.append(score)
    return tuple(result)


def aggregate_layer_scores(layer_scores, *, config):
    """Combine explicitly supplied CPU score vectors from all six full layers."""
    _require(isinstance(config, EarlySensitivityConfig), "validated explicit config required")
    _require(isinstance(layer_scores, Mapping)
             and all(_integer(k) for k in layer_scores)
             and set(layer_scores) == set(EARLY_LAYERS),
             "scores must contain exactly layers 0..5, not missing/extra/aliased layers")
    vectors = [layer_scores[layer] for layer in EARLY_LAYERS]
    _require(all(isinstance(v, (list, tuple)) for v in vectors)
             and len(vectors[0]) > 0 and all(len(v) == len(vectors[0]) for v in vectors),
             "equal-length nonempty layer score vectors required")
    _require(all(_finite(x) and x >= 0 for vector in vectors for x in vector),
             "finite nonnegative scores required for every layer/token")
    means = tuple(math.fsum(float(vector[i])/6 for vector in vectors)
                  for i in range(len(vectors[0])))
    _require(all(math.isfinite(value) for value in means), "nonfinite aggregate score")
    return means


def score_layer_pairs(layer_pairs, *, config):
    """CPU same-layer reference path: {0: (online_rows, offline_rows), ..., 5: ...}."""
    _require(isinstance(layer_pairs, Mapping)
             and all(_integer(k) for k in layer_pairs)
             and set(layer_pairs) == set(EARLY_LAYERS), "exact six-layer KV pairing required")
    _require(all(isinstance(v, (list, tuple)) and len(v) == 2 for v in layer_pairs.values()),
             "each layer needs an online/offline pair")
    return aggregate_layer_scores({layer: reference_layer_scores(*layer_pairs[layer], config=config)
                                   for layer in EARLY_LAYERS}, config=config)


@dataclass(frozen=True)
class LayerRoles:
    """The native adapter must explicitly attest every source role; no defaults."""

    layer: int
    native_global_source: bool
    native_index_source: bool
    engram_source: bool
    yoco_source: bool

    def __post_init__(self):
        _require(_integer(self.layer) and self.layer >= 0, "nonnegative native layer required")
        _require(all(type(value) is bool for value in (self.native_global_source,
            self.native_index_source, self.engram_source, self.yoco_source)),
            "all native layer source roles must be explicitly known booleans")


def eligible_sparse_layers(roles, *, layer_count):
    """Reject incomplete geometry; exclude early and all cache/index source layers.

    Even on eligible layers this does NOT authorize skipping global attention,
    its index path, or native state updates; only the separately verified pilot
    adapter can implement an ordinary-layer/token path.
    """
    _require(_integer(layer_count) and layer_count >= 6, "explicit native layer count required")
    _require(isinstance(roles, (list, tuple)) and len(roles) == layer_count
             and all(isinstance(role, LayerRoles) for role in roles)
             and sorted(role.layer for role in roles) == list(range(layer_count)),
             "complete unique native layer roles required")
    return tuple(sorted(role.layer for role in roles if role.layer >= 6 and not any((
        role.native_global_source, role.native_index_source, role.engram_source, role.yoco_source))))


def _position_set(values, start, end, name):
    _require(isinstance(values, (list, tuple, set, frozenset, range)), name+" must be explicit positions")
    result = set(values)
    _require(all(_integer(value) and start <= value < end for value in values),
             name+" contains invalid or out-of-chunk positions")
    _require(len(result) == len(values), name+" contains duplicate positions")
    return result


def plan_from_scores(scores, *, start, end, input_tokens, document_spans, config,
                     eligible_positions=None, mandatory_positions=()):
    """Pure, deterministic ranking oracle for a tensor selector implementation.

    ``scores[i]`` belongs to absolute position ``start+i``. Document intervals
    are [start,end) actual independent 32K cache chunks, not natural paragraphs.
    Non-document rows, every document/native-chunk first and last 128 rows,
    caller-ineligible rows and explicit mandatory rows are all retained.
    """
    _require(isinstance(config, EarlySensitivityConfig), "validated explicit config required")
    _require(all(_integer(value) for value in (start, end, input_tokens))
             and 0 <= start < end <= input_tokens
             and start % config.native_chunk_tokens == 0
             and end == min(start+config.native_chunk_tokens, input_tokens),
             "exact aligned native chunk and request length required")
    _require(isinstance(scores, (list, tuple)) and len(scores) == end-start
             and all(_finite(value) and value >= 0 for value in scores),
             "finite nonnegative score for every native-chunk row required")
    _require(isinstance(document_spans, (list, tuple)) and bool(document_spans),
             "explicit actual independent-document spans required")
    previous_end = -1
    spans = []
    for span in document_spans:
        _require(isinstance(span, (list, tuple)) and len(span) == 2
                 and all(_integer(value) for value in span), "invalid document interval")
        lo, hi = span
        _require(0 <= lo < hi <= input_tokens and hi-lo == config.document_tokens
                 and lo >= previous_end, "sorted nonoverlapping actual 32K documents required")
        spans.append((lo, hi))
        previous_end = hi
    all_positions = set(range(start, end))
    requested_eligible = (all_positions if eligible_positions is None else
                          _position_set(eligible_positions, start, end, "eligible_positions"))
    extra_mandatory = _position_set(mandatory_positions, start, end, "mandatory_positions")
    doc_interior = set()
    for lo, hi in spans:
        doc_interior.update(range(max(start, lo+config.guard_tokens), min(end, hi-config.guard_tokens)))
    chunk_interior = set(range(min(end, start+config.guard_tokens),
                               max(start, end-config.guard_tokens)))
    optional = requested_eligible & doc_interior & chunk_interior - extra_mandatory
    mandatory = all_positions - optional
    ranking = sorted(optional, key=lambda pos: (-float(scores[pos-start]), pos))
    selections = []
    previous_retained = set()
    for fraction in config.optional_fractions:
        # Ceil keeps the requested sensitivity budget conservative at tiny N.
        count = min(len(ranking), math.ceil(float(fraction)*len(ranking)))
        selected = set(ranking[:count])
        retained = mandatory | selected
        _require(previous_retained <= retained, "internal nonnested selection")
        previous_retained = retained
        selections.append(dict(optional_fraction=float(fraction), optional_selected_rows=count,
            optional_selected_positions=sorted(selected), retained_positions=sorted(retained),
            bypass_positions=sorted(all_positions-retained), retained_rows=len(retained),
            actual_retained_fraction=len(retained)/(end-start), mandatory_rows=len(mandatory),
            optional_rows=len(optional)))
    return dict(schema=SCHEMA, chunk_start=start, chunk_end=end, input_tokens=input_tokens,
        protocol=config.protocol(), document_spans=[list(span) for span in spans],
        mandatory_positions=sorted(mandatory), optional_positions=sorted(optional),
        optional_ranked_positions=ranking, selections=selections,
        first_six_layers_require_full_computation=True,
        source_layers_require_full_computation=True,
        global_index_and_cache_state_updates_must_be_preserved=True,
        accuracy_admission=False, cache_reuse_permission_granted=False)


def plan_chunk(layer_scores, **kwargs):
    """Six-layer CPU scores → deterministic per-chunk plans; invalid input raises."""
    return plan_from_scores(aggregate_layer_scores(layer_scores, config=kwargs["config"]), **kwargs)
