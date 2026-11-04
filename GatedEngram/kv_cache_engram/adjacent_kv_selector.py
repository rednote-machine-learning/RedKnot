"""Pure CPU reference for the explicitly chosen adjacent-SWA score and top15%.

Human layer numbers 3..6 correspond to zero-based layer indices 2..5. At each
same input position the score is the mean of 2->3, 3->4, 4->5 relative L2
changes, with the *previous* layer norm floored at 1e-8. These are different
layer projection spaces: the score is an experimental ranking heuristic, not
a proof of semantic instability, safe KV reuse, or an accuracy guarantee.

Ranking is independently performed within each actual native 8192-token
chunk. The denominator is all document tokens in that chunk; document-prefix
SWA refresh tokens are not an additional FFN guard. Exactly ceil(15*n/100)
document positions are chosen, with descending score and ascending absolute
position as the tie break. All non-document positions are mandatory for FFN.
This module does not select SWA refresh rows or infer YOCO layer eligibility.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .core import ValidationError, normalize_rows, validate_digest

LAYER_INDICES = (2, 3, 4, 5)
TRANSITIONS = ((2, 3), (3, 4), (4, 5))
FEATURE_KIND = "swa_kv"
REPRESENTATION = "swa-postnorm-prerope-v1"
NORM_FLOOR = 1e-8
CHUNK_TOKENS = 8192
SCHEMA = "adjacent-swa-3to6-top15-selector-v1"


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _finite_nonnegative(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, ValueError):
        return False


@dataclass(frozen=True)
class SWAKVFeatures:
    """Explicit, copied CPU provenance; no unverified device read or alias."""

    layer_index: int
    input_ids_sha256: str
    positions: tuple
    rows: tuple
    feature_kind: str
    representation: str
    source_layer_index: int

    def __post_init__(self):
        _require(type(self.layer_index) is int and self.layer_index in LAYER_INDICES, "features must come from zero-based SWA layers2..5")
        _require(
            type(self.source_layer_index) is int and self.source_layer_index == self.layer_index,
            "shared/aliased feature producers cannot stand in for distinct SWA layers",
        )
        _require(
            self.feature_kind == FEATURE_KIND and self.representation == REPRESENTATION,
            "only same-position SWA postnorm/preRoPE features are valid; never shared global KV",
        )
        validate_digest(self.input_ids_sha256, "input token identity")
        _require(
            type(self.positions) in (list, tuple)
            and bool(self.positions)
            and all(type(p) is int and p >= 0 for p in self.positions)
            and list(self.positions) == sorted(set(self.positions)),
            "positions must be explicit, nonempty, unique ascending absolute integers",
        )
        positions = tuple(self.positions)
        rows = normalize_rows(self.rows, count=len(positions))
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True)
class AdjacentScore:
    input_ids_sha256: str
    positions: tuple
    transition_scores: tuple
    scores: tuple

    def as_dict(self):
        return dict(
            schema=SCHEMA,
            input_ids_sha256=self.input_ids_sha256,
            positions=list(self.positions),
            scores=list(self.scores),
            transition_scores=[
                dict(previous_layer=a, next_layer=b, values=list(values))
                for (a, b), values in zip(TRANSITIONS, self.transition_scores, strict=False)
            ],
            layer_indices_zero_based=list(LAYER_INDICES),
            human_layer_numbers=[3, 4, 5, 6],
            feature_kind=FEATURE_KIND,
            representation=REPRESENTATION,
            denominator="previous_layer_l2_norm",
            norm_floor=NORM_FLOOR,
            aggregation="arithmetic_mean_of_three_adjacent_changes_per_token",
            cohort_aggregation=False,
            accuracy_admission=False,
        )


def adjacent_relative_scores(layer_features):
    """Return all three per-token changes and their per-token arithmetic mean."""
    _require(
        isinstance(layer_features, Mapping)
        and all(type(key) is int for key in layer_features)
        and set(layer_features) == set(LAYER_INDICES),
        "exact distinct feature layers2..5 are required",
    )
    features = []
    for layer in LAYER_INDICES:
        feature = layer_features[layer]
        _require(
            type(feature) is SWAKVFeatures and feature.layer_index == layer,
            "feature mapping keys must match their explicit layer provenance",
        )
        # Revalidate/copy at the boundary rather than trusting an injected
        # dataclass instance with bypassed post-init validation.
        feature = SWAKVFeatures(**vars(feature))
        features.append(feature)
    reference = features[0]
    _require(
        all(
            f.input_ids_sha256 == reference.input_ids_sha256
            and f.positions == reference.positions
            and len(f.rows[0]) == len(reference.rows[0])
            for f in features
        ),
        "adjacent features must bind the same input, same absolute positions and same width",
    )
    changes = []
    for previous, following in zip(features, features[1:], strict=False):
        values = []
        for a, b in zip(previous.rows, following.rows, strict=False):
            denominator, delta = math.hypot(*a), math.hypot(*(y - x for x, y in zip(a, b, strict=False)))
            _require(math.isfinite(denominator) and math.isfinite(delta), "nonfinite derived KV norm")
            value = delta / max(denominator, NORM_FLOOR)
            _require(math.isfinite(value), "nonfinite adjacent relative change")
            values.append(value)
        changes.append(tuple(values))
    scores = tuple(math.fsum(v / 3 for v in values) for values in zip(*changes, strict=False))
    _require(all(math.isfinite(value) and value >= 0 for value in scores), "nonfinite mean adjacent score")
    return AdjacentScore(reference.input_ids_sha256, reference.positions, tuple(changes), scores)


def _chunk_geometry(input_tokens, chunk_start, chunk_end, document_spans):
    _require(
        type(input_tokens) is int
        and input_tokens > 0
        and type(chunk_start) is int
        and type(chunk_end) is int
        and 0 <= chunk_start < chunk_end <= input_tokens
        and chunk_start % CHUNK_TOKENS == 0
        and chunk_end == min(chunk_start + CHUNK_TOKENS, input_tokens),
        "one exact native8K chunk or final remainder is required",
    )
    _require(type(document_spans) in (list, tuple), "explicit actual document spans are required")
    spans, last = [], 0
    for span in document_spans:
        _require(type(span) in (list, tuple) and len(span) == 2, "document span must be a start/end pair")
        start, end = span
        _require(
            type(start) is int and type(end) is int and 0 <= start < end <= input_tokens and start >= last,
            "document spans must be ordered, disjoint, nonempty and in input range",
        )
        spans.append((start, end))
        last = end
    return tuple(spans)


def select_chunk_by_scores(scores, *, input_tokens, chunk_start, chunk_end, document_spans):
    """Rank supplied CPU per-token scores; this function never invents a metric."""
    spans = _chunk_geometry(input_tokens, chunk_start, chunk_end, document_spans)
    _require(
        type(scores) in (list, tuple) and len(scores) == chunk_end - chunk_start,
        "one score is required for every input position in the exact chunk",
    )
    _require(all(_finite_nonnegative(value) for value in scores), "scores must be finite nonnegative CPU numbers, not bools")
    eligible = [position for start, end in spans for position in range(max(start, chunk_start), min(end, chunk_end))]
    eligible_set = set(eligible)
    mandatory = [position for position in range(chunk_start, chunk_end) if position not in eligible_set]
    ranked = sorted(eligible, key=lambda position: (-scores[position - chunk_start], position))
    count = (15 * len(eligible) + 99) // 100
    sensitive = sorted(ranked[:count])
    selected = sorted(sensitive + mandatory)
    return dict(
        schema=SCHEMA,
        input_tokens=input_tokens,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        document_spans=[list(span) for span in spans],
        document_eligible_positions=eligible,
        ranked_document_positions=ranked,
        sensitive_indices=sensitive,
        mandatory_ffn_indices=mandatory,
        ffn_selected_rows=selected,
        scores=list(scores),
        document_eligible_count=len(eligible),
        sensitive_count=count,
        fraction_numerator=15,
        fraction_denominator=100,
        rounding="ceil",
        ranking="descending_score_then_ascending_absolute_position",
        swa_prefix_rows_are_not_automatic_ffn_guards=True,
        swa_online_rows_not_selected_here=True,
        accuracy_admission=False,
        native_layer_eligibility_inferred=False,
        runtime_hook_installed=False,
    )


def plan_adjacent_chunk(layer_features, *, input_tokens, chunk_start, chunk_end, document_spans):
    """Combine provenance-checked adjacent scores and independent native-chunk ranking."""
    _chunk_geometry(input_tokens, chunk_start, chunk_end, document_spans)
    scored = adjacent_relative_scores(layer_features)
    _require(
        scored.positions == tuple(range(chunk_start, chunk_end)),
        "feature positions must exactly cover this native chunk, without compacted or missing rows",
    )
    plan = select_chunk_by_scores(
        scored.scores, input_tokens=input_tokens, chunk_start=chunk_start, chunk_end=chunk_end, document_spans=document_spans
    )
    plan["feature_provenance"] = scored.as_dict()
    return plan
