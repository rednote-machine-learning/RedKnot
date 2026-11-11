"""CPU-only geometry for independently planned SWA-KV and FFN row sets.

This is not a native adapter, a selector, or an accuracy-admission policy. It
accepts sensitive positions chosen elsewhere and does not define a score,
choose the score's layer numbering, or infer which YOCO/global layers support
reuse. A caller must explicitly keep a model-required layer fully native.

Positions and ``layer_index`` are zero based. Document spans are explicit,
ordered, non-overlapping half-open intervals in the actual model input. At
layers 0..5 both paths are full. Later SWA-KV is online only for each document's
first min(127, document_length) tokens and all non-document input tokens.
There is intentionally no document-tail or native-chunk boundary protection.
Sensitive FFN positions do not implicitly enter the SWA online row set.

All work uses ordinary Python values. Importing or calling this module never
imports Torch, installs hooks, validates real KV, or starts an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import ValidationError

SCHEMA = "ffn-only-independent-row-geometry-v1"
FIRST_FULL_LAYER_COUNT = 6
DOCUMENT_PREFIX_TOKENS = 127


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _indices(values, input_tokens, label):
    _require(type(values) in (list, tuple, range, set, frozenset), label + " must be an explicit finite collection of integer positions")
    _require(
        all(type(value) is int and 0 <= value < input_tokens for value in values),
        label + " must contain only in-range integers, not bools or numeric substitutes",
    )
    return tuple(sorted(set(values)))


def _spans(values, input_tokens):
    _require(type(values) in (list, tuple), "explicit ordered document spans are required")
    spans, previous_end = [], 0
    for value in values:
        _require(type(value) in (list, tuple) and len(value) == 2, "every document span must be a half-open (start, end) pair")
        start, end = value
        _require(
            type(start) is int and type(end) is int and 0 <= start < end <= input_tokens and start >= previous_end,
            "document spans must be nonempty, ordered, disjoint, integer and in range",
        )
        spans.append((start, end))
        previous_end = end
    return tuple(spans)


@dataclass(frozen=True)
class FFNOnlyMaskPlan:
    """Immutable CPU geometry; no tensor/runtime or semantic-safety proof."""

    input_tokens: int
    layer_index: int
    document_spans: tuple
    sensitive_indices: tuple
    mandatory_question_indices: tuple
    non_document_indices: tuple
    swa_online_rows: tuple
    swa_reuse_rows: tuple
    ffn_selected_rows: tuple
    ffn_skipped_rows: tuple
    first_six_native: bool
    native_required_layer: bool

    @property
    def full_native(self):
        return self.first_six_native or self.native_required_layer

    def as_dict(self):
        """JSON-ready evidence without inventing model qualification claims."""
        return dict(
            schema=SCHEMA,
            input_tokens=self.input_tokens,
            layer_index=self.layer_index,
            layer_index_convention="zero_based_geometry_only",
            document_spans=[list(span) for span in self.document_spans],
            sensitive_indices=list(self.sensitive_indices),
            mandatory_question_indices=list(self.mandatory_question_indices),
            non_document_indices=list(self.non_document_indices),
            swa_online_rows=list(self.swa_online_rows),
            swa_reuse_rows=list(self.swa_reuse_rows),
            ffn_selected_rows=list(self.ffn_selected_rows),
            ffn_skipped_rows=list(self.ffn_skipped_rows),
            counts=dict(
                swa_online=len(self.swa_online_rows),
                swa_reuse=len(self.swa_reuse_rows),
                ffn_selected=len(self.ffn_selected_rows),
                ffn_skipped=len(self.ffn_skipped_rows),
            ),
            first_six_native=self.first_six_native,
            native_required_layer=self.native_required_layer,
            full_native=self.full_native,
            document_prefix_tokens=DOCUMENT_PREFIX_TOKENS,
            document_tail_guard_tokens=0,
            native_chunk_guards=False,
            sensitive_ffn_implies_swa_recompute=False,
            score_policy_selected=False,
            model_layer_eligibility_inferred=False,
            runtime_adapter_installed=False,
            accuracy_or_speedup_proven=False,
        )


def plan_layer_masks(
    *, input_tokens, layer_index, document_spans, sensitive_indices, mandatory_question_indices=(), native_required_layer=False
):
    """Plan independent row sets from explicit positions, not from scores.

    Sensitive indices may repeat or be unordered; outputs are deduplicated and
    ascending. Mandatory question indices must lie outside every actual
    document span. Other non-document tokens are automatically online for SWA,
    but are not automatically selected for FFN unless explicitly supplied.

    Empty document spans mean the whole input is non-document. A document
    shorter than 127 tokens has its entire span online. Adjacent documents keep
    their own prefix boundaries. No implicit layer upper bound or model-specific
    sparse-layer whitelist is asserted here; this helper cannot authorize a
    native layer. Invalid arguments are rejected even for a full-native layer.
    """
    _require(type(input_tokens) is int and input_tokens > 0, "positive integer input length required")
    _require(type(layer_index) is int and layer_index >= 0, "nonnegative integer layer index required")
    _require(type(native_required_layer) is bool, "native-required override must be an explicit bool")
    spans = _spans(document_spans, input_tokens)
    sensitive = _indices(sensitive_indices, input_tokens, "sensitive indices")
    questions = _indices(mandatory_question_indices, input_tokens, "mandatory question indices")

    non_document, cursor = [], 0
    for start, end in spans:
        non_document.extend(range(cursor, start))
        cursor = end
    non_document.extend(range(cursor, input_tokens))
    non_document = tuple(non_document)
    non_document_set = set(non_document)
    _require(set(questions) <= non_document_set, "mandatory question positions must be outside actual document spans")

    first_six_native = layer_index < FIRST_FULL_LAYER_COUNT
    if first_six_native or native_required_layer:
        swa_online = ffn_selected = tuple(range(input_tokens))
    else:
        online = set(non_document)
        for start, end in spans:
            online.update(range(start, min(start + DOCUMENT_PREFIX_TOKENS, end)))
        swa_online = tuple(sorted(online))
        ffn_selected = tuple(sorted(set(sensitive) | set(questions)))

    swa_set, ffn_set = set(swa_online), set(ffn_selected)
    return FFNOnlyMaskPlan(
        input_tokens=input_tokens,
        layer_index=layer_index,
        document_spans=spans,
        sensitive_indices=sensitive,
        mandatory_question_indices=questions,
        non_document_indices=non_document,
        swa_online_rows=swa_online,
        swa_reuse_rows=tuple(position for position in range(input_tokens) if position not in swa_set),
        ffn_selected_rows=ffn_selected,
        ffn_skipped_rows=tuple(position for position in range(input_tokens) if position not in ffn_set),
        first_six_native=first_six_native,
        native_required_layer=native_required_layer,
    )
