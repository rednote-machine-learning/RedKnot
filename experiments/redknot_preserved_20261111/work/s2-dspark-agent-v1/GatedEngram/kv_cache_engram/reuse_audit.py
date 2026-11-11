"""CPU paired-capture audit of *KV error*, not task accuracy or inference speed.

The full-online rows are a research reference supplied by the caller. This code
does not run a model, authenticate a capture producer, tune thresholds, authorize
runtime reuse, or prove that probes were computed independently. Every pair and
every row is retained in the result; no difficult examples are filtered out.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional, Tuple

from .cache import CacheArtifact
from .core import (Binding, CacheIdentity, OnlineRows, Rows, integer,
                   normalize_rows, require, token_digest, validate_digest)
from .gate import GatePolicy, evaluate, make_plan


@dataclass(frozen=True)
class PairedKV:
    """One complete document/layer in one full online request.

    ``online_identity`` must describe the independently computed online rows,
    not merely be copied to bypass a mismatch. Token IDs must include the exact
    request tokenization, including any template/prefix and question. The
    document is an exact slice at ``document_start``. Values are canonical
    postnorm/pre-RoPE SWA KV, in the CPU representation defined by CacheIdentity.

    ``data_origin`` and ``capture_sha256`` are provenance declarations, not
    verification that real model weights were executed. Model-capture mode
    requires an external evidence fingerprint to avoid unlabelled provenance.
    """

    offline: CacheArtifact
    online_identity: CacheIdentity
    document_token_ids: Tuple[int, ...]
    request_token_ids: Tuple[int, ...]
    document_start: int
    online_rows: Rows
    request_id: str
    data_origin: str
    capture_sha256: Optional[str] = None

    def __post_init__(self):
        require(type(self.offline) is CacheArtifact, "CacheArtifact required")
        require(type(self.online_identity) is CacheIdentity, "online identity required")
        self.offline.validate(self.online_identity)
        require(type(self.document_token_ids) in (tuple, list), "document IDs must be CPU IDs")
        require(type(self.request_token_ids) in (tuple, list), "request IDs must be CPU IDs")
        document = tuple(self.document_token_ids)
        request = tuple(self.request_token_ids)
        require(token_digest(document) == self.online_identity.tokens_sha256,
                "document token fingerprint mismatch")
        token_digest(request)  # Validate every request token, not just the slice.
        require(len(document) == self.online_identity.token_count, "document length mismatch")
        integer(self.document_start, "document_start")
        require(request[self.document_start:self.document_start + len(document)] == document,
                "document is not the claimed exact request slice")
        require(isinstance(self.request_id, str) and bool(self.request_id.strip()),
                "request_id required")
        require(self.data_origin in ("synthetic", "model_capture"),
                "data_origin must be synthetic or model_capture")
        if self.capture_sha256 is not None:
            validate_digest(self.capture_sha256, "capture_sha256")
        require(self.data_origin != "model_capture" or self.capture_sha256 is not None,
                "model_capture requires an external capture evidence fingerprint")
        object.__setattr__(self, "document_token_ids", document)
        object.__setattr__(self, "request_token_ids", request)
        object.__setattr__(self, "online_rows", normalize_rows(self.online_rows,
            width=self.online_identity.width, count=len(document)))

    @property
    def request_sha256(self):
        return token_digest(self.request_token_ids)

    def binding(self, start: int, end: int) -> Binding:
        return Binding(self.request_id, "full-online-paired-kv-audit",
            self.online_identity, self.document_start, start, end,
            self.request_sha256)


def _tolerance(value, name):
    require(type(value) in (int, float), name + " must be an explicit real number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        require(False, name + " must be finite")
    require(math.isfinite(number) and number >= 0, name + " must be finite and nonnegative")
    return number


def _metrics(online, offline, floor):
    difference = tuple(a-b for a, b in zip(online, offline))
    absolute = max(abs(value) for value in difference)
    distance = math.hypot(*difference)
    reference = math.hypot(*online)
    if not all(math.isfinite(value) for value in (absolute, distance, reference)):
        return {"relative_l2": None, "absolute_error": None, "finite_metrics": False}
    relative = distance / max(reference, floor)
    if not math.isfinite(relative):
        return {"relative_l2": None, "absolute_error": None, "finite_metrics": False}
    return {"relative_l2": relative, "absolute_error": absolute, "finite_metrics": True}


def _bucket(position, count, policy):
    head = position < policy.head_guard
    tail = position >= count-policy.tail_guard
    return "both_guards" if head and tail else "head_guard" if head else "tail_guard" if tail else "interior"


def _row_summary(rows):
    if not rows:
        return {"rows": 0, "outside_oracle_tolerances": 0, "nonfinite_metrics": 0,
                "max_relative_l2": None, "max_absolute_error": None}
    nonfinite = sum(not row["finite_metrics"] for row in rows)
    return {"rows": len(rows),
            "outside_oracle_tolerances": sum(not row["within_oracle_tolerances"] for row in rows),
            "nonfinite_metrics": nonfinite,
            "max_relative_l2": None if nonfinite else max(row["relative_l2"] for row in rows),
            "max_absolute_error": None if nonfinite else max(row["absolute_error"] for row in rows)}


def audit_pair(pair: PairedKV, policy: GatePolicy, *,
               oracle_relative_tolerance: float,
               oracle_absolute_tolerance: float) -> dict:
    """Compare fixed sampled decisions with a full-row offline oracle.

    Oracle thresholds are explicit *KV-error* limits, not accepted EM/F1 loss.
    The oracle inspects the unprobed/unprotected rows that could actually be
    reused. Probe and guard errors are still reported, but those rows are fresh
    online values in a successful merge. False acceptance means at least one
    reused row violates an oracle limit. False rejection means the sampled
    numerical gate denied reuse even though all candidate reused rows met both
    oracle limits. Disabled policy and all-online blocks are not gate mistakes.

    Callback-row counts simulate the existing prepare semantics. They count
    repeated probes when falling back to a complete block, but are not FLOPs,
    latency, cost of obtaining hidden states, or the full-row audit's own work.
    No CalibrationRecord is required: this diagnoses a gate, not runtime reuse.
    """
    require(type(pair) is PairedKV and type(policy) is GatePolicy,
            "PairedKV and GatePolicy required")
    pair.__post_init__()  # Recheck immutable-envelope integrity at audit time.
    policy.__post_init__()
    relative_limit = _tolerance(oracle_relative_tolerance, "oracle_relative_tolerance")
    absolute_limit = _tolerance(oracle_absolute_tolerance, "oracle_absolute_tolerance")
    count = pair.online_identity.token_count
    tokens = []
    for position, (online, cached) in enumerate(zip(pair.online_rows, pair.offline.rows)):
        metric = _metrics(online, cached, policy.norm_floor)
        within = (metric["finite_metrics"] and metric["relative_l2"] <= relative_limit
                  and metric["absolute_error"] <= absolute_limit)
        tokens.append(dict(metric, position=position,
            absolute_position=pair.document_start + position,
            layer=pair.online_identity.layer,
            boundary_bucket=_bucket(position, count, policy),
            position_decile=min(9, position * 10 // count),
            within_oracle_tolerances=within))
    blocks = []
    for start in range(0, count, policy.block_size):
        end = min(start + policy.block_size, count)
        bound = pair.binding(start, end)
        plan = make_plan(bound, policy)
        probes = set(plan.probe_positions)
        mandatory = set(plan.mandatory_positions)
        candidates = tuple(p for p in plan.positions if p not in probes and p not in mandatory)
        online_probes = OnlineRows(bound.sha256, plan.probe_positions,
            tuple(pair.online_rows[p] for p in plan.probe_positions))
        decision = evaluate(plan, policy,
            tuple(pair.offline.rows[p] for p in plan.probe_positions), online_probes, bound)
        oracle_allows = bool(candidates) and all(tokens[p]["within_oracle_tolerances"] for p in candidates)
        actually_reused = candidates if decision.allow_reuse else ()
        # Enabled numerical evaluation takes probes first; a denial then takes
        # the full current block. Disabled and all-mandatory paths skip probes.
        sampled = len(probes) if policy.enabled and probes else 0
        if decision.allow_reuse:
            calls = [sampled] + ([len(mandatory)] if mandatory else [])
        else:
            calls = ([sampled] if sampled else []) + [end-start]
        reuse_set = set(actually_reused)
        for position in plan.positions:
            tokens[position].update(role="guard" if position in mandatory else
                "probe" if position in probes else "candidate",
                reused=position in reuse_set)
        classified = bool(policy.enabled and probes and candidates)
        blocks.append({"start": start, "end": end,
            "absolute_start": pair.document_start + start,
            "absolute_end": pair.document_start + end,
            "layer": pair.online_identity.layer,
            "binding_sha256": bound.sha256,
            "probe_positions": list(plan.probe_positions),
            "mandatory_positions": list(plan.mandatory_positions),
            "candidate_positions": list(candidates),
            "gate": asdict(decision),
            "oracle_allows_candidate_reuse": oracle_allows,
            "gate_comparison_applicable": classified,
            "false_accept": classified and decision.allow_reuse and not oracle_allows,
            "false_reject": classified and not decision.allow_reuse and oracle_allows,
            "actual_reused_positions": list(actually_reused),
            "bad_reused_positions": [p for p in actually_reused if not tokens[p]["within_oracle_tolerances"]],
            "unsampled_candidate_errors": _row_summary([tokens[p] for p in candidates]),
            "online_callback_row_counts": calls,
            "fallback_probe_overhead_rows": sampled if not decision.allow_reuse else 0,
            "mode": "gated_reuse" if actually_reused else "full_recompute"})
    reused = [row for row in tokens if row["reused"]]
    callback_rows = sum(sum(block["online_callback_row_counts"]) for block in blocks)
    return {"schema": "kv-cache-engram-paired-kv-audit-v1",
        "request_id": pair.request_id, "request_sha256": pair.request_sha256,
        "document_start": pair.document_start, "identity": asdict(pair.online_identity),
        "offline_cache_checksum": pair.offline.checksum,
        "data_origin_declared": pair.data_origin,
        "capture_sha256_declared": pair.capture_sha256,
        "policy": asdict(policy),
        "oracle_tolerances": {"relative_l2": relative_limit, "absolute_error": absolute_limit},
        "tokens": tokens, "blocks": blocks,
        "summary": {"token_rows": count, "blocks": len(blocks),
            "compared_blocks": sum(block["gate_comparison_applicable"] for block in blocks),
            "false_accept_blocks": sum(block["false_accept"] for block in blocks),
            "false_reject_blocks": sum(block["false_reject"] for block in blocks),
            "reused_rows": len(reused), "reused_row_fraction": len(reused)/count,
            "bad_reused_rows": sum(not row["within_oracle_tolerances"] for row in reused),
            "online_callback_rows": callback_rows,
            "fallback_probe_overhead_rows": sum(block["fallback_probe_overhead_rows"] for block in blocks),
            "extra_callback_rows_over_full": max(0, callback_rows-count),
            "unsampled_candidate_errors": _row_summary([row for row in tokens if row["role"] == "candidate"]),
            "all_row_errors": _row_summary(tokens)},
        "by_boundary": {name: _row_summary([row for row in tokens if row["boundary_bucket"] == name])
                        for name in ("head_guard", "tail_guard", "both_guards", "interior")},
        "by_position_decile": {str(i): _row_summary([row for row in tokens if row["position_decile"] == i])
                               for i in range(10)},
        "limitations": ["Full-online reference and capture provenance are caller-supplied, not authenticated.",
            "Oracle labels bound KV errors only; no EM/F1, answer-quality or no-loss claim.",
            "Row counts exclude hidden-state preparation and audit overhead; they are not FLOPs or TTFT.",
            "Local fallback cannot undo upstream approximation; captures must come from a full-online reference.",
            "This report diagnoses a fixed gate; it does not authorize runtime reuse or fit thresholds."]}


def audit_splits(calibration_pairs, evaluation_pairs, policy: GatePolicy, *,
                 oracle_relative_tolerance: float,
                 oracle_absolute_tolerance: float) -> dict:
    """Audit both declared splits without fitting, rejecting document leakage.

    All layers/context variants of a token-identical document must belong to
    the same split. Different tokenizations/overlapping text cannot be detected
    by this exact-ID check; callers must additionally split by source document.
    Empty splits are rejected. Duplicate exact captures within a split are
    rejected rather than silently weighted twice. Nothing is dropped or selected.
    """
    calibration = tuple(calibration_pairs)
    evaluation = tuple(evaluation_pairs)
    require(bool(calibration) and bool(evaluation), "both splits require at least one pair")
    for pair in calibration + evaluation:
        require(type(pair) is PairedKV, "split members must be PairedKV")
    # Applying one policy across layers is explicit in the report; combining
    # different weights/tokenizers/execution regimes into one cohort is not.
    scope_fields = ("model_sha256", "tokenizer_sha256", "execution_sha256",
                    "representation", "storage_dtype", "width")
    scopes = {tuple(getattr(pair.online_identity, key) for key in scope_fields)
              for pair in calibration + evaluation}
    require(len(scopes) == 1, "mixed model or execution configuration across splits")
    require(len({pair.data_origin for pair in calibration + evaluation}) == 1,
            "synthetic and model-capture pairs must be audited in separate cohorts")
    calibration_docs = {pair.online_identity.tokens_sha256 for pair in calibration}
    evaluation_docs = {pair.online_identity.tokens_sha256 for pair in evaluation}
    require(calibration_docs.isdisjoint(evaluation_docs), "document token identity leaks across splits")
    reports = {}
    for name, pairs in (("calibration", calibration), ("evaluation", evaluation)):
        keys = [(pair.online_identity.sha256, pair.request_sha256, pair.document_start) for pair in pairs]
        require(len(set(keys)) == len(keys), "duplicate exact capture in " + name)
        items = [audit_pair(pair, policy,
            oracle_relative_tolerance=oracle_relative_tolerance,
            oracle_absolute_tolerance=oracle_absolute_tolerance) for pair in pairs]
        layers = sorted({item["identity"]["layer"] for item in items})
        totals = {}
        for layer in layers:
            summaries = [item["summary"] for item in items if item["identity"]["layer"] == layer]
            names = ("token_rows", "blocks", "compared_blocks", "false_accept_blocks",
                     "false_reject_blocks", "reused_rows", "bad_reused_rows", "online_callback_rows",
                     "fallback_probe_overhead_rows")
            total = {key: sum(summary[key] for summary in summaries) for key in names}
            total["reused_row_fraction"] = total["reused_rows"] / total["token_rows"]
            totals[str(layer)] = total
        reports[name] = {"pair_count": len(items), "document_count": len({p.online_identity.tokens_sha256 for p in pairs}),
                         "pairs": items, "by_layer": totals}
    return {"schema": "kv-cache-engram-audit-splits-v1", "thresholds_fitted": False,
        "document_token_overlap": [], "splits": reports,
        "limitations": ["Exact token-hash disjointness is not independent-source or semantic holdout proof.",
                        "No parameter selection, model quality validation or speed experiment was performed."]}
