"""CPU-only fixed-candidate calibration and untouched-document validation.

This workflow selects a numerical KV gate on calibration data, then freezes it.
Its oracle measures KV error, not EM/F1, TTFT, or a statistical safety guarantee.
It never produces a runtime CalibrationRecord or enables live cache reuse.
Seals are reproducibility checks, not signatures authenticating capture origin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Optional

from .core import (CacheIdentity, ValidationError, canonical_bytes, digest,
                   integer, require, validate_digest)
from .gate import GatePolicy
from .reuse_audit import PairedKV, audit_pair


_SCHEMA = "kv-cache-engram-frozen-calibration-v1"
_SCOPE_FIELDS = ("model_sha256", "tokenizer_sha256", "execution_sha256", "layer",
                 "width", "representation", "storage_dtype")
_COUNTS = ("token_rows", "blocks", "compared_blocks", "false_accept_blocks",
           "false_reject_blocks", "reused_rows", "bad_reused_rows",
           "online_callback_rows", "fallback_probe_overhead_rows",
           "extra_callback_rows_over_full")
_LIMITATIONS = [
    "KV-oracle validation is not task-accuracy validation or an EM/F1 guarantee.",
    "Row counts are not FLOPs or TTFT; capture and audit overhead are excluded.",
    "Source IDs, capture origin and fingerprints are declarations, not authentication.",
    "A held-out failure cannot trigger candidate reselection on those held-out data.",
    "No runtime CalibrationRecord or authorization to reuse is produced.",
]


@dataclass(frozen=True)
class CalibrationExample:
    pair: PairedKV
    source_document_id: str
    declared_split: Optional[str] = None
    paired_capture_sha256: Optional[str] = None

    def __post_init__(self):
        require(type(self.pair) is PairedKV, "PairedKV required")
        self.pair.__post_init__()
        require(type(self.source_document_id) is str and bool(self.source_document_id)
                and self.source_document_id == self.source_document_id.strip(),
                "source_document_id must be nonempty without surrounding whitespace")
        require(self.declared_split in (None, "calibration", "evaluation"),
                "invalid declared split")
        if self.paired_capture_sha256 is not None:
            validate_digest(self.paired_capture_sha256, "paired_capture_sha256")
            require(self.declared_split is not None, "paired capture requires its declared split")


def from_paired_capture(capture, layer: int, *, max_values=1_048_576):
    """Import an open sealed pair, preserving its source group and split.

    Reading is CPU-only and does not start a capture/model. The bridge checks
    both complete matrices against the combined max_values budget before reads.
    """
    from .paired_capture import PairedCapture
    from .snapshot_audit import pair_from_snapshots
    require(type(capture) is PairedCapture, "open PairedCapture required")
    capture.validate()
    pair = pair_from_snapshots(capture.independent, capture.reference, layer,
                              max_values=max_values)
    return CalibrationExample(pair, capture.spec.source_document_id,
                              capture.spec.split, capture.seal)


def _number(value, name):
    require(type(value) in (int, float), name + " must be an explicit real number")
    try:
        value = float(value)
    except (ValueError, OverflowError):
        raise ValidationError(name + " must be finite") from None
    require(math.isfinite(value) and value >= 0, name + " must be finite and nonnegative")
    return value


def _limits(relative, absolute, false_accept, bad_rows):
    integer(false_accept, "max_false_accept_blocks")
    integer(bad_rows, "max_bad_reused_rows")
    return {"oracle_relative_tolerance": _number(relative, "oracle_relative_tolerance"),
            "oracle_absolute_tolerance": _number(absolute, "oracle_absolute_tolerance"),
            "max_false_accept_blocks": false_accept, "max_bad_reused_rows": bad_rows}


def _scope(pair):
    return dict({key: getattr(pair.online_identity, key) for key in _SCOPE_FIELDS},
                data_origin_declared=pair.data_origin)


def _evidence(example):
    pair = example.pair
    fields = {"source_document_id": example.source_document_id,
              "declared_split": example.declared_split,
              "paired_capture_sha256": example.paired_capture_sha256,
              "identity": asdict(pair.online_identity),
              "identity_sha256": pair.online_identity.sha256,
              "document_token_sha256": pair.online_identity.tokens_sha256,
              "offline_cache_checksum": pair.offline.checksum,
              "online_rows_sha256": digest(pair.online_rows),
              "request_sha256": pair.request_sha256,
              "request_id": pair.request_id, "document_start": pair.document_start,
              "data_origin_declared": pair.data_origin,
              "capture_sha256_declared": pair.capture_sha256}
    return dict(fields, evidence_sha256=digest(fields))


def _examples(values, role):
    require(type(values) in (tuple, list), "examples must be a fixed CPU list or tuple")
    values = tuple(values)
    require(bool(values), role + " examples cannot be empty")
    evidence = []
    duplicate_keys = set()
    for example in values:
        require(type(example) is CalibrationExample, "CalibrationExample required")
        example.__post_init__()
        require(example.declared_split in (None, role), "declared split conflicts with workflow role")
        item = _evidence(example)
        # Renaming a request/source ID must not silently count identical data twice.
        key = tuple(item[k] for k in ("identity_sha256", "offline_cache_checksum",
                    "online_rows_sha256", "request_sha256", "document_start"))
        require(key not in duplicate_keys, "duplicate paired evidence in split")
        duplicate_keys.add(key)
        evidence.append(item)
    scope = _scope(values[0].pair)
    require(all(_scope(example.pair) == scope for example in values),
            "workflow requires one model/config/layer/representation/origin scope")
    return values, evidence, scope


def _aggregate(reports):
    counts = {key: 0 for key in _COUNTS}
    counts["nonfinite_oracle_metrics"] = 0
    for report in reports:
        summary = report["summary"]
        for key in _COUNTS:
            integer(summary[key], key)
            counts[key] += summary[key]
        nonfinite = summary["all_row_errors"]["nonfinite_metrics"]
        integer(nonfinite, "nonfinite_oracle_metrics")
        counts["nonfinite_oracle_metrics"] += nonfinite
    counts["reused_row_fraction"] = counts["reused_rows"] / counts["token_rows"]
    return counts


def _reasons(summary, limits):
    reasons = []
    if summary["nonfinite_oracle_metrics"]:
        reasons.append("nonfinite_oracle_metrics")
    if summary["false_accept_blocks"] > limits["max_false_accept_blocks"]:
        reasons.append("false_accept_block_budget_exceeded")
    if summary["bad_reused_rows"] > limits["max_bad_reused_rows"]:
        reasons.append("bad_reused_row_budget_exceeded")
    if summary["reused_rows"] == 0:
        reasons.append("no_actual_reuse")
    return reasons


def _run(examples, evidence, policy, limits):
    reports = [audit_pair(example.pair, policy,
                oracle_relative_tolerance=limits["oracle_relative_tolerance"],
                oracle_absolute_tolerance=limits["oracle_absolute_tolerance"])
               for example in examples]
    summary = _aggregate(reports)
    reasons = _reasons(summary, limits)
    return {"policy": asdict(policy), "policy_sha256": digest(asdict(policy)),
            "summary": summary, "eligible": not reasons, "rejection_reasons": reasons,
            "examples": [{"evidence_sha256": item["evidence_sha256"], "audit": report}
                         for item, report in zip(evidence, reports)]}


def _selected(reports):
    eligible = [report for report in reports if report["eligible"]]
    # Every candidate sees exactly the same calibration rows. Maximizing reused
    # rows therefore exactly maximizes the fraction, without floating-point ties.
    return min(eligible, key=lambda report: (-report["summary"]["reused_rows"],
                report["summary"]["online_callback_rows"], report["policy_sha256"])) if eligible else None


def _json_copy(value):
    try:
        return json.loads(canonical_bytes(value))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("artifact must be finite canonical JSON") from exc


@dataclass(frozen=True)
class FrozenCalibration:
    """Immutable sealed JSON; integrity is checked again before evaluation.

    The artifact includes every calibration audit, not just the winning policy.
    A caller able to rewrite and reseal it is not authenticated by the checksum.
    """
    payload_json: str
    seal_sha256: str

    def __post_init__(self):
        require(type(self.payload_json) is str, "payload_json must be text")
        try:
            payload = json.loads(self.payload_json)
        except (ValueError, TypeError) as exc:
            raise ValidationError("invalid frozen calibration JSON") from exc
        validate_digest(self.seal_sha256, "seal_sha256")
        require(canonical_bytes(_json_copy(payload)).decode("utf-8") == self.payload_json,
                "frozen payload must use canonical JSON")
        require(digest(payload) == self.seal_sha256, "frozen calibration seal mismatch")
        try:
            _validate_payload(payload)
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValidationError("malformed frozen calibration payload") from exc

    @classmethod
    def from_dict(cls, envelope):
        require(type(envelope) is dict and set(envelope) == {"payload", "seal_sha256"},
                "invalid frozen calibration envelope")
        payload = _json_copy(envelope["payload"])
        return cls(canonical_bytes(payload).decode("utf-8"), envelope["seal_sha256"])

    def to_dict(self):
        self.__post_init__()
        return {"payload": json.loads(self.payload_json), "seal_sha256": self.seal_sha256}

    @property
    def mode(self):
        return self.to_dict()["payload"]["mode"]

    @property
    def selected_policy(self):
        data = self.to_dict()["payload"]["selected_policy"]
        return None if data is None else GatePolicy(**data)


def _validate_payload(payload):
    """Validate the envelope and selection invariants, without original tensors."""
    required = {"schema", "scope", "limits", "calibration_inputs", "candidate_policies",
                "candidate_reports", "selected_policy_sha256", "selected_policy", "mode",
                "reason", "fallback_policy", "runtime_admission_created", "limitations"}
    require(type(payload) is dict and set(payload) == required, "invalid frozen payload fields")
    require(payload["schema"] == _SCHEMA, "unsupported frozen schema")
    require(payload["runtime_admission_created"] is False, "runtime authorization is forbidden")
    limits = payload["limits"]
    require(type(limits) is dict and set(limits) == {
        "oracle_relative_tolerance", "oracle_absolute_tolerance",
        "max_false_accept_blocks", "max_bad_reused_rows"}, "invalid frozen limits")
    _limits(limits["oracle_relative_tolerance"], limits["oracle_absolute_tolerance"],
            limits["max_false_accept_blocks"], limits["max_bad_reused_rows"])
    require(payload["fallback_policy"] == asdict(GatePolicy()), "fallback must be disabled full recompute")
    inputs = payload["calibration_inputs"]
    require(type(inputs) is list and bool(inputs), "missing calibration evidence")
    for item in inputs:
        require(type(item) is dict, "invalid calibration evidence")
        fields = dict(item)
        checksum = fields.pop("evidence_sha256", None)
        require(checksum == digest(fields), "calibration evidence fingerprint mismatch")
        identity = CacheIdentity(**item["identity"])
        require(identity.sha256 == item["identity_sha256"]
                and identity.tokens_sha256 == item["document_token_sha256"], "evidence identity mismatch")
        scope = dict({key: getattr(identity, key) for key in _SCOPE_FIELDS},
                     data_origin_declared=item["data_origin_declared"])
        require(scope == payload["scope"], "evidence scope mismatch")
        require(item["declared_split"] in (None, "calibration"), "wrong frozen split")
    candidates, reports = payload["candidate_policies"], payload["candidate_reports"]
    require(type(candidates) is list and bool(candidates) and type(reports) is list
            and len(candidates) == len(reports), "candidate/report coverage mismatch")
    hashes = []
    for candidate, report in zip(candidates, reports):
        policy = GatePolicy(**candidate)
        policy_hash = digest(asdict(policy))
        hashes.append(policy_hash)
        require(report["policy"] == candidate and report["policy_sha256"] == policy_hash,
                "candidate policy mismatch")
        require(len(report["examples"]) == len(inputs), "candidate dropped calibration examples")
        audits = []
        for example_report, item in zip(report["examples"], inputs):
            require(example_report["evidence_sha256"] == item["evidence_sha256"], "audit/evidence mismatch")
            audit = example_report["audit"]
            require(audit["policy"] == candidate and audit["identity"] == item["identity"]
                    and audit["request_sha256"] == item["request_sha256"]
                    and audit["offline_cache_checksum"] == item["offline_cache_checksum"],
                    "audit binding mismatch")
            require(audit["oracle_tolerances"] == {"relative_l2": limits["oracle_relative_tolerance"],
                    "absolute_error": limits["oracle_absolute_tolerance"]}, "audit oracle mismatch")
            audits.append(audit)
        summary = _aggregate(audits)
        require(summary == report["summary"], "candidate aggregate mismatch")
        reasons = _reasons(summary, limits)
        require(report["eligible"] is (not reasons) and report["rejection_reasons"] == reasons,
                "candidate eligibility mismatch")
    require(hashes == sorted(set(hashes)), "candidate hashes must be unique canonical order")
    selected = _selected(reports)
    if selected is None:
        require(payload["selected_policy"] is None and payload["selected_policy_sha256"] is None
                and payload["mode"] == "full_recompute" and payload["reason"] == "no_eligible_candidate",
                "invalid full-recompute selection")
    else:
        require(payload["selected_policy"] == selected["policy"]
                and payload["selected_policy_sha256"] == selected["policy_sha256"]
                and payload["mode"] == "gated_reuse_candidate" and payload["reason"] == "eligible_candidate_selected",
                "selection does not follow frozen calibration criterion")


def fit_candidates(calibration_examples, candidate_policies, *,
                   oracle_relative_tolerance, oracle_absolute_tolerance,
                   max_false_accept_blocks, max_bad_reused_rows) -> FrozenCalibration:
    """Select only among supplied candidates; no thresholds or probes are fitted.

    The two budgets are aggregate integer counts over all calibration examples,
    not rates or probabilities. No held-out examples are accepted by this API.
    Every candidate and every calibration example is retained, including failures.
    Validation errors abort the entire operation instead of dropping bad examples.
    """
    limits = _limits(oracle_relative_tolerance, oracle_absolute_tolerance,
                     max_false_accept_blocks, max_bad_reused_rows)
    examples, evidence, scope = _examples(calibration_examples, "calibration")
    require(type(candidate_policies) in (list, tuple) and bool(candidate_policies),
            "candidate_policies must be a nonempty fixed list or tuple")
    candidates = {}
    for policy in candidate_policies:
        require(type(policy) is GatePolicy, "GatePolicy candidate required")
        policy.__post_init__()
        policy_hash = digest(asdict(policy))
        require(policy_hash not in candidates, "duplicate candidate policy")
        candidates[policy_hash] = policy
    reports = [_run(examples, evidence, candidates[key], limits) for key in sorted(candidates)]
    selected = _selected(reports)
    payload = {"schema": _SCHEMA, "scope": scope, "limits": limits,
        "calibration_inputs": evidence, "candidate_policies": [report["policy"] for report in reports],
        "candidate_reports": reports,
        "selected_policy_sha256": selected["policy_sha256"] if selected else None,
        "selected_policy": selected["policy"] if selected else None,
        "mode": "gated_reuse_candidate" if selected else "full_recompute",
        "reason": "eligible_candidate_selected" if selected else "no_eligible_candidate",
        "fallback_policy": asdict(GatePolicy()), "runtime_admission_created": False,
        "limitations": list(_LIMITATIONS)}
    return FrozenCalibration(canonical_bytes(payload).decode("utf-8"), digest(payload))


def evaluate_frozen(frozen: FrozenCalibration, heldout_examples) -> dict:
    """Apply the chosen policy once to a disjoint document group; never reselect."""
    require(type(frozen) is FrozenCalibration, "FrozenCalibration required")
    payload = frozen.to_dict()["payload"]
    examples, evidence, scope = _examples(heldout_examples, "evaluation")
    require(scope == payload["scope"], "heldout model/config/layer/origin scope mismatch")
    calibration_sources = {item["source_document_id"] for item in payload["calibration_inputs"]}
    calibration_tokens = {item["document_token_sha256"] for item in payload["calibration_inputs"]}
    calibration_requests = {item["request_sha256"] for item in payload["calibration_inputs"]}
    require(not calibration_sources.intersection(item["source_document_id"] for item in evidence),
            "source document group leakage between calibration and heldout")
    require(not calibration_tokens.intersection(item["document_token_sha256"] for item in evidence),
            "document token hash leakage between calibration and heldout")
    require(not calibration_requests.intersection(item["request_sha256"] for item in evidence),
            "full request context leakage between calibration and heldout")
    policy = GatePolicy(**(payload["selected_policy"] or payload["fallback_policy"]))
    report = _run(examples, evidence, policy, payload["limits"])
    passed = payload["selected_policy"] is not None and report["eligible"]
    return {"schema": "kv-cache-engram-heldout-validation-v1",
        "frozen_calibration_seal": frozen.seal_sha256, "scope": scope,
        "limits": payload["limits"], "heldout_inputs": evidence,
        "selected_policy_sha256": payload["selected_policy_sha256"],
        "mode": payload["mode"], "validation_result": "kv_validation_pass" if passed else "rejected",
        "kv_validation_pass": passed,
        "reason": ("within_explicit_kv_budgets" if passed else
                   "no_eligible_candidate" if payload["selected_policy"] is None else "heldout_gate_rejected"),
        "report": report, "candidate_reselection_performed": False,
        "runtime_admission_created": False, "limitations": list(_LIMITATIONS)}
