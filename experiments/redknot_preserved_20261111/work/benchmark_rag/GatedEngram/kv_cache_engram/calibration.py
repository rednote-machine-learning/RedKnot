"""CPU-only binding between a declared gate policy and external evidence.

This module contains no real calibration data. ``bind`` creates an integrity
record; it does not run calibration, authenticate evidence, prove model accuracy,
or qualify a policy for production. The caller must independently establish the
meaning and trustworthiness of the evidence whose digest is supplied here.
Fingerprints prevent accidental scope/policy reuse, not deliberate forgery.
"""

from dataclasses import asdict, dataclass

from .core import CacheIdentity, digest, require, validate_digest
from .gate import GatePolicy

SCHEMA = "kv-cache-engram-calibration-binding-v1"


def _scope_sha256(identity: CacheIdentity) -> str:
    require(type(identity) is CacheIdentity, "CacheIdentity required")
    fields = asdict(identity)
    # Revalidate even a frozen instance before deriving the scope.
    CacheIdentity(**fields)
    # Calibration is document-independent, but never model/layer/format-free.
    fields.pop("tokens_sha256")
    fields.pop("token_count")
    return digest(fields)


def _policy_sha256(policy: GatePolicy) -> str:
    require(type(policy) is GatePolicy, "GatePolicy required")
    policy.__post_init__()
    require(
        isinstance(policy.calibration_id, str) and bool(policy.calibration_id.strip()),
        "calibration binding requires a nonempty policy calibration_id",
    )
    # Includes enabled, thresholds, norm_floor, block/probe/guard configuration,
    # seed, calibration_id, and any fields added to GatePolicy in the future.
    return digest(asdict(policy))


def _checksum(record_id: str, scope_sha256: str, policy_sha256: str, evidence_sha256: str) -> str:
    return digest(
        {
            "schema": SCHEMA,
            "record_id": record_id,
            "scope_sha256": scope_sha256,
            "policy_sha256": policy_sha256,
            "evidence_sha256": evidence_sha256,
        }
    )


@dataclass(frozen=True)
class CalibrationRecord:
    """Immutable scope/policy binding, not an accuracy certificate.

    The document token digest/count are intentionally outside the scope. All
    remaining CacheIdentity fields and every GatePolicy field must match.
    Directly constructed or externally loaded records must pass ``validate``
    before use; a checksum is not an evidence signature or trust decision.
    """

    record_id: str
    scope_sha256: str
    policy_sha256: str
    evidence_sha256: str
    checksum: str

    def __post_init__(self) -> None:
        require(isinstance(self.record_id, str) and bool(self.record_id.strip()), "calibration record_id must be nonempty")
        for name in ("scope_sha256", "policy_sha256", "evidence_sha256", "checksum"):
            validate_digest(getattr(self, name), name)

    @classmethod
    def bind(cls, identity: CacheIdentity, policy: GatePolicy, evidence_sha256: str) -> "CalibrationRecord":
        """Bind declared external evidence without claiming to verify it."""
        scope_sha256 = _scope_sha256(identity)
        policy_sha256 = _policy_sha256(policy)
        validate_digest(evidence_sha256, "evidence_sha256")
        record_id = policy.calibration_id
        return cls(
            record_id, scope_sha256, policy_sha256, evidence_sha256, _checksum(record_id, scope_sha256, policy_sha256, evidence_sha256)
        )

    def validate(self, identity: CacheIdentity, policy: GatePolicy) -> None:
        """Reject damaged records and any changed scope or policy.

        Successful validation means only that this integrity binding matches
        these supplied contracts. It does not inspect or authenticate evidence.
        """
        require(type(self) is CalibrationRecord, "CalibrationRecord required")
        self.__post_init__()
        require(
            self.checksum == _checksum(self.record_id, self.scope_sha256, self.policy_sha256, self.evidence_sha256),
            "calibration record checksum mismatch",
        )
        require(self.scope_sha256 == _scope_sha256(identity), "calibration model/execution/layer/representation scope mismatch")
        require(self.policy_sha256 == _policy_sha256(policy), "calibration policy mismatch")
        require(self.record_id == policy.calibration_id, "calibration record_id mismatch")
