"""Deterministic production-eligibility governance gate v2.

This additive gate determines whether immutable v2 governance evidence is
sufficient for production eligibility. It does not mutate the registry,
promote factors, submit orders, hold credentials, authorize a venue adapter,
or enable LIVE. ELIGIBLE is a governance decision only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from typing import Any

from std0_quant.execution.execution_governance_bridge_v2 import (
    ACCEPTED,
    ExecutionGovernanceBridgeArtifactV2,
    verify_execution_governance_bridge_artifact_v2,
)
from std0_quant.execution.execution_validation import ExecutionValidationTarget
from std0_quant.execution.execution_validation_policy_v2 import (
    ExecutionValidationDecisionV2,
    ExecutionValidationPolicyV2,
    execution_validation_decision_v2_hash,
    execution_validation_policy_v2_hash,
)
from std0_quant.research.factors.contracts import FactorStatus, ValidationStatus
from std0_quant.research.factors.registry import (
    FactorRegistryRecord,
    registry_record_hash,
)
from std0_quant.storage import canonical_json


PRODUCTION_ELIGIBILITY_GATE_V2 = "production_eligibility_gate_v2"
PRODUCTION_ELIGIBILITY_DECISION_SCHEMA_V2 = (
    "production_eligibility_decision_v2"
)

ELIGIBLE = "ELIGIBLE"
BLOCKED = "BLOCKED"


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _sha256(value: Any, name: str) -> str:
    text = _nonempty(value, name)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be lowercase SHA256 hex")
    return text


@dataclass(frozen=True)
class ProductionEligibilityDecisionV2:
    gate_run_id: str
    target: ExecutionValidationTarget
    status: str
    reasons: tuple[str, ...]
    registry_record_hash: str
    governance_bridge_artifact_hash: str
    governance_bridge_run_id: str
    execution_decision_artifact_hash: str
    execution_run_id: str
    provenance_run_id: str
    policy_id: str
    policy_version: str
    policy_hash: str
    source_qualification_policy_hash: str
    trusted_public_key_policy_hash: str
    artifact_hash: str
    gate_version: str = PRODUCTION_ELIGIBILITY_GATE_V2
    schema_version: str = PRODUCTION_ELIGIBILITY_DECISION_SCHEMA_V2

    def __post_init__(self) -> None:
        for name in (
            "gate_run_id",
            "governance_bridge_run_id",
            "execution_run_id",
            "provenance_run_id",
            "policy_id",
            "policy_version",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name),
            )

        if not isinstance(self.target, ExecutionValidationTarget):
            raise TypeError("target must be ExecutionValidationTarget")

        if self.status not in {ELIGIBLE, BLOCKED}:
            raise ValueError("unsupported production eligibility status")

        reasons = tuple(_nonempty(reason, "reason") for reason in self.reasons)
        object.__setattr__(self, "reasons", reasons)

        if self.status == ELIGIBLE and reasons:
            raise ValueError("ELIGIBLE cannot contain reasons")
        if self.status == BLOCKED and not reasons:
            raise ValueError("BLOCKED requires reasons")

        for name in (
            "registry_record_hash",
            "governance_bridge_artifact_hash",
            "execution_decision_artifact_hash",
            "policy_hash",
            "source_qualification_policy_hash",
            "trusted_public_key_policy_hash",
            "artifact_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name),
            )

        if self.gate_version != PRODUCTION_ELIGIBILITY_GATE_V2:
            raise ValueError("unsupported gate_version")
        if self.schema_version != PRODUCTION_ELIGIBILITY_DECISION_SCHEMA_V2:
            raise ValueError("unsupported schema_version")


def _decision_payload(
    decision: ProductionEligibilityDecisionV2,
) -> dict[str, Any]:
    return {
        "target": asdict(decision.target),
        "status": decision.status,
        "reasons": decision.reasons,
        "registry_record_hash": decision.registry_record_hash,
        "governance_bridge_artifact_hash": (
            decision.governance_bridge_artifact_hash
        ),
        "governance_bridge_run_id": decision.governance_bridge_run_id,
        "execution_decision_artifact_hash": (
            decision.execution_decision_artifact_hash
        ),
        "execution_run_id": decision.execution_run_id,
        "provenance_run_id": decision.provenance_run_id,
        "policy_id": decision.policy_id,
        "policy_version": decision.policy_version,
        "policy_hash": decision.policy_hash,
        "source_qualification_policy_hash": (
            decision.source_qualification_policy_hash
        ),
        "trusted_public_key_policy_hash": (
            decision.trusted_public_key_policy_hash
        ),
        "gate_version": decision.gate_version,
        "schema_version": decision.schema_version,
    }


def production_eligibility_decision_v2_hash(
    decision: ProductionEligibilityDecisionV2,
) -> str:
    if not isinstance(decision, ProductionEligibilityDecisionV2):
        raise TypeError(
            "decision must be ProductionEligibilityDecisionV2"
        )
    return hashlib.sha256(
        canonical_json(_decision_payload(decision)).encode("utf-8")
    ).hexdigest()


def evaluate_production_eligibility_v2(
    record: FactorRegistryRecord,
    bridge: ExecutionGovernanceBridgeArtifactV2,
    decision: ExecutionValidationDecisionV2,
    policy: ExecutionValidationPolicyV2,
    *,
    gate_run_id: str,
) -> ProductionEligibilityDecisionV2:
    if not isinstance(record, FactorRegistryRecord):
        raise TypeError("record must be FactorRegistryRecord")
    if not isinstance(bridge, ExecutionGovernanceBridgeArtifactV2):
        raise TypeError(
            "bridge must be ExecutionGovernanceBridgeArtifactV2"
        )
    if not isinstance(decision, ExecutionValidationDecisionV2):
        raise TypeError("decision must be ExecutionValidationDecisionV2")
    if not isinstance(policy, ExecutionValidationPolicyV2):
        raise TypeError("policy must be ExecutionValidationPolicyV2")

    gate_run_id = _nonempty(gate_run_id, "gate_run_id")
    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if record.status != FactorStatus.VALIDATED:
        add("REGISTRY_NOT_VALIDATED")

    if (
        record.factor_id != decision.target.factor_id
        or record.factor_version != decision.target.factor_version
        or record.definition_hash != decision.target.definition_hash
    ):
        add("REGISTRY_TARGET_IDENTITY_MISMATCH")

    if execution_validation_decision_v2_hash(decision) != decision.artifact_hash:
        add("EXECUTION_DECISION_V2_ARTIFACT_HASH_MISMATCH")

    expected_policy_hash = execution_validation_policy_v2_hash(policy)
    if (
        decision.policy_id != policy.policy_id
        or decision.policy_version != policy.version
        or decision.policy_hash != expected_policy_hash
    ):
        add("EXECUTION_POLICY_V2_PROVENANCE_MISMATCH")

    if (
        decision.source_qualification_policy_hash
        != policy.source_qualification_policy_hash
    ):
        add("SOURCE_QUALIFICATION_POLICY_HASH_MISMATCH")

    if (
        decision.trusted_public_key_policy_hash
        != policy.trusted_public_key_policy_hash
    ):
        add("TRUSTED_PUBLIC_KEY_POLICY_HASH_MISMATCH")

    for reason in verify_execution_governance_bridge_artifact_v2(
        bridge,
        record,
        decision,
        policy,
    ):
        add(reason)

    if bridge.target != decision.target:
        add("GOVERNANCE_BRIDGE_V2_TARGET_MISMATCH")

    expected_registry_hash = registry_record_hash(record)
    if bridge.registry_record_hash != expected_registry_hash:
        add("GOVERNANCE_BRIDGE_V2_REGISTRY_HASH_MISMATCH")

    if bridge.execution_decision_artifact_hash != decision.artifact_hash:
        add("GOVERNANCE_BRIDGE_V2_EXECUTION_DECISION_HASH_MISMATCH")

    if (
        bridge.execution_run_id != decision.execution_run_id
        or bridge.provenance_run_id != decision.provenance_run_id
    ):
        add("GOVERNANCE_BRIDGE_V2_EXECUTION_PROVENANCE_MISMATCH")

    if (
        bridge.policy_id != policy.policy_id
        or bridge.policy_version != policy.version
        or bridge.policy_hash != expected_policy_hash
    ):
        add("GOVERNANCE_BRIDGE_V2_POLICY_PROVENANCE_MISMATCH")

    if (
        bridge.source_qualification_policy_hash
        != policy.source_qualification_policy_hash
    ):
        add("GOVERNANCE_BRIDGE_V2_SOURCE_POLICY_HASH_MISMATCH")

    if (
        bridge.trusted_public_key_policy_hash
        != policy.trusted_public_key_policy_hash
    ):
        add("GOVERNANCE_BRIDGE_V2_TRUSTED_KEY_POLICY_HASH_MISMATCH")

    if bridge.status != ACCEPTED:
        add("GOVERNANCE_BRIDGE_V2_NOT_ACCEPTED")

    if bridge.execution_validation_status != decision.validation_status:
        add("GOVERNANCE_BRIDGE_V2_EXECUTION_STATUS_MISMATCH")

    if decision.validation_status != ValidationStatus.PASS:
        add("EXECUTION_VALIDATION_V2_NOT_PASS")

    status = BLOCKED if reasons else ELIGIBLE

    provisional = ProductionEligibilityDecisionV2(
        gate_run_id=gate_run_id,
        target=decision.target,
        status=status,
        reasons=tuple(reasons),
        registry_record_hash=expected_registry_hash,
        governance_bridge_artifact_hash=bridge.artifact_hash,
        governance_bridge_run_id=bridge.bridge_run_id,
        execution_decision_artifact_hash=decision.artifact_hash,
        execution_run_id=decision.execution_run_id,
        provenance_run_id=decision.provenance_run_id,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        policy_hash=expected_policy_hash,
        source_qualification_policy_hash=(
            policy.source_qualification_policy_hash
        ),
        trusted_public_key_policy_hash=(
            policy.trusted_public_key_policy_hash
        ),
        artifact_hash="0" * 64,
    )

    return replace(
        provisional,
        artifact_hash=production_eligibility_decision_v2_hash(provisional),
    )


def verify_production_eligibility_decision_v2(
    gate_decision: ProductionEligibilityDecisionV2,
    record: FactorRegistryRecord,
    bridge: ExecutionGovernanceBridgeArtifactV2,
    decision: ExecutionValidationDecisionV2,
    policy: ExecutionValidationPolicyV2,
) -> tuple[str, ...]:
    if not isinstance(gate_decision, ProductionEligibilityDecisionV2):
        raise TypeError(
            "gate_decision must be ProductionEligibilityDecisionV2"
        )
    if not isinstance(record, FactorRegistryRecord):
        raise TypeError("record must be FactorRegistryRecord")
    if not isinstance(bridge, ExecutionGovernanceBridgeArtifactV2):
        raise TypeError(
            "bridge must be ExecutionGovernanceBridgeArtifactV2"
        )
    if not isinstance(decision, ExecutionValidationDecisionV2):
        raise TypeError("decision must be ExecutionValidationDecisionV2")
    if not isinstance(policy, ExecutionValidationPolicyV2):
        raise TypeError("policy must be ExecutionValidationPolicyV2")

    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if (
        production_eligibility_decision_v2_hash(gate_decision)
        != gate_decision.artifact_hash
    ):
        add("PRODUCTION_ELIGIBILITY_DECISION_V2_HASH_MISMATCH")

    expected = evaluate_production_eligibility_v2(
        record,
        bridge,
        decision,
        policy,
        gate_run_id=gate_decision.gate_run_id,
    )
    if gate_decision != expected:
        add("PRODUCTION_ELIGIBILITY_DECISION_V2_SEMANTICS_MISMATCH")

    return tuple(reasons)
