"""Code-only final authority for SHADOW classification."""

from __future__ import annotations

from .schemas import GateEvidence, JevJudgments, ManagerReport


def classify_shadow(evidence: GateEvidence, *, context_pit_valid: bool,
                    k3_outputs_valid: bool, jev_outputs_valid: bool,
                    manager: ManagerReport | None,
                    jev: JevJudgments | None) -> tuple[str, tuple[str, ...]]:
    hard = {
        "COVERAGE_INVALID_OR_UNKNOWN": evidence.coverage_valid,
        "PROVENANCE_INVALID_OR_UNKNOWN": evidence.provenance_valid,
        "SANITY_INVALID_OR_UNKNOWN": evidence.sanity_valid,
        "CONTEXT_PIT_INVALID": context_pit_valid,
        "K3_OUTPUTS_INVALID": k3_outputs_valid,
        "JEV_OUTPUTS_INVALID": jev_outputs_valid,
    }
    blocked = tuple(name for name, value in hard.items() if value is not True)
    if blocked or manager is None or jev is None:
        return "BLOCKED", blocked or ("MODEL_OUTPUT_UNAVAILABLE",)
    reasons: list[str] = []
    if manager.direction == "ABSTAIN" or manager.confidence < 0.6 or len(manager.supporting_roles) < 2:
        reasons.append("K3_INSUFFICIENT_SUPPORT")
    consistent = jev.directional_evidence_consistent
    if consistent["answer"] is not True or consistent["probability"] < 0.6:
        reasons.append("JEV_DIRECTION_NOT_CONSISTENT")
    anomalous = jev.book_state_anomalous
    if anomalous["answer"] is not False or anomalous["probability"] > 0.4:
        reasons.append("JEV_BOOK_ANOMALY")
    risk = jev.execution_risk
    if risk["level"] == "HIGH" or risk["confidence"] < 0.5:
        reasons.append("JEV_EXECUTION_RISK")
    if jev.candidate_quality["score"] < 0.6:
        reasons.append("JEV_LOW_CANDIDATE_QUALITY")
    return ("SHADOW_REJECT", tuple(reasons)) if reasons else ("SHADOW_ACCEPT", ())
