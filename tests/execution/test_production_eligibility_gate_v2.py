# Fixture constructors copied from test_execution_governance_bridge_v2.py.
from dataclasses import replace

import pytest

from std0_quant.execution.execution_governance_bridge_v2 import (
    ACCEPTED,
    BLOCKED,
    ExecutionGovernanceBridgeArtifactV2,
    evaluate_execution_governance_bridge_v2,
    execution_governance_bridge_artifact_v2_hash,
    verify_execution_governance_bridge_artifact_v2,
)
from std0_quant.execution.execution_validation import ExecutionValidationTarget
from std0_quant.execution.execution_validation_policy import (
    MEASURED_VENUE_EXECUTION,
    ExecutionValidationDecision,
    ExecutionValidationPolicy,
)
from std0_quant.execution.execution_validation_policy_v2 import (
    ExecutionValidationDecisionV2,
    ExecutionValidationPolicyV2,
    execution_validation_decision_v2_hash,
    execution_validation_policy_v2_hash,
)
from std0_quant.research.factors.contracts import FactorStatus, ValidationStatus
from std0_quant.research.factors.registry import (
    FactorRegistryRecord,
    FactorTransition,
    registry_record_hash,
)


FACTOR_ID = "factor-a"
FACTOR_VERSION = "1"
DEFINITION_HASH = "a" * 64


def target(**changes):
    values = {
        "factor_id": FACTOR_ID,
        "factor_version": FACTOR_VERSION,
        "definition_hash": DEFINITION_HASH,
        "alpha_id": "alpha-a",
        "alpha_version": "1",
        "risk_policy_version": "risk-v1",
    }
    values.update(changes)
    return ExecutionValidationTarget(**values)


def validated_record(**transition_changes):
    transition_values = {
        "status_before": FactorStatus.CANDIDATE,
        "status_after": FactorStatus.VALIDATED,
        "research_validation_status": ValidationStatus.PASS,
        "temporal_integrity": ValidationStatus.PASS,
        "research_artifact_hash": "b" * 64,
        "research_run_id": "research-run",
        "execution_validation_status": None,
        "execution_artifact_hash": None,
        "execution_run_id": None,
        "decided_at": "2026-09-01T00:01:00Z",
        "research_policy_id": "research-policy",
        "research_policy_version": "3",
        "research_policy_hash": "c" * 64,
        "research_validation_reasons": (),
        "validation_evidence_bundle_hash": "d" * 64,
    }
    transition_values.update(transition_changes)
    transition = FactorTransition(**transition_values)
    return FactorRegistryRecord(
        factor_id=FACTOR_ID,
        factor_version=FACTOR_VERSION,
        definition_hash=DEFINITION_HASH,
        status=FactorStatus.VALIDATED,
        created_by="test",
        created_at="2026-09-01T00:00:00Z",
        transitions=(transition,),
    )


def policy_v2(**changes):
    values = {
        "policy_id": "execution-validation-policy",
        "version": "2",
        "required_pass_evidence_kind": MEASURED_VENUE_EXECUTION,
        "source_qualification_policy_id": "source-policy",
        "source_qualification_policy_version": "1",
        "source_qualification_policy_hash": "1" * 64,
        "trusted_public_key_policy_id": "trusted-key-policy",
        "trusted_public_key_policy_version": "1",
        "trusted_public_key_policy_hash": "2" * 64,
    }
    values.update(changes)
    return ExecutionValidationPolicyV2(**values)


def decision_v2(
    *,
    validation_status=ValidationStatus.PASS,
    reasons=(),
    t=None,
    p=None,
    execution_run_id="execution-v2-run",
    **changes,
):
    p = p or policy_v2()
    values = {
        "execution_run_id": execution_run_id,
        "provenance_run_id": "provenance-run",
        "measured_evidence_run_id": "measured-run",
        "qualification_run_id": "qualification-run",
        "attestation_run_id": "attestation-run",
        "verification_run_id": "verification-run",
        "coverage_run_id": "coverage-run",
        "target": t or target(),
        "validation_status": validation_status,
        "reasons": tuple(reasons),
        "provenance_artifact_hash": "3" * 64,
        "measured_execution_artifact_hash": "4" * 64,
        "source_qualification_artifact_hash": "5" * 64,
        "attestation_artifact_hash": "6" * 64,
        "signature_verification_artifact_hash": "7" * 64,
        "coverage_artifact_hash": "8" * 64,
        "source_artifact_hash": "9" * 64,
        "coverage_manifest_hash": "a" * 64,
        "source_qualification_policy_hash": p.source_qualification_policy_hash,
        "trusted_public_key_policy_hash": p.trusted_public_key_policy_hash,
        "policy_id": p.policy_id,
        "policy_version": p.version,
        "policy_hash": execution_validation_policy_v2_hash(p),
        "artifact_hash": "0" * 64,
    }
    values.update(changes)
    provisional = ExecutionValidationDecisionV2(**values)
    return replace(
        provisional,
        artifact_hash=execution_validation_decision_v2_hash(provisional),
    )


from dataclasses import asdict

from std0_quant.execution.production_eligibility_gate_v2 import (
    BLOCKED as GATE_BLOCKED,
    ELIGIBLE,
    evaluate_production_eligibility_v2 as evaluate_gate,
    production_eligibility_decision_v2_hash as gate_hash,
    verify_production_eligibility_decision_v2 as verify_gate,
)


def _rehash(value, hasher, **changes):
    provisional = replace(value, **{**changes, "artifact_hash": "0" * 64})
    return replace(provisional, artifact_hash=hasher(provisional))


def _evaluate(inputs, **changes):
    return evaluate_gate(**{**inputs, "gate_run_id": "gate-v2-run", **changes})


def _blocked(inputs, expected_reason, **changes):
    supplied = {**inputs, **changes}
    result = _evaluate(supplied)
    assert result.status == GATE_BLOCKED
    assert expected_reason in result.reasons
    assert result.reasons == tuple(dict.fromkeys(result.reasons))
    assert gate_hash(result) == result.artifact_hash
    assert verify_gate(result, **supplied) == ()
    assert _evaluate(supplied) == result
    return result


@pytest.fixture
def inputs():
    record = validated_record()
    policy = policy_v2()
    decision = decision_v2(p=policy)
    bridge = evaluate_execution_governance_bridge_v2(
        record, decision, policy, bridge_run_id="bridge-v2-run"
    )
    assert bridge.status == ACCEPTED
    assert verify_execution_governance_bridge_artifact_v2(
        bridge, record, decision, policy
    ) == ()
    values = dict(record=record, bridge=bridge, decision=decision, policy=policy)
    baseline = _evaluate(values)
    assert baseline.status == ELIGIBLE
    assert baseline.reasons == ()
    assert verify_gate(baseline, **values) == ()
    return values


def test_gate_v2_accepts_complete_bound_v2_evidence(inputs):
    result = _evaluate(inputs)
    assert result.status == ELIGIBLE
    assert result.reasons == ()
    assert result.target == inputs["decision"].target
    assert result.registry_record_hash == registry_record_hash(inputs["record"])
    assert result.governance_bridge_artifact_hash == inputs["bridge"].artifact_hash
    assert result.governance_bridge_run_id == inputs["bridge"].bridge_run_id
    assert result.execution_decision_artifact_hash == inputs["decision"].artifact_hash
    assert result.execution_run_id == inputs["decision"].execution_run_id
    assert result.provenance_run_id == inputs["decision"].provenance_run_id
    assert result.policy_id == inputs["policy"].policy_id
    assert result.policy_version == inputs["policy"].version
    assert result.policy_hash == execution_validation_policy_v2_hash(inputs["policy"])
    for field in ("source_qualification_policy_hash", "trusted_public_key_policy_hash"):
        assert getattr(result, field) == getattr(inputs["policy"], field)
    assert result.gate_version == "production_eligibility_gate_v2"
    assert result.schema_version == "production_eligibility_decision_v2"
    assert gate_hash(result) == result.artifact_hash
    assert verify_gate(result, **inputs) == ()


@pytest.mark.parametrize("status", (ValidationStatus.PENDING, ValidationStatus.FAIL))
def test_gate_v2_blocks_non_pass_execution(inputs, status):
    decision = decision_v2(p=inputs["policy"], validation_status=status,
                           reasons=("EVIDENCE_BLOCKED",))
    bridge = evaluate_execution_governance_bridge_v2(
        inputs["record"], decision, inputs["policy"], bridge_run_id="bridge-v2-run"
    )
    assert bridge.status == BLOCKED
    result = _blocked(inputs, "EXECUTION_VALIDATION_V2_NOT_PASS",
                      decision=decision, bridge=bridge)
    assert "GOVERNANCE_BRIDGE_V2_NOT_ACCEPTED" in result.reasons


def test_gate_v2_blocks_non_validated_registry(inputs):
    record = replace(inputs["record"], status=FactorStatus.CANDIDATE, transitions=())
    _blocked(inputs, "REGISTRY_NOT_VALIDATED", record=record)


@pytest.mark.parametrize("field,value", (
    ("factor_id", "different-factor"),
    ("factor_version", "2"),
    ("definition_hash", "f" * 64),
))
def test_gate_v2_blocks_registry_target_mismatch(inputs, field, value):
    decision = decision_v2(p=inputs["policy"], t=target(**{field: value}))
    _blocked(inputs, "REGISTRY_TARGET_IDENTITY_MISMATCH", decision=decision)


@pytest.mark.parametrize("component,reason", (
    ("decision", "EXECUTION_DECISION_V2_ARTIFACT_HASH_MISMATCH"),
    ("bridge", "EXECUTION_GOVERNANCE_BRIDGE_V2_HASH_MISMATCH"),
))
def test_gate_v2_blocks_tampered_upstream_hash(inputs, component, reason):
    tampered = replace(inputs[component], artifact_hash="f" * 64)
    _blocked(inputs, reason, **{component: tampered})


@pytest.mark.parametrize("field,value,reason", (
    ("policy_id", "other-policy", "EXECUTION_POLICY_V2_PROVENANCE_MISMATCH"),
    ("policy_version", "3", "EXECUTION_POLICY_V2_PROVENANCE_MISMATCH"),
    ("policy_hash", "e" * 64, "EXECUTION_POLICY_V2_PROVENANCE_MISMATCH"),
    ("source_qualification_policy_hash", "e" * 64,
     "SOURCE_QUALIFICATION_POLICY_HASH_MISMATCH"),
    ("trusted_public_key_policy_hash", "e" * 64,
     "TRUSTED_PUBLIC_KEY_POLICY_HASH_MISMATCH"),
    ("coverage_artifact_hash", "e" * 64,
     "GOVERNANCE_BRIDGE_V2_EXECUTION_DECISION_HASH_MISMATCH"),
    ("execution_run_id", "other-execution",
     "GOVERNANCE_BRIDGE_V2_EXECUTION_PROVENANCE_MISMATCH"),
    ("provenance_run_id", "other-provenance",
     "GOVERNANCE_BRIDGE_V2_EXECUTION_PROVENANCE_MISMATCH"),
    ("coverage_run_id", "other-coverage",
     "EXECUTION_GOVERNANCE_BRIDGE_V2_SEMANTICS_MISMATCH"),
))
def test_gate_v2_blocks_rehashed_decision_splice(inputs, field, value, reason):
    original = inputs["decision"]
    forged = _rehash(original, execution_validation_decision_v2_hash,
                     **{field: value})
    if field.endswith("_run_id"):
        assert forged.artifact_hash == original.artifact_hash
    result = _blocked(inputs, reason, decision=forged)
    assert "EXECUTION_DECISION_V2_ARTIFACT_HASH_MISMATCH" not in result.reasons


def test_gate_v2_blocks_different_registry_snapshot(inputs):
    record = validated_record(research_artifact_hash="e" * 64)
    assert registry_record_hash(record) != registry_record_hash(inputs["record"])
    _blocked(inputs, "GOVERNANCE_BRIDGE_V2_REGISTRY_HASH_MISMATCH", record=record)


def test_gate_v2_blocks_supplied_policy_change(inputs):
    _blocked(inputs, "EXECUTION_POLICY_V2_PROVENANCE_MISMATCH",
             policy=policy_v2(version="3"))


@pytest.mark.parametrize("field,value,reason", (
    ("target", target(alpha_id="other-alpha"), "GOVERNANCE_BRIDGE_V2_TARGET_MISMATCH"),
    ("registry_record_hash", "e" * 64, "GOVERNANCE_BRIDGE_V2_REGISTRY_HASH_MISMATCH"),
    ("execution_decision_artifact_hash", "e" * 64,
     "GOVERNANCE_BRIDGE_V2_EXECUTION_DECISION_HASH_MISMATCH"),
    ("execution_run_id", "other-execution",
     "GOVERNANCE_BRIDGE_V2_EXECUTION_PROVENANCE_MISMATCH"),
    ("provenance_run_id", "other-provenance",
     "GOVERNANCE_BRIDGE_V2_EXECUTION_PROVENANCE_MISMATCH"),
    ("policy_id", "other-policy", "GOVERNANCE_BRIDGE_V2_POLICY_PROVENANCE_MISMATCH"),
    ("policy_version", "3", "GOVERNANCE_BRIDGE_V2_POLICY_PROVENANCE_MISMATCH"),
    ("policy_hash", "e" * 64, "GOVERNANCE_BRIDGE_V2_POLICY_PROVENANCE_MISMATCH"),
    ("source_qualification_policy_hash", "e" * 64,
     "GOVERNANCE_BRIDGE_V2_SOURCE_POLICY_HASH_MISMATCH"),
    ("trusted_public_key_policy_hash", "e" * 64,
     "GOVERNANCE_BRIDGE_V2_TRUSTED_KEY_POLICY_HASH_MISMATCH"),
    ("coverage_run_id", "other-coverage",
     "EXECUTION_GOVERNANCE_BRIDGE_V2_SEMANTICS_MISMATCH"),
    ("research_run_id", "other-research",
     "EXECUTION_GOVERNANCE_BRIDGE_V2_SEMANTICS_MISMATCH"),
))
def test_gate_v2_blocks_rehashed_bridge_splice(inputs, field, value, reason):
    bridge = _rehash(inputs["bridge"], execution_governance_bridge_artifact_v2_hash,
                     **{field: value})
    result = _blocked(inputs, reason, bridge=bridge)
    assert "EXECUTION_GOVERNANCE_BRIDGE_V2_SEMANTICS_MISMATCH" in result.reasons
    assert "EXECUTION_GOVERNANCE_BRIDGE_V2_HASH_MISMATCH" not in result.reasons


def test_gate_v2_blocks_blocked_bridge_even_when_execution_passes(inputs):
    bridge = _rehash(inputs["bridge"], execution_governance_bridge_artifact_v2_hash,
                     status=BLOCKED, reasons=("TEST_BLOCKED",))
    _blocked(inputs, "GOVERNANCE_BRIDGE_V2_NOT_ACCEPTED", bridge=bridge)


def test_gate_v2_blocks_bridge_execution_status_mismatch(inputs):
    bridge = _rehash(inputs["bridge"], execution_governance_bridge_artifact_v2_hash,
                     status=BLOCKED, reasons=("TEST_BLOCKED",),
                     execution_validation_status=ValidationStatus.PENDING)
    _blocked(inputs, "GOVERNANCE_BRIDGE_V2_EXECUTION_STATUS_MISMATCH", bridge=bridge)


def test_gate_v2_checks_bridge_semantics_not_just_accepted_label_and_hash(inputs):
    # Keep PASS, ACCEPTED and every direct gate binding valid, but remove
    # the research transition. Only full bridge verification catches this.
    record = replace(inputs["record"], transitions=())
    bridge = _rehash(inputs["bridge"], execution_governance_bridge_artifact_v2_hash,
                     registry_record_hash=registry_record_hash(record))
    assert record.status == FactorStatus.VALIDATED
    assert bridge.status == ACCEPTED
    assert inputs["decision"].validation_status == ValidationStatus.PASS
    result = _blocked(inputs, "EXECUTION_GOVERNANCE_BRIDGE_V2_SEMANTICS_MISMATCH",
                      record=record, bridge=bridge)
    for unrelated in ("EXECUTION_GOVERNANCE_BRIDGE_V2_HASH_MISMATCH",
                      "REGISTRY_NOT_VALIDATED", "GOVERNANCE_BRIDGE_V2_NOT_ACCEPTED",
                      "GOVERNANCE_BRIDGE_V2_REGISTRY_HASH_MISMATCH",
                      "EXECUTION_VALIDATION_V2_NOT_PASS"):
        assert unrelated not in result.reasons


@pytest.mark.parametrize("component,expected", (
    ("decision", "ExecutionValidationDecisionV2"),
    ("policy", "ExecutionValidationPolicyV2"),
))
def test_gate_v2_rejects_real_v1_types(inputs, component, expected):
    v1_policy = ExecutionValidationPolicy(
        policy_id="execution-validation-policy", version="1",
        required_pass_evidence_kind=MEASURED_VENUE_EXECUTION,
    )
    v1_decision = ExecutionValidationDecision(
        execution_run_id="execution-v1-run", provenance_run_id="provenance-run",
        target=target(), validation_status=ValidationStatus.PENDING,
        reasons=("MEASURED_VENUE_EXECUTION_EVIDENCE_MISSING",),
        provenance_artifact_hash="e" * 64, policy_id=v1_policy.policy_id,
        policy_version=v1_policy.version, policy_hash="f" * 64,
        artifact_hash="a" * 64,
    )
    invalid = {**inputs, component: {"decision": v1_decision, "policy": v1_policy}[component]}
    with pytest.raises(TypeError, match=expected):
        _evaluate(invalid)
    with pytest.raises(TypeError, match=expected):
        verify_gate(_evaluate(inputs), **invalid)


@pytest.mark.parametrize("component,expected", (
    ("record", "FactorRegistryRecord"),
    ("bridge", "ExecutionGovernanceBridgeArtifactV2"),
))
def test_gate_v2_rejects_wrong_input_type(inputs, component, expected):
    invalid = {**inputs, component: object()}
    with pytest.raises(TypeError, match=expected):
        _evaluate(invalid)
    with pytest.raises(TypeError, match=expected):
        verify_gate(_evaluate(inputs), **invalid)


@pytest.mark.parametrize("recompute_hash", (False, True))
def test_gate_v2_verifier_rejects_forged_eligible(inputs, recompute_hash):
    decision = decision_v2(p=inputs["policy"], validation_status=ValidationStatus.FAIL,
                           reasons=("PROVENANCE_BLOCKED",))
    supplied = {**inputs, "decision": decision}
    blocked = _blocked(supplied, "EXECUTION_VALIDATION_V2_NOT_PASS")
    forged = replace(blocked, status=ELIGIBLE, reasons=())
    assert gate_hash(forged) != blocked.artifact_hash
    if recompute_hash:
        forged = _rehash(forged, gate_hash)
    reasons = verify_gate(forged, **supplied)
    assert "PRODUCTION_ELIGIBILITY_DECISION_V2_SEMANTICS_MISMATCH" in reasons
    assert ("PRODUCTION_ELIGIBILITY_DECISION_V2_HASH_MISMATCH" in reasons) == (not recompute_hash)


def test_gate_v2_verifier_rejects_tampered_gate_hash(inputs):
    forged = replace(_evaluate(inputs), artifact_hash="f" * 64)
    reasons = verify_gate(forged, **inputs)
    assert "PRODUCTION_ELIGIBILITY_DECISION_V2_HASH_MISMATCH" in reasons
    assert "PRODUCTION_ELIGIBILITY_DECISION_V2_SEMANTICS_MISMATCH" in reasons


@pytest.mark.parametrize("field,value", (
    ("target", target(alpha_id="other-alpha")),
    ("registry_record_hash", "e" * 64),
    ("governance_bridge_artifact_hash", "e" * 64),
    ("governance_bridge_run_id", "other-bridge"),
    ("execution_decision_artifact_hash", "e" * 64),
    ("execution_run_id", "other-execution"),
    ("provenance_run_id", "other-provenance"),
    ("policy_id", "other-policy"),
    ("policy_version", "3"),
    ("policy_hash", "e" * 64),
    ("source_qualification_policy_hash", "e" * 64),
    ("trusted_public_key_policy_hash", "e" * 64),
))
def test_gate_v2_hash_binds_fields_and_verifier_rejects_rehashed_splice(inputs, field, value):
    original = _evaluate(inputs)
    forged = _rehash(original, gate_hash, **{field: value})
    assert forged.artifact_hash != original.artifact_hash
    assert verify_gate(forged, **inputs) == (
        "PRODUCTION_ELIGIBILITY_DECISION_V2_SEMANTICS_MISMATCH",
    )


def test_gate_v2_hash_binds_blocked_reasons(inputs):
    supplied = {**inputs, "record": replace(inputs["record"], transitions=())}
    original = _blocked(supplied, "GOVERNANCE_BRIDGE_V2_REGISTRY_HASH_MISMATCH")
    forged = _rehash(original, gate_hash, reasons=("DIFFERENT_REASON",))
    assert forged.artifact_hash != original.artifact_hash
    assert verify_gate(forged, **supplied) == (
        "PRODUCTION_ELIGIBILITY_DECISION_V2_SEMANTICS_MISMATCH",
    )


def test_gate_v2_hash_excludes_gate_run_id(inputs):
    first = _evaluate(inputs, gate_run_id="gate-a")
    second = _evaluate(inputs, gate_run_id="gate-b")
    assert first.gate_run_id != second.gate_run_id
    assert first.artifact_hash == second.artifact_hash
    assert verify_gate(first, **inputs) == ()
    assert verify_gate(second, **inputs) == ()


@pytest.mark.parametrize("field", ("bridge_run_id", "execution_run_id", "provenance_run_id"))
def test_gate_v2_hash_changes_for_valid_upstream_run_provenance(inputs, field):
    decision = inputs["decision"]
    if field != "bridge_run_id":
        decision = _rehash(decision, execution_validation_decision_v2_hash,
                           **{field: "different-run"})
    bridge = evaluate_execution_governance_bridge_v2(
        inputs["record"], decision, inputs["policy"],
        bridge_run_id="different-run" if field == "bridge_run_id" else inputs["bridge"].bridge_run_id,
    )
    supplied = {**inputs, "decision": decision, "bridge": bridge}
    changed = _evaluate(supplied)
    assert changed.status == ELIGIBLE
    assert verify_gate(changed, **supplied) == ()
    assert changed.artifact_hash != _evaluate(inputs).artifact_hash


@pytest.mark.parametrize("blocked", (False, True))
def test_gate_v2_is_deterministic_and_does_not_mutate_inputs(inputs, blocked):
    supplied = dict(inputs)
    if blocked:
        supplied["record"] = replace(inputs["record"], transitions=())
    before = {name: asdict(value) for name, value in supplied.items()}
    first = _evaluate(supplied)
    second = _evaluate(supplied)
    assert first == second
    assert first.status == (GATE_BLOCKED if blocked else ELIGIBLE)
    assert {name: asdict(value) for name, value in supplied.items()} == before


@pytest.mark.parametrize("field,value,error", (
    ("gate_run_id", "", ValueError),
    ("gate_run_id", 123, TypeError),
    ("target", object(), TypeError),
    ("status", "UNKNOWN", ValueError),
    ("status", GATE_BLOCKED, ValueError),
    ("reasons", ("NOT_ALLOWED_FOR_ELIGIBLE",), ValueError),
    ("artifact_hash", "invalid", ValueError),
    ("gate_version", "wrong-version", ValueError),
    ("schema_version", "wrong-schema", ValueError),
))
def test_gate_v2_artifact_rejects_invalid_structure(inputs, field, value, error):
    with pytest.raises(error):
        replace(_evaluate(inputs), **{field: value})
