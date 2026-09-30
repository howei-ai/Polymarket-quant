import pytest

from agent_shadow_fakes import FakeJudge, FakeReasoner, judge_response, pit_row
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import GateEvidence


@pytest.mark.parametrize("evidence,reason", [
    (GateEvidence(False, True, True), "COVERAGE_INVALID_OR_UNKNOWN"),
    (GateEvidence(None, True, True), "COVERAGE_INVALID_OR_UNKNOWN"),
    (GateEvidence(True, False, True), "PROVENANCE_INVALID_OR_UNKNOWN"),
    (GateEvidence(True, True, None), "SANITY_INVALID_OR_UNKNOWN"),
])
def test_hard_invariant_blocks_before_models(evidence, reason):
    k3, jev = FakeReasoner(), FakeJudge()
    result = ShadowRunner(k3, jev).run(pit_row(), evidence)
    assert result.gate_status == "BLOCKED" and reason in result.gate_reasons
    assert k3.requests == [] and jev.requests == []


@pytest.mark.parametrize("judge", [
    judge_response(candidate_quality={"score": 0.1}),
    judge_response(execution_risk={"level": "HIGH", "confidence": 0.9}),
    judge_response(book_state_anomalous={"answer": True, "probability": 0.9}),
])
def test_valid_but_unfavorable_judgment_rejects(judge):
    result = ShadowRunner(FakeReasoner(), FakeJudge(judge)).run(
        pit_row(), GateEvidence(True, True, True))
    assert result.gate_status == "SHADOW_REJECT"


def test_low_manager_confidence_rejects():
    manager = {"direction": "UP", "confidence": 0.1,
               "supporting_roles": ["MICROSTRUCTURE", "MOMENTUM"],
               "contradicting_roles": ["BEAR_RESEARCHER"], "key_risks": []}
    result = ShadowRunner(FakeReasoner({"RESEARCH_MANAGER": manager}), FakeJudge()).run(
        pit_row(), GateEvidence(True, True, True))
    assert result.gate_status == "SHADOW_REJECT"


@pytest.mark.parametrize("row", [
    pit_row(btc_pre30_coverage_pct=0.98),
    pit_row(book_pre10_coverage_pct=None),
])
def test_reported_coverage_true_cannot_bypass_public_99_percent_gate(row):
    k3, jev = FakeReasoner(), FakeJudge()
    result = ShadowRunner(k3, jev).run(row, GateEvidence(True, True, True))
    assert result.gate_status == "BLOCKED"
    assert "COVERAGE_INVALID_OR_UNKNOWN" in result.gate_reasons
    assert k3.requests == [] and jev.requests == []


def test_provenance_false_blocks_even_if_models_are_positive():
    k3, jev = FakeReasoner(), FakeJudge()
    result = ShadowRunner(k3, jev).run(pit_row(), GateEvidence(True, False, True))
    assert result.gate_status == "BLOCKED"
    assert result.gate_reasons == ("PROVENANCE_INVALID_OR_UNKNOWN",)
    assert k3.requests == [] and jev.requests == []


def test_sanity_false_blocks_even_if_models_are_positive():
    k3, jev = FakeReasoner(), FakeJudge()
    result = ShadowRunner(k3, jev).run(pit_row(), GateEvidence(True, True, False))
    assert result.gate_status == "BLOCKED"
    assert result.gate_reasons == ("SANITY_INVALID_OR_UNKNOWN",)
    assert k3.requests == [] and jev.requests == []
