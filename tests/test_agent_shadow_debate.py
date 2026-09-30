import pytest

from agent_shadow_fakes import FakeJudge, FakeReasoner, pit_row
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import AnalystReport, GateEvidence


def _debate_report(role, direction, abstain=False):
    return {"role": role, "direction": direction, "confidence": 0.8,
            "evidence": ["pre-cutoff state"], "risks": [], "abstain": abstain}


def test_bull_non_abstain_down_is_rejected():
    with pytest.raises(ValueError):
        AnalystReport.parse(_debate_report("BULL_RESEARCHER", "DOWN"), "BULL_RESEARCHER")


def test_bear_non_abstain_up_is_rejected():
    with pytest.raises(ValueError):
        AnalystReport.parse(_debate_report("BEAR_RESEARCHER", "UP"), "BEAR_RESEARCHER")


def test_bull_abstain_requires_unclear():
    with pytest.raises(ValueError):
        AnalystReport.parse(_debate_report("BULL_RESEARCHER", "UP", True), "BULL_RESEARCHER")
    assert AnalystReport.parse(_debate_report("BULL_RESEARCHER", "UNCLEAR", True), "BULL_RESEARCHER").abstain


def test_bear_abstain_requires_unclear():
    with pytest.raises(ValueError):
        AnalystReport.parse(_debate_report("BEAR_RESEARCHER", "DOWN", True), "BEAR_RESEARCHER")
    assert AnalystReport.parse(_debate_report("BEAR_RESEARCHER", "UNCLEAR", True), "BEAR_RESEARCHER").abstain


def test_swapped_bull_bear_end_to_end_blocks():
    k3 = FakeReasoner({
        "BULL_RESEARCHER": _debate_report("BULL_RESEARCHER", "DOWN"),
        "BEAR_RESEARCHER": _debate_report("BEAR_RESEARCHER", "UP"),
    })
    jev = FakeJudge()
    result = ShadowRunner(k3, jev).run(pit_row(), GateEvidence(True, True, True))
    assert result.gate_status == "BLOCKED"
    assert "K3_OUTPUTS_INVALID" in result.gate_reasons
    assert jev.requests == []


def test_exactly_one_fixed_round_and_order():
    k3 = FakeReasoner()
    ShadowRunner(k3, FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    assert [request["role"] for request in k3.requests] == [
        "MICROSTRUCTURE", "MOMENTUM", "LIQUIDITY", "REGIME",
        "BULL_RESEARCHER", "BEAR_RESEARCHER", "RESEARCH_MANAGER",
    ]
    assert all(request.get("round", 1) == 1 for request in k3.requests)
    with pytest.raises(ValueError):
        ShadowRunner(FakeReasoner(), FakeJudge(), rounds=2)


def test_manager_cannot_claim_abstaining_role_as_support():
    abstainer = {"role": "MOMENTUM", "direction": "UP", "confidence": 0.8,
                 "evidence": [], "risks": [], "abstain": True}
    result = ShadowRunner(FakeReasoner({"MOMENTUM": abstainer}), FakeJudge()).run(
        pit_row(), GateEvidence(True, True, True))
    assert result.gate_status == "BLOCKED"
    assert "K3_OUTPUTS_INVALID" in result.gate_reasons


def test_manager_cannot_claim_opposing_role_as_support():
    opposing = {"role": "MOMENTUM", "direction": "DOWN", "confidence": 0.8,
                "evidence": [], "risks": [], "abstain": False}
    result = ShadowRunner(FakeReasoner({"MOMENTUM": opposing}), FakeJudge()).run(
        pit_row(), GateEvidence(True, True, True))
    assert result.gate_status == "BLOCKED"
