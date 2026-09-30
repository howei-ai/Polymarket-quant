import pytest

from std0_quant.research.agent_shadow.schemas import AnalystReport, GateEvidence, ManagerReport


def report(**changes):
    value = {"role": "MOMENTUM", "direction": "UP", "confidence": 0.8,
             "evidence": ["pre-cutoff return"], "risks": [], "abstain": False}
    value.update(changes)
    return value


@pytest.mark.parametrize("change", [
    {"direction": "BUY"}, {"confidence": -0.1}, {"confidence": 1.1},
    {"confidence": float("nan")}, {"confidence": True}, {"abstain": "false"},
    {"role": "LIQUIDITY"}, {"extra": "submit_order"},
])
def test_malformed_k3_report_rejected(change):
    with pytest.raises(ValueError):
        AnalystReport.parse(report(**change), "MOMENTUM")


def test_gate_evidence_unknown_is_representable_but_not_truthy():
    assert GateEvidence(None, True, True).coverage_valid is False
    with pytest.raises(TypeError):
        GateEvidence(1, True, True)


def test_manager_invalid_roles_rejected():
    with pytest.raises(ValueError):
        ManagerReport.parse({"direction": "UP", "confidence": 0.8,
                             "supporting_roles": ["TRADER"], "contradicting_roles": [],
                             "key_risks": []})
