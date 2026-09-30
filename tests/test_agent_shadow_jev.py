import pytest

from agent_shadow_fakes import FakeJudge, FakeReasoner, judge_response, pit_row
from std0_quant.research.agent_shadow.interfaces import ModelUnavailable
from std0_quant.research.agent_shadow.jev_client import JevClient
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import GateEvidence, JevJudgments


def test_jev_adapter_disabled_and_latest_rejected():
    called = []
    client = JevClient(model_id="jev-1.13.0", transport=lambda state, questions: called.append(questions) or {})
    with pytest.raises(ModelUnavailable):
        client.judge({}, ())
    with pytest.raises(ValueError):
        JevClient(model_id="jev-latest")
    assert called == []


@pytest.mark.parametrize("change", [
    {"directional_evidence_consistent": {"answer": "YES", "probability": 0.8}},
    {"book_state_anomalous": {"answer": None, "probability": 0.1}},
    {"candidate_quality": {"score": float("inf")}},
    {"execution_risk": {"level": "BUY", "confidence": 0.8}},
    {"regime": {"label": "UNKNOWN", "confidence": 0.8}},
])
def test_jev_malformed_typed_response_rejected(change):
    with pytest.raises(ValueError):
        JevJudgments.parse(judge_response(**change))


def test_jev_timeout_blocks_and_records_request():
    result = ShadowRunner(FakeReasoner(), FakeJudge(error=TimeoutError())).run(
        pit_row(), GateEvidence(True, True, True))
    assert result.gate_status == "BLOCKED" and "JEV_TIMEOUT" in result.gate_reasons
    assert len(result.jev_request_hash) == 64 and len(result.jev_response_hash) == 64


def test_real_jev_client_requires_explicit_model_id():
    with pytest.raises(TypeError):
        JevClient()
    assert JevClient(model_id="jev-1.13.0").model_id == "jev-1.13.0"


@pytest.mark.parametrize("model_id", ["jev-latest", "latest", "JEV-LATEST", ""])
def test_real_jev_client_rejects_latest_alias(model_id):
    with pytest.raises(ValueError):
        JevClient(model_id=model_id)


@pytest.mark.parametrize("model_id", ["jev-1", "jev-1.13", "jev-unconfigured-v1", "other-1.2.3"])
def test_real_jev_client_rejects_unversioned_id(model_id):
    with pytest.raises(ValueError):
        JevClient(model_id=model_id)
