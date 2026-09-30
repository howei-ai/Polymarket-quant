import pytest

from agent_shadow_fakes import FakeJudge, FakeReasoner, pit_row
from std0_quant.research.agent_shadow.k3_client import K3Client
from std0_quant.research.agent_shadow.interfaces import ModelUnavailable
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import GateEvidence


def test_adapter_network_off_by_default():
    calls = []
    client = K3Client(lambda request: calls.append(request) or {})
    with pytest.raises(ModelUnavailable):
        client.analyze({"role": "MOMENTUM"})
    assert calls == []


def test_malformed_k3_blocks_before_jev():
    k3 = FakeReasoner({"MOMENTUM": {"role": "MOMENTUM", "direction": "BUY NOW",
                                      "confidence": 1, "evidence": [], "risks": [], "abstain": False}})
    jev = FakeJudge()
    result = ShadowRunner(k3, jev).run(pit_row(), GateEvidence(True, True, True))
    assert result.gate_status == "BLOCKED" and "K3_OUTPUTS_INVALID" in result.gate_reasons
    assert jev.requests == []


def test_k3_timeout_blocks():
    result = ShadowRunner(FakeReasoner(error=TimeoutError()), FakeJudge()).run(
        pit_row(), GateEvidence(True, True, True))
    assert result.gate_status == "BLOCKED" and "K3_TIMEOUT" in result.gate_reasons
