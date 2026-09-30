from copy import deepcopy

from agent_shadow_fakes import FakeJudge, FakeReasoner, pit_row
from std0_quant.research.agent_shadow.analysts import analyst_request
from std0_quant.research.agent_shadow.audit import hash_k3_request
from std0_quant.research.agent_shadow.context import build_shadow_context
from std0_quant.research.agent_shadow.role_instructions import ROLE_INSTRUCTIONS
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import GateEvidence, K3_ROLES


def test_each_k3_role_has_distinct_instruction():
    k3 = FakeReasoner()
    ShadowRunner(k3, FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    actual = {request["role"]: request["instruction"] for request in k3.requests}
    assert set(actual) == set(K3_ROLES) == set(ROLE_INSTRUCTIONS)
    assert len(set(actual.values())) == len(K3_ROLES)
    for role, instruction in actual.items():
        assert instruction == ROLE_INSTRUCTIONS[role]


def test_k3_request_hash_binds_role_instruction():
    context = build_shadow_context(pit_row())
    request = analyst_request("MOMENTUM", context)
    assert hash_k3_request("kimi-k3", request) != hash_k3_request("other-k3", request)
    assert request["role"] == "MOMENTUM"
    assert request["context"]["context_sha256"] == context.context_sha256
    assert request["schema_version"]


def test_changed_role_instruction_changes_request_hash():
    request = analyst_request("MICROSTRUCTURE", build_shadow_context(pit_row()))
    modified = deepcopy(request)
    modified["instruction"] += " Changed instruction."
    assert hash_k3_request("kimi-k3", request) != hash_k3_request("kimi-k3", modified)


def test_runner_records_model_bound_k3_request_hash():
    k3 = FakeReasoner()
    result = ShadowRunner(k3, FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    for request in k3.requests:
        assert result.k3_request_hashes[request["role"]] == hash_k3_request(k3.model_id, request)
