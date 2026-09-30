from copy import deepcopy

import pytest

from agent_shadow_fakes import FakeJudge, FakeReasoner, pit_row
from std0_quant.research.agent_shadow.audit import sha256_json
from std0_quant.research.agent_shadow.jev_questions import (
    JEV_QUESTION_SCHEMA_VERSION, JEV_QUESTION_SPECS, build_jev_request,
    question_specs_payload,
)
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import GateEvidence


def _request(**overrides):
    return build_jev_request("fake-jev-1.0.0", {"pit": "pre-cutoff"}, **overrides)


def test_jev_request_hash_binds_full_question_specs():
    request = _request()
    assert request["questions"]["schema_version"] == JEV_QUESTION_SCHEMA_VERSION
    assert set(request["questions"]["specs"]) == set(JEV_QUESTION_SPECS)
    assert all(set(spec) == {"type", "instructions", "criteria"} for spec in request["questions"]["specs"].values())
    assert len(sha256_json(request)) == 64


def test_changed_jev_instruction_changes_request_hash():
    specs = question_specs_payload()
    specs["directional_evidence_consistent"]["instructions"] += " Additional bounded instruction."
    assert sha256_json(_request()) != sha256_json(_request(specs=specs))


def test_changed_jev_criteria_changes_request_hash():
    specs = question_specs_payload()
    specs["execution_risk"]["criteria"]["high"] += " Additional PIT criterion."
    assert sha256_json(_request()) != sha256_json(_request(specs=specs))


def test_changed_jev_schema_version_changes_request_hash():
    assert sha256_json(_request()) != sha256_json(_request(schema_version="jev-shadow-q-v2"))


def test_changed_jev_model_id_changes_request_hash():
    assert sha256_json(_request()) != sha256_json(build_jev_request("fake-jev-2.0.0", {"pit": "pre-cutoff"}))


def test_jev_question_names_alone_are_not_hash_contract():
    names = tuple(JEV_QUESTION_SPECS)
    assert sha256_json(names) != sha256_json(_request())


def test_runner_hashes_exact_full_request_and_passes_specs_to_fake_judge():
    judge = FakeJudge()
    result = ShadowRunner(FakeReasoner(), judge).run(pit_row(), GateEvidence(True, True, True))
    state, questions = judge.requests[0]
    expected = build_jev_request(judge.model_id, state)
    assert questions == expected["questions"]
    assert result.jev_request_hash == sha256_json(expected)
    assert result.jev_question_schema_version == JEV_QUESTION_SCHEMA_VERSION


def test_question_spec_payload_isolated_from_frozen_source():
    copy = deepcopy(question_specs_payload())
    copy["candidate_quality"]["instructions"] = "mutated"
    assert question_specs_payload()["candidate_quality"]["instructions"] != "mutated"
    with pytest.raises(TypeError):
        JEV_QUESTION_SPECS["candidate_quality"]["criteria"]["1"] = "mutated"


def test_question_spec_type_and_criteria_keys_cannot_be_spliced():
    bad_type = question_specs_payload()
    bad_type["book_state_anomalous"]["type"] = "score"
    with pytest.raises(ValueError):
        _request(specs=bad_type)
    bad_criteria = question_specs_payload()
    bad_criteria["execution_risk"]["criteria"] = {"buy": "now"}
    with pytest.raises(ValueError):
        _request(specs=bad_criteria)
