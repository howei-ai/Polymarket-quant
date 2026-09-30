from agent_shadow_fakes import FakeJudge, FakeReasoner, pit_row
from dataclasses import replace
from std0_quant.research.agent_shadow.audit import sha256_json
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import GateEvidence


def test_offline_pipeline_accept_and_hash_reproducibility():
    judge = FakeJudge()
    a = ShadowRunner(FakeReasoner(), judge).run(pit_row(), GateEvidence(True, True, True))
    b = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    assert a.gate_status == "SHADOW_ACCEPT"
    assert a.artifact_sha256 == b.artifact_sha256
    assert a.execution_allowed is False and a.formal_cohort_effect is False
    assert len(a.k3_request_hashes) == len(a.k3_response_hashes) == 7
    assert len(a.jev_request_hash) == len(a.jev_response_hash) == 64
    state, questions = judge.requests[0]
    assert len(state["analyst_reports"]) == 4
    assert state["bull_report"]["role"] == "BULL_RESEARCHER"
    assert state["bear_report"]["role"] == "BEAR_RESEARCHER"
    assert questions["model_id"] == judge.model_id
    assert questions["schema_version"] == "jev-shadow-q-v1"
    assert set(questions["specs"]) == {
        "directional_evidence_consistent", "book_state_anomalous",
        "execution_risk", "regime", "candidate_quality",
    }


def test_shadow_result_records_gate_evidence():
    result = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    assert result.to_dict()["gate_evidence"] == {
        "coverage_valid": True, "provenance_valid": True, "sanity_valid": True,
    }


def test_gate_evidence_false_is_preserved_in_artifact():
    result = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, False, True))
    assert result.gate_status == "BLOCKED"
    assert result.to_dict()["gate_evidence"]["provenance_valid"] is False


def test_gate_evidence_changes_artifact_hash():
    a = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, False, True))
    b = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, True, False))
    assert a.k3_response_hashes == b.k3_response_hashes == {}
    assert a.jev_response_hash == b.jev_response_hash
    assert a.artifact_sha256 != b.artifact_sha256


def test_same_model_response_hashes_different_evidence_changes_artifact_hash():
    blocked = ShadowRunner(FakeReasoner(error=TimeoutError()), FakeJudge()).run(
        pit_row(), GateEvidence(True, True, True))
    different_evidence = replace(blocked, gate_evidence=GateEvidence(True, False, True))
    assert blocked.k3_response_hashes == different_evidence.k3_response_hashes
    assert blocked.jev_response_hash == different_evidence.jev_response_hash
    assert blocked.artifact_sha256 != different_evidence.artifact_sha256


def test_artifact_hash_recomputes_exactly():
    result = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    payload = result.to_dict()
    reported_hash = payload.pop("artifact_sha256")
    assert reported_hash == sha256_json(payload)


def test_artifact_hash_changes_if_jev_spec_version_changes(monkeypatch):
    from std0_quant.research.agent_shadow import runner
    first = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    monkeypatch.setattr(runner, "JEV_QUESTION_SCHEMA_VERSION", "jev-shadow-q-v2")
    second = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    assert first.artifact_sha256 != second.artifact_sha256
    assert first.jev_request_hash != second.jev_request_hash


def test_duplicate_condition_id_blocked_without_model_call():
    k3, jev = FakeReasoner(), FakeJudge()
    runner = ShadowRunner(k3, jev)
    runner.run(pit_row(), GateEvidence(True, True, True))
    result = runner.run(pit_row(), GateEvidence(True, True, True))
    assert result.gate_status == "BLOCKED"
    assert "DUPLICATE_CONDITION_ID" in result.gate_reasons
    assert len(k3.requests) == 7 and len(jev.requests) == 1


def test_leakage_blocked_before_any_model_call():
    k3, jev = FakeReasoner(), FakeJudge()
    result = ShadowRunner(k3, jev).run(pit_row(y30=1), GateEvidence(True, True, True))
    assert result.gate_status == "BLOCKED"
    assert k3.requests == [] and jev.requests == []
