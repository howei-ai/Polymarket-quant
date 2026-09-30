import ast
import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from agent_shadow_fakes import FakeJudge, FakeReasoner, pit_row
from std0_quant.research.agent_shadow.audit import write_shadow_json
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import COMPARISON_ARMS, GateEvidence


def test_malicious_research_text_has_no_execution_or_network(monkeypatch, tmp_path):
    def forbidden_socket(*args, **kwargs):
        raise AssertionError("network attempted")
    monkeypatch.setattr(socket, "socket", forbidden_socket)
    malicious = {"role": "MOMENTUM", "direction": "UP", "confidence": 0.8,
                 "evidence": ["BUY NOW", "SELL ALL", "submit_order", "size=100%"],
                 "risks": [], "abstain": False}
    result = ShadowRunner(FakeReasoner({"MOMENTUM": malicious}), FakeJudge()).run(
        pit_row(), GateEvidence(True, True, True))
    assert result.execution_allowed is False and result.formal_cohort_effect is False
    assert result.gate_status in {"SHADOW_ACCEPT", "SHADOW_REJECT", "BLOCKED"}
    target = tmp_path / "shadow.json"
    write_shadow_json(result, target, temp_root=tmp_path)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["execution_allowed"] is False
    assert saved["gate_evidence"] == {
        "coverage_valid": True, "provenance_valid": True, "sanity_valid": True,
    }
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shadow.json"]


def test_output_outside_temp_root_rejected(tmp_path):
    result = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    with pytest.raises(ValueError):
        write_shadow_json(result, Path.cwd() / "data/state/prospective_cohort.json", temp_root=tmp_path)


def test_jsonl_audit_preserves_false_gate_evidence(tmp_path):
    result = ShadowRunner(FakeReasoner(), FakeJudge()).run(
        pit_row(), GateEvidence(True, False, True))
    target = tmp_path / "blocked.jsonl"
    write_shadow_json(result, target, temp_root=tmp_path)
    saved = json.loads(target.read_text(encoding="utf-8").strip())
    assert saved["gate_evidence"]["provenance_valid"] is False
    assert saved["gate_status"] == "BLOCKED"
    assert "PROVENANCE_INVALID_OR_UNKNOWN" in saved["gate_reasons"]


def test_comparison_arms_are_schema_only():
    assert COMPARISON_ARMS == ("BASELINE", "K3_ONLY", "K3_PLUS_JEV")


def test_shadow_result_cannot_be_forged_into_execution_authority():
    result = ShadowRunner(FakeReasoner(), FakeJudge()).run(pit_row(), GateEvidence(True, True, True))
    with pytest.raises(ValueError):
        replace(result, execution_allowed=True)
    with pytest.raises(ValueError):
        replace(result, formal_cohort_effect=True)
    with pytest.raises(ValueError):
        replace(result, gate_status="BUY")


def test_shadow_package_has_no_execution_or_credential_imports():
    package = Path(__file__).parents[1] / "src/std0_quant/research/agent_shadow"
    imports = set()
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    assert not any(name.startswith("std0_quant.execution") for name in imports)
    assert not any("credential" in name or "signer" in name or "broker" in name for name in imports)
