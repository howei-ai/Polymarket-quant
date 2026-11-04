from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from agent_shadow_fakes import judge_response, pit_row
from agent_shadow_provider_fakes import FakeOpenAIK3Client, FakeOpenAIClient
from std0_quant.research.agent_shadow import provider_clients
from std0_quant.research.agent_shadow.audit import canonical_json, sha256_json
from std0_quant.research.agent_shadow.model_registry import CostBudget
from std0_quant.research.agent_shadow.replay import (
    REDACTED_CUTOFF_MS, ReplayRow, ReplayWriter, redacted_condition_id, replay_run_binding, run_replay,
)
from std0_quant.research.agent_shadow.replay_report import ReplayReportError, summarize_replay
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import GateEvidence

ROOT = Path(__file__).resolve().parents[1]


def _attempt(directory, cid="private-condition-ABC", *, fake=None, jev_fake=None, budget_usd=5.0, evidence=None):
    redacted = redacted_condition_id(cid)
    row = ReplayRow(cid, redacted,
                    pit_row(condition_id=redacted, decision_cutoff_ts_ms=REDACTED_CUTOFF_MS,
                            btc_observed_ts_ms=REDACTED_CUTOFF_MS - 100,
                            book_observed_ts_ms=REDACTED_CUTOFF_MS - 200),
                    evidence or GateEvidence(True, True, True), (), 1_700_000_000_000)
    budget = CostBudget(budget_usd)
    models = provider_clients.build_shadow_models(fake or FakeOpenAIK3Client(), jev_fake or FakeOpenAIClient(),
                                                budget=budget, network_enabled=True, allow_test_clients=True,
                                                route="relay_mtu")
    writer = ReplayWriter(directory, forbidden_roots=(ROOT,))
    run_replay([row], ShadowRunner(*models, allow_unverified_relay=True), writer, budget,
               max_rows=1, row_headroom_usd=0)
    return writer


def _edit(directory, mutate, *, resign=True):
    path = directory / "artifacts.jsonl"
    payload = json.loads(path.read_text())
    mutate(payload)
    if resign:
        data = {key: value for key, value in payload.items() if key != "artifact_sha256"}
        payload["artifact_sha256"] = sha256_json(data)
    path.write_text(canonical_json(payload) + "\n")


def test_accepts_real_worker_artifacts_without_identifiers_or_writes(tmp_path):
    directory = tmp_path / "replay"
    writer = _attempt(directory)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    report = summarize_replay(directory)
    assert (report["attempts"], report["rows"], report["final_rows"], report["unresolved_rows"]) == (1, 1, 1, 0)
    assert report["final_status_counts"] == {"SHADOW_ACCEPT": 1} and report["final_accept_rate"] == 1.0
    assert (report["route"], report["provenance"]) == ("relay_mtu", "TEST_CLIENT")
    assert report["model_calls"]["records"] == 8 and report["model_calls"]["unpriced_records"] == 0
    assert report["model_calls_by_role"]["MICROSTRUCTURE"]["input_tokens_known_sum"] == 1200
    payload = json.loads(writer.artifacts.read_text())
    expected_cost = sum(Decimal(str(call["cost_usd"])) for call in payload["model_calls"])
    assert Decimal(report["model_calls"]["priced_cost_usd"]) == expected_cost
    for secret in ("private-condition-ABC", redacted_condition_id("private-condition-ABC"),
                   "req_k3_1", "req_openai_1", "1700000000000"):
        assert secret not in json.dumps(report)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert report["sources"]["artifacts_sha256"] == hashlib.sha256(writer.artifacts.read_bytes()).hexdigest()
    assert report["report_sha256"] == sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    assert report["execution_allowed"] is False and report["formal_cohort_effect"] is False


def test_retry_costs_are_counted_but_final_row_is_counted_once(tmp_path):
    directory = tmp_path / "replay"
    _attempt(directory, fake=FakeOpenAIK3Client(error=TimeoutError()))
    _attempt(directory)
    report = summarize_replay(directory)
    assert (report["attempts"], report["nonfinal_attempts"], report["rows"], report["final_rows"]) == (2, 1, 1, 1)
    assert report["attempt_status_counts"] == {"BLOCKED": 1, "SHADOW_ACCEPT": 1}
    assert report["latest_status_counts"] == report["final_status_counts"] == {"SHADOW_ACCEPT": 1}
    assert report["attempt_gate_reason_counts"] == {"K3_TIMEOUT": 1} and report["final_gate_reason_counts"] == {}
    assert report["attempt_count_histogram"] == {"2": 1}
    calls = report["model_calls"]
    assert calls["records"] == 9 and calls["upper_bound_records"] == 1 and calls["input_tokens_unknown_records"] == 1


def test_budget_cut_stays_unresolved_and_unpriced_record_is_visible(tmp_path):
    directory = tmp_path / "replay"
    _attempt(directory, budget_usd=1.0)
    report = summarize_replay(directory)
    assert report["final_rows"] == 0 and report["unresolved_rows"] == 1 and report["final_accept_rate"] is None
    assert report["final_status_counts"] == {} and report["latest_status_counts"] == {"BLOCKED": 1}
    assert report["model_calls"]["outcomes"]["BUDGET_EXCEEDED"] == 1
    assert report["model_calls"]["unpriced_records"] == 1 and report["model_calls"]["all_records_have_cost"] is False


def test_unknown_full_cost_is_not_converted_to_a_priced_record(tmp_path):
    directory = tmp_path / "replay"
    _attempt(directory, fake=FakeOpenAIK3Client(usage=(10 ** 315, 300), cache_tokens=None))
    calls = summarize_replay(directory)["model_calls"]
    assert calls["records"] == calls["unpriced_records"] == calls["budget_overrun_records"] == 2
    assert calls["outcomes"] == {"OK": 1, "BUDGET_EXCEEDED": 1}
    assert calls["priced_records"] == 0 and calls["priced_cost_usd"] == "0" and not calls["all_records_have_cost"]


def test_cost_totals_can_exceed_float_range(tmp_path):
    directory = tmp_path / "replay"
    for cid in ("a", "b", "c"):
        _attempt(directory, cid, fake=FakeOpenAIK3Client(usage=(4 * 10 ** 312, 300), cache_tokens=None))
    report = summarize_replay(directory)
    assert Decimal(report["model_calls"]["priced_cost_usd"]) == Decimal("3e308")
    assert report["unresolved_rows"] == 3
    canonical_json(report)


def test_accept_rate_uses_final_rows_and_includes_final_blocks(tmp_path):
    directory = tmp_path / "replay"
    _attempt(directory, "accepted")
    reply = judge_response(candidate_quality={"score": 0.1})
    _attempt(directory, "rejected", jev_fake=FakeOpenAIClient(reply=json.dumps(reply)))
    _attempt(directory, "ineligible", evidence=GateEvidence(False, True, True))
    report = summarize_replay(directory)
    assert report["final_status_counts"] == {"SHADOW_ACCEPT": 1, "SHADOW_REJECT": 1, "BLOCKED": 1}
    assert report["final_accept_rate"] == pytest.approx(1 / 3)
    assert report["evidence_invalid_attempt_counts"] == {"coverage_valid": 1}
    assert report["final_gate_reason_counts"] == {"JEV_LOW_CANDIDATE_QUALITY": 1, "COVERAGE_INVALID_OR_UNKNOWN": 1}


@pytest.mark.parametrize("change", [{"direction": "DOWN"}, {"abstain": True}])
def test_manager_support_must_match_the_cited_report(tmp_path, change):
    directory = tmp_path / "replay"
    _attempt(directory)
    def mutate(payload):
        report = payload["analyst_reports"][1]
        assert report["role"] in payload["manager_report"]["supporting_roles"]
        report.update(change)
        payload["k3_response_hashes"][report["role"]] = sha256_json(report)
    _edit(directory, mutate)
    with pytest.raises(ReplayReportError, match="invalid artifact at line 1"):
        summarize_replay(directory)


@pytest.mark.parametrize("change", [
    {"cost_usd": 0.0},
    {"cost_usd": 0.0, "output_tokens": None},
    {"cost_usd": 0.0, "input_tokens": None},
    {"cost_usd": 0.027, "billable_input_tokens": 2400},
], ids=["raw-and-output", "input-only", "output-only", "billable-input"])
def test_estimated_cost_cannot_be_below_known_usage(tmp_path, change):
    directory = tmp_path / "replay"
    _attempt(directory, fake=FakeOpenAIK3Client(cache_tokens=None), budget_usd=100)
    def mutate(payload):
        call = payload["model_calls"][0]
        assert call["cost_is_upper_bound"] is True
        assert (call["input_tokens"], call["output_tokens"], call["billable_input_tokens"]) == (1200, 300, None)
        call.update(change)
    _edit(directory, mutate)
    with pytest.raises(ReplayReportError, match="invalid artifact at line 1"):
        summarize_replay(directory)


@pytest.mark.parametrize("field", ["k3_request_hashes", "k3_response_hashes"])
@pytest.mark.parametrize("missing", ["all", "one"])
def test_completed_call_hashes_must_cover_every_k3_role(tmp_path, field, missing):
    directory = tmp_path / "replay"
    _attempt(directory)
    def mutate(payload):
        if missing == "all":
            payload[field].clear()
        else:
            del payload[field]["MOMENTUM"]
    _edit(directory, mutate)
    with pytest.raises(ReplayReportError, match="invalid artifact at line 1"):
        summarize_replay(directory)


@pytest.mark.parametrize("mutate,resign", [
    (lambda p: p.update(gate_status="SHADOW_REJECT"), False),
    (lambda p: p.update(execution_allowed=True), True),
    (lambda p: p.update(provenance="UNVERIFIED_RELAY"), True),
    (lambda p: p.update(model_calls=[]), True),
    (lambda p: p.update(k3_provider="unspecified"), True),
    (lambda p: p["model_calls"][0].update(cost_usd=0), True),
    (lambda p: p["model_calls"][0].update(test_client=False), True),
    (lambda p: p.update(gate_reasons=["PRIVATE_ROW_SECRET"]), True),
    (lambda p: p.update(decision_cutoff_ts=1_700_000_000_000), True),
    (lambda p: p.update(schema_version="legacy"), True),
])
def test_tampered_or_inconsistent_artifacts_are_refused(tmp_path, mutate, resign):
    directory = tmp_path / "replay"
    _attempt(directory)
    _edit(directory, mutate, resign=resign)
    with pytest.raises(ReplayReportError, match="invalid artifact at line 1"):
        summarize_replay(directory)


@pytest.mark.parametrize("kind", ["torn", "no_newline", "duplicate_key", "after_final"])
def test_incomplete_or_ambiguous_jsonl_is_refused(tmp_path, kind):
    directory = tmp_path / "replay"
    writer = _attempt(directory)
    line = writer.artifacts.read_text()
    if kind == "torn":
        line += '{"PRIVATE_ROW_SECRET":'
    elif kind == "no_newline":
        line = line.rstrip("\n")
    elif kind == "duplicate_key":
        line = line.replace('{', '{"execution_allowed":false,', 1)
    else:
        line += line
    writer.artifacts.write_text(line)
    with pytest.raises(ReplayReportError, match="invalid artifact at line"):
        summarize_replay(directory)


@pytest.mark.parametrize("name", ["replay_binding.json", "artifacts.jsonl"])
def test_symlinked_inputs_are_refused(tmp_path, name):
    directory = tmp_path / "replay"
    _attempt(directory)
    path = directory / name
    target = tmp_path / name
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ReplayReportError, match="symlink"):
        summarize_replay(directory)


def test_bound_empty_run_is_valid_but_legacy_without_binding_is_not(tmp_path):
    (tmp_path / "replay_binding.json").write_text(canonical_json(replay_run_binding("relay_mtu", test_clients=True)))
    report = summarize_replay(tmp_path)
    assert report["attempts"] == report["rows"] == 0 and report["final_accept_rate"] is None
    assert report["sources"]["artifacts_present"] is False
    (tmp_path / "replay_binding.json").unlink()
    with pytest.raises(OSError):
        summarize_replay(tmp_path)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(schema_version="legacy"),
    lambda p: p["models"]["K3"].update(effort="low"),
    lambda p: p["models"]["K3"].update(input_usd_per_mtok=0.0001),
    lambda p: p["redaction"].update(cutoff_ts_ms=123),
])
def test_changed_run_binding_is_refused(tmp_path, mutate):
    directory = tmp_path / "replay"
    _attempt(directory)
    path = directory / "replay_binding.json"
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(canonical_json(value))
    with pytest.raises(ReplayReportError, match="invalid replay binding"):
        summarize_replay(directory)


def test_report_does_not_read_the_redaction_map(tmp_path, monkeypatch):
    directory = tmp_path / "replay"
    _attempt(directory)
    original = Path.open
    def open_without_map(path, *args, **kwargs):
        if path.name == "redaction_map.jsonl" or path.suffix == ".parquet":
            raise AssertionError("must not read identifying data or features")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_without_map)
    assert summarize_replay(directory)["final_rows"] == 1


def _script():
    spec = importlib.util.spec_from_file_location("replay_report_script", ROOT / "scripts/summarize_agent_shadow_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_builds_no_models_and_outputs_only_aggregates(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "replay"
    _attempt(directory)
    def forbidden(*args, **kwargs):
        raise AssertionError("must not build a client or model")
    for name in ("build_openai_client", "build_anthropic_client", "build_shadow_models"):
        monkeypatch.setattr(provider_clients, name, forbidden)
    assert _script().main(["--input-dir", str(directory)]) == 0
    output = capsys.readouterr()
    assert output.err == "" and json.loads(output.out)["final_rows"] == 1


def test_cli_rejects_bad_input_without_partial_report_or_row_data(tmp_path, capsys):
    directory = tmp_path / "replay"
    writer = _attempt(directory)
    writer.artifacts.write_text('{"PRIVATE_ROW_SECRET":')
    assert _script().main(["--input-dir", str(directory)]) == 2
    output = capsys.readouterr()
    assert output.out == "" and "line 1" in output.err and "PRIVATE_ROW_SECRET" not in output.err
