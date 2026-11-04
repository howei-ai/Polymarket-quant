import dataclasses
from decimal import Decimal
import json
import os
from pathlib import Path

import pytest

from agent_shadow_provider_fakes import FakeOpenAIK3Client, FakeOpenAIClient
from std0_quant.research.agent_shadow import provider_clients
from std0_quant import config as config_module
from std0_quant.research.agent_shadow.audit import canonical_json, sha256_json
from std0_quant.research.agent_shadow.model_registry import CostBudget, binding_for
from std0_quant.research.agent_shadow.replay import ReplayWriter, build_replay_plan, replay_run_binding, run_replay
from std0_quant.research.agent_shadow.replay_planning import plan_replay_run
from std0_quant.research.agent_shadow.replay_report import ReplayReportError
from std0_quant.research.agent_shadow.runner import ShadowRunner
from test_agent_shadow_replay import _feature, _provenance, _script, _write_inputs
from test_agent_shadow_replay_report import _attempt, _edit


ROOT = Path(__file__).resolve().parents[1]


def _rows(*ids, **changes):
    features = [_feature(cid, **changes) for cid in ids]
    provenance = [item for cid in ids for item in _provenance(cid)]
    return build_replay_plan(features, provenance, eligible_only=False).rows


def _plan(rows, directory=None, **kwargs):
    return plan_replay_run(rows, route="relay_mtu", out_dir=directory, forbidden_roots=(ROOT,),
                           test_clients=True, **kwargs)


def test_fresh_plan_is_read_only_and_uses_configured_reservations(tmp_path):
    directory = tmp_path / "not-created" / "nested"
    plan = _plan(_rows("a", "b", "c"), directory, max_rows=2, budget_usd=83.52)
    assert not directory.parent.exists()
    assert (plan["pending_rows"], plan["new_rows"], plan["selected_rows"]) == (3, 3, 2)
    assert plan["selected_model_call_candidates"] == 2 and plan["selected_retry_rows"] == 0
    assert not plan["existing_binding"] and plan["sources"]["binding_sha256"] is None
    estimate = plan["budget_estimate"]
    assert Decimal(estimate["configured_reservation_usd_per_model_row"]) == Decimal("41.76")
    assert Decimal(estimate["selected_configured_reservation_usd"]) == Decimal("83.52")
    assert estimate["model_rows_covered_by_estimate"] == 2 and estimate["budget_meets_start_headroom"] is True
    assert plan["plan_sha256"] == sha256_json({k: v for k, v in plan.items() if k != "plan_sha256"})
    assert plan["planning_only"] and not plan["execution_allowed"] and not plan["formal_cohort_effect"]


def test_resume_distinguishes_final_retry_exhausted_and_outside_rows(tmp_path):
    directory = tmp_path / "out"
    _attempt(directory, "a-retry", fake=FakeOpenAIK3Client(error=TimeoutError()))
    for _ in range(3):
        _attempt(directory, "b-exhausted", fake=FakeOpenAIK3Client(error=TimeoutError()))
    _attempt(directory, "c-final")
    _attempt(directory, "x-outside")
    rows = _rows("a-retry", "b-exhausted", "c-final", "d-new")
    plan = _plan(rows, directory, max_rows=1, budget_usd=50)
    assert (plan["already_final_rows"], plan["attempts_exhausted_rows"], plan["pending_rows"]) == (1, 1, 2)
    assert (plan["new_rows"], plan["retry_ready_rows"], plan["history_rows_outside_input_plan"]) == (1, 1, 1)
    assert (plan["selected_rows"], plan["selected_new_rows"], plan["selected_retry_rows"]) == (1, 0, 1)
    raised = _plan(rows, directory, max_attempts=4)
    assert raised["attempts_exhausted_rows"] == 0 and raised["retry_ready_rows"] == 2
    budget = CostBudget(50)
    models = provider_clients.build_shadow_models(FakeOpenAIK3Client(), FakeOpenAIClient(), budget=budget,
                                                network_enabled=True, allow_test_clients=True, route="relay_mtu")
    actual = run_replay(rows, ShadowRunner(*models, allow_unverified_relay=True),
                        ReplayWriter(directory, forbidden_roots=(ROOT,)), budget, max_rows=1)
    assert actual["attempted"] == plan["selected_rows"]
    assert json.loads((directory / "artifacts.jsonl").read_text().splitlines()[-1])["condition_id"] == rows[0].redacted_id


def test_start_headroom_and_configured_coverage_are_separate(tmp_path):
    below = _plan(_rows("a"), budget_usd=1)
    enough_to_start = _plan(_rows("a"), budget_usd=5)
    assert below["selected_rows"] == enough_to_start["selected_rows"] == 1
    assert below["budget_estimate"]["budget_meets_start_headroom"] is False
    assert enough_to_start["budget_estimate"]["budget_meets_start_headroom"] is True
    assert enough_to_start["budget_estimate"]["model_rows_covered_by_estimate"] == 0


def test_preflight_blocks_have_no_model_reservation_estimate():
    plan = _plan(_rows("a", opp_mid=1.5), budget_usd=5)
    assert plan["selected_rows"] == plan["selected_preflight_blocked_rows"] == 1
    assert plan["selected_model_call_candidates"] == 0
    assert Decimal(plan["budget_estimate"]["selected_configured_reservation_usd"]) == 0


def test_unknown_historical_cost_does_not_consume_a_new_run_budget(tmp_path):
    directory = tmp_path / "out"
    _attempt(directory, "a", fake=FakeOpenAIK3Client(usage=(10 ** 315, 300), cache_tokens=None))
    plan = _plan(_rows("a"), directory, budget_usd=50)
    assert plan["retry_ready_rows"] == plan["selected_retry_rows"] == 1
    assert plan["budget_estimate"]["budget_scope"] == "NEW_RUN"
    assert plan["budget_estimate"]["model_rows_covered_by_estimate"] == 1


@pytest.mark.parametrize("budget,covered", [(83.51, 1), (83.52, 2), (1e308, 3)])
def test_budget_estimate_boundaries_and_large_values(budget, covered):
    assert _plan(_rows("a", "b", "c"), budget_usd=budget)["budget_estimate"]["model_rows_covered_by_estimate"] == covered


def test_tightened_model_limits_use_the_runtime_reservation_policy():
    models = {
        "K3": dataclasses.replace(binding_for("K3"), max_input_bytes=1000, max_output_tokens=100),
        "JEV": dataclasses.replace(binding_for("JEV"), max_input_bytes=2000, max_output_tokens=50),
    }
    plan = _plan(_rows("a"), bindings=models, budget_usd=1)
    expected = (7 * Decimal(str(models["K3"].max_call_cost_usd(1000)))
                + Decimal(str(models["JEV"].max_call_cost_usd(2000))))
    assert Decimal(plan["budget_estimate"]["configured_reservation_usd_per_model_row"]) == expected


def test_no_budget_and_no_remaining_rows_are_explicit(tmp_path):
    directory = tmp_path / "out"
    _attempt(directory, "a")
    plan = _plan(_rows("a"), directory)
    assert plan["selected_rows"] == plan["pending_rows"] == 0
    assert plan["budget_estimate"]["model_rows_covered_by_estimate"] is None
    assert plan["limits"]["budget_usd"] is None


@pytest.mark.parametrize("kwargs", [
    {"max_rows": 0}, {"max_rows": True}, {"max_attempts": 0}, {"max_attempts": False},
    {"budget_usd": 0}, {"budget_usd": float("nan")}, {"budget_usd": 10 ** 400},
    {"row_headroom_usd": -1}, {"row_headroom_usd": float("inf")},
])
def test_invalid_planning_limits_are_rejected_without_creating_outputs(tmp_path, kwargs):
    directory = tmp_path / "out"
    with pytest.raises(ValueError):
        _plan(_rows("a"), directory, **kwargs)
    assert not directory.exists()


def test_duplicate_input_rows_are_not_given_an_ambiguous_plan():
    rows = _rows("a")
    with pytest.raises(ValueError, match="duplicate replay row"):
        _plan(rows + rows)


def test_existing_history_is_validated_and_not_read_again(tmp_path, monkeypatch):
    directory = tmp_path / "out"
    _attempt(directory, "PRIVATE-ROW")
    rows = _rows("PRIVATE-ROW")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    original = Path.open
    opened = []
    def checked_open(path, *args, **kwargs):
        assert path.name in ("replay_binding.json", "artifacts.jsonl")
        opened.append(path.name)
        return original(path, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("model or client constructed during planning")
    with monkeypatch.context() as local:
        local.setattr(Path, "open", checked_open)
        for name in ("build_anthropic_client", "build_openai_client", "build_shadow_models"):
            local.setattr(provider_clients, name, forbidden)
        plan = _plan(rows, directory)
    assert opened == ["replay_binding.json", "artifacts.jsonl"]
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    for secret in ("PRIVATE-ROW", rows[0].redacted_id, "req_k3_1", "1700000000000", "1756000000000"):
        assert secret not in json.dumps(plan)


@pytest.mark.parametrize("damage", ["hash", "missing_binding", "different_binding"])
def test_damaged_or_incompatible_history_is_refused(tmp_path, damage):
    directory = tmp_path / "out"
    _attempt(directory, "a")
    if damage == "hash":
        _edit(directory, lambda p: p.update(gate_status="BLOCKED"), resign=False)
    elif damage == "missing_binding":
        (directory / "replay_binding.json").unlink()
    else:
        _edit(directory, lambda p: None)
    with pytest.raises(ReplayReportError):
        if damage == "different_binding":
            plan_replay_run(_rows("a"), route="relay_mtu", out_dir=directory)  # real CLI cannot resume test clients
        else:
            _plan(_rows("a"), directory)


def test_cli_plan_only_uses_limits_without_creating_a_directory(tmp_path, monkeypatch, capsys):
    pins = _write_inputs(tmp_path / "in", [_feature("a"), _feature("b")], _provenance("a") + _provenance("b"))
    out = tmp_path / "absent" / "out"
    script = _script(monkeypatch, pins)
    assert script.main(["--input-dir", str(tmp_path / "in"), "--plan-only", "--out-dir", str(out),
                        "--route", "relay_mtu", "--max-rows", "1", "--budget-usd", "5", "--max-attempts", "4"]) == 0
    plan = json.loads(capsys.readouterr().out)["run_plan"]
    assert plan["selected_rows"] == 1 and plan["limits"]["max_attempts"] == 4
    assert plan["provenance"] == "UNVERIFIED_RELAY" and plan["relay_opt_in_required_for_run"]
    assert not out.parent.exists()


def test_cli_resume_plan_accepts_declared_real_history_without_calls(tmp_path, monkeypatch, capsys):
    pins = _write_inputs(tmp_path / "in", [_feature("a"), _feature("b")], _provenance("a") + _provenance("b"))
    directory = tmp_path / "out"
    _attempt(directory, "a", fake=FakeOpenAIK3Client(error=TimeoutError()))
    # Only declared provenance is checked: this synthetic fixture makes no real calls.
    binding = replay_run_binding("relay_mtu")
    (directory / "replay_binding.json").write_text(canonical_json(binding) + "\n")
    def relabel(payload):
        payload["provenance"] = "UNVERIFIED_RELAY"
        for role, prefix in (("K3", "k3"), ("JEV", "jev")):
            payload[prefix + "_transport_version"] = binding["models"][role]["transport_version"]
        for call in payload["model_calls"]:
            call["test_client"] = False
    _edit(directory, relabel)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert _script(monkeypatch, pins).main(["--input-dir", str(tmp_path / "in"), "--plan-only",
                                           "--out-dir", str(directory), "--route", "relay_mtu"]) == 0
    plan = json.loads(capsys.readouterr().out)["run_plan"]
    assert plan["retry_ready_rows"] == plan["new_rows"] == 1
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("extra,message", [
    (["--max-rows", "0"], "--max-rows"), (["--budget-usd", "nan"], "--budget-usd"),
    (["--max-attempts", "0"], "--max-attempts"), (["--row-headroom-usd", "-1"], "--row-headroom-usd"),
])
def test_cli_bad_limits_are_checked_before_loading_inputs(tmp_path, monkeypatch, extra, message):
    script = _script(monkeypatch)
    monkeypatch.setattr(script, "load_frozen_inputs", lambda *a: pytest.fail("inputs read before argument checks"))
    with pytest.raises(SystemExit, match=message):
        script.main(["--input-dir", str(tmp_path), "--plan-only", *extra])


def test_cli_failed_plan_has_no_partial_output_or_directory(tmp_path, monkeypatch, capsys):
    pins = _write_inputs(tmp_path / "in", [_feature("a")], _provenance("a"))
    out = tmp_path / "in" / "forbidden"
    assert _script(monkeypatch, pins).main(["--input-dir", str(tmp_path / "in"), "--plan-only", "--out-dir", str(out)]) == 2
    output = capsys.readouterr()
    assert output.out == "" and "offline run plan unavailable" in output.err
    assert not out.exists()


@pytest.mark.parametrize("contents,valid", [("", True), (" \n", False)])
def test_unbound_empty_artifacts_match_the_workers_fresh_directory_rule(tmp_path, contents, valid):
    out = tmp_path / "out"
    out.mkdir()
    path = out / "artifacts.jsonl"
    path.write_text(contents)
    if valid:
        plan = _plan(_rows("a"), out)
        assert not plan["existing_binding"] and plan["new_rows"] == 1
    else:
        with pytest.raises(ReplayReportError, match="no run binding"):
            _plan(_rows("a"), out)
    assert path.read_text() == contents and not (out / "replay_binding.json").exists()


def test_run_cli_passes_limits_to_the_worker_and_records_them(tmp_path, monkeypatch, capsys):
    pins = _write_inputs(tmp_path / "in", [_feature("a")], _provenance("a"))
    original_models = provider_clients.build_shadow_models
    script = _script(monkeypatch, pins)
    clients = [FakeOpenAIK3Client(), FakeOpenAIClient()]
    monkeypatch.setattr(script.provider_clients, "build_openai_client", lambda **kwargs: clients.pop(0))
    monkeypatch.setattr(script.provider_clients, "build_shadow_models",
                        lambda *args, **kwargs: original_models(*args, **kwargs, allow_test_clients=True))
    passed = []
    def run(*args, **kwargs):
        passed.append(kwargs)
        return run_replay(*args, **kwargs)
    monkeypatch.setattr(script, "run_replay", run)
    out = tmp_path / "out"
    assert script.main(["--input-dir", str(tmp_path / "in"), "--out-dir", str(out),
                        "--max-rows", "1", "--budget-usd", "0.5", "--max-attempts", "2",
                        "--row-headroom-usd", "0"]) == 0
    assert passed == [{"max_rows": 1, "max_attempts": 2, "row_headroom_usd": 0.0}]
    summary = json.loads(capsys.readouterr().out)
    assert summary["attempted"] == 1 and summary["stop_reason"] == "BUDGET"
    manifest = json.loads(next(out.glob("run_manifest_*.json")).read_text())
    assert (manifest["max_attempts"], manifest["row_headroom_usd"]) == (2, 0.0)


def test_plan_does_not_load_dotenv_or_mutate_the_process_environment(tmp_path, monkeypatch, capsys):
    pins = _write_inputs(tmp_path / "in", [_feature("a")], _provenance("a"))
    script = _script(monkeypatch, pins)
    calls = []
    def mock_dotenv(*args, **kwargs):
        calls.append(True)
        monkeypatch.setenv("STD0_PLAN_DOTENV_PROBE", "injected-by-mock")
    monkeypatch.setattr(config_module, "load_dotenv", mock_dotenv)
    before = dict(os.environ)
    out = tmp_path / "missing" / "out"
    assert script.main(["--input-dir", str(tmp_path / "in"), "--plan-only", "--out-dir", str(out)]) == 0
    assert calls == [] and dict(os.environ) == before
    assert not out.parent.exists()
    assert json.loads(capsys.readouterr().out)["run_plan"]["planning_only"]


@pytest.mark.parametrize("target", ["file", "file-parent"])
def test_planning_and_writer_refuse_files_as_directories(tmp_path, target):
    marker = tmp_path / "marker"
    marker.write_text("synthetic-marker")
    out = marker if target == "file" else marker / "child" / "nested"
    for operation in (lambda: _plan(_rows("a"), out), lambda: ReplayWriter(out, forbidden_roots=(ROOT,))):
        with pytest.raises(ValueError, match="director"):
            operation()
    assert marker.read_text() == "synthetic-marker"


@pytest.mark.parametrize("with_dotenv", [False, True])
def test_settings_can_skip_dotenv_while_preserving_process_overrides(monkeypatch, with_dotenv):
    monkeypatch.setenv("STD0_NAME", "from-process")
    calls = []
    def mock_dotenv(*args, **kwargs):
        calls.append(True)
        monkeypatch.setenv("STD0_NAME", "from-dotenv")
    monkeypatch.setattr(config_module, "load_dotenv", mock_dotenv)
    kwargs = {} if with_dotenv else {"load_env_file": False}
    settings = config_module.load_settings(project_root=ROOT, **kwargs)
    assert settings.trader.name == ("from-dotenv" if with_dotenv else "from-process")
    assert calls == ([True] if with_dotenv else [])
