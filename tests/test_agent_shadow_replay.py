import importlib.util
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agent_shadow_provider_fakes import FakeAnthropicClient, FakeOpenAIClient
from std0_quant.research.agent_shadow import provider_clients
from std0_quant.research.agent_shadow.context import FEATURE_ALLOWLIST, META_ALLOWLIST
from std0_quant.research.agent_shadow.model_registry import CostBudget
from std0_quant.research.agent_shadow.replay import (
    REDACTED_CUTOFF_MS, REDACTED_FEATURES, ReplayWriter, build_replay_plan, is_final, load_frozen_inputs,
    redacted_condition_id, run_replay, sha256_bytes,
)
from std0_quant.research.agent_shadow.runner import ShadowRunner


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = 1_756_000_000_000
LEAKY = ("y30", "first_opp_fill_price", "fill_minus_prev_mid", "prediction_ts_ms", "market_end_ms",
         "initial_direction_up", "btc_move_toward_opp_5s", "direction_sign", "feature_row_id", "opp_ratio")


def _feature(cid, cutoff=CUTOFF, **changes):
    row = {
        "feature_row_id": f"row-{cid}", "condition_id": cid, "prediction_ts_ms": cutoff + 1000,
        "feature_cutoff_ms": cutoff, "cutoff_mode": "cutoff_1", "market_start_ms": cutoff - 200_000,
        "market_end_ms": cutoff + 100_000, "y30": 1, "first_opp_fill_price": 0.47, "fill_minus_prev_mid": -0.05,
        "initial_direction_up": True, "btc_move_toward_opp_5s": 0.0001, "direction_sign": -1, "opp_ratio": 0.3,
        "btc_last_price": 112345.5, "btc_start_price": 112300.25, "btc_ret_5s": 0.00002,
        "btc_pre30_coverage_pct": 1.0, "book_pre10_coverage_pct": 1.0,
        "opp_mid": 0.52, "opp_spread": 0.02, "opp_best_bid": 0.51, "opp_best_ask": 0.53, "opp_obi_1": 0.1,
        "model_eligible": True, "model_ineligible_reason": None,
    }
    row.update(changes)
    return row


def _provenance(cid, cutoff=CUTOFF, *, btc_max=None, book_max=None, skip=(), missing=()):
    btc_max = cutoff - 200 if btc_max is None else btc_max
    book_max = cutoff - 300 if book_max is None else book_max
    rows = []
    for source, names, maximum in (
            ("binance_btc", ("btc_last_price", "btc_start_price", "btc_ret_5s", "btc_pre30_coverage_pct"), btc_max),
            ("polymarket_book", ("opp_mid", "opp_spread", "opp_obi_1", "book_pre10_coverage_pct"), book_max)):
        for name in names:
            if name not in skip:
                rows.append({"condition_id": cid, "feature_name": name, "source_type": source,
                             "source_timestamp_min_ms": cutoff - 30_000, "source_timestamp_max_ms": maximum,
                             "prediction_ts_ms": cutoff + 1000, "feature_cutoff_ms": cutoff,
                             "missing_reason": "NO_DATA" if name in missing else None})
    # Non-allowlisted public rows (fill-dependent) and t0-known state must not move observed timestamps.
    rows.append({"condition_id": cid, "feature_name": "fill_minus_prev_mid", "source_type": "polymarket_book",
                 "source_timestamp_min_ms": None, "source_timestamp_max_ms": cutoff - 1,
                 "prediction_ts_ms": cutoff + 1000, "feature_cutoff_ms": cutoff, "missing_reason": None})
    rows.append({"condition_id": cid, "feature_name": "opp_ratio", "source_type": "phase1_truth",
                 "source_timestamp_min_ms": None, "source_timestamp_max_ms": cutoff + 1000,
                 "prediction_ts_ms": cutoff + 1000, "feature_cutoff_ms": cutoff, "missing_reason": None})
    return rows


def _write_inputs(directory, features, provenance, rows=True):
    directory.mkdir(parents=True, exist_ok=True)
    pins = {}
    for name, records in (("features", features), ("provenance", provenance)):
        path = directory / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(records), path)
        pins[name] = (path.name, sha256_bytes(path.read_bytes()), len(records) if rows else None)
    return pins


def _plan(features, provenance, **kwargs):
    return build_replay_plan(features, provenance, **kwargs)


def _models(budget, anthropic_fake=None, openai_fake=None):
    return provider_clients.build_shadow_models(anthropic_fake or FakeAnthropicClient(), openai_fake or FakeOpenAIClient(),
                                                budget=budget, network_enabled=True, allow_test_clients=True)


def _writer(tmp_path, *extra):
    return ReplayWriter(tmp_path / "out", forbidden_roots=(ROOT, *extra))


# Inputs (F6)

def test_pinned_inputs_must_match_sha_and_rows(tmp_path):
    pins = _write_inputs(tmp_path / "in", [_feature("c1")], _provenance("c1"))
    loaded = load_frozen_inputs(tmp_path / "in", pins)
    assert [record["condition_id"] for record in loaded["features"]["records"]] == ["c1"]
    for bad in ({"features": ("features.parquet", "0" * 64, 1)},
                {"features": ("features.parquet", pins["features"][1], 2)},
                {"features": ("missing.parquet", pins["features"][1], 1)}):
        with pytest.raises(ValueError):
            load_frozen_inputs(tmp_path / "in", {**pins, **bad})


# Projection and redaction (F10)

def test_rows_are_projected_to_the_allowlist_and_redacted():
    (row,) = _plan([_feature("0xabc")], _provenance("0xabc")).rows
    assert set(row.pit_row) <= FEATURE_ALLOWLIST | META_ALLOWLIST
    assert not any(name in row.pit_row for name in LEAKY + tuple(REDACTED_FEATURES))
    assert row.pit_row["condition_id"] == redacted_condition_id("0xabc") != "0xabc"
    assert row.pit_row["decision_cutoff_ts_ms"] == REDACTED_CUTOFF_MS
    assert row.pit_row["btc_observed_ts_ms"] == REDACTED_CUTOFF_MS - 200   # relative timing kept
    assert row.pit_row["book_observed_ts_ms"] == REDACTED_CUTOFF_MS - 300  # fill rows did not move it
    assert (row.condition_id, row.real_cutoff_ms) == ("0xabc", CUTOFF)


def test_models_never_see_real_ids_times_prices_or_labels(tmp_path):
    rows = _plan([_feature("0xabc")], _provenance("0xabc")).rows
    anthropic_fake, openai_fake = FakeAnthropicClient(), FakeOpenAIClient()
    budget = CostBudget(5.0)
    writer = _writer(tmp_path)
    run_replay(rows, ShadowRunner(*_models(budget, anthropic_fake, openai_fake)), writer, budget, max_rows=1)
    sent = json.dumps([call["messages"] for call in anthropic_fake.calls] + [call["input"] for call in openai_fake.calls])
    for secret in ("0xabc", str(CUTOFF), str(CUTOFF - 200), "112345.5", "112300.25", *LEAKY):
        assert secret not in sent
    (artifact,) = [json.loads(line) for line in writer.artifacts.read_text().splitlines()]
    (mapping,) = [json.loads(line) for line in writer.redaction_map.read_text().splitlines()]
    assert artifact["condition_id"] == mapping["redacted_id"] == redacted_condition_id("0xabc")
    assert (mapping["condition_id"], mapping["real_cutoff_ms"]) == ("0xabc", CUTOFF)
    assert mapping["artifact_sha256"] == artifact["artifact_sha256"]


def test_non_finite_features_are_missing_and_order_is_by_real_cutoff():
    features = [_feature("b", cutoff=CUTOFF + 5), _feature("a", cutoff=CUTOFF + 5, btc_ret_5s=math.nan),
                _feature("z", cutoff=CUTOFF)]
    provenance = _provenance("b", CUTOFF + 5) + _provenance("a", CUTOFF + 5) + _provenance("z")
    rows = _plan(features, provenance).rows
    assert [row.condition_id for row in rows] == ["z", "a", "b"]
    assert rows[1].pit_row["btc_ret_5s"] is None


def test_unreadable_rows_are_counted_not_silently_dropped():
    features = [_feature("ok"), _feature("ineligible", model_eligible=False), _feature("mode", cutoff_mode="cutoff_5"),
                _feature("float_cutoff", feature_cutoff_ms=float(CUTOFF)), _feature(None),
                _feature("none_cutoff", feature_cutoff_ms=None)]
    plan = _plan(features, _provenance("ok"))
    assert [row.condition_id for row in plan.rows] == ["ok"]
    assert dict(plan.dropped) == {"NOT_MODEL_ELIGIBLE": 1, "NOT_CUTOFF_1": 1, "BAD_CUTOFF": 2, "BAD_CONDITION_ID": 1}


# Evidence: fail closed (F1, F9)

def test_complete_row_has_all_evidence():
    (row,) = _plan([_feature("c1")], _provenance("c1")).rows
    assert (row.evidence.coverage_valid, row.evidence.provenance_valid, row.evidence.sanity_valid) == (True,) * 3


@pytest.mark.parametrize("feature_changes, provenance_kwargs, failure", [
    ({"btc_pre30_coverage_pct": 0.98}, {}, "COVERAGE"),
    ({"model_eligible": None}, {}, "COVERAGE"),
    ({}, {"btc_max": CUTOFF + 1}, "PROVENANCE"),
    ({}, {"skip": ("opp_mid",)}, "PROVENANCE"),
    ({}, {"missing": ("opp_mid",)}, "PROVENANCE"),
    ({"opp_mid": 1.5}, {}, "SANITY"),
    ({"opp_best_bid": 0.6, "opp_best_ask": 0.5}, {}, "SANITY"),
    ({"opp_best_bid": math.nan}, {}, "SANITY"),
    ({"opp_obi_1": math.inf}, {}, "SANITY"),
])
def test_failed_evidence_is_false(feature_changes, provenance_kwargs, failure):
    (row,) = _plan([_feature("c1", **feature_changes)], _provenance("c1", **provenance_kwargs),
                   eligible_only=False).rows
    assert failure in row.evidence_failures


def test_missing_provenance_blocks_the_row_without_calls(tmp_path):
    (row,) = _plan([_feature("c1")], []).rows
    assert row.pit_row["btc_observed_ts_ms"] is None and "PROVENANCE" in row.evidence_failures
    anthropic_fake = FakeAnthropicClient()
    budget = CostBudget(5.0)
    summary = run_replay([row], ShadowRunner(*_models(budget, anthropic_fake)), _writer(tmp_path), budget, max_rows=5)
    assert summary["gate_status_counts"] == {"BLOCKED": 1} and not anthropic_fake.calls


# Replay loop, retry semantics and budget (F2)

def _three_rows():
    return _plan([_feature(f"c{i}", cutoff=CUTOFF + i) for i in range(3)],
                 [p for i in range(3) for p in _provenance(f"c{i}", CUTOFF + i)]).rows


def test_replay_resumes_and_respects_max_rows(tmp_path):
    rows, writer = _three_rows(), _writer(tmp_path)
    budget = CostBudget(5.0)
    first = run_replay(rows, ShadowRunner(*_models(budget)), writer, budget, max_rows=2)
    assert (first["attempted"], first["written"], first["stop_reason"]) == (2, 2, "MAX_ROWS")
    second = run_replay(rows, ShadowRunner(*_models(budget)), writer, budget, max_rows=2)
    assert (second["attempted"], second["skipped_already_done"]) == (1, 2)
    assert len(writer.artifacts.read_text().splitlines()) == 3


def test_budget_cut_rows_are_audited_and_retried(tmp_path):
    rows, writer = _three_rows(), _writer(tmp_path)
    small = CostBudget(1.0)  # one row's later reservations exceed it
    first = run_replay(rows, ShadowRunner(*_models(small)), writer, small, max_rows=3, row_headroom_usd=0.0)
    assert (first["stop_reason"], first["written"], first["written_retryable"]) == ("BUDGET", 1, 1)
    assert writer.done_ids() == set()
    big = CostBudget(10.0)
    second = run_replay(rows, ShadowRunner(*_models(big)), writer, big, max_rows=3)
    assert second["written"] == 3 and second["skipped_already_done"] == 0
    assert writer.done_ids() == {row.redacted_id for row in rows}


def test_headroom_check_stops_before_a_row_starts(tmp_path):
    budget = CostBudget(0.5)
    anthropic_fake = FakeAnthropicClient()
    summary = run_replay(_three_rows(), ShadowRunner(*_models(budget, anthropic_fake)), _writer(tmp_path), budget,
                         max_rows=3, row_headroom_usd=1.0)
    assert (summary["stop_reason"], summary["attempted"]) == ("BUDGET", 0) and not anthropic_fake.calls


def test_transient_failures_are_written_for_audit_but_retried(tmp_path):
    rows, writer = _three_rows()[:1], _writer(tmp_path)
    budget = CostBudget(5.0)
    failing = FakeAnthropicClient(error=TimeoutError("slow"))
    first = run_replay(rows, ShadowRunner(*_models(budget, failing)), writer, budget, max_rows=1)
    assert (first["written"], first["written_retryable"]) == (1, 1)
    assert writer.done_ids() == set()
    second = run_replay(rows, ShadowRunner(*_models(budget)), writer, budget, max_rows=1)
    assert second["written"] == 1 and writer.done_ids() == {rows[0].redacted_id}
    assert len(writer.artifacts.read_text().splitlines()) == 2


def test_broken_setup_stops_the_run(tmp_path):
    rows = _three_rows()
    budget = CostBudget(5.0)
    k3, jev = _models(budget)
    k3.transport_version = "forged"
    summary = run_replay(rows, ShadowRunner(k3, jev), _writer(tmp_path), budget, max_rows=3)
    assert (summary["stop_reason"], summary["written"]) == ("SETUP_INVALID", 1)


def test_is_final_rules():
    assert is_final({"gate_reasons": [], "model_calls": [{"outcome": "OK"}]})
    assert not is_final({"gate_reasons": ["K3_TIMEOUT"], "model_calls": []})
    assert not is_final({"gate_reasons": [], "model_calls": [{"outcome": "API_ERROR"}]})
    assert not is_final({"gate_reasons": ["MODEL_PROVENANCE_INVALID"], "model_calls": []})


# Writer (F3)

def test_writer_refuses_forbidden_roots_and_never_overwrites(tmp_path):
    input_dir = tmp_path / "frozen"
    input_dir.mkdir()
    for out in (ROOT / "data" / "replay", input_dir, input_dir / "sub", tmp_path):
        with pytest.raises(ValueError):
            ReplayWriter(out, forbidden_roots=(ROOT, input_dir))
    writer = _writer(tmp_path, input_dir)
    first, second = writer.write_once("run_manifest", {"a": 1}), writer.write_once("run_manifest", {"a": 2})
    assert first != second and json.loads(first.read_text()) == {"a": 1}
    writer.artifacts.write_text("not json\n")
    with pytest.raises(ValueError):
        writer.done_ids()


def test_writer_refuses_symlinked_artifacts(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "artifacts.jsonl").symlink_to(tmp_path / "elsewhere.jsonl")
    with pytest.raises(ValueError):
        ReplayWriter(out, forbidden_roots=(ROOT,))


def test_run_replay_validates_arguments(tmp_path):
    budget = CostBudget(1.0)
    for kwargs in ({"max_rows": 0}, {"max_rows": 1, "row_headroom_usd": -1.0}):
        with pytest.raises(ValueError):
            run_replay([], ShadowRunner(*_models(budget)), _writer(tmp_path), budget, **kwargs)


# Script (F3, F4, F5, F8)

def _script(monkeypatch, pins=None):
    spec = importlib.util.spec_from_file_location("run_agent_shadow_replay", ROOT / "scripts/run_agent_shadow_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if pins is not None:
        monkeypatch.setattr(module, "load_frozen_inputs", lambda directory: load_frozen_inputs(directory, pins))
    for name in ("build_anthropic_client", "build_openai_client", "build_shadow_models"):
        monkeypatch.setattr(module.provider_clients, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} called"))
    return module


def test_script_plan_only_makes_no_calls(tmp_path, monkeypatch, capsys):
    pins = _write_inputs(tmp_path / "in", [_feature("c1"), _feature("c2", opp_mid=1.5)],
                         _provenance("c1") + _provenance("c2"))
    assert _script(monkeypatch, pins).main(["--input-dir", str(tmp_path / "in"), "--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["rows"] == 2 and plan["evidence_all_true"] == 1 and plan["evidence_failures"] == {"SANITY": 1}


@pytest.mark.parametrize("extra, message", [
    ([], "required"),
    (["--max-rows", "0", "--budget-usd", "1"], "--max-rows"),
    (["--max-rows", "1", "--budget-usd", "nan"], "--budget-usd"),
    (["--max-rows", "1", "--budget-usd", "1", "--route", "relay_mtu"], "--allow-unverified-relay"),
])
def test_script_checks_run_arguments_before_reading_inputs(tmp_path, monkeypatch, extra, message):
    script = _script(monkeypatch)
    monkeypatch.setattr(script, "load_frozen_inputs", lambda directory: pytest.fail("inputs read before checks"))
    with pytest.raises(SystemExit, match=message):
        script.main(["--input-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), *extra])


def test_script_refuses_output_inside_the_input_dir(tmp_path, monkeypatch):
    pins = _write_inputs(tmp_path / "in", [_feature("c1")], _provenance("c1"))
    with pytest.raises(ValueError):
        _script(monkeypatch, pins).main(["--input-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "in" / "o"),
                                         "--max-rows", "1", "--budget-usd", "1"])


def test_script_refuses_changed_frozen_inputs(tmp_path, monkeypatch):
    pins = _write_inputs(tmp_path / "in", [_feature("c1")], _provenance("c1"))
    script = _script(monkeypatch)
    from std0_quant.research.agent_shadow import replay

    tampered = {**pins, "features": (pins["features"][0], "f" * 64, 1)}
    monkeypatch.setattr(script, "load_frozen_inputs", lambda directory: replay.load_frozen_inputs(directory, tampered))
    with pytest.raises(ValueError, match="sha256"):
        script.main(["--input-dir", str(tmp_path / "in"), "--plan-only"])


def test_proxy_credentials_are_never_recorded(monkeypatch):
    script = _script(monkeypatch)
    assert script._proxy_label("http://user:secret@127.0.0.1:7897") == "http://127.0.0.1:7897"
    assert script._proxy_label(None) is None


# Re-check fixes: duplicates, attempt cap, configured paths, proxy schemes

def test_duplicate_condition_ids_are_dropped():
    plan = _plan([_feature("dup"), _feature("dup", cutoff=CUTOFF + 1), _feature("ok")],
                 _provenance("dup") + _provenance("ok"))
    assert [row.condition_id for row in plan.rows] == ["ok"]
    assert plan.dropped["DUPLICATE_CONDITION_ID"] == 2


def test_rows_that_keep_failing_stop_being_retried(tmp_path):
    rows, writer = _three_rows()[:2], _writer(tmp_path)
    budget = CostBudget(20.0)
    for _ in range(2):
        run_replay(rows[:1], ShadowRunner(*_models(budget, FakeAnthropicClient(error=TimeoutError("slow")))),
                   writer, budget, max_rows=1)
    summary = run_replay(rows, ShadowRunner(*_models(budget)), writer, budget, max_rows=1, max_attempts=2)
    assert summary["skipped_attempts_exhausted"] == 1 and summary["written"] == 1
    assert writer.done_ids() == {rows[1].redacted_id}
    with pytest.raises(ValueError):
        run_replay(rows, ShadowRunner(*_models(budget)), writer, budget, max_rows=1, max_attempts=0)


def test_script_forbids_every_configured_data_path(monkeypatch):
    script = _script(monkeypatch)
    roots = {str(path) for path in script._forbidden_roots(Path("/tmp/frozen-input"))}
    assert str(ROOT) in roots and "/tmp/frozen-input" in roots
    assert any(path.endswith("data/derived") for path in roots) and any(path.endswith("data/state") for path in roots)


def test_proxy_urls_need_a_known_scheme(monkeypatch):
    script = _script(monkeypatch)
    for bad in ("user:pass@127.0.0.1:7897", "ftp://127.0.0.1:21", "http://"):
        with pytest.raises(SystemExit):
            script._proxy_label(bad)
