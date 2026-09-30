import pytest

from agent_shadow_fakes import pit_row
from std0_quant.research.agent_shadow.context import build_shadow_context


def test_explicit_allowlist_and_deterministic_hash():
    first = build_shadow_context(pit_row())
    second = build_shadow_context(pit_row())
    assert first.context_sha256 == second.context_sha256
    assert len(first.input_feature_sha256) == 64
    assert first.to_dict()["btc_observed_ts_ms"] < first.decision_cutoff_ts_ms
    assert "y30" not in first.to_dict()["features"]


@pytest.mark.parametrize("name,value", [
    ("y30", 1), ("y30_event_ts_ms", 99), ("settlement", "UP"),
    ("final_outcome", "UP"), ("realized_pnl", 1),
    ("future_fills", []), ("backtest_result", "PASS"),
    ("post_cutoff_book_data", []), ("post_cutoff_btc_data", []),
    ("formal_cohort_future_eligibility_label", True),
    ("private_key", "secret"),
])
def test_leakage_or_unknown_fields_fail_closed(name, value):
    with pytest.raises(ValueError):
        build_shadow_context(pit_row(**{name: value}))


@pytest.mark.parametrize("name", ["btc_observed_ts_ms", "book_observed_ts_ms"])
def test_post_cutoff_source_rejected(name):
    with pytest.raises(ValueError):
        build_shadow_context(pit_row(**{name: 1_700_000_000_001}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_feature_rejected(value):
    with pytest.raises(ValueError):
        build_shadow_context(pit_row(btc_ret_5s=value))
