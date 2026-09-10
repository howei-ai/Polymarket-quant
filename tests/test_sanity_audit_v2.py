"""Opt-in numeric semantics, legacy compatibility and malformed-value guards."""
from __future__ import annotations

import __future__
import ast
from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext
import json
import math
from pathlib import Path
import random

import pytest

from std0_quant.audit.sanity_audit_v2 import (
    AUDIT_SCOPE, AUDIT_VERSION, OBI_CHANGE_FIELDS, sanity_audit_v2,
)


def legacy_checker():
    path = Path(__file__).resolve().parents[1] / "src/std0_quant/audit/prospective.py"
    selected = [n for n in ast.parse(path.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == "sanity_audit"]
    assert len(selected) == 1 and not selected[0].decorator_list
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    return namespace["sanity_audit"]


def checked(**fields):
    return sanity_audit_v2([{"condition_id": "c", **fields}])


@pytest.mark.parametrize("field", sorted(OBI_CHANGE_FIELDS))
@pytest.mark.parametrize("value", [-2, -1.6, -1, 0, 1, 1.6, 2])
def test_exact_delta_domain_inclusive(field, value):
    out = checked(**{field: value})
    assert out["status"] == "PASS" and out["checked_value_count"] == 1


@pytest.mark.parametrize("field", sorted(OBI_CHANGE_FIELDS))
@pytest.mark.parametrize("value", [math.nextafter(2, math.inf), math.nextafter(-2, -math.inf),
                                   "2.00000000000000000001", Decimal("-2.00000000000000000001")])
def test_delta_strictly_outside_is_not_rounded_or_clipped(field, value):
    out = checked(**{field: value})
    assert out["violation_count"] == 1 and out["violations"][0]["reason"] == "OUT_OF_RANGE"


@pytest.mark.parametrize("field", ["opp_obi_1", "opp_obi_3", "initial_obi_1", "initial_obi_3",
                                   "pm_obi_change_10s", "prefix_pm_obi_change_1s",
                                   "pm_obi_change_5s_suffix", "pm_obi_change_1S"])
@pytest.mark.parametrize("value", [-1.1, 1.1])
def test_level_or_unknown_obi_names_are_not_widened(field, value):
    assert checked(**{field: value})["status"] == "DATA_SANITY_WARNING"


@pytest.mark.parametrize("value", [-1, 1, -0.0])
def test_level_boundaries_preserved(value):
    assert checked(opp_obi_1=value, initial_obi_3=value)["status"] == "PASS"


@pytest.mark.parametrize("field", ["pm_obi_change_1s", "opp_obi_1", "opp_bid_depth_1", "btc_last_price", "opp_mid"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "NaN", Decimal("sNaN")])
def test_all_governed_domains_reject_nonfinite(field, value):
    out = checked(**{field: value})
    assert out["status"] == "DATA_SANITY_WARNING"
    assert out["violations"][0]["reason"] == "NONFINITE_VALUE"
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("value", [True, False, "not-a-number", [], {}, object()])
def test_invalid_numeric_values_diagnosed_not_silently_accepted(value):
    out = checked(pm_obi_change_5s=value)
    assert out["violation_count"] == 1
    assert out["violations"][0]["reason"] == "INVALID_NUMERIC_VALUE"
    json.dumps(out, allow_nan=False)


def test_arbitrary_float_or_repr_hooks_not_invoked():
    class Hostile:
        def __float__(self):
            raise AssertionError("must not call arbitrary float")
        def __repr__(self):
            raise AssertionError("must not call arbitrary repr")
    assert checked(pm_obi_change_1s=Hostile())["status"] == "DATA_SANITY_WARNING"


@pytest.mark.parametrize("field,value", [("opp_best_bid", -0.01), ("opp_best_ask", 1.01),
    ("opp_mid", 1.01), ("opp_bid_depth_1", -1), ("initial_ask_depth_3", -1),
    ("btc_start_price", 0), ("btc_last_price", -1), ("btc_cutoff_price", 0)])
def test_other_old_numeric_constraints_retained(field, value):
    out = checked(**{field: value})
    assert out["status"] == "DATA_SANITY_WARNING"
    assert out["violations"][0]["field"] == field


def test_crossed_book_is_not_masked_by_delta_fix():
    out = checked(opp_best_bid=0.6, opp_best_ask=0.5, pm_obi_change_1s=1.7)
    assert out["violation_count"] == 1
    assert out["violations"][0]["field"] == "crossed_book"
    assert out["violations"][0]["value"] == [0.6, 0.5]


def test_bad_price_has_own_error_without_crossed_comparison_exception():
    out = checked(opp_best_bid="bad", opp_best_ask=0.5)
    assert out["violation_count"] == 1
    assert out["violations"][0]["reason"] == "INVALID_NUMERIC_VALUE"


def test_unknown_depth_obi_field_obeys_both_existing_rules():
    assert checked(depth_obi=-0.5)["status"] == "DATA_SANITY_WARNING"
    assert checked(depth_obi=0.5)["status"] == "PASS"


def test_none_absence_and_empty_input_do_not_claim_completeness():
    for out in (sanity_audit_v2([]), checked(pm_obi_change_1s=None), checked()):
        assert out["checked_value_count"] == 0 and out["status"] == "PASS"
        assert out["scope"] == AUDIT_SCOPE
        assert "eligible" not in out and "fully_covered" not in out


def test_unrelated_fields_and_coverage_are_not_reinterpreted():
    # This is a narrow value checker, not a new coverage or schema validator.
    out = checked(book_pre10_coverage_pct=0.1, btc_pre30_coverage_pct=0.2,
                  model_eligible=False, model_ineligible_reason="MISSING", label="text")
    assert out["checked_value_count"] == 0


@pytest.mark.parametrize("rows", [{}, "bad", b"bad", [None], [{1: 0}], 42])
def test_bad_structural_inputs_raise(rows):
    with pytest.raises(TypeError):
        sanity_audit_v2(rows)


def test_input_and_caller_flags_unchanged_and_details_detached():
    rows = [{"condition_id": "c", "opp_best_bid": 0.9, "opp_best_ask": 0.1,
             "model_eligible": False, "sanity_pass": False, "pm_obi_change_1s": 1.6}]
    before = deepcopy(rows)
    out = sanity_audit_v2(rows)
    out["violations"][0]["value"].append("edit")
    assert rows == before


def test_generator_is_consumed_once_and_mapping_order_does_not_change_result():
    row = {"condition_id": "c", "pm_obi_change_5s": 2.1, "opp_obi_1": 1.1, "opp_mid": -0.1}
    assert sanity_audit_v2(iter([row])) == sanity_audit_v2([dict(reversed(list(row.items())))])


def test_total_count_not_silently_truncated_to_details():
    out = sanity_audit_v2({"condition_id": str(i), "opp_obi_1": 2} for i in range(137))
    assert out["row_count"] == out["violation_count"] == 137
    assert len(out["violations"]) == 100 and out["violations_truncated"] is True
    assert out["violations"][0]["row_index"] == 0 and out["violations"][-1]["row_index"] == 99


def test_version_explicit_and_no_automatic_switch_of_legacy():
    old = legacy_checker()
    row = {"condition_id": "c", "pm_obi_change_1s": 1.6}
    before = old([row])
    out = sanity_audit_v2([row])
    assert out["audit_version"] == AUDIT_VERSION and out["status"] == "PASS"
    assert old([row]) == before and before["status"] == "DATA_SANITY_WARNING"
    assert "audit_version" not in before


def test_frozen_range_bug_reproduced_by_legitimate_endpoint_difference():
    current, previous = 0.8, -0.8
    delta = current - previous
    row = {"condition_id": "c", "opp_obi_1": current, "pm_obi_change_1s": delta}
    assert legacy_checker()([row])["violation_count"] == 1
    assert sanity_audit_v2([row])["violation_count"] == 0


def test_delta_does_not_hide_bad_current_level():
    out = checked(opp_obi_1=1.5, pm_obi_change_1s=1.6)
    assert [v["field"] for v in out["violations"]] == ["opp_obi_1"]


def test_no_raw_endpoint_validation_is_invented():
    # In-range values alone do not prove that either historical endpoint exists.
    out = checked(pm_obi_change_1s=1.6)
    assert out["status"] == "PASS" and out["scope"] == AUDIT_SCOPE


def test_decimal_context_does_not_round_comparisons():
    with localcontext() as ctx:
        ctx.prec = 2
        assert checked(pm_obi_change_1s=Decimal("2.00000000000000000001"))["status"] == "DATA_SANITY_WARNING"
        assert checked(pm_obi_change_1s=Decimal("1.99999999999999999999"))["status"] == "PASS"


def test_finite_native_and_numeric_string_values_supported():
    out = checked(pm_obi_change_1s="1.6", opp_obi_1=Decimal("0.3"),
                  opp_bid_depth_1=10**1000, btc_last_price="65000", opp_mid=0.5)
    assert out["status"] == "PASS"


def test_legacy_agreement_for_ordinary_finite_values_outside_changed_fields():
    old = legacy_checker()
    rng = random.Random(410)
    for _ in range(120):
        row = {"condition_id": "c", "opp_best_bid": rng.uniform(-0.3, 1.3),
               "opp_best_ask": rng.uniform(-0.3, 1.3), "opp_mid": rng.uniform(-0.3, 1.3),
               "opp_obi_1": rng.uniform(-1.5, 1.5), "opp_bid_depth_1": rng.uniform(-1, 10),
               "btc_start_price": rng.uniform(-1, 10)}
        a, b = old([row]), sanity_audit_v2([row])
        assert a["status"] == b["status"] and a["violation_count"] == b["violation_count"]
        assert sorted((v["field"], v["value"]) for v in a["violations"]) == sorted(
            (v["field"], v["value"]) for v in b["violations"])


def test_malformed_number_reason_independent_of_callers_decimal_traps():
    before = checked(pm_obi_change_1s="bad")
    with localcontext() as context:
        context.traps[InvalidOperation] = False
        assert checked(pm_obi_change_1s="bad") == before
        assert context.traps[InvalidOperation] is False
