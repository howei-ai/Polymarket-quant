"""Opt-in, present-value sanity checks v2; NOT research eligibility.

The original prospective.sanity_audit and all existing callers stay unchanged.
Only the exact pm_obi_change_1s / pm_obi_change_5s fields use [-2, 2].
Other lowercase 'obi' fields retain [-1, 1]; existing probability, crossed
book, nonnegative depth and positive BTC-price checks are retained.

v2 also diagnoses invalid / nonfinite values in those governed fields. Native
int, float, Decimal and numeric str values are supported; bool is not numeric.
Decimal comparisons prevent conversion rounding from admitting out-of-range
values. No clipping, epsilon, imputation or input mutation is performed.

Missing / None fields remain outside these checks, as in v1. Empty input or
only absent fields can therefore return PASS with checked_value_count=0.
PASS never establishes completeness, coverage, raw-data correctness, lineage,
collector provenance, historical truth, cohort membership or trading approval.
This module performs no I/O and changes no stored flags or policy thresholds.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal, DecimalException, InvalidOperation, localcontext
import math
from typing import Any

AUDIT_VERSION = "sanity_audit_v2"
AUDIT_SCOPE = "PRESENT_GOVERNED_VALUES_ONLY_NOT_ELIGIBILITY"
OBI_CHANGE_FIELDS = frozenset({"pm_obi_change_1s", "pm_obi_change_5s"})
_PRICE_FIELDS = ("opp_best_bid", "opp_best_ask", "opp_mid")
_BTC_FIELDS = ("btc_start_price", "btc_last_price", "btc_cutoff_price")
_MAX_DETAILS = 100


def _finite_number(value: Any) -> tuple[Decimal | None, str | None]:
    if type(value) not in (int, float, str, Decimal):
        return None, "INVALID_NUMERIC_VALUE"
    try:
        with localcontext() as context:
            context.traps[InvalidOperation] = True
            number = Decimal.from_float(value) if type(value) is float else Decimal(value)
    except (DecimalException, ValueError, OverflowError):
        return None, "INVALID_NUMERIC_VALUE"
    if not number.is_finite():
        return None, "NONFINITE_VALUE"
    return number, None


def _report_value(value: Any) -> Any:
    # Detached, JSON-safe diagnostics; never invoke arbitrary repr/float hooks.
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite_float": str(value)}
    if type(value) is Decimal:
        return {"decimal": str(value)}
    if type(value) in (str, int, float, bool) or value is None:
        return value
    return {"unsupported_type": type(value).__name__}


def sanity_audit_v2(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return version-tagged diagnostics; consume rows once without editing them.

    Shape errors raise TypeError. Bad present governed values produce WARNING.
    Counts cover all rows; at most the first 100 violations are retained.
    Rows retain input order; fields are evaluated in stable lexical order.
    """
    if isinstance(rows, (Mapping, str, bytes)):
        raise TypeError("rows must be an iterable of mappings, not a single mapping/string")
    violations: list[dict[str, Any]] = []
    row_count = checked_count = violation_count = 0

    def add(index: int, cid: Any, field: str, value: Any, reason: str) -> None:
        nonlocal violation_count
        violation_count += 1
        if len(violations) < _MAX_DETAILS:
            violations.append({"row_index": index, "condition_id": _report_value(cid),
                               "field": field, "value": value, "reason": reason})

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or any(type(key) is not str for key in row):
            raise TypeError("each row must be a mapping with string keys")
        row_count += 1
        cid = row.get("condition_id")
        fields = sorted(name for name in row if name in _PRICE_FIELDS or name in _BTC_FIELDS
                        or "obi" in name or "depth" in name)
        numbers: dict[str, Decimal] = {}
        for name in fields:
            value = row[name]
            if value is None:
                continue
            checked_count += 1
            number, problem = _finite_number(value)
            if problem is not None:
                add(index, cid, name, _report_value(value), problem)
                continue
            assert number is not None
            numbers[name] = number
            invalid = (name in _PRICE_FIELDS and not 0 <= number <= 1)
            invalid = invalid or (name in _BTC_FIELDS and number <= 0)
            invalid = invalid or ("depth" in name and number < 0)
            if "obi" in name:
                bound = 2 if name in OBI_CHANGE_FIELDS else 1
                invalid = invalid or not -bound <= number <= bound
            if invalid:
                add(index, cid, name, _report_value(value), "OUT_OF_RANGE")
        bid, ask = numbers.get("opp_best_bid"), numbers.get("opp_best_ask")
        if bid is not None and ask is not None and bid > ask:
            add(index, cid, "crossed_book",
                [_report_value(row["opp_best_bid"]), _report_value(row["opp_best_ask"])],
                "CROSSED_BOOK")

    return {"audit_version": AUDIT_VERSION, "scope": AUDIT_SCOPE,
            "status": "PASS" if not violation_count else "DATA_SANITY_WARNING",
            "row_count": row_count, "checked_value_count": checked_count,
            "violation_count": violation_count, "violations": violations,
            "violations_truncated": violation_count > len(violations)}
