"""Read-only resume counts and configured reservation estimates for a replay.

The plan describes the input order and a new run's budget. It does not predict
model responses, actual token usage, early stops or a relay's billed prices.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .audit import canonical_json, sha256_json
from .context import build_shadow_context
from .model_registry import ModelBinding, binding_for, route_for
from .replay import (
    ARTIFACTS_FILE, DEFAULT_MAX_ATTEMPTS, DEFAULT_ROW_HEADROOM_USD, REDACTION_MAP_FILE,
    REPLAY_VERSION, RUN_BINDING_FILE, ReplayRow, replay_output_path, replay_run_binding,
)
from .replay_report import ReplayReportError, ReplayState, read_replay_state
from .schemas import K3_ROLES


PLAN_VERSION = "agent_shadow_replay_run_plan_v1"


def _amount(value: object, name: str, *, positive: bool) -> Decimal:
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return Decimal(str(value))


def _money(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _history(directory: Path) -> ReplayState | None:
    for name in (ARTIFACTS_FILE, RUN_BINDING_FILE, REDACTION_MAP_FILE):
        if (directory / name).is_symlink():
            raise ReplayReportError("replay inputs must not be symlinks")
    if not (directory / RUN_BINDING_FILE).exists():
        artifacts = directory / ARTIFACTS_FILE
        if artifacts.exists() and artifacts.stat().st_size:
            raise ReplayReportError("replay artifacts have no run binding; use a new --out-dir")
        return None
    return read_replay_state(directory)


def _may_call_models(row: ReplayRow) -> bool:
    if not all(getattr(row.evidence, name) is True for name in
               ("coverage_valid", "provenance_valid", "sanity_valid")):
        return False
    try:
        build_shadow_context(row.pit_row)
    except (ValueError, TypeError):
        return False
    return True


def plan_replay_run(rows: Iterable[ReplayRow], *, route: str, out_dir: Path | None = None,
                    forbidden_roots: Iterable[Path] = (), max_rows: int | None = None,
                    budget_usd: float | None = None, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                    row_headroom_usd: float = DEFAULT_ROW_HEADROOM_USD,
                    test_clients: bool = False, bindings: Mapping[str, ModelBinding] | None = None) -> dict[str, Any]:
    """Plan valid replay rows without creating outputs, clients or model calls.

    Existing history is fully validated in one pass. Selection follows the
    worker's input order, final-row exclusion and non-final attempt cap. Row
    limits and a fresh budget can change; the directory binding must match.
    TEST_CLIENT is supported by this local API, not by the real replay CLI.
    """
    if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
        raise ValueError("max_rows must be a positive integer or null")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    headroom = _amount(row_headroom_usd, "row_headroom_usd", positive=False)
    budget = _amount(budget_usd, "budget_usd", positive=True) if budget_usd is not None else None
    models = {role: binding_for(role) if bindings is None else bindings[role] for role in ("K3", "JEV")}
    expected = replay_run_binding(route, test_clients=test_clients, bindings=models)
    history = None
    directory = None
    if out_dir is not None:
        directory = replay_output_path(out_dir, forbidden_roots)
        history = _history(directory)
        if history is not None and canonical_json(history.binding) != canonical_json(expected):
            raise ReplayReportError("replay run binding differs; use a new --out-dir")
    items = tuple(rows)
    ids = {row.redacted_id for row in items}
    if len(ids) != len(items):
        raise ValueError("duplicate replay row identifier")
    done = history.done_ids if history is not None else frozenset()
    attempts = history.nonfinal_attempts if history is not None else {}
    pending = tuple(row for row in items if row.redacted_id not in done
                    and attempts.get(row.redacted_id, 0) < max_attempts)
    selected = pending[:max_rows] if max_rows is not None else pending
    model_rows = sum(_may_call_models(row) for row in selected)
    new_rows = sum(attempts.get(row.redacted_id, 0) == 0 for row in pending)
    with localcontext() as context:
        context.prec = 100
        # Reuse the runtime reservation policy at each binding's input limit.
        per_row = (len(K3_ROLES) * Decimal(str(models["K3"].max_call_cost_usd(models["K3"].max_input_bytes)))
                   + Decimal(str(models["JEV"].max_call_cost_usd(models["JEV"].max_input_bytes))))
        total = model_rows * per_row
        covered = None if budget is None else model_rows if budget >= total else int(budget // per_row)
        estimate = {
            "pricing_source": "REGISTERED_LIST_PRICES", "budget_scope": "NEW_RUN",
            "configured_reservation_usd_per_model_row": _money(per_row),
            "selected_configured_reservation_usd": _money(total),
            "model_rows_covered_by_estimate": covered,
            "budget_meets_start_headroom": None if budget is None else budget >= headroom,
        }
    sources = dict(history.sources) if history is not None else {
        "binding_sha256": None, "artifacts_sha256": None,
        "artifacts_present": directory is not None and (directory / ARTIFACTS_FILE).exists(),
    }
    report = {
        "schema_version": PLAN_VERSION, "replay_version": REPLAY_VERSION,
        "route": route, "provenance": expected["provenance"],
        "relay_opt_in_required_for_run": route_for(route).provenance != "OFFICIAL",
        "models": {role: {"model_id": model.model_id, "effort": model.effort} for role, model in models.items()},
        "planned_binding_sha256": sha256_json(expected), "existing_binding": history is not None,
        "sources": sources, "row_order": "INPUT_ORDER",
        "limits": {"max_rows": max_rows, "max_attempts": max_attempts,
                   "budget_usd": None if budget is None else _money(budget), "row_headroom_usd": _money(headroom)},
        "input_rows": len(items), "already_final_rows": sum(row.redacted_id in done for row in items),
        "attempts_exhausted_rows": sum(row.redacted_id not in done and attempts.get(row.redacted_id, 0) >= max_attempts
                                       for row in items),
        "new_rows": new_rows, "retry_ready_rows": len(pending) - new_rows, "pending_rows": len(pending),
        "selected_rows": len(selected),
        "selected_new_rows": sum(attempts.get(row.redacted_id, 0) == 0 for row in selected),
        "selected_retry_rows": sum(attempts.get(row.redacted_id, 0) > 0 for row in selected),
        "selected_model_call_candidates": model_rows, "selected_preflight_blocked_rows": len(selected) - model_rows,
        "history_rows_outside_input_plan": len((done | set(attempts)) - ids),
        "budget_estimate": estimate, "planning_only": True,
        "execution_allowed": False, "formal_cohort_effect": False,
    }
    report["plan_sha256"] = sha256_json(report)
    return report
