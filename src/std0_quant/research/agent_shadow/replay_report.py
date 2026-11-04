"""Read-only summaries of bound, redacted replay artifacts.

Hashes establish integrity, not authenticity of the input evidence or relay.
Reports contain aggregates only, never identifiers, model text or request ids.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .audit import canonical_json, sha256_json
from .deterministic_gate import classify_shadow
from .model_registry import ModelBinding
from .replay import ARTIFACTS_FILE, RUN_BINDING_FILE, STOP_REASONS, is_final, replay_run_binding
from .research_manager import validate_manager_synthesis
from .runner import SHADOW_SCHEMA_VERSION
from .schemas import (
    ANALYST_ROLES, K3_ROLES, AnalystReport, GateEvidence, JevJudgments, ManagerReport,
    ModelCallRecord, ShadowResult,
)


REPORT_VERSION = "agent_shadow_replay_report_v1"
MAX_BINDING_BYTES = 65_536
MAX_ARTIFACT_BYTES = 8 * 1_024 * 1_024
GATE_REASONS = frozenset({
    "COVERAGE_INVALID_OR_UNKNOWN", "PROVENANCE_INVALID_OR_UNKNOWN", "SANITY_INVALID_OR_UNKNOWN",
    "CONTEXT_PIT_INVALID", "K3_OUTPUTS_INVALID", "JEV_OUTPUTS_INVALID", "MODEL_OUTPUT_UNAVAILABLE",
    "K3_INSUFFICIENT_SUPPORT", "JEV_DIRECTION_NOT_CONSISTENT", "JEV_BOOK_ANOMALY",
    "JEV_EXECUTION_RISK", "JEV_LOW_CANDIDATE_QUALITY", "MODEL_PROVENANCE_INVALID",
    "BUDGET_OVERRUN", "DUPLICATE_CONDITION_ID", "K3_TIMEOUT", "JEV_TIMEOUT",
    "RELAY_NOT_ALLOWED", "MODEL_ROUTE_MIXED", "TEST_CLIENT_MIXED",
})


class ReplayReportError(ValueError):
    """Input cannot be summarized consistently; messages never include row data."""


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ReplayReportError("duplicate JSON field")
        result[key] = value
    return result


def _constant(_: str) -> None:
    raise ReplayReportError("non-finite JSON number")


def _json(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    if type(value) is not dict:
        raise ReplayReportError("JSON object required")
    return value


def _binding(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise ReplayReportError("binding must not be a symlink")
    with path.open("rb") as handle:
        raw = handle.read(MAX_BINDING_BYTES + 1)
    if len(raw) > MAX_BINDING_BYTES:
        raise ReplayReportError("binding exceeds the size limit")
    try:
        value = _json(raw)
        models = {}
        for role in ("K3", "JEV"):
            model = dict(value["models"][role])
            model.pop("transport_version")
            model["accepted_served_model_ids"] = tuple(model["accepted_served_model_ids"])
            models[role] = ModelBinding(**model)
        expected = replay_run_binding(value["route"], test_clients=value["provenance"] == "TEST_CLIENT",
                                      bindings=models)
        if canonical_json(value) != canonical_json(expected):
            raise ValueError("binding mismatch")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ReplayReportError("unsupported or invalid replay binding") from exc
    return value, hashlib.sha256(raw).hexdigest()


def _artifact(payload: dict[str, Any], binding: dict[str, Any]) -> ShadowResult:
    expected_keys = {field.name for field in fields(ShadowResult)} | {"artifact_sha256"}
    if set(payload) != expected_keys or payload["schema_version"] != SHADOW_SCHEMA_VERSION:
        raise ReplayReportError("unsupported artifact schema")
    data = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if payload["artifact_sha256"] != sha256_json(data):
        raise ReplayReportError("artifact hash mismatch")
    if (type(data["condition_id"]) is not str or not re.fullmatch(r"row_[0-9a-f]{24}", data["condition_id"])
            or type(data["decision_cutoff_ts"]) is not int
            or data["decision_cutoff_ts"] != binding["redaction"]["cutoff_ts_ms"]):
        raise ReplayReportError("artifact is not a redacted replay row")
    if type(data["gate_reasons"]) is not list or any(reason not in GATE_REASONS for reason in data["gate_reasons"]):
        raise ReplayReportError("unsupported gate reason")
    data["gate_reasons"] = tuple(data["gate_reasons"])
    data["gate_evidence"] = GateEvidence(**data["gate_evidence"])
    data["model_calls"] = tuple(ModelCallRecord(**call) for call in data["model_calls"])
    data["analyst_reports"] = tuple(AnalystReport.parse(report, report["role"]) for report in data["analyst_reports"])
    for name, role in (("bull_report", "BULL_RESEARCHER"), ("bear_report", "BEAR_RESEARCHER")):
        data[name] = AnalystReport.parse(data[name], role) if data[name] is not None else None
    data["manager_report"] = ManagerReport.parse(data["manager_report"]) if data["manager_report"] is not None else None
    data["jev_judgments"] = JevJudgments.parse(data["jev_judgments"]) if data["jev_judgments"] is not None else None
    result = ShadowResult(**data)
    if canonical_json(result.to_dict()) != canonical_json(payload):
        raise ReplayReportError("artifact requires normalization")
    if any(report.role != role for report, role in zip(result.analyst_reports, ANALYST_ROLES)) or len(result.analyst_reports) > 4:
        raise ReplayReportError("invalid analyst order")
    for role, prefix in (("K3", "k3"), ("JEV", "jev")):
        model = binding["models"][role]
        if (getattr(result, prefix + "_model_id") != model["model_id"]
                or getattr(result, prefix + "_provider") not in (model["provider"], "unspecified")
                or getattr(result, prefix + "_transport_version") not in (model["transport_version"], "unspecified")):
            raise ReplayReportError("model does not match replay binding")
    if (result.k3_prompt_schema_version != binding["k3_prompt_schema_version"]
            or result.jev_question_schema_version != binding["jev_question_schema_version"]):
        raise ReplayReportError("prompt contract does not match replay binding")
    if result.route == "unspecified":
        if result.gate_status != "BLOCKED" or not set(result.gate_reasons) & STOP_REASONS:
            raise ReplayReportError("unverified route requires a setup failure")
    elif result.route != binding["route"] or result.provenance != binding["provenance"]:
        raise ReplayReportError("route or provenance does not match replay binding")
    else:
        for role, prefix in (("K3", "k3"), ("JEV", "jev")):
            model = binding["models"][role]
            if (getattr(result, prefix + "_provider") != model["provider"]
                    or getattr(result, prefix + "_transport_version") != model["transport_version"]):
                raise ReplayReportError("registered labels do not match replay binding")
        roles = tuple(call.role for call in result.model_calls)
        if roles != (K3_ROLES + ("JEV",))[:len(roles)]:
            raise ReplayReportError("invalid call order")
    for call in result.model_calls:
        role = "JEV" if call.role == "JEV" else "K3"
        model = binding["models"][role]
        if (call.route != binding["route"] or call.test_client is not (binding["provenance"] == "TEST_CLIENT")
                or call.provider != model["provider"] or call.requested_model_id != model["model_id"]):
            raise ReplayReportError("call does not match replay binding")
        if call.outcome == "OK" and call.served_model_id not in model["accepted_served_model_ids"]:
            raise ReplayReportError("successful call has an unexpected served model")
        if call.cost_usd is not None:
            if call.cost_is_upper_bound:
                # Unknown cache counters still leave raw input and any known
                # output as a minimum cost; known billable input is stronger.
                input_tokens = call.billable_input_tokens
                if input_tokens is None:
                    input_tokens = call.input_tokens or 0
                output_tokens = call.output_tokens or 0
            else:
                if call.billable_input_tokens is None or call.output_tokens is None:
                    raise ReplayReportError("priced usage is missing")
                input_tokens, output_tokens = call.billable_input_tokens, call.output_tokens
            expected_cost = (input_tokens / 1_000_000 * model["input_usd_per_mtok"]
                             + output_tokens / 1_000_000 * model["output_usd_per_mtok"])
            if not math.isfinite(expected_cost):
                raise ReplayReportError("cost does not match billable usage")
            close = math.isclose(call.cost_usd, expected_cost, rel_tol=1e-12, abs_tol=1e-12)
            if call.cost_is_upper_bound:
                if call.cost_usd < expected_cost and not close:
                    raise ReplayReportError("estimated cost is below known usage")
            elif not close:
                raise ReplayReportError("cost does not match billable usage")
    if any(call.budget_overrun for call in result.model_calls) and (
            result.gate_status != "BLOCKED" or "BUDGET_OVERRUN" not in result.gate_reasons):
        raise ReplayReportError("budget overrun requires a blocked result")
    if result.gate_status != "BLOCKED":
        if (len(result.analyst_reports) != 4 or result.bull_report is None or result.bear_report is None
                or result.manager_report is None or result.jev_judgments is None
                or tuple(call.role for call in result.model_calls) != K3_ROLES + ("JEV",)
                or any(call.outcome != "OK" for call in result.model_calls)):
            raise ReplayReportError("completed classification requires complete model outputs")
        if (set(result.k3_request_hashes) != set(K3_ROLES)
                or set(result.k3_response_hashes) != set(K3_ROLES)):
            raise ReplayReportError("completed classification requires complete K3 call hashes")
        validate_manager_synthesis(result.manager_report, result.analyst_reports + (result.bull_report, result.bear_report))
        status, reasons = classify_shadow(result.gate_evidence, context_pit_valid=True, k3_outputs_valid=True,
                                         jev_outputs_valid=True, manager=result.manager_report, jev=result.jev_judgments)
        if status != result.gate_status or set(reasons) != set(result.gate_reasons):
            raise ReplayReportError("classification disagrees with declared model outputs")
    return result


def _calls() -> dict[str, Any]:
    return {"records": 0, "outcomes": Counter(), "priced_records": 0, "unpriced_records": 0,
            "upper_bound_records": 0, "budget_overrun_records": 0, "priced_cost_usd": Decimal(0),
            **{name + suffix: 0 for name in ("input_tokens", "billable_input_tokens", "output_tokens")
               for suffix in ("_known_sum", "_unknown_records")}}


def _add_calls(stats: dict[str, Any], call: ModelCallRecord) -> None:
    stats["records"] += 1
    stats["outcomes"][call.outcome] += 1
    for name in ("input_tokens", "billable_input_tokens", "output_tokens"):
        value = getattr(call, name)
        stats[name + ("_unknown_records" if value is None else "_known_sum")] += 1 if value is None else value
    if call.cost_usd is None:
        stats["unpriced_records"] += 1
    else:
        stats["priced_records"] += 1
        stats["priced_cost_usd"] += Decimal(str(call.cost_usd))
    stats["upper_bound_records"] += call.cost_is_upper_bound
    stats["budget_overrun_records"] += call.budget_overrun


def summarize_replay(directory: Path) -> dict[str, Any]:
    """Validate a v3 directory and aggregate every attempt, including retry costs.

    Final row counts use replay.is_final; unresolved rows may have exhausted
    their attempt cap. Only files named artifacts.jsonl and replay_binding.json
    are read. No credential, redaction map, feature, manifest or model is used.
    """
    directory = Path(directory).expanduser()
    binding, binding_hash = _binding(directory / RUN_BINDING_FILE)
    path = directory / ARTIFACTS_FILE
    if path.is_symlink():
        raise ReplayReportError("artifacts must not be a symlink")
    present = path.exists()
    source_hash = hashlib.sha256()
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    evidence: Counter[str] = Counter()
    rows: dict[str, dict[str, Any]] = {}
    calls, by_role = _calls(), {role: _calls() for role in K3_ROLES + ("JEV",)}
    attempts = nonfinal = 0
    # Finite binary64 costs span fewer than 650 decimal places; preserve their
    # sums even when the total is larger than a representable float.
    with localcontext() as context:
        context.prec = 1_000
        if present:
            with path.open("rb") as handle:
                number = 0
                while raw := handle.readline(MAX_ARTIFACT_BYTES + 1):
                    number += 1
                    source_hash.update(raw)
                    try:
                        if len(raw) > MAX_ARTIFACT_BYTES:
                            raise ReplayReportError("artifact exceeds the size limit")
                        if not raw.strip():
                            continue
                        if not raw.endswith(b"\n"):
                            raise ReplayReportError("incomplete JSONL line")
                        payload = _json(raw)
                        result = _artifact(payload, binding)
                        row = rows.setdefault(result.condition_id, {"attempts": 0, "final": False})
                        if row["final"]:
                            raise ReplayReportError("attempt after a final result")
                        row.update(attempts=row["attempts"] + 1, final=is_final(payload),
                                   status=result.gate_status, reasons=result.gate_reasons)
                        attempts += 1
                        nonfinal += not row["final"]
                        counts[result.gate_status] += 1
                        reasons.update(set(result.gate_reasons))
                        evidence.update(name for name in ("coverage_valid", "provenance_valid", "sanity_valid")
                                        if getattr(result.gate_evidence, name) is not True)
                        for call in result.model_calls:
                            _add_calls(calls, call)
                            _add_calls(by_role[call.role], call)
                    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
                        raise ReplayReportError(f"invalid artifact at line {number}") from exc
        for stats in (calls, *by_role.values()):
            stats["outcomes"] = dict(stats["outcomes"])
            stats["priced_cost_usd"] = format(stats["priced_cost_usd"], "f")
            stats["all_records_have_cost"] = stats["unpriced_records"] == 0
    final = [row for row in rows.values() if row["final"]]
    report = {
        "schema_version": REPORT_VERSION, "scope": "DECLARED_SHADOW_ARTIFACTS_ONLY",
        "replay_version": binding["replay_version"], "route": binding["route"], "provenance": binding["provenance"],
        "models": {role: {key: binding["models"][role][key] for key in ("model_id", "effort")} for role in ("K3", "JEV")},
        "sources": {"binding_sha256": binding_hash, "artifacts_sha256": source_hash.hexdigest(), "artifacts_present": present},
        "attempts": attempts, "nonfinal_attempts": nonfinal, "rows": len(rows), "final_rows": len(final),
        "unresolved_rows": len(rows) - len(final), "attempt_status_counts": dict(counts),
        "latest_status_counts": dict(Counter(row["status"] for row in rows.values())),
        "final_status_counts": dict(Counter(row["status"] for row in final)),
        "final_accept_rate": sum(row["status"] == "SHADOW_ACCEPT" for row in final) / len(final) if final else None,
        "attempt_gate_reason_counts": dict(reasons), "final_gate_reason_counts": dict(Counter(reason for row in final for reason in set(row["reasons"]))),
        "attempt_count_histogram": dict(Counter(str(row["attempts"]) for row in rows.values())),
        "evidence_invalid_attempt_counts": dict(evidence), "model_calls": calls, "model_calls_by_role": by_role,
        "execution_allowed": False, "formal_cohort_effect": False,
    }
    report["report_sha256"] = sha256_json(report)
    return report
