"""Pure, bounded SHADOW orchestration over caller-provided PIT rows."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .analysts import K3_PROMPT_SCHEMA_VERSION, analyst_request
from .audit import hash_k3_request, sha256_json
from .context import ShadowContext, build_shadow_context
from .debate import debate_request
from .deterministic_gate import classify_shadow
from .interfaces import ProbabilisticJudge, ReasoningModel
from .jev_questions import JEV_QUESTION_SCHEMA_VERSION, build_jev_request
from .research_manager import manager_request, validate_manager_synthesis
from .schemas import (ANALYST_ROLES, DEBATE_ROLES, AnalystReport, GateEvidence,
                      JevJudgments, ManagerReport, ShadowResult)


SHADOW_SCHEMA_VERSION = "k3_jev_agent_shadow_v1"
_EMPTY_HASH = sha256_json(None)


def _response_hash(raw: Any) -> str:
    try:
        return sha256_json(raw)
    except (TypeError, ValueError, OverflowError):
        return sha256_json({"unserializable_response_type": type(raw).__name__})


class ShadowRunner:
    """Exactly four analysts, one bull/bear round, one manager and one judge."""

    def __init__(self, reasoner: ReasoningModel, judge: ProbabilisticJudge, *, rounds: int = 1) -> None:
        if type(rounds) is not int or rounds != 1:
            raise ValueError("only one bounded debate round is supported")
        for model, name in ((reasoner, "K3"), (judge, "Jev")):
            if not isinstance(getattr(model, "model_id", None), str) or not model.model_id.strip():
                raise TypeError(f"{name} requires a recorded model_id")
        if "latest" in judge.model_id.lower():
            raise ValueError("Jev requires a versioned model_id")
        self.reasoner = reasoner
        self.judge = judge
        self._seen: set[str] = set()

    def run(self, row: Mapping[str, Any], evidence: GateEvidence) -> ShadowResult:
        if not isinstance(evidence, GateEvidence):
            raise TypeError("GateEvidence required")
        cid = row.get("condition_id") if isinstance(row, Mapping) else None
        if not isinstance(cid, str) or not cid.strip():
            cid = "UNKNOWN"
        cutoff = row.get("decision_cutoff_ts_ms") if isinstance(row, Mapping) else None
        if type(cutoff) is not int or cutoff <= 0:
            cutoff = 0
        context: ShadowContext | None = None
        requests: dict[str, str] = {}
        responses: dict[str, str] = {}
        jev_request_hash = _EMPTY_HASH
        jev_response_hash = _EMPTY_HASH
        analysts: list[AnalystReport] = []
        bull: AnalystReport | None = None
        bear: AnalystReport | None = None
        manager: ManagerReport | None = None
        judgments: JevJudgments | None = None

        def result(status: str, reasons: tuple[str, ...]) -> ShadowResult:
            return ShadowResult(
                schema_version=SHADOW_SCHEMA_VERSION,
                condition_id=cid,
                decision_cutoff_ts=cutoff,
                input_feature_sha256=context.input_feature_sha256 if context else _EMPTY_HASH,
                context_sha256=context.context_sha256 if context else _EMPTY_HASH,
                gate_evidence=evidence,
                k3_model_id=self.reasoner.model_id,
                k3_prompt_schema_version=K3_PROMPT_SCHEMA_VERSION,
                k3_request_hashes=dict(requests),
                k3_response_hashes=dict(responses),
                jev_model_id=self.judge.model_id,
                jev_question_schema_version=JEV_QUESTION_SCHEMA_VERSION,
                jev_request_hash=jev_request_hash,
                jev_response_hash=jev_response_hash,
                analyst_reports=tuple(analysts),
                bull_report=bull,
                bear_report=bear,
                manager_report=manager,
                jev_judgments=judgments,
                gate_status=status,
                gate_reasons=reasons,
            )

        try:
            context = build_shadow_context(row)
        except (TypeError, ValueError):
            return result("BLOCKED", ("CONTEXT_PIT_INVALID",))
        if cid in self._seen:
            return result("BLOCKED", ("DUPLICATE_CONDITION_ID",))
        self._seen.add(cid)
        coverage_ok = all(
            type(context.features.get(name)) in (int, float)
            and 0.99 <= context.features[name] <= 1.0
            for name in ("btc_pre30_coverage_pct", "book_pre10_coverage_pct")
        )
        hard_reasons = tuple(name for name, valid in (
            ("COVERAGE_INVALID_OR_UNKNOWN", evidence.coverage_valid is True and coverage_ok),
            ("PROVENANCE_INVALID_OR_UNKNOWN", evidence.provenance_valid is True),
            ("SANITY_INVALID_OR_UNKNOWN", evidence.sanity_valid is True),
        ) if not valid)
        if hard_reasons:
            return result("BLOCKED", hard_reasons)

        def call_k3(role: str, request: dict[str, Any]) -> Mapping[str, Any]:
            requests[role] = hash_k3_request(self.reasoner.model_id, request)
            try:
                raw = self.reasoner.analyze(request)
            except Exception as exc:
                responses[role] = sha256_json({"error_type": type(exc).__name__})
                raise
            responses[role] = _response_hash(raw)
            return raw

        try:
            for role in ANALYST_ROLES:
                analysts.append(AnalystReport.parse(call_k3(role, analyst_request(role, context)), role))
            analyst_tuple = tuple(analysts)
            bull = AnalystReport.parse(
                call_k3("BULL_RESEARCHER", debate_request("BULL_RESEARCHER", context, analyst_tuple)),
                "BULL_RESEARCHER",
            )
            bear = AnalystReport.parse(
                call_k3("BEAR_RESEARCHER", debate_request("BEAR_RESEARCHER", context, analyst_tuple)),
                "BEAR_RESEARCHER",
            )
            manager = ManagerReport.parse(call_k3("RESEARCH_MANAGER", manager_request(context, analyst_tuple, bull, bear)))
            validate_manager_synthesis(manager, analyst_tuple + (bull, bear))
        except Exception as exc:
            reason = "K3_TIMEOUT" if isinstance(exc, TimeoutError) else "K3_OUTPUTS_INVALID"
            return result("BLOCKED", (reason,))

        state = {
            "context": context.to_dict(),
            "analyst_reports": [asdict(report) for report in analysts],
            "bull_report": asdict(bull),
            "bear_report": asdict(bear),
            "manager_report": asdict(manager),
        }
        question_request = build_jev_request(
            self.judge.model_id, state, schema_version=JEV_QUESTION_SCHEMA_VERSION,
        )
        jev_request_hash = sha256_json(question_request)
        try:
            raw = self.judge.judge(question_request["state"], question_request["questions"])
            jev_response_hash = _response_hash(raw)
            judgments = JevJudgments.parse(raw)
        except Exception as exc:
            if jev_response_hash == _EMPTY_HASH:
                jev_response_hash = sha256_json({"error_type": type(exc).__name__})
            reason = "JEV_TIMEOUT" if isinstance(exc, TimeoutError) else "JEV_OUTPUTS_INVALID"
            return result("BLOCKED", (reason,))
        status, reasons = classify_shadow(
            evidence, context_pit_valid=True, k3_outputs_valid=True,
            jev_outputs_valid=True, manager=manager, jev=judgments,
        )
        return result(status, reasons)
