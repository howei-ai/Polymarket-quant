"""Pure, bounded SHADOW orchestration over caller-provided PIT rows."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Mapping

from . import anthropic_k3_transport, openai_jev_transport
from .analysts import K3_PROMPT_SCHEMA_VERSION, analyst_request
from .audit import hash_k3_request, sha256_json
from .context import ShadowContext, build_shadow_context
from .debate import debate_request
from .deterministic_gate import classify_shadow
from .interfaces import ProbabilisticJudge, ReasoningModel
from .jev_client import JevClient
from .jev_questions import JEV_QUESTION_SCHEMA_VERSION, build_jev_request
from .k3_client import K3Client
from .provider_common import CallLedger
from .research_manager import manager_request, validate_manager_synthesis
from .schemas import (ANALYST_ROLES, DEBATE_ROLES, K3_ROLES, AnalystReport, GateEvidence,
                      JevJudgments, ManagerReport, ModelCallRecord, ShadowResult)


SHADOW_SCHEMA_VERSION = "k3_jev_agent_shadow_v3"
_EMPTY_HASH = sha256_json(None)
_LOG = logging.getLogger(__name__)
# (client class, transport class, transport version, roles) of each registered model path.
_K3_PATH = (K3Client, anthropic_k3_transport.AnthropicK3Transport, anthropic_k3_transport.TRANSPORT_VERSION, K3_ROLES)
_JEV_PATH = (JevClient, openai_jev_transport.OpenAIJevTransport, openai_jev_transport.TRANSPORT_VERSION, ("JEV",))


def _response_hash(raw: Any) -> str:
    try:
        return sha256_json(raw)
    except (TypeError, ValueError, OverflowError):
        return sha256_json({"unserializable_response_type": type(raw).__name__})


def _drain_calls(model: Any) -> tuple[ModelCallRecord, ...]:
    drain = getattr(model, "drain_call_records", None)
    records = tuple(drain()) if callable(drain) else ()
    if any(not isinstance(record, ModelCallRecord) for record in records):
        raise TypeError("model call records must be ModelCallRecord")
    return records


def _registered_transport(model: Any, path: tuple[Any, ...]) -> Any:
    """The transport of an exact registered client around the exact registered transport, else None."""
    client_type, transport_type = path[0], path[1]
    if type(model) is not client_type:
        return None
    transport = vars(model).get("_transport")
    return transport if type(transport) is transport_type else None


def _ledger_records(transport: Any) -> tuple[ModelCallRecord, ...]:
    """Drains the transport's ledger through the class functions, so instance overrides are ignored."""
    ledger = vars(transport).get("_ledger")
    if type(ledger) is not CallLedger:
        raise TypeError("registered transport must keep its CallLedger")
    records = CallLedger.drain(ledger)
    if any(not isinstance(record, ModelCallRecord) for record in records):
        raise TypeError("model call records must be ModelCallRecord")
    return records


def _provenance(model: Any, path: tuple[Any, ...],
                expected_roles: tuple[str, ...]) -> tuple[str, str, tuple[ModelCallRecord, ...], bool]:
    """(provider, transport_version, call records, consistent) for one model.

    Labels and records are taken only from the exact registered client wrapping the exact
    registered transport whose client and binding still have the trust level its version claims
    (checked through the class functions); any other object is "unspecified" and its
    self-reported records are discarded. A registered pair whose client, binding, labels or
    records disagree is inconsistent: labels become "unspecified", records are dropped, and the
    row is BLOCKED. Each call the runner made (expected_roles, in order) must have exactly one
    record, so outputs produced without the transport cannot carry genuine labels. A registered
    client whose transport is no longer the registered one is inconsistent too. Deliberate
    in-process tampering beyond this is outside the threat model (see README).
    """
    client_type, transport_type, version, roles = path
    transport = _registered_transport(model, path)
    if transport is None and type(model) is client_type:
        return "unspecified", "unspecified", (), False
    if transport is None:
        try:
            discarded = len(_drain_calls(model))
        except Exception:
            discarded = -1
        if discarded:
            _LOG.warning("discarded call records from an unregistered model object (count=%d)", discarded)
        return "unspecified", "unspecified", (), True
    try:
        records = _ledger_records(transport)
        binding = transport.binding
        test_client = transport._ledger.test_client is True
        expected_version = version + ("-test-client" if test_client else "")
        consistent = (
            transport_type._client_still_trusted(transport)
            and model.transport_version == expected_version == transport.transport_version
            and model.provider == binding.provider and model.model_id == binding.model_id
            and all(record.provider == binding.provider and record.requested_model_id == binding.model_id
                    and record.role in roles and record.test_client is test_client for record in records)
            and tuple(record.role for record in records) == expected_roles
        )
    except Exception:
        consistent = False
    if not consistent:
        return "unspecified", "unspecified", (), False
    return model.provider, model.transport_version, records, True


class ShadowRunner:
    """Exactly four analysts, one bull/bear round, one manager and one judge.

    Not thread-safe: use one runner (with its own transports) per thread.
    """

    def __init__(self, reasoner: ReasoningModel, judge: ProbabilisticJudge, *, rounds: int = 1) -> None:
        if type(rounds) is not int or rounds != 1:
            raise ValueError("only one bounded debate round is supported")
        # Read once: a model attribute that later changes or raises cannot alter or break a row.
        model_ids = []
        for model, name in ((reasoner, "K3"), (judge, "Jev")):
            model_id = getattr(model, "model_id", None)
            if not isinstance(model_id, str) or not model_id.strip():
                raise TypeError(f"{name} requires a recorded model_id")
            model_ids.append(model_id)
        self._k3_model_id, self._jev_model_id = model_ids
        if "latest" in self._jev_model_id.lower():
            raise ValueError("Jev requires a versioned model_id")
        self.reasoner = reasoner
        self.judge = judge
        self._seen: set[str] = set()

    def run(self, row: Mapping[str, Any], evidence: GateEvidence) -> ShadowResult:
        if not isinstance(evidence, GateEvidence):
            raise TypeError("GateEvidence required")
        # Records left by calls outside run() must not be attributed to this row; log metadata only.
        stale: tuple[ModelCallRecord, ...] = ()
        for model, path in ((self.reasoner, _K3_PATH), (self.judge, _JEV_PATH)):
            try:
                transport = _registered_transport(model, path)
                stale += _ledger_records(transport) if transport is not None else _drain_calls(model)
            except Exception:  # registered models with unreadable records are BLOCKED by result(); others are unspecified
                _LOG.warning("could not read stale model call records from %s", type(model).__name__)
        if stale:
            _LOG.warning("discarded %d model call records not made by run(): roles=%s outcomes=%s cost_usd=%.6f",
                         len(stale), [r.role for r in stale], [r.outcome for r in stale],
                         sum(r.cost_usd or 0.0 for r in stale))
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

        k3_roles_called: list[str] = []
        jev_roles_called: list[str] = []

        def result(status: str, reasons: tuple[str, ...]) -> ShadowResult:
            k3_provider, k3_version, k3_calls, k3_consistent = _provenance(
                self.reasoner, _K3_PATH, tuple(k3_roles_called))
            jev_provider, jev_version, jev_calls, jev_consistent = _provenance(
                self.judge, _JEV_PATH, tuple(jev_roles_called))
            if not (k3_consistent and jev_consistent):
                status, reasons = "BLOCKED", tuple(reasons) + ("MODEL_PROVENANCE_INVALID",)
            if any(call.budget_overrun for call in k3_calls + jev_calls):
                status, reasons = "BLOCKED", tuple(reasons) + ("BUDGET_OVERRUN",)
            return ShadowResult(
                schema_version=SHADOW_SCHEMA_VERSION,
                condition_id=cid,
                decision_cutoff_ts=cutoff,
                input_feature_sha256=context.input_feature_sha256 if context else _EMPTY_HASH,
                context_sha256=context.context_sha256 if context else _EMPTY_HASH,
                gate_evidence=evidence,
                k3_model_id=self._k3_model_id,
                k3_prompt_schema_version=K3_PROMPT_SCHEMA_VERSION,
                k3_request_hashes=dict(requests),
                k3_response_hashes=dict(responses),
                jev_model_id=self._jev_model_id,
                jev_question_schema_version=JEV_QUESTION_SCHEMA_VERSION,
                jev_request_hash=jev_request_hash,
                jev_response_hash=jev_response_hash,
                k3_provider=k3_provider,
                k3_transport_version=k3_version,
                jev_provider=jev_provider,
                jev_transport_version=jev_version,
                model_calls=k3_calls + jev_calls,
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
            requests[role] = hash_k3_request(self._k3_model_id, request)
            try:
                k3_roles_called.append(role)
                if _registered_transport(self.reasoner, _K3_PATH) is not None:
                    raw = K3Client.analyze(self.reasoner, request)  # class function: instance overrides ignored
                else:
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

        try:
            state = {
                "context": context.to_dict(),
                "analyst_reports": [asdict(report) for report in analysts],
                "bull_report": asdict(bull),
                "bear_report": asdict(bear),
                "manager_report": asdict(manager),
            }
            question_request = build_jev_request(
                self._jev_model_id, state, schema_version=JEV_QUESTION_SCHEMA_VERSION,
            )
            jev_request_hash = sha256_json(question_request)
            jev_roles_called.append("JEV")
            if _registered_transport(self.judge, _JEV_PATH) is not None:
                raw = JevClient.judge(self.judge, question_request["state"], question_request["questions"])
            else:
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
