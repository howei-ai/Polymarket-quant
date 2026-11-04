"""Pure, bounded SHADOW orchestration over caller-provided PIT rows."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Mapping, NamedTuple

from . import openai_k3_transport, openai_jev_transport
from .analysts import K3_PROMPT_SCHEMA_VERSION, analyst_request
from .audit import hash_k3_request, sha256_json
from .context import ShadowContext, build_shadow_context
from .debate import debate_request
from .deterministic_gate import classify_shadow
from .interfaces import ProbabilisticJudge, ReasoningModel
from .jev_client import JevClient
from .jev_questions import JEV_QUESTION_SCHEMA_VERSION, build_jev_request
from .k3_client import K3Client
from .model_registry import CostBudget, ModelBinding, binding_for, require_registered_identity, route_for
from .provider_common import CallLedger, route_policy, transport_version_for
from .research_manager import manager_request, validate_manager_synthesis
from .schemas import (ANALYST_ROLES, DEBATE_ROLES, K3_ROLES, AnalystReport, GateEvidence,
                      JevJudgments, ManagerReport, ModelCallRecord, ShadowResult)


SHADOW_SCHEMA_VERSION = "k3_jev_agent_shadow_v4"
_EMPTY_HASH = sha256_json(None)
_LOG = logging.getLogger(__name__)
# (client class, transport class, transport version, roles) of each registered model path.
_K3_PATH = (K3Client, openai_k3_transport.OpenAIK3Transport, openai_k3_transport.TRANSPORT_VERSION, K3_ROLES)
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


_INVALID = "INVALID"


def _registered_setup(model: Any, path: tuple[Any, ...]) -> tuple[str | None, bool | None]:
    """(route, test_client) of a registered model's ledger: (None, None) for unregistered models,
    (_INVALID, None) for a registered model whose ledger or route is unreadable."""
    transport = _registered_transport(model, path)
    if transport is None:
        return (_INVALID, None) if type(model) is path[0] else (None, None)
    ledger = vars(transport).get("_ledger")
    try:
        return route_for(ledger.route).name, ledger.test_client is True
    except Exception:
        return _INVALID, None


class _Provenance(NamedTuple):
    provider: str
    transport_version: str
    records: tuple[ModelCallRecord, ...]
    consistent: bool
    route: str | None  # None: unregistered model or inconsistent provenance
    test_client: bool


_UNSPECIFIED = ("unspecified", "unspecified", ())


def _provenance(model: Any, path: tuple[Any, ...], expected_roles: tuple[str, ...]) -> _Provenance:
    """Labels, records and route of one model.

    Labels and records are taken only from the exact registered client wrapping the exact
    registered transport whose client, binding and route still have the trust level its version
    claims (checked through the class functions); any other object is "unspecified" and its
    self-reported records are discarded. A registered pair whose client, binding, route, labels
    or records disagree is inconsistent: labels become "unspecified", records are dropped, and
    the row is BLOCKED. Each call the runner made (expected_roles, in order) must have exactly
    one record, so outputs produced without the transport cannot carry genuine labels. A
    registered client whose transport is no longer the registered one is inconsistent too.
    Deliberate in-process tampering beyond this is outside the threat model (see README).
    """
    client_type, transport_type, version, roles = path
    transport = _registered_transport(model, path)
    if transport is None and type(model) is client_type:
        return _Provenance(*_UNSPECIFIED, False, None, False)
    if transport is None:
        try:
            discarded = len(_drain_calls(model))
        except Exception:
            discarded = -1
        if discarded:
            _LOG.warning("discarded call records from an unregistered model object (count=%d)", discarded)
        return _Provenance(*_UNSPECIFIED, True, None, False)
    try:
        records = _ledger_records(transport)
        binding = transport.binding
        test_client = transport._ledger.test_client is True
        route = route_for(transport._ledger.route).name
        expected_version = transport_version_for(version, route, test_client)
        consistent = (
            transport_type._client_still_trusted(transport)
            and model.transport_version == expected_version == transport.transport_version
            and model.provider == binding.provider and model.model_id == binding.model_id
            and all(record.provider == binding.provider and record.requested_model_id == binding.model_id
                    and record.role in roles and record.test_client is test_client and record.route == route
                    for record in records)
            and tuple(record.role for record in records) == expected_roles
        )
    except Exception:
        consistent = False
    if not consistent:
        return _Provenance(*_UNSPECIFIED, False, None, False)
    return _Provenance(model.provider, model.transport_version, records, True, route, test_client)


def _route_reasons(k3_route: str | None, jev_route: str | None, allow_relay: bool,
                   k3_test: bool | None = None, jev_test: bool | None = None) -> tuple[str, ...]:
    """Pre-flight route rules (D2: one route per row; relay only with the explicit opt-in).
    MODEL_ROUTE_MIXED also covers a relay model next to an unregistered one, whose route is unknown."""
    if _INVALID in (k3_route, jev_route):
        return ("MODEL_PROVENANCE_INVALID",)
    reasons: list[str] = []
    known = [route for route in (k3_route, jev_route) if route is not None]
    relay = [route for route in known if route_for(route).provenance != "OFFICIAL"]
    if len(set(known)) > 1 or (relay and len(known) == 1):
        reasons.append("MODEL_ROUTE_MIXED")
    if relay and not allow_relay:
        reasons.append("RELAY_NOT_ALLOWED")
    if k3_test is not None and jev_test is not None and k3_test != jev_test:
        reasons.append("TEST_CLIENT_MIXED")
    return tuple(reasons)


class ShadowRunner:
    """Exactly four analysts, one bull/bear round, one manager and one judge.

    Not thread-safe: use one runner (with its own transports) per thread.
    """

    def __init__(self, reasoner: ReasoningModel, judge: ProbabilisticJudge, *, rounds: int = 1,
                 allow_unverified_relay: bool = False) -> None:
        if type(rounds) is not int or rounds != 1:
            raise ValueError("only one bounded debate round is supported")
        if type(allow_unverified_relay) is not bool:
            raise TypeError("allow_unverified_relay must be an explicit boolean")
        # Relay routes send every request to a third party whose backend cannot be verified; rows
        # on such a route are BLOCKED (RELAY_NOT_ALLOWED) before any call unless this is True.
        self._allow_relay = allow_unverified_relay
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

    def replay_setup(self, budget: CostBudget) -> tuple[str, bool, dict[str, ModelBinding]]:
        """Recorded route and trust level for one registered replay pair.

        The worker binds these to its output directory before resuming; custom
        or mixed model paths cannot establish a comparable replay configuration.
        This does not drain records or replace the per-call provenance checks.
        """
        setups = tuple(_registered_setup(model, path) for model, path in
                       ((self.reasoner, _K3_PATH), (self.judge, _JEV_PATH)))
        if (setups[0] != setups[1] or setups[0][0] in (None, _INVALID)
                or (self._k3_model_id, self._jev_model_id)
                != (binding_for("K3").model_id, binding_for("JEV").model_id)):
            raise ValueError("replay requires registered models on the same route and trust level")
        if any(_registered_transport(model, path)._ledger.budget is not budget
               for model, path in ((self.reasoner, _K3_PATH), (self.judge, _JEV_PATH))):
            raise ValueError("replay budget must be the shared budget of both transports")
        bindings = {role: require_registered_identity(_registered_transport(model, path).binding, role)
                    for role, model, path in (("K3", self.reasoner, _K3_PATH), ("JEV", self.judge, _JEV_PATH))}
        return setups[0][0], setups[0][1], bindings

    def run(self, row: Mapping[str, Any], evidence: GateEvidence) -> ShadowResult:
        """Every provider call of the row runs inside one route policy: relay calls need this
        runner's opt-in and all calls must share one route, however the models are wrapped."""
        routes = {_registered_setup(model, path)[0] for model, path in ((self.reasoner, _K3_PATH),
                                                                          (self.judge, _JEV_PATH))}
        routes.discard(None)
        routes.discard(_INVALID)
        with route_policy(routes.pop() if len(routes) == 1 else None, allow_unverified_relay=self._allow_relay):
            return self._run(row, evidence)

    def _run(self, row: Mapping[str, Any], evidence: GateEvidence) -> ShadowResult:
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
            k3 = _provenance(self.reasoner, _K3_PATH, tuple(k3_roles_called))
            jev = _provenance(self.judge, _JEV_PATH, tuple(jev_roles_called))
            reasons = tuple(reasons)
            if not (k3.consistent and jev.consistent):
                status, reasons = "BLOCKED", reasons + ("MODEL_PROVENANCE_INVALID",)
            if any(call.budget_overrun for call in k3.records + jev.records):
                status, reasons = "BLOCKED", reasons + ("BUDGET_OVERRUN",)
            outcomes = {call.outcome for call in k3.records + jev.records}
            late = _route_reasons(k3.route, jev.route, self._allow_relay,
                                  k3.test_client if k3.route else None, jev.test_client if jev.route else None)
            late += tuple(reason for outcome, reason in (("RELAY_NOT_ALLOWED", "RELAY_NOT_ALLOWED"),
                                                         ("ROUTE_MIXED", "MODEL_ROUTE_MIXED")) if outcome in outcomes)
            late = tuple(dict.fromkeys(reason for reason in late if reason not in reasons))
            if late:  # routes refused at call time or changed after the pre-flight check
                status, reasons = "BLOCKED", reasons + late
            if k3.route is None or jev.route is None or k3.route != jev.route:
                route, provenance = "unspecified", "UNSPECIFIED"
            else:
                route = k3.route
                # A real call outranks a test stand-in, so a real relay call is never hidden as TEST_CLIENT.
                provenance = ("TEST_CLIENT" if k3.test_client and jev.test_client
                              else route_for(route).provenance)
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
                k3_provider=k3.provider,
                k3_transport_version=k3.transport_version,
                jev_provider=jev.provider,
                jev_transport_version=jev.transport_version,
                model_calls=k3.records + jev.records,
                route=route,
                provenance=provenance,
                analyst_reports=tuple(analysts),
                bull_report=bull,
                bear_report=bear,
                manager_report=manager,
                jev_judgments=judgments,
                gate_status=status,
                gate_reasons=reasons,
            )

        (k3_route, k3_test), (jev_route, jev_test) = (_registered_setup(self.reasoner, _K3_PATH),
                                                      _registered_setup(self.judge, _JEV_PATH))
        preflight = _route_reasons(k3_route, jev_route, self._allow_relay, k3_test, jev_test)
        if preflight:  # before any context work or provider call
            return result("BLOCKED", preflight)
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
