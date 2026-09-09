"""Exclusive submit path v1: OFFLINE orchestration, NOT a LIVE gateway.

Reuse the frozen OrderIntent, production-eligibility verifier, risk gate,
portfolio reservations, and DurableIntentLedgerV1. This module has no venue,
network transport, credentials, fill processing, or deployment switch.
Only the exact built-in OfflineSubmitStubV1 is accepted, never an arbitrary
transport. Python object privacy is not a security sandbox.

One coordinator owns one locked ledger and a COPY of one reconciled portfolio.
The operator must supply trusted policy/evidence anchors, a trusted clock, and
one authoritative ledger per account. Multiple independent ledgers do not
coordinate account risk. External checkpoints must be independently trusted.

A nonempty recovered ledger blocks ALL new dispatch until external
reconciliation, which v1 deliberately does not implement. An intent record
cannot establish whether a previous dispatch happened. Session outcomes and
reservations are IN MEMORY ONLY. ACK reserves resources but does not fill.
Unknown outcomes and persistence failures latch the path closed and retain
reservations; neither retry nor close releases them. A known offline rejection
or a post-persistence/pre-dispatch stop releases only its own reservation.

Pending BUY notionals count against market/gross limits in addition to held
positions. Outstanding SELLs reserve inventory and never reduce exposure until
actual fills (outside this version). Decisions, reservation, precommit and
stub dispatch are serialized. Kill switch is sampled at entry and immediately
before dispatch; it cannot cancel an already started call. This is not an
exactly-once venue guarantee, PRECOMMITTED_COVERAGE, or production authorization.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable

from std0_quant.execution.contracts import OrderEvent, OrderEventType, OrderIntent
from std0_quant.execution.durable_intent_ledger_v1 import (
    DurableIntentLedgerV1,
    DurableIntentReceiptV1,
    LedgerCheckpointV1,
    order_intent_payload_hash_v1,
)
from std0_quant.execution.execution_governance_bridge_v2 import ExecutionGovernanceBridgeArtifactV2
from std0_quant.execution.execution_validation import ExecutionValidationTarget
from std0_quant.execution.execution_validation_policy_v2 import (
    ExecutionValidationDecisionV2, ExecutionValidationPolicyV2,
)
from std0_quant.execution.portfolio import PortfolioState, Position
from std0_quant.execution.production_eligibility_gate_v2 import (
    ELIGIBLE, ProductionEligibilityDecisionV2,
    verify_production_eligibility_decision_v2,
)
from std0_quant.execution.risk import (
    RiskContext, RiskLimits, RiskOrderIntent, RiskResult, evaluate_order_risk,
)
from std0_quant.research.factors.registry import FactorRegistryRecord

OFFLINE = "OFFLINE"
POLICY_SCHEMA = "exclusive_submit_policy_v1"
RESULT_SCHEMA = "exclusive_submit_result_v1"


class SubmitPathUnavailableError(RuntimeError):
    """Closed, forked, reentrant, or structurally unavailable coordinator."""


class SubmitIntentConflictError(ValueError):
    """A consumed ID was reused with different normalized content."""


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be str")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty, without surrounding whitespace")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be int/float, not bool")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _hash(value: object, name: str) -> str:
    value = _text(value, name)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _now_ms() -> float:
    return time.time_ns() / 1_000_000


@dataclass(frozen=True)
class ExclusiveSubmitPolicyV1:
    """Trusted OFFLINE configuration, not a caller-supplied approval flag.

    The gate hash is pinned by the operator; self-consistent hashes alone do
    not authenticate evidence. Strategy identity follows build_order_intent:
    strategy_id/version == target.alpha_id/alpha_version.
    """
    policy_id: str
    version: str
    target: ExecutionValidationTarget
    eligibility_artifact_hash: str
    limits: RiskLimits
    max_decision_age_ms: float
    fee_reserve_bps: float = 0.0
    mode: str = OFFLINE
    schema_version: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        _text(self.policy_id, "policy_id")
        _text(self.version, "version")
        if type(self.target) is not ExecutionValidationTarget:
            raise TypeError("target must be ExecutionValidationTarget")
        if type(self.limits) is not RiskLimits:
            raise TypeError("limits must be RiskLimits")
        for key, value in asdict(self.target).items():
            _text(value, "target." + key)
        _hash(self.target.definition_hash, "definition_hash")
        _hash(self.eligibility_artifact_hash, "eligibility_artifact_hash")
        for key, value in asdict(self.limits).items():
            _number(value, "limits." + key)
        for key in ("max_decision_age_ms", "fee_reserve_bps"):
            object.__setattr__(self, key, _number(getattr(self, key), key))
        if self.mode != OFFLINE or self.schema_version != POLICY_SCHEMA:
            raise ValueError("only OFFLINE exclusive submit policy v1 is supported")


def exclusive_submit_policy_v1_hash(policy: ExclusiveSubmitPolicyV1) -> str:
    if type(policy) is not ExclusiveSubmitPolicyV1:
        raise TypeError("policy must be ExclusiveSubmitPolicyV1")
    policy = deepcopy(policy)
    policy.__post_init__()
    return hashlib.sha256(b"std0-quant/exclusive-submit-policy/v1\n"
                          + _canonical(asdict(policy))).hexdigest()


@dataclass(frozen=True)
class SubmitGovernanceEvidenceV1:
    record: FactorRegistryRecord
    bridge: ExecutionGovernanceBridgeArtifactV2
    decision: ExecutionValidationDecisionV2
    policy: ExecutionValidationPolicyV2
    eligibility: ProductionEligibilityDecisionV2

    def __post_init__(self) -> None:
        for name, expected in (
            ("record", FactorRegistryRecord),
            ("bridge", ExecutionGovernanceBridgeArtifactV2),
            ("decision", ExecutionValidationDecisionV2),
            ("policy", ExecutionValidationPolicyV2),
            ("eligibility", ProductionEligibilityDecisionV2),
        ):
            if type(getattr(self, name)) is not expected:
                raise TypeError(f"{name} must be {expected.__name__}")


@dataclass(frozen=True)
class OfflineDispatchRequestV1:
    intent_json: str
    intent_hash: str
    receipt: DurableIntentReceiptV1
    eligibility_artifact_hash: str
    submit_policy_hash: str
    submit_ts_ms: float
    mode: str = OFFLINE


@dataclass(frozen=True)
class OfflineStubResponseV1:
    mode: str
    intent_hash: str
    record_hash: str
    event: OrderEvent


class OfflineSubmitStubV1:
    """Fixed no-network fixture. ACK and REJECT always have zero fills.

    TIMEOUT and MALFORMED are explicit offline fault scenarios. This is NOT
    an adapter protocol accepting arbitrary implementations or live endpoints.
    """
    def __init__(self, scenario: str = "ACK") -> None:
        if scenario not in {"ACK", "REJECT", "TIMEOUT", "MALFORMED"}:
            raise ValueError("unknown offline stub scenario")
        self._scenario = scenario
        self._token: object | None = None
        self._requests: list[OfflineDispatchRequestV1] = []
        self._lock = threading.Lock()

    @property
    def requests(self) -> tuple[OfflineDispatchRequestV1, ...]:
        with self._lock:
            return deepcopy(tuple(self._requests))

    def _claim(self, token: object) -> None:
        with self._lock:
            if self._token is not None:
                raise SubmitPathUnavailableError("offline stub already owned")
            self._token = token

    def _release(self, token: object) -> None:
        with self._lock:
            if self._token is token:
                self._token = None

    def _dispatch(self, token: object, request: OfflineDispatchRequestV1) -> OfflineStubResponseV1 | None:
        with self._lock:
            if token is not self._token or token is None or request.mode != OFFLINE:
                raise SubmitPathUnavailableError("dispatch must use the owning offline path")
            self._requests.append(deepcopy(request))
            if self._scenario == "TIMEOUT":
                raise TimeoutError("injected offline timeout after accepting the call")
            if self._scenario == "MALFORMED":
                return None
            intent = OrderIntent.from_json(request.intent_json)
            event_type = OrderEventType.VENUE_ACK if self._scenario == "ACK" else OrderEventType.REJECTED
            event = OrderEvent(
                event_id=f"offline:{intent.intent_id}:{request.receipt.sequence}",
                intent_id=intent.intent_id, event_type=event_type,
                receive_ts_ms=request.submit_ts_ms, venue_ts_ms=None,
                venue_order_id=f"offline:{request.receipt.sequence}" if self._scenario == "ACK" else None,
                fill_qty=0.0, fill_price=None, cumulative_filled_qty=0.0,
                remaining_qty=intent.qty, reason="OFFLINE_STUB_NO_FILL",
            )
            return OfflineStubResponseV1(OFFLINE, request.intent_hash,
                                         request.receipt.record_hash, event)


@dataclass(frozen=True)
class SubmitAttemptResultV1:
    intent_id: str
    intent_hash: str
    submit_policy_hash: str
    eligibility_artifact_hash: str
    status: str
    reasons: tuple[str, ...]
    adapter_called: bool = False
    receipt: DurableIntentReceiptV1 | None = None
    risk: RiskResult | None = None
    event: OrderEvent | None = None
    reserved_cash: float = 0.0
    reserved_sell_qty: float = 0.0
    previous_status: str | None = None
    mode: str = OFFLINE
    schema_version: str = RESULT_SCHEMA


@dataclass(frozen=True)
class _Hold:
    condition_id: str
    outcome: str
    cash: float
    sell_qty: float
    buy_notional: float


def _portfolio_copy(value: PortfolioState) -> PortfolioState:
    if type(value) is not PortfolioState:
        raise TypeError("portfolio must be exactly PortfolioState")
    value = deepcopy(value)
    for name in ("cash", "reserved_cash"):
        _number(getattr(value, name), name)
    if type(value.realized_pnl) not in (float, int) or not math.isfinite(value.realized_pnl):
        raise ValueError("invalid realized_pnl")
    if value.reserved_cash != 0 or value.reserved_positions:
        raise ValueError("initial portfolio must be reconciled with no outstanding reservations")
    if type(value.positions) is not dict or type(value.reserved_positions) is not dict:
        raise TypeError("portfolio positions and reservations must be dictionaries")
    for key, position in value.positions.items():
        if (type(key) is not tuple or len(key) != 2 or type(position) is not Position
                or key != (position.condition_id, position.outcome)):
            raise ValueError("invalid position identity")
        _text(key[0], "condition_id")
        _text(key[1], "outcome")
        _number(position.qty, "position qty")
        _number(position.cost_basis, "cost_basis")
        if position.qty == 0 and position.cost_basis != 0:
            raise ValueError("empty position cannot have a cost basis")
    _number(value.gross_cost_exposure, "gross exposure")
    return value


class ExclusiveSubmitPathV1:
    """Owned Linux ledger + owned portfolio snapshot + built-in offline stub.

    No public recovery-unblock, cancellation, fill settlement, or manual
    reservation release exists. Use create/open_existing and explicit close.
    """
    def __init__(self) -> None:
        raise TypeError("use create() or open_existing()")

    @classmethod
    def create(cls, path: Path | str, *, ledger_id: str, portfolio: PortfolioState,
               policy: ExclusiveSubmitPolicyV1, stub: OfflineSubmitStubV1,
               clock_ms: Callable[[], float] = _now_ms) -> ExclusiveSubmitPathV1:
        return cls._open(path, ledger_id=ledger_id, portfolio=portfolio,
                         policy=policy, stub=stub, clock_ms=clock_ms, create=True,
                         checkpoint=None)

    @classmethod
    def open_existing(cls, path: Path | str, *, ledger_id: str, portfolio: PortfolioState,
                      policy: ExclusiveSubmitPolicyV1, stub: OfflineSubmitStubV1,
                      clock_ms: Callable[[], float] = _now_ms,
                      checkpoint: LedgerCheckpointV1 | None = None) -> ExclusiveSubmitPathV1:
        return cls._open(path, ledger_id=ledger_id, portfolio=portfolio,
                         policy=policy, stub=stub, clock_ms=clock_ms, create=False,
                         checkpoint=checkpoint)

    @classmethod
    def _open(cls, path: Path | str, *, ledger_id: str, portfolio: PortfolioState,
              policy: ExclusiveSubmitPolicyV1, stub: OfflineSubmitStubV1,
              clock_ms: Callable[[], float], create: bool,
              checkpoint: LedgerCheckpointV1 | None) -> ExclusiveSubmitPathV1:
        if type(stub) is not OfflineSubmitStubV1:
            raise TypeError("only the exact built-in OfflineSubmitStubV1 is accepted")
        policy_hash = exclusive_submit_policy_v1_hash(policy)
        owned_portfolio = _portfolio_copy(portfolio)
        if not callable(clock_ms):
            raise TypeError("clock_ms must be a trusted callable")
        self = cls.__new__(cls)
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._kill = threading.Event()
        self._closed = False
        self._in_submit = False
        self._halt_reason: str | None = None
        self._portfolio = owned_portfolio
        self._policy = deepcopy(policy)
        self._policy_hash = policy_hash
        self._clock = clock_ms
        self._stub = stub
        self._token = object()
        self._ledger: DurableIntentLedgerV1 | None = None
        self._holds: dict[str, _Hold] = {}
        self._results: dict[str, SubmitAttemptResultV1] = {}
        self._consumed_hashes: dict[str, str] = {}
        stub._claim(self._token)
        try:
            self._ledger = (DurableIntentLedgerV1.create(path, ledger_id=ledger_id) if create
                            else DurableIntentLedgerV1.open_existing(path, ledger_id=ledger_id,
                                                                     checkpoint=checkpoint))
            self._tip = self._ledger.checkpoint()
            if self._tip.sequence:
                self._halt_reason = "RECOVERED_HISTORY_REQUIRES_RECONCILIATION"
            return self
        except BaseException:
            if self._ledger is not None:
                self._ledger.close()
            stub._release(self._token)
            self._closed = True
            raise

    def _owner(self) -> None:
        if os.getpid() != self._pid:
            raise SubmitPathUnavailableError("fork-inherited submit path")
        if self._closed:
            raise SubmitPathUnavailableError("submit path is closed")

    @property
    def halt_reason(self) -> str | None:
        self._owner()
        with self._lock:
            return self._halt_reason

    def portfolio_snapshot(self) -> PortfolioState:
        """Copy for diagnostics; never a production reconciliation result."""
        self._owner()
        with self._lock:
            return deepcopy(self._portfolio)

    def set_kill_switch(self, active: bool) -> None:
        self._owner()
        if type(active) is not bool:
            raise TypeError("kill switch must be bool")
        # No submit mutex: another thread can stop a call still fsyncing.
        self._kill.set() if active else self._kill.clear()

    def _time_reasons(self, intent: OrderIntent, now: float) -> list[str]:
        reasons = []
        if self._kill.is_set():
            reasons.append("KILL_SWITCH_ACTIVE")
        if now < intent.decision_ts_ms:
            reasons.append("DECISION_FROM_FUTURE")
        elif now - intent.decision_ts_ms > self._policy.max_decision_age_ms:
            reasons.append("STALE_DECISION")
        if now < intent.market_data_ts_ms:
            reasons.append("MARKET_DATA_FROM_FUTURE")
        elif now - intent.market_data_ts_ms > self._policy.limits.max_market_data_age_ms:
            reasons.append("STALE_MARKET_DATA")
        return reasons

    def _governance_reasons(self, intent: OrderIntent, evidence: SubmitGovernanceEvidenceV1) -> list[str]:
        evidence.__post_init__()
        gate = evidence.eligibility
        reasons = list(verify_production_eligibility_decision_v2(
            gate, evidence.record, evidence.bridge, evidence.decision, evidence.policy))
        if gate.status != ELIGIBLE:
            reasons.append("PRODUCTION_ELIGIBILITY_NOT_ELIGIBLE")
        if gate.artifact_hash != self._policy.eligibility_artifact_hash:
            reasons.append("ELIGIBILITY_ANCHOR_MISMATCH")
        if gate.target != self._policy.target:
            reasons.append("GOVERNANCE_TARGET_MISMATCH")
        expected = self._policy.target
        if (intent.strategy_id, intent.strategy_version) != (expected.alpha_id, expected.alpha_version):
            reasons.append("INTENT_STRATEGY_IDENTITY_MISMATCH")
        if intent.risk_policy_version != expected.risk_policy_version:
            reasons.append("INTENT_RISK_POLICY_MISMATCH")
        return reasons

    def _reserve(self, intent: OrderIntent, cash: float) -> None:
        hold = _Hold(intent.condition_id, intent.outcome, cash,
                     intent.qty if intent.side.value == "SELL" else 0.0,
                     intent.order_notional if intent.side.value == "BUY" else 0.0)
        # Reserve on a copy so an exception cannot partially mutate shared state.
        updated = deepcopy(self._portfolio)
        if hold.cash:
            updated.reserve_buy_cash(hold.cash)
        if hold.sell_qty:
            updated.reserve_sell_qty(hold.condition_id, hold.outcome, hold.sell_qty)
        holds = {**self._holds, intent.intent_id: hold}
        self._validate_reservation_candidate(updated, holds)
        self._portfolio = updated
        self._holds = holds

    def _release_hold(self, intent_id: str) -> None:
        hold = self._holds[intent_id]
        updated = deepcopy(self._portfolio)
        if hold.cash:
            updated.release_reserved_cash(hold.cash)
        if hold.sell_qty:
            updated.release_reserved_sell_qty(hold.condition_id, hold.outcome, hold.sell_qty)
        holds = {key: value for key, value in self._holds.items() if key != intent_id}
        self._validate_reservation_candidate(updated, holds)
        self._portfolio = updated
        self._holds = holds

    def _validate_reservation_candidate(self, updated: PortfolioState,
                                        holds: dict[str, _Hold]) -> None:
        # Frozen portfolio helpers normalize tiny values to zero. A candidate
        # snapshot must not erase a still-live hold or change cash/positions.
        # Relative float roundoff is allowed; there is NO absolute tolerance
        # that could make a positive reservation equivalent to zero.
        if (updated.cash != self._portfolio.cash
                or updated.realized_pnl != self._portfolio.realized_pnl
                or updated.positions != self._portfolio.positions):
            raise SubmitPathUnavailableError("reservation update changed non-reservation state")
        expected_cash = math.fsum(hold.cash for hold in holds.values())
        quantities: dict[tuple[str, str], list[float]] = {}
        for hold in holds.values():
            if hold.sell_qty:
                quantities.setdefault((hold.condition_id, hold.outcome), []).append(hold.sell_qty)
        expected_positions = {key: math.fsum(values) for key, values in quantities.items()}

        def matches(actual: float, expected: float) -> bool:
            return (math.isfinite(actual) and actual >= 0
                    and math.isfinite(expected) and expected >= 0
                    and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0))

        if not matches(updated.reserved_cash, expected_cash):
            raise SubmitPathUnavailableError("cash reservation does not match active holds")
        for key in set(updated.reserved_positions) | set(expected_positions):
            if not matches(updated.reserved_positions.get(key, 0.0), expected_positions.get(key, 0.0)):
                raise SubmitPathUnavailableError("inventory reservation does not match active holds")

    def _result(self, intent: OrderIntent, intent_hash: str, gate_hash: str,
                status: str, reasons: tuple[str, ...] = (), **extra: object) -> SubmitAttemptResultV1:
        hold = self._holds.get(intent.intent_id)
        return SubmitAttemptResultV1(
            intent_id=intent.intent_id, intent_hash=intent_hash,
            submit_policy_hash=self._policy_hash, eligibility_artifact_hash=gate_hash,
            status=status, reasons=tuple(dict.fromkeys(reasons)),
            reserved_cash=hold.cash if hold else 0.0,
            reserved_sell_qty=hold.sell_qty if hold else 0.0, **extra,
        )

    def _store(self, result: SubmitAttemptResultV1) -> SubmitAttemptResultV1:
        self._results[result.intent_id] = deepcopy(result)
        return deepcopy(result)

    def _validate_response(self, response: object, request: OfflineDispatchRequestV1,
                           intent: OrderIntent) -> OrderEvent:
        if type(response) is not OfflineStubResponseV1:
            raise ValueError("invalid offline response type")
        if (response.mode != OFFLINE or response.intent_hash != request.intent_hash
                or response.record_hash != request.receipt.record_hash
                or type(response.event) is not OrderEvent):
            raise ValueError("offline response binding mismatch")
        event = OrderEvent.from_dict(response.event.to_dict())
        if (event.intent_id != intent.intent_id
                or event.event_type not in (OrderEventType.VENUE_ACK, OrderEventType.REJECTED)
                or event.fill_qty != 0 or event.fill_price is not None
                or event.cumulative_filled_qty != 0 or event.remaining_qty != intent.qty
                or event.venue_ts_ms is not None or event.receive_ts_ms != request.submit_ts_ms
                or event.reason != "OFFLINE_STUB_NO_FILL"):
            raise ValueError("unsupported offline event semantics")
        if event.event_type == OrderEventType.VENUE_ACK:
            if event.venue_order_id != f"offline:{request.receipt.sequence}":
                raise ValueError("offline acknowledgement identity mismatch")
        elif event.venue_order_id is not None:
            raise ValueError("rejection cannot acknowledge an order")
        return event

    def submit(self, intent: OrderIntent, *, evidence: SubmitGovernanceEvidenceV1) -> SubmitAttemptResultV1:
        self._owner()
        if type(evidence) is not SubmitGovernanceEvidenceV1:
            raise TypeError("evidence must be SubmitGovernanceEvidenceV1")
        digest = order_intent_payload_hash_v1(intent)
        intent = OrderIntent.from_dict(intent.to_dict())
        evidence = deepcopy(evidence)
        evidence.__post_init__()
        with self._lock:
            self._owner()
            if self._in_submit:
                raise SubmitPathUnavailableError("reentrant submit is forbidden")
            self._in_submit = True
            try:
                return self._submit_locked(intent, digest, evidence)
            except SubmitIntentConflictError:
                raise
            except BaseException:
                # Unexpected exceptions must not leave a route open to new risk.
                self._halt_reason = self._halt_reason or "UNEXPECTED_FAILURE_REQUIRES_RECONCILIATION"
                raise
            finally:
                self._in_submit = False

    def _submit_locked(self, intent: OrderIntent, digest: str,
                       evidence: SubmitGovernanceEvidenceV1) -> SubmitAttemptResultV1:
        gate_hash = evidence.eligibility.artifact_hash
        old_hash = self._consumed_hashes.get(intent.intent_id)
        if old_hash is not None:
            if old_hash != digest:
                raise SubmitIntentConflictError("intent ID already consumed with different content")
            old = self._results.get(intent.intent_id)
            original_gate_hash = old.eligibility_artifact_hash if old else self._policy.eligibility_artifact_hash
            return self._result(intent, digest, original_gate_hash, "DUPLICATE",
                                ("ALREADY_CONSUMED_NO_REDISPATCH",),
                                receipt=old.receipt if old else None,
                                previous_status=old.status if old else "UNKNOWN")
        if self._halt_reason:
            return self._result(intent, digest, gate_hash, "RECONCILIATION_REQUIRED", (self._halt_reason,))
        try:
            if self._ledger.checkpoint() != self._tip:
                raise SubmitPathUnavailableError("ledger changed outside this coordinator")
        except Exception:
            self._halt_reason = "LEDGER_UNAVAILABLE_REQUIRES_RECONCILIATION"
            return self._result(intent, digest, gate_hash, "RECONCILIATION_REQUIRED", (self._halt_reason,))
        reasons = self._governance_reasons(intent, evidence)
        now = _number(self._clock(), "clock_ms")
        reasons.extend(self._time_reasons(intent, now))
        if reasons:
            return self._result(intent, digest, gate_hash, "BLOCKED", tuple(reasons))
        notional = _number(intent.order_notional, "order notional")
        if notional <= 0:
            raise ValueError("order notional must remain positive after multiplication")
        fee = _number(notional * self._policy.fee_reserve_bps / 10_000, "fee reserve")
        cash = _number(notional + fee, "cash requirement") if intent.side.value == "BUY" else 0.0
        risk = evaluate_order_risk(
            portfolio=deepcopy(self._portfolio),
            intent=RiskOrderIntent(intent.condition_id, intent.outcome, intent.side.value,
                                   intent.qty, intent.limit_price),
            limits=self._policy.limits,
            context=RiskContext(now, intent.market_data_ts_ms, self._kill.is_set(), fee),
        )
        reasons.extend(risk.reasons)
        if not risk.allowed and not risk.reasons:
            reasons.append("RISK_NOT_ALLOWED")
        pending_market = sum(h.buy_notional for h in self._holds.values()
                             if h.condition_id == intent.condition_id)
        pending_gross = sum(h.buy_notional for h in self._holds.values())
        # Do not credit an unfilled SELL with reducing current exposure.
        market_worst = max(risk.market_exposure_before, risk.market_exposure_after) + pending_market
        gross_worst = max(risk.gross_exposure_before, risk.gross_exposure_after) + pending_gross
        if not math.isfinite(market_worst) or market_worst > self._policy.limits.max_market_exposure:
            reasons.append("PENDING_MAX_MARKET_EXPOSURE_EXCEEDED")
        if not math.isfinite(gross_worst) or gross_worst > self._policy.limits.max_gross_exposure:
            reasons.append("PENDING_MAX_GROSS_EXPOSURE_EXCEEDED")
        # Avoid the upstream numerical tolerance permitting actual overspend.
        if cash > self._portfolio.available_cash:
            reasons.append("INSUFFICIENT_AVAILABLE_CASH")
        if intent.side.value == "SELL" and intent.qty > self._portfolio.available_position_qty(
                intent.condition_id, intent.outcome):
            reasons.append("INSUFFICIENT_AVAILABLE_POSITION")
        if reasons:
            return self._result(intent, digest, gate_hash, "BLOCKED", tuple(reasons), risk=risk)
        self._reserve(intent, cash)
        self._consumed_hashes[intent.intent_id] = digest
        receipt = None
        try:
            receipt = self._ledger.append(intent)
            tip = self._ledger.checkpoint()
            stored = self._ledger.get_intent(intent.intent_id)
            if (type(receipt) is not DurableIntentReceiptV1 or stored is None
                    or order_intent_payload_hash_v1(stored) != digest
                    or receipt.intent_id != intent.intent_id or receipt.intent_hash != digest
                    or receipt.ledger_id != self._tip.ledger_id
                    or receipt.sequence != self._tip.sequence + 1
                    or tip != LedgerCheckpointV1(receipt.ledger_id, receipt.sequence, receipt.record_hash)
                    or self._ledger.receipts()[-1] != receipt):
                raise SubmitPathUnavailableError("precommit receipt binding mismatch")
            self._tip = tip
        except Exception:
            self._halt_reason = "PRECOMMIT_FAILED_REQUIRES_RECONCILIATION"
            return self._store(self._result(intent, digest, gate_hash, "PRECOMMIT_FAILED",
                                            (self._halt_reason,), risk=risk))
        # Disk sync latency can make an initially fresh decision stale.
        dispatch_now = _number(self._clock(), "clock_ms")
        reasons = self._time_reasons(intent, dispatch_now)
        if dispatch_now < now:
            reasons.append("CLOCK_MOVED_BACKWARDS")
        try:
            if self._ledger.checkpoint() != self._tip:
                raise SubmitPathUnavailableError("ledger changed after precommit")
        except Exception:
            self._halt_reason = "LEDGER_CHANGED_BEFORE_DISPATCH_REQUIRES_RECONCILIATION"
            return self._store(self._result(intent, digest, gate_hash, "RECONCILIATION_REQUIRED",
                                            (self._halt_reason,), receipt=receipt, risk=risk))
        # The checkpoint itself may block long enough for time/kill to change.
        # Re-sample trusted controls after that I/O; preserve any earlier stop.
        # The trusted clock must not mutate ledger state. An already started
        # dispatch still cannot be cancelled by a later kill-switch update.
        final_now = _number(self._clock(), "clock_ms")
        reasons.extend(self._time_reasons(intent, final_now))
        if final_now < dispatch_now:
            reasons.append("CLOCK_MOVED_BACKWARDS")
        dispatch_now = final_now
        if reasons:
            self._release_hold(intent.intent_id)  # We know no adapter call occurred.
            return self._store(self._result(intent, digest, gate_hash, "NOT_DISPATCHED",
                                            tuple(reasons), receipt=receipt, risk=risk))
        request = OfflineDispatchRequestV1(intent.to_json(), digest, receipt, gate_hash,
                                           self._policy_hash, dispatch_now)
        try:
            response = self._stub._dispatch(self._token, request)
            event = self._validate_response(response, request, intent)
        except Exception:
            self._halt_reason = "DISPATCH_OUTCOME_UNKNOWN_REQUIRES_RECONCILIATION"
            return self._store(self._result(intent, digest, gate_hash, "UNKNOWN",
                                            (self._halt_reason,), adapter_called=True,
                                            receipt=receipt, risk=risk))
        if event.event_type == OrderEventType.REJECTED:
            self._release_hold(intent.intent_id)
            status = "OFFLINE_REJECTED"
        else:
            status = "OFFLINE_ACK"
        return self._store(self._result(intent, digest, gate_hash, status,
                                        adapter_called=True, receipt=receipt, risk=risk, event=event))

    def close(self) -> None:
        if os.getpid() != self._pid:
            raise SubmitPathUnavailableError("fork-inherited submit path must not be used")
        with self._lock:
            if self._in_submit:
                raise SubmitPathUnavailableError("cannot close inside submit")
            if self._closed:
                return
            self._closed = True
            try:
                if self._ledger is not None:
                    self._ledger.close()
            finally:
                self._stub._release(self._token)
            # Deliberately no reservation release and no ledger rewrite.

    def __enter__(self) -> ExclusiveSubmitPathV1:
        self._owner()
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        try:
            self.close()
        except Exception:
            if exc is None:
                raise
            exc.add_note("submit path close also failed")
