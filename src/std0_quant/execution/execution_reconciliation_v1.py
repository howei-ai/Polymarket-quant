"""OFFLINE, checkpoint-scoped execution-observation reconciliation v1.

This additive module reuses OrderIntent/OrderEvent, ledger v1 receipts and the
OrderStatus vocabulary. It never submits, cancels, releases a reservation,
settles a fill, changes a portfolio, unlocks a submit path, or loads credentials.
A terminal observation is NOT venue truth, complete evidence, or permission to
resume. Authentication, authoritative account/ledger selection, event capture
and durable storage of events/results remain external.

The caller supplies an independently trusted checkpoint. Every intent in its
EXACT prefix is reconstructed and hash-chain checked. No event-only universe.
The optional capture helper reads an already-open locked ledger; it neither
opens/closes that handle nor writes/fsyncs it. A concurrent append during
capture is rejected, not silently adopted. Other ledger methods may mark a
corrupt handle unusable as defined by frozen ledger v1.

Events are a caller-ordered tuple, scoped to this single ledger, bound to full
receipts. Exact duplicates are counted but replayed once. Conflicting event IDs
or a venue order ID reused for different intents are conflicts. No sorting to
repair missing causal history. Receive timestamps must be nondecreasing per
intent and within the explicit receive-time cutoff. Venue timestamps stay
separate; they are not compared with local clocks or used to invent order.

Replay follows frozen order_state.py transition rules, but does not use its
mutable, epsilon-normalizing arithmetic. Quantities use exact rational values
of the normalized contract's decimal strings (Fraction(str(value))); outputs
are exact decimal strings. There is no absolute tolerance that could erase a
small positive fill. This conservative profile may flag rounded source data;
source-specific precision adapters and venue-specific TIF validation are NOT
implemented here. Missing remaining
quantities or missing lifecycle prerequisites remain unresolved. CANCEL_REQUESTED
is not terminal and permits fills while cancellation is pending.

Hashes bind the entire snapshot, ordered supplied events including duplicate
multiplicity, cutoff, outcomes and declared scope; the local reconciliation run
ID alone is excluded. The verifier recomputes semantics from supplied inputs,
not just a checksum. Limits are in-memory single-batch guardrails, not a scale
or power-loss certification. No PRECOMMITTED_COVERAGE or LIVE authorization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from typing import Any

from std0_quant.execution.contracts import OrderEvent, OrderEventType, OrderIntent
from std0_quant.execution.durable_intent_ledger_v1 import (
    DurableIntentLedgerV1, DurableIntentReceiptV1, LedgerCheckpointV1,
    HEADER_SCHEMA, RECORD_SCHEMA, MAX_LINE_BYTES, order_intent_payload_hash_v1,
)
from std0_quant.execution.order_state import OrderStatus, TERMINAL_STATUSES

OFFLINE = "OFFLINE"
SCOPE = "SUPPLIED_EVENTS_ONLY"
SNAPSHOT_SCHEMA = "execution_ledger_snapshot_v1"
EVENT_SCHEMA = "execution_reconciliation_event_v1"
REPORT_SCHEMA = "execution_reconciliation_report_v1"
RECONCILER_VERSION = "execution_reconciliation_v1"
QUANTITY_RULE = "exact_normalized_decimal_v1"
OBSERVED_TERMINAL = "OBSERVED_TERMINAL"
OBSERVATIONS_CONSISTENT = "OBSERVATIONS_CONSISTENT"
UNRESOLVED = "UNRESOLVED"
CONFLICT = "CONFLICT"
EMPTY_SCOPE = "EMPTY_SCOPE"
MAX_INTENTS = 100_000
MAX_EVENTS = 500_000
_HASH_DOMAIN = b"std0-quant/execution-reconciliation/v1/"
_LEDGER_DOMAIN = b"std0-quant/durable-intent-ledger/v1/"


class ReconciliationInputError(ValueError):
    """Malformed, changed or unanchored input; no trustworthy report produced."""


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be str")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty without surrounding whitespace")
    value.encode("utf-8")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be int/float, not bool")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(kind: str, value: Any, *, domain: bytes = _HASH_DOMAIN) -> str:
    return hashlib.sha256(domain + kind.encode("ascii") + b"\n" + _canonical(value)).hexdigest()


def _hash_text(value: object) -> str:
    value = _text(value, "hash")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("hash must be lowercase SHA256")
    return value


def _checkpoint(value: LedgerCheckpointV1) -> LedgerCheckpointV1:
    if type(value) is not LedgerCheckpointV1:
        raise TypeError("checkpoint must be exactly LedgerCheckpointV1")
    result = LedgerCheckpointV1(**asdict(value))
    if len(result.ledger_id.encode("utf-8")) > 256:
        raise ValueError("ledger_id exceeds the ledger v1 limit")
    return result


def _receipt(value: DurableIntentReceiptV1) -> DurableIntentReceiptV1:
    if type(value) is not DurableIntentReceiptV1:
        raise TypeError("receipt must be exactly DurableIntentReceiptV1")
    return DurableIntentReceiptV1(**asdict(value))


def _intent(value: OrderIntent) -> OrderIntent:
    # This frozen helper revalidates normalization and every OrderIntent field.
    order_intent_payload_hash_v1(value)
    return OrderIntent.from_dict(value.to_dict())


def _event(value: OrderEvent) -> OrderEvent:
    if type(value) is not OrderEvent:
        raise TypeError("event must be exactly contracts.OrderEvent")
    row = value.to_dict()
    normalized = OrderEvent.from_dict(row)
    if _canonical(row) != _canonical(normalized.to_dict()):
        raise ReconciliationInputError("event is not in normalized contract form")
    if normalized.venue_order_id is not None:
        _text(normalized.venue_order_id, "venue_order_id")
    if normalized.reason is not None and type(normalized.reason) is not str:
        raise TypeError("event reason must be str or None")
    return normalized


@dataclass(frozen=True)
class ReconciliationIntentV1:
    intent: OrderIntent
    receipt: DurableIntentReceiptV1

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _intent(self.intent))
        object.__setattr__(self, "receipt", _receipt(self.receipt))


@dataclass(frozen=True)
class ExecutionLedgerSnapshotV1:
    checkpoint: LedgerCheckpointV1
    entries: tuple[ReconciliationIntentV1, ...]
    schema_version: str = SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", _checkpoint(self.checkpoint))
        if type(self.entries) is not tuple:
            raise TypeError("snapshot entries must be a tuple")
        if len(self.entries) > MAX_INTENTS:
            raise ValueError("snapshot exceeds MAX_INTENTS")
        if any(type(entry) is not ReconciliationIntentV1 for entry in self.entries):
            raise TypeError("snapshot requires ReconciliationIntentV1 entries")
        if self.schema_version != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported snapshot schema")
        object.__setattr__(self, "entries", tuple(
            ReconciliationIntentV1(entry.intent, entry.receipt) for entry in self.entries))


def _validate_snapshot(snapshot: ExecutionLedgerSnapshotV1,
                       checkpoint: LedgerCheckpointV1) -> None:
    if snapshot.checkpoint != checkpoint or len(snapshot.entries) != checkpoint.sequence:
        raise ReconciliationInputError("snapshot does not match the exact trusted checkpoint scope")
    # Independent reconstruction of the public ledger v1 schema, not its private helpers.
    previous = _digest("header", {"schema_version": HEADER_SCHEMA,
                                 "ledger_id": checkpoint.ledger_id}, domain=_LEDGER_DOMAIN)
    seen: set[str] = set()
    for sequence, entry in enumerate(snapshot.entries, 1):
        value, receipt = entry.intent, entry.receipt
        intent_hash = order_intent_payload_hash_v1(value)
        payload = {"schema_version": RECORD_SCHEMA, "ledger_id": checkpoint.ledger_id,
                   "sequence": sequence, "previous_hash": previous,
                   "intent": value.to_dict(), "intent_hash": intent_hash}
        current = _digest("record", payload, domain=_LEDGER_DOMAIN)
        expected = DurableIntentReceiptV1(checkpoint.ledger_id, sequence,
                                          value.intent_id, intent_hash, current)
        if receipt != expected or value.intent_id in seen:
            raise ReconciliationInputError("receipt identity/sequence/hash chain mismatch")
        if len(_canonical({**payload, "record_hash": current})) + 1 > MAX_LINE_BYTES:
            raise ReconciliationInputError("snapshot contains an oversized ledger record")
        seen.add(value.intent_id)
        previous = current
    if previous != checkpoint.record_hash:
        raise ReconciliationInputError("reconstructed history does not reach trusted checkpoint")


def capture_execution_ledger_snapshot_v1(
    ledger: DurableIntentLedgerV1, *, checkpoint: LedgerCheckpointV1,
) -> ExecutionLedgerSnapshotV1:
    """Read a stable prefix of an already-open ledger; no append/close/repair."""
    if type(ledger) is not DurableIntentLedgerV1:
        raise TypeError("ledger must be exactly DurableIntentLedgerV1")
    anchor = _checkpoint(checkpoint)
    before = ledger.checkpoint()
    if (before.ledger_id != anchor.ledger_id or anchor.sequence > before.sequence
            or anchor.sequence > MAX_INTENTS):
        raise ReconciliationInputError("requested checkpoint is outside available history")
    receipts = ledger.receipts()
    if len(receipts) != before.sequence:
        raise ReconciliationInputError("ledger snapshot changed while reading receipts")
    entries = []
    for receipt in receipts[:anchor.sequence]:
        intent = ledger.get_intent(receipt.intent_id)
        if intent is None:
            raise ReconciliationInputError("ledger intent missing during snapshot")
        entries.append(ReconciliationIntentV1(intent, receipt))
    snapshot = ExecutionLedgerSnapshotV1(anchor, tuple(entries))
    _validate_snapshot(snapshot, anchor)
    if ledger.checkpoint() != before:
        raise ReconciliationInputError("ledger changed during snapshot capture; retry explicitly")
    return snapshot


@dataclass(frozen=True)
class ReconciliationEventV1:
    receipt: DurableIntentReceiptV1
    event: OrderEvent
    mode: str = OFFLINE
    schema_version: str = EVENT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt", _receipt(self.receipt))
        object.__setattr__(self, "event", _event(self.event))
        if self.mode != OFFLINE or self.schema_version != EVENT_SCHEMA:
            raise ValueError("only OFFLINE reconciliation event v1 is supported")


@dataclass(frozen=True)
class IntentReconciliationV1:
    sequence: int
    intent_id: str
    intent_hash: str
    record_hash: str
    status: str
    reasons: tuple[str, ...]
    observed_event_ids: tuple[str, ...]
    order_status: str | None = None
    filled_qty_decimal: str | None = None
    remaining_qty_decimal: str | None = None
    venue_order_id: str | None = None


@dataclass(frozen=True)
class ExecutionReconciliationReportV1:
    reconciliation_run_id: str
    checkpoint: LedgerCheckpointV1
    snapshot_hash: str
    observations_hash: str
    as_of_receive_ts_ms: float
    status: str
    reasons: tuple[str, ...]
    rows: tuple[IntentReconciliationV1, ...]
    unresolved_intent_ids: tuple[str, ...]
    conflicted_intent_ids: tuple[str, ...]
    observed_terminal_intent_ids: tuple[str, ...]
    orphan_event_ids: tuple[str, ...]
    conflicting_event_ids: tuple[str, ...]
    input_event_count: int
    distinct_event_id_count: int
    duplicate_event_count: int
    artifact_hash: str
    mode: str = OFFLINE
    observation_scope: str = SCOPE
    quantity_rule: str = QUANTITY_RULE
    resume_authorized: bool = False
    reservation_release_authorized: bool = False
    redispatch_authorized: bool = False
    reconciler_version: str = RECONCILER_VERSION
    schema_version: str = REPORT_SCHEMA

    def __post_init__(self) -> None:
        _text(self.reconciliation_run_id, "reconciliation_run_id")
        _checkpoint(self.checkpoint)
        for value in (self.snapshot_hash, self.observations_hash, self.artifact_hash):
            _hash_text(value)
        _number(self.as_of_receive_ts_ms, "as_of_receive_ts_ms")
        if self.status not in {OBSERVATIONS_CONSISTENT, UNRESOLVED, CONFLICT, EMPTY_SCOPE}:
            raise ValueError("unsupported reconciliation status")
        if (self.mode != OFFLINE or self.observation_scope != SCOPE
                or self.quantity_rule != QUANTITY_RULE
                or self.reconciler_version != RECONCILER_VERSION
                or self.schema_version != REPORT_SCHEMA):
            raise ValueError("unsupported offline report scope/schema")
        for name in ("resume_authorized", "reservation_release_authorized", "redispatch_authorized"):
            if getattr(self, name) is not False:
                raise ValueError("a reconciliation report cannot authorize an action")
        if type(self.rows) is not tuple or any(type(row) is not IntentReconciliationV1 for row in self.rows):
            raise TypeError("report rows must be a tuple of IntentReconciliationV1")
        for name in ("input_event_count", "distinct_event_id_count", "duplicate_event_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("report counts must be nonnegative integers")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _canonical(self.to_dict()).decode("utf-8")


def _quantity(value: float) -> Fraction:
    return Fraction(str(value))


def _decimal(value: Fraction) -> str:
    """Exact finite decimal representation, independent of Decimal contexts."""
    denominator, twos, fives = value.denominator, 0, 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1 or value < 0:
        raise ReconciliationInputError("quantity is not a nonnegative finite decimal")
    scale = max(twos, fives)
    integer = value.numerator * 2 ** (scale - twos) * 5 ** (scale - fives)
    if scale == 0:
        return str(integer)
    digits = str(integer).rjust(scale + 1, "0")
    return (digits[:-scale] + "." + digits[-scale:]).rstrip("0").rstrip(".")


_TRANSITIONS = {
    OrderEventType.SUBMITTED: ({OrderStatus.NEW}, OrderStatus.SENT),
    OrderEventType.VENUE_ACK: ({OrderStatus.SENT}, OrderStatus.ACKNOWLEDGED),
    OrderEventType.REJECTED: ({OrderStatus.SENT}, OrderStatus.REJECTED),
    OrderEventType.CANCEL_REQUESTED: (
        {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}, OrderStatus.CANCEL_REQUESTED),
    OrderEventType.CANCELLED: ({OrderStatus.CANCEL_REQUESTED}, OrderStatus.CANCELLED),
    OrderEventType.EXPIRED: (
        {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}, OrderStatus.EXPIRED),
}
_FILL_FROM = {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCEL_REQUESTED}
_FILL_TYPES = {OrderEventType.PARTIAL_FILL, OrderEventType.FILLED}


def _replay(entry: ReconciliationIntentV1, observations: list[ReconciliationEventV1],
            cutoff: float, inherited_reasons: set[str]) -> IntentReconciliationV1:
    value, receipt = entry.intent, entry.receipt
    ids = tuple(dict.fromkeys(item.event.event_id for item in observations))

    def row(status: str, reasons: tuple[str, ...], **extra: Any) -> IntentReconciliationV1:
        return IntentReconciliationV1(receipt.sequence, value.intent_id, receipt.intent_hash,
                                      receipt.record_hash, status, reasons, ids, **extra)

    if inherited_reasons:
        return row(CONFLICT, tuple(sorted(inherited_reasons)))
    if value.decision_ts_ms > cutoff:
        return row(CONFLICT, ("INTENT_AFTER_OBSERVATION_CUTOFF",))
    if not observations:
        return row(UNRESOLVED, ("NO_EXECUTION_EVENTS",))
    state, filled = OrderStatus.NEW, Fraction(0)
    qty = _quantity(value.qty)
    last_time = value.decision_ts_ms
    venue_id = None
    for item in observations:
        event = item.event
        kind = event.event_type
        if event.receive_ts_ms < last_time:
            return row(CONFLICT, ("NONMONOTONIC_OR_PREDECISION_RECEIVE_TIME",))
        if event.receive_ts_ms > cutoff:
            return row(CONFLICT, ("EVENT_AFTER_OBSERVATION_CUTOFF",))
        last_time = event.receive_ts_ms
        if state in TERMINAL_STATUSES:
            return row(CONFLICT, ("EVENT_AFTER_TERMINAL",))
        if state == OrderStatus.NEW and kind != OrderEventType.SUBMITTED:
            return row(UNRESOLVED, ("MISSING_SUBMITTED_EVENT",))
        if state == OrderStatus.SENT and kind in {
            OrderEventType.PARTIAL_FILL, OrderEventType.FILLED,
            OrderEventType.CANCEL_REQUESTED, OrderEventType.CANCELLED, OrderEventType.EXPIRED,
        }:
            return row(UNRESOLVED, ("MISSING_ACKNOWLEDGEMENT",))
        if (kind == OrderEventType.CANCELLED
                and state in {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}):
            return row(UNRESOLVED, ("MISSING_CANCEL_REQUEST",))
        if kind in _FILL_TYPES:
            if state not in _FILL_FROM:
                return row(CONFLICT, ("INVALID_ORDER_TRANSITION",))
        elif state not in _TRANSITIONS[kind][0]:
            return row(CONFLICT, ("INVALID_ORDER_TRANSITION",))
        if kind == OrderEventType.SUBMITTED:
            if event.venue_order_id is not None or event.venue_ts_ms is not None:
                return row(CONFLICT, ("SUBMITTED_CANNOT_PROVE_VENUE_ACK",))
        elif kind != OrderEventType.REJECTED and event.venue_order_id is None:
            return row(UNRESOLVED, ("MISSING_VENUE_ORDER_ID",))
        if event.venue_order_id is not None:
            if venue_id is not None and venue_id != event.venue_order_id:
                return row(CONFLICT, ("VENUE_ORDER_ID_CHANGED",))
            venue_id = event.venue_order_id
        next_filled = filled + (_quantity(event.fill_qty) if kind in _FILL_TYPES else 0)
        if next_filled > qty:
            return row(CONFLICT, ("OVERFILL",))
        if _quantity(event.cumulative_filled_qty) != next_filled:
            return row(CONFLICT, ("CUMULATIVE_FILL_MISMATCH",))
        if event.remaining_qty is None:
            return row(UNRESOLVED, ("MISSING_REMAINING_QUANTITY",))
        if _quantity(event.remaining_qty) != qty - next_filled:
            return row(CONFLICT, ("REMAINING_QUANTITY_MISMATCH",))
        if kind in _FILL_TYPES:
            if ((value.side.value == "BUY" and event.fill_price > value.limit_price)
                    or (value.side.value == "SELL" and event.fill_price < value.limit_price)):
                return row(CONFLICT, ("FILL_OUTSIDE_LIMIT",))
            if kind == OrderEventType.FILLED:
                if next_filled != qty:
                    return row(CONFLICT, ("FILLED_WITH_UNFILLED_QUANTITY",))
                state = OrderStatus.FILLED
            else:
                if not 0 < next_filled < qty:
                    return row(CONFLICT, ("INVALID_PARTIAL_FILL_QUANTITY",))
                if state != OrderStatus.CANCEL_REQUESTED:
                    state = OrderStatus.PARTIALLY_FILLED
        else:
            state = _TRANSITIONS[kind][1]
        filled = next_filled
    terminal = state in TERMINAL_STATUSES
    return row(OBSERVED_TERMINAL if terminal else UNRESOLVED,
               () if terminal else ("ORDER_NOT_TERMINAL",), order_status=state.value,
               filled_qty_decimal=_decimal(filled), remaining_qty_decimal=_decimal(qty - filled),
               venue_order_id=venue_id)


def execution_reconciliation_report_v1_hash(report: ExecutionReconciliationReportV1) -> str:
    if type(report) is not ExecutionReconciliationReportV1:
        raise TypeError("report must be ExecutionReconciliationReportV1")
    payload = asdict(report)
    del payload["reconciliation_run_id"]
    del payload["artifact_hash"]
    return _digest("report", payload)


def reconcile_execution_v1(
    snapshot: ExecutionLedgerSnapshotV1, observations: tuple[ReconciliationEventV1, ...], *,
    checkpoint: LedgerCheckpointV1, as_of_receive_ts_ms: float, reconciliation_run_id: str,
) -> ExecutionReconciliationReportV1:
    """Pure report over supplied observations. Never an authorization decision."""
    _text(reconciliation_run_id, "reconciliation_run_id")
    cutoff = _number(as_of_receive_ts_ms, "as_of_receive_ts_ms")
    anchor = _checkpoint(checkpoint)
    if type(snapshot) is not ExecutionLedgerSnapshotV1:
        raise TypeError("snapshot must be ExecutionLedgerSnapshotV1")
    snapshot = ExecutionLedgerSnapshotV1(snapshot.checkpoint, snapshot.entries, snapshot.schema_version)
    _validate_snapshot(snapshot, anchor)
    if type(observations) is not tuple:
        raise TypeError("observations must be a tuple")
    if len(observations) > MAX_EVENTS:
        raise ValueError("observations exceed MAX_EVENTS")
    if any(type(item) is not ReconciliationEventV1 for item in observations):
        raise TypeError("observations require ReconciliationEventV1")
    observations = tuple(ReconciliationEventV1(item.receipt, item.event, item.mode, item.schema_version)
                         for item in observations)
    entries = {entry.intent.intent_id: entry for entry in snapshot.entries}
    reasons_by_id: dict[str, set[str]] = {identifier: set() for identifier in entries}
    by_id: dict[str, list[ReconciliationEventV1]] = {identifier: [] for identifier in entries}
    global_reasons: set[str] = set()
    orphan_ids: set[str] = set()
    variants: dict[str, set[bytes]] = {}
    distinct: list[ReconciliationEventV1] = []
    duplicates = 0
    for item in observations:
        eid = item.event.event_id
        payload = _canonical(asdict(item))
        seen = variants.setdefault(eid, set())
        if payload in seen:
            duplicates += 1
        else:
            seen.add(payload)
            distinct.append(item)
    conflicting = {eid for eid, payloads in variants.items() if len(payloads) > 1}
    venue_owners: dict[str, set[str]] = {}
    for item in distinct:
        eid, identifier = item.event.event_id, item.event.intent_id
        related = {identifier, item.receipt.intent_id} & entries.keys()
        if eid in conflicting:
            global_reasons.add("EVENT_ID_CONTENT_CONFLICT")
            for key in related:
                reasons_by_id[key].add("EVENT_ID_CONTENT_CONFLICT")
        if identifier not in entries:
            orphan_ids.add(eid)
            global_reasons.add("EVENT_OUTSIDE_CHECKPOINT_SCOPE")
        valid_binding = (identifier in entries
                         and item.receipt == entries[identifier].receipt
                         and item.receipt.intent_id == identifier)
        if not valid_binding:
            global_reasons.add("EVENT_RECEIPT_BINDING_MISMATCH")
            for key in related:
                reasons_by_id[key].add("EVENT_RECEIPT_BINDING_MISMATCH")
        for key in related:
            by_id[key].append(item)
        if item.event.venue_order_id is not None:
            venue_owners.setdefault(item.event.venue_order_id, set()).add(identifier)
    for owners in venue_owners.values():
        if len(owners) > 1:
            global_reasons.add("VENUE_ORDER_ID_REUSED")
            for identifier in owners & entries.keys():
                reasons_by_id[identifier].add("VENUE_ORDER_ID_REUSED")
    rows = tuple(_replay(entry, by_id[entry.intent.intent_id], cutoff,
                         reasons_by_id[entry.intent.intent_id]) for entry in snapshot.entries)
    unresolved = tuple(row.intent_id for row in rows if row.status == UNRESOLVED)
    conflicted = tuple(row.intent_id for row in rows if row.status == CONFLICT)
    terminal = tuple(row.intent_id for row in rows if row.status == OBSERVED_TERMINAL)
    if conflicted:
        global_reasons.add("CONFLICTING_INTENT_HISTORY")
    if global_reasons:
        status = CONFLICT
    elif unresolved:
        status = UNRESOLVED
    elif not rows:
        status = EMPTY_SCOPE
    else:
        status = OBSERVATIONS_CONSISTENT
    if unresolved:
        global_reasons.add("UNRESOLVED_INTENT_HISTORY")
    if not rows:
        global_reasons.add("NO_INTENTS_IN_CHECKPOINT_SCOPE")
    report = ExecutionReconciliationReportV1(
        reconciliation_run_id=reconciliation_run_id, checkpoint=anchor,
        snapshot_hash=_digest("snapshot", asdict(snapshot)),
        observations_hash=_digest("observations", [asdict(item) for item in observations]),
        as_of_receive_ts_ms=cutoff, status=status, reasons=tuple(sorted(global_reasons)),
        rows=rows, unresolved_intent_ids=unresolved, conflicted_intent_ids=conflicted,
        observed_terminal_intent_ids=terminal, orphan_event_ids=tuple(sorted(orphan_ids)),
        conflicting_event_ids=tuple(sorted(conflicting)), input_event_count=len(observations),
        distinct_event_id_count=len(variants), duplicate_event_count=duplicates,
        artifact_hash="0" * 64,
    )
    return replace(report, artifact_hash=execution_reconciliation_report_v1_hash(report))


def verify_execution_reconciliation_report_v1(
    report: ExecutionReconciliationReportV1,
    snapshot: ExecutionLedgerSnapshotV1, observations: tuple[ReconciliationEventV1, ...], *,
    checkpoint: LedgerCheckpointV1, as_of_receive_ts_ms: float,
) -> tuple[str, ...]:
    """Recompute semantics even if a forged report has a recomputed checksum."""
    if type(report) is not ExecutionReconciliationReportV1:
        raise TypeError("report must be ExecutionReconciliationReportV1")
    reasons = []
    if execution_reconciliation_report_v1_hash(report) != report.artifact_hash:
        reasons.append("EXECUTION_RECONCILIATION_REPORT_V1_HASH_MISMATCH")
    expected = reconcile_execution_v1(
        snapshot, observations, checkpoint=checkpoint, as_of_receive_ts_ms=as_of_receive_ts_ms,
        reconciliation_run_id=report.reconciliation_run_id,
    )
    if _canonical(asdict(report)) != _canonical(asdict(expected)):
        reasons.append("EXECUTION_RECONCILIATION_REPORT_V1_SEMANTICS_MISMATCH")
    return tuple(reasons)
