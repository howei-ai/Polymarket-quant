"""Offline integration with actual ledger, contracts, and order-status vocabulary.

No venue calls, credentials, automatic recovery, or portfolio mutation.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from std0_quant.execution.contracts import OrderEvent, OrderIntent
from std0_quant.execution.durable_intent_ledger_v1 import (
    DurableIntentLedgerV1 as Ledger, LedgerCheckpointV1, LedgerUnavailableError,
)
from std0_quant.execution.order_state import OrderStateMachine, OrderStatus
import std0_quant.execution.execution_reconciliation_v1 as module
from std0_quant.execution.execution_reconciliation_v1 import (
    ExecutionLedgerSnapshotV1, ExecutionReconciliationReportV1,
    ReconciliationEventV1, ReconciliationInputError, ReconciliationIntentV1,
    capture_execution_ledger_snapshot_v1 as capture, reconcile_execution_v1 as reconcile,
    execution_reconciliation_report_v1_hash as report_hash,
    verify_execution_reconciliation_report_v1 as verify,
    OBSERVED_TERMINAL, OBSERVATIONS_CONSISTENT, UNRESOLVED, CONFLICT, EMPTY_SCOPE,
)

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-local ledger v1")


def intent(**changes):
    values = dict(intent_id="i-1", condition_id="m1", outcome="Up", side="BUY", qty=10,
                  limit_price=0.5, time_in_force="GTC", decision_ts_ms=1001,
                  market_data_ts_ms=1000, strategy_id="alpha-a", strategy_version="1",
                  risk_policy_version="risk-v1")
    return OrderIntent(**{**values, **changes})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline reconciliation tests must not access the network")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def factory(tmp_path):
    handles = []
    def build(values=None, ledger_id="offline-ledger"):
        values = (intent(),) if values is None else values
        path = tmp_path / f"ledger-{len(handles)}.ndjson"
        ledger = Ledger.create(path, ledger_id=ledger_id)
        handles.append(ledger)
        for value in values:
            ledger.append(value)
        checkpoint = ledger.checkpoint()
        return SimpleNamespace(path=path, ledger=ledger, checkpoint=checkpoint,
                               snapshot=capture(ledger, checkpoint=checkpoint))
    yield build
    for ledger in handles:
        ledger.close()


def events_for(entry, steps=("SUBMITTED", "VENUE_ACK", "FILLED"), *, venue_id=None):
    output = []
    total = Fraction(0)
    qty = Fraction(str(entry.intent.qty))
    venue_id = venue_id or f"offline:order-{entry.receipt.sequence}"
    for index, step in enumerate(steps):
        kind = step if isinstance(step, str) else step[0]
        fill = Fraction(0)
        if kind in {"PARTIAL_FILL", "FILLED"}:
            fill = qty - total if isinstance(step, str) else Fraction(str(step[1]))
        total += fill
        event = OrderEvent(
            event_id=f"{entry.intent.intent_id}:e-{index}", intent_id=entry.intent.intent_id,
            event_type=kind, receive_ts_ms=1002 + index,
            venue_order_id=None if kind in {"SUBMITTED", "REJECTED"} else venue_id,
            venue_ts_ms=None if kind in {"SUBMITTED", "CANCEL_REQUESTED"} else 500 + index,
            fill_qty=float(fill), fill_price=entry.intent.limit_price if fill else None,
            cumulative_filled_qty=float(total), remaining_qty=float(qty - total),
        )
        output.append(ReconciliationEventV1(entry.receipt, event))
    return tuple(output)


def run(case, observations=(), *, cutoff=2000, run_id="reconcile-1"):
    return reconcile(case.snapshot, observations, checkpoint=case.checkpoint,
                     as_of_receive_ts_ms=cutoff, reconciliation_run_id=run_id)


def assert_verified(case, observations, result, cutoff=2000):
    assert report_hash(result) == result.artifact_hash
    assert verify(result, case.snapshot, observations, checkpoint=case.checkpoint,
                  as_of_receive_ts_ms=cutoff) == ()
    assert result.resume_authorized is False
    assert result.reservation_release_authorized is False
    assert result.redispatch_authorized is False
    assert result.mode == "OFFLINE" and result.observation_scope == "SUPPLIED_EVENTS_ONLY"


@pytest.mark.parametrize("side", ("BUY", "SELL"))
@pytest.mark.parametrize("steps,terminal,filled,remaining", (
    (("SUBMITTED", "VENUE_ACK", "FILLED"), "FILLED", "10", "0"),
    (("SUBMITTED", "REJECTED"), "REJECTED", "0", "10"),
    (("SUBMITTED", "VENUE_ACK", "EXPIRED"), "EXPIRED", "0", "10"),
    (("SUBMITTED", "VENUE_ACK", ("PARTIAL_FILL", 3), "EXPIRED"), "EXPIRED", "3", "7"),
    (("SUBMITTED", "VENUE_ACK", "CANCEL_REQUESTED", "CANCELLED"), "CANCELLED", "0", "10"),
    (("SUBMITTED", "VENUE_ACK", "CANCEL_REQUESTED", ("PARTIAL_FILL", 3), "CANCELLED"),
     "CANCELLED", "3", "7"),
    (("SUBMITTED", "VENUE_ACK", ("PARTIAL_FILL", 3), "CANCEL_REQUESTED", "FILLED"),
     "FILLED", "10", "0"),
))
def test_valid_terminal_observations_never_authorize_actions(factory, side, steps, terminal, filled, remaining):
    case = factory((intent(side=side),))
    observations = events_for(case.snapshot.entries[0], steps)
    result = run(case, observations)
    assert result.status == OBSERVATIONS_CONSISTENT and result.reasons == ()
    row = result.rows[0]
    assert row.status == OBSERVED_TERMINAL and row.order_status == terminal
    assert row.filled_qty_decimal == filled and row.remaining_qty_decimal == remaining
    assert result.observed_terminal_intent_ids == ("i-1",)
    assert not result.unresolved_intent_ids and not result.conflicted_intent_ids
    assert result.input_event_count == result.distinct_event_id_count == len(observations)
    assert result.duplicate_event_count == 0
    assert_verified(case, observations, result)


@pytest.mark.parametrize("steps,status", (
    (("SUBMITTED",), "SENT"),
    (("SUBMITTED", "VENUE_ACK"), "ACKNOWLEDGED"),
    (("SUBMITTED", "VENUE_ACK", ("PARTIAL_FILL", 3)), "PARTIALLY_FILLED"),
    (("SUBMITTED", "VENUE_ACK", "CANCEL_REQUESTED"), "CANCEL_REQUESTED"),
    (("SUBMITTED", "VENUE_ACK", "CANCEL_REQUESTED", ("PARTIAL_FILL", 3)), "CANCEL_REQUESTED"),
))
def test_nonterminal_orders_stay_unresolved(factory, steps, status):
    case = factory()
    observations = events_for(case.snapshot.entries[0], steps)
    result = run(case, observations)
    assert result.status == UNRESOLVED and result.rows[0].status == UNRESOLVED
    assert result.rows[0].order_status == status
    assert "ORDER_NOT_TERMINAL" in result.rows[0].reasons
    assert result.unresolved_intent_ids == ("i-1",)
    assert_verified(case, observations, result)


def test_no_events_cannot_prove_not_sent_or_release_resources(factory):
    case = factory()
    result = run(case)
    assert result.status == UNRESOLVED
    row = result.rows[0]
    assert row.reasons == ("NO_EXECUTION_EVENTS",)
    assert row.order_status is row.filled_qty_decimal is row.remaining_qty_decimal is None
    assert_verified(case, (), result)


def test_empty_scope_is_explicit_not_vacuous_success(factory):
    case = factory(())
    result = run(case)
    assert result.status == EMPTY_SCOPE and result.rows == ()
    assert result.reasons == ("NO_INTENTS_IN_CHECKPOINT_SCOPE",)
    assert_verified(case, (), result)


def test_complete_universe_includes_intents_without_events(factory):
    case = factory((intent(), intent(intent_id="i-2"), intent(intent_id="i-3")))
    observations = events_for(case.snapshot.entries[0])
    result = run(case, observations)
    assert tuple(row.intent_id for row in result.rows) == ("i-1", "i-2", "i-3")
    assert result.unresolved_intent_ids == ("i-2", "i-3")
    assert result.observed_terminal_intent_ids == ("i-1",) and result.status == UNRESOLVED
    assert_verified(case, observations, result)


@pytest.mark.parametrize("steps,reason", (
    (("VENUE_ACK",), "MISSING_SUBMITTED_EVENT"),
    (("FILLED",), "MISSING_SUBMITTED_EVENT"),
    (("SUBMITTED", "FILLED"), "MISSING_ACKNOWLEDGEMENT"),
    (("SUBMITTED", "EXPIRED"), "MISSING_ACKNOWLEDGEMENT"),
    (("SUBMITTED", "VENUE_ACK", "CANCELLED"), "MISSING_CANCEL_REQUEST"),
))
def test_incomplete_history_is_not_repaired_or_declared_terminal(factory, steps, reason):
    case = factory()
    observations = events_for(case.snapshot.entries[0], steps)
    result = run(case, observations)
    assert result.status == UNRESOLVED and result.rows[0].reasons == (reason,)
    assert result.rows[0].order_status is None
    assert_verified(case, observations, result)


@pytest.mark.parametrize("index", (0, 1, 2))
def test_exact_duplicate_event_is_counted_but_never_replayed(factory, index):
    case = factory()
    original = events_for(case.snapshot.entries[0])
    observations = original + (original[index], original[index])
    clean, result = run(case, original), run(case, observations)
    assert clean.rows == result.rows and result.status == OBSERVATIONS_CONSISTENT
    assert result.input_event_count == 5 and result.distinct_event_id_count == 3
    assert result.duplicate_event_count == 2
    assert result.observations_hash != clean.observations_hash
    assert result.artifact_hash != clean.artifact_hash  # Evidence multiplicity is bound.
    assert_verified(case, observations, result)


@pytest.mark.parametrize("change", ("time", "reason", "receipt"))
def test_same_event_id_conflicting_content_blocks_even_after_terminal(factory, change):
    case = factory()
    observations = events_for(case.snapshot.entries[0])
    original = observations[-1]
    if change == "time":
        altered = replace(original, event=replace(original.event, receive_ts_ms=1009))
    elif change == "reason":
        altered = replace(original, event=replace(original.event, reason="different"))
    else:
        altered = replace(original, receipt=replace(original.receipt, record_hash="f" * 64))
    observations += (altered, altered)
    result = run(case, observations)
    assert result.status == CONFLICT and result.rows[0].status == CONFLICT
    assert "EVENT_ID_CONTENT_CONFLICT" in result.rows[0].reasons
    assert result.conflicting_event_ids == (original.event.event_id,)
    assert result.duplicate_event_count == 1
    assert_verified(case, observations, result)


def test_conflicting_event_id_marks_both_intents(factory):
    case = factory((intent(), intent(intent_id="i-2")))
    left, right = (events_for(entry) for entry in case.snapshot.entries)
    right = (replace(right[0], event=replace(right[0].event, event_id=left[0].event.event_id)),) + right[1:]
    result = run(case, left + right)
    assert result.conflicted_intent_ids == ("i-1", "i-2")
    assert result.status == CONFLICT


def test_venue_order_identifier_cannot_be_reused_across_intents(factory):
    case = factory((intent(), intent(intent_id="i-2")))
    observations = tuple(item for entry in case.snapshot.entries
                         for item in events_for(entry, venue_id="same-order"))
    result = run(case, observations)
    assert result.status == CONFLICT and result.conflicted_intent_ids == ("i-1", "i-2")
    assert all("VENUE_ORDER_ID_REUSED" in row.reasons for row in result.rows)
    assert_verified(case, observations, result)


@pytest.mark.parametrize("field,value", (
    ("ledger_id", "different"), ("sequence", 2), ("intent_id", "other"),
    ("intent_hash", "e" * 64), ("record_hash", "f" * 64),
))
def test_event_full_receipt_binding_is_required(factory, field, value):
    case = factory()
    observations = events_for(case.snapshot.entries[0])
    first = replace(observations[0], receipt=replace(observations[0].receipt, **{field: value}))
    observations = (first,) + observations[1:]
    result = run(case, observations)
    assert result.status == CONFLICT
    assert "EVENT_RECEIPT_BINDING_MISMATCH" in result.rows[0].reasons
    assert_verified(case, observations, result)


def test_orphan_event_is_not_silently_dropped(factory):
    case, other = factory(), factory((intent(intent_id="unknown"),), ledger_id="other-ledger")
    observations = events_for(case.snapshot.entries[0]) + events_for(other.snapshot.entries[0], venue_id="other-ledger-order")
    result = run(case, observations)
    assert result.status == CONFLICT
    assert result.orphan_event_ids == tuple(item.event.event_id for item in observations[3:])
    assert result.rows[0].status == OBSERVED_TERMINAL
    assert_verified(case, observations, result)


def test_snapshot_can_select_exact_historical_prefix_but_not_events_outside_it(factory):
    case = factory()
    old = case.checkpoint
    case.ledger.append(intent(intent_id="i-2"))
    before = case.path.read_bytes()
    selected = capture(case.ledger, checkpoint=old)
    assert selected == case.snapshot
    full = capture(case.ledger, checkpoint=case.ledger.checkpoint())
    outside = events_for(full.entries[1])
    result = reconcile(selected, outside, checkpoint=old, as_of_receive_ts_ms=2000,
                       reconciliation_run_id="prefix")
    assert result.status == CONFLICT and result.orphan_event_ids
    assert tuple(row.intent_id for row in result.rows) == ("i-1",)
    assert case.path.read_bytes() == before


@pytest.mark.parametrize("mutation", ("drop", "duplicate", "reorder", "intent", "receipt", "checkpoint"))
def test_snapshot_chain_and_complete_prefix_are_independently_reconstructed(factory, mutation):
    case = factory((intent(), intent(intent_id="i-2")))
    snapshot = case.snapshot
    entries = list(snapshot.entries)
    if mutation == "drop": entries.pop()
    elif mutation == "duplicate": entries[1] = entries[0]
    elif mutation == "reorder": entries.reverse()
    elif mutation == "intent": entries[1] = replace(entries[1], intent=intent(intent_id="i-2", qty=11))
    elif mutation == "receipt": entries[1] = replace(entries[1], receipt=replace(entries[1].receipt, record_hash="e" * 64))
    else: snapshot = replace(snapshot, checkpoint=replace(snapshot.checkpoint, record_hash="f" * 64))
    snapshot = replace(snapshot, entries=tuple(entries))
    with pytest.raises(ReconciliationInputError):
        reconcile(snapshot, (), checkpoint=case.checkpoint, as_of_receive_ts_ms=2000,
                  reconciliation_run_id="bad")


def test_self_consistent_alternative_history_does_not_match_external_anchor(factory):
    original = factory()
    other = factory((intent(qty=11),))
    with pytest.raises(ReconciliationInputError, match="trusted checkpoint"):
        reconcile(other.snapshot, (), checkpoint=original.checkpoint, as_of_receive_ts_ms=2000,
                  reconciliation_run_id="forged")


@pytest.mark.parametrize("variant", ("future", "hash", "ledger"))
def test_capture_rejects_wrong_trusted_checkpoint(factory, variant):
    case = factory()
    changes = {"future": {"sequence": 2}, "hash": {"record_hash": "e" * 64},
               "ledger": {"ledger_id": "elsewhere"}}[variant]
    with pytest.raises(ReconciliationInputError):
        capture(case.ledger, checkpoint=replace(case.checkpoint, **changes))


def test_capture_is_read_only_and_does_not_close_existing_handle(factory, monkeypatch):
    case = factory()
    before = case.path.read_bytes()
    def forbidden(*args):
        pytest.fail("snapshot capture must not append, fsync, write, or close")
    with monkeypatch.context() as patch:
        for name in ("append", "close"):
            patch.setattr(Ledger, name, forbidden)
        patch.setattr(os, "write", forbidden)
        patch.setattr(os, "fsync", forbidden)
        assert capture(case.ledger, checkpoint=case.checkpoint) == case.snapshot
    assert case.path.read_bytes() == before
    assert case.ledger.append(intent(intent_id="i-2")).sequence == 2


@pytest.mark.parametrize("stage", ("receipts", "intent", "validation"))
def test_concurrent_append_during_capture_is_not_adopted(factory, monkeypatch, stage):
    case = factory()
    target = {"receipts": "receipts", "intent": "get_intent", "validation": None}[stage]
    fired = False
    if target:
        original = getattr(Ledger, target)
        def changed(self, *args):
            nonlocal fired
            result = original(self, *args)
            if self is case.ledger and not fired:
                fired = True
                self.append(intent(intent_id="concurrent"))
            return result
        monkeypatch.setattr(Ledger, target, changed)
    else:
        original = module._validate_snapshot
        def changed(*args):
            original(*args)
            case.ledger.append(intent(intent_id="concurrent"))
        monkeypatch.setattr(module, "_validate_snapshot", changed)
    with pytest.raises(ReconciliationInputError, match="changed"):
        capture(case.ledger, checkpoint=case.checkpoint)
    assert case.ledger.checkpoint().sequence == 2


@pytest.mark.parametrize("variant", ("closed", "external-corruption"))
def test_unavailable_ledger_does_not_produce_a_snapshot(factory, variant):
    case = factory()
    if variant == "closed": case.ledger.close()
    else:
        with case.path.open("ab") as output: output.write(b"{")
    before = case.path.read_bytes()
    with pytest.raises(LedgerUnavailableError):
        capture(case.ledger, checkpoint=case.checkpoint)
    assert case.path.read_bytes() == before


@pytest.mark.parametrize("variant,reason", (
    ("overfill", "OVERFILL"), ("cumulative", "CUMULATIVE_FILL_MISMATCH"),
    ("remaining", "REMAINING_QUANTITY_MISMATCH"), ("price", "FILL_OUTSIDE_LIMIT"),
    ("venue-id", "VENUE_ORDER_ID_CHANGED"), ("repeated-ack", "INVALID_ORDER_TRANSITION"),
    ("after-terminal", "EVENT_AFTER_TERMINAL"), ("receive-backwards", "NONMONOTONIC_OR_PREDECISION_RECEIVE_TIME"),
    ("cutoff", "EVENT_AFTER_OBSERVATION_CUTOFF"), ("submitted-venue", "SUBMITTED_CANNOT_PROVE_VENUE_ACK"),
    ("partial-is-full", "INVALID_PARTIAL_FILL_QUANTITY"),
))
def test_conflicting_lifecycle_or_quantities_fail_closed(factory, variant, reason):
    case = factory()
    observations = events_for(case.snapshot.entries[0])
    index, changed = 2, {}
    if variant == "overfill": changed = dict(fill_qty=11, cumulative_filled_qty=11)
    elif variant == "cumulative": changed = dict(cumulative_filled_qty=9)
    elif variant == "remaining":
        index = 1; changed = dict(remaining_qty=9)
    elif variant == "price": changed = dict(fill_price=0.6)
    elif variant == "venue-id": changed = dict(venue_order_id="other-order")
    elif variant == "repeated-ack":
        observations = observations[:2] + (replace(observations[1], event=replace(observations[1].event,
                                   event_id="extra-ack", receive_ts_ms=1004)),) + observations[2:]
    elif variant == "after-terminal":
        observations += (replace(observations[-1], event=replace(observations[-1].event, event_id="new-fill")),)
    elif variant == "receive-backwards": changed = dict(receive_ts_ms=1002)
    elif variant == "cutoff": changed = dict(receive_ts_ms=2001)
    elif variant == "submitted-venue":
        index = 0; changed = dict(venue_order_id="unconfirmed")
    elif variant == "partial-is-full":
        # Keep upstream PARTIAL_FILL structurally legal; report's exact remaining check
        # cannot call a full fill partial. Missing remaining is a separate unresolved case.
        index = 2; changed = dict(event_type="PARTIAL_FILL", remaining_qty=None)
    if changed:
        observations = observations[:index] + (replace(observations[index], event=replace(
            observations[index].event, **changed)),) + observations[index + 1:]
    result = run(case, observations)
    if variant == "partial-is-full":
        assert result.status == UNRESOLVED
        assert result.rows[0].reasons == ("MISSING_REMAINING_QUANTITY",)
    else:
        assert result.status == CONFLICT and reason in result.rows[0].reasons
    assert_verified(case, observations, result)


@pytest.mark.parametrize("kind", ("SUBMITTED", "VENUE_ACK", "FILLED"))
def test_missing_remaining_quantity_is_not_assumed_zero(factory, kind):
    case = factory()
    observations = events_for(case.snapshot.entries[0])
    observations = tuple(replace(item, event=replace(item.event, remaining_qty=None))
                         if item.event.event_type.value == kind else item for item in observations)
    result = run(case, observations)
    assert result.status == UNRESOLVED
    assert result.rows[0].reasons == ("MISSING_REMAINING_QUANTITY",)


@pytest.mark.parametrize("kind", ("VENUE_ACK", "FILLED"))
def test_missing_venue_order_id_is_unresolved_not_authenticated_by_receipt(factory, kind):
    case = factory()
    observations = tuple(replace(item, event=replace(item.event, venue_order_id=None))
                         if item.event.event_type.value == kind else item
                         for item in events_for(case.snapshot.entries[0]))
    result = run(case, observations)
    assert result.status == UNRESOLVED
    assert result.rows[0].reasons == ("MISSING_VENUE_ORDER_ID",)


def test_receive_time_cutoff_is_inclusive_and_remote_clock_is_not_assumed_synced(factory):
    case = factory()
    observations = events_for(case.snapshot.entries[0])
    observations = tuple(replace(item, event=replace(item.event, venue_ts_ms=1e9 - index))
                         if index else item for index, item in enumerate(observations))
    result = run(case, observations, cutoff=1004)
    assert result.status == OBSERVATIONS_CONSISTENT
    assert_verified(case, observations, result, cutoff=1004)


def test_equal_receive_timestamps_use_supplied_order_not_event_id_sort(factory):
    case = factory()
    observations = tuple(replace(item, event=replace(item.event, receive_ts_ms=1002,
                         event_id=f"z-{3-index}")) for index, item in enumerate(events_for(case.snapshot.entries[0])))
    assert run(case, observations).status == OBSERVATIONS_CONSISTENT
    assert run(case, tuple(reversed(observations))).status == UNRESOLVED


@pytest.mark.parametrize("qty", (1e-13, 1e-300, 5e-324, 1e308))
def test_positive_extreme_quantity_is_not_erased_by_absolute_tolerance(factory, qty):
    case = factory((intent(qty=qty),))
    observations = events_for(case.snapshot.entries[0])
    result = run(case, observations)
    assert result.status == OBSERVATIONS_CONSISTENT
    assert Fraction(result.rows[0].filled_qty_decimal) == Fraction(str(float(qty)))
    assert result.rows[0].remaining_qty_decimal == "0"
    wrong = replace(observations[-1], event=replace(observations[-1].event, cumulative_filled_qty=0))
    assert run(case, observations[:-1] + (wrong,)).status == CONFLICT


def test_decimal_fractional_fills_sum_exactly_without_binary_epsilon(factory):
    case = factory((intent(qty=0.3),))
    observations = events_for(case.snapshot.entries[0], ("SUBMITTED", "VENUE_ACK", ("PARTIAL_FILL", 0.1), ("FILLED", 0.2)))
    result = run(case, observations)
    assert result.status == OBSERVATIONS_CONSISTENT
    assert result.rows[0].filled_qty_decimal == "0.3"
    assert_verified(case, observations, result)


def test_no_tolerance_turns_a_tiny_remaining_position_into_filled(factory):
    case = factory((intent(qty=0.30000000000000004),))
    observations = events_for(case.snapshot.entries[0], ("SUBMITTED", "VENUE_ACK", ("PARTIAL_FILL", 0.1)))
    last = replace(observations[-1].event, event_id="claimed-filled", event_type="FILLED",
                   receive_ts_ms=1005, fill_qty=0.2, cumulative_filled_qty=0.3, remaining_qty=0)
    result = run(case, observations + (ReconciliationEventV1(case.snapshot.entries[0].receipt, last),))
    assert result.status == CONFLICT
    assert result.rows[0].reasons == ("REMAINING_QUANTITY_MISMATCH",)


def test_transition_vocabulary_matches_frozen_state_machine_for_normal_cancel_fill_race(factory):
    case = factory()
    machine = OrderStateMachine(order_qty=10)
    machine.mark_sent(1002)
    machine.acknowledge(1003)
    machine.request_cancel(1004)
    machine.apply_fill(3, 1005)
    steps = ("SUBMITTED", "VENUE_ACK", "CANCEL_REQUESTED", ("PARTIAL_FILL", 3))
    result = run(case, events_for(case.snapshot.entries[0], steps))
    assert result.rows[0].order_status == machine.status.value == OrderStatus.CANCEL_REQUESTED.value
    assert result.status == UNRESOLVED
    machine.acknowledge_cancel(1006)
    result = run(case, events_for(case.snapshot.entries[0], steps + ("CANCELLED",)))
    assert result.rows[0].order_status == machine.status.value
    assert Fraction(result.rows[0].filled_qty_decimal) == machine.filled_qty


@pytest.mark.parametrize("cutoff", (True, -1, float("inf"), float("nan"), "2000"))
def test_bad_cutoff_is_rejected_before_report(factory, cutoff):
    with pytest.raises((TypeError, ValueError)):
        run(factory(), cutoff=cutoff)


@pytest.mark.parametrize("mode", ("LIVE", "SHADOW", "", "offline"))
def test_envelope_refuses_nonoffline_scope(factory, mode):
    case = factory()
    obs = events_for(case.snapshot.entries[0])[0]
    with pytest.raises(ValueError): replace(obs, mode=mode)


@pytest.mark.parametrize("kind", ("intent", "receipt", "event", "snapshot", "observations", "checkpoint"))
def test_wrong_input_types_are_not_duck_typed(factory, kind):
    case = factory()
    entry = case.snapshot.entries[0]
    with pytest.raises(TypeError):
        if kind == "intent": ReconciliationIntentV1({}, entry.receipt)
        elif kind == "receipt": ReconciliationIntentV1(entry.intent, object())
        elif kind == "event": ReconciliationEventV1(entry.receipt, object())
        elif kind == "snapshot": reconcile({}, (), checkpoint=case.checkpoint, as_of_receive_ts_ms=2000, reconciliation_run_id="r")
        elif kind == "observations": run(case, [])
        else: capture(case.ledger, checkpoint={})


@pytest.mark.parametrize("field,value", (("fill_qty", float("nan")), ("cumulative_filled_qty", True),
                                          ("event_id", " e "), ("venue_order_id", "")))
def test_mutated_event_is_revalidated_not_trusted(factory, field, value):
    case = factory()
    obs = deepcopy(events_for(case.snapshot.entries[0])[-1])
    object.__setattr__(obs.event, field, value)
    with pytest.raises((TypeError, ValueError)):
        run(case, (obs,))


def test_evaluation_is_deterministic_and_does_not_mutate_inputs_or_ledger(factory):
    case = factory()
    observations = events_for(case.snapshot.entries[0])
    before = (asdict(case.snapshot), tuple(asdict(item) for item in observations), case.path.read_bytes())
    first, second = run(case, observations), run(case, observations)
    assert first == second
    assert (asdict(case.snapshot), tuple(asdict(item) for item in observations), case.path.read_bytes()) == before
    assert case.ledger.checkpoint() == case.checkpoint


def test_policy_free_report_cannot_accept_allow_or_resume_flags(factory):
    case = factory()
    with pytest.raises(TypeError):
        reconcile(case.snapshot, (), checkpoint=case.checkpoint, as_of_receive_ts_ms=2000,
                  reconciliation_run_id="r", resume_authorized=True)
    report = run(case)
    for field in ("resume_authorized", "reservation_release_authorized", "redispatch_authorized"):
        with pytest.raises(ValueError): replace(report, **{field: True})


def test_run_id_exclusion_and_independent_reference_hash(factory):
    case = factory()
    observations = events_for(case.snapshot.entries[0])
    first, second = run(case, observations, run_id="r1"), run(case, observations, run_id="r2")
    assert first.artifact_hash == second.artifact_hash
    payload = asdict(first)
    del payload["artifact_hash"]
    del payload["reconciliation_run_id"]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    expected = hashlib.sha256(b"std0-quant/execution-reconciliation/v1/report\n" + raw).hexdigest()
    assert first.artifact_hash == expected
    assert_verified(case, observations, first)
    assert_verified(case, observations, second)


@pytest.mark.parametrize("field", ("status", "rows", "unresolved_intent_ids", "snapshot_hash", "observations_hash",
                                    "input_event_count", "as_of_receive_ts_ms"))
def test_recomputed_hash_cannot_hide_forged_report_semantics(factory, field):
    case = factory()
    original = run(case)
    changes = {
        "status": OBSERVATIONS_CONSISTENT, "rows": (), "unresolved_intent_ids": (),
        "snapshot_hash": "e" * 64, "observations_hash": "f" * 64,
        "input_event_count": 9, "as_of_receive_ts_ms": 10000.0,
    }
    forged = replace(original, **{field: changes[field]})
    forged = replace(forged, artifact_hash=report_hash(forged))
    reasons = verify(forged, case.snapshot, (), checkpoint=case.checkpoint, as_of_receive_ts_ms=2000)
    assert reasons == ("EXECUTION_RECONCILIATION_REPORT_V1_SEMANTICS_MISMATCH",)


def test_hash_tampering_and_object_level_authorization_forgery_are_detected(factory):
    case = factory()
    original = run(case)
    tampered = replace(original, artifact_hash="f" * 64)
    assert verify(tampered, case.snapshot, (), checkpoint=case.checkpoint, as_of_receive_ts_ms=2000) == (
        "EXECUTION_RECONCILIATION_REPORT_V1_HASH_MISMATCH", "EXECUTION_RECONCILIATION_REPORT_V1_SEMANTICS_MISMATCH")
    forged = deepcopy(original)
    object.__setattr__(forged, "resume_authorized", True)
    object.__setattr__(forged, "artifact_hash", report_hash(forged))
    assert verify(forged, case.snapshot, (), checkpoint=case.checkpoint, as_of_receive_ts_ms=2000) == (
        "EXECUTION_RECONCILIATION_REPORT_V1_SEMANTICS_MISMATCH",)


def test_report_does_not_verify_against_changed_observations(factory):
    case = factory()
    observations = events_for(case.snapshot.entries[0])
    report = run(case, observations)
    assert verify(report, case.snapshot, observations[:-1], checkpoint=case.checkpoint,
                  as_of_receive_ts_ms=2000) == ("EXECUTION_RECONCILIATION_REPORT_V1_SEMANTICS_MISMATCH",)


def test_cross_process_recovery_and_reconciliation_are_offline_and_do_not_rewrite_ledger(factory):
    case = factory()
    case.ledger.close()
    before = case.path.read_bytes()
    root = str(Path(module.__file__).resolve().parents[2])
    code = """
import json, socket, sys
from std0_quant.execution.durable_intent_ledger_v1 import DurableIntentLedgerV1, LedgerCheckpointV1
from std0_quant.execution.execution_reconciliation_v1 import capture_execution_ledger_snapshot_v1, reconcile_execution_v1
def forbidden(*args, **kwargs):
    raise AssertionError('network forbidden')
socket.socket = socket.create_connection = forbidden
checkpoint = LedgerCheckpointV1(**json.loads(sys.argv[2]))
with DurableIntentLedgerV1.open_existing(sys.argv[1], ledger_id=checkpoint.ledger_id, checkpoint=checkpoint) as ledger:
    snapshot = capture_execution_ledger_snapshot_v1(ledger, checkpoint=checkpoint)
    result = reconcile_execution_v1(snapshot, (), checkpoint=checkpoint, as_of_receive_ts_ms=2000, reconciliation_run_id='child')
    assert result.status == 'UNRESOLVED' and result.resume_authorized is False
    print(result.artifact_hash)
"""
    env = {**os.environ, "PYTHONPATH": root, "PYTHONDONTWRITEBYTECODE": "1"}
    child = subprocess.run([sys.executable, "-B", "-c", code, str(case.path), json.dumps(asdict(case.checkpoint))],
                           env=env, capture_output=True, text=True, timeout=20)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == run(case).artifact_hash
    assert case.path.read_bytes() == before


def test_json_report_is_canonical_and_detached_from_internal_rows(factory):
    report = run(factory())
    payload = report.to_dict()
    assert json.loads(report.to_json())["resume_authorized"] is False
    assert report.to_json() == json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False)
    payload["checkpoint"]["record_hash"] = "f" * 64
    assert report.checkpoint.record_hash != "f" * 64


@pytest.mark.parametrize("field,value", (("filled_qty_decimal", "999"), ("remaining_qty_decimal", "0"),
                                          ("order_status", "FILLED"), ("status", OBSERVED_TERMINAL)))
def test_forged_individual_row_is_rejected_even_with_new_report_hash(factory, field, value):
    case = factory()
    report = run(case)
    forged_row = replace(report.rows[0], **{field: value})
    forged = replace(report, rows=(forged_row,))
    forged = replace(forged, artifact_hash=report_hash(forged))
    assert verify(forged, case.snapshot, (), checkpoint=case.checkpoint, as_of_receive_ts_ms=2000) == (
        "EXECUTION_RECONCILIATION_REPORT_V1_SEMANTICS_MISMATCH",)


def test_size_guardrails_are_explicit_before_evaluation(factory, monkeypatch):
    case = factory()
    observations = events_for(case.snapshot.entries[0])
    monkeypatch.setattr(module, "MAX_EVENTS", 2)
    with pytest.raises(ValueError, match="MAX_EVENTS"):
        run(case, observations)
    monkeypatch.setattr(module, "MAX_INTENTS", 0)
    with pytest.raises(ValueError, match="MAX_INTENTS"):
        run(case)


def test_empty_checkpoint_can_be_scoped_with_longer_history_but_is_not_full_account_evidence(factory):
    case = factory(())
    header = case.checkpoint
    case.ledger.append(intent())
    snapshot = capture(case.ledger, checkpoint=header)
    assert snapshot.entries == ()
    result = reconcile(snapshot, (), checkpoint=header, as_of_receive_ts_ms=2000,
                       reconciliation_run_id="header-only")
    assert result.status == EMPTY_SCOPE and result.resume_authorized is False


def test_sell_fill_below_limit_is_conflicting(factory):
    case = factory((intent(side="SELL"),))
    observations = events_for(case.snapshot.entries[0])
    observations = observations[:-1] + (replace(observations[-1], event=replace(
        observations[-1].event, fill_price=0.49)),)
    result = run(case, observations)
    assert result.status == CONFLICT
    assert result.rows[0].reasons == ("FILL_OUTSIDE_LIMIT",)


def test_intent_after_cutoff_remains_explicit_conflict_even_without_events(factory):
    result = run(factory(), cutoff=1000)
    assert result.status == CONFLICT
    assert result.rows[0].reasons == ("INTENT_AFTER_OBSERVATION_CUTOFF",)


def test_independent_order_interleaving_is_preserved_in_evidence_hash(factory):
    case = factory((intent(), intent(intent_id="i-2")))
    one, two = (events_for(entry) for entry in case.snapshot.entries)
    interleaved = tuple(item for pair in zip(one, two) for item in pair)
    first, second = run(case, one + two), run(case, interleaved)
    assert first.status == second.status == OBSERVATIONS_CONSISTENT
    assert first.rows == second.rows
    assert first.observations_hash != second.observations_hash
    assert_verified(case, interleaved, second)
