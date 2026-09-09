"""Offline real-ledger integration; real governance verifier and risk gate.

No key material or venue calls. Built-in fault stubs do not certify a LIVE
adapter, production account isolation, power-loss safety or reconciliation.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import errno
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from std0_quant.execution.contracts import OrderEvent, OrderEventType, OrderIntent
from std0_quant.execution.durable_intent_ledger_v1 import (
    DurableIntentLedgerV1, LedgerBusyError, LedgerCheckpointV1, LedgerCorruptionError,
)
from std0_quant.execution.execution_validation import ExecutionValidationTarget
from std0_quant.execution.portfolio import PortfolioState, Position
from std0_quant.execution.production_eligibility_gate_v2 import (
    ELIGIBLE, evaluate_production_eligibility_v2, production_eligibility_decision_v2_hash,
)
from std0_quant.execution.risk import RiskLimits
from std0_quant.research.factors.contracts import ValidationStatus
import std0_quant.execution.exclusive_submit_path_v1 as module
from std0_quant.execution.exclusive_submit_path_v1 import (
    ExclusiveSubmitPathV1 as SubmitPath, ExclusiveSubmitPolicyV1,
    OfflineSubmitStubV1, SubmitGovernanceEvidenceV1, SubmitIntentConflictError,
    SubmitPathUnavailableError, exclusive_submit_policy_v1_hash,
)

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-local ledger v1")

# Reuse the reviewed real evidence constructors, not mocks of verifiers.
_fixture_file = Path(__file__).with_name("test_production_eligibility_gate_v2.py")
_spec = importlib.util.spec_from_file_location("_exclusive_submit_gate_fixtures_v1", _fixture_file)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


def evidence_for(status=ValidationStatus.PASS):
    record = _fixtures.validated_record()
    policy = _fixtures.policy_v2()
    decision = _fixtures.decision_v2(
        p=policy, validation_status=status,
        reasons=() if status == ValidationStatus.PASS else ("TEST_EXECUTION_BLOCKED",),
    )
    bridge = _fixtures.evaluate_execution_governance_bridge_v2(
        record, decision, policy, bridge_run_id="offline-bridge")
    gate = evaluate_production_eligibility_v2(record, bridge, decision, policy,
                                             gate_run_id="offline-gate")
    return SubmitGovernanceEvidenceV1(record, bridge, decision, policy, gate)


def intent(**changes):
    values = dict(intent_id="intent-1", condition_id="m1", outcome="Up", side="BUY",
                  qty=10, limit_price=0.5, time_in_force="GTC", decision_ts_ms=1001,
                  market_data_ts_ms=1000, strategy_id="alpha-a", strategy_version="1",
                  risk_policy_version="risk-v1")
    return OrderIntent(**{**values, **changes})


def policy_for(evidence, **changes):
    values = dict(policy_id="offline-submit", version="1", target=evidence.eligibility.target,
                  eligibility_artifact_hash=evidence.eligibility.artifact_hash,
                  limits=RiskLimits(100, 100, 200, 10, 1000), max_decision_age_ms=1000,
                  fee_reserve_bps=100)
    return ExclusiveSubmitPolicyV1(**{**values, **changes})


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("exclusive submit tests must not access the network")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def evidence():
    return evidence_for()


@pytest.fixture
def factory(tmp_path, evidence):
    handles = []
    def build(*, portfolio=None, policy=None, scenario="ACK", clock=None, stub=None, **policy_changes):
        clock = clock or SimpleNamespace(now=1002.0)
        time_fn = clock if callable(clock) else lambda: clock.now
        stub = stub or OfflineSubmitStubV1(scenario)
        path = tmp_path / f"ledger-{len(handles)}.ndjson"
        original = portfolio if portfolio is not None else PortfolioState(cash=100.0)
        policy = policy or policy_for(evidence, **policy_changes)
        route = SubmitPath.create(path, ledger_id="offline-ledger", portfolio=original,
                                  policy=policy, stub=stub, clock_ms=time_fn)
        handles.append(route)
        return SimpleNamespace(route=route, stub=stub, path=path, clock=clock,
                               original=original, policy=policy, evidence=evidence)
    yield build
    for handle in handles:
        handle.close()


def submit(case, value=None, evidence=None):
    return case.route.submit(value or intent(), evidence=evidence or case.evidence)


def persisted_count(case):
    return len(case.path.read_bytes().splitlines()) - 1


def test_ack_binds_evidence_and_receipt_without_filling_or_mutating_input(factory):
    case = factory()
    before = asdict(case.original)
    result = submit(case)
    assert result.status == "OFFLINE_ACK" and result.adapter_called
    assert result.risk.allowed
    assert result.receipt.intent_hash == result.intent_hash
    assert result.submit_policy_hash == exclusive_submit_policy_v1_hash(case.policy)
    assert result.eligibility_artifact_hash == case.evidence.eligibility.artifact_hash
    assert result.event.event_type == OrderEventType.VENUE_ACK
    assert result.event.fill_qty == 0
    assert result.reserved_cash == pytest.approx(5.05)
    snapshot = case.route.portfolio_snapshot()
    assert snapshot.cash == 100 and snapshot.positions == {}
    assert snapshot.reserved_cash == pytest.approx(5.05)
    assert asdict(case.original) == before
    assert persisted_count(case) == len(case.stub.requests) == 1
    request = case.stub.requests[0]
    assert request.mode == result.mode == "OFFLINE"
    assert request.receipt == result.receipt
    assert OrderIntent.from_json(request.intent_json) == intent()


def test_real_fsync_precedes_dispatch_and_reservation_is_visible(factory, monkeypatch):
    case = factory()
    events = []
    real_sync, real_dispatch = os.fsync, OfflineSubmitStubV1._dispatch
    def sync(fd):
        value = real_sync(fd)
        if fd == case.route._ledger._fd:
            events.append("durable-file-sync")
        return value
    def dispatch(self, token, request):
        events.append("dispatch")
        assert case.route.portfolio_snapshot().reserved_cash == pytest.approx(5.05)
        row = json.loads(case.path.read_bytes().splitlines()[-1])
        assert row["intent_hash"] == request.intent_hash
        assert row["record_hash"] == request.receipt.record_hash
        return real_dispatch(self, token, request)
    monkeypatch.setattr(module.os, "fsync", sync)
    monkeypatch.setattr(OfflineSubmitStubV1, "_dispatch", dispatch)
    assert submit(case).status == "OFFLINE_ACK"
    assert events == ["durable-file-sync", "dispatch"]


def test_risk_gate_is_invoked_internally_with_current_owned_state(factory, monkeypatch):
    case = factory()
    original = module.evaluate_order_risk
    calls = []
    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(module, "evaluate_order_risk", spy)
    submit(case)
    submit(case, intent(intent_id="second"))
    assert len(calls) == 2
    assert calls[1]["portfolio"].reserved_cash == pytest.approx(5.05)
    assert calls[0]["intent"].qty == intent().qty
    assert calls[0]["context"].estimated_fee_cost == pytest.approx(0.05)


@pytest.mark.parametrize("field,value,reason", (
    ("strategy_id", "other-alpha", "INTENT_STRATEGY_IDENTITY_MISMATCH"),
    ("strategy_version", "2", "INTENT_STRATEGY_IDENTITY_MISMATCH"),
    ("risk_policy_version", "other-risk", "INTENT_RISK_POLICY_MISMATCH"),
))
def test_intent_provenance_mismatch_blocks_before_persistence(factory, field, value, reason):
    case = factory()
    before = case.path.read_bytes()
    result = submit(case, intent(**{field: value}))
    assert result.status == "BLOCKED" and reason in result.reasons
    assert not result.adapter_called
    assert case.path.read_bytes() == before
    assert case.route.portfolio_snapshot().reserved_cash == 0
    assert not case.stub.requests


@pytest.mark.parametrize("status", (ValidationStatus.FAIL, ValidationStatus.PENDING))
def test_real_noneligible_governance_is_blocked_even_with_pinned_hash(factory, status):
    supplied = evidence_for(status)
    case = factory(policy=policy_for(supplied))
    result = submit(case, evidence=supplied)
    assert "PRODUCTION_ELIGIBILITY_NOT_ELIGIBLE" in result.reasons
    assert result.status == "BLOCKED"
    assert persisted_count(case) == len(case.stub.requests) == 0


def test_forged_eligible_with_recomputed_and_pinned_hash_is_rejected(factory):
    supplied = evidence_for(ValidationStatus.FAIL)
    forged = replace(supplied.eligibility, status=ELIGIBLE, reasons=(), artifact_hash="0" * 64)
    forged = replace(forged, artifact_hash=production_eligibility_decision_v2_hash(forged))
    supplied = replace(supplied, eligibility=forged)
    case = factory(policy=policy_for(supplied))
    result = submit(case, evidence=supplied)
    assert "PRODUCTION_ELIGIBILITY_DECISION_V2_SEMANTICS_MISMATCH" in result.reasons
    assert "ELIGIBILITY_ANCHOR_MISMATCH" not in result.reasons
    assert persisted_count(case) == len(case.stub.requests) == 0


@pytest.mark.parametrize("part", ("decision", "bridge", "eligibility"))
def test_tampered_evidence_hashes_do_not_reach_stub(factory, evidence, part):
    forged = replace(getattr(evidence, part), artifact_hash="f" * 64)
    case = factory()
    result = submit(case, evidence=replace(evidence, **{part: forged}))
    assert result.status == "BLOCKED"
    assert any("MISMATCH" in reason for reason in result.reasons)
    assert persisted_count(case) == len(case.stub.requests) == 0


def test_policy_requires_independently_pinned_eligibility(factory):
    case = factory(eligibility_artifact_hash="e" * 64)
    result = submit(case)
    assert "ELIGIBILITY_ANCHOR_MISMATCH" in result.reasons
    assert persisted_count(case) == len(case.stub.requests) == 0


def test_trusted_target_identity_is_not_inferred_from_an_unrelated_gate(factory, evidence):
    target = replace(evidence.eligibility.target, factor_id="different-factor")
    case = factory(target=target)
    result = submit(case)
    assert "GOVERNANCE_TARGET_MISMATCH" in result.reasons
    assert not case.stub.requests


@pytest.mark.parametrize("variant,reason", (
    ("order", "MAX_ORDER_NOTIONAL_EXCEEDED"),
    ("cash", "INSUFFICIENT_AVAILABLE_CASH"),
    ("fees", "INSUFFICIENT_AVAILABLE_CASH"),
    ("loss", "MAX_DAILY_LOSS_REACHED"),
    ("position", "INSUFFICIENT_AVAILABLE_POSITION"),
))
def test_real_risk_rejections_have_no_reservation_or_journal_side_effect(factory, evidence, variant, reason):
    portfolio = PortfolioState(cash=100)
    limits = policy_for(evidence).limits
    proposed = intent()
    if variant == "order": limits = replace(limits, max_order_notional=4)
    elif variant == "cash": portfolio.cash = 4
    elif variant == "fees": portfolio.cash = 5
    elif variant == "loss": portfolio.realized_pnl = -11
    else: proposed = intent(side="SELL")
    case = factory(portfolio=portfolio, limits=limits)
    before = (case.path.read_bytes(), asdict(case.route.portfolio_snapshot()))
    result = submit(case, proposed)
    assert result.status == "BLOCKED" and reason in result.reasons
    assert (case.path.read_bytes(), asdict(case.route.portfolio_snapshot())) == before
    assert not case.stub.requests


@pytest.mark.parametrize("scope", ("market", "gross"))
def test_unfilled_buy_orders_count_towards_pending_exposure_caps(factory, scope):
    limits = RiskLimits(100, 8 if scope == "market" else 100,
                        8 if scope == "gross" else 200, 10, 1000)
    case = factory(limits=limits, portfolio=PortfolioState(cash=1000))
    assert submit(case).status == "OFFLINE_ACK"
    second = intent(intent_id="second", condition_id="m2" if scope == "gross" else "m1")
    result = submit(case, second)
    assert result.status == "BLOCKED"
    assert f"PENDING_MAX_{scope.upper()}_EXPOSURE_EXCEEDED" in result.reasons
    assert persisted_count(case) == len(case.stub.requests) == 1


def test_sell_inventory_is_reserved_without_changing_position_or_crediting_cash(factory):
    portfolio = PortfolioState(cash=100, positions={('m1', 'Up'): Position('m1', 'Up', 10, 4)})
    case = factory(portfolio=portfolio)
    first = submit(case, intent(side="SELL", qty=8))
    assert first.status == "OFFLINE_ACK" and first.reserved_sell_qty == 8
    second = submit(case, intent(intent_id="second", side="SELL", qty=3))
    assert "INSUFFICIENT_AVAILABLE_POSITION" in second.reasons
    owned = case.route.portfolio_snapshot()
    assert owned.cash == 100 and owned.positions['m1', 'Up'].qty == 10
    assert owned.reserved_positions['m1', 'Up'] == 8
    assert not portfolio.reserved_positions
    assert len(case.stub.requests) == 1


def test_unfilled_sell_does_not_reduce_exposure_for_following_buy(factory):
    portfolio = PortfolioState(cash=100, positions={('m1', 'Up'): Position('m1', 'Up', 10, 8)})
    case = factory(portfolio=portfolio, limits=RiskLimits(100, 10, 200, 10, 1000))
    assert submit(case, intent(side="SELL", qty=10)).status == "OFFLINE_ACK"
    second = submit(case, intent(intent_id="buy", qty=6))
    assert second.status == "BLOCKED"
    assert len(case.stub.requests) == 1


@pytest.mark.parametrize("side", ("BUY", "SELL"))
def test_known_offline_rejection_releases_only_own_reservation(factory, side):
    portfolio = PortfolioState(cash=100, positions={('m1', 'Up'): Position('m1', 'Up', 10, 4)})
    case = factory(portfolio=portfolio)
    assert submit(case, intent(qty=2, side=side)).status == "OFFLINE_ACK"
    before = asdict(case.route.portfolio_snapshot())
    # Only change the exact fixture's offline scenario; no live adapter involved.
    case.stub._scenario = "REJECT"
    result = submit(case, intent(intent_id="reject", qty=2, side=side))
    assert result.status == "OFFLINE_REJECTED"
    assert result.reserved_cash == result.reserved_sell_qty == 0
    assert asdict(case.route.portfolio_snapshot()) == before
    assert persisted_count(case) == len(case.stub.requests) == 2
    again = submit(case, intent(intent_id="reject", qty=2, side=side))
    assert again.status == "DUPLICATE" and again.previous_status == "OFFLINE_REJECTED"
    assert not again.adapter_called and len(case.stub.requests) == 2


def test_duplicate_is_not_new_authorization_and_does_not_redispatch(factory):
    case = factory()
    first = submit(case)
    before = case.path.read_bytes(), asdict(case.route.portfolio_snapshot())
    case.route.set_kill_switch(True)
    repeated = submit(case)
    assert repeated.status == "DUPLICATE" and repeated.previous_status == "OFFLINE_ACK"
    assert not repeated.adapter_called and repeated.receipt == first.receipt
    assert (case.path.read_bytes(), asdict(case.route.portfolio_snapshot())) == before
    assert len(case.stub.requests) == 1


def test_consumed_id_content_conflict_never_reaches_adapter(factory):
    case = factory()
    submit(case)
    before = case.path.read_bytes()
    with pytest.raises(SubmitIntentConflictError):
        submit(case, intent(qty=11))
    assert case.path.read_bytes() == before and len(case.stub.requests) == 1


@pytest.mark.parametrize("scenario", ("TIMEOUT", "MALFORMED"))
@pytest.mark.parametrize("side", ("BUY", "SELL"))
def test_unknown_outcome_latches_all_new_dispatch_and_retains_reservations(factory, scenario, side):
    portfolio = PortfolioState(cash=100, positions={('m1', 'Up'): Position('m1', 'Up', 10, 4)})
    case = factory(scenario=scenario, portfolio=portfolio)
    result = submit(case, intent(side=side))
    assert result.status == "UNKNOWN" and result.adapter_called and result.receipt is not None
    assert result.reserved_cash > 0 if side == "BUY" else result.reserved_sell_qty == 10
    assert case.route.halt_reason
    case.route.set_kill_switch(False)
    repeated = submit(case, intent(side=side))
    assert repeated.status == "DUPLICATE" and not repeated.adapter_called
    assert submit(case, intent(intent_id="new")).status == "RECONCILIATION_REQUIRED"
    held = case.route._portfolio
    snapshot = asdict(held)
    case.route.close()
    assert asdict(held) == snapshot and len(case.stub.requests) == 1


@pytest.mark.parametrize("mutation", ("mode", "intent-hash", "record-hash", "intent-id",
                                       "cumulative-fill", "remaining", "timestamp", "order-id", "fill"))
def test_invalid_stub_response_is_unknown_not_a_success_or_safe_rejection(factory, monkeypatch, mutation):
    case = factory()
    original = OfflineSubmitStubV1._dispatch
    def forged(self, token, request):
        response = original(self, token, request)
        if mutation == "mode": return replace(response, mode="LIVE")
        if mutation == "intent-hash": return replace(response, intent_hash="f" * 64)
        if mutation == "record-hash": return replace(response, record_hash="f" * 64)
        changes = {
            "intent-id": dict(intent_id="other"),
            "cumulative-fill": dict(cumulative_filled_qty=1),
            "remaining": dict(remaining_qty=9),
            "timestamp": dict(receive_ts_ms=request.submit_ts_ms + 1),
            "order-id": dict(venue_order_id="untrusted-venue"),
            "fill": dict(event_type=OrderEventType.FILLED, fill_qty=10, fill_price=0.5,
                         cumulative_filled_qty=10, remaining_qty=0),
        }[mutation]
        return replace(response, event=replace(response.event, **changes))
    monkeypatch.setattr(OfflineSubmitStubV1, "_dispatch", forged)
    result = submit(case)
    assert result.status == "UNKNOWN" and result.reserved_cash == pytest.approx(5.05)
    assert submit(case, intent(intent_id="new")).status == "RECONCILIATION_REQUIRED"
    assert len(case.stub.requests) == 1


@pytest.mark.parametrize("fault", ("fsync", "zero-write", "partial-write", "enospc"))
def test_persistence_failure_never_dispatches_and_requires_reconciliation(factory, monkeypatch, fault):
    case = factory()
    real_write, real_sync = os.write, os.fsync
    ledger_fd = case.route._ledger._fd
    count = 0
    def write(fd, data):
        nonlocal count
        if fd != ledger_fd: return real_write(fd, data)
        count += 1
        if fault == "zero-write": return 0
        if fault == "partial-write" and count == 1: return real_write(fd, data[:11])
        raise OSError(errno.ENOSPC, "offline injected full disk")
    def sync(fd):
        if fd == ledger_fd: raise OSError(errno.EIO, "offline injected fsync failure")
        return real_sync(fd)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "fsync" if fault == "fsync" else "write",
                      sync if fault == "fsync" else write)
        result = submit(case)
    assert result.status == "PRECOMMIT_FAILED" and not result.adapter_called
    assert result.receipt is None and result.reserved_cash == pytest.approx(5.05)
    assert not case.stub.requests
    assert submit(case, intent(intent_id="new")).status == "RECONCILIATION_REQUIRED"


def test_forged_precommit_receipt_cannot_authorize_stub(factory, monkeypatch):
    case = factory()
    original = DurableIntentLedgerV1.append
    def forged(self, value):
        receipt = original(self, value)
        return replace(receipt, intent_hash="f" * 64)
    monkeypatch.setattr(DurableIntentLedgerV1, "append", forged)
    result = submit(case)
    assert result.status == "PRECOMMIT_FAILED" and not result.adapter_called
    assert not case.stub.requests and persisted_count(case) == 1


def test_out_of_path_ledger_append_halts_coordinator(factory):
    case = factory()
    case.route._ledger.append(intent(intent_id="out-of-path"))
    result = submit(case)
    assert result.status == "RECONCILIATION_REQUIRED"
    assert len(case.stub.requests) == 0 and persisted_count(case) == 1


@pytest.mark.parametrize("now,reason", ((999, "DECISION_FROM_FUTURE"),
    (2500, "STALE_DECISION"), (float("inf"), None), (True, None), (-1, None)))
def test_bad_or_stale_clock_cannot_dispatch(factory, now, reason):
    case = factory(clock=SimpleNamespace(now=now))
    if reason is None:
        with pytest.raises((TypeError, ValueError)): submit(case)
    else:
        result = submit(case)
        assert result.status == "BLOCKED" and reason in result.reasons
    assert not case.stub.requests and persisted_count(case) == 0


def test_stale_market_data_is_independent_of_decision_age(factory):
    case = factory(limits=RiskLimits(100, 100, 200, 10, 1))
    result = submit(case)
    assert "STALE_MARKET_DATA" in result.reasons
    assert not case.stub.requests


def test_active_kill_switch_blocks_without_side_effect(factory):
    case = factory()
    case.route.set_kill_switch(True)
    result = submit(case)
    assert "KILL_SWITCH_ACTIVE" in result.reasons
    assert persisted_count(case) == len(case.stub.requests) == 0
    case.route.set_kill_switch(False)
    assert submit(case).status == "OFFLINE_ACK"


@pytest.mark.parametrize("change", ("stale", "backwards", "kill"))
def test_post_fsync_checks_stop_dispatch_and_release_known_unsent_hold(factory, monkeypatch, change):
    case = factory()
    original = os.fsync
    def sync(fd):
        result = original(fd)
        if fd == case.route._ledger._fd:
            if change == "kill": case.route.set_kill_switch(True)
            else: case.clock.now = 3000 if change == "stale" else 1001
        return result
    monkeypatch.setattr(module.os, "fsync", sync)
    result = submit(case)
    assert result.status == "NOT_DISPATCHED" and result.receipt is not None
    assert not result.adapter_called and result.reserved_cash == 0
    assert case.route.portfolio_snapshot().reserved_cash == 0
    assert persisted_count(case) == 1 and not case.stub.requests
    case.route.set_kill_switch(False)
    case.clock.now = 1002
    assert submit(case).status == "DUPLICATE"
    assert not case.stub.requests


@pytest.mark.parametrize("scenario", ("ACK", "REJECT", "TIMEOUT"))
def test_nonempty_recovery_blocks_all_ids_not_just_previously_seen(factory, scenario):
    case = factory(scenario=scenario)
    submit(case)
    case.route.close()
    stub = OfflineSubmitStubV1()
    with SubmitPath.open_existing(case.path, ledger_id="offline-ledger", portfolio=PortfolioState(cash=100),
                                  policy=case.policy, stub=stub, clock_ms=lambda: 1002) as route:
        assert route.halt_reason == "RECOVERED_HISTORY_REQUIRES_RECONCILIATION"
        for identifier in ("intent-1", "totally-new"):
            result = route.submit(intent(intent_id=identifier), evidence=case.evidence)
            assert result.status == "RECONCILIATION_REQUIRED" and not result.adapter_called
        assert not stub.requests


def test_empty_existing_ledger_may_start_but_missing_or_corrupt_never_recreated(factory, tmp_path):
    case = factory()
    case.route.close()
    with SubmitPath.open_existing(case.path, ledger_id="offline-ledger", portfolio=PortfolioState(cash=100),
                                  policy=case.policy, stub=OfflineSubmitStubV1(), clock_ms=lambda: 1002) as route:
        assert route.submit(intent(), evidence=case.evidence).status == "OFFLINE_ACK"
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        SubmitPath.open_existing(missing, ledger_id="offline-ledger", portfolio=PortfolioState(cash=100),
                                 policy=case.policy, stub=OfflineSubmitStubV1())
    assert not missing.exists()
    before = case.path.read_bytes() + b"{"
    case.path.write_bytes(before)
    with pytest.raises(LedgerCorruptionError):
        SubmitPath.open_existing(case.path, ledger_id="offline-ledger", portfolio=PortfolioState(cash=100),
                                 policy=case.policy, stub=OfflineSubmitStubV1())
    assert case.path.read_bytes() == before


def test_trusted_recovery_checkpoint_is_forwarded_not_ignored(factory):
    case = factory()
    submit(case)
    checkpoint = case.route._ledger.checkpoint()
    case.route.close()
    wrong = replace(checkpoint, sequence=checkpoint.sequence + 1)
    with pytest.raises(LedgerCorruptionError):
        SubmitPath.open_existing(case.path, ledger_id="offline-ledger", portfolio=PortfolioState(cash=100),
                                 policy=case.policy, stub=OfflineSubmitStubV1(), checkpoint=wrong)


def test_same_ledger_excludes_second_coordinator(factory):
    case = factory()
    with pytest.raises(LedgerBusyError):
        SubmitPath.open_existing(case.path, ledger_id="offline-ledger", portfolio=PortfolioState(cash=100),
                                 policy=case.policy, stub=OfflineSubmitStubV1())
    assert submit(case).status == "OFFLINE_ACK"


def test_same_stub_cannot_be_owned_by_two_coordinators(factory, tmp_path):
    case = factory()
    other = tmp_path / "second-ledger"
    with pytest.raises(SubmitPathUnavailableError):
        SubmitPath.create(other, ledger_id="other", portfolio=PortfolioState(cash=100),
                          policy=case.policy, stub=case.stub)
    assert not other.exists()
    assert submit(case).status == "OFFLINE_ACK"


@pytest.mark.parametrize("same_id", (False, True))
def test_concurrent_calls_cannot_double_spend_or_double_dispatch(factory, same_id):
    case = factory(portfolio=PortfolioState(cash=1), fee_reserve_bps=0)
    def attempt(index):
        return submit(case, intent(intent_id="shared" if same_id else f"i-{index}", qty=1))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(32)))
    expected = 1 if same_id else 2
    assert sum(result.status == "OFFLINE_ACK" for result in results) == expected
    assert len(case.stub.requests) == persisted_count(case) == expected
    assert case.route.portfolio_snapshot().reserved_cash == 0.5 * expected
    assert sum(result.status == "DUPLICATE" for result in results) == (31 if same_id else 0)


def test_reentrant_submit_is_rejected_without_deadlock_or_redispatch(factory, monkeypatch):
    case = factory()
    original = OfflineSubmitStubV1._dispatch
    def reentrant(self, token, request):
        with pytest.raises(SubmitPathUnavailableError, match="reentrant"):
            submit(case, intent(intent_id="nested"))
        return original(self, token, request)
    monkeypatch.setattr(OfflineSubmitStubV1, "_dispatch", reentrant)
    assert submit(case).status == "OFFLINE_ACK"
    assert len(case.stub.requests) == 1


def test_unexpected_interruption_after_precommit_keeps_hold_and_latches(factory, monkeypatch):
    case = factory()
    def interrupted(*args): raise KeyboardInterrupt()
    monkeypatch.setattr(OfflineSubmitStubV1, "_dispatch", interrupted)
    with pytest.raises(KeyboardInterrupt): submit(case)
    assert case.route.portfolio_snapshot().reserved_cash == pytest.approx(5.05)
    assert persisted_count(case) == 1
    assert submit(case, intent(intent_id="next")).status == "RECONCILIATION_REQUIRED"


@pytest.mark.parametrize("value", (None, {}, "intent", 123))
def test_wrong_intent_types_do_not_modify_state(factory, value):
    case = factory()
    with pytest.raises(TypeError):
        case.route.submit(value, evidence=case.evidence)
    assert persisted_count(case) == len(case.stub.requests) == 0


def test_wrong_evidence_and_caller_allow_flag_are_not_accepted(factory):
    case = factory()
    with pytest.raises(TypeError): case.route.submit(intent(), evidence=True)
    with pytest.raises(TypeError): case.route.submit(intent(), evidence=case.evidence, allowed=True)
    assert not case.stub.requests


@pytest.mark.parametrize("mode", ("LIVE", "SHADOW", "", "live"))
def test_nonoffline_modes_are_rejected(evidence, mode):
    with pytest.raises(ValueError): policy_for(evidence, mode=mode)


@pytest.mark.parametrize("field,value", (("max_decision_age_ms", True), ("max_decision_age_ms", -1),
    ("fee_reserve_bps", float("nan")), ("fee_reserve_bps", -1),
    ("eligibility_artifact_hash", "invalid"), ("schema_version", "v2")))
def test_invalid_policy_rejected(evidence, field, value):
    with pytest.raises((TypeError, ValueError)):
        policy_for(evidence, **{field: value})


def test_arbitrary_transport_and_stub_subclass_are_rejected_before_file_creation(tmp_path, evidence):
    class ForeignTransport:
        def submit(self, *args): raise AssertionError("must never run")
    class StubSubclass(OfflineSubmitStubV1): pass
    for index, stub in enumerate((ForeignTransport(), StubSubclass())):
        path = tmp_path / str(index)
        with pytest.raises(TypeError, match="exact built-in"):
            SubmitPath.create(path, ledger_id="offline", portfolio=PortfolioState(cash=100),
                              policy=policy_for(evidence), stub=stub)
        assert not path.exists()


@pytest.mark.parametrize("variant", ("cash-reservation", "inventory-reservation", "bad-position", "nan"))
def test_unreconciled_initial_portfolio_rejected_before_creation(tmp_path, evidence, variant):
    portfolio = PortfolioState(cash=100)
    if variant == "cash-reservation": portfolio.reserved_cash = 1
    elif variant == "inventory-reservation": portfolio.reserved_positions['m1', 'Up'] = 1
    elif variant == "bad-position": portfolio.positions['m1', 'Up'] = Position('other', 'Up', 10, 4)
    else: portfolio.cash = float("nan")
    path = tmp_path / "ledger"
    with pytest.raises(ValueError):
        SubmitPath.create(path, ledger_id="offline", portfolio=portfolio,
                          policy=policy_for(evidence), stub=OfflineSubmitStubV1())
    assert not path.exists()


def test_external_portfolio_and_snapshot_mutations_cannot_change_owned_budget(factory):
    case = factory(portfolio=PortfolioState(cash=5), fee_reserve_bps=0)
    case.original.cash = 10000
    returned = case.route.portfolio_snapshot()
    returned.cash = 10000
    assert submit(case).status == "OFFLINE_ACK"
    assert submit(case, intent(intent_id="new")).status == "BLOCKED"
    assert len(case.stub.requests) == 1


def test_closed_and_forked_handles_are_rejected(factory, monkeypatch):
    case = factory()
    pid = os.getpid()
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "getpid", lambda: pid + 1)
        with pytest.raises(SubmitPathUnavailableError, match="fork"): submit(case)
    case.route.close()
    case.route.close()
    with pytest.raises(SubmitPathUnavailableError, match="closed"): submit(case)


def test_repeated_runs_are_deterministic_with_identical_inputs(factory):
    first, second = factory(), factory()
    assert submit(first) == submit(second)
    assert first.path.read_bytes() == second.path.read_bytes()


@pytest.mark.parametrize("field,value", (("version", "2"), ("policy_id", "other"),
    ("max_decision_age_ms", 999), ("fee_reserve_bps", 0), ("eligibility_artifact_hash", "f" * 64)))
def test_policy_hash_binds_configuration(evidence, field, value):
    first = policy_for(evidence)
    changed = replace(first, **{field: value})
    assert exclusive_submit_policy_v1_hash(first) != exclusive_submit_policy_v1_hash(changed)


def test_direct_constructor_is_not_an_open_shortcut():
    with pytest.raises(TypeError, match="create"): SubmitPath()


def test_unsupported_offline_scenario_is_not_a_live_configuration():
    with pytest.raises(ValueError): OfflineSubmitStubV1("LIVE")


def test_process_exit_after_ack_cannot_cause_automatic_resend(factory):
    case = factory()
    case.route.close()
    source_root = str(Path(module.__file__).resolve().parents[2])
    code = """
import importlib.util, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('offline_submit_test_fixtures', sys.argv[2])
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
evidence = f.evidence_for()
route = f.SubmitPath.open_existing(sys.argv[1], ledger_id='offline-ledger',
    portfolio=f.PortfolioState(cash=100), policy=f.policy_for(evidence),
    stub=f.OfflineSubmitStubV1(), clock_ms=lambda: 1002)
result = route.submit(f.intent(), evidence=evidence)
assert result.status == 'OFFLINE_ACK'
print('OFFLINE_ACK_THEN_EXIT', flush=True)
os._exit(0)
"""
    env = {**os.environ, "PYTHONPATH": source_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "PYTHONDONTWRITEBYTECODE": "1"}
    child = subprocess.run([sys.executable, "-B", "-c", code, str(case.path), __file__],
                            env=env, capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "OFFLINE_ACK_THEN_EXIT"
    stub = OfflineSubmitStubV1()
    with SubmitPath.open_existing(case.path, ledger_id="offline-ledger", portfolio=PortfolioState(cash=100),
                                  policy=case.policy, stub=stub, clock_ms=lambda: 1002) as route:
        assert route.submit(intent(), evidence=case.evidence).status == "RECONCILIATION_REQUIRED"
        assert not stub.requests



def test_duplicate_keeps_original_governance_provenance_not_new_supplied_hash(factory):
    case = factory()
    first = submit(case)
    forged = replace(case.evidence, eligibility=replace(case.evidence.eligibility, artifact_hash="e" * 64))
    repeated = submit(case, evidence=forged)
    assert repeated.status == "DUPLICATE" and not repeated.adapter_called
    assert repeated.eligibility_artifact_hash == first.eligibility_artifact_hash
    assert len(case.stub.requests) == 1


def test_ledger_change_after_precommit_but_before_dispatch_stops_path(factory):
    counter = 0
    case = None
    def clock():
        nonlocal counter
        counter += 1
        if counter == 2:
            with case.path.open("ab") as output:
                output.write(b"{")
        return 1002.0
    case = factory(clock=clock)
    result = submit(case)
    assert result.status == "RECONCILIATION_REQUIRED"
    assert "LEDGER_CHANGED_BEFORE_DISPATCH_REQUIRES_RECONCILIATION" in result.reasons
    assert not result.adapter_called and not case.stub.requests
    assert result.reserved_cash == pytest.approx(5.05)


@pytest.mark.parametrize("qty,price", ((1e308, 1e308), (1e-300, 1e-300)))
def test_nonfinite_or_underflowed_order_notional_is_not_executable(factory, qty, price):
    case = factory()
    with pytest.raises(ValueError):
        submit(case, intent(qty=qty, limit_price=price))
    assert persisted_count(case) == len(case.stub.requests) == 0


@pytest.mark.parametrize("side", ("BUY", "SELL"))
@pytest.mark.parametrize("change,reason", (
    ("kill", "KILL_SWITCH_ACTIVE"),
    ("stale", "STALE_DECISION"),
    ("backwards", "CLOCK_MOVED_BACKWARDS"),
))
def test_review_final_controls_are_sampled_after_last_checkpoint(factory, monkeypatch, side, change, reason):
    portfolio = PortfolioState(cash=100, positions={('m1', 'Up'): Position('m1', 'Up', 10, 4)})
    case = factory(portfolio=portfolio)
    original = DurableIntentLedgerV1.checkpoint
    calls = 0

    def checkpoint(self):
        nonlocal calls
        result = original(self)
        if self is case.route._ledger:
            calls += 1
            if calls == 3:  # Entry, post-append receipt binding, final pre-dispatch.
                if change == "kill":
                    case.route.set_kill_switch(True)
                else:
                    case.clock.now = 3000 if change == "stale" else 1001
        return result

    monkeypatch.setattr(DurableIntentLedgerV1, "checkpoint", checkpoint)
    result = submit(case, intent(side=side))
    assert calls == 3
    assert result.status == "NOT_DISPATCHED" and reason in result.reasons
    assert not result.adapter_called and not case.stub.requests
    assert result.receipt is not None and persisted_count(case) == 1
    assert result.reserved_cash == result.reserved_sell_qty == 0
    snapshot = case.route.portfolio_snapshot()
    assert snapshot.reserved_cash == 0 and snapshot.reserved_positions == {}
    assert snapshot.cash == 100 and snapshot.positions['m1', 'Up'].qty == 10


@pytest.mark.parametrize("cash", (100.0, 5e-13))
def test_review_tiny_cash_reservation_cannot_be_silently_normalized_away(factory, cash):
    case = factory(portfolio=PortfolioState(cash=cash), fee_reserve_bps=0)
    before = asdict(case.route.portfolio_snapshot())
    with pytest.raises(SubmitPathUnavailableError, match="reservation"):
        submit(case, intent(qty=1e-13, limit_price=0.5))
    assert asdict(case.route.portfolio_snapshot()) == before
    assert not case.route._holds
    assert persisted_count(case) == 0 and not case.stub.requests
    assert case.route.halt_reason == "UNEXPECTED_FAILURE_REQUIRES_RECONCILIATION"
    assert submit(case, intent(intent_id="next")).status == "RECONCILIATION_REQUIRED"


@pytest.mark.parametrize("tiny_qty", (1e-13, 1e-14))
def test_review_rejection_cannot_erase_another_live_tiny_inventory_hold(factory, tiny_qty):
    portfolio = PortfolioState(cash=100, positions={('m1', 'Up'): Position('m1', 'Up', 10, 4)})
    case = factory(portfolio=portfolio, fee_reserve_bps=0)
    first = submit(case, intent(side="SELL", qty=tiny_qty))
    assert first.status == "OFFLINE_ACK"
    assert case.route.portfolio_snapshot().reserved_positions['m1', 'Up'] == tiny_qty
    case.stub._scenario = "REJECT"
    with pytest.raises(SubmitPathUnavailableError, match="inventory reservation"):
        submit(case, intent(intent_id="rejected", side="SELL", qty=2))
    # The rejection is known but publishing its accounting update was unsafe:
    # preserve the previous candidate state and both holds, then halt the path.
    assert case.route._holds['intent-1'].sell_qty == tiny_qty
    assert case.route._holds['rejected'].sell_qty == 2
    assert case.route.portfolio_snapshot().reserved_positions['m1', 'Up'] == 2 + tiny_qty
    assert case.route.halt_reason == "UNEXPECTED_FAILURE_REQUIRES_RECONCILIATION"
    assert len(case.stub.requests) == persisted_count(case) == 2
    assert submit(case, intent(intent_id="next")).status == "RECONCILIATION_REQUIRED"
    assert len(case.stub.requests) == 2


@pytest.mark.parametrize("side", ("BUY", "SELL"))
def test_review_ordinary_fractional_holds_still_release_only_the_rejected_order(factory, side):
    portfolio = PortfolioState(cash=100, positions={('m1', 'Up'): Position('m1', 'Up', 10, 4)})
    case = factory(portfolio=portfolio, fee_reserve_bps=0)
    for identifier, qty in (("first", 0.1), ("second", 0.2)):
        assert submit(case, intent(intent_id=identifier, side=side, qty=qty)).status == "OFFLINE_ACK"
    before = case.route.portfolio_snapshot()
    case.stub._scenario = "REJECT"
    result = submit(case, intent(intent_id="rejected", side=side, qty=0.3))
    assert result.status == "OFFLINE_REJECTED"
    after = case.route.portfolio_snapshot()
    assert after.cash == before.cash and after.positions == before.positions
    assert after.reserved_cash == pytest.approx(before.reserved_cash, rel=1e-12, abs=0)
    assert after.reserved_positions == pytest.approx(before.reserved_positions, rel=1e-12, abs=0)
    assert set(case.route._holds) == {"first", "second"}
    assert case.route.halt_reason is None
    assert len(case.stub.requests) == 3
