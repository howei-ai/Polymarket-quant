"""Real Linux-local I/O, process recovery, fault injection and frozen reconciliation.

All fixtures are synthetic OFFLINE observations; no credentials or venue calls.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from std0_quant.execution.contracts import OrderEvent, OrderIntent
from std0_quant.execution.durable_intent_ledger_v1 import DurableIntentLedgerV1 as Ledger
from std0_quant.execution.execution_reconciliation_v1 import (
    ReconciliationEventV1, capture_execution_ledger_snapshot_v1 as capture_intents,
    reconcile_execution_v1 as reconcile, verify_execution_reconciliation_report_v1 as verify_report,
)
import std0_quant.execution.durable_execution_event_journal_v1 as module
from std0_quant.execution.durable_execution_event_journal_v1 import (
    DurableExecutionEventJournalV1 as Journal, DurableExecutionEventReceiptV1,
    ExecutionEventJournalCheckpointV1 as Checkpoint, ExecutionEventJournalEntryV1 as Entry,
    ExecutionEventJournalExportV1 as Export, EventJournalBusyError as Busy,
    EventJournalCorruptionError as Corrupt, EventJournalUnavailableError as Unavailable,
    CaptureConflictError, execution_event_journal_export_v1_hash as export_hash,
    verify_execution_event_journal_export_v1 as verify_export,
    execution_events_from_journal_export_v1 as to_events,
)

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-local journal v1")
JID, LID = "offline-journal", "offline-intents"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def reference_hash(kind, payload):
    return hashlib.sha256(b"std0-quant/durable-execution-event-journal/v1/" + kind.encode() + b"\n"
                          + canonical(payload)).hexdigest()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("OFFLINE tests must not open network connections")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def case(tmp_path):
    intent_path, path = tmp_path / "intents.ndjson", tmp_path / "events.ndjson"
    intent = OrderIntent("i-1", "market-1", "Up", "BUY", 10, 0.5, "GTC", 1001, 1000, "alpha", "1", "risk-1")
    ledger = Ledger.create(intent_path, ledger_id=LID)
    receipt = ledger.append(intent)
    anchor = ledger.checkpoint()
    snapshot = capture_intents(ledger, checkpoint=anchor)
    submitted = ReconciliationEventV1(receipt, OrderEvent("e-s", "i-1", "SUBMITTED", 1002, remaining_qty=10))
    ack = ReconciliationEventV1(receipt, OrderEvent("e-a", "i-1", "VENUE_ACK", 1003,
                                                  venue_order_id="venue-1", remaining_qty=10))
    fill = ReconciliationEventV1(receipt, OrderEvent("e-f", "i-1", "FILLED", 1004, venue_order_id="venue-1",
                                                   fill_qty=10, fill_price=0.5, cumulative_filled_qty=10, remaining_qty=0))
    journal = Journal.create(path, journal_id=JID, ledger_id=LID)
    yield SimpleNamespace(path=path, intent_path=intent_path, ledger=ledger, journal=journal, snapshot=snapshot,
                          intent_checkpoint=anchor, receipt=receipt, observation=submitted,
                          observations=(submitted, ack, fill))
    journal.close()
    ledger.close()


def reopen(case, **kwargs):
    return Journal.open_existing(case.path, journal_id=JID, ledger_id=LID, **kwargs)


def add_all(case, observations=None):
    observations = case.observations if observations is None else observations
    return tuple(case.journal.append(f"capture-{index}", observation)
                 for index, observation in enumerate(observations))


def checked_export(journal):
    checkpoint = journal.checkpoint()
    result = journal.export(checkpoint=checkpoint)
    assert result.artifact_hash == export_hash(result)
    assert verify_export(result, checkpoint=checkpoint) == ()
    return checkpoint, result


def assert_poisoned(case):
    for operation in (lambda: case.journal.append("capture-0", case.observation),
                      case.journal.checkpoint, lambda: case.journal.get_capture("capture-0"),
                      lambda: case.journal.export(checkpoint=Checkpoint(JID, LID, 0, "0" * 64)),
                      case.journal.__enter__):
        with pytest.raises(Unavailable):
            operation()


def test_create_empty_header_private_mode_and_fd_properties(case):
    raw = case.path.read_bytes()
    row = json.loads(raw)
    digest = row.pop("record_hash")
    assert row == {"schema_version": module.HEADER_SCHEMA, "journal_id": JID, "ledger_id": LID, "mode": "OFFLINE"}
    assert digest == reference_hash("header", row)
    assert canonical({**row, "record_hash": digest}) + b"\n" == raw
    assert stat.S_IMODE(case.path.stat().st_mode) == 0o600
    assert not os.get_inheritable(case.journal._fd) and not os.get_inheritable(case.journal._dir_fd)
    assert module.fcntl.fcntl(case.journal._fd, module.fcntl.F_GETFL) & os.O_APPEND
    cp, export = checked_export(case.journal)
    assert cp.sequence == 0 and cp.record_hash == digest
    assert to_events(export, checkpoint=cp) == ()
    assert case.journal.get_capture("missing") is None


def test_new_capture_has_independent_reference_hash_and_full_receipt(case):
    old = case.path.read_bytes()
    receipt = case.journal.append("capture-0", case.observation)
    header_raw, raw = case.path.read_bytes().splitlines(keepends=True)
    assert header_raw == old
    row = json.loads(raw)
    record_hash = row.pop("record_hash")
    assert record_hash == reference_hash("record", row) == receipt.record_hash
    assert row["previous_hash"] == json.loads(header_raw)["record_hash"]
    assert row["observation"] == json.loads(canonical(asdict(case.observation)))
    assert row["observation_hash"] == reference_hash("observation", row["observation"]) == receipt.observation_hash
    assert receipt == DurableExecutionEventReceiptV1(JID, LID, 1, "capture-0", row["observation_hash"], record_hash)
    assert case.journal.get_capture("capture-0") == Entry(receipt, case.observation)


@pytest.mark.parametrize("method", ("create", "open_existing"))
def test_initialization_syncs_file_then_directory_before_return(tmp_path, monkeypatch, method):
    path = tmp_path / "events"
    if method == "open_existing":
        Journal.create(path, journal_id=JID, ledger_id=LID).close()
    calls, real = [], os.fsync
    def sync(fd):
        calls.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real(fd)
    monkeypatch.setattr(module.os, "fsync", sync)
    with getattr(Journal, method)(path, journal_id=JID, ledger_id=LID):
        assert calls == ["file", "directory"]


def test_append_calls_sync_before_receipt_and_readback_is_exact(case, monkeypatch):
    events, write, sync, pread = [], os.write, os.fsync, os.pread
    def traced_write(fd, raw):
        events.append("write")
        return write(fd, raw)
    def traced_read(fd, size, offset):
        events.append("readback")
        return pread(fd, size, offset)
    def traced_sync(fd):
        assert case.journal._records == []  # Not published in RAM before fsync.
        assert b'"capture_id":"capture-0"' in case.path.read_bytes()
        events.append("fsync")
        sync(fd)
    monkeypatch.setattr(module.os, "write", traced_write)
    monkeypatch.setattr(module.os, "pread", traced_read)
    monkeypatch.setattr(module.os, "fsync", traced_sync)
    receipt = case.journal.append("capture-0", case.observation)
    assert events == ["write", "readback", "fsync"] and receipt.sequence == 1


def test_existing_create_never_overwrites(case):
    add_all(case)
    before = case.path.read_bytes()
    with pytest.raises(FileExistsError):
        Journal.create(case.path, journal_id=JID, ledger_id=LID)
    assert case.path.read_bytes() == before


@pytest.mark.parametrize("method", ("create", "open_existing"))
def test_missing_parent_is_not_created(tmp_path, method):
    path = tmp_path / "not-provisioned" / "events"
    with pytest.raises(FileNotFoundError):
        getattr(Journal, method)(path, journal_id=JID, ledger_id=LID)
    assert not path.parent.exists()


def test_missing_recovery_does_not_create_file(tmp_path):
    path = tmp_path / "not-found"
    with pytest.raises(FileNotFoundError):
        Journal.open_existing(path, journal_id=JID, ledger_id=LID)
    assert not path.exists()


@pytest.mark.parametrize("invalid", ("", " padded", "padded ", 1, None, "x" * 257, "界" * 86, "\ud800"))
@pytest.mark.parametrize("field", ("journal_id", "ledger_id"))
def test_invalid_identity_rejected_before_file_creation(tmp_path, field, invalid):
    values = {"journal_id": JID, "ledger_id": LID, field: invalid}
    path = tmp_path / "events"
    with pytest.raises((TypeError, ValueError)):
        Journal.create(path, **values)
    assert not path.exists()


@pytest.mark.parametrize("invalid", ("", " padded", "padded ", False, None, "x" * 257, "\ud800"))
def test_bad_capture_id_is_nonmutating(case, invalid):
    before = case.path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        case.journal.append(invalid, case.observation)
    assert case.path.read_bytes() == before and case.journal.checkpoint().sequence == 0


def test_unicode_id_and_text_roundtrip(case):
    observation = replace(case.observation, event=replace(case.observation.event, reason="测试\n🚀"))
    case.journal.append("采集-一", observation)
    cp, first = checked_export(case.journal)
    case.journal.close()
    with reopen(case, checkpoint=cp) as journal:
        assert journal.export(checkpoint=cp) == first
        assert journal.get_capture("采集-一").observation == observation


def test_capture_retry_idempotent_no_extra_write_or_fsync(case, monkeypatch):
    expected = case.journal.append("stable", case.observation)
    before = case.path.read_bytes()
    def forbidden(*args):
        pytest.fail("idempotent retry must not rewrite or fsync")
    monkeypatch.setattr(module.os, "write", forbidden)
    monkeypatch.setattr(module.os, "fsync", forbidden)
    assert case.journal.append("stable", deepcopy(case.observation)) == expected
    assert case.journal.checkpoint().sequence == 1 and case.path.read_bytes() == before


@pytest.mark.parametrize("field,value", (
    ("event_id", "new-id"), ("receive_ts_ms", 1009), ("reason", "other"),
    ("remaining_qty", None), ("venue_order_id", "other"), ("intent_id", "another-intent"),
))
def test_capture_id_conflict_not_silently_overwritten_or_poisoned(case, field, value):
    case.journal.append("stable", case.observation)
    before = case.path.read_bytes()
    changed = replace(case.observation, event=replace(case.observation.event, **{field: value}))
    with pytest.raises(CaptureConflictError):
        case.journal.append("stable", changed)
    assert case.path.read_bytes() == before
    assert case.journal.append("different-capture", changed).sequence == 2


@pytest.mark.parametrize("field,value", (("sequence", 2), ("intent_hash", "e" * 64), ("record_hash", "f" * 64)))
def test_retry_binds_entire_intent_receipt(case, field, value):
    case.journal.append("stable", case.observation)
    changed = replace(case.observation, receipt=replace(case.receipt, **{field: value}))
    with pytest.raises(CaptureConflictError):
        case.journal.append("stable", changed)
    assert case.journal.checkpoint().sequence == 1


def test_different_capture_ids_retain_event_duplicates_for_reconciliation(case):
    add_all(case, case.observations + (case.observation, case.observation))
    cp, export = checked_export(case.journal)
    observations = to_events(export, checkpoint=cp)
    assert len(observations) == 5 and observations == case.observations + (case.observation, case.observation)
    report = reconcile(case.snapshot, observations, checkpoint=case.intent_checkpoint,
                       as_of_receive_ts_ms=2000, reconciliation_run_id="from-journal")
    assert report.status == "OBSERVATIONS_CONSISTENT" and report.duplicate_event_count == 2
    assert report.rows[0].filled_qty_decimal == "10"
    assert verify_report(report, case.snapshot, observations, checkpoint=case.intent_checkpoint,
                         as_of_receive_ts_ms=2000) == ()
    assert (report.resume_authorized, report.reservation_release_authorized, report.redispatch_authorized) == (False,) * 3


def test_conflicting_event_ids_survive_storage_and_reopen(case):
    changed = replace(case.observation, event=replace(case.observation.event, reason="conflicting-claim"))
    add_all(case, case.observations + (changed,))
    cp, export = checked_export(case.journal)
    case.journal.close()
    with reopen(case, checkpoint=cp) as journal:
        assert journal.export(checkpoint=cp) == export
        obs = to_events(export, checkpoint=cp)
        result = reconcile(case.snapshot, obs, checkpoint=case.intent_checkpoint,
                           as_of_receive_ts_ms=2000, reconciliation_run_id="conflicting")
        assert result.status == "CONFLICT" and result.conflicting_event_ids == ("e-s",)


@pytest.mark.parametrize("kind", ("receipt-hash", "intent-id", "out-of-order", "after-cutoff"))
def test_storage_is_not_lifecycle_validation_and_never_filters_claims(case, kind):
    values = case.observations
    if kind == "receipt-hash":
        values = (replace(values[0], receipt=replace(case.receipt, record_hash="f" * 64)),) + values[1:]
    elif kind == "intent-id":
        values = (replace(values[0], event=replace(values[0].event, intent_id="orphan")),) + values[1:]
    elif kind == "out-of-order":
        values = tuple(reversed(values))
    else:
        values = values[:-1] + (replace(values[-1], event=replace(values[-1].event, receive_ts_ms=9999)),)
    add_all(case, values)
    cp, export = checked_export(case.journal)
    assert to_events(export, checkpoint=cp) == values
    result = reconcile(case.snapshot, values, checkpoint=case.intent_checkpoint,
                       as_of_receive_ts_ms=2000, reconciliation_run_id="claims")
    assert result.status in {"CONFLICT", "UNRESOLVED"} and result.resume_authorized is False


def test_receipt_from_another_ledger_rejected_before_write(case):
    changed = replace(case.observation, receipt=replace(case.receipt, ledger_id="other-ledger"))
    before = case.path.read_bytes()
    with pytest.raises(ValueError, match="another intent ledger"):
        case.journal.append("c", changed)
    assert case.path.read_bytes() == before and case.journal.checkpoint().sequence == 0


@pytest.mark.parametrize("field,value", (("fill_qty", float("nan")), ("remaining_qty", True),
                                          ("event_id", " padded "), ("venue_order_id", "")))
def test_mutated_observations_revalidated(case, field, value):
    changed = deepcopy(case.observation)
    object.__setattr__(changed.event, field, value)
    before = case.path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        case.journal.append("c", changed)
    assert case.path.read_bytes() == before


@pytest.mark.parametrize("mode", ("LIVE", "SHADOW", "offline"))
def test_mutated_nonoffline_envelope_never_written(case, mode):
    changed = deepcopy(case.observation)
    object.__setattr__(changed, "mode", mode)
    with pytest.raises(ValueError):
        case.journal.append("c", changed)
    assert case.journal.checkpoint().sequence == 0


def test_input_and_returned_objects_do_not_alias_stored_evidence(case):
    value = deepcopy(case.observation)
    receipt = case.journal.append("c", value)
    object.__setattr__(value.event, "reason", "mutated-input")
    object.__setattr__(receipt, "record_hash", "e" * 64)
    one = case.journal.get_capture("c")
    object.__setattr__(one.observation.event, "reason", "mutated-output")
    cp, export = checked_export(case.journal)
    converted = to_events(export, checkpoint=cp)
    object.__setattr__(converted[0].event, "reason", "mutated-conversion")
    object.__setattr__(export.entries[0].observation.receipt, "record_hash", "f" * 64)
    two = case.journal.get_capture("c")
    assert two.observation == case.observation and two.receipt.record_hash == cp.record_hash
    assert case.journal.append("c", case.observation) == two.receipt


@pytest.mark.parametrize("sequence", (0, 1, 3))
def test_exact_export_prefix_not_implicit_latest(case, sequence):
    anchors = [case.journal.checkpoint()]
    for index, observation in enumerate(case.observations):
        case.journal.append(str(index), observation)
        anchors.append(case.journal.checkpoint())
    anchor = anchors[sequence]
    before = case.path.read_bytes()
    export = case.journal.export(checkpoint=anchor)
    assert len(export.entries) == sequence
    assert to_events(export, checkpoint=anchor) == case.observations[:sequence]
    assert verify_export(export, checkpoint=anchors[-1]) == (() if sequence == 3 else ("EVENT_JOURNAL_EXPORT_V1_PREFIX_MISMATCH",))
    assert case.path.read_bytes() == before


@pytest.mark.parametrize("field,value", (("journal_id", "other"), ("ledger_id", "other"),
                                         ("sequence", 4), ("record_hash", "f" * 64)))
def test_wrong_checkpoint_rejected_on_export_and_open(case, field, value):
    add_all(case)
    anchor = replace(case.journal.checkpoint(), **{field: value})
    with pytest.raises(Corrupt):
        case.journal.export(checkpoint=anchor)
    case.journal.close()
    before = case.path.read_bytes()
    with pytest.raises(Corrupt):
        reopen(case, checkpoint=anchor)
    assert case.path.read_bytes() == before
    with reopen(case):
        pass


@pytest.mark.parametrize("change", ("drop", "reorder", "duplicate", "observation", "capture", "receipt", "checkpoint"))
def test_export_semantic_forgery_with_recomputed_hash_is_detected(case, change):
    add_all(case)
    cp, export = checked_export(case.journal)
    entries = list(export.entries)
    if change == "drop": entries.pop()
    elif change == "reorder": entries.reverse()
    elif change == "duplicate": entries[1] = entries[0]
    elif change == "observation": entries[0] = replace(entries[0], observation=replace(
        entries[0].observation, event=replace(entries[0].observation.event, reason="forged")))
    elif change == "capture": entries[0] = replace(entries[0], receipt=replace(entries[0].receipt, capture_id="new"))
    elif change == "receipt": entries[0] = replace(entries[0], receipt=replace(entries[0].receipt, record_hash="e" * 64))
    else: export = replace(export, checkpoint=replace(cp, record_hash="f" * 64))
    forged = replace(export, entries=tuple(entries))
    forged = replace(forged, artifact_hash=export_hash(forged))
    assert verify_export(forged, checkpoint=cp) == ("EVENT_JOURNAL_EXPORT_V1_PREFIX_MISMATCH",)
    with pytest.raises(Corrupt):
        to_events(forged, checkpoint=cp)


def test_export_hash_tamper_and_json_detachment(case):
    add_all(case)
    cp, export = checked_export(case.journal)
    assert verify_export(replace(export, artifact_hash="f" * 64), checkpoint=cp) == ("EVENT_JOURNAL_EXPORT_V1_HASH_MISMATCH",)
    payload = export.to_dict()
    raw = canonical(payload)
    assert export.to_json().encode() == raw
    del payload["artifact_hash"]
    assert reference_hash("export", payload) == export.artifact_hash
    payload["checkpoint"]["record_hash"] = "f" * 64
    assert export.checkpoint.record_hash == cp.record_hash


def test_same_process_second_handle_cannot_open_or_write(case):
    with pytest.raises(Busy):
        reopen(case)
    assert case.journal.checkpoint().sequence == 0


@pytest.mark.parametrize("existing", (False, True))
def test_thread_serialization_has_one_record_per_capture(case, existing):
    def write(index):
        return case.journal.append("same" if existing else f"c-{index}", case.observation)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write, range(32)))
    count = 1 if existing else 32
    assert case.journal.checkpoint().sequence == count
    assert set(receipt.sequence for receipt in results) == set(range(1, count + 1))
    cp, export = checked_export(case.journal)
    assert len(to_events(export, checkpoint=cp)) == count


def test_short_writes_and_interrupted_syscalls_are_retried(case, monkeypatch):
    write, pread = os.write, os.pread
    wcount = rcount = 0
    def partial_write(fd, raw):
        nonlocal wcount
        wcount += 1
        if wcount == 1: raise InterruptedError()
        return write(fd, raw[:17])
    def partial_read(fd, size, offset):
        nonlocal rcount
        rcount += 1
        if rcount == 1: raise InterruptedError()
        return pread(fd, min(size, 19), offset)
    monkeypatch.setattr(module.os, "write", partial_write)
    monkeypatch.setattr(module.os, "pread", partial_read)
    assert case.journal.append("c", case.observation).sequence == 1
    assert wcount > 2 and rcount > 2
    checked_export(case.journal)


@pytest.mark.parametrize("fault", ("write-error", "short-error", "zero", "negative", "overshoot",
                                  "sync-error", "read-error", "read-eof", "read-mismatch", "interrupt"))
def test_append_faults_never_acknowledge_and_poison_all_operations(case, monkeypatch, fault):
    write, pread = os.write, os.pread
    before = case.path.read_bytes()
    def bad_write(fd, raw):
        if fault == "short-error": write(fd, raw[:23])
        if fault == "zero": return 0
        if fault == "negative": return -1
        if fault == "overshoot": return len(raw) + 1
        if fault == "interrupt": raise KeyboardInterrupt("injected")
        raise OSError(errno.ENOSPC, "injected full disk")
    def bad_sync(fd): raise OSError(errno.EIO, "injected fsync failure")
    def bad_read(fd, size, offset):
        if fault == "read-eof": return b""
        if fault == "read-mismatch": return b"x" * size
        raise OSError(errno.EIO, "injected read failure")
    with monkeypatch.context() as patch:
        if fault in {"sync-error"}: patch.setattr(module.os, "fsync", bad_sync)
        elif fault.startswith("read-"): patch.setattr(module.os, "pread", bad_read)
        else: patch.setattr(module.os, "write", bad_write)
        expected = KeyboardInterrupt if fault == "interrupt" else Unavailable
        with pytest.raises(expected):
            case.journal.append("capture-0", case.observation)
    assert_poisoned(case)
    after = case.path.read_bytes()
    case.journal.close()
    assert case.path.read_bytes() == after  # close cannot repair or remove evidence.
    if fault == "short-error":
        with pytest.raises(Corrupt): reopen(case)
    else:
        with reopen(case) as journal:
            # A complete unacknowledged write may survive. Recovery re-establishes persistence.
            complete = fault in {"sync-error", "read-error", "read-eof", "read-mismatch"}
            assert journal.checkpoint().sequence == int(complete)
            assert journal.append("capture-0", case.observation).sequence == 1
    assert after.startswith(before)


@pytest.mark.parametrize("method", ("create", "open_existing"))
@pytest.mark.parametrize("stage", ("file", "directory"))
def test_initialization_sync_failure_releases_lock_without_repair(tmp_path, monkeypatch, method, stage):
    path = tmp_path / "events"
    if method == "open_existing": Journal.create(path, journal_id=JID, ledger_id=LID).close()
    sync = os.fsync
    def bad_sync(fd):
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if directory == (stage == "directory"): raise OSError(errno.EIO, "injected")
        sync(fd)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "fsync", bad_sync)
        with pytest.raises(OSError): getattr(Journal, method)(path, journal_id=JID, ledger_id=LID)
    before = path.read_bytes()
    with Journal.open_existing(path, journal_id=JID, ledger_id=LID): pass
    assert path.read_bytes() == before


@pytest.mark.parametrize("method", ("create", "open_existing"))
@pytest.mark.parametrize("stage", ("file", "directory"))
@pytest.mark.parametrize("mutation", ("append", "truncate", "same-size"))
def test_initialization_sync_cannot_adopt_unvalidated_change(tmp_path, monkeypatch, method, stage, mutation):
    path = tmp_path / "events"
    if method == "open_existing": Journal.create(path, journal_id=JID, ledger_id=LID).close()
    sync, changed = os.fsync, []
    def injected(fd):
        sync(fd)
        if changed or stat.S_ISDIR(os.fstat(fd).st_mode) != (stage == "directory"): return
        original, info = path.read_bytes(), path.stat()
        raw = original + b"{" if mutation == "append" else original[:-1] if mutation == "truncate" else original.replace(JID.encode(), b"X" + JID.encode()[1:], 1)
        assert raw != original
        path.write_bytes(raw)
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
        changed.append(raw)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "fsync", injected)
        with pytest.raises(Corrupt): getattr(Journal, method)(path, journal_id=JID, ledger_id=LID)
    assert changed and path.read_bytes() == changed[0]


@pytest.mark.parametrize("mutation", ("append", "truncate", "same-size"))
def test_append_sync_cannot_adopt_unvalidated_change(case, monkeypatch, mutation):
    sync, changed = os.fsync, []
    def injected(fd):
        sync(fd)
        before, info = case.path.read_bytes(), case.path.stat()
        raw = before + b"{" if mutation == "append" else before[:-1] if mutation == "truncate" else before.replace(b"capture-0", b"capture-X")
        case.path.write_bytes(raw)
        os.utime(case.path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
        changed.append(raw)
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "fsync", injected)
        with pytest.raises(Unavailable): case.journal.append("capture-0", case.observation)
    assert_poisoned(case)
    assert case.path.read_bytes() == changed[0]


@pytest.mark.parametrize("mutation", ("partial-tail", "empty", "newline", "whitespace", "utf8", "duplicate-key",
    "nan", "extra-field", "missing-field", "sequence", "bool-sequence", "record-hash", "previous-hash",
    "observation-hash", "event-payload", "header-id", "header-schema", "record-schema", "mode", "missing-event-field"))
def test_recovery_rejects_corruption_without_truncation(case, mutation):
    add_all(case)
    cp = case.journal.checkpoint()
    case.journal.close()
    lines = case.path.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[-1])
    if mutation == "partial-tail": lines[-1] = lines[-1][:-1]
    elif mutation == "empty": lines = []
    elif mutation == "newline": lines.append(b"\n")
    elif mutation == "whitespace": lines[-1] = b" " + lines[-1]
    elif mutation == "utf8": lines[-1] = b"\xff\n"
    elif mutation == "duplicate-key": lines[-1] = lines[-1].replace(b'{', b'{"capture_id":"duplicate",', 1)
    elif mutation == "nan": lines[-1] = lines[-1].replace(b'"fill_qty":10.0', b'"fill_qty":NaN')
    elif mutation == "header-id": lines[0] = lines[0].replace(JID.encode(), b"other-journal")
    elif mutation == "header-schema": lines[0] = lines[0].replace(module.HEADER_SCHEMA.encode(), b"unknown")
    else:
        if mutation == "extra-field": row["extra"] = 1
        elif mutation == "missing-field": del row["capture_id"]
        elif mutation == "sequence": row["sequence"] = 2
        elif mutation == "bool-sequence": row["sequence"] = True
        elif mutation == "record-hash": row["record_hash"] = "f" * 64
        elif mutation == "previous-hash": row["previous_hash"] = "f" * 64
        elif mutation == "observation-hash": row["observation_hash"] = "f" * 64
        elif mutation == "event-payload": row["observation"]["event"]["reason"] = "changed"
        elif mutation == "record-schema": row["schema_version"] = "unknown"
        elif mutation == "mode": row["observation"]["mode"] = "LIVE"
        elif mutation == "missing-event-field": del row["observation"]["event"]["reason"]
        lines[-1] = canonical(row) + b"\n"
    raw = b"".join(lines)
    case.path.write_bytes(raw)
    with pytest.raises(Corrupt): reopen(case, checkpoint=cp)
    assert case.path.read_bytes() == raw


def test_corrupted_suffix_rejected_even_when_checkpoint_selects_earlier_prefix(case):
    first = case.journal.append("first", case.observation)
    anchor = case.journal.checkpoint()
    case.journal.append("second", case.observations[1])
    case.journal.close()
    with case.path.open("ab") as output: output.write(b"{")
    before = case.path.read_bytes()
    with pytest.raises(Corrupt): reopen(case, checkpoint=anchor)
    assert case.path.read_bytes() == before


def test_complete_suffix_loss_requires_independently_saved_later_checkpoint(case):
    old = case.journal.checkpoint()
    add_all(case)
    latest = case.journal.checkpoint()
    case.journal.close()
    case.path.write_bytes(case.path.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(Corrupt): reopen(case, checkpoint=latest)
    # Earlier/no anchor cannot detect this internally self-consistent deletion.
    with reopen(case, checkpoint=old) as journal: assert journal.checkpoint().sequence == 0
    with reopen(case) as journal: assert journal.checkpoint().sequence == 0


@pytest.mark.parametrize("violation", ("duplicate-capture", "foreign-ledger", "unnormalized"))
def test_rehashed_but_structurally_invalid_record_still_rejected(case, violation):
    case.journal.append("first", case.observation)
    case.journal.close()
    lines = case.path.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[-1])
    if violation == "duplicate-capture":
        row["sequence"] = 2
        row["previous_hash"] = row["record_hash"]
    elif violation == "foreign-ledger": row["observation"]["receipt"]["ledger_id"] = "foreign"
    else: row["observation"]["event"]["receive_ts_ms"] = 1002  # Not normalized float JSON.
    row["observation_hash"] = reference_hash("observation", row["observation"])
    del row["record_hash"]
    row["record_hash"] = reference_hash("record", row)
    if violation == "duplicate-capture": lines.append(canonical(row) + b"\n")
    else: lines[-1] = canonical(row) + b"\n"
    case.path.write_bytes(b"".join(lines))
    with pytest.raises(Corrupt): reopen(case)


@pytest.mark.parametrize("mutation", ("append", "truncate", "rewrite", "unlink", "replace", "hardlink", "symlink", "parent-move"))
def test_external_identity_or_content_change_poison_handle(case, mutation):
    case.journal.append("capture-0", case.observation)
    before = case.path.read_bytes()
    if mutation == "append":
        with case.path.open("ab") as output: output.write(b"{")
    elif mutation == "truncate": case.path.write_bytes(before[:-1])
    elif mutation == "rewrite":
        info = case.path.stat(); case.path.write_bytes(before.replace(b"capture-0", b"capture-X"))
        os.utime(case.path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
    elif mutation == "unlink": case.path.unlink()
    elif mutation == "hardlink": os.link(case.path, case.path.with_suffix(".link"))
    elif mutation == "parent-move":
        parent = case.path.parent
        moved = parent.with_name(parent.name + "-moved")
        parent.rename(moved); parent.mkdir()
    else:
        saved = case.path.with_suffix(".saved"); case.path.rename(saved)
        if mutation == "replace": case.path.write_bytes(before)
        else: case.path.symlink_to(saved)
    with pytest.raises(Unavailable): case.journal.checkpoint()
    assert_poisoned(case)


@pytest.mark.parametrize("kind", ("symlink", "hardlink", "directory", "fifo"))
def test_unsafe_file_type_not_opened_or_overwritten(tmp_path, kind):
    path = tmp_path / "events"
    real = tmp_path / "real"
    Journal.create(real, journal_id=JID, ledger_id=LID).close()
    before = real.read_bytes()
    if kind == "symlink": path.symlink_to(real)
    elif kind == "hardlink": os.link(real, path)
    elif kind == "directory": path.mkdir()
    else: os.mkfifo(path)
    with pytest.raises((OSError, Unavailable)):
        Journal.open_existing(path, journal_id=JID, ledger_id=LID)
    assert real.read_bytes() == before


@pytest.mark.parametrize("limit", ("MAX_LINE_BYTES", "MAX_RECORDS", "MAX_FILE_BYTES"))
def test_append_limits_fail_before_write_without_poisoning(case, monkeypatch, limit):
    before = case.path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(module, limit, 0 if limit == "MAX_RECORDS" else len(before))
        with pytest.raises(ValueError): case.journal.append("capture-0", case.observation)
    assert case.path.read_bytes() == before and case.journal.checkpoint().sequence == 0
    assert case.journal.append("capture-0", case.observation).sequence == 1


@pytest.mark.parametrize("limit", ("MAX_LINE_BYTES", "MAX_RECORDS", "MAX_FILE_BYTES"))
def test_recovery_limits_reject_without_repair(case, monkeypatch, limit):
    add_all(case); case.journal.close()
    before = case.path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(module, limit, 0 if limit == "MAX_RECORDS" else 300)
        with pytest.raises(Corrupt): reopen(case)
    assert case.path.read_bytes() == before


def test_retry_is_allowed_at_capacity_but_cannot_add_new_capture(case, monkeypatch):
    receipt = case.journal.append("c", case.observation)
    monkeypatch.setattr(module, "MAX_RECORDS", 1)
    assert case.journal.append("c", case.observation) == receipt
    with pytest.raises(ValueError): case.journal.append("d", case.observation)


def test_close_is_idempotent_no_deferred_write_fsync_or_repair(case, monkeypatch):
    add_all(case)
    def forbidden(*args): pytest.fail("close/read/export must not write or sync")
    monkeypatch.setattr(module.os, "write", forbidden)
    monkeypatch.setattr(module.os, "fsync", forbidden)
    checked_export(case.journal)
    case.journal.get_capture("capture-0")
    case.journal.close(); case.journal.close()
    assert_poisoned(case)


def test_export_final_guard_detects_external_change_during_copy(case, monkeypatch):
    add_all(case)
    anchor = case.journal.checkpoint()
    original = module.execution_event_journal_export_v1_hash
    def changed(export):
        result = original(export)
        with case.path.open("ab") as output: output.write(b"{")
        return result
    monkeypatch.setattr(module, "execution_event_journal_export_v1_hash", changed)
    with pytest.raises(Unavailable): case.journal.export(checkpoint=anchor)
    assert_poisoned(case)


def test_close_failure_not_retried_and_other_descriptor_still_closed(case, monkeypatch):
    fd, directory, close, calls = case.journal._fd, case.journal._dir_fd, os.close, []
    def fail_once(value):
        calls.append(value)
        close(value)
        if value == fd: raise OSError(errno.EIO, "injected close failure")
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "close", fail_once)
        with pytest.raises(Unavailable): case.journal.close()
        case.journal.close()
    assert calls == [fd, directory]
    assert_poisoned(case)


def test_context_exit_does_not_mask_primary_failure(case, monkeypatch):
    original = Journal.close
    def bad_close(self):
        original(self)
        raise Unavailable("injected close error")
    with monkeypatch.context() as patch:
        patch.setattr(Journal, "close", bad_close)
        with pytest.raises(ValueError, match="primary") as caught:
            with case.journal: raise ValueError("primary")
        assert any("close also failed" in note for note in caught.value.__notes__)


def test_wrong_public_types_and_checkpoints_are_not_duck_typed(case):
    with pytest.raises(TypeError): Journal()
    with pytest.raises(TypeError): case.journal.append("c", {})
    with pytest.raises(TypeError): case.journal.export(checkpoint={})
    with pytest.raises(TypeError): export_hash({})
    with pytest.raises(TypeError): verify_export({}, checkpoint=case.journal.checkpoint())
    for invalid in (True, -1, 0.0):
        with pytest.raises(ValueError): Checkpoint(JID, LID, invalid, "e" * 64)
    with pytest.raises(ValueError): Checkpoint(JID, LID, 0, "E" * 64)


def test_unavailable_platform_fails_before_open(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "fcntl", None)
    path = tmp_path / "events"
    with pytest.raises(Unavailable): Journal.create(path, journal_id=JID, ledger_id=LID)
    assert not path.exists()


def test_cross_process_lock_and_crash_recovery_of_durable_capture(case):
    root = str(Path(module.__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": root, "PYTHONDONTWRITEBYTECODE": "1"}
    code = """
import os, sys, socket
from std0_quant.execution.durable_execution_event_journal_v1 import DurableExecutionEventJournalV1 as J, EventJournalBusyError
try:
    j = J.open_existing(sys.argv[1], journal_id='offline-journal', ledger_id='offline-intents')
except EventJournalBusyError:
    print('BUSY', flush=True)
else:
    raise AssertionError('expected exclusive lock')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(case.path)], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and result.stdout.strip() == "BUSY", result.stderr
    add_all(case)
    cp, expected = checked_export(case.journal)
    case.journal.close()
    before, intent_before = case.path.read_bytes(), case.intent_path.read_bytes()
    code = """
import os, sys, json, socket
from std0_quant.execution.durable_execution_event_journal_v1 import (
    DurableExecutionEventJournalV1 as J, ExecutionEventJournalCheckpointV1 as CP,
    execution_events_from_journal_export_v1 as events)
def forbidden(*a, **kw): raise AssertionError('network forbidden')
socket.socket = socket.create_connection = forbidden
cp = CP(**json.loads(sys.argv[2]))
j = J.open_existing(sys.argv[1], journal_id=cp.journal_id, ledger_id=cp.ledger_id, checkpoint=cp)
e = j.export(checkpoint=cp)
assert len(events(e, checkpoint=cp)) == 3
j.append('after-crash', events(e, checkpoint=cp)[0])
print(e.to_json(), flush=True)
os._exit(0)  # No close/finalizer, analogous to abrupt process termination.
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(case.path), json.dumps(asdict(cp))],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected.to_json()
    with reopen(case, checkpoint=cp) as journal:
        assert journal.checkpoint().sequence == 4
        assert journal.export(checkpoint=cp) == expected
        assert journal.append("after-crash", case.observation).sequence == 4
    assert case.path.read_bytes().startswith(before) and case.intent_path.read_bytes() == intent_before


def test_fork_inherited_handle_rejected_before_mutex_and_child_close_keeps_parent_lock(case):
    # Isolate fork in a fresh single-threaded interpreter, avoiding pytest's threads.
    case.journal.close()
    root = str(Path(module.__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": root, "PYTHONDONTWRITEBYTECODE": "1"}
    code = """
import os, sys
from std0_quant.execution.durable_execution_event_journal_v1 import (
    DurableExecutionEventJournalV1 as J, EventJournalBusyError, EventJournalUnavailableError)
j = J.open_existing(sys.argv[1], journal_id='offline-journal', ledger_id='offline-intents')
pid = os.fork()
if pid == 0:
    class BadMutex:
        def __enter__(self): raise AssertionError('child reached inherited mutex')
        def __exit__(self, *a): pass
    j._mutex = BadMutex()
    try: j.checkpoint()
    except EventJournalUnavailableError: pass
    else: os._exit(2)
    j.close()
    os._exit(0)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
try: J.open_existing(sys.argv[1], journal_id='offline-journal', ledger_id='offline-intents')
except EventJournalBusyError: pass
else: raise AssertionError('child unlocked parent')
assert j.checkpoint().sequence == 0
j.close()
print('FORK_GUARDS_OK')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(case.path)], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and result.stdout.strip() == "FORK_GUARDS_OK", result.stderr


def test_verified_conversion_uses_detached_copy_not_caller_object_after_validation(case, monkeypatch):
    add_all(case)
    cp, export = checked_export(case.journal)
    original = module.verify_execution_event_journal_export_v1
    def validate_then_change_caller(selected, *, checkpoint):
        result = original(selected, checkpoint=checkpoint)
        object.__setattr__(export.entries[0].observation.event, "reason", "changed-after-check")
        return result
    monkeypatch.setattr(module, "verify_execution_event_journal_export_v1", validate_then_change_caller)
    assert to_events(export, checkpoint=cp) == case.observations
    assert export.entries[0].observation.event.reason == "changed-after-check"


def test_recovery_validation_change_releases_handle_without_adopting_it(case, monkeypatch):
    add_all(case); case.journal.close()
    original = Journal._recover
    def mutate(self):
        original(self)
        with self._path.open("ab") as output: output.write(b"{")
    with monkeypatch.context() as patch:
        patch.setattr(Journal, "_recover", mutate)
        with pytest.raises(Corrupt, match="validation"):
            reopen(case)
    assert case.path.read_bytes().endswith(b"{")


@pytest.mark.parametrize("stage", ("partial-header", "wrong-readback"))
def test_create_io_failure_keeps_file_for_explicit_inspection(tmp_path, monkeypatch, stage):
    path = tmp_path / "events"
    write = os.write
    def fail(fd, raw):
        write(fd, raw[:17])
        raise OSError(errno.ENOSPC, "injected")
    with monkeypatch.context() as patch:
        if stage == "partial-header": patch.setattr(module.os, "write", fail)
        else: patch.setattr(module.os, "pread", lambda fd, size, offset: b"x" * size)
        with pytest.raises((OSError, Corrupt)):
            Journal.create(path, journal_id=JID, ledger_id=LID)
    before = path.read_bytes()
    if stage == "partial-header":
        with pytest.raises(Corrupt): Journal.open_existing(path, journal_id=JID, ledger_id=LID)
    else:
        with Journal.open_existing(path, journal_id=JID, ledger_id=LID): pass
    assert path.read_bytes() == before


def test_fully_rehashed_history_needs_an_independent_anchor_not_just_internal_hashes(case):
    add_all(case)
    trusted = case.journal.checkpoint()
    case.journal.close()
    lines = case.path.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[-1])
    row["observation"]["event"]["reason"] = "forged-but-valid-contract"
    row["observation_hash"] = reference_hash("observation", row["observation"])
    del row["record_hash"]
    row["record_hash"] = reference_hash("record", row)
    lines[-1] = canonical(row) + b"\n"
    case.path.write_bytes(b"".join(lines))
    with pytest.raises(Corrupt): reopen(case, checkpoint=trusted)
    with reopen(case) as journal:
        own_cp, export = checked_export(journal)
        assert own_cp != trusted
        assert verify_export(export, checkpoint=trusted) == ("EVENT_JOURNAL_EXPORT_V1_PREFIX_MISMATCH",)
        assert to_events(export, checkpoint=own_cp)[-1].event.reason == "forged-but-valid-contract"


def test_recovered_capture_retry_reuses_original_record_and_conflict_is_explicit(case):
    receipt = case.journal.append("stable", case.observation)
    anchor = case.journal.checkpoint()
    case.journal.close()
    before = case.path.read_bytes()
    with reopen(case, checkpoint=anchor) as journal:
        assert journal.append("stable", case.observation) == receipt
        with pytest.raises(CaptureConflictError):
            journal.append("stable", replace(case.observation, event=replace(case.observation.event, reason="different")))
        assert journal.checkpoint() == anchor
    assert case.path.read_bytes() == before


def test_mutated_scope_cannot_be_reauthorized_by_recomputing_export_hash(case):
    cp, export = checked_export(case.journal)
    object.__setattr__(export, "mode", "LIVE")
    object.__setattr__(export, "artifact_hash", export_hash(export))
    assert verify_export(export, checkpoint=cp) == ("EVENT_JOURNAL_EXPORT_V1_PREFIX_MISMATCH",)
    with pytest.raises(ValueError): to_events(export, checkpoint=cp)
