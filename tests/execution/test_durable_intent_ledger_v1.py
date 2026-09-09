"""Offline real-file tests, OS-failure injection, and subprocess recovery.

These tests do not certify hardware power-loss behavior or venue submission.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from std0_quant.execution.contracts import OrderIntent
import std0_quant.execution.durable_intent_ledger_v1 as ledger_module
from std0_quant.execution.durable_intent_ledger_v1 import (
    DurableIntentLedgerV1 as Ledger,
    DurableIntentReceiptV1,
    IntentConflictError,
    LedgerBusyError,
    LedgerCheckpointV1,
    LedgerCorruptionError,
    LedgerUnavailableError,
    MAX_LINE_BYTES,
    order_intent_payload_hash_v1,
)

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="ledger v1 supports Linux local filesystems")
LEDGER_ID = "offline-test-ledger"


def intent(**changes):
    values = dict(intent_id="intent-1", condition_id="m1", outcome="Up", side="BUY",
                  qty=10, limit_price=0.5, time_in_force="GTC", decision_ts_ms=1001,
                  market_data_ts_ms=1000, strategy_id="std0_candidate",
                  strategy_version="v1", risk_policy_version="risk_v1")
    return OrderIntent(**{**values, **changes})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(kind, payload):
    # Independent schema/hash reference, not the implementation helper.
    domain = b"std0-quant/durable-intent-ledger/v1/" + kind.encode() + b"\n"
    return hashlib.sha256(domain + canonical(payload)).hexdigest()


def rehash(row, kind="record"):
    payload = {k: v for k, v in row.items() if k != "record_hash"}
    return {**payload, "record_hash": digest(kind, payload)}


def encode_rows(rows):
    return b"".join(canonical(row) + b"\n" for row in rows)


def child(code, *args):
    source_root = str(Path(ledger_module.__file__).resolve().parents[2])
    env = {**os.environ, "PYTHONPATH": source_root + os.pathsep
           + os.environ.get("PYTHONPATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-B", "-c", code, *map(str, args)],
                          env=env, capture_output=True, text=True, timeout=20)


@pytest.fixture
def populated(tmp_path):
    path = tmp_path / "intents.ndjson"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        first = ledger.append(intent())
        second = ledger.append(intent(intent_id="intent-2", qty=11))
        checkpoint = ledger.checkpoint()
    return path, first, second, checkpoint


def test_create_append_receipt_schema_and_hashes_independently(tmp_path):
    path = tmp_path / "intents.ndjson"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        assert ledger.receipts() == ()
        assert ledger.get_intent("missing") is None
        empty = ledger.checkpoint()
        receipt = ledger.append(intent())
        assert receipt.sequence == 1
        assert receipt.schema_version == "durable_intent_receipt_v1"
        assert ledger.get_intent("intent-1") == intent()
        assert ledger.receipts() == (receipt,)
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
        header, row = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert header == rehash({"schema_version": "durable_intent_ledger_header_v1",
                                 "ledger_id": LEDGER_ID}, "header")
        assert empty == LedgerCheckpointV1(LEDGER_ID, 0, header["record_hash"])
        assert row["schema_version"] == "durable_intent_record_v1"
        assert row["intent"] == intent().to_dict()
        assert row["intent_hash"] == digest("intent", intent().to_dict())
        assert row == rehash(row)
        assert row["previous_hash"] == header["record_hash"]
        assert receipt.record_hash == row["record_hash"]
        assert ledger.checkpoint() == LedgerCheckpointV1(LEDGER_ID, 1, receipt.record_hash)


def test_reopen_restores_all_intents_receipts_and_append_sequence(populated):
    path, first, second, checkpoint = populated
    with Ledger.open_existing(path, ledger_id=LEDGER_ID, checkpoint=checkpoint) as ledger:
        assert ledger.receipts() == (first, second)
        assert ledger.get_intent("intent-2") == intent(intent_id="intent-2", qty=11)
        assert ledger.append(intent()) == first
        assert ledger.append(intent(intent_id="intent-3")).sequence == 3


def test_deterministic_files_and_hashes_for_same_inputs(tmp_path):
    outputs = []
    for name in ("a", "b"):
        path = tmp_path / name
        with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
            ledger.append(intent())
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]


def test_ledger_identity_changes_record_hash_not_intent_hash(tmp_path):
    receipts = []
    for name in ("a", "b"):
        with Ledger.create(tmp_path / name, ledger_id=name) as ledger:
            receipts.append(ledger.append(intent()))
    assert receipts[0].intent_hash == receipts[1].intent_hash
    assert receipts[0].record_hash != receipts[1].record_hash


@pytest.mark.parametrize("field,value", (
    ("intent_id", "intent-2"), ("condition_id", "m2"), ("outcome", "Down"),
    ("side", "SELL"), ("qty", 11), ("limit_price", 0.6), ("time_in_force", "IOC"),
    ("decision_ts_ms", 1002), ("market_data_ts_ms", 999), ("strategy_id", "other"),
    ("strategy_version", "v2"), ("risk_policy_version", "risk-v2"),
))
def test_intent_hash_binds_every_normalized_business_field(field, value):
    first = intent()
    changed = intent(**{field: value})
    assert order_intent_payload_hash_v1(first) != order_intent_payload_hash_v1(changed)
    assert order_intent_payload_hash_v1(changed) == digest("intent", changed.to_dict())


def test_unicode_payload_roundtrip(tmp_path):
    value = intent(strategy_id="策略-α", outcome="上涨")
    path = tmp_path / "账本.ndjson"
    with Ledger.create(path, ledger_id="测试账本") as ledger:
        receipt = ledger.append(value)
    assert "策略".encode() in path.read_bytes()
    with Ledger.open_existing(path, ledger_id="测试账本") as ledger:
        assert ledger.get_intent(value.intent_id) == value
        assert ledger.append(value) == receipt


def test_repeated_append_is_idempotent_without_write_or_fsync(tmp_path, monkeypatch):
    path = tmp_path / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        first = ledger.append(intent())
        before = path.read_bytes()
        def forbidden(*args):
            pytest.fail("duplicate append must not perform new write or fsync")
        with monkeypatch.context() as patch:
            patch.setattr(ledger_module.os, "write", forbidden)
            patch.setattr(ledger_module.os, "fsync", forbidden)
            assert ledger.append(intent()) == first
        assert path.read_bytes() == before
        assert len(ledger.receipts()) == 1


@pytest.mark.parametrize("field,value", (("qty", 12), ("strategy_version", "v2"),
                                         ("market_data_ts_ms", 900)))
def test_same_id_different_content_rejected_without_poison(tmp_path, field, value):
    path = tmp_path / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        first = ledger.append(intent())
        before = path.read_bytes()
        with pytest.raises(IntentConflictError):
            ledger.append(intent(**{field: value}))
        assert path.read_bytes() == before
        assert ledger.append(intent()) == first
        assert ledger.append(intent(intent_id="intent-2")).sequence == 2


def test_create_never_overwrites_and_existing_open_never_creates(tmp_path, populated):
    path = populated[0]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        Ledger.create(path, ledger_id=LEDGER_ID)
    assert path.read_bytes() == before
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        Ledger.open_existing(missing, ledger_id=LEDGER_ID)
    assert not missing.exists()


@pytest.mark.parametrize("method", ("create", "open_existing"))
def test_missing_parent_not_created(tmp_path, method):
    path = tmp_path / "absent" / "ledger"
    with pytest.raises(FileNotFoundError):
        getattr(Ledger, method)(path, ledger_id=LEDGER_ID)
    assert not path.parent.exists()


@pytest.mark.parametrize("ledger_id,error", (("", ValueError), (" x ", ValueError),
    (123, TypeError), (None, TypeError), ("x" * 257, ValueError)))
def test_bad_ledger_id_rejected_before_file_creation(tmp_path, ledger_id, error):
    path = tmp_path / "ledger"
    with pytest.raises(error):
        Ledger.create(path, ledger_id=ledger_id)
    assert not path.exists()


@pytest.mark.parametrize("value", (None, {}, "intent", 123))
def test_wrong_intent_type_rejected_without_write(tmp_path, value):
    path = tmp_path / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        before = path.read_bytes()
        with pytest.raises(TypeError):
            ledger.append(value)
        assert path.read_bytes() == before
        assert ledger.append(intent()).sequence == 1


def test_oversized_intent_rejected_before_write_without_poison(tmp_path):
    path = tmp_path / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        before = path.read_bytes()
        with pytest.raises(ValueError, match="MAX_LINE_BYTES"):
            ledger.append(intent(strategy_id="x" * MAX_LINE_BYTES))
        assert path.read_bytes() == before
        assert ledger.append(intent()).sequence == 1


def test_mutated_nonfinite_intent_rejected_before_write(tmp_path):
    value = intent()
    object.__setattr__(value, "qty", float("nan"))
    path = tmp_path / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        with pytest.raises(ValueError):
            ledger.append(value)
        assert ledger.receipts() == ()


def test_get_intent_does_not_expose_internal_mutable_payload(populated):
    path, first, second, _ = populated
    with Ledger.open_existing(path, ledger_id=LEDGER_ID) as ledger:
        value = ledger.get_intent("intent-1")
        object.__setattr__(value, "qty", 999.0)
        assert ledger.get_intent("intent-1") == intent()
        assert ledger.append(intent()) == first
        assert ledger.receipts() == (first, second)


def test_each_new_append_syncs_before_return_and_create_syncs_parent(tmp_path, monkeypatch):
    events = []
    real_write, real_sync = os.write, os.fsync
    def write(fd, payload):
        events.append("write")
        return real_write(fd, payload)
    def sync(fd):
        events.append("dir-sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file-sync")
        return real_sync(fd)
    path = tmp_path / "ledger"
    with monkeypatch.context() as patch:
        patch.setattr(ledger_module.os, "write", write)
        patch.setattr(ledger_module.os, "fsync", sync)
        ledger = Ledger.create(path, ledger_id=LEDGER_ID)
        assert events == ["write", "file-sync", "dir-sync"]
        events.clear()
        ledger.append(intent())
        assert events == ["write", "file-sync"]
        events.clear()
        ledger.close()
        assert events == []
        with Ledger.open_existing(path, ledger_id=LEDGER_ID):
            assert events == ["file-sync", "dir-sync"]


@pytest.mark.parametrize("stage", ("write", "file-sync", "dir-sync"))
def test_create_io_failure_preserves_file_and_never_returns_handle(tmp_path, monkeypatch, stage):
    path = tmp_path / "ledger"
    real_sync = os.fsync
    def fail_write(*args):
        raise OSError(errno.ENOSPC, "injected full disk")
    def sync(fd):
        is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if is_directory == (stage == "dir-sync"):
            raise OSError(errno.EIO, "injected fsync failure")
        return real_sync(fd)
    with monkeypatch.context() as patch:
        if stage == "write":
            patch.setattr(ledger_module.os, "write", fail_write)
        else:
            patch.setattr(ledger_module.os, "fsync", sync)
        with pytest.raises(OSError):
            Ledger.create(path, ledger_id=LEDGER_ID)
    assert path.exists()
    if stage == "write":
        assert path.read_bytes() == b""
        with pytest.raises(LedgerCorruptionError):
            Ledger.open_existing(path, ledger_id=LEDGER_ID)
    else:
        with Ledger.open_existing(path, ledger_id=LEDGER_ID) as ledger:
            assert ledger.receipts() == ()


@pytest.mark.parametrize("stage", ("file-sync", "dir-sync"))
def test_recovery_requires_successful_resync(populated, monkeypatch, stage):
    path = populated[0]
    before = path.read_bytes()
    real_sync = os.fsync
    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) == (stage == "dir-sync"):
            raise OSError(errno.EIO, "injected recovery sync error")
        return real_sync(fd)
    with monkeypatch.context() as patch:
        patch.setattr(ledger_module.os, "fsync", sync)
        with pytest.raises(OSError):
            Ledger.open_existing(path, ledger_id=LEDGER_ID)
    assert path.read_bytes() == before
    with Ledger.open_existing(path, ledger_id=LEDGER_ID) as ledger:
        assert len(ledger.receipts()) == 2


def test_short_writes_and_interrupted_write_are_completed(tmp_path, monkeypatch):
    path = tmp_path / "ledger"
    real_write = os.write
    count = 0
    def short_write(fd, payload):
        nonlocal count
        count += 1
        if count == 1:
            raise InterruptedError(errno.EINTR, "interrupted")
        return real_write(fd, payload[:17])
    with monkeypatch.context() as patch:
        patch.setattr(ledger_module.os, "write", short_write)
        with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
            receipt = ledger.append(intent())
    assert count > 3
    with Ledger.open_existing(path, ledger_id=LEDGER_ID) as ledger:
        assert ledger.append(intent()) == receipt


@pytest.mark.parametrize("kind", ("zero-write", "enospc", "partial-write", "fsync", "interrupt"))
def test_append_fault_poisoning_and_explicit_recovery(tmp_path, monkeypatch, kind):
    path = tmp_path / "ledger"
    ledger = Ledger.create(path, ledger_id=LEDGER_ID)
    real_write = os.write
    attempts = 0
    def failing_write(fd, payload):
        nonlocal attempts
        attempts += 1
        if kind == "zero-write":
            return 0
        if kind in {"partial-write", "interrupt"} and attempts == 1:
            return real_write(fd, payload[:17])
        if kind == "interrupt":
            raise KeyboardInterrupt()
        raise OSError(errno.ENOSPC, "injected write failure")
    def failing_sync(fd):
        raise OSError(errno.EIO, "injected sync failure")
    try:
        with monkeypatch.context() as patch:
            if kind == "fsync":
                patch.setattr(ledger_module.os, "fsync", failing_sync)
            else:
                patch.setattr(ledger_module.os, "write", failing_write)
            with pytest.raises(KeyboardInterrupt if kind == "interrupt" else LedgerUnavailableError):
                ledger.append(intent())
        for action in (lambda: ledger.append(intent()), ledger.receipts,
                       ledger.checkpoint, lambda: ledger.get_intent("intent-1")):
            with pytest.raises(LedgerUnavailableError):
                action()
        failed_bytes = path.read_bytes()
    finally:
        ledger.close()
    assert path.read_bytes() == failed_bytes
    if kind in {"partial-write", "interrupt"}:
        with pytest.raises(LedgerCorruptionError):
            Ledger.open_existing(path, ledger_id=LEDGER_ID)
        assert path.read_bytes() == failed_bytes
    else:
        with Ledger.open_existing(path, ledger_id=LEDGER_ID) as recovered:
            assert len(recovered.receipts()) == (1 if kind == "fsync" else 0)
            assert recovered.append(intent()).sequence == 1
            assert len(recovered.receipts()) == 1


@pytest.mark.parametrize("mutation", (
    "empty", "header-only-partial", "missing-newline", "partial-tail", "blank-tail",
    "invalid-utf8", "duplicate-key", "extra-field", "header-hash", "header-id",
    "record-schema", "intent-tamper", "intent-extra-field", "reverse-rows",
    "missing-first-record", "duplicate-row", "duplicate-id-rehashed",
    "previous-hash-rehashed", "sequence-bool-rehashed", "noncanonical-number",
    "nan", "noncanonical-whitespace", "oversized-line",
))
def test_corrupt_history_is_blocked_without_silent_repair(populated, mutation):
    path = populated[0]
    original = path.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    raw = None
    if mutation == "empty": raw = b""
    elif mutation == "header-only-partial": raw = original.splitlines()[0][:-1]
    elif mutation == "missing-newline": raw = original[:-1]
    elif mutation == "partial-tail": raw = original + b"{"
    elif mutation == "blank-tail": raw = original + b"\n"
    elif mutation == "invalid-utf8": raw = original + b"\xff\n"
    elif mutation == "duplicate-key":
        raw = original.replace(b'"sequence":1', b'"sequence":1,"sequence":1', 1)
    elif mutation == "extra-field": rows[1]["unauthorized"] = True
    elif mutation == "header-hash": rows[0]["record_hash"] = "f" * 64
    elif mutation == "header-id": rows[0] = rehash({**rows[0], "ledger_id": "other"}, "header")
    elif mutation == "record-schema": rows[1]["schema_version"] = "unknown"
    elif mutation == "intent-tamper": rows[1]["intent"]["qty"] = 99.0
    elif mutation == "intent-extra-field": rows[1]["intent"]["unknown"] = "x"
    elif mutation == "reverse-rows": rows[1:] = reversed(rows[1:])
    elif mutation == "missing-first-record": del rows[1]
    elif mutation == "duplicate-row": rows.append(rows[2])
    elif mutation == "duplicate-id-rehashed":
        rows[2]["intent"] = rows[1]["intent"]
        rows[2]["intent_hash"] = digest("intent", rows[2]["intent"])
        rows[2] = rehash(rows[2])
    elif mutation == "previous-hash-rehashed":
        rows[2] = rehash({**rows[2], "previous_hash": "f" * 64})
    elif mutation == "sequence-bool-rehashed": rows[1] = rehash({**rows[1], "sequence": True})
    elif mutation == "noncanonical-number":
        rows[2]["intent"]["qty"] = 11
        rows[2]["intent_hash"] = digest("intent", rows[2]["intent"])
        rows[2] = rehash(rows[2])
    elif mutation == "nan": raw = original.replace(b'"qty":10.0', b'"qty":NaN', 1)
    elif mutation == "noncanonical-whitespace": raw = b" " + original
    elif mutation == "oversized-line": raw = original + b"x" * MAX_LINE_BYTES + b"\n"
    if raw is None:
        raw = encode_rows(rows)
    assert raw != original
    path.write_bytes(raw)
    with pytest.raises(LedgerCorruptionError):
        Ledger.open_existing(path, ledger_id=LEDGER_ID)
    assert path.read_bytes() == raw


def test_checkpoint_detects_complete_suffix_removal_without_overclaim(populated):
    path, first, _, checkpoint = populated
    shorter = b"\n".join(path.read_bytes().splitlines()[:2]) + b"\n"
    path.write_bytes(shorter)
    with pytest.raises(LedgerCorruptionError, match="shorter"):
        Ledger.open_existing(path, ledger_id=LEDGER_ID, checkpoint=checkpoint)
    assert path.read_bytes() == shorter
    # A valid prefix is undetectably shorter without an external anchor.
    with Ledger.open_existing(path, ledger_id=LEDGER_ID) as ledger:
        assert ledger.receipts() == (first,)


def test_checkpoint_detects_rehashed_history_and_accepts_longer_valid_prefix(populated):
    path, first, _, _ = populated
    checkpoint = LedgerCheckpointV1(LEDGER_ID, first.sequence, first.record_hash)
    with Ledger.open_existing(path, ledger_id=LEDGER_ID, checkpoint=checkpoint) as ledger:
        assert len(ledger.receipts()) == 2
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[1]["intent"]["qty"] = 12.0
    for index in range(1, len(rows)):
        rows[index]["previous_hash"] = rows[index - 1]["record_hash"]
        rows[index]["intent_hash"] = digest("intent", rows[index]["intent"])
        rows[index] = rehash(rows[index])
    path.write_bytes(encode_rows(rows))
    with pytest.raises(LedgerCorruptionError, match="checkpoint hash"):
        Ledger.open_existing(path, ledger_id=LEDGER_ID, checkpoint=checkpoint)


@pytest.mark.parametrize("checkpoint", ("wrong-type", "wrong-id", "wrong-hash", "header"))
def test_checkpoint_input_validation(populated, checkpoint):
    path, _, _, tip = populated
    if checkpoint == "wrong-type":
        anchor, error = object(), TypeError
    elif checkpoint == "wrong-id":
        anchor, error = replace(tip, ledger_id="other"), LedgerCorruptionError
    elif checkpoint == "wrong-hash":
        anchor, error = replace(tip, record_hash="f" * 64), LedgerCorruptionError
    else:
        header = json.loads(path.read_bytes().splitlines()[0])
        anchor = LedgerCheckpointV1(LEDGER_ID, 0, header["record_hash"])
        with Ledger.open_existing(path, ledger_id=LEDGER_ID, checkpoint=anchor):
            pass
        return
    with pytest.raises(error):
        Ledger.open_existing(path, ledger_id=LEDGER_ID, checkpoint=anchor)


def test_second_handle_same_process_and_child_are_excluded(tmp_path):
    path = tmp_path / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        with pytest.raises(LedgerBusyError):
            Ledger.open_existing(path, ledger_id=LEDGER_ID)
        result = child("""
import sys
from std0_quant.execution.durable_intent_ledger_v1 import DurableIntentLedgerV1, LedgerBusyError
try:
    DurableIntentLedgerV1.open_existing(sys.argv[1], ledger_id=sys.argv[2])
except LedgerBusyError:
    print("LOCK_BLOCKED")
else:
    raise SystemExit(3)
""", path, LEDGER_ID)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "LOCK_BLOCKED"
        ledger.append(intent())
    with Ledger.open_existing(path, ledger_id=LEDGER_ID):
        pass


def test_subprocess_crash_after_ack_preserves_record_without_close(tmp_path):
    path = tmp_path / "ledger"
    result = child("""
import json, os, sys
from dataclasses import asdict
from std0_quant.execution.contracts import OrderIntent
from std0_quant.execution.durable_intent_ledger_v1 import DurableIntentLedgerV1
ledger = DurableIntentLedgerV1.create(sys.argv[1], ledger_id=sys.argv[2])
receipt = ledger.append(OrderIntent.from_json(sys.argv[3]))
print(json.dumps(asdict(receipt)), flush=True)
os._exit(0)
""", path, LEDGER_ID, intent().to_json())
    assert result.returncode == 0, result.stderr
    receipt = DurableIntentReceiptV1(**json.loads(result.stdout))
    with Ledger.open_existing(path, ledger_id=LEDGER_ID) as ledger:
        assert ledger.receipts() == (receipt,)
        assert ledger.append(intent()) == receipt


def test_subprocess_crash_mid_write_preserves_incomplete_tail_and_blocks(tmp_path):
    path = tmp_path / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID):
        pass
    result = child("""
import os, sys
from std0_quant.execution.contracts import OrderIntent
from std0_quant.execution.durable_intent_ledger_v1 import DurableIntentLedgerV1
ledger = DurableIntentLedgerV1.open_existing(sys.argv[1], ledger_id=sys.argv[2])
original = os.write
def interrupted(fd, payload):
    original(fd, payload[:19])
    os._exit(7)
os.write = interrupted
ledger.append(OrderIntent.from_json(sys.argv[3]))
""", path, LEDGER_ID, intent().to_json())
    assert result.returncode == 7, result.stderr
    before = path.read_bytes()
    with pytest.raises(LedgerCorruptionError):
        Ledger.open_existing(path, ledger_id=LEDGER_ID)
    assert path.read_bytes() == before


def test_fork_inherited_handle_rejected_and_child_close_does_not_unlock_parent(tmp_path):
    path = tmp_path / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        code = """
import os, sys
from std0_quant.execution.durable_intent_ledger_v1 import DurableIntentLedgerV1, LedgerUnavailableError, LedgerBusyError
ledger = DurableIntentLedgerV1.open_existing(sys.argv[1], ledger_id=sys.argv[2])
pid = os.fork()
if pid == 0:
    try:
        ledger.receipts()
    except LedgerUnavailableError:
        ledger.close()
        os._exit(0)
    os._exit(8)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
try:
    DurableIntentLedgerV1.open_existing(sys.argv[1], ledger_id=sys.argv[2])
except LedgerBusyError:
    pass
else:
    raise SystemExit(9)
ledger.close()
"""
        ledger.close()
        result = child(code, path, LEDGER_ID)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("same_id", (False, True))
def test_same_handle_thread_serialization(tmp_path, same_id):
    path = tmp_path / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        def append(index):
            return ledger.append(intent(intent_id="shared" if same_id else f"i-{index}"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            receipts = list(pool.map(append, range(32)))
        expected = 1 if same_id else 32
        assert len(ledger.receipts()) == expected
        assert len(set(receipts)) == expected
        assert sorted({r.sequence for r in receipts}) == list(range(1, expected + 1))
    with Ledger.open_existing(path, ledger_id=LEDGER_ID) as ledger:
        assert len(ledger.receipts()) == expected


@pytest.mark.parametrize("mode", ("symlink", "hardlink", "directory", "fifo"))
def test_unsafe_file_types_rejected(tmp_path, mode):
    target, path = tmp_path / "real", tmp_path / "ledger"
    if mode in {"symlink", "hardlink"}:
        with Ledger.create(target, ledger_id=LEDGER_ID):
            pass
        before = target.read_bytes()
        if mode == "symlink": path.symlink_to(target)
        else: os.link(target, path)
    elif mode == "directory": path.mkdir()
    else: os.mkfifo(path)
    with pytest.raises((OSError, LedgerUnavailableError)):
        Ledger.open_existing(path, ledger_id=LEDGER_ID)
    if mode in {"symlink", "hardlink"}:
        assert target.read_bytes() == before


@pytest.mark.parametrize("mode", ("append", "truncate", "replace", "unlink", "hardlink", "parent-replace"))
def test_out_of_band_file_change_poisoning(tmp_path, mode):
    directory = tmp_path / "directory"
    directory.mkdir()
    path = directory / "ledger"
    with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
        ledger.append(intent())
        before = path.read_bytes()
        if mode == "append":
            with path.open("ab") as output: output.write(b"junk")
        elif mode == "truncate": path.write_bytes(before[:-1])
        elif mode == "replace":
            replacement = directory / "replacement"
            replacement.write_bytes(before)
            os.replace(replacement, path)
        elif mode == "unlink": path.unlink()
        elif mode == "hardlink": os.link(path, directory / "alias")
        else:
            directory.rename(tmp_path / "old-directory")
            directory.mkdir()
            path.write_bytes(before)
        with pytest.raises(LedgerUnavailableError):
            ledger.append(intent(intent_id="new"))
        with pytest.raises(LedgerUnavailableError):
            ledger.receipts()


def test_file_mutation_during_recovery_rejected(populated, monkeypatch):
    path = populated[0]
    original = Ledger._recover
    def change_after_scan(self):
        original(self)
        with path.open("ab") as out: out.write(b"junk")
    with monkeypatch.context() as patch:
        patch.setattr(Ledger, "_recover", change_after_scan)
        with pytest.raises(LedgerCorruptionError, match="during recovery"):
            Ledger.open_existing(path, ledger_id=LEDGER_ID)


def test_closed_handle_rejects_operations_and_close_is_idempotent(tmp_path):
    ledger = Ledger.create(tmp_path / "ledger", ledger_id=LEDGER_ID)
    ledger.close()
    ledger.close()
    for action in (ledger.receipts, ledger.checkpoint, lambda: ledger.get_intent("i"),
                   lambda: ledger.append(intent()), ledger.__enter__):
        with pytest.raises(LedgerUnavailableError):
            action()


def test_non_linux_support_is_explicit_and_does_not_create_file(tmp_path, monkeypatch):
    path = tmp_path / "ledger"
    monkeypatch.setattr(ledger_module, "fcntl", None)
    with pytest.raises(LedgerUnavailableError, match="Linux"):
        Ledger.create(path, ledger_id=LEDGER_ID)
    assert not path.exists()


@pytest.mark.parametrize("sequence", (True, -1, 1.0, "1"))
def test_checkpoint_rejects_bad_sequence(sequence):
    with pytest.raises(ValueError):
        LedgerCheckpointV1(LEDGER_ID, sequence, "a" * 64)


def test_direct_constructor_is_not_an_unsafe_open_shortcut():
    with pytest.raises(TypeError, match="create"):
        Ledger()


@pytest.mark.parametrize("primary_error", (False, True))
def test_close_reports_error_without_retry_or_masking_primary(tmp_path, monkeypatch, primary_error):
    ledger = Ledger.create(tmp_path / "ledger", ledger_id=LEDGER_ID)
    fd = ledger._fd
    real_close = os.close
    closed = []
    def failing_close(value):
        closed.append(value)
        real_close(value)
        if value == fd:
            raise OSError(errno.EIO, "injected delayed close failure")
    with monkeypatch.context() as patch:
        patch.setattr(ledger_module.os, "close", failing_close)
        expected = ValueError if primary_error else LedgerUnavailableError
        with pytest.raises(expected):
            with ledger:
                if primary_error:
                    raise ValueError("original operation error")
    assert closed.count(fd) == 1
    ledger.close()
    with Ledger.open_existing(tmp_path / "ledger", ledger_id=LEDGER_ID):
        pass


@pytest.mark.parametrize("method", ("create", "open_existing"))
@pytest.mark.parametrize("stage", ("file-sync", "dir-sync"))
@pytest.mark.parametrize("mutation", ("partial-tail", "truncate", "same-size-rewrite"))
def test_initialization_rejects_external_changes_during_final_sync(
    tmp_path, monkeypatch, method, stage, mutation
):
    """A final fsync must not adopt a file snapshot that was never validated.

    This models an external tool ignoring the advisory lock. It is a bounded
    fail-closed check, not a guarantee against arbitrary malicious writers.
    """
    path = tmp_path / "ledger"
    if method == "open_existing":
        with Ledger.create(path, ledger_id=LEDGER_ID) as ledger:
            ledger.append(intent())
    real_sync = os.fsync
    injected_bytes = None

    def sync_with_external_change(fd):
        nonlocal injected_bytes
        real_sync(fd)
        is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if injected_bytes is not None or is_directory != (stage == "dir-sync"):
            return
        info = path.stat()
        original = path.read_bytes()
        if mutation == "partial-tail":
            injected_bytes = original + b"{"
        elif mutation == "truncate":
            injected_bytes = original[:-1]
        else:
            identity = LEDGER_ID.encode("utf-8")
            injected_bytes = original.replace(identity, b"X" + identity[1:], 1)
            assert len(injected_bytes) == len(original)
        assert injected_bytes != original
        path.write_bytes(injected_bytes)
        # Deterministic timestamp distinction even on a coarse-resolution mount.
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))

    with monkeypatch.context() as patch:
        patch.setattr(ledger_module.os, "fsync", sync_with_external_change)
        with pytest.raises(LedgerCorruptionError):
            with getattr(Ledger, method)(path, ledger_id=LEDGER_ID):
                pass
    assert injected_bytes is not None
    assert path.read_bytes() == injected_bytes  # Never truncate or repair.
