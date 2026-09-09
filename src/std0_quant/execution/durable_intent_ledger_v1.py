"""Linux-local, single-writer durable intent ledger v1 (offline infrastructure).

Only create() may create a file. open_existing() never creates or repairs it.
Every successful new append uses unbuffered writes and fsync before returning.
Opening also fsyncs the validated file and its parent: a complete record left
by a failed/crashed writer must not be treated as previously acknowledged.

A receipt means durable local recording, NOT permission or proof of venue
submission, execution, or PRECOMMITTED_COVERAGE. Recovery never submits orders.
The existing OrderIntent contract is reused without changing its semantics.

Assumptions/limits: trusted, durably provisioned parent directories, cooperating writers,
Linux local filesystem and storage that honors fsync. No NFS/distributed locks,
malicious-writer protection, rotation, replication, or power-loss certification.
flock is advisory. A hash chain is NOT authentication: without an independently
trusted checkpoint, removal of a complete suffix or a fully rehashed history
cannot be detected. A supplied checkpoint is a required historical prefix, not
necessarily the latest record. Record payloads and an ID index are held in RAM.
Do not fork with an open ledger; inherited handles cannot read or append.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import threading
from typing import Any

try:
    import fcntl
except ImportError:  # Importable on other platforms; opening is Linux-only.
    fcntl = None

from std0_quant.execution.contracts import OrderIntent


HEADER_SCHEMA = "durable_intent_ledger_header_v1"
RECORD_SCHEMA = "durable_intent_record_v1"
RECEIPT_SCHEMA = "durable_intent_receipt_v1"
MAX_LINE_BYTES = 1024 * 1024
_HASH_DOMAIN = b"std0-quant/durable-intent-ledger/v1/"
_INTENT_FIELDS = frozenset(field.name for field in fields(OrderIntent))


class LedgerError(RuntimeError):
    """Ledger operation failed; no submit authorization is implied."""


class LedgerBusyError(LedgerError):
    """Another handle holds the exclusive lock."""


class LedgerCorruptionError(LedgerError):
    """Invalid history or an unsatisfied trusted checkpoint; no repair."""


class LedgerUnavailableError(LedgerError):
    """Closed, forked, poisoned, changed, or failed persistence handle."""


class IntentConflictError(LedgerError):
    """An existing intent ID is bound to different complete content."""


def _text(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    value.encode("utf-8")
    return value


def _hash_text(value: Any) -> str:
    value = _text(value, "hash")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("hash must be lowercase SHA256 hex")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(kind: str, payload: Any) -> str:
    return hashlib.sha256(_HASH_DOMAIN + kind.encode("ascii") + b"\n"
                          + _canonical(payload)).hexdigest()


def _intent_payload(intent: OrderIntent) -> dict[str, Any]:
    if type(intent) is not OrderIntent:
        raise TypeError("intent must be exactly OrderIntent")
    row = OrderIntent.to_dict(intent)
    normalized = OrderIntent.from_dict(row)
    if _canonical(row) != _canonical(normalized.to_dict()):
        raise ValueError("intent is not in normalized OrderIntent form")
    return row


def order_intent_payload_hash_v1(intent: OrderIntent) -> str:
    """Bind all normalized OrderIntent fields, including intent_id and times."""
    return _digest("intent", _intent_payload(intent))


@dataclass(frozen=True)
class LedgerCheckpointV1:
    """A prefix anchor; authenticity and independent storage are external."""

    ledger_id: str
    sequence: int
    record_hash: str

    def __post_init__(self) -> None:
        _text(self.ledger_id, "ledger_id")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("checkpoint sequence must be a nonnegative integer")
        _hash_text(self.record_hash)


@dataclass(frozen=True)
class DurableIntentReceiptV1:
    ledger_id: str
    sequence: int
    intent_id: str
    intent_hash: str
    record_hash: str
    schema_version: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _text(self.ledger_id, "ledger_id")
        _text(self.intent_id, "intent_id")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("receipt sequence must be a positive integer")
        _hash_text(self.intent_hash)
        _hash_text(self.record_hash)
        if self.schema_version != RECEIPT_SCHEMA:
            raise ValueError("unsupported receipt schema")


def _header(ledger_id: str) -> dict[str, Any]:
    payload = {"schema_version": HEADER_SCHEMA, "ledger_id": ledger_id}
    return {**payload, "record_hash": _digest("header", payload)}


def _record(ledger_id: str, sequence: int, previous_hash: str,
            intent: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": RECORD_SCHEMA, "ledger_id": ledger_id,
        "sequence": sequence, "previous_hash": previous_hash,
        "intent": intent, "intent_hash": _digest("intent", intent),
    }
    return {**payload, "record_hash": _digest("record", payload)}


def _receipt(row: dict[str, Any]) -> DurableIntentReceiptV1:
    return DurableIntentReceiptV1(
        row["ledger_id"], row["sequence"], row["intent"]["intent_id"],
        row["intent_hash"], row["record_hash"],
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number: " + value)


def _decode_line(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
        raise ValueError("oversized or incomplete ledger line")
    row = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                     parse_constant=_reject_constant)
    if type(row) is not dict or _canonical(row) + b"\n" != line:
        raise ValueError("noncanonical ledger line")
    return row


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written <= 0 or written > len(view):
            raise OSError(errno.EIO, "write made invalid progress")
        view = view[written:]


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


class DurableIntentLedgerV1:
    """Exclusive handle. Use a context manager, then explicitly close it."""

    def __init__(self) -> None:
        raise TypeError("use create() or open_existing()")

    @classmethod
    def create(cls, path: Path | str, *, ledger_id: str) -> DurableIntentLedgerV1:
        return cls._open(path, ledger_id=ledger_id, create=True, checkpoint=None)

    @classmethod
    def open_existing(
        cls, path: Path | str, *, ledger_id: str,
        checkpoint: LedgerCheckpointV1 | None = None,
    ) -> DurableIntentLedgerV1:
        return cls._open(path, ledger_id=ledger_id, create=False,
                         checkpoint=checkpoint)

    @classmethod
    def _open(cls, path: Path | str, *, ledger_id: str, create: bool,
              checkpoint: LedgerCheckpointV1 | None) -> DurableIntentLedgerV1:
        if not sys.platform.startswith("linux") or fcntl is None:
            raise LedgerUnavailableError("Linux local filesystems only")
        ledger_id = _text(ledger_id, "ledger_id")
        if len(ledger_id.encode("utf-8")) > 256:
            raise ValueError("ledger_id exceeds 256 UTF-8 bytes")
        if checkpoint is not None:
            if type(checkpoint) is not LedgerCheckpointV1:
                raise TypeError("checkpoint must be LedgerCheckpointV1")
            if checkpoint.ledger_id != ledger_id:
                raise LedgerCorruptionError("checkpoint ledger identity mismatch")
        self = cls.__new__(cls)
        self._fd = self._dir_fd = -1
        self._pid = os.getpid()
        self._mutex = threading.RLock()
        self._poisoned = False
        self._path = Path(os.path.abspath(os.fspath(path)))
        self._ledger_id = ledger_id
        self._records: list[tuple[DurableIntentReceiptV1, bytes]] = []
        self._by_id: dict[str, int] = {}
        self._tip = _header(ledger_id)["record_hash"]
        try:
            self._dir_fd = os.open(self._path.parent, os.O_RDONLY | os.O_DIRECTORY
                                   | os.O_CLOEXEC)
            flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            if create:
                flags |= os.O_CREAT | os.O_EXCL
            self._fd = os.open(self._path.name, flags, 0o600, dir_fd=self._dir_fd)
            self._identity()
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LedgerBusyError("ledger already locked") from exc
            info = self._identity()  # Snapshot only after taking the lock.
            if create:
                raw = _canonical(_header(ledger_id)) + b"\n"
                _write_all(self._fd, raw)
                info = self._identity()
                if info.st_size != len(raw):
                    raise LedgerCorruptionError("unexpected size after header write")
                before = _stamp(info)
            else:
                before = _stamp(info)
                self._recover()
                if _stamp(self._identity()) != before:
                    raise LedgerCorruptionError("file changed during recovery")
                if checkpoint is not None:
                    self._check_checkpoint(checkpoint)
            # Also re-establish persistence for records surviving an unacked write.
            os.fsync(self._fd)
            os.fsync(self._dir_fd)
            info = self._identity()
            if _stamp(info) != before:
                raise LedgerCorruptionError("file changed during initialization sync")
            # Keep the validated snapshot; do not bless intervening external writes.
            self._stamp = before
            return self
        except BaseException:
            self._release()
            raise

    def _identity(self) -> os.stat_result:
        info = os.fstat(self._fd)
        named = os.stat(self._path.name, dir_fd=self._dir_fd, follow_symlinks=False)
        parent = os.stat(self._path.parent)
        pinned_parent = os.fstat(self._dir_fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or not stat.S_ISREG(named.st_mode)
                or (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino)
                or (parent.st_dev, parent.st_ino)
                != (pinned_parent.st_dev, pinned_parent.st_ino)):
            raise LedgerUnavailableError("ledger path/inode/link identity changed")
        return info

    def _recover(self) -> None:
        os.lseek(self._fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(self._fd), "rb") as reader:
            line_number = 0
            while True:
                line = reader.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                line_number += 1
                try:
                    row = _decode_line(line)
                    if line_number == 1:
                        if row != _header(self._ledger_id):
                            raise ValueError("header/hash/ledger identity mismatch")
                        continue
                    payload = row.get("intent")
                    if type(payload) is not dict or set(payload) != _INTENT_FIELDS:
                        raise ValueError("intent fields mismatch")
                    intent = OrderIntent.from_dict(payload)
                    normalized = _intent_payload(intent)
                    if _canonical(payload) != _canonical(normalized):
                        raise ValueError("non-normalized intent payload")
                    sequence = len(self._records) + 1
                    if type(row.get("sequence")) is not int:
                        raise ValueError("invalid sequence type")
                    expected = _record(self._ledger_id, sequence, self._tip, normalized)
                    if _canonical(row) != _canonical(expected):
                        raise ValueError("record/schema/sequence/hash-chain mismatch")
                    if intent.intent_id in self._by_id:
                        raise ValueError("duplicate intent ID in persisted history")
                    receipt = _receipt(expected)
                    self._by_id[intent.intent_id] = len(self._records)
                    self._records.append((receipt, _canonical(normalized)))
                    self._tip = receipt.record_hash
                except (ValueError, TypeError, KeyError, OverflowError,
                        RecursionError, AttributeError) as exc:
                    raise LedgerCorruptionError(f"invalid ledger line {line_number}") from exc
            if line_number == 0:
                raise LedgerCorruptionError("empty ledger; header is required")

    def _check_checkpoint(self, checkpoint: LedgerCheckpointV1) -> None:
        if checkpoint.sequence > len(self._records):
            raise LedgerCorruptionError("history shorter than trusted checkpoint")
        actual = (_header(self._ledger_id)["record_hash"] if checkpoint.sequence == 0
                  else self._records[checkpoint.sequence - 1][0].record_hash)
        if actual != checkpoint.record_hash:
            raise LedgerCorruptionError("trusted checkpoint hash mismatch")

    def _owner(self) -> None:
        if os.getpid() != self._pid:
            raise LedgerUnavailableError("fork-inherited handle; close it and reopen")

    def _guard(self) -> None:
        if self._fd < 0 or self._poisoned:
            raise LedgerUnavailableError("closed or poisoned ledger; reopen and reconcile")
        try:
            if _stamp(self._identity()) != self._stamp:
                raise LedgerUnavailableError("ledger changed outside this handle")
        except (OSError, LedgerError) as exc:
            self._poisoned = True
            raise LedgerUnavailableError("ledger identity/content changed; stop") from exc

    def append(self, intent: OrderIntent) -> DurableIntentReceiptV1:
        """Idempotently persist an intent; never perform a venue action."""
        self._owner()
        with self._mutex:
            self._guard()
            payload = _intent_payload(intent)
            encoded = _canonical(payload)
            index = self._by_id.get(intent.intent_id)
            if index is not None:
                receipt, existing = self._records[index]
                if encoded != existing:
                    raise IntentConflictError("intent ID already binds different content")
                return receipt
            row = _record(self._ledger_id, len(self._records) + 1, self._tip, payload)
            raw = _canonical(row) + b"\n"
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError("intent record exceeds MAX_LINE_BYTES")
            receipt = _receipt(row)
            size_before = self._stamp[3]
            try:
                _write_all(self._fd, raw)
                os.fsync(self._fd)
                info = self._identity()
                if info.st_size != size_before + len(raw):
                    raise LedgerUnavailableError("unexpected size after append")
                self._by_id[intent.intent_id] = len(self._records)
                self._records.append((receipt, encoded))
                self._tip = receipt.record_hash
                self._stamp = _stamp(info)
                return receipt
            except BaseException as exc:
                self._poisoned = True
                if isinstance(exc, (OSError, LedgerError)):
                    raise LedgerUnavailableError(
                        "append not acknowledged; preserve file, close, reopen, reconcile"
                    ) from exc
                raise

    def get_intent(self, intent_id: str) -> OrderIntent | None:
        self._owner()
        with self._mutex:
            self._guard()
            index = self._by_id.get(_text(intent_id, "intent_id"))
            if index is None:
                return None
            return OrderIntent.from_dict(json.loads(self._records[index][1]))

    def receipts(self) -> tuple[DurableIntentReceiptV1, ...]:
        self._owner()
        with self._mutex:
            self._guard()
            return tuple(receipt for receipt, _ in self._records)

    def checkpoint(self) -> LedgerCheckpointV1:
        self._owner()
        with self._mutex:
            self._guard()
            return LedgerCheckpointV1(self._ledger_id, len(self._records), self._tip)

    def _release(self) -> OSError | None:
        # Close only; never LOCK_UN on a fork-shared open file description.
        first_error = None
        for name in ("_fd", "_dir_fd"):
            fd = getattr(self, name, -1)
            setattr(self, name, -1)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError as exc:
                    if first_error is None:
                        first_error = exc  # Never retry close() on Linux.
        return first_error

    def close(self) -> None:
        if os.getpid() != self._pid:
            error = self._release()
        else:
            with self._mutex:
                error = self._release()  # No deferred writes or repair.
        if error is not None:
            raise LedgerUnavailableError("descriptor close failed; handle is unusable") from error

    def __enter__(self) -> DurableIntentLedgerV1:
        self._owner()
        with self._mutex:
            self._guard()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        try:
            self.close()
        except LedgerUnavailableError:
            if exc is None:
                raise
            exc.add_note("ledger descriptor close also failed")

    def __del__(self) -> None:
        self._release()
