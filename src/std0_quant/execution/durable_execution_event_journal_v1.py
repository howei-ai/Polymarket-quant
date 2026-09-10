"""Linux-local execution-event journal v1. OFFLINE infrastructure only.

A separate append-only NDJSON file for one journal_id and intent ledger_id.
create() alone creates; open_existing() never creates, truncates or repairs.
Every new capture is written and fsynced before a receipt is returned. Recovery
validates the ENTIRE file and fsyncs it and its directory, including any complete
record surviving an unacknowledged write. This does NOT prove that an earlier
caller received a receipt, that an event is authentic, or that an order ran.

capture_id identifies an ingestion attempt, NOT a venue event. Reusing it with
identical normalized observation content is idempotent; changed content raises
CaptureConflictError. Different capture IDs preserve even identical/conflicting
event IDs, receipt-binding problems and out-of-order observations for the frozen
reconciler. Only the receipt ledger_id must match this journal. Actual receipt
membership in a trusted intent checkpoint is checked by reconciliation, NOT by
storage. Invalid contract-shaped inputs are rejected; raw-message quarantine and
network collection are outside this module. Callers must persist/reuse stable
capture IDs for retries; no automatic source acknowledgement is sent here.

Exports contain an EXACT journal checkpoint prefix without sorting/filtering or
deduplication. Verify against an independently trusted journal checkpoint before
converting to ReconciliationEventV1. The intent checkpoint remains a separate
required input to reconcile_execution_v1. A hash is not authentication; without
an independent anchor, valid-suffix deletion or a fully rehashed history cannot
be detected. A prefix checkpoint is not proof of the latest or complete history.

Assumes cooperating writers, trusted durably provisioned parent directories,
Linux LOCAL storage honoring fsync and one authoritative journal per scope.
flock is advisory, not a malicious-writer defense. No NFS, rotation, replication,
power-loss certification, live transport, credentials, portfolio mutation,
reservation release, redispatch, or recovery unlock. Records/indexes are in RAM;
size caps are guardrails, not a scalability guarantee. Do not fork with an open
handle: child operations fail before taking the inherited mutex; child close
closes its descriptors without unlocking the parent's flock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
except ImportError:
    fcntl = None

from std0_quant.execution.contracts import OrderEvent
from std0_quant.execution.durable_intent_ledger_v1 import DurableIntentReceiptV1
from std0_quant.execution.execution_reconciliation_v1 import ReconciliationEventV1

HEADER_SCHEMA = "execution_event_journal_header_v1"
RECORD_SCHEMA = "execution_event_journal_record_v1"
RECEIPT_SCHEMA = "durable_execution_event_receipt_v1"
EXPORT_SCHEMA = "execution_event_journal_export_v1"
OFFLINE = "OFFLINE"
EXPORT_SCOPE = "EXACT_JOURNAL_PREFIX"
MAX_LINE_BYTES = 1024 * 1024
MAX_RECORDS = 500_000
MAX_FILE_BYTES = 512 * 1024 * 1024
_HASH_DOMAIN = b"std0-quant/durable-execution-event-journal/v1/"


class EventJournalError(RuntimeError):
    """Storage failure; no trading action or source acknowledgement implied."""


class EventJournalBusyError(EventJournalError):
    """Another open handle owns the advisory exclusive lock."""


class EventJournalCorruptionError(EventJournalError):
    """Invalid file/export/anchor; preserve evidence, never repair implicitly."""


class EventJournalUnavailableError(EventJournalError):
    """Closed, forked, poisoned, changed, or failed-persistence handle."""


class CaptureConflictError(EventJournalError):
    """One capture_id was reused for different observation bytes."""


def _text(value: Any, name: str, *, limit: int = 256) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be str")
    if not value or value != value.strip() or len(value.encode("utf-8")) > limit:
        raise ValueError(f"{name} must be nonempty, unpadded, <= {limit} UTF-8 bytes")
    return value


def _hash_text(value: Any) -> str:
    value = _text(value, "hash")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("hash must be lowercase SHA256")
    return value


def _sequence(value: Any, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise ValueError("sequence must be a nonnegative/positive integer, not bool")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(kind: str, payload: Any) -> str:
    return hashlib.sha256(_HASH_DOMAIN + kind.encode("ascii") + b"\n"
                          + _canonical(payload)).hexdigest()


def _observation(value: ReconciliationEventV1) -> ReconciliationEventV1:
    if type(value) is not ReconciliationEventV1:
        raise TypeError("observation must be exactly ReconciliationEventV1")
    # Revalidate/copy through the frozen public contract, not private helpers.
    result = ReconciliationEventV1(value.receipt, value.event, value.mode, value.schema_version)
    if _canonical(asdict(result)) != _canonical(asdict(value)):
        raise ValueError("observation is not normalized")
    return result


def _observation_from_dict(row: Any) -> ReconciliationEventV1:
    if type(row) is not dict or set(row) != {"receipt", "event", "mode", "schema_version"}:
        raise ValueError("observation fields mismatch")
    result = ReconciliationEventV1(DurableIntentReceiptV1(**row["receipt"]),
                                   OrderEvent.from_dict(row["event"]),
                                   row["mode"], row["schema_version"])
    if _canonical(asdict(result)) != _canonical(row):
        raise ValueError("persisted observation is not normalized or has missing fields")
    return result


@dataclass(frozen=True)
class ExecutionEventJournalCheckpointV1:
    """Required historical prefix; authenticity and latestness remain external."""

    journal_id: str
    ledger_id: str
    sequence: int
    record_hash: str

    def __post_init__(self) -> None:
        _text(self.journal_id, "journal_id")
        _text(self.ledger_id, "ledger_id")
        _sequence(self.sequence)
        _hash_text(self.record_hash)


def _checkpoint(value: ExecutionEventJournalCheckpointV1) -> ExecutionEventJournalCheckpointV1:
    if type(value) is not ExecutionEventJournalCheckpointV1:
        raise TypeError("checkpoint must be exactly ExecutionEventJournalCheckpointV1")
    return ExecutionEventJournalCheckpointV1(**asdict(value))


@dataclass(frozen=True)
class DurableExecutionEventReceiptV1:
    journal_id: str
    ledger_id: str
    sequence: int
    capture_id: str
    observation_hash: str
    record_hash: str
    schema_version: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _text(self.journal_id, "journal_id")
        _text(self.ledger_id, "ledger_id")
        _text(self.capture_id, "capture_id")
        _sequence(self.sequence, positive=True)
        _hash_text(self.observation_hash)
        _hash_text(self.record_hash)
        if self.schema_version != RECEIPT_SCHEMA:
            raise ValueError("unsupported event receipt schema")


@dataclass(frozen=True)
class ExecutionEventJournalEntryV1:
    receipt: DurableExecutionEventReceiptV1
    observation: ReconciliationEventV1

    def __post_init__(self) -> None:
        if type(self.receipt) is not DurableExecutionEventReceiptV1:
            raise TypeError("entry receipt must be DurableExecutionEventReceiptV1")
        object.__setattr__(self, "receipt", DurableExecutionEventReceiptV1(**asdict(self.receipt)))
        object.__setattr__(self, "observation", _observation(self.observation))


@dataclass(frozen=True)
class ExecutionEventJournalExportV1:
    checkpoint: ExecutionEventJournalCheckpointV1
    entries: tuple[ExecutionEventJournalEntryV1, ...]
    artifact_hash: str
    mode: str = OFFLINE
    scope: str = EXPORT_SCOPE
    schema_version: str = EXPORT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", _checkpoint(self.checkpoint))
        if type(self.entries) is not tuple or len(self.entries) > MAX_RECORDS:
            raise ValueError("export entries must be a tuple within MAX_RECORDS")
        if any(type(entry) is not ExecutionEventJournalEntryV1 for entry in self.entries):
            raise TypeError("export requires ExecutionEventJournalEntryV1 entries")
        object.__setattr__(self, "entries", tuple(
            ExecutionEventJournalEntryV1(entry.receipt, entry.observation) for entry in self.entries))
        _hash_text(self.artifact_hash)
        if self.mode != OFFLINE or self.scope != EXPORT_SCOPE or self.schema_version != EXPORT_SCHEMA:
            raise ValueError("unsupported event export scope/schema")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _canonical(self.to_dict()).decode("utf-8")


def _header(journal_id: str, ledger_id: str) -> dict[str, Any]:
    payload = {"schema_version": HEADER_SCHEMA, "mode": OFFLINE,
               "journal_id": journal_id, "ledger_id": ledger_id}
    return {**payload, "record_hash": _digest("header", payload)}


def _record(journal_id: str, ledger_id: str, sequence: int, previous: str,
            capture_id: str, observation: dict[str, Any]) -> dict[str, Any]:
    payload = {"schema_version": RECORD_SCHEMA, "journal_id": journal_id,
               "ledger_id": ledger_id, "sequence": sequence, "previous_hash": previous,
               "capture_id": capture_id, "observation": observation,
               "observation_hash": _digest("observation", observation)}
    return {**payload, "record_hash": _digest("record", payload)}


def _receipt(row: dict[str, Any]) -> DurableExecutionEventReceiptV1:
    return DurableExecutionEventReceiptV1(row["journal_id"], row["ledger_id"], row["sequence"],
                                          row["capture_id"], row["observation_hash"], row["record_hash"])


def _entry(raw: bytes) -> ExecutionEventJournalEntryV1:
    row = json.loads(raw)
    return ExecutionEventJournalEntryV1(_receipt(row), _observation_from_dict(row["observation"]))


def execution_event_journal_export_v1_hash(export: ExecutionEventJournalExportV1) -> str:
    if type(export) is not ExecutionEventJournalExportV1:
        raise TypeError("export must be exactly ExecutionEventJournalExportV1")
    payload = asdict(export)
    del payload["artifact_hash"]
    return _digest("export", payload)


def verify_execution_event_journal_export_v1(
    export: ExecutionEventJournalExportV1, *, checkpoint: ExecutionEventJournalCheckpointV1,
) -> tuple[str, ...]:
    """Rebuild the full prefix and compare with an independently trusted anchor."""
    if type(export) is not ExecutionEventJournalExportV1:
        raise TypeError("export must be exactly ExecutionEventJournalExportV1")
    anchor = _checkpoint(checkpoint)
    reasons = []
    if execution_event_journal_export_v1_hash(export) != export.artifact_hash:
        reasons.append("EVENT_JOURNAL_EXPORT_V1_HASH_MISMATCH")
    try:
        clean = ExecutionEventJournalExportV1(export.checkpoint, export.entries, export.artifact_hash,
                                             export.mode, export.scope, export.schema_version)
        if clean.checkpoint != anchor or len(clean.entries) != anchor.sequence:
            raise ValueError("exact checkpoint prefix mismatch")
        header = _header(anchor.journal_id, anchor.ledger_id)
        previous = header["record_hash"]
        total = len(_canonical(header)) + 1
        captures: set[str] = set()
        for sequence, entry in enumerate(clean.entries, 1):
            receipt, observation = entry.receipt, entry.observation
            if receipt.capture_id in captures or observation.receipt.ledger_id != anchor.ledger_id:
                raise ValueError("capture duplication or different intent ledger")
            row = _record(anchor.journal_id, anchor.ledger_id, sequence, previous,
                          receipt.capture_id, asdict(observation))
            length = len(_canonical(row)) + 1
            total += length
            if receipt != _receipt(row) or length > MAX_LINE_BYTES or total > MAX_FILE_BYTES:
                raise ValueError("entry receipt/hash/sequence/size mismatch")
            previous = row["record_hash"]
            captures.add(receipt.capture_id)
        if previous != anchor.record_hash or _canonical(asdict(clean)) != _canonical(asdict(export)):
            raise ValueError("history does not reach the trusted checkpoint")
    except (TypeError, ValueError, KeyError, OverflowError, AttributeError):
        reasons.append("EVENT_JOURNAL_EXPORT_V1_PREFIX_MISMATCH")
    return tuple(reasons)


def execution_events_from_journal_export_v1(
    export: ExecutionEventJournalExportV1, *, checkpoint: ExecutionEventJournalCheckpointV1,
) -> tuple[ReconciliationEventV1, ...]:
    """Verified exact prefix, retaining sequence and duplicate/conflicting events."""
    if type(export) is not ExecutionEventJournalExportV1:
        raise TypeError("export must be exactly ExecutionEventJournalExportV1")
    # Do not verify caller-owned objects and then consume a possibly changed copy.
    selected = ExecutionEventJournalExportV1(export.checkpoint, export.entries, export.artifact_hash,
                                            export.mode, export.scope, export.schema_version)
    reasons = verify_execution_event_journal_export_v1(selected, checkpoint=checkpoint)
    if reasons:
        raise EventJournalCorruptionError("; ".join(reasons))
    return tuple(_observation(entry.observation) for entry in selected.entries)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: " + value)


def _decode_line(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
        raise ValueError("oversized or incomplete event journal line")
    row = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                     parse_constant=_reject_constant)
    if type(row) is not dict or _canonical(row) + b"\n" != raw:
        raise ValueError("noncanonical event journal line")
    return row


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written <= 0 or written > len(view):
            raise OSError(errno.EIO, "write made invalid progress")
        view = view[written:]


def _read_exact(fd: int, size: int, offset: int) -> bytes:
    chunks = []
    while size:
        try:
            chunk = os.pread(fd, size, offset)
        except InterruptedError:
            continue
        if not chunk:
            raise OSError(errno.EIO, "unexpected EOF during write verification")
        chunks.append(chunk)
        offset += len(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


class DurableExecutionEventJournalV1:
    """One exclusive handle; explicit close or use a context manager."""

    def __init__(self) -> None:
        raise TypeError("use create() or open_existing()")

    @classmethod
    def create(cls, path: Path | str, *, journal_id: str, ledger_id: str) -> DurableExecutionEventJournalV1:
        return cls._open(path, journal_id=journal_id, ledger_id=ledger_id, create=True, checkpoint=None)

    @classmethod
    def open_existing(
        cls, path: Path | str, *, journal_id: str, ledger_id: str,
        checkpoint: ExecutionEventJournalCheckpointV1 | None = None,
    ) -> DurableExecutionEventJournalV1:
        return cls._open(path, journal_id=journal_id, ledger_id=ledger_id, create=False, checkpoint=checkpoint)

    @classmethod
    def _open(cls, path: Path | str, *, journal_id: str, ledger_id: str, create: bool,
              checkpoint: ExecutionEventJournalCheckpointV1 | None) -> DurableExecutionEventJournalV1:
        if not sys.platform.startswith("linux") or fcntl is None:
            raise EventJournalUnavailableError("Linux local storage only")
        _text(journal_id, "journal_id")
        _text(ledger_id, "ledger_id")
        anchor = None if checkpoint is None else _checkpoint(checkpoint)
        if anchor is not None and (anchor.journal_id, anchor.ledger_id) != (journal_id, ledger_id):
            raise EventJournalCorruptionError("checkpoint journal/ledger identity mismatch")
        self = cls.__new__(cls)
        self._fd = self._dir_fd = -1
        self._pid = os.getpid()
        self._mutex = threading.RLock()
        self._poisoned = False
        self._path = Path(os.path.abspath(os.fspath(path)))
        self._journal_id, self._ledger_id = journal_id, ledger_id
        self._records: list[bytes] = []  # Immutable bytes; callers never share internal objects.
        self._by_capture: dict[str, int] = {}
        self._tip = _header(journal_id, ledger_id)["record_hash"]
        try:
            self._dir_fd = os.open(self._path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
            if create:
                flags |= os.O_CREAT | os.O_EXCL
            self._fd = os.open(self._path.name, flags, 0o600, dir_fd=self._dir_fd)
            self._identity()
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise EventJournalBusyError("journal already locked") from exc
            info = self._identity()
            if info.st_size > MAX_FILE_BYTES:
                raise EventJournalCorruptionError("journal exceeds MAX_FILE_BYTES")
            if create:
                raw = _canonical(_header(journal_id, ledger_id)) + b"\n"
                _write_all(self._fd, raw)
                before = _stamp(self._identity())
                if before[3] != len(raw) or _read_exact(self._fd, len(raw), 0) != raw:
                    raise EventJournalCorruptionError("header write verification failed")
            else:
                before = _stamp(info)
                self._recover()
                if anchor is not None:
                    self._check_checkpoint(anchor)
            if _stamp(self._identity()) != before:
                raise EventJournalCorruptionError("journal changed during initialization validation")
            os.fsync(self._fd)
            os.fsync(self._dir_fd)
            if _stamp(self._identity()) != before:
                raise EventJournalCorruptionError("journal changed during initialization sync")
            self._stamp = before  # Never adopt an unvalidated post-sync snapshot.
            return self
        except BaseException:
            self._release()
            raise

    def _identity(self) -> os.stat_result:
        info = os.fstat(self._fd)
        named = os.stat(self._path.name, dir_fd=self._dir_fd, follow_symlinks=False)
        parent, pinned = os.stat(self._path.parent), os.fstat(self._dir_fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or not stat.S_ISREG(named.st_mode)
                or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
                or (parent.st_dev, parent.st_ino) != (pinned.st_dev, pinned.st_ino)):
            raise EventJournalUnavailableError("journal path/inode/link identity changed")
        return info

    def _recover(self) -> None:
        os.lseek(self._fd, 0, os.SEEK_SET)
        total, line_number = 0, 0
        with os.fdopen(os.dup(self._fd), "rb") as reader:
            while True:
                raw = reader.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                total += len(raw)
                line_number += 1
                try:
                    if total > MAX_FILE_BYTES:
                        raise ValueError("journal exceeds MAX_FILE_BYTES")
                    row = _decode_line(raw)
                    if line_number == 1:
                        if row != _header(self._journal_id, self._ledger_id):
                            raise ValueError("header/schema/identity/hash mismatch")
                        continue
                    if len(self._records) >= MAX_RECORDS:
                        raise ValueError("journal exceeds MAX_RECORDS")
                    capture_id = _text(row.get("capture_id"), "capture_id")
                    if capture_id in self._by_capture:
                        raise ValueError("duplicate persisted capture_id")
                    observation = _observation_from_dict(row.get("observation"))
                    if observation.receipt.ledger_id != self._ledger_id:
                        raise ValueError("observation from another intent ledger")
                    _sequence(row.get("sequence"), positive=True)
                    expected = _record(self._journal_id, self._ledger_id, len(self._records) + 1,
                                       self._tip, capture_id, asdict(observation))
                    if _canonical(row) != _canonical(expected):
                        raise ValueError("record/schema/sequence/hash chain mismatch")
                    self._by_capture[capture_id] = len(self._records)
                    self._records.append(raw)
                    self._tip = row["record_hash"]
                except (ValueError, TypeError, KeyError, OverflowError, RecursionError, AttributeError) as exc:
                    raise EventJournalCorruptionError(f"invalid event journal line {line_number}") from exc
        if line_number == 0:
            raise EventJournalCorruptionError("empty journal; header is required")

    def _check_checkpoint(self, checkpoint: ExecutionEventJournalCheckpointV1) -> None:
        if ((checkpoint.journal_id, checkpoint.ledger_id) != (self._journal_id, self._ledger_id)
                or checkpoint.sequence > len(self._records)):
            raise EventJournalCorruptionError("checkpoint outside this journal history")
        actual = (_header(self._journal_id, self._ledger_id)["record_hash"] if checkpoint.sequence == 0
                  else json.loads(self._records[checkpoint.sequence - 1])["record_hash"])
        if actual != checkpoint.record_hash:
            raise EventJournalCorruptionError("trusted journal checkpoint hash mismatch")

    def _owner(self) -> None:
        if os.getpid() != self._pid:
            raise EventJournalUnavailableError("fork-inherited handle; close and reopen")

    def _guard(self) -> None:
        if self._fd < 0 or self._poisoned:
            raise EventJournalUnavailableError("closed or poisoned journal; preserve file and reopen explicitly")
        try:
            if _stamp(self._identity()) != self._stamp:
                raise EventJournalUnavailableError("journal changed outside this handle")
        except (OSError, EventJournalError) as exc:
            self._poisoned = True
            raise EventJournalUnavailableError("journal identity/content changed; stop") from exc

    def append(self, capture_id: str, observation: ReconciliationEventV1) -> DurableExecutionEventReceiptV1:
        """Persist one capture. A receipt attests only local storage, not event truth."""
        self._owner()
        with self._mutex:
            self._guard()
            capture_id = _text(capture_id, "capture_id")
            value = _observation(observation)
            if value.receipt.ledger_id != self._ledger_id:
                raise ValueError("observation belongs to another intent ledger")
            payload = asdict(value)
            index = self._by_capture.get(capture_id)
            if index is not None:
                previous = json.loads(self._records[index])
                if _canonical(previous["observation"]) != _canonical(payload):
                    raise CaptureConflictError("capture_id already binds different normalized content")
                return _receipt(previous)
            row = _record(self._journal_id, self._ledger_id, len(self._records) + 1,
                          self._tip, capture_id, payload)
            raw = _canonical(row) + b"\n"
            if len(raw) > MAX_LINE_BYTES or len(self._records) >= MAX_RECORDS:
                raise ValueError("journal MAX_LINE_BYTES/MAX_RECORDS exceeded")
            old_size = self._stamp[3]
            if old_size + len(raw) > MAX_FILE_BYTES:
                raise ValueError("journal MAX_FILE_BYTES exceeded")
            receipt = _receipt(row)
            self._guard()
            try:
                _write_all(self._fd, raw)
                validated = _stamp(self._identity())
                if (validated[3] != old_size + len(raw)
                        or _read_exact(self._fd, len(raw), old_size) != raw
                        or _stamp(self._identity()) != validated):
                    raise EventJournalUnavailableError("append write verification failed")
                os.fsync(self._fd)
                if _stamp(self._identity()) != validated:
                    raise EventJournalUnavailableError("journal changed during append sync")
                self._by_capture[capture_id] = len(self._records)
                self._records.append(raw)
                self._tip, self._stamp = receipt.record_hash, validated
                return receipt
            except BaseException as exc:
                self._poisoned = True
                if isinstance(exc, (OSError, EventJournalError)):
                    raise EventJournalUnavailableError(
                        "append not acknowledged; preserve journal, close, reopen, inspect"
                    ) from exc
                raise

    def checkpoint(self) -> ExecutionEventJournalCheckpointV1:
        self._owner()
        with self._mutex:
            self._guard()
            return ExecutionEventJournalCheckpointV1(self._journal_id, self._ledger_id,
                                                      len(self._records), self._tip)

    def get_capture(self, capture_id: str) -> ExecutionEventJournalEntryV1 | None:
        self._owner()
        with self._mutex:
            self._guard()
            index = self._by_capture.get(_text(capture_id, "capture_id"))
            result = None if index is None else _entry(self._records[index])
            self._guard()
            return result

    def export(self, *, checkpoint: ExecutionEventJournalCheckpointV1) -> ExecutionEventJournalExportV1:
        """Copy a validated exact prefix, no event/time filtering or deduplication."""
        self._owner()
        with self._mutex:
            self._guard()
            anchor = _checkpoint(checkpoint)
            self._check_checkpoint(anchor)
            result = ExecutionEventJournalExportV1(anchor, tuple(
                _entry(raw) for raw in self._records[:anchor.sequence]), "0" * 64)
            result = replace(result, artifact_hash=execution_event_journal_export_v1_hash(result))
            reasons = verify_execution_event_journal_export_v1(result, checkpoint=anchor)
            if reasons:
                self._poisoned = True
                raise EventJournalCorruptionError("; ".join(reasons))
            self._guard()
            return result

    def _release(self) -> OSError | None:
        first_error = None
        for name in ("_fd", "_dir_fd"):
            fd = getattr(self, name, -1)
            setattr(self, name, -1)
            if fd >= 0:
                try:
                    os.close(fd)  # No retry and no explicit LOCK_UN on fork-shared descriptors.
                except OSError as exc:
                    if first_error is None:
                        first_error = exc
        return first_error

    def close(self) -> None:
        if os.getpid() != self._pid:
            error = self._release()
        else:
            with self._mutex:
                error = self._release()
        if error is not None:
            raise EventJournalUnavailableError("descriptor close failed; handle unusable") from error

    def __enter__(self) -> DurableExecutionEventJournalV1:
        self._owner()
        with self._mutex:
            self._guard()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        try:
            self.close()
        except EventJournalUnavailableError:
            if exc is None:
                raise
            exc.add_note("event journal descriptor close also failed")

    def __del__(self) -> None:
        self._release()
