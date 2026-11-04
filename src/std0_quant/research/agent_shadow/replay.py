"""Offline replay of frozen pretrade feature rows through ShadowRunner (SHADOW only).

Inputs are the frozen prospective-v4 feature and provenance parquet files, read only and
verified against pinned SHA-256 values and row counts (the bytes that were hashed are the
bytes that are parsed). Each cutoff_1 row is projected to the point-in-time allowlist
(labels, fills and t0-known state never reach a model), its observed timestamps are derived
from the provenance file, and its GateEvidence is derived fail-closed: any check that cannot
be established is False, which blocks the row.

Rows are redacted before they reach a model, so a model cannot recognise the real market
or minute: the condition id becomes a one-way hash, every timestamp is shifted so the
decision cutoff sits at a fixed synthetic epoch (relative timing is kept), and absolute BTC
price levels are dropped (relative BTC features are kept). The real identifiers stay only in
a local mapping file next to the artifacts.

Outputs are append-only SHADOW artifacts outside the repository and outside the inputs.
Nothing here writes the formal cohort, frozen artifacts or data/, submits orders, publishes,
or runs a backtest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ...audit.sanity_audit_v2 import sanity_audit_v2
from ...features.provenance import PUBLIC_SOURCES, validate_provenance
from .audit import canonical_json
from .context import FEATURE_ALLOWLIST
from .model_registry import CostBudget
from .runner import ShadowRunner
from .schemas import GateEvidence, ShadowResult


REPLAY_VERSION = "agent_shadow_replay_v2"
FROZEN_RUN_SUFFIX = "prospective-v4-lowmem-20260829T073235Z.parquet"
# File names inside the frozen features directory, pinned SHA-256 (same pins as
# scripts/compare_sanity_audit_v2.py) and row counts. Any mismatch refuses the replay.
FROZEN_INPUTS = MappingProxyType({
    "features": ("pretrade_features_" + FROZEN_RUN_SUFFIX,
                 "d7562ebd0c06df2b7441f65264b09d6971dbc5f87d965fbd08d8ced8a1a8aa3c", 463),
    "provenance": ("feature_provenance_" + FROZEN_RUN_SUFFIX,
                   "347ff2d7ea282bb9454a0379f243080c0ed83a39620de46d612449acb29a0bec", 36577),
})
OBSERVED_COLUMNS = MappingProxyType({"binance_btc": "btc_observed_ts_ms", "polymarket_book": "book_observed_ts_ms"})
COVERAGE_THRESHOLD = 0.99
# Redaction: the decision cutoff of every row sent to a model sits at this synthetic epoch.
REDACTED_CUTOFF_MS = 1_000_000_000_000
REDACTION_SALT = "std0-agent-shadow-replay-v2"
# Absolute price levels date a row; relative BTC features (returns, distance, volatility) stay.
REDACTED_FEATURES = frozenset({"btc_last_price", "btc_start_price"})
ARTIFACTS_FILE = "artifacts.jsonl"
REDACTION_MAP_FILE = "redaction_map.jsonl"
# Call outcomes and gate reasons that mean the row should be attempted again, not counted as done.
RETRYABLE_OUTCOMES = frozenset({"TIMEOUT", "API_ERROR", "BUDGET_EXCEEDED"})
RETRYABLE_REASONS = frozenset({"K3_TIMEOUT", "JEV_TIMEOUT", "BUDGET_OVERRUN"})
# Reasons that mean the setup itself is broken: stop the run instead of writing more BLOCKED rows.
STOP_REASONS = frozenset({"MODEL_PROVENANCE_INVALID", "RELAY_NOT_ALLOWED", "MODEL_ROUTE_MIXED", "TEST_CLIENT_MIXED"})
# Budget headroom required before a row starts (about one row's worst-case reservations).
DEFAULT_ROW_HEADROOM_USD = 1.5
# Attempts after which a row that never reached a final result is no longer retried.
DEFAULT_MAX_ATTEMPTS = 3


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_frozen_inputs(directory: Path, pins: Mapping[str, tuple] = FROZEN_INPUTS) -> dict[str, dict[str, Any]]:
    """{name: {"path", "sha256", "records"}}; reads each file once, hashes and parses the same bytes.

    pins: name -> (file name, sha256, expected row count or None).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    loaded: dict[str, dict[str, Any]] = {}
    for name, (file_name, expected_sha, expected_rows) in pins.items():
        path = Path(directory) / file_name
        if not path.is_file():
            raise ValueError(f"frozen input {name} not found: {path}")
        raw = path.read_bytes()
        actual = sha256_bytes(raw)
        if actual != expected_sha:
            raise ValueError(f"frozen input {name} sha256 {actual} != pinned {expected_sha}")
        records = pq.read_table(pa.BufferReader(raw)).to_pylist()
        if expected_rows is not None and len(records) != expected_rows:
            raise ValueError(f"frozen input {name} has {len(records)} rows, pinned {expected_rows}")
        loaded[name] = {"path": path, "sha256": actual, "records": records}
    return loaded


def _plain(value: Any) -> Any:
    """JSON-safe scalar: non-finite floats and missing values become None."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    return _plain(item()) if callable(item) else value


def redacted_condition_id(condition_id: str) -> str:
    return "row_" + hashlib.sha256(f"{REDACTION_SALT}|{condition_id}".encode()).hexdigest()[:24]


@dataclass(frozen=True)
class ReplayRow:
    condition_id: str  # real id, local only
    redacted_id: str
    pit_row: dict[str, Any]  # redacted; this is what a model sees
    evidence: GateEvidence
    evidence_failures: tuple[str, ...] = field(default=())
    real_cutoff_ms: int = 0


@dataclass(frozen=True)
class ReplayPlan:
    rows: tuple[ReplayRow, ...]
    dropped: Mapping[str, int]


def _observed(provenance: list[dict[str, Any]], cutoff: int) -> dict[str, int | None]:
    observed: dict[str, int | None] = {}
    for source_type, column in OBSERVED_COLUMNS.items():
        maxima = [row["source_timestamp_max_ms"] for row in provenance
                  if row.get("source_type") == source_type and row.get("feature_cutoff_ms") == cutoff
                  and row.get("feature_name") in FEATURE_ALLOWLIST
                  and type(row.get("source_timestamp_max_ms")) is int]
        observed[column] = max(maxima) if maxima else None
    return observed


def _evidence(record: dict[str, Any], pit_row: dict[str, Any], provenance: list[dict[str, Any]],
              cutoff: int) -> tuple[GateEvidence, tuple[str, ...]]:
    """Evidence on the real (unredacted) row. Sanity sees the raw record, so NaN/inf warn."""
    failures: list[str] = []
    coverage = (record.get("model_eligible") is True and all(
        type(pit_row.get(name)) in (int, float) and COVERAGE_THRESHOLD <= pit_row[name] <= 1.0
        for name in ("btc_pre30_coverage_pct", "book_pre10_coverage_pct")))
    if not coverage:
        failures.append("COVERAGE")
    public = [row for row in provenance if row.get("source_type") in PUBLIC_SOURCES]
    covered = {row.get("feature_name") for row in public
               if row.get("feature_cutoff_ms") == cutoff and type(row.get("source_timestamp_max_ms")) is int
               and not row.get("missing_reason")}
    sent = {name for name in FEATURE_ALLOWLIST if pit_row.get(name) is not None}
    try:
        provenance_ok = bool(public) and sent <= covered and validate_provenance(public) is True
    except (AssertionError, KeyError, TypeError):
        provenance_ok = False
    if not provenance_ok:
        failures.append("PROVENANCE")
    try:
        sanity_ok = sanity_audit_v2([dict(record)])["status"] == "PASS"
    except (TypeError, ValueError):
        sanity_ok = False
    if not sanity_ok:
        failures.append("SANITY")
    return GateEvidence(coverage, provenance_ok, sanity_ok), tuple(failures)


def _redact(pit_row: dict[str, Any], redacted_id: str) -> dict[str, Any]:
    shift = REDACTED_CUTOFF_MS - pit_row["decision_cutoff_ts_ms"]
    redacted = {name: value for name, value in pit_row.items() if name not in REDACTED_FEATURES}
    redacted["condition_id"] = redacted_id
    for name in ("decision_cutoff_ts_ms", "btc_observed_ts_ms", "book_observed_ts_ms"):
        if type(pit_row.get(name)) is int:
            redacted[name] = pit_row[name] + shift
    return redacted


def build_replay_plan(features: Iterable[Mapping[str, Any]], provenance: Iterable[Mapping[str, Any]], *,
                      eligible_only: bool = True) -> ReplayPlan:
    """cutoff_1 rows in (real cutoff, condition_id) order, projected to the PIT allowlist and redacted.

    Only the condition id, the decision cutoff, the derived observed timestamps and allowlisted
    features are kept, so labels (y30), fills (first_opp_*, fill_minus_prev_*), t0-known state
    and every other column never reach a model. Rows that cannot be read are counted in dropped.
    """
    by_condition: dict[Any, list[dict[str, Any]]] = {}
    for row in provenance:
        by_condition.setdefault(row.get("condition_id"), []).append(dict(row))
    dropped: Counter[str] = Counter()
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for record in features:
        record = dict(record)
        if record.get("cutoff_mode") != "cutoff_1":
            dropped["NOT_CUTOFF_1"] += 1
            continue
        if eligible_only and record.get("model_eligible") is not True:
            dropped["NOT_MODEL_ELIGIBLE"] += 1
            continue
        cid, cutoff = record.get("condition_id"), _plain(record.get("feature_cutoff_ms"))
        if not isinstance(cid, str) or not cid.strip():
            dropped["BAD_CONDITION_ID"] += 1
            continue
        if type(cutoff) is not int or cutoff <= 0:
            dropped["BAD_CUTOFF"] += 1
            continue
        candidates.append((cutoff, cid, record))
    counts = Counter(cid for _, cid, _ in candidates)
    duplicates = {cid for cid, count in counts.items() if count > 1}
    if duplicates:  # one redacted id per condition: ambiguous rows are never replayed
        dropped["DUPLICATE_CONDITION_ID"] += sum(counts[cid] for cid in duplicates)
        candidates = [item for item in candidates if item[1] not in duplicates]
    candidates.sort(key=lambda item: (item[0], item[1]))
    rows: list[ReplayRow] = []
    for cutoff, cid, record in candidates:
        prov = by_condition.get(cid, [])
        pit_row: dict[str, Any] = {"condition_id": cid, "decision_cutoff_ts_ms": cutoff, **_observed(prov, cutoff)}
        for name in sorted(FEATURE_ALLOWLIST & record.keys()):
            pit_row[name] = _plain(record[name])
        evidence, failures = _evidence(record, pit_row, prov, cutoff)
        redacted_id = redacted_condition_id(cid)
        rows.append(ReplayRow(cid, redacted_id, _redact(pit_row, redacted_id), evidence, failures, cutoff))
    return ReplayPlan(tuple(rows), MappingProxyType(dict(dropped)))


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def is_final(payload: Mapping[str, Any]) -> bool:
    """A written attempt counts as done unless a retryable failure or a broken setup blocked it."""
    reasons = set(payload.get("gate_reasons") or ())
    outcomes = {call.get("outcome") for call in payload.get("model_calls") or ()}
    return not (reasons & (RETRYABLE_REASONS | STOP_REASONS) or outcomes & RETRYABLE_OUTCOMES)


class ReplayWriter:
    """Append-only SHADOW artifacts in an output directory outside every forbidden root."""

    def __init__(self, out_dir: Path, *, forbidden_roots: Iterable[Path]) -> None:
        resolved = Path(out_dir).expanduser().resolve()
        for root in forbidden_roots:
            root = Path(root).expanduser().resolve()
            if _inside(resolved, root) or _inside(root, resolved):
                raise ValueError(f"replay output must be outside (and not contain) {root}")
        resolved.mkdir(parents=True, exist_ok=True)
        self.out_dir = resolved
        self.artifacts = resolved / ARTIFACTS_FILE
        self.redaction_map = resolved / REDACTION_MAP_FILE
        for path in (self.artifacts, self.redaction_map):
            if path.is_symlink():
                raise ValueError(f"{path} must not be a symlink")

    def done_ids(self) -> set[str]:
        """Redacted ids with a final attempt; retryable attempts stay eligible for another run."""
        return self.attempt_state()[0]

    def attempt_state(self) -> tuple[set[str], Counter[str]]:
        """(ids with a final attempt, number of non-final attempts per id)."""
        done: set[str] = set()
        attempts: Counter[str] = Counter()
        if not self.artifacts.exists():
            return done, attempts
        with open(self.artifacts, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    condition_id = payload["condition_id"]
                except (ValueError, KeyError, TypeError):
                    raise ValueError(f"{self.artifacts} line {number} is not a replay artifact") from None
                if is_final(payload):
                    done.add(condition_id)
                else:
                    attempts[condition_id] += 1
        return done, attempts

    @staticmethod
    def _append(path: Path, payload: Mapping[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, row: ReplayRow, result: ShadowResult) -> str:
        if not isinstance(result, ShadowResult) or result.execution_allowed or result.formal_cohort_effect:
            raise ValueError("only SHADOW results without execution or cohort effect can be written")
        if result.condition_id != row.redacted_id:
            raise ValueError("artifact must carry the redacted id of its row")
        payload = result.to_dict()
        self._append(self.redaction_map, {"redacted_id": row.redacted_id, "condition_id": row.condition_id,
                                          "real_cutoff_ms": row.real_cutoff_ms,
                                          "artifact_sha256": payload["artifact_sha256"]})
        self._append(self.artifacts, payload)
        return payload["artifact_sha256"]

    def write_once(self, prefix: str, payload: Mapping[str, Any]) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.out_dir / f"{prefix}_{stamp}.json"
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path


def run_replay(rows: Iterable[ReplayRow], runner: ShadowRunner, writer: ReplayWriter, budget: CostBudget, *,
               max_rows: int, row_headroom_usd: float = DEFAULT_ROW_HEADROOM_USD,
               max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> dict[str, Any]:
    """Attempts up to max_rows rows without a final attempt yet.

    Stops before a row when the remaining budget is below row_headroom_usd. Every attempt is
    written for audit; retryable ones (timeouts, API errors, a row the budget cut short) stay
    eligible until max_attempts non-final attempts. A budget cut or a broken setup (e.g.
    MODEL_PROVENANCE_INVALID) stops the run.
    """
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if type(row_headroom_usd) not in (int, float) or not math.isfinite(row_headroom_usd) or row_headroom_usd < 0:
        raise ValueError("row_headroom_usd must be finite and nonnegative")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    done, attempts = writer.attempt_state()
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    attempted = written = skipped = retryable = exhausted = 0
    stop = "INPUT_EXHAUSTED"
    for row in rows:
        if row.redacted_id in done:
            skipped += 1
            continue
        if attempts[row.redacted_id] >= max_attempts:
            exhausted += 1
            continue
        if attempted >= max_rows:
            stop = "MAX_ROWS"
            break
        if budget.locked or budget.limit_usd - budget.spent_usd - budget.held_usd < row_headroom_usd:
            stop = "BUDGET"
            break
        result = runner.run(row.pit_row, row.evidence)
        attempted += 1
        payload_reasons = set(result.gate_reasons)
        budget_cut = budget.locked or "BUDGET_OVERRUN" in payload_reasons or any(
            call.outcome == "BUDGET_EXCEEDED" for call in result.model_calls)
        writer.append(row, result)  # a budget-cut attempt is non-final (is_final) and retried
        written += 1
        statuses[result.gate_status] += 1
        reasons.update(result.gate_reasons)
        if not is_final(result.to_dict()) or budget_cut:
            retryable += 1
        if budget_cut:
            stop = "BUDGET"
            break
        if payload_reasons & STOP_REASONS:
            stop = "SETUP_INVALID"
            break
    return {
        "replay_version": REPLAY_VERSION, "attempted": attempted, "written": written,
        "written_retryable": retryable, "skipped_already_done": skipped,
        "skipped_attempts_exhausted": exhausted, "stop_reason": stop,
        "gate_status_counts": dict(statuses), "gate_reason_counts": dict(reasons),
        "budget": {"limit_usd": budget.limit_usd, "spent_usd": budget.spent_usd, "held_usd": budget.held_usd,
                   "locked": budget.locked},
    }
