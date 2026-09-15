"""Feature provenance repair v2 with declared-source membership binding."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

PROVENANCE_CONTRACT_VERSION = "publication_provenance_v2"
PHASE1_SOURCE_TYPE = "phase1_truth"
FULL_ARTIFACT = "FULL_ARTIFACT"
CANDIDATE_SUBSET = "CANDIDATE_SUBSET"


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class FullArtifactProvenanceExpectations:
    """Pinned counts for the complete feature-provenance artifact."""

    total_rows: int
    phase1_rows: int
    phase1_unique_conditions: int

    def __post_init__(self) -> None:
        for name in (
            "total_rows",
            "phase1_rows",
            "phase1_unique_conditions",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        if self.phase1_rows > self.total_rows:
            raise ValueError("phase1_rows cannot exceed total_rows")
        if self.phase1_unique_conditions > self.phase1_rows:
            raise ValueError(
                "phase1_unique_conditions cannot exceed phase1_rows"
            )


@dataclass(frozen=True)
class CandidateSubsetProvenanceExpectations:
    """Pinned counts for an explicitly selected candidate subset."""

    phase1_rows: int
    phase1_unique_conditions: int

    def __post_init__(self) -> None:
        for name in ("phase1_rows", "phase1_unique_conditions"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        if self.phase1_unique_conditions > self.phase1_rows:
            raise ValueError(
                "phase1_unique_conditions cannot exceed phase1_rows"
            )


CURRENT_CANDIDATE_SUBSET_EXPECTATIONS = (
    CandidateSubsetProvenanceExpectations(
        phase1_rows=1376,
        phase1_unique_conditions=86,
    )
)

# Complete-artifact data pins for the feature-provenance artifact whose SHA256
# is 347ff2d7ea282bb9454a0379f243080c0ed83a39620de46d612449acb29a0bec.
# The dry-run verifies these values rather than deriving one from another.
CURRENT_FULL_ARTIFACT_EXPECTATIONS = FullArtifactProvenanceExpectations(
    total_rows=36577,
    phase1_rows=7408,
    phase1_unique_conditions=463,
)


def _condition_ids(rows: Iterable[Mapping[str, Any]], *, label: str) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        cid = row.get("condition_id")
        if cid in (None, ""):
            raise ValueError(f"{label} row missing condition_id")
        key = str(cid)
        if key in ids:
            raise ValueError(f"{label} duplicate condition_id {key}")
        ids.add(key)
    return ids


def _phase1_counts(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str], dict[int, int]]:
    phase1 = [
        dict(row)
        for row in rows
        if row.get("source_type") == PHASE1_SOURCE_TYPE
    ]
    per_condition: Counter[str] = Counter()
    for row in phase1:
        cid = row.get("condition_id")
        if cid in (None, ""):
            raise ValueError("phase1 provenance row missing condition_id")
        per_condition[str(cid)] += 1
    histogram = dict(sorted(Counter(per_condition.values()).items()))
    return phase1, per_condition, histogram


def _distribution(
    per_condition: Counter[str],
    histogram: Mapping[int, int],
) -> dict[str, Any]:
    counts = tuple(per_condition.values())
    return {
        "phase1_rows_per_condition_min": min(counts) if counts else 0,
        "phase1_rows_per_condition_max": max(counts) if counts else 0,
        "phase1_rows_per_condition_histogram": dict(histogram),
    }


def audit_candidate_subset_provenance(
    provenance_rows: Iterable[Mapping[str, Any]],
    *,
    expected: CandidateSubsetProvenanceExpectations,
) -> dict[str, Any]:
    """Validate counts for rows already selected into the candidate scope.

    This deliberately has no publication-membership role and cannot accept
    full-artifact expectations. Candidate selection remains the caller's
    responsibility; this function never reads or mutates cohort state.
    """
    if not isinstance(expected, CandidateSubsetProvenanceExpectations):
        raise TypeError(
            "expected must be CandidateSubsetProvenanceExpectations"
        )

    phase1, per_condition, histogram = _phase1_counts(provenance_rows)
    if len(phase1) != expected.phase1_rows:
        raise AssertionError(
            "candidate-subset phase1 row expectation mismatch: "
            f"expected {expected.phase1_rows}, got {len(phase1)}"
        )
    if len(per_condition) != expected.phase1_unique_conditions:
        raise AssertionError(
            "candidate-subset phase1 condition expectation mismatch: "
            f"expected {expected.phase1_unique_conditions}, "
            f"got {len(per_condition)}"
        )

    return {
        "status": "PASS",
        "contract_scope": CANDIDATE_SUBSET,
        "phase1_rows": len(phase1),
        "phase1_unique_conditions": len(per_condition),
        **_distribution(per_condition, histogram),
    }
def preflight_phase1_provenance_membership(
    provenance_rows: Iterable[Mapping[str, Any]],
    *,
    publication_rows: Iterable[Mapping[str, Any]],
    expected: FullArtifactProvenanceExpectations | None = None,
) -> dict[str, Any]:
    """Read-only full-artifact gate before an output artifact SHA exists.

    This validates the semantic part of the repair contract:
    every phase1_truth provenance condition must exist in the proposed
    behavioral-truth publication.  Actual source_file + artifact-SHA binding
    remains a publish-time gate after Parquet serialization.

    Candidate-subset expectations are rejected by type so candidate counts
    cannot accidentally gate the complete provenance artifact.
    """
    if expected is not None and not isinstance(
        expected,
        FullArtifactProvenanceExpectations,
    ):
        raise TypeError("expected must be FullArtifactProvenanceExpectations")

    publication_ids = _condition_ids(
        publication_rows, label="publication source"
    )
    rows = [dict(row) for row in provenance_rows]
    phase1, per_condition, histogram = _phase1_counts(rows)
    phase1_conditions = set(per_condition)

    missing = sorted(phase1_conditions - publication_ids)
    if missing:
        raise AssertionError(
            "phase1 provenance membership preflight failure: "
            f"{len(missing)} condition_id(s) absent from publication: "
            + ", ".join(missing[:10])
        )

    if expected is not None:
        if len(rows) != expected.total_rows:
            raise AssertionError(
                "provenance total-row expectation mismatch: "
                f"expected {expected.total_rows}, got {len(rows)}"
            )
        if len(phase1) != expected.phase1_rows:
            raise AssertionError(
                "phase1 provenance row expectation mismatch: "
                f"expected {expected.phase1_rows}, got {len(phase1)}"
            )
        if len(phase1_conditions) != expected.phase1_unique_conditions:
            raise AssertionError(
                "phase1 provenance condition expectation mismatch: "
                f"expected {expected.phase1_unique_conditions}, "
                f"got {len(phase1_conditions)}"
            )

    return {
        "status": "PASS",
        "contract_scope": FULL_ARTIFACT,
        "provenance_rows": len(rows),
        "phase1_rows": len(phase1),
        "phase1_unique_conditions": len(phase1_conditions),
        "phase1_null_condition_ids": 0,
        "publication_unique_conditions": len(publication_ids),
        "membership_pass_conditions": len(phase1_conditions),
        "membership_failures": 0,
        **_distribution(per_condition, histogram),
    }


def repair_phase1_provenance_v2(
    provenance_rows: Iterable[Mapping[str, Any]],
    *,
    publication_source_file: str,
    publication_source_sha256: str,
    publication_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a new provenance artifact; never mutate the supplied rows.

    Every phase1_truth row must bind to a condition_id physically present in
    the declared publication artifact before source_file is changed.
    """
    if not publication_source_file:
        raise ValueError("publication_source_file is required")
    if re.fullmatch(r"[0-9a-f]{64}", publication_source_sha256) is None:
        raise ValueError(
            "publication_source_sha256 must be a lowercase hexadecimal SHA256"
        )

    membership = _condition_ids(
        publication_rows, label="publication source"
    )

    repaired: list[dict[str, Any]] = []
    phase1_rows = 0
    phase1_conditions: set[str] = set()

    for original in provenance_rows:
        row = deepcopy(dict(original))

        if row.get("source_type") == PHASE1_SOURCE_TYPE:
            row["provenance_contract_version"] = PROVENANCE_CONTRACT_VERSION
            cid = row.get("condition_id")
            if cid in (None, ""):
                raise ValueError("phase1 provenance row missing condition_id")
            key = str(cid)
            if key not in membership:
                raise ValueError(
                    "declared source membership failure before repair: "
                    f"condition_id {key} not present in publication source"
                )

            previous_source_file = row.get("previous_source_file")
            if (
                not isinstance(previous_source_file, str)
                or not previous_source_file.strip()
            ):
                row["previous_source_file"] = row.get("source_file")
            row["source_file"] = publication_source_file
            row["source_artifact_sha256"] = publication_source_sha256
            row["source_membership_verified"] = True
            phase1_rows += 1
            phase1_conditions.add(key)
        repaired.append(row)

    report = validate_phase1_provenance_membership(
        repaired,
        source_membership={publication_source_file: membership},
        source_sha256={publication_source_file: publication_source_sha256},
    )
    report.update(
        {
            "status": "PASS",
            "contract_version": PROVENANCE_CONTRACT_VERSION,
            "repaired_rows": len(repaired),
            "phase1_rows": phase1_rows,
            "phase1_unique_conditions": len(phase1_conditions),
        }
    )
    return repaired, report


def validate_phase1_provenance_membership(
    provenance_rows: Iterable[Mapping[str, Any]],
    *,
    source_membership: Mapping[str, set[str]],
    source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Require condition_id ∈ declared source artifact for phase1 truth."""
    checked = 0
    conditions: set[str] = set()
    failures: list[dict[str, Any]] = []

    for row in provenance_rows:
        if row.get("source_type") != PHASE1_SOURCE_TYPE:
            continue

        checked += 1
        cid = str(row.get("condition_id") or "")
        conditions.add(cid)

        source_file = row.get("source_file")
        if not isinstance(source_file, str) or not source_file:
            failures.append(
                {"condition_id": cid, "reason": "MISSING_DECLARED_SOURCE"}
            )
            continue

        # Phase-1 truth must name one authoritative publication artifact, not a
        # semicolon-joined set that makes membership ambiguous.
        if ";" in source_file:
            failures.append(
                {"condition_id": cid, "reason": "AMBIGUOUS_DECLARED_SOURCE"}
            )
            continue

        members = source_membership.get(source_file)
        if members is None:
            failures.append(
                {"condition_id": cid, "reason": "DECLARED_SOURCE_NOT_INDEXED"}
            )
            continue

        if cid not in members:
            failures.append(
                {"condition_id": cid, "reason": "CONDITION_NOT_IN_DECLARED_SOURCE"}
            )
            continue

        if source_sha256 is not None:
            expected = source_sha256.get(source_file)
            actual = row.get("source_artifact_sha256")
            if expected is None:
                failures.append(
                    {"condition_id": cid, "reason": "DECLARED_SOURCE_SHA_NOT_INDEXED"}
                )
                continue
            if actual != expected:
                failures.append(
                    {"condition_id": cid, "reason": "DECLARED_SOURCE_SHA_MISMATCH"}
                )
                continue

        if row.get("source_membership_verified") is not True:
            failures.append(
                {"condition_id": cid, "reason": "MEMBERSHIP_VERIFICATION_FLAG_MISSING"}
            )

    if failures:
        raise AssertionError(
            "phase1 provenance membership failure: "
            + repr(failures[:5])
        )

    return {
        "status": "PASS",
        "phase1_rows_checked": checked,
        "phase1_unique_conditions_checked": len(conditions),
        "membership_failures": 0,
    }
