"""Publication / provenance repair v2 core contract.

This module is intentionally pure: it does not read or write project state.
The caller supplies the frozen canonical ledger, the isolated prospective
ledger, a coverage-bypassed prospective behavioral-truth rebuild, and the
forensic resolutions for every non-exact overlap.

The historical canonical artifact is evidence only and is never rewritten.
Publication v2 is a separate prospective-universe truth artifact.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

PUBLICATION_CONTRACT_VERSION = "publication_provenance_v2"

RESOLUTION_PROSPECTIVE_NEW = "PROSPECTIVE_NEW"
RESOLUTION_EXACT_OVERLAP = "EXACT_OVERLAP"
RESOLUTION_COVERAGE_PATH_ONLY = "COVERAGE_PATH_ONLY"
RESOLUTION_COVERAGE_EARLY_RETURN = "COVERAGE_EARLY_RETURN_MASKING_CANONICAL"
RESOLUTION_CANONICAL_PROVENANCE_GAP = (
    "CANONICAL_ARTIFACT_NOT_REPRODUCIBLE_UNDER_PINNED_SEMANTICS"
)
RESOLUTION_RAW_SNAPSHOT = "RAW_SNAPSHOT_EXPLAINS_ARTIFACT_DIFFERENCE"

FORENSIC_RESOLUTION_CLASSES = frozenset(
    {
        RESOLUTION_COVERAGE_PATH_ONLY,
        RESOLUTION_COVERAGE_EARLY_RETURN,
        RESOLUTION_CANONICAL_PROVENANCE_GAP,
        RESOLUTION_RAW_SNAPSHOT,
    }
)

SELECTED_TRUTH_SOURCE = "prospective_behavioral_truth_v2"


@dataclass(frozen=True)
class PublicationExpectations:
    prospective_rows: int
    overlap_rows: int
    new_rows: int
    exact_overlap_rows: int
    conflict_rows: int
    forensic_resolution_counts: Mapping[str, int]


CURRENT_REPAIR_EXPECTATIONS = PublicationExpectations(
    prospective_rows=1675,
    overlap_rows=321,
    new_rows=1354,
    exact_overlap_rows=78,
    conflict_rows=243,
    forensic_resolution_counts={
        RESOLUTION_COVERAGE_PATH_ONLY: 93,
        RESOLUTION_COVERAGE_EARLY_RETURN: 147,
        RESOLUTION_CANONICAL_PROVENANCE_GAP: 1,
        RESOLUTION_RAW_SNAPSHOT: 2,
    },
)


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=lambda x: str(x))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    # PyArrow/Pandas scalar-like objects should have stable string forms for
    # ledger primitive types; unsupported rich objects remain explicit.
    return {"__type__": type(value).__name__, "__repr__": repr(value)}


def row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_value(dict(row)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rows_content_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Order-independent content digest over unique condition rows."""
    indexed = _index_unique(rows, label="content rows")
    payload = "\n".join(
        f"{cid}:{row_sha256(indexed[cid])}" for cid in sorted(indexed)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _index_unique(
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = row.get("condition_id")
        if cid in (None, ""):
            raise ValueError(f"{label} row missing condition_id")
        key = str(cid)
        if key in indexed:
            raise ValueError(f"{label} duplicate condition_id {key}")
        indexed[key] = dict(row)
    return indexed


def _index_forensic_resolutions(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed = _index_unique(rows, label="forensic resolution")
    for cid, row in indexed.items():
        klass = str(row.get("resolution_class") or "")
        if klass not in FORENSIC_RESOLUTION_CLASSES:
            raise ValueError(
                f"forensic resolution {cid} has unsupported class {klass!r}"
            )
        if bool(row.get("unresolved", False)):
            raise ValueError(f"forensic resolution {cid} is unresolved")
        if bool(row.get("true_semantic_conflict", False)):
            raise ValueError(
                f"forensic resolution {cid} declares true semantic conflict"
            )
    return indexed


def _check_expectations(
    report: Mapping[str, Any],
    forensic_counts: Mapping[str, int],
    expected: PublicationExpectations,
) -> None:
    scalar_pairs = {
        "prospective_rows": expected.prospective_rows,
        "overlap_rows": expected.overlap_rows,
        "new_rows": expected.new_rows,
        "exact_overlap_rows": expected.exact_overlap_rows,
        "conflict_rows": expected.conflict_rows,
    }
    for key, wanted in scalar_pairs.items():
        got = int(report[key])
        if got != int(wanted):
            raise ValueError(
                f"publication expectation mismatch for {key}: "
                f"expected {wanted}, got {got}"
            )

    got_counts = dict(sorted((str(k), int(v)) for k, v in forensic_counts.items()))
    wanted_counts = dict(
        sorted((str(k), int(v)) for k, v in expected.forensic_resolution_counts.items())
    )
    if got_counts != wanted_counts:
        raise ValueError(
            "publication expectation mismatch for forensic resolution counts: "
            f"expected {wanted_counts}, got {got_counts}"
        )


def build_publication_v2(
    *,
    canonical_rows: Iterable[Mapping[str, Any]],
    prospective_ledger_rows: Iterable[Mapping[str, Any]],
    prospective_truth_rows: Iterable[Mapping[str, Any]],
    forensic_resolutions: Iterable[Mapping[str, Any]],
    expected: PublicationExpectations | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build prospective publication-v2 rows and reconciliation manifest.

    The output publication contains *only* the prospective universe. Historical
    canonical rows remain in their frozen artifact and are referenced as
    evidence through manifest hashes.

    Every non-exact overlap must have one forensic resolution. The function
    fails closed for missing/extra resolution rows, unresolved cases, or any
    declared true semantic conflict.
    """
    canonical = _index_unique(canonical_rows, label="canonical ledger")
    prospective = _index_unique(
        prospective_ledger_rows, label="prospective ledger"
    )
    truth = _index_unique(
        prospective_truth_rows, label="prospective behavioral truth"
    )

    if set(prospective) != set(truth):
        missing_truth = sorted(set(prospective) - set(truth))
        extra_truth = sorted(set(truth) - set(prospective))
        raise ValueError(
            "prospective truth membership mismatch: "
            f"missing={missing_truth[:10]} extra={extra_truth[:10]}"
        )

    overlap = set(canonical) & set(prospective)
    new_ids = set(prospective) - set(canonical)
    exact_ids = {
        cid
        for cid in overlap
        if row_sha256(canonical[cid]) == row_sha256(prospective[cid])
    }
    conflict_ids = overlap - exact_ids

    resolutions = _index_forensic_resolutions(forensic_resolutions)
    resolution_ids = set(resolutions)

    missing_resolution = sorted(conflict_ids - resolution_ids)
    extra_resolution = sorted(resolution_ids - conflict_ids)
    if missing_resolution or extra_resolution:
        raise ValueError(
            "forensic resolution coverage mismatch: "
            f"missing={missing_resolution[:10]} extra={extra_resolution[:10]}"
        )

    forensic_counts = Counter(
        str(row["resolution_class"]) for row in resolutions.values()
    )

    publication_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for cid in sorted(prospective):
        canonical_row = canonical.get(cid)
        prospective_row = prospective[cid]
        truth_row = truth[cid]

        if cid in new_ids:
            resolution_class = RESOLUTION_PROSPECTIVE_NEW
            resolution = {}
        elif cid in exact_ids:
            resolution_class = RESOLUTION_EXACT_OVERLAP
            resolution = {}
        else:
            resolution = resolutions[cid]
            resolution_class = str(resolution["resolution_class"])

        publication_rows.append(dict(truth_row))
        manifest_rows.append(
            {
                "contract_version": PUBLICATION_CONTRACT_VERSION,
                "condition_id": cid,
                "resolution_class": resolution_class,
                "canonical_present": canonical_row is not None,
                "prospective_present": True,
                "selected_truth_source": SELECTED_TRUTH_SOURCE,
                "canonical_row_sha256": (
                    row_sha256(canonical_row) if canonical_row is not None else None
                ),
                "prospective_ledger_row_sha256": row_sha256(prospective_row),
                "published_truth_row_sha256": row_sha256(truth_row),
                "raw_relation": resolution.get("raw_relation"),
                "reason": resolution.get("reason"),
                "canonical_evidence_retained": canonical_row is not None,
                "unresolved": False,
                "true_semantic_conflict": False,
            }
        )

    report = {
        "status": "PASS",
        "contract_version": PUBLICATION_CONTRACT_VERSION,
        "canonical_rows": len(canonical),
        "prospective_rows": len(prospective),
        "overlap_rows": len(overlap),
        "new_rows": len(new_ids),
        "exact_overlap_rows": len(exact_ids),
        "conflict_rows": len(conflict_ids),
        "forensic_resolution_counts": dict(sorted(forensic_counts.items())),
        "unresolved_rows": 0,
        "true_semantic_conflicts": 0,
        "published_rows": len(publication_rows),
        "published_unique_condition_ids": len(
            {str(row["condition_id"]) for row in publication_rows}
        ),
        "publication_content_sha256": rows_content_sha256(publication_rows),
    }

    if expected is not None:
        _check_expectations(report, forensic_counts, expected)

    if report["published_rows"] != report["published_unique_condition_ids"]:
        raise AssertionError("publication output lost condition-id uniqueness")

    return publication_rows, manifest_rows, report
