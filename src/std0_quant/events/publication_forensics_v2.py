"""Convert Stage-1/Stage-2 read-only forensic logs into repair resolutions."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from std0_quant.events.publication_provenance_v2 import (
    RESOLUTION_CANONICAL_PROVENANCE_GAP,
    RESOLUTION_COVERAGE_EARLY_RETURN,
    RESOLUTION_COVERAGE_PATH_ONLY,
    RESOLUTION_RAW_SNAPSHOT,
)

EXPECTED_STAGE1_CONFLICTS = 243
EXPECTED_STAGE2_TARGETS = 150

_STAGE2_MAP = {
    "COVERAGE_EARLY_RETURN_MASKING_CANONICAL":
        RESOLUTION_COVERAGE_EARLY_RETURN,
    "CURRENT_FROZEN_REPLAY_SUPPORTS_PROSPECTIVE":
        RESOLUTION_CANONICAL_PROVENANCE_GAP,
    "RAW_SNAPSHOT_EXPLAINS_ARTIFACT_DIFFERENCE":
        RESOLUTION_RAW_SNAPSHOT,
}


def _parse_prefixed_json(path: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="strict") as fh:
        for number, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            if not line.startswith(prefix):
                continue
            try:
                obj = json.loads(line[len(prefix):])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid {prefix} JSON at line {number}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}: {prefix} at line {number} is not an object"
                )
            rows.append(obj)
    return rows


def _scalar(path: Path, key: str) -> str | None:
    prefix = key + "="
    found: str | None = None
    with path.open("r", encoding="utf-8-sig", errors="strict") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.startswith(prefix):
                if found is not None:
                    raise ValueError(f"{path}: duplicate scalar {key}")
                found = line[len(prefix):]
    return found


def _replay_required(row: Mapping[str, Any], cid: str) -> bool:
    value = row.get("semantic_replay_required")
    if not isinstance(value, bool):
        raise ValueError(
            f"{cid}: semantic_replay_required must be boolean"
        )
    return value


def build_forensic_resolutions_from_logs(
    stage1_log: Path | str,
    stage2_log: Path | str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Require complete 243/150 evidence and emit one final row per conflict."""
    stage1_path = Path(stage1_log)
    stage2_path = Path(stage2_log)

    if _scalar(stage1_path, "STAGE1_STATUS") != "PASS":
        raise ValueError("Stage-1 log is not PASS")
    if _scalar(stage2_path, "STAGE2_STATUS") != "PASS":
        raise ValueError("Stage-2 log is not PASS")

    conflicts_text = _scalar(stage1_path, "CONFLICTS")
    targets_text = _scalar(stage2_path, "STAGE2_TARGETS")
    unresolved_text = _scalar(stage2_path, "UNRESOLVED_COUNT")
    true_conflict_text = _scalar(stage2_path, "TRUE_SEMANTIC_CONFLICT_ASSIGNED")

    if int(conflicts_text or -1) != EXPECTED_STAGE1_CONFLICTS:
        raise ValueError("Stage-1 conflict count is not the frozen 243")
    if int(targets_text or -1) != EXPECTED_STAGE2_TARGETS:
        raise ValueError("Stage-2 target count is not the frozen 150")
    if int(unresolved_text or -1) != 0:
        raise ValueError("Stage-2 contains unresolved conditions")
    if int(true_conflict_text or -1) != 0:
        raise ValueError("Stage-2 assigned true semantic conflicts")

    stage1_details = _parse_prefixed_json(stage1_path, "DETAIL=")
    stage2_details = _parse_prefixed_json(stage2_path, "DETAIL2=")

    s1: dict[str, dict[str, Any]] = {}
    for row in stage1_details:
        cid = str(row.get("condition_id") or "")
        if not cid:
            raise ValueError("Stage-1 detail missing condition_id")
        if cid in s1:
            raise ValueError(f"duplicate Stage-1 detail {cid}")
        s1[cid] = row

    s2: dict[str, dict[str, Any]] = {}
    for row in stage2_details:
        cid = str(row.get("condition_id") or "")
        if not cid:
            raise ValueError("Stage-2 detail missing condition_id")
        if cid in s2:
            raise ValueError(f"duplicate Stage-2 detail {cid}")
        s2[cid] = row

    if len(s1) != EXPECTED_STAGE1_CONFLICTS:
        raise ValueError(
            f"Stage-1 DETAIL coverage incomplete: expected 243, got {len(s1)}"
        )

    replay_required = {
        cid: _replay_required(row, cid)
        for cid, row in s1.items()
    }
    replay_ids = {cid for cid, required in replay_required.items() if required}
    if len(replay_ids) != EXPECTED_STAGE2_TARGETS:
        raise ValueError(
            f"Stage-1 replay target mismatch: expected 150, got {len(replay_ids)}"
        )

    if set(s2) != replay_ids:
        missing = sorted(replay_ids - set(s2))
        extra = sorted(set(s2) - replay_ids)
        raise ValueError(
            "Stage-2 DETAIL2 coverage mismatch: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    resolutions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for cid in sorted(s1):
        first = s1[cid]

        if not replay_required[cid]:
            if first.get("root_class_stage1") != RESOLUTION_COVERAGE_PATH_ONLY:
                raise ValueError(
                    f"{cid}: non-replay Stage-1 class is not COVERAGE_PATH_ONLY"
                )
            klass = RESOLUTION_COVERAGE_PATH_ONLY
            second = None
            reason = "Stage-1 exact-raw coverage-path-only conflict"
        else:
            second = s2[cid]
            root = str(second.get("stage2_root_class") or "")
            klass = _STAGE2_MAP.get(root)
            if klass is None:
                raise ValueError(
                    f"{cid}: unsupported Stage-2 root class {root!r}"
                )
            reason = f"Stage-2 frozen-semantics replay: {root}"

        raw_relation = str(first.get("raw_relation") or "")
        resolution = {
            "condition_id": cid,
            "resolution_class": klass,
            "raw_relation": raw_relation,
            "reason": reason,
            "stage1_ledger_diff_fields": list(
                first.get("ledger_diff_fields") or []
            ),
            "unresolved": False,
            "true_semantic_conflict": False,
        }
        if second is not None:
            resolution["stage2_root_class"] = second.get("stage2_root_class")
            resolution[
                "historical_vs_prospective_replay_diff_fields"
            ] = list(
                second.get(
                    "historical_vs_prospective_replay_diff_fields"
                ) or []
            )

        resolutions.append(resolution)
        counts[klass] += 1

    report = {
        "status": "PASS",
        "stage1_conflicts": len(s1),
        "stage2_targets": len(replay_ids),
        "resolution_rows": len(resolutions),
        "resolution_counts": dict(sorted(counts.items())),
        "unresolved": 0,
        "true_semantic_conflicts": 0,
    }
    return resolutions, report
