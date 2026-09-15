#!/usr/bin/env python3
"""Standalone real-data DRY RUN for Publication / Provenance Repair v2.

No project files are written.  This harness:
- pins Stage-1 and full-detail Stage-2 logs;
- pins canonical/prospective ledgers, prospective raw, and feature provenance;
- pins the known Stage-2 frozen-semantics replay script;
- rebuilds all 1,675 prospective behavioral-truth rows from prospective raw
  using the exact embedded Stage-2 frozen semantics;
- requires all 243 conflicts to have complete final forensic resolution;
- builds publication-v2 and reconciliation manifest, then serializes and
  reload-validates the complete bundle in a fail-safe system-temp directory;
- verifies the complete pinned provenance artifact (not a candidate subset),
  including its measured phase1_truth distribution and publication binding;
- writes zero project files and deletes the private validation bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXPECTED = {
    "stage1_log_sha256": "946dcb144f3d11f75aade2635505c4fb634f4fc676612ebdd85cf0e0fc4e9054",
    "stage2_log_sha256": "13d264dc259ebc12c9eadbe870c27fefc83623d7d5fa5e350c6df12cc7f635b8",
    "stage2_script_sha256": "818a7f2a286964c00586dee41d5a567d3129c01ad339204eb616de5f7f117344",
    "canonical_ledger_sha256": "63d58236fdfa88dbf0e35e5da74b16db8b40259368929d24d1291644b93b52dd",
    "prospective_ledger_sha256": "ff818c83860d031c080d9a4f97b6035dfbae417a7ab96c8e17837e05632b3770",
    "prospective_raw_sha256": "ddc5eee63b3f58451c9620161518884708e6da724a95bd9c29125cee8c8620d9",
    "feature_provenance_sha256": "347ff2d7ea282bb9454a0379f243080c0ed83a39620de46d612449acb29a0bec",
    "prospective_rows": 1675,
    "provenance_rows": 36577,
    "phase1_rows": 7408,
    "phase1_conditions": 463,
    "phase1_rows_per_condition_min": 16,
    "phase1_rows_per_condition_max": 16,
    "phase1_rows_per_condition_histogram": {16: 463},
    "publication_content_sha256": "30be62ae728f09db145734237dd2e84d4b2619d235cfc915fbd04aaf6a75d045",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label}_MISSING:{path}")
    actual = sha256(path)
    print(f"{label}_SHA256={actual}")
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label}_SHA_MISMATCH")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-log", required=True, type=Path)
    ap.add_argument("--stage2-log", required=True, type=Path)
    ap.add_argument("--stage2-script", required=True, type=Path)
    ap.add_argument("--canonical-ledger", required=True, type=Path)
    ap.add_argument("--prospective-ledger", required=True, type=Path)
    ap.add_argument("--prospective-raw", required=True, type=Path)
    ap.add_argument("--feature-provenance", required=True, type=Path)
    return ap.parse_args()


def require_original_inputs(args: argparse.Namespace, *, prefix: str = "") -> None:
    pins = (
        (args.stage1_log, EXPECTED["stage1_log_sha256"], "STAGE1_LOG"),
        (args.stage2_log, EXPECTED["stage2_log_sha256"], "STAGE2_LOG"),
        (args.stage2_script, EXPECTED["stage2_script_sha256"], "STAGE2_SCRIPT"),
        (
            args.canonical_ledger,
            EXPECTED["canonical_ledger_sha256"],
            "CANONICAL_LEDGER",
        ),
        (
            args.prospective_ledger,
            EXPECTED["prospective_ledger_sha256"],
            "PROSPECTIVE_LEDGER",
        ),
        (args.prospective_raw, EXPECTED["prospective_raw_sha256"], "PROSPECTIVE_RAW"),
        (
            args.feature_provenance,
            EXPECTED["feature_provenance_sha256"],
            "FEATURE_PROVENANCE",
        ),
    )
    for path, expected, label in pins:
        require_sha(path, expected, prefix + label)


def load_build_module(root: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "publication_v2_build_for_realdata_validation",
        root / "scripts" / "build_publication_provenance_v2.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load publication-v2 build module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_verified_realdata(
    args: argparse.Namespace,
    *,
    original_args: argparse.Namespace,
    root: Path,
    build,
) -> int:
    from std0_quant.events import publication_forensics_v2 as forensics
    from std0_quant.events import publication_provenance_v2 as publication
    from std0_quant.features import provenance_v2 as provenance

    import importlib.util

    replay_spec = importlib.util.spec_from_file_location(
        "stage2_frozen_replay",
        args.stage2_script,
    )
    if replay_spec is None or replay_spec.loader is None:
        raise RuntimeError("cannot load pinned Stage-2 replay script")
    replay = importlib.util.module_from_spec(replay_spec)
    sys.modules["stage2_frozen_replay"] = replay
    replay_spec.loader.exec_module(replay)

    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(f"PYARROW_IMPORT_FAILED:{type(exc).__name__}") from exc

    canonical_rows = pq.read_table(args.canonical_ledger).to_pylist()
    prospective_rows = pq.read_table(args.prospective_ledger).to_pylist()
    provenance_rows = pq.read_table(args.feature_provenance).to_pylist()

    prospective_by = {}
    for row in prospective_rows:
        cid = str(row.get("condition_id") or "")
        if not cid:
            raise RuntimeError("PROSPECTIVE_LEDGER_ROW_MISSING_CONDITION_ID")
        if cid in prospective_by:
            raise RuntimeError(f"PROSPECTIVE_LEDGER_DUPLICATE:{cid}")
        prospective_by[cid] = row

    if len(prospective_by) != EXPECTED["prospective_rows"]:
        raise RuntimeError(
            f"PROSPECTIVE_ROW_COUNT_MISMATCH:{len(prospective_by)}"
        )

    target_ids = set(prospective_by)
    raw_by, raw_lines, matched_lines, duplicate_ids = replay.scan_raw(
        args.prospective_raw,
        target_ids,
        "PROSPECTIVE_RAW_ALL_1675",
    )
    missing_raw = sorted(target_ids - set(raw_by))
    if missing_raw:
        raise RuntimeError(
            f"PROSPECTIVE_RAW_MEMBERSHIP_MISSING:{len(missing_raw)}:"
            + ",".join(missing_raw[:10])
        )

    truth_rows = [
        replay.build_semantic_row(cid, raw_by[cid])
        for cid in sorted(target_ids)
    ]
    print(f"PROSPECTIVE_RAW_LINES={raw_lines}")
    print(f"PROSPECTIVE_RAW_MATCHED_LINES={matched_lines}")
    print(f"PROSPECTIVE_RAW_DUPLICATE_FILL_IDS_SKIPPED={duplicate_ids}")
    print(f"BEHAVIORAL_TRUTH_ROWS={len(truth_rows)}")

    resolutions, forensic_report = (
        forensics.build_forensic_resolutions_from_logs(
            args.stage1_log,
            args.stage2_log,
        )
    )
    print("FORENSIC_REPORT=" + json.dumps(forensic_report, sort_keys=True))

    publication_rows, manifest_rows, publication_report = (
        publication.build_publication_v2(
            canonical_rows=canonical_rows,
            prospective_ledger_rows=prospective_rows,
            prospective_truth_rows=truth_rows,
            forensic_resolutions=resolutions,
            expected=publication.CURRENT_REPAIR_EXPECTATIONS,
        )
    )
    print(
        "PUBLICATION_REPORT="
        + json.dumps(publication_report, sort_keys=True)
    )
    if (
        publication_report["publication_content_sha256"]
        != EXPECTED["publication_content_sha256"]
    ):
        raise RuntimeError("PUBLICATION_CONTENT_SHA256_MISMATCH")

    provenance_expected = provenance.FullArtifactProvenanceExpectations(
        total_rows=EXPECTED["provenance_rows"],
        phase1_rows=EXPECTED["phase1_rows"],
        phase1_unique_conditions=EXPECTED["phase1_conditions"],
    )
    provenance_report = provenance.preflight_phase1_provenance_membership(
        provenance_rows,
        publication_rows=publication_rows,
        expected=provenance_expected,
    )
    for key in (
        "phase1_rows_per_condition_min",
        "phase1_rows_per_condition_max",
        "phase1_rows_per_condition_histogram",
    ):
        if provenance_report[key] != EXPECTED[key]:
            raise RuntimeError(f"{key.upper()}_MISMATCH")
    print(
        "PROVENANCE_PREFLIGHT="
        + json.dumps(provenance_report, sort_keys=True)
    )
    print(
        "PHASE1_NULL_CONDITION_IDS="
        f"{provenance_report['phase1_null_condition_ids']}"
    )
    build.report_candidate_subset_contract()
    if len(manifest_rows) != EXPECTED["prospective_rows"]:
        raise RuntimeError(
            f"RECONCILIATION_ROW_COUNT_MISMATCH:{len(manifest_rows)}"
        )

    builder_state = build.read_builder_state(root)

    def stage(staging: Path) -> dict:
        return build.build_validated_staged_bundle(
            root=root,
            staging=staging,
            pq=pq,
            atomic_write_parquet=build.atomic_write_parquet_v2,
            publication_rows=publication_rows,
            reconciliation_rows=manifest_rows,
            input_provenance_rows=provenance_rows,
            report=publication_report,
            builder_state=builder_state,
            provenance_expected=provenance_expected,
        )

    staged_result = build.run_private_staged_validation(
        stage,
        project_root=root,
    )
    print(
        "PROVENANCE_REPORT="
        + json.dumps(staged_result["provenance_report"], sort_keys=True)
    )
    print("TEMP_VALIDATION_BUNDLE_CLEANED=YES")

    require_original_inputs(original_args, prefix="POST_")
    print("PROTECTED_INPUTS_UNCHANGED=YES")
    print("PROJECT_OUTPUT_FILES_WRITTEN=0")
    print("OUTPUT_FILES_WRITTEN=0")
    return 0


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    build = load_build_module(root)

    print("MODE=READ_ONLY_PUBLICATION_PROVENANCE_V2_REALDATA_DRYRUN")
    print("FORMAL_COHORT_ACCESS=NONE")
    print("BACKTEST_EXECUTED=false")

    require_original_inputs(args)
    with build.verified_input_snapshots(args) as snapshot_args:
        result = run_verified_realdata(
            snapshot_args,
            original_args=args,
            root=root,
            build=build,
        )
    print("TEMP_INPUT_SNAPSHOTS_CLEANED=YES")
    print("SNAPSHOT_TEMP_CLEANED=YES")
    print("REALDATA_DRYRUN_STATUS=PASS")
    print("FORMAL_COHORT_WRITTEN=false")
    print("BACKTEST_EXECUTED=false")
    print("COMMIT=NOT_RUN")
    print("PUSH=NOT_RUN")
    print("PR=NOT_RUN")
    print("MERGE=NOT_RUN")
    print("DEPLOY=NOT_RUN")
    print("EXIT=0")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
