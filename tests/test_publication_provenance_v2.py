from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from std0_quant.events.publication_provenance_v2 import (
    PublicationExpectations,
    RESOLUTION_CANONICAL_PROVENANCE_GAP,
    RESOLUTION_COVERAGE_EARLY_RETURN,
    RESOLUTION_COVERAGE_PATH_ONLY,
    RESOLUTION_RAW_SNAPSHOT,
    build_publication_v2,
    row_sha256,
    rows_content_sha256,
)
from std0_quant.features.provenance_v2 import (
    CandidateSubsetProvenanceExpectations,
    FullArtifactProvenanceExpectations,
    audit_candidate_subset_provenance,
    preflight_phase1_provenance_membership,
    repair_phase1_provenance_v2,
    validate_phase1_provenance_membership,
)


def ledger(cid: str, *, value: int = 1, coverage: float | None = 1.0) -> dict:
    return {
        "condition_id": cid,
        "y30": value,
        "n_buy_fills": 2,
        "poly_book_coverage_pct": coverage,
        "btc_coverage_pct": coverage,
    }


def resolution(cid: str, klass: str) -> dict:
    return {
        "condition_id": cid,
        "resolution_class": klass,
        "raw_relation": "EXACT_RAW",
        "reason": "test evidence",
        "unresolved": False,
        "true_semantic_conflict": False,
    }


def load_build_script_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_publication_provenance_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_publication_provenance_v2_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publication_contract_selects_prospective_behavioral_truth():
    canonical = [
        ledger("exact"),
        ledger("coverage", value=0),
        ledger("early", value=0),
        ledger("gap", value=0),
        ledger("snapshot", value=0),
    ]
    prospective = [
        ledger("new"),
        ledger("exact"),
        ledger("coverage", value=1),
        ledger("early", value=1),
        ledger("gap", value=1),
        ledger("snapshot", value=1),
    ]
    truth = [
        {**row, "poly_book_coverage_pct": None, "btc_coverage_pct": None}
        for row in prospective
    ]

    resolutions = [
        resolution("coverage", RESOLUTION_COVERAGE_PATH_ONLY),
        resolution("early", RESOLUTION_COVERAGE_EARLY_RETURN),
        resolution("gap", RESOLUTION_CANONICAL_PROVENANCE_GAP),
        resolution("snapshot", RESOLUTION_RAW_SNAPSHOT),
    ]

    expected = PublicationExpectations(
        prospective_rows=6,
        overlap_rows=5,
        new_rows=1,
        exact_overlap_rows=1,
        conflict_rows=4,
        forensic_resolution_counts={
            RESOLUTION_COVERAGE_PATH_ONLY: 1,
            RESOLUTION_COVERAGE_EARLY_RETURN: 1,
            RESOLUTION_CANONICAL_PROVENANCE_GAP: 1,
            RESOLUTION_RAW_SNAPSHOT: 1,
        },
    )

    published, manifest, report = build_publication_v2(
        canonical_rows=canonical,
        prospective_ledger_rows=prospective,
        prospective_truth_rows=truth,
        forensic_resolutions=resolutions,
        expected=expected,
    )

    assert report["status"] == "PASS"
    assert report["published_rows"] == 6
    assert report["true_semantic_conflicts"] == 0
    assert report["unresolved_rows"] == 0
    assert {r["condition_id"] for r in published} == {
        "new", "exact", "coverage", "early", "gap", "snapshot"
    }
    assert all(r["poly_book_coverage_pct"] is None for r in published)

    by = {r["condition_id"]: r for r in manifest}
    assert by["new"]["resolution_class"] == "PROSPECTIVE_NEW"
    assert by["exact"]["resolution_class"] == "EXACT_OVERLAP"
    assert by["coverage"]["resolution_class"] == RESOLUTION_COVERAGE_PATH_ONLY
    assert by["early"]["resolution_class"] == RESOLUTION_COVERAGE_EARLY_RETURN
    assert by["gap"]["resolution_class"] == RESOLUTION_CANONICAL_PROVENANCE_GAP
    assert by["snapshot"]["resolution_class"] == RESOLUTION_RAW_SNAPSHOT


def test_missing_forensic_resolution_fails_closed():
    canonical = [ledger("x", value=0)]
    prospective = [ledger("x", value=1)]
    truth = [ledger("x", value=1, coverage=None)]
    with pytest.raises(ValueError, match="coverage mismatch"):
        build_publication_v2(
            canonical_rows=canonical,
            prospective_ledger_rows=prospective,
            prospective_truth_rows=truth,
            forensic_resolutions=[],
        )


def test_extra_forensic_resolution_fails_closed():
    canonical = []
    prospective = [ledger("new")]
    truth = [ledger("new", coverage=None)]
    with pytest.raises(ValueError, match="coverage mismatch"):
        build_publication_v2(
            canonical_rows=canonical,
            prospective_ledger_rows=prospective,
            prospective_truth_rows=truth,
            forensic_resolutions=[
                resolution("new", RESOLUTION_COVERAGE_PATH_ONLY)
            ],
        )


def test_unresolved_and_true_semantic_conflict_are_forbidden():
    canonical = [ledger("x", value=0)]
    prospective = [ledger("x", value=1)]
    truth = [ledger("x", value=1, coverage=None)]

    unresolved = resolution("x", RESOLUTION_COVERAGE_PATH_ONLY)
    unresolved["unresolved"] = True
    with pytest.raises(ValueError, match="unresolved"):
        build_publication_v2(
            canonical_rows=canonical,
            prospective_ledger_rows=prospective,
            prospective_truth_rows=truth,
            forensic_resolutions=[unresolved],
        )

    conflict = resolution("x", RESOLUTION_COVERAGE_PATH_ONLY)
    conflict["true_semantic_conflict"] = True
    with pytest.raises(ValueError, match="true semantic conflict"):
        build_publication_v2(
            canonical_rows=canonical,
            prospective_ledger_rows=prospective,
            prospective_truth_rows=truth,
            forensic_resolutions=[conflict],
        )


def test_truth_membership_must_equal_prospective_membership():
    with pytest.raises(ValueError, match="truth membership mismatch"):
        build_publication_v2(
            canonical_rows=[],
            prospective_ledger_rows=[ledger("p1"), ledger("p2")],
            prospective_truth_rows=[ledger("p1", coverage=None)],
            forensic_resolutions=[],
        )


def test_duplicate_condition_id_fails_closed():
    with pytest.raises(ValueError, match="duplicate condition_id"):
        build_publication_v2(
            canonical_rows=[],
            prospective_ledger_rows=[ledger("p1"), ledger("p1")],
            prospective_truth_rows=[ledger("p1", coverage=None)],
            forensic_resolutions=[],
        )


def test_expected_counts_are_enforced():
    wrong = PublicationExpectations(
        prospective_rows=999,
        overlap_rows=0,
        new_rows=1,
        exact_overlap_rows=0,
        conflict_rows=0,
        forensic_resolution_counts={},
    )
    with pytest.raises(ValueError, match="expectation mismatch"):
        build_publication_v2(
            canonical_rows=[],
            prospective_ledger_rows=[ledger("p1")],
            prospective_truth_rows=[ledger("p1", coverage=None)],
            forensic_resolutions=[],
            expected=wrong,
        )


def test_row_hash_is_key_order_stable_and_handles_nan():
    a = {"condition_id": "x", "z": math.nan, "a": [1, 2]}
    b = {"a": [1, 2], "z": math.nan, "condition_id": "x"}
    assert row_sha256(a) == row_sha256(b)


def test_provenance_repair_binds_declared_source_membership_without_mutation():
    original = [
        {
            "condition_id": "c1",
            "feature_name": "old_direction_qty",
            "source_type": "phase1_truth",
            "source_file": "old/event_ledger.parquet",
        },
        {
            "condition_id": "c1",
            "feature_name": "btc_ret_1s",
            "source_type": "binance_btc",
            "source_file": "raw/btc.ndjson",
        },
    ]
    before = copy.deepcopy(original)
    publication = [ledger("c1", coverage=None)]

    repaired, report = repair_phase1_provenance_v2(
        original,
        publication_source_file="publication_v2/event_ledger.parquet",
        publication_source_sha256="a" * 64,
        publication_rows=publication,
    )

    assert original == before
    assert report["status"] == "PASS"
    phase1 = repaired[0]
    public = repaired[1]

    assert phase1["previous_source_file"] == "old/event_ledger.parquet"
    assert phase1["source_file"] == "publication_v2/event_ledger.parquet"
    assert phase1["source_artifact_sha256"] == "a" * 64
    assert phase1["source_membership_verified"] is True

    assert public["source_file"] == "raw/btc.ndjson"
    assert "source_membership_verified" not in public


def test_provenance_repair_rejects_non_hex_source_sha():
    with pytest.raises(ValueError, match="lowercase hexadecimal SHA256"):
        repair_phase1_provenance_v2(
            [
                {
                    "condition_id": "c1",
                    "feature_name": "old_direction_qty",
                    "source_type": "phase1_truth",
                    "source_file": "old.parquet",
                }
            ],
            publication_source_file="publication_v2/event_ledger.parquet",
            publication_source_sha256="z" * 64,
            publication_rows=[ledger("c1", coverage=None)],
        )


def test_provenance_repair_preserves_prior_chain_and_non_phase1_fields():
    original = [
        {
            "condition_id": "c1",
            "feature_name": "old_direction_qty",
            "source_type": "phase1_truth",
            "source_file": "intermediate.parquet",
            "previous_source_file": "original.parquet",
        },
        {
            "condition_id": "c1",
            "feature_name": "btc_ret_1s",
            "source_type": "binance_btc",
            "source_file": "raw/btc.ndjson",
            "source_membership_verified": True,
            "source_artifact_sha256": "b" * 64,
        },
    ]

    repaired, _ = repair_phase1_provenance_v2(
        original,
        publication_source_file="publication_v2/event_ledger.parquet",
        publication_source_sha256="a" * 64,
        publication_rows=[ledger("c1", coverage=None)],
    )

    assert repaired[0]["previous_source_file"] == "original.parquet"
    assert repaired[1]["source_file"] == "raw/btc.ndjson"
    assert repaired[1]["source_membership_verified"] is True
    assert repaired[1]["source_artifact_sha256"] == "b" * 64


def test_provenance_repair_leaves_btc_and_book_source_fields_unchanged():
    source_rows = [
        {
            "condition_id": "c1",
            "feature_name": "old_direction_qty",
            "source_type": "phase1_truth",
            "source_file": "old.parquet",
        },
        {
            "condition_id": "c1",
            "feature_name": "btc_ret_1s",
            "source_type": "binance_btc",
            "source_file": "raw/btc.ndjson",
            "source_artifact_sha256": "b" * 64,
            "source_membership_verified": False,
        },
        {
            "condition_id": "c1",
            "feature_name": "book_spread",
            "source_type": "polymarket_book",
            "source_file": "raw/book.ndjson",
            "source_artifact_sha256": "c" * 64,
            "source_membership_verified": True,
        },
    ]
    before = copy.deepcopy(source_rows)

    repaired, _ = repair_phase1_provenance_v2(
        source_rows,
        publication_source_file="publication_v2/event_ledger.parquet",
        publication_source_sha256="a" * 64,
        publication_rows=[ledger("c1", coverage=None)],
    )

    assert repaired[1] == before[1]
    assert repaired[2] == before[2]


@pytest.mark.parametrize("previous_source_file", [None, "", "   "])
def test_provenance_repair_fills_empty_previous_source_file(
    previous_source_file,
):
    repaired, _ = repair_phase1_provenance_v2(
        [
            {
                "condition_id": "c1",
                "feature_name": "old_direction_qty",
                "source_type": "phase1_truth",
                "source_file": "old.parquet",
                "previous_source_file": previous_source_file,
            }
        ],
        publication_source_file="publication_v2/event_ledger.parquet",
        publication_source_sha256="a" * 64,
        publication_rows=[ledger("c1", coverage=None)],
    )

    assert repaired[0]["previous_source_file"] == "old.parquet"


def test_repeated_provenance_repair_preserves_earliest_non_empty_lineage():
    source = [
        {
            "condition_id": "c1",
            "feature_name": "old_direction_qty",
            "source_type": "phase1_truth",
            "source_file": "original.parquet",
        }
    ]
    publication = [ledger("c1", coverage=None)]

    first, _ = repair_phase1_provenance_v2(
        source,
        publication_source_file="publication_v2/first.parquet",
        publication_source_sha256="a" * 64,
        publication_rows=publication,
    )
    second, _ = repair_phase1_provenance_v2(
        first,
        publication_source_file="publication_v2/second.parquet",
        publication_source_sha256="b" * 64,
        publication_rows=publication,
    )

    assert first[0]["previous_source_file"] == "original.parquet"
    assert second[0]["previous_source_file"] == "original.parquet"
    assert second[0]["source_file"] == "publication_v2/second.parquet"


def test_provenance_repair_rejects_missing_condition_membership():
    with pytest.raises(ValueError, match="membership failure before repair"):
        repair_phase1_provenance_v2(
            [
                {
                    "condition_id": "missing",
                    "feature_name": "old_direction_qty",
                    "source_type": "phase1_truth",
                    "source_file": "old.parquet",
                }
            ],
            publication_source_file="publication_v2/event_ledger.parquet",
            publication_source_sha256="b" * 64,
            publication_rows=[ledger("other", coverage=None)],
        )


def test_membership_validator_rejects_wrong_declared_source_and_sha():
    row = {
        "condition_id": "c1",
        "feature_name": "old_direction_qty",
        "source_type": "phase1_truth",
        "source_file": "publication_v2/event_ledger.parquet",
        "source_artifact_sha256": "a" * 64,
        "source_membership_verified": True,
    }

    with pytest.raises(AssertionError, match="DECLARED_SOURCE_NOT_INDEXED"):
        validate_phase1_provenance_membership(
            [row],
            source_membership={"other.parquet": {"c1"}},
        )

    with pytest.raises(AssertionError, match="DECLARED_SOURCE_SHA_MISMATCH"):
        validate_phase1_provenance_membership(
            [row],
            source_membership={
                "publication_v2/event_ledger.parquet": {"c1"}
            },
            source_sha256={
                "publication_v2/event_ledger.parquet": "b" * 64
            },
        )


def test_membership_validator_rejects_ambiguous_semicolon_source():
    row = {
        "condition_id": "c1",
        "feature_name": "old_direction_qty",
        "source_type": "phase1_truth",
        "source_file": "a.parquet;b.parquet",
        "source_membership_verified": True,
    }
    with pytest.raises(AssertionError, match="AMBIGUOUS_DECLARED_SOURCE"):
        validate_phase1_provenance_membership(
            [row],
            source_membership={"a.parquet": {"c1"}, "b.parquet": {"c1"}},
        )


def test_forensic_log_parser_requires_complete_detail_coverage(tmp_path):
    from std0_quant.events.publication_forensics_v2 import (
        build_forensic_resolutions_from_logs,
    )

    s1 = tmp_path / "stage1.txt"
    s2 = tmp_path / "stage2.txt"
    s1.write_text("CONFLICTS=243\nSTAGE1_STATUS=PASS\n", encoding="utf-8")
    s2.write_text(
        "STAGE2_TARGETS=150\nUNRESOLVED_COUNT=0\n"
        "TRUE_SEMANTIC_CONFLICT_ASSIGNED=0\nSTAGE2_STATUS=PASS\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DETAIL coverage incomplete"):
        build_forensic_resolutions_from_logs(s1, s2)


def test_forensic_log_parser_rejects_duplicate_scalar(tmp_path):
    from std0_quant.events.publication_forensics_v2 import (
        build_forensic_resolutions_from_logs,
    )

    s1 = tmp_path / "stage1.txt"
    s2 = tmp_path / "stage2.txt"
    s1.write_text(
        "CONFLICTS=243\nSTAGE1_STATUS=FAIL\nSTAGE1_STATUS=PASS\n",
        encoding="utf-8",
    )
    s2.write_text("STAGE2_STATUS=PASS\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate scalar STAGE1_STATUS"):
        build_forensic_resolutions_from_logs(s1, s2)


def test_forensic_log_parser_rejects_non_boolean_replay_flag(tmp_path):
    from std0_quant.events.publication_forensics_v2 import (
        build_forensic_resolutions_from_logs,
    )
    import json

    stage1_lines = ["CONFLICTS=243", "STAGE1_STATUS=PASS"]
    stage2_lines = [
        "STAGE2_TARGETS=150",
        "UNRESOLVED_COUNT=0",
        "TRUE_SEMANTIC_CONFLICT_ASSIGNED=0",
        "STAGE2_STATUS=PASS",
    ]

    for i in range(93):
        cid = f"coverage-{i:03d}"
        replay_required = "false" if i == 0 else False
        stage1_lines.append(
            "DETAIL="
            + json.dumps(
                {
                    "condition_id": cid,
                    "semantic_replay_required": replay_required,
                    "root_class_stage1": "COVERAGE_PATH_ONLY",
                    "raw_relation": "EXACT_RAW",
                }
            )
        )
        if i == 0:
            stage2_lines.append(
                "DETAIL2="
                + json.dumps(
                    {
                        "condition_id": cid,
                        "stage2_root_class":
                            "COVERAGE_EARLY_RETURN_MASKING_CANONICAL",
                    }
                )
            )

    # Keep the current truthiness-based parser at exactly 150 apparent replay
    # targets so this test detects the type bug, not a downstream count error.
    for i in range(150):
        cid = f"replay-{i:03d}"
        replay_required = False if i == 0 else True
        stage1_lines.append(
            "DETAIL="
            + json.dumps(
                {
                    "condition_id": cid,
                    "semantic_replay_required": replay_required,
                    "root_class_stage1": (
                        "COVERAGE_PATH_ONLY" if i == 0 else "EXACT_RAW"
                    ),
                    "raw_relation": "EXACT_RAW",
                }
            )
        )
        if replay_required:
            stage2_lines.append(
                "DETAIL2="
                + json.dumps(
                    {
                        "condition_id": cid,
                        "stage2_root_class":
                            "COVERAGE_EARLY_RETURN_MASKING_CANONICAL",
                    }
                )
            )

    s1 = tmp_path / "stage1.txt"
    s2 = tmp_path / "stage2.txt"
    s1.write_text("\n".join(stage1_lines) + "\n", encoding="utf-8")
    s2.write_text("\n".join(stage2_lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="semantic_replay_required must be boolean"):
        build_forensic_resolutions_from_logs(s1, s2)


def test_build_entrypoint_pins_every_external_evidence_input(
    tmp_path,
    monkeypatch,
):
    module = load_build_script_module()
    names = (
        "canonical_ledger",
        "prospective_ledger",
        "prospective_raw",
        "stage1_log",
        "stage2_log",
        "stage2_script",
        "feature_provenance",
    )
    paths = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes((name + "\n").encode())
        paths[name] = path

    pin_names = {
        "canonical_ledger": "PINNED_CANONICAL_SHA256",
        "prospective_ledger": "PINNED_PROSPECTIVE_LEDGER_SHA256",
        "prospective_raw": "PINNED_PROSPECTIVE_RAW_SHA256",
        "stage1_log": "PINNED_STAGE1_LOG_SHA256",
        "stage2_log": "PINNED_STAGE2_LOG_SHA256",
        "stage2_script": "PINNED_STAGE2_SCRIPT_SHA256",
        "feature_provenance": "PINNED_FEATURE_PROVENANCE_SHA256",
    }
    for name, constant in pin_names.items():
        monkeypatch.setattr(module, constant, file_sha256(paths[name]))

    args = SimpleNamespace(**paths)
    module.verify_pinned_inputs(args)

    paths["stage1_log"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="STAGE1_LOG_SHA_MISMATCH"):
        module.verify_pinned_inputs(args)


def test_build_dry_run_executes_frozen_source_guard_before_inputs(monkeypatch):
    module = load_build_script_module()
    args = SimpleNamespace(
        publish=False,
        canonical_ledger=Path("missing-canonical"),
        prospective_ledger=Path("missing-prospective"),
        prospective_raw=Path("missing-raw"),
        stage1_log=Path("missing-stage1"),
        stage2_log=Path("missing-stage2"),
        stage2_script=Path("missing-stage2-script"),
        feature_provenance=Path("missing-provenance"),
        output_dir=None,
    )
    calls = []
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(
        module,
        "verify_frozen_behavioral_sources",
        lambda root: calls.append("frozen"),
    )
    monkeypatch.setattr(
        module,
        "verify_pinned_inputs",
        lambda parsed: (_ for _ in ()).throw(RuntimeError("INPUT_STOP")),
    )

    with pytest.raises(RuntimeError, match="INPUT_STOP"):
        module.main()
    assert calls == ["frozen"]


def test_build_entrypoint_rejects_unpinned_publication_content(monkeypatch):
    module = load_build_script_module()
    monkeypatch.setattr(module, "PINNED_PUBLICATION_CONTENT_SHA256", "a" * 64)

    with pytest.raises(RuntimeError, match="PUBLICATION_CONTENT_SHA256_MISMATCH"):
        module.verify_publication_content_sha256(
            {"publication_content_sha256": "b" * 64}
        )


def test_staging_validation_rejects_serialized_provenance_sha_splice(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    module = load_build_script_module()
    publication_rows = [
        ledger("c1", coverage=None),
        {
            "condition_id": "c2",
            "y30": 0,
            "n_buy_fills": 1,
            # Parquet will materialize the coverage keys as explicit nulls.
        },
    ]
    reconciliation_rows = [
        {
            "condition_id": "c1",
            "resolution_class": "PROSPECTIVE_NEW",
        },
        {
            "condition_id": "c2",
            "resolution_class": "PROSPECTIVE_NEW",
        },
    ]
    final_truth_rel = "data/derived/publication_v2/behavioral_truth.parquet"

    behavioral_truth_path = tmp_path / module.BEHAVIORAL_TRUTH_NAME
    reconciliation_path = tmp_path / module.RECONCILIATION_NAME
    provenance_path = tmp_path / module.PROVENANCE_NAME
    report_path = tmp_path / module.REPORT_NAME
    bundle_manifest_path = tmp_path / module.BUNDLE_MANIFEST_NAME

    pq.write_table(pa.Table.from_pylist(publication_rows), behavioral_truth_path)
    pq.write_table(
        pa.Table.from_pylist(reconciliation_rows),
        reconciliation_path,
    )
    behavioral_truth_sha = file_sha256(behavioral_truth_path)
    spliced_provenance_rows = [
        {
            "condition_id": "c1",
            "feature_name": "old_direction_qty",
            "source_type": "phase1_truth",
            "source_file": final_truth_rel,
            "previous_source_file": "old.parquet",
            "source_artifact_sha256": "0" * 64,
            "source_membership_verified": True,
            "provenance_contract_version": "publication_provenance_v2",
        }
    ]
    pq.write_table(
        pa.Table.from_pylist(spliced_provenance_rows),
        provenance_path,
    )
    report = {
        "publication_content_sha256": rows_content_sha256(publication_rows)
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    bundle_manifest = {
        "outputs": {
            module.BEHAVIORAL_TRUTH_NAME: file_sha256(behavioral_truth_path),
            module.RECONCILIATION_NAME: file_sha256(reconciliation_path),
            module.PROVENANCE_NAME: file_sha256(provenance_path),
            module.REPORT_NAME: file_sha256(report_path),
        }
    }
    bundle_manifest_path.write_text(
        json.dumps(bundle_manifest),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="DECLARED_SOURCE_SHA_MISMATCH"):
        module.validate_staged_bundle(
            staging=tmp_path,
            pq=pq,
            publication_rows=publication_rows,
            reconciliation_rows=reconciliation_rows,
            provenance_rows=spliced_provenance_rows,
            report=report,
            bundle_manifest=bundle_manifest,
            final_truth_rel=final_truth_rel,
            behavioral_truth_sha=behavioral_truth_sha,
            provenance_expected=FullArtifactProvenanceExpectations(
                total_rows=1,
                phase1_rows=1,
                phase1_unique_conditions=1,
            ),
        )


def make_valid_staged_bundle(module, staging):
    import pyarrow.parquet as pq

    publication_rows = [ledger("c1", coverage=None)]
    reconciliation_rows = [
        {
            "condition_id": "c1",
            "resolution_class": "PROSPECTIVE_NEW",
        }
    ]
    input_provenance_rows = [
        {
            "condition_id": "c1",
            "feature_name": "old_direction_qty",
            "source_type": "phase1_truth",
            "source_file": "old.parquet",
        }
    ]
    report = {
        "publication_content_sha256": rows_content_sha256(publication_rows)
    }
    builder_state = {
        "builder_branch": "fix/publication-provenance-v2",
        "builder_head": "f" * 40,
        "builder_worktree_clean": False,
        "builder_code_sha256": {
            path: hashlib.sha256(path.encode()).hexdigest()
            for path in module.BUILDER_CODE_PATHS
        },
    }
    expected = FullArtifactProvenanceExpectations(
        total_rows=1,
        phase1_rows=1,
        phase1_unique_conditions=1,
    )
    result = module.build_validated_staged_bundle(
        root=Path(__file__).resolve().parents[1],
        staging=staging,
        pq=pq,
        atomic_write_parquet=module.atomic_write_parquet_v2,
        publication_rows=publication_rows,
        reconciliation_rows=reconciliation_rows,
        input_provenance_rows=input_provenance_rows,
        report=report,
        builder_state=builder_state,
        provenance_expected=expected,
    )
    provenance_rows = pq.read_table(
        staging / module.PROVENANCE_NAME
    ).to_pylist()
    return {
        "pq": pq,
        "publication_rows": publication_rows,
        "reconciliation_rows": reconciliation_rows,
        "provenance_rows": provenance_rows,
        "report": report,
        "bundle_manifest": result["bundle_manifest"],
        "final_truth_rel": (
            module.OUTPUT_RELATIVE_DIR / module.BEHAVIORAL_TRUTH_NAME
        ).as_posix(),
        "behavioral_truth_sha": result["behavioral_truth_sha256"],
        "provenance_expected": expected,
    }


def validate_fixture_bundle(module, staging, fixture):
    module.validate_staged_bundle(staging=staging, **fixture)


def test_staging_validation_rejects_serialized_provenance_tamper(tmp_path):
    import pyarrow as pa

    module = load_build_script_module()
    fixture = make_valid_staged_bundle(module, tmp_path)
    tampered = copy.deepcopy(fixture["provenance_rows"])
    tampered[0]["feature_name"] = "tampered"
    fixture["pq"].write_table(
        pa.Table.from_pylist(tampered),
        tmp_path / module.PROVENANCE_NAME,
    )

    with pytest.raises(RuntimeError, match="STAGED_PROVENANCE_SEMANTICS_MISMATCH"):
        validate_fixture_bundle(module, tmp_path, fixture)


def test_staging_validation_rejects_serialized_report_tamper(tmp_path):
    module = load_build_script_module()
    fixture = make_valid_staged_bundle(module, tmp_path)
    (tmp_path / module.REPORT_NAME).write_text(
        json.dumps({"publication_content_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="STAGED_REPORT_SEMANTICS_MISMATCH"):
        validate_fixture_bundle(module, tmp_path, fixture)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_staging_validation_rejects_wrong_file_set(tmp_path, mutation):
    module = load_build_script_module()
    fixture = make_valid_staged_bundle(module, tmp_path)
    if mutation == "missing":
        (tmp_path / module.PROVENANCE_NAME).unlink()
    else:
        (tmp_path / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(RuntimeError, match="STAGING_FILE_SET_MISMATCH"):
        validate_fixture_bundle(module, tmp_path, fixture)


def test_verified_snapshots_isolate_consumption_from_source_mutation(
    tmp_path,
    monkeypatch,
):
    module = load_build_script_module()
    names = (
        "canonical_ledger",
        "prospective_ledger",
        "prospective_raw",
        "stage1_log",
        "stage2_log",
        "stage2_script",
        "feature_provenance",
    )
    paths = {}
    original_bytes = {}
    for name in names:
        path = tmp_path / name
        content = (name + "-pinned\n").encode()
        path.write_bytes(content)
        paths[name] = path
        original_bytes[name] = content

    pin_names = {
        "canonical_ledger": "PINNED_CANONICAL_SHA256",
        "prospective_ledger": "PINNED_PROSPECTIVE_LEDGER_SHA256",
        "prospective_raw": "PINNED_PROSPECTIVE_RAW_SHA256",
        "stage1_log": "PINNED_STAGE1_LOG_SHA256",
        "stage2_log": "PINNED_STAGE2_LOG_SHA256",
        "stage2_script": "PINNED_STAGE2_SCRIPT_SHA256",
        "feature_provenance": "PINNED_FEATURE_PROVENANCE_SHA256",
    }
    for name, constant in pin_names.items():
        monkeypatch.setattr(module, constant, file_sha256(paths[name]))

    args = SimpleNamespace(**paths)
    snapshot_root = None
    with module.verified_input_snapshots(args) as snapshots:
        snapshot_root = snapshots.stage1_log.parent
        paths["stage1_log"].write_bytes(b"mutated-after-snapshot\n")
        assert snapshots.stage1_log.read_bytes() == original_bytes["stage1_log"]
        paths["stage1_log"].write_bytes(original_bytes["stage1_log"])
        assert snapshots.stage1_log.read_bytes() == original_bytes["stage1_log"]
        module.verify_pinned_inputs(snapshots)

    assert snapshot_root is not None
    assert not snapshot_root.exists()


def make_snapshot_args(module, tmp_path, monkeypatch):
    pin_names = {
        "canonical_ledger": "PINNED_CANONICAL_SHA256",
        "prospective_ledger": "PINNED_PROSPECTIVE_LEDGER_SHA256",
        "prospective_raw": "PINNED_PROSPECTIVE_RAW_SHA256",
        "stage1_log": "PINNED_STAGE1_LOG_SHA256",
        "stage2_log": "PINNED_STAGE2_LOG_SHA256",
        "stage2_script": "PINNED_STAGE2_SCRIPT_SHA256",
        "feature_provenance": "PINNED_FEATURE_PROVENANCE_SHA256",
    }
    paths = {}
    for name, constant in pin_names.items():
        path = tmp_path / name
        path.write_bytes((name + "-pinned\n").encode())
        paths[name] = path
        monkeypatch.setattr(module, constant, file_sha256(path))
    return SimpleNamespace(**paths)


def test_verified_snapshots_reject_sha_mismatch_and_clean_temp(
    tmp_path,
    monkeypatch,
):
    module = load_build_script_module()
    args = make_snapshot_args(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "PINNED_STAGE1_LOG_SHA256", "0" * 64)
    copied_roots = []
    real_copy = module.shutil.copyfile

    def recording_copy(source, target):
        copied_roots.append(Path(target).parent)
        return real_copy(source, target)

    monkeypatch.setattr(module.shutil, "copyfile", recording_copy)
    with pytest.raises(RuntimeError, match="SNAPSHOT_STAGE1_LOG_SHA_MISMATCH"):
        with module.verified_input_snapshots(args):
            pytest.fail("mismatched snapshot must not be yielded")

    assert copied_roots
    assert all(not path.exists() for path in copied_roots)


def test_verified_snapshots_reject_missing_snapshot_and_clean_temp(
    tmp_path,
    monkeypatch,
):
    module = load_build_script_module()
    args = make_snapshot_args(module, tmp_path, monkeypatch)
    copied_roots = []
    real_copy = module.shutil.copyfile

    def omit_stage1_snapshot(source, target):
        copied_roots.append(Path(target).parent)
        if Path(source) == args.stage1_log:
            return str(target)
        return real_copy(source, target)

    monkeypatch.setattr(module.shutil, "copyfile", omit_stage1_snapshot)
    with pytest.raises(FileNotFoundError):
        with module.verified_input_snapshots(args):
            pytest.fail("missing snapshot must not be yielded")

    assert copied_roots
    assert all(not path.exists() for path in copied_roots)


def test_verified_snapshots_clean_temp_when_consumer_raises(
    tmp_path,
    monkeypatch,
):
    module = load_build_script_module()
    args = make_snapshot_args(module, tmp_path, monkeypatch)
    snapshot_root = None

    with pytest.raises(RuntimeError, match="consumer failed"):
        with module.verified_input_snapshots(args) as snapshots:
            snapshot_root = snapshots.stage1_log.parent
            raise RuntimeError("consumer failed")

    assert snapshot_root is not None
    assert not snapshot_root.exists()


def test_private_staged_validation_always_cleans_temp_bundle():
    module = load_build_script_module()
    seen = []

    def validate(staging):
        seen.append(staging)
        (staging / "sentinel").write_text("validated", encoding="utf-8")
        return "PASS"

    assert module.run_private_staged_validation(validate) == "PASS"
    assert len(seen) == 1
    assert not seen[0].exists()


def test_private_staged_validation_cleans_temp_bundle_on_exception():
    module = load_build_script_module()
    seen = []

    def validate(staging):
        seen.append(staging)
        (staging / "sentinel").write_text("partial", encoding="utf-8")
        raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError, match="validation failed"):
        module.run_private_staged_validation(validate)

    assert len(seen) == 1
    assert not seen[0].exists()


def test_publish_builder_policy_rejects_dirty_worktree(monkeypatch):
    module = load_build_script_module()
    state = {
        "builder_branch": "main",
        "builder_head": "f" * 40,
        "builder_worktree_clean": False,
        "builder_code_sha256": {},
    }
    monkeypatch.setattr(module, "read_builder_state", lambda root: state)

    with pytest.raises(RuntimeError, match="BUILDER_WORKTREE_NOT_CLEAN"):
        module.verify_publish_builder_policy(
            Path("."),
            expected_branch="main",
            expected_head="f" * 40,
        )


@pytest.mark.parametrize(
    ("actual_branch", "actual_head", "expected_branch", "expected_head", "error"),
    [
        (
            "fix/publication-provenance-v2",
            "f" * 40,
            "main",
            "f" * 40,
            "BUILDER_BRANCH_MISMATCH",
        ),
        (
            "main",
            "e" * 40,
            "main",
            "f" * 40,
            "BUILDER_HEAD_MISMATCH",
        ),
    ],
)
def test_publish_builder_policy_rejects_wrong_identity(
    monkeypatch,
    actual_branch,
    actual_head,
    expected_branch,
    expected_head,
    error,
):
    module = load_build_script_module()
    state = {
        "builder_branch": actual_branch,
        "builder_head": actual_head,
        "builder_worktree_clean": True,
        "builder_code_sha256": {},
    }
    monkeypatch.setattr(module, "read_builder_state", lambda root: state)

    with pytest.raises(RuntimeError, match=error):
        module.verify_publish_builder_policy(
            Path("."),
            expected_branch=expected_branch,
            expected_head=expected_head,
        )


def test_publish_builder_policy_rejects_branch_outside_allowlist(monkeypatch):
    module = load_build_script_module()
    state = {
        "builder_branch": "fix/publication-provenance-v2",
        "builder_head": "f" * 40,
        "builder_worktree_clean": True,
        "builder_code_sha256": {},
    }
    monkeypatch.setattr(module, "read_builder_state", lambda root: state)

    with pytest.raises(RuntimeError, match="PUBLISH_BUILDER_BRANCH_NOT_ALLOWED"):
        module.verify_publish_builder_policy(
            Path("."),
            expected_branch="fix/publication-provenance-v2",
            expected_head="f" * 40,
        )


def test_publish_builder_policy_accepts_exact_clean_main_identity(monkeypatch):
    module = load_build_script_module()
    state = {
        "builder_branch": "main",
        "builder_head": "f" * 40,
        "builder_worktree_clean": True,
        "builder_code_sha256": {"module.py": "a" * 64},
    }
    monkeypatch.setattr(module, "read_builder_state", lambda root: state)

    assert module.verify_publish_builder_policy(
        Path("."),
        expected_branch="main",
        expected_head="f" * 40,
    ) == state


def test_publish_rejects_builder_module_bytes_changed_during_build():
    module = load_build_script_module()
    initial = {
        "builder_branch": "fix/publication-provenance-v2",
        "builder_head": "f" * 40,
        "builder_worktree_clean": True,
        "builder_code_sha256": {"module.py": "a" * 64},
    }
    changed = copy.deepcopy(initial)
    changed["builder_code_sha256"]["module.py"] = "b" * 64

    with pytest.raises(RuntimeError, match="BUILDER_CODE_SHA_CHANGED_DURING_BUILD"):
        module.verify_builder_state_unchanged(initial, changed)


def test_dry_run_manifest_records_dirty_builder_and_flat_module_hashes(tmp_path):
    module = load_build_script_module()
    fixture = make_valid_staged_bundle(module, tmp_path)
    manifest = fixture["bundle_manifest"]

    assert manifest["builder_worktree_clean"] is False
    assert manifest["builder_branch"] == "fix/publication-provenance-v2"
    assert manifest["builder_head"] == "f" * 40
    for relative, manifest_key in module.BUILDER_MANIFEST_SHA_KEYS.items():
        assert manifest[manifest_key] == manifest["builder_code_sha256"][relative]


def test_atomic_noreplace_success(tmp_path):
    module = load_build_script_module()
    staging = tmp_path / "staging"
    final = tmp_path / "publication_v2"
    staging.mkdir()
    (staging / "new").write_text("new", encoding="utf-8")

    module.atomic_publish_directory_noreplace(
        staging,
        final,
        _rename_noreplace=lambda source, target: source.rename(target),
    )

    assert not staging.exists()
    assert (final / "new").read_text(encoding="utf-8") == "new"


def test_atomic_noreplace_rejects_existing_empty_target(tmp_path):
    module = load_build_script_module()
    staging = tmp_path / "staging"
    final = tmp_path / "publication_v2"
    staging.mkdir()
    final.mkdir()

    with pytest.raises(RuntimeError, match="PUBLICATION_TARGET_ALREADY_EXISTS"):
        module.atomic_publish_directory_noreplace(staging, final)

    assert staging.is_dir()
    assert final.is_dir()


def test_atomic_noreplace_unavailable_fails_closed(tmp_path):
    module = load_build_script_module()
    staging = tmp_path / "staging"
    final = tmp_path / "publication_v2"
    staging.mkdir()

    def unavailable(source, target):
        raise RuntimeError("ATOMIC_NOREPLACE_UNAVAILABLE_RENAMEAT2")

    with pytest.raises(RuntimeError, match="ATOMIC_NOREPLACE_UNAVAILABLE"):
        module.atomic_publish_directory_noreplace(
            staging,
            final,
            _rename_noreplace=unavailable,
        )

    assert staging.is_dir()
    assert not final.exists()


def test_atomic_noreplace_race_preserves_existing_target(tmp_path):
    module = load_build_script_module()
    staging = tmp_path / "staging"
    final = tmp_path / "publication_v2"
    staging.mkdir()
    (staging / "new").write_text("new", encoding="utf-8")

    def racing_rename(source, target):
        assert source == staging
        assert target == final
        target.mkdir()
        (target / "existing").write_text("existing", encoding="utf-8")
        raise FileExistsError("simulated concurrent publisher")

    with pytest.raises(RuntimeError, match="PUBLICATION_TARGET_ALREADY_EXISTS"):
        module.atomic_publish_directory_noreplace(
            staging,
            final,
            _rename_noreplace=racing_rename,
        )

    assert (final / "existing").read_text(encoding="utf-8") == "existing"
    assert (staging / "new").read_text(encoding="utf-8") == "new"


def test_v2_atomic_parquet_writer_roundtrip_and_no_temp(tmp_path):
    import pyarrow.parquet as pq

    module = load_build_script_module()
    target = tmp_path / "artifact.parquet"
    rows = [{"condition_id": "c1", "value": 1}]

    module.atomic_write_parquet_v2(rows, target)

    assert pq.read_table(target).to_pylist() == rows
    assert not list(tmp_path.glob(".artifact.parquet.*.tmp"))


def test_v2_atomic_json_writer_roundtrip_and_no_temp(tmp_path):
    module = load_build_script_module()
    target = tmp_path / "artifact.json"
    payload = {"status": "PASS", "rows": 3}

    module.atomic_write_json_v2(target, payload)

    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert target.read_bytes().endswith(b"\n")
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))


@pytest.mark.parametrize("failure_point", ["write", "flush"])
def test_v2_atomic_json_writer_cleans_temp_after_write_or_flush_failure(
    tmp_path,
    monkeypatch,
    failure_point,
):
    module = load_build_script_module()
    target = tmp_path / "artifact.json"
    target.write_text("old\n", encoding="utf-8")
    real_open = Path.open

    class FailingBufferedWriter:
        def __init__(self, path):
            self.path = path
            self.inner = None

        def __enter__(self):
            self.inner = real_open(self.path, "xb")
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.inner.close()

        def write(self, data):
            if failure_point == "write":
                self.inner.write(b"partial")
                raise OSError("simulated write failure")
            return self.inner.write(data)

        def flush(self):
            self.inner.flush()
            if failure_point == "flush":
                raise OSError("simulated flush failure")

        def fileno(self):
            return self.inner.fileno()

    def failing_open(path, mode="r", *args, **kwargs):
        if Path(path).name.startswith(".artifact.json.") and mode == "xb":
            return FailingBufferedWriter(path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match=failure_point):
        module.atomic_write_json_v2(target, {"status": "PASS"})

    assert target.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))


def test_v2_atomic_json_writer_cleans_temp_after_fsync_failure(
    tmp_path,
    monkeypatch,
):
    module = load_build_script_module()
    target = tmp_path / "artifact.json"

    def fail_fsync(fd):
        raise OSError("simulated JSON fsync failure")

    monkeypatch.setattr(module, "_flush_file_descriptor", fail_fsync)
    with pytest.raises(OSError, match="JSON fsync failure"):
        module.atomic_write_json_v2(target, {"status": "PASS"})

    assert not target.exists()
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))


def test_durable_publish_staging_fsync_failure_prevents_rename(tmp_path):
    module = load_build_script_module()
    staging = tmp_path / "staging"
    final = tmp_path / "publication_v2"
    staging.mkdir()
    renamed = []

    def open_directory(path):
        return 10

    def fail_staging_fsync(fd):
        assert fd == 10
        raise OSError("staging fsync failure")

    with pytest.raises(OSError, match="staging fsync failure"):
        module.durable_publish_directory_v2(
            staging,
            final,
            _open_directory=open_directory,
            _fsync_directory=fail_staging_fsync,
            _close_fd=lambda fd: None,
            _publish_noreplace=lambda source, target: renamed.append(True),
        )

    assert renamed == []
    assert staging.is_dir()
    assert not final.exists()


def test_durable_publish_parent_open_failure_prevents_rename(tmp_path):
    module = load_build_script_module()
    staging = tmp_path / "staging"
    final = tmp_path / "publication_v2"
    staging.mkdir()
    calls = []

    def open_directory(path):
        calls.append(("open", Path(path).name))
        if Path(path) == final.parent:
            raise OSError("parent open failure")
        return 10

    with pytest.raises(OSError, match="parent open failure"):
        module.durable_publish_directory_v2(
            staging,
            final,
            _open_directory=open_directory,
            _fsync_directory=lambda fd: calls.append(("fsync", fd)),
            _close_fd=lambda fd: calls.append(("close", fd)),
            _publish_noreplace=lambda source, target: calls.append(("rename",)),
        )

    assert ("rename",) not in calls
    assert staging.is_dir()
    assert not final.exists()


def test_durable_publish_fsyncs_parent_after_rename(tmp_path):
    module = load_build_script_module()
    staging = tmp_path / "staging"
    final = tmp_path / "publication_v2"
    staging.mkdir()
    (staging / "artifact").write_text("ready", encoding="utf-8")
    calls = []
    descriptors = iter((10, 20))

    def publish(source, target):
        calls.append(("rename",))
        source.rename(target)

    module.durable_publish_directory_v2(
        staging,
        final,
        _open_directory=lambda path: next(descriptors),
        _fsync_directory=lambda fd: calls.append(("fsync", fd)),
        _close_fd=lambda fd: calls.append(("close", fd)),
        _publish_noreplace=publish,
    )

    assert calls == [
        ("fsync", 10),
        ("close", 10),
        ("rename",),
        ("fsync", 20),
        ("close", 20),
    ]
    assert (final / "artifact").read_text(encoding="utf-8") == "ready"


def test_durable_publish_parent_fsync_failure_is_explicit_terminal_state(
    tmp_path,
):
    module = load_build_script_module()
    staging = tmp_path / "staging"
    final = tmp_path / "publication_v2"
    staging.mkdir()
    (staging / "artifact").write_text("renamed", encoding="utf-8")
    descriptors = iter((10, 20))

    def fsync_directory(fd):
        if fd == 20:
            raise OSError("parent fsync failure")

    with pytest.raises(
        RuntimeError,
        match="PUBLICATION_RENAMED_DURABILITY_UNCONFIRMED",
    ):
        module.durable_publish_directory_v2(
            staging,
            final,
            _open_directory=lambda path: next(descriptors),
            _fsync_directory=fsync_directory,
            _close_fd=lambda fd: None,
            _publish_noreplace=lambda source, target: source.rename(target),
        )

    assert not staging.exists()
    assert (final / "artifact").read_text(encoding="utf-8") == "renamed"


def test_forensic_log_parser_reconstructs_frozen_243_resolution_counts(tmp_path):
    import json
    from collections import Counter
    from std0_quant.events.publication_forensics_v2 import (
        build_forensic_resolutions_from_logs,
    )

    stage1_lines = ["CONFLICTS=243", "STAGE1_STATUS=PASS"]
    stage2_lines = [
        "STAGE2_TARGETS=150",
        "UNRESOLVED_COUNT=0",
        "TRUE_SEMANTIC_CONFLICT_ASSIGNED=0",
        "STAGE2_STATUS=PASS",
    ]

    # 93 Stage-1-only coverage-path cases.
    for i in range(93):
        cid = f"cov-{i:03d}"
        stage1_lines.append(
            "DETAIL=" + json.dumps({
                "condition_id": cid,
                "semantic_replay_required": False,
                "root_class_stage1": "COVERAGE_PATH_ONLY",
                "raw_relation": "EXACT_RAW",
                "ledger_diff_fields": ["btc_coverage_pct"],
            })
        )

    # 147 coverage early-return cases.
    for i in range(147):
        cid = f"early-{i:03d}"
        stage1_lines.append(
            "DETAIL=" + json.dumps({
                "condition_id": cid,
                "semantic_replay_required": True,
                "root_class_stage1": "EXACT_RAW",
                "raw_relation": "EXACT_RAW",
                "ledger_diff_fields": ["y30"],
            })
        )
        stage2_lines.append(
            "DETAIL2=" + json.dumps({
                "condition_id": cid,
                "stage2_root_class":
                    "COVERAGE_EARLY_RETURN_MASKING_CANONICAL",
                "historical_vs_prospective_replay_diff_fields": [],
            })
        )

    # 1 canonical build-provenance gap.
    cid = "gap-000"
    stage1_lines.append(
        "DETAIL=" + json.dumps({
            "condition_id": cid,
            "semantic_replay_required": True,
            "root_class_stage1": "EXACT_RAW",
            "raw_relation": "EXACT_RAW",
            "ledger_diff_fields": ["n_buy_fills"],
        })
    )
    stage2_lines.append(
        "DETAIL2=" + json.dumps({
            "condition_id": cid,
            "stage2_root_class":
                "CURRENT_FROZEN_REPLAY_SUPPORTS_PROSPECTIVE",
            "historical_vs_prospective_replay_diff_fields": [],
        })
    )

    # 2 raw-snapshot-explained cases.
    for i in range(2):
        cid = f"snapshot-{i:03d}"
        stage1_lines.append(
            "DETAIL=" + json.dumps({
                "condition_id": cid,
                "semantic_replay_required": True,
                "root_class_stage1": "CANONICAL_RAW_SUPERSET",
                "raw_relation": "CANONICAL_RAW_SUPERSET",
                "ledger_diff_fields": ["n_buy_fills"],
            })
        )
        stage2_lines.append(
            "DETAIL2=" + json.dumps({
                "condition_id": cid,
                "stage2_root_class":
                    "RAW_SNAPSHOT_EXPLAINS_ARTIFACT_DIFFERENCE",
                "historical_vs_prospective_replay_diff_fields":
                    ["n_buy_fills"],
            })
        )

    s1 = tmp_path / "stage1.txt"
    s2 = tmp_path / "stage2.txt"
    s1.write_text("\n".join(stage1_lines) + "\n", encoding="utf-8")
    s2.write_text("\n".join(stage2_lines) + "\n", encoding="utf-8")

    rows, report = build_forensic_resolutions_from_logs(s1, s2)

    assert len(rows) == 243
    assert report["resolution_counts"] == {
        "CANONICAL_ARTIFACT_NOT_REPRODUCIBLE_UNDER_PINNED_SEMANTICS": 1,
        "COVERAGE_EARLY_RETURN_MASKING_CANONICAL": 147,
        "COVERAGE_PATH_ONLY": 93,
        "RAW_SNAPSHOT_EXPLAINS_ARTIFACT_DIFFERENCE": 2,
    }


def test_provenance_preflight_membership_and_frozen_counts():
    rows = [
        {
            "condition_id": "c1",
            "feature_name": "x",
            "source_type": "phase1_truth",
            "source_file": "old.parquet",
        },
        {
            "condition_id": "c1",
            "feature_name": "y",
            "source_type": "phase1_truth",
            "source_file": "old.parquet",
        },
        {
            "condition_id": "c2",
            "feature_name": "btc_ret_1s",
            "source_type": "binance_btc",
            "source_file": "btc.ndjson",
        },
    ]
    report = preflight_phase1_provenance_membership(
        rows,
        publication_rows=[ledger("c1", coverage=None)],
        expected=FullArtifactProvenanceExpectations(
            total_rows=3,
            phase1_rows=2,
            phase1_unique_conditions=1,
        ),
    )
    assert report["membership_failures"] == 0
    assert report["membership_pass_conditions"] == 1

    with pytest.raises(AssertionError, match="absent from publication"):
        preflight_phase1_provenance_membership(
            rows,
            publication_rows=[ledger("other", coverage=None)],
        )

    with pytest.raises(AssertionError, match="row expectation mismatch"):
        preflight_phase1_provenance_membership(
            rows,
            publication_rows=[ledger("c1", coverage=None)],
            expected=FullArtifactProvenanceExpectations(
                total_rows=3,
                phase1_rows=3,
                phase1_unique_conditions=1,
            ),
        )


def test_full_artifact_provenance_counts_pass_with_distribution():
    rows = [
        {
            "condition_id": cid,
            "feature_name": f"phase1-{number}",
            "source_type": "phase1_truth",
            "source_file": "old.parquet",
        }
        for cid, count in (("c1", 2), ("c2", 3))
        for number in range(count)
    ]
    rows.append(
        {
            "condition_id": "c1",
            "feature_name": "btc-ret",
            "source_type": "binance_btc",
            "source_file": "btc.ndjson",
        }
    )

    report = preflight_phase1_provenance_membership(
        rows,
        publication_rows=[
            ledger("c1", coverage=None),
            ledger("c2", coverage=None),
        ],
        expected=FullArtifactProvenanceExpectations(
            total_rows=6,
            phase1_rows=5,
            phase1_unique_conditions=2,
        ),
    )

    assert report["status"] == "PASS"
    assert report["contract_scope"] == "FULL_ARTIFACT"
    assert report["phase1_rows_per_condition_min"] == 2
    assert report["phase1_rows_per_condition_max"] == 3
    assert report["phase1_rows_per_condition_histogram"] == {2: 1, 3: 1}
    assert report["phase1_null_condition_ids"] == 0
    assert report["membership_failures"] == 0


def test_candidate_subset_counts_pass_independently():
    candidate_rows = [
        {
            "condition_id": f"candidate-{condition:03d}",
            "feature_name": f"phase1-{feature:02d}",
            "source_type": "phase1_truth",
        }
        for condition in range(86)
        for feature in range(16)
    ]

    report = audit_candidate_subset_provenance(
        candidate_rows,
        expected=CandidateSubsetProvenanceExpectations(
            phase1_rows=1376,
            phase1_unique_conditions=86,
        ),
    )

    assert report == {
        "status": "PASS",
        "contract_scope": "CANDIDATE_SUBSET",
        "phase1_rows": 1376,
        "phase1_unique_conditions": 86,
        "phase1_rows_per_condition_min": 16,
        "phase1_rows_per_condition_max": 16,
        "phase1_rows_per_condition_histogram": {16: 86},
    }


def test_full_and_candidate_expectations_cannot_be_mixed():
    full = FullArtifactProvenanceExpectations(
        total_rows=1,
        phase1_rows=1,
        phase1_unique_conditions=1,
    )
    candidate = CandidateSubsetProvenanceExpectations(
        phase1_rows=1,
        phase1_unique_conditions=1,
    )
    rows = [
        {
            "condition_id": "c1",
            "feature_name": "x",
            "source_type": "phase1_truth",
        }
    ]

    with pytest.raises(TypeError, match="FullArtifactProvenanceExpectations"):
        preflight_phase1_provenance_membership(
            rows,
            publication_rows=[ledger("c1", coverage=None)],
            expected=candidate,
        )

    with pytest.raises(
        TypeError,
        match="CandidateSubsetProvenanceExpectations",
    ):
        audit_candidate_subset_provenance(rows, expected=full)


def test_full_artifact_wrong_total_row_count_fails():
    rows = [
        {
            "condition_id": "c1",
            "feature_name": "x",
            "source_type": "phase1_truth",
        }
    ]

    with pytest.raises(AssertionError, match="total-row expectation mismatch"):
        preflight_phase1_provenance_membership(
            rows,
            publication_rows=[ledger("c1", coverage=None)],
            expected=FullArtifactProvenanceExpectations(
                total_rows=2,
                phase1_rows=1,
                phase1_unique_conditions=1,
            ),
        )


def test_full_artifact_wrong_unique_condition_count_fails():
    rows = [
        {
            "condition_id": "c1",
            "feature_name": feature,
            "source_type": "phase1_truth",
        }
        for feature in ("x", "y")
    ]

    with pytest.raises(AssertionError, match="condition expectation mismatch"):
        preflight_phase1_provenance_membership(
            rows,
            publication_rows=[ledger("c1", coverage=None)],
            expected=FullArtifactProvenanceExpectations(
                total_rows=2,
                phase1_rows=2,
                phase1_unique_conditions=2,
            ),
        )
