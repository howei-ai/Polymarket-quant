#!/usr/bin/env python3
"""Build Publication / Provenance Repair v2.

Default mode is DRY RUN and writes only a fail-safe system-temp validation
bundle that is deleted before exit; it writes no project output. ``--publish`` is deliberately
fail-closed: it requires the pinned inputs, complete Stage-1/Stage-2 forensic
evidence, the pinned feature-provenance artifact, unchanged frozen behavioral
source files relative to the pinned main commit, and an absent versioned target
directory.

This script never reads or writes formal-cohort state and never runs backtests.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PINNED_MAIN = "e9246443aca33155e6cbb34cc4d592f9e423eded"
PINNED_STAGE1_LOG_SHA256 = "946dcb144f3d11f75aade2635505c4fb634f4fc676612ebdd85cf0e0fc4e9054"
PINNED_STAGE2_LOG_SHA256 = "13d264dc259ebc12c9eadbe870c27fefc83623d7d5fa5e350c6df12cc7f635b8"
PINNED_STAGE2_SCRIPT_SHA256 = "818a7f2a286964c00586dee41d5a567d3129c01ad339204eb616de5f7f117344"
PINNED_CANONICAL_SHA256 = "63d58236fdfa88dbf0e35e5da74b16db8b40259368929d24d1291644b93b52dd"
PINNED_PROSPECTIVE_LEDGER_SHA256 = "ff818c83860d031c080d9a4f97b6035dfbae417a7ab96c8e17837e05632b3770"
PINNED_PROSPECTIVE_RAW_SHA256 = "ddc5eee63b3f58451c9620161518884708e6da724a95bd9c29125cee8c8620d9"
PINNED_FEATURE_PROVENANCE_SHA256 = "347ff2d7ea282bb9454a0379f243080c0ed83a39620de46d612449acb29a0bec"
PINNED_PUBLICATION_CONTENT_SHA256 = "30be62ae728f09db145734237dd2e84d4b2619d235cfc915fbd04aaf6a75d045"

OUTPUT_RELATIVE_DIR = Path("data/derived/publication_v2")
BEHAVIORAL_TRUTH_NAME = "behavioral_truth.parquet"
RECONCILIATION_NAME = "reconciliation_manifest.parquet"
PROVENANCE_NAME = "provenance.parquet"
REPORT_NAME = "publication_report.json"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"

# The repair is allowed to add new modules/scripts/tests, but the frozen
# behavioral semantics used for replay must remain byte-identical to the pinned
# base commit.
FROZEN_BEHAVIORAL_PATHS = (
    "src/std0_quant/events/event_ledger.py",
    "src/std0_quant/events/episode_builder.py",
    "src/std0_quant/events/first_opposite.py",
    "src/std0_quant/events/fills.py",
    "src/std0_quant/collectors/std0_trades.py",
    "src/std0_quant/timeutil.py",
)

BUILDER_CODE_PATHS = (
    "scripts/build_publication_provenance_v2.py",
    "src/std0_quant/events/publication_forensics_v2.py",
    "src/std0_quant/events/publication_provenance_v2.py",
    "src/std0_quant/features/provenance_v2.py",
)
BUILDER_MANIFEST_SHA_KEYS = {
    "scripts/build_publication_provenance_v2.py": "build_script_sha256",
    "src/std0_quant/events/publication_forensics_v2.py": (
        "publication_forensics_v2_sha256"
    ),
    "src/std0_quant/events/publication_provenance_v2.py": (
        "publication_provenance_v2_sha256"
    ),
    "src/std0_quant/features/provenance_v2.py": "provenance_v2_sha256",
}
ALLOWED_PUBLICATION_BRANCHES = frozenset({"main"})

SNAPSHOT_INPUT_NAMES = {
    "stage1_log": "stage1_full_output.txt",
    "stage2_log": "stage2_full_detail_output.txt",
    "stage2_script": "replay_150_conflicts_stage2.py",
    "canonical_ledger": "canonical_event_ledger.parquet",
    "prospective_ledger": "prospective_event_ledger.parquet",
    "prospective_raw": "prospective_trades.ndjson",
    "feature_provenance": "feature_provenance.parquet",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    print(f"{label}_SHA256={actual}")
    if actual != expected:
        raise RuntimeError(f"{label}_SHA_MISMATCH")


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def verify_frozen_behavioral_sources(root: Path) -> None:
    inside = git(root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise RuntimeError("GIT_WORKTREE_REQUIRED_FOR_FROZEN_SOURCE_VERIFICATION")

    ancestor = git(root, "merge-base", "--is-ancestor", PINNED_MAIN, "HEAD")
    if ancestor.returncode != 0:
        raise RuntimeError("PINNED_MAIN_IS_NOT_ANCESTOR_OF_HEAD")

    diff = git(root, "diff", "--quiet", PINNED_MAIN, "--", *FROZEN_BEHAVIORAL_PATHS)
    if diff.returncode == 1:
        raise RuntimeError("FROZEN_BEHAVIORAL_SOURCE_DIFFERS_FROM_PINNED_MAIN")
    if diff.returncode != 0:
        raise RuntimeError(
            "FROZEN_BEHAVIORAL_SOURCE_VERIFICATION_FAILED:"
            + diff.stderr.strip()[:200]
        )

    print("FROZEN_BEHAVIORAL_SOURCES=PASS")


def verify_pinned_inputs(
    args: argparse.Namespace,
    *,
    label_prefix: str = "",
) -> None:
    """Require every external evidence artifact before any replay or write."""
    pins = (
        (args.stage1_log, PINNED_STAGE1_LOG_SHA256, "STAGE1_LOG"),
        (args.stage2_log, PINNED_STAGE2_LOG_SHA256, "STAGE2_LOG"),
        (args.stage2_script, PINNED_STAGE2_SCRIPT_SHA256, "STAGE2_SCRIPT"),
        (args.canonical_ledger, PINNED_CANONICAL_SHA256, "CANONICAL_LEDGER"),
        (
            args.prospective_ledger,
            PINNED_PROSPECTIVE_LEDGER_SHA256,
            "PROSPECTIVE_LEDGER",
        ),
        (args.prospective_raw, PINNED_PROSPECTIVE_RAW_SHA256, "PROSPECTIVE_RAW"),
        (
            args.feature_provenance,
            PINNED_FEATURE_PROVENANCE_SHA256,
            "FEATURE_PROVENANCE",
        ),
    )
    for path, expected, label in pins:
        effective_label = label_prefix + label
        if not isinstance(path, Path):
            raise RuntimeError(f"{effective_label}_PATH_REQUIRED")
        require_sha(path, expected, effective_label)


@contextmanager
def verified_input_snapshots(args: argparse.Namespace):
    """Copy once, verify the copies, and expose only immutable input paths."""
    with tempfile.TemporaryDirectory(
        prefix="std0-publication-v2-input-snapshots-"
    ) as temp_name:
        snapshot_root = Path(temp_name)
        snapshot_values = vars(args).copy()
        for attribute, filename in SNAPSHOT_INPUT_NAMES.items():
            source = getattr(args, attribute)
            if not isinstance(source, Path) or not source.is_file():
                raise FileNotFoundError(source)
            snapshot = snapshot_root / filename
            shutil.copyfile(source, snapshot)
            snapshot_values[attribute] = snapshot

        snapshots = argparse.Namespace(**snapshot_values)
        verify_pinned_inputs(snapshots, label_prefix="SNAPSHOT_")
        print("SNAPSHOT_INPUTS=7/7")
        print("SNAPSHOT_SHA_MATCH=7/7")
        print("SNAPSHOT_PINNING=PASS")
        yield snapshots


def read_builder_state(root: Path) -> dict:
    """Read the exact code identity that produced a candidate bundle."""
    branch_result = git(root, "branch", "--show-current")
    head_result = git(root, "rev-parse", "HEAD")
    status_result = git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    for label, result in (
        ("BRANCH", branch_result),
        ("HEAD", head_result),
        ("STATUS", status_result),
    ):
        if result.returncode != 0:
            raise RuntimeError(
                f"BUILDER_GIT_{label}_FAILED:{result.stderr.strip()[:200]}"
            )

    code_sha256 = {}
    for relative in BUILDER_CODE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"BUILDER_CODE_FILE_MISSING:{relative}")
        code_sha256[relative] = sha256(path)

    return {
        "builder_branch": branch_result.stdout.strip(),
        "builder_head": head_result.stdout.strip(),
        "builder_worktree_clean": not bool(status_result.stdout),
        "builder_code_sha256": code_sha256,
    }


def verify_publish_builder_policy(
    root: Path,
    *,
    expected_branch: str | None,
    expected_head: str | None,
) -> dict:
    """Require an explicitly authorized clean builder identity for publish."""
    if not expected_branch:
        raise RuntimeError("PUBLISH_EXPECTED_BUILDER_BRANCH_REQUIRED")
    if expected_branch not in ALLOWED_PUBLICATION_BRANCHES:
        raise RuntimeError("PUBLISH_BUILDER_BRANCH_NOT_ALLOWED")
    if not expected_head:
        raise RuntimeError("PUBLISH_EXPECTED_BUILDER_HEAD_REQUIRED")
    if len(expected_head) != 40 or any(
        char not in "0123456789abcdef" for char in expected_head
    ):
        raise RuntimeError("PUBLISH_EXPECTED_BUILDER_HEAD_INVALID")

    state = read_builder_state(root)
    if state["builder_branch"] != expected_branch:
        raise RuntimeError("BUILDER_BRANCH_MISMATCH")
    if state["builder_head"] != expected_head:
        raise RuntimeError("BUILDER_HEAD_MISMATCH")
    if state["builder_worktree_clean"] is not True:
        raise RuntimeError("BUILDER_WORKTREE_NOT_CLEAN")
    return state


def verify_builder_state_unchanged(initial: dict, final: dict) -> None:
    """Fail closed if identity or any builder module byte changes mid-build."""
    if initial.get("builder_code_sha256") != final.get("builder_code_sha256"):
        raise RuntimeError("BUILDER_CODE_SHA_CHANGED_DURING_BUILD")
    if initial != final:
        raise RuntimeError("BUILDER_STATE_CHANGED_DURING_BUILD")


def _linux_rename_noreplace(source: Path, target: Path) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("ATOMIC_NOREPLACE_UNAVAILABLE_NON_LINUX")

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("ATOMIC_NOREPLACE_UNAVAILABLE_RENAMEAT2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return

    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    raise OSError(error_number, os.strerror(error_number), target)


def atomic_publish_directory_noreplace(
    staging: Path,
    final_dir: Path,
    *,
    _rename_noreplace=None,
) -> None:
    """Atomically publish a directory without replacing any existing target."""
    if final_dir.exists():
        raise RuntimeError(f"PUBLICATION_TARGET_ALREADY_EXISTS:{final_dir}")
    rename_noreplace = _rename_noreplace or _linux_rename_noreplace
    try:
        rename_noreplace(staging, final_dir)
    except FileExistsError as exc:
        raise RuntimeError(
            f"PUBLICATION_TARGET_ALREADY_EXISTS:{final_dir}"
        ) from exc
    if staging.exists() or not final_dir.is_dir():
        raise RuntimeError("ATOMIC_NOREPLACE_POSTCONDITION_FAILED")


def run_private_staged_validation(validate, *, project_root: Path | None = None):
    """Run full serialization validation outside project publication state."""
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix="std0-publication-v2-validation-"
    ) as temp_name:
        temporary_path = Path(temp_name)
        if project_root is not None:
            try:
                temporary_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                pass
            else:
                raise RuntimeError("TEMP_VALIDATION_BUNDLE_INSIDE_PROJECT")
        result = validate(temporary_path)
        if not any(temporary_path.iterdir()):
            raise RuntimeError("TEMP_VALIDATION_BUNDLE_NOT_WRITTEN")
        print("TEMP_VALIDATION_BUNDLE_WRITTEN=true")
    if temporary_path.exists():
        raise RuntimeError("TEMP_VALIDATION_BUNDLE_CLEANUP_FAILED")
    return result


def verify_publication_content_sha256(report: dict) -> None:
    actual = report.get("publication_content_sha256")
    if actual != PINNED_PUBLICATION_CONTENT_SHA256:
        raise RuntimeError("PUBLICATION_CONTENT_SHA256_MISMATCH")
    print(f"PUBLICATION_CONTENT_SHA256={actual}")


def report_candidate_subset_contract() -> None:
    """Report the separately typed, frozen candidate-subset audit contract."""
    from std0_quant.features.provenance_v2 import (
        CANDIDATE_SUBSET,
        CURRENT_CANDIDATE_SUBSET_EXPECTATIONS,
        FULL_ARTIFACT,
        CandidateSubsetProvenanceExpectations,
        FullArtifactProvenanceExpectations,
    )

    candidate = CURRENT_CANDIDATE_SUBSET_EXPECTATIONS
    if not isinstance(candidate, CandidateSubsetProvenanceExpectations):
        raise RuntimeError("CANDIDATE_SUBSET_EXPECTATION_TYPE_MISMATCH")
    if isinstance(candidate, FullArtifactProvenanceExpectations):
        raise RuntimeError("PROVENANCE_EXPECTATION_SCOPE_COLLISION")
    if CANDIDATE_SUBSET == FULL_ARTIFACT:
        raise RuntimeError("PROVENANCE_CONTRACT_SCOPE_COLLISION")
    print(f"CANDIDATE_PHASE1_ROWS={candidate.phase1_rows}")
    print(
        "CANDIDATE_UNIQUE_CONDITIONS="
        f"{candidate.phase1_unique_conditions}"
    )
    print("FULL_AND_CANDIDATE_EXPECTATIONS_NOT_INTERCHANGEABLE=PASS")


def _normalized_row_multiset_sha256(left: list[dict], right: list[dict]) -> tuple[str, str]:
    """Compare serialized rows while treating schema-added nulls as absent."""
    from decimal import Decimal

    from std0_quant.events.publication_provenance_v2 import row_sha256

    columns = sorted(
        {
            str(key)
            for row in (*left, *right)
            for key in row
        }
    )

    def normalize(value):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int):
            return {"__number__": str(value)}
        if isinstance(value, float) and not (value != value or abs(value) == float("inf")):
            number = Decimal(str(value)).normalize()
            return {"__number__": "0" if number == 0 else format(number, "f")}
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    def digest(rows: list[dict]) -> str:
        hashes = sorted(
            row_sha256(
                {
                    column: normalize(row.get(column))
                    for column in columns
                }
            )
            for row in rows
        )
        return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()

    return digest(left), digest(right)


def validate_staged_bundle(
    *,
    staging: Path,
    pq,
    publication_rows: list[dict],
    reconciliation_rows: list[dict],
    provenance_rows: list[dict],
    report: dict,
    bundle_manifest: dict,
    final_truth_rel: str,
    behavioral_truth_sha: str,
    provenance_expected,
) -> None:
    """Re-read and independently validate every staged byte before rename."""
    from std0_quant.events.publication_provenance_v2 import rows_content_sha256
    from std0_quant.features.provenance_v2 import (
        preflight_phase1_provenance_membership,
        validate_phase1_provenance_membership,
    )

    expected_names = {
        BEHAVIORAL_TRUTH_NAME,
        RECONCILIATION_NAME,
        PROVENANCE_NAME,
        REPORT_NAME,
        BUNDLE_MANIFEST_NAME,
    }
    actual_names = {path.name for path in staging.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(
            "STAGING_FILE_SET_MISMATCH:"
            f"expected={sorted(expected_names)} actual={sorted(actual_names)}"
        )

    behavioral_truth_path = staging / BEHAVIORAL_TRUTH_NAME
    reconciliation_path = staging / RECONCILIATION_NAME
    provenance_path = staging / PROVENANCE_NAME
    report_path = staging / REPORT_NAME
    bundle_manifest_path = staging / BUNDLE_MANIFEST_NAME

    if sha256(behavioral_truth_path) != behavioral_truth_sha:
        raise RuntimeError("STAGED_BEHAVIORAL_TRUTH_SHA_MISMATCH")

    staged_publication = pq.read_table(behavioral_truth_path).to_pylist()
    staged_reconciliation = pq.read_table(reconciliation_path).to_pylist()
    staged_provenance = pq.read_table(provenance_path).to_pylist()

    expected_digest, staged_digest = _normalized_row_multiset_sha256(
        publication_rows,
        staged_publication,
    )
    if staged_digest != expected_digest:
        raise RuntimeError("STAGED_PUBLICATION_SEMANTICS_MISMATCH")
    if (
        rows_content_sha256(publication_rows)
        != report.get("publication_content_sha256")
    ):
        raise RuntimeError("STAGED_PUBLICATION_REPORT_HASH_MISMATCH")

    expected_digest, staged_digest = _normalized_row_multiset_sha256(
        reconciliation_rows,
        staged_reconciliation,
    )
    if staged_digest != expected_digest:
        raise RuntimeError("STAGED_RECONCILIATION_SEMANTICS_MISMATCH")

    expected_digest, staged_digest = _normalized_row_multiset_sha256(
        provenance_rows,
        staged_provenance,
    )
    if staged_digest != expected_digest:
        raise RuntimeError("STAGED_PROVENANCE_SEMANTICS_MISMATCH")

    preflight_phase1_provenance_membership(
        staged_provenance,
        publication_rows=staged_publication,
        expected=provenance_expected,
    )
    publication_membership = {
        str(row["condition_id"])
        for row in staged_publication
    }
    validate_phase1_provenance_membership(
        staged_provenance,
        source_membership={final_truth_rel: publication_membership},
        source_sha256={final_truth_rel: behavioral_truth_sha},
    )

    staged_report = json.loads(report_path.read_text(encoding="utf-8"))
    if staged_report != report:
        raise RuntimeError("STAGED_REPORT_SEMANTICS_MISMATCH")
    staged_bundle_manifest = json.loads(
        bundle_manifest_path.read_text(encoding="utf-8")
    )
    if staged_bundle_manifest != bundle_manifest:
        raise RuntimeError("STAGED_BUNDLE_MANIFEST_SEMANTICS_MISMATCH")

    actual_output_hashes = {
        BEHAVIORAL_TRUTH_NAME: sha256(behavioral_truth_path),
        RECONCILIATION_NAME: sha256(reconciliation_path),
        PROVENANCE_NAME: sha256(provenance_path),
        REPORT_NAME: sha256(report_path),
    }
    if bundle_manifest.get("outputs") != actual_output_hashes:
        raise RuntimeError("STAGED_BUNDLE_OUTPUT_SHA_MISMATCH")

    print("STAGED_BUNDLE_VALIDATION=PASS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-ledger", type=Path, required=True)
    p.add_argument("--prospective-ledger", type=Path, required=True)
    p.add_argument("--prospective-raw", type=Path, required=True)
    p.add_argument("--stage1-log", type=Path, required=True)
    p.add_argument("--stage2-log", type=Path, required=True)
    p.add_argument("--stage2-script", type=Path, required=True)
    p.add_argument("--feature-provenance", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Versioned output directory. In publish mode this must resolve "
            "exactly to data/derived/publication_v2 under the repo root."
        ),
    )
    p.add_argument(
        "--expected-builder-branch",
        help="Required exact branch policy when --publish is requested.",
    )
    p.add_argument(
        "--expected-builder-head",
        help="Required exact committed builder HEAD when --publish is requested.",
    )
    p.add_argument("--publish", action="store_true")
    return p.parse_args()


def _flush_file_descriptor(fd: int) -> None:
    if os.name != "nt":
        os.fsync(fd)
        return

    import msvcrt

    handle = msvcrt.get_osfhandle(fd)
    flush_file_buffers = ctypes.windll.kernel32.FlushFileBuffers
    flush_file_buffers.argtypes = (ctypes.c_void_p,)
    flush_file_buffers.restype = ctypes.c_int
    if flush_file_buffers(ctypes.c_void_p(handle)) == 0:
        raise ctypes.WinError()


def _write_json_temp(temporary: Path, data: bytes) -> None:
    with temporary.open("xb") as fh:
        fh.write(data)
        fh.flush()
        _flush_file_descriptor(fh.fileno())


def atomic_write_json_v2(path: Path, payload: dict) -> Path:
    """Write deterministic JSON through a flushed same-directory temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"STALE_JSON_TEMP_EXISTS:{temporary}")

    data = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        _write_json_temp(temporary, data)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def atomic_write_parquet_v2(rows: list[dict], path: Path) -> Path:
    """Serialize inside staging with a same-directory atomic file replace."""
    if not rows:
        raise ValueError("refusing to serialize empty publication artifact")

    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"STALE_PARQUET_TEMP_EXISTS:{temporary}")

    try:
        pq.write_table(
            pa.Table.from_pylist(rows),
            temporary,
            compression="zstd",
        )
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
        fd = os.open(str(temporary), flags)
        try:
            _flush_file_descriptor(fd)
        finally:
            os.close(fd)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    return target


def _open_directory_fd(path: Path) -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("DIRECTORY_FSYNC_UNAVAILABLE_NON_LINUX")
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        raise RuntimeError("DIRECTORY_FSYNC_UNAVAILABLE_O_DIRECTORY")
    return os.open(str(path), os.O_RDONLY | directory_flag)


def durable_publish_directory_v2(
    staging: Path,
    final_dir: Path,
    *,
    _open_directory=None,
    _fsync_directory=None,
    _close_fd=None,
    _publish_noreplace=None,
) -> None:
    """Durably publish only after all pre-rename directory checks succeed."""
    open_directory = _open_directory or _open_directory_fd
    fsync_directory = _fsync_directory or os.fsync
    close_fd = _close_fd or os.close
    publish_noreplace = (
        _publish_noreplace or atomic_publish_directory_noreplace
    )

    staging_fd = open_directory(staging)
    try:
        fsync_directory(staging_fd)
    finally:
        close_fd(staging_fd)

    # The parent descriptor must be acquired before rename. If this fails,
    # publication has not happened and must not be attempted.
    parent_fd = open_directory(final_dir.parent)
    renamed = False
    try:
        publish_noreplace(staging, final_dir)
        renamed = True
        try:
            fsync_directory(parent_fd)
        except BaseException as exc:
            raise RuntimeError(
                "PUBLICATION_RENAMED_DURABILITY_UNCONFIRMED"
            ) from exc
    finally:
        try:
            close_fd(parent_fd)
        except BaseException as exc:
            if renamed:
                raise RuntimeError(
                    "PUBLICATION_RENAMED_DURABILITY_UNCONFIRMED"
                ) from exc
            raise


def build_validated_staged_bundle(
    *,
    root: Path,
    staging: Path,
    pq,
    atomic_write_parquet,
    publication_rows: list[dict],
    reconciliation_rows: list[dict],
    input_provenance_rows: list[dict],
    report: dict,
    builder_state: dict,
    provenance_expected,
) -> dict:
    """Repair, serialize, and re-validate the complete staged bundle."""
    from std0_quant.features.provenance_v2 import repair_phase1_provenance_v2

    if any(staging.iterdir()):
        raise RuntimeError(f"STAGING_DIRECTORY_NOT_EMPTY:{staging}")

    behavioral_truth_path = staging / BEHAVIORAL_TRUTH_NAME
    reconciliation_path = staging / RECONCILIATION_NAME
    provenance_path = staging / PROVENANCE_NAME
    report_path = staging / REPORT_NAME

    atomic_write_parquet(publication_rows, behavioral_truth_path)
    behavioral_truth_sha = sha256(behavioral_truth_path)
    final_truth_rel = (
        (root / OUTPUT_RELATIVE_DIR / BEHAVIORAL_TRUTH_NAME)
        .relative_to(root)
        .as_posix()
    )

    repaired_provenance, provenance_report = repair_phase1_provenance_v2(
        input_provenance_rows,
        publication_source_file=final_truth_rel,
        publication_source_sha256=behavioral_truth_sha,
        publication_rows=publication_rows,
    )

    atomic_write_parquet(reconciliation_rows, reconciliation_path)
    atomic_write_parquet(repaired_provenance, provenance_path)
    atomic_write_json_v2(report_path, report)

    bundle_manifest = {
        "contract_version": "publication_provenance_v2",
        "pinned_main": PINNED_MAIN,
        "builder_branch": builder_state["builder_branch"],
        "builder_head": builder_state["builder_head"],
        "builder_worktree_clean": builder_state["builder_worktree_clean"],
        "builder_code_sha256": builder_state["builder_code_sha256"],
        "inputs": {
            "canonical_ledger_sha256": PINNED_CANONICAL_SHA256,
            "prospective_ledger_sha256": PINNED_PROSPECTIVE_LEDGER_SHA256,
            "prospective_raw_sha256": PINNED_PROSPECTIVE_RAW_SHA256,
            "feature_provenance_sha256": PINNED_FEATURE_PROVENANCE_SHA256,
            "stage1_log_sha256": PINNED_STAGE1_LOG_SHA256,
            "stage2_log_sha256": PINNED_STAGE2_LOG_SHA256,
            "stage2_script_sha256": PINNED_STAGE2_SCRIPT_SHA256,
        },
        "outputs": {
            BEHAVIORAL_TRUTH_NAME: behavioral_truth_sha,
            RECONCILIATION_NAME: sha256(reconciliation_path),
            PROVENANCE_NAME: sha256(provenance_path),
            REPORT_NAME: sha256(report_path),
        },
        "publication_report": report,
        "provenance_report": provenance_report,
        "formal_cohort_written": False,
        "backtest_executed": False,
    }
    for relative, manifest_key in BUILDER_MANIFEST_SHA_KEYS.items():
        bundle_manifest[manifest_key] = builder_state["builder_code_sha256"][
            relative
        ]
    atomic_write_json_v2(staging / BUNDLE_MANIFEST_NAME, bundle_manifest)

    validate_staged_bundle(
        staging=staging,
        pq=pq,
        publication_rows=publication_rows,
        reconciliation_rows=reconciliation_rows,
        provenance_rows=repaired_provenance,
        report=report,
        bundle_manifest=bundle_manifest,
        final_truth_rel=final_truth_rel,
        behavioral_truth_sha=behavioral_truth_sha,
        provenance_expected=provenance_expected,
    )

    return {
        "provenance_report": provenance_report,
        "bundle_manifest": bundle_manifest,
        "behavioral_truth_sha256": behavioral_truth_sha,
        "reconciliation_sha256": sha256(reconciliation_path),
        "provenance_sha256": sha256(provenance_path),
        "bundle_manifest_sha256": sha256(staging / BUNDLE_MANIFEST_NAME),
    }


def _execute_with_verified_inputs(
    *,
    args: argparse.Namespace,
    original_args: argparse.Namespace,
    root: Path,
    builder_state: dict,
    pq,
    SlugWindowMetadataProvider,
    build_ledger_rows,
    load_fills,
    atomic_write_parquet,
    build_forensic_resolutions_from_logs,
    build_publication_v2,
    current_repair_expectations,
    provenance_expected,
    preflight_phase1_provenance_membership,
) -> int:
    canonical_rows = pq.read_table(args.canonical_ledger).to_pylist()
    prospective_rows = pq.read_table(args.prospective_ledger).to_pylist()

    fills = sorted(
        list(load_fills(args.prospective_raw, keep_raw_json=False)),
        key=lambda fill: (
            fill.timestamp_ms if fill.timestamp_ms is not None else 0,
            fill.condition_id or "",
            fill.fill_id,
        ),
    )
    if not fills:
        raise RuntimeError("PROSPECTIVE_RAW_REPLAY_EMPTY")

    metadata = SlugWindowMetadataProvider.from_fills(
        fills,
        slug_prefix="btc-updown-5m-",
        window_seconds=300,
    )
    truth_rows = build_ledger_rows(
        fills,
        metadata,
        coverage_provider=None,
        scope_slug_prefix="btc-updown-5m-",
    )

    resolutions, forensic_report = build_forensic_resolutions_from_logs(
        args.stage1_log,
        args.stage2_log,
    )
    print("FORENSIC_REPORT=" + json.dumps(forensic_report, sort_keys=True))

    publication_rows, manifest_rows, report = build_publication_v2(
        canonical_rows=canonical_rows,
        prospective_ledger_rows=prospective_rows,
        prospective_truth_rows=truth_rows,
        forensic_resolutions=resolutions,
        expected=current_repair_expectations,
    )
    print("PUBLICATION_REPORT=" + json.dumps(report, sort_keys=True))
    verify_publication_content_sha256(report)

    provenance_rows = pq.read_table(args.feature_provenance).to_pylist()
    provenance_preflight = preflight_phase1_provenance_membership(
        provenance_rows,
        publication_rows=publication_rows,
        expected=provenance_expected,
    )
    print(
        "PROVENANCE_PREFLIGHT="
        + json.dumps(provenance_preflight, sort_keys=True)
    )
    print(
        "PHASE1_NULL_CONDITION_IDS="
        f"{provenance_preflight['phase1_null_condition_ids']}"
    )
    report_candidate_subset_contract()

    def stage(staging: Path) -> dict:
        return build_validated_staged_bundle(
            root=root,
            staging=staging,
            pq=pq,
            atomic_write_parquet=atomic_write_parquet,
            publication_rows=publication_rows,
            reconciliation_rows=manifest_rows,
            input_provenance_rows=provenance_rows,
            report=report,
            builder_state=builder_state,
            provenance_expected=provenance_expected,
        )

    if not args.publish:
        staged_result = run_private_staged_validation(
            stage,
            project_root=root,
        )
        print(
            "PROVENANCE_REPORT="
            + json.dumps(staged_result["provenance_report"], sort_keys=True)
        )
        print("TEMP_VALIDATION_BUNDLE_CLEANED=YES")
        verify_pinned_inputs(original_args, label_prefix="POST_")
        print("PROTECTED_INPUTS_UNCHANGED=YES")
        print("PROJECT_OUTPUT_FILES_WRITTEN=0")
        print("OUTPUT_FILES_WRITTEN=0")
        return 0

    final_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (root / OUTPUT_RELATIVE_DIR).resolve()
    )
    expected_dir = (root / OUTPUT_RELATIVE_DIR).resolve()
    if final_dir != expected_dir:
        raise RuntimeError(f"PUBLISH_OUTPUT_DIR_MUST_EQUAL:{expected_dir}")
    if final_dir.exists():
        raise RuntimeError(f"PUBLICATION_V2_TARGET_ALREADY_EXISTS:{final_dir}")

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = final_dir.parent / f".{final_dir.name}.{os.getpid()}.tmp"
    if staging.exists():
        raise RuntimeError(f"STALE_STAGING_DIR_EXISTS:{staging}")

    try:
        staging.mkdir(parents=False, exist_ok=False)
        staged_result = stage(staging)
        verify_pinned_inputs(original_args, label_prefix="POST_")
        print("PROTECTED_INPUTS_UNCHANGED=YES")

        final_builder_state = verify_publish_builder_policy(
            root,
            expected_branch=args.expected_builder_branch,
            expected_head=args.expected_builder_head,
        )
        verify_builder_state_unchanged(builder_state, final_builder_state)

        durable_publish_directory_v2(staging, final_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(
        "PROVENANCE_REPORT="
        + json.dumps(staged_result["provenance_report"], sort_keys=True)
    )
    print(f"PUBLICATION_DIR={final_dir}")
    print(
        f"BEHAVIORAL_TRUTH_SHA256="
        f"{staged_result['behavioral_truth_sha256']}"
    )
    print(f"RECONCILIATION_SHA256={staged_result['reconciliation_sha256']}")
    print(f"PROVENANCE_SHA256={staged_result['provenance_sha256']}")
    print(
        f"BUNDLE_MANIFEST_SHA256="
        f"{staged_result['bundle_manifest_sha256']}"
    )
    return 0


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    import pyarrow.parquet as pq

    from std0_quant.events.event_ledger import (
        SlugWindowMetadataProvider,
        build_ledger_rows,
    )
    from std0_quant.events.fills import load_fills
    from std0_quant.events.publication_forensics_v2 import (
        build_forensic_resolutions_from_logs,
    )
    from std0_quant.events.publication_provenance_v2 import (
        CURRENT_REPAIR_EXPECTATIONS,
        build_publication_v2,
    )
    from std0_quant.features.provenance_v2 import (
        CURRENT_FULL_ARTIFACT_EXPECTATIONS,
        preflight_phase1_provenance_membership,
    )

    mode = "PUBLISH" if args.publish else "DRY_RUN"
    print(f"MODE={mode}")
    print(f"PINNED_MAIN={PINNED_MAIN}")
    print("FORMAL_COHORT_ACCESS=NONE")
    print("BACKTEST_EXECUTED=false")

    verify_frozen_behavioral_sources(root)
    builder_state = read_builder_state(root)
    if args.publish:
        builder_state = verify_publish_builder_policy(
            root,
            expected_branch=args.expected_builder_branch,
            expected_head=args.expected_builder_head,
        )
    print("BUILDER_STATE=" + json.dumps(builder_state, sort_keys=True))

    # The first source hash is diagnostic. The snapshot hash is authoritative
    # for every byte consumed below.
    verify_pinned_inputs(args)
    with verified_input_snapshots(args) as snapshot_args:
        result = _execute_with_verified_inputs(
            args=snapshot_args,
            original_args=args,
            root=root,
            builder_state=builder_state,
            pq=pq,
            SlugWindowMetadataProvider=SlugWindowMetadataProvider,
            build_ledger_rows=build_ledger_rows,
            load_fills=load_fills,
            atomic_write_parquet=atomic_write_parquet_v2,
            build_forensic_resolutions_from_logs=(
                build_forensic_resolutions_from_logs
            ),
            build_publication_v2=build_publication_v2,
            current_repair_expectations=CURRENT_REPAIR_EXPECTATIONS,
            provenance_expected=CURRENT_FULL_ARTIFACT_EXPECTATIONS,
            preflight_phase1_provenance_membership=(
                preflight_phase1_provenance_membership
            ),
        )
    print("TEMP_INPUT_SNAPSHOTS_CLEANED=YES")
    print("SNAPSHOT_TEMP_CLEANED=YES")
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
