"""Explicit fixed-input v1/v2 numeric comparison; stdout only, no cohort write.

This is NOT a checkpoint runner or backtest. Run only after reviewing the
pinned data and source hashes. It reads small existing Parquet/JSON artifacts,
never their raw-market-data references. All production callers remain on v1.
The surrounding installer saves stdout in a NEW /tmp run directory, not reports.
"""
from __future__ import annotations

import __future__
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Any

SUFFIX = "prospective-v4-lowmem-20260829T073235Z.parquet"
PINNED_INPUTS = {
    "features": ("data/derived/features/pretrade_features_" + SUFFIX,
                 "d7562ebd0c06df2b7441f65264b09d6971dbc5f87d965fbd08d8ced8a1a8aa3c"),
    "provenance": ("data/derived/features/feature_provenance_" + SUFFIX,
                   "347ff2d7ea282bb9454a0379f243080c0ed83a39620de46d612449acb29a0bec"),
    "manifest": ("data/state/prospective_cohort.json",
                 "59ed0b09de6149ea1492bbe663bf45938bf092566a4992af8b79ca6b0be683f3"),
    "historical_report": ("data/reports/prospective_checkpoint_1_20260829T083521Z.json",
                          "ecb71b75d1394a754c1b5adfc206c60e2313b1199021ea1cfad6fec911d1a3d6"),
}
LEGACY_BLOB = "0ff11a7fbbf11e833f67468a8374c5484d2cc6a5"
PRIOR_PROFILE = "ac45038f769e14f3f79d118355e01280a452e138977b15b44ad5369fed535279"
MAX_BYTES = 32 * 1024 * 1024


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def input_snapshot(value: Any) -> bytes:
    # Private mutation fingerprint only; excluded records may contain NaN.
    # Reports still use strict canonical() JSON and do not export excluded values.
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=True).encode("utf-8")


def stamp(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def read_stable(path: Path, limit: int = MAX_BYTES) -> tuple[bytes, tuple[int, ...]]:
    require(path.is_absolute() and ".." not in path.parts, "absolute non-traversing path required")
    for parent in reversed(path.parents):
        require(stat.S_ISDIR(parent.lstat().st_mode), "unsafe parent: " + str(parent))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                "expected single-link regular file: " + str(path))
        require(before.st_size <= limit, "byte budget exceeded: " + str(path))
        chunks, size = [], 0
        while size <= limit:
            chunk = os.read(fd, min(65536, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        raw = b"".join(chunks)
        require(len(raw) == before.st_size and len(raw) <= limit
                and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()),
                "input changed while reading: " + str(path))
        return raw, stamp(before)
    finally:
        os.close(fd)


def extract_legacy_sanity(raw: bytes):
    # No module-level import, state writer, CohortManifest or project runner runs.
    nodes = [node for node in ast.parse(raw).body
             if isinstance(node, ast.FunctionDef) and node.name == "sanity_audit"]
    require(len(nodes) == 1 and not nodes[0].decorator_list, "unexpected frozen function shape")
    namespace: dict[str, Any] = {}
    code = compile(ast.Module(body=nodes, type_ignores=[]), "<frozen-sanity-only>", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)
    return namespace["sanity_audit"]


def load_parquet(raw: bytes, required: set[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(io.BytesIO(raw))
    try:
        names = parquet.schema_arrow.names
        require(len(names) == len(set(names)) and required.issubset(names),
                "missing or duplicate Parquet columns")
        decoded = sum(parquet.metadata.row_group(i).total_byte_size
                      for i in range(parquet.metadata.num_row_groups))
        require(parquet.metadata.num_rows <= 200000 and decoded <= 128 * 1024 * 1024,
                "Parquet row/decompressed-size budget exceeded")
        rows = parquet.read(use_threads=False).to_pylist()
        require(len(rows) == parquet.metadata.num_rows, "Parquet row count mismatch")
        return rows
    finally:
        parquet.close()


def compare_rows(features: list[dict[str, Any]], provenance: list[dict[str, Any]],
                 legacy, v2) -> dict[str, Any]:
    """Compare identical records. No filtering by v2 results or eligibility write."""
    before = input_snapshot([features, provenance])
    conditions = [r["condition_id"] for r in features]
    require(all(type(x) is str and x for x in conditions), "invalid condition identity")
    require(len(set(conditions)) == len(conditions), "duplicate feature condition")
    require(all(type(r["model_eligible"]) is bool for r in features), "nonboolean stored flag")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in provenance:
        grouped[row["condition_id"]].append(row)
    require(set(grouped).issubset(conditions), "orphan provenance condition")
    candidates = [r for r in features if r["cutoff_mode"] == "cutoff_1"
                  and r["model_eligible"] is True]
    details, profile = [], []
    transitions: Counter[str] = Counter()
    old_fields: Counter[str] = Counter()
    new_fields: Counter[str] = Counter()
    dates: Counter[str] = Counter()
    for row in candidates:
        cid, prediction = row["condition_id"], row["prediction_ts_ms"]
        require(type(row["feature_row_id"]) is str and bool(row["feature_row_id"]), "missing feature id")
        require(type(prediction) is int and type(row["market_start_ms"]) is int,
                "nonintegral identity timestamp")
        bound = grouped[cid]
        require(bound and all(p["prediction_ts_ms"] == prediction for p in bound),
                "missing/mismatched prediction provenance binding")
        require(len({(p["feature_name"], p["source_type"]) for p in bound}) == len(bound),
                "duplicate provenance feature/source binding")
        old, new = legacy([row]), v2([row])
        require(old["violation_count"] == len(old["violations"])
                and new["violation_count"] == len(new["violations"]), "truncated per-row comparison")
        profile.append((cid, row["feature_row_id"], prediction, old["status"], old["violation_count"]))
        transitions[old["status"] + " -> " + new["status"]] += 1
        old_fields.update(v["field"] for v in old["violations"])
        new_fields.update(v["field"] for v in new["violations"])
        dates[datetime.fromtimestamp(row["market_start_ms"] / 1000, timezone.utc).date().isoformat()] += 1
        details.append({"condition_id": cid, "feature_row_id": row["feature_row_id"],
                        "prediction_ts_ms": prediction, "stored_model_eligible": True,
                        "bound_provenance_rows": len(bound), "v1": old, "v2": new})
    require(input_snapshot([features, provenance]) == before, "comparison mutated input")
    # Exactly the previously recorded profile format; do not hash counts alone.
    profile_hash = hashlib.sha256(json.dumps(sorted(profile), separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    return {"schema_version": "sanity_audit_comparison_v2",
            "scope": "SAME_STORED_CUTOFF1_CANDIDATES_SANITY_ONLY_NOT_REQUALIFICATION",
            "feature_rows": len(features), "candidate_rows": len(candidates),
            "unselected_rows": len(features) - len(candidates),
            "legacy_profile_sha256": profile_hash,
            "transitions": dict(sorted(transitions.items())),
            "v1_status_counts": dict(sorted(Counter(x["v1"]["status"] for x in details).items())),
            "v2_status_counts": dict(sorted(Counter(x["v2"]["status"] for x in details).items())),
            "v1_violation_field_counts": dict(sorted(old_fields.items())),
            "v2_violation_field_counts": dict(sorted(new_fields.items())),
            "candidate_utc_date_counts": dict(sorted(dates.items())), "rows": details,
            "input_values_and_flags_unchanged": True,
            "canonical_cohort_written": False, "raw_quality_revalidated": False,
            "eligibility_recertified": False, "backtest_executed": False,
            "production_callers_switched": False}


def build_pinned_comparison(root: Path) -> dict[str, Any]:
    snapshots, data, inputs = {}, {}, {}
    for label, (relative, digest) in PINNED_INPUTS.items():
        path = root / relative
        raw, identity = read_stable(path)
        require(hashlib.sha256(raw).hexdigest() == digest, "pinned input changed: " + label)
        snapshots[path] = (identity, digest)
        data[label] = raw
        inputs[label] = {"path": str(path), "sha256": digest, "bytes": len(raw)}
    legacy_path = root / "src/std0_quant/audit/prospective.py"
    legacy_raw, identity = read_stable(legacy_path, 1024 * 1024)
    blob = hashlib.sha1(b"blob " + str(len(legacy_raw)).encode() + b"\0" + legacy_raw).hexdigest()
    require(blob == LEGACY_BLOB, "legacy source differs from reviewed blob")
    snapshots[legacy_path] = (identity, hashlib.sha256(legacy_raw).hexdigest())
    legacy = extract_legacy_sanity(legacy_raw)
    new_path = root / "src/std0_quant/audit/sanity_audit_v2.py"
    new_raw, identity = read_stable(new_path, 1024 * 1024)
    snapshots[new_path] = (identity, hashlib.sha256(new_raw).hexdigest())
    # Execute only the explicitly selected new stdlib-only module from these bytes.
    namespace: dict[str, Any] = {"__name__": "sanity_audit_v2_comparison"}
    exec(compile(new_raw, str(new_path), "exec"), namespace)
    features = load_parquet(data["features"], {"condition_id", "feature_row_id", "prediction_ts_ms",
                           "market_start_ms", "model_eligible", "cutoff_mode"})
    provenance = load_parquet(data["provenance"], {"condition_id", "prediction_ts_ms",
                             "feature_name", "source_type"})
    require(len(features) == 463 and len(provenance) == 36577, "pinned artifact shape changed")
    report = compare_rows(features, provenance, legacy, namespace["sanity_audit_v2"])
    require(report["legacy_profile_sha256"] == PRIOR_PROFILE, "prior membership/results changed")
    require(report["candidate_rows"] == 86, "pinned candidate count changed")
    for path, (before, digest) in snapshots.items():
        raw, after = read_stable(path)
        require(before == after and hashlib.sha256(raw).hexdigest() == digest,
                "input/source changed during comparison: " + str(path))
    report.update({"inputs": inputs, "legacy_source_git_blob": LEGACY_BLOB,
                   "v2_source_sha256": snapshots[new_path][1],
                   "prior_profile_matched": True, "protected_input_snapshots_unchanged": True})
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        report = build_pinned_comparison(root)
        print(canonical(report).decode(), flush=True)
    except Exception as exc:
        print(canonical({"comparison_completed": False, "error_type": type(exc).__name__,
                         "error": str(exc), "canonical_cohort_written": False}).decode(), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
