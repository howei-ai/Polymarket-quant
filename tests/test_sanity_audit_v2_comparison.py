"""Pure comparison and bounded reader tests; no production dataset required."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path

import pytest

from std0_quant.audit.sanity_audit_v2 import sanity_audit_v2

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sanity_compare_test", ROOT / "scripts/compare_sanity_audit_v2.py")
assert spec is not None and spec.loader is not None
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
legacy = compare.extract_legacy_sanity((ROOT / "src/std0_quant/audit/prospective.py").read_bytes())


def inputs():
    rows = [{"condition_id": str(i), "feature_row_id": "f" + str(i),
             "prediction_ts_ms": 1787832723000 + i, "market_start_ms": 1787832600000,
             "cutoff_mode": "cutoff_1", "model_eligible": True,
             "pm_obi_change_1s": value} for i, value in enumerate([0.2, 1.6, 2.1])]
    provenance = [{"condition_id": r["condition_id"], "prediction_ts_ms": r["prediction_ts_ms"],
                   "feature_name": "pm_obi_change_1s", "source_type": "polymarket_book"}
                  for r in rows]
    return rows, provenance


def run(rows, provenance):
    return compare.compare_rows(rows, provenance, legacy, sanity_audit_v2)


def test_comparison_preserves_v1_results_and_bad_data_remains_bad():
    rows, provenance = inputs()
    before = deepcopy((rows, provenance))
    result = run(rows, provenance)
    assert result["transitions"] == {"PASS -> PASS": 1,
        "DATA_SANITY_WARNING -> PASS": 1, "DATA_SANITY_WARNING -> DATA_SANITY_WARNING": 1}
    assert result["v1_violation_field_counts"] == {"pm_obi_change_1s": 2}
    assert result["v2_violation_field_counts"] == {"pm_obi_change_1s": 1}
    assert (rows, provenance) == before
    assert result["eligibility_recertified"] is False and result["canonical_cohort_written"] is False
    assert result["backtest_executed"] is False and result["production_callers_switched"] is False
    json.dumps(result, allow_nan=False)


def test_unselected_rows_remain_outside_sanity_comparison():
    rows, provenance = inputs()
    rows[0]["model_eligible"] = False
    rows[0]["opp_bid_depth_1"] = float("nan")  # excluded rows are not promoted or re-audited
    rows[1]["cutoff_mode"] = "cutoff_2"
    result = run(rows, provenance)
    assert result["candidate_rows"] == 1 and result["unselected_rows"] == 2
    assert not rows[0]["model_eligible"]


def test_profile_binds_membership_not_only_total_counts():
    rows, provenance = inputs()
    before = run(rows, provenance)
    rows[0]["feature_row_id"] = "changed"
    after = run(rows, provenance)
    assert before["v1_status_counts"] == after["v1_status_counts"]
    assert before["legacy_profile_sha256"] != after["legacy_profile_sha256"]


@pytest.mark.parametrize("failure", ["duplicate", "flag", "provenance-time", "provenance-key", "orphan", "missing"])
def test_bad_binding_stops_instead_of_emitting_pass(failure):
    rows, provenance = inputs()
    if failure == "duplicate": rows.append(deepcopy(rows[0]))
    elif failure == "flag": rows[0]["model_eligible"] = "true"
    elif failure == "provenance-time": provenance[0]["prediction_ts_ms"] += 1
    elif failure == "provenance-key": provenance.append(deepcopy(provenance[0]))
    elif failure == "orphan": provenance[0]["condition_id"] = "unknown"
    else: provenance.pop(0)
    with pytest.raises(ValueError): run(rows, provenance)


def test_comparison_detects_checker_input_mutation():
    rows, provenance = inputs()
    def mutator(values):
        values[0]["model_eligible"] = False
        return sanity_audit_v2(values)
    with pytest.raises(ValueError, match="mutated"):
        compare.compare_rows(rows, provenance, legacy, mutator)


def test_only_legacy_function_not_module_body_is_executed():
    source = (ROOT / "src/std0_quant/audit/prospective.py").read_bytes()
    extract = compare.extract_legacy_sanity(b'raise RuntimeError("must not run")\n' + source.replace(
        b"from __future__ import annotations", b"# future removed for AST test"))
    assert extract([{"opp_obi_1": 2}])["status"] == "DATA_SANITY_WARNING"


def test_stable_reader_success_preserves_content(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    raw, identity = compare.read_stable(path)
    assert raw == b"{}" and identity == compare.stamp(path.stat())
    assert path.read_bytes() == b"{}"


@pytest.mark.parametrize("mode", ["symlink", "parent-symlink", "hardlink", "oversize", "fifo"])
def test_reader_rejects_unsafe_file_states(tmp_path, mode):
    path = tmp_path / "file"
    if mode == "fifo": os.mkfifo(path)
    elif mode == "parent-symlink":
        real = tmp_path / "real"; real.mkdir(); (real / "file").write_bytes(b"{}")
        (tmp_path / "alias").symlink_to(real, target_is_directory=True)
        path = tmp_path / "alias/file"
    elif mode == "symlink":
        real = tmp_path / "real"; real.write_bytes(b"{}"); path.symlink_to(real)
    else:
        path.write_bytes(b"abc")
        if mode == "hardlink": os.link(path, tmp_path / "link")
    with pytest.raises((ValueError, OSError)):
        compare.read_stable(path, 1 if mode == "oversize" else 100)


def test_reader_detects_changed_stamp(tmp_path, monkeypatch):
    path = tmp_path / "input"; path.write_bytes(b"ok")
    real_read = compare.os.read
    calls = 0
    def changed(fd, count):
        nonlocal calls
        out = real_read(fd, count)
        calls += 1
        if calls == 1:
            with path.open("ab") as f: f.write(b"!")
        return out
    monkeypatch.setattr(compare.os, "read", changed)
    with pytest.raises(ValueError, match="changed"):
        compare.read_stable(path)


def test_wrong_input_pin_stops_before_parquet_or_module_execution(tmp_path, monkeypatch):
    first = next(iter(compare.PINNED_INPUTS.values()))[0]
    path = tmp_path / first; path.parent.mkdir(parents=True); path.write_bytes(b"wrong")
    def forbidden(*args, **kwargs): raise AssertionError("must stop before Parquet")
    monkeypatch.setattr(compare, "load_parquet", forbidden)
    with pytest.raises(ValueError, match="pinned input changed"):
        compare.build_pinned_comparison(tmp_path)


def test_real_parquet_reader_roundtrip_when_dependency_available():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows, _ = inputs()
    buffer = io.BytesIO(); pq.write_table(pa.Table.from_pylist(rows), buffer)
    assert compare.load_parquet(buffer.getvalue(), {"condition_id"}) == rows


def test_real_parquet_reader_rejects_missing_column_when_dependency_available():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    buffer = io.BytesIO(); pq.write_table(pa.table({"a": [1]}), buffer)
    with pytest.raises(ValueError, match="columns"):
        compare.load_parquet(buffer.getvalue(), {"condition_id"})


def test_real_parquet_reader_rejects_bad_bytes_when_dependency_available():
    pa = pytest.importorskip("pyarrow")
    with pytest.raises(pa.ArrowInvalid):
        compare.load_parquet(b"not-parquet", {"condition_id"})
