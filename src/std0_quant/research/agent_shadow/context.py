"""Explicit PIT public-state projection; unknown columns fail closed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .audit import sha256_json


FEATURE_ALLOWLIST = frozenset({
    "btc_last_price", "btc_start_price", "btc_distance_bps",
    "btc_ret_1s", "btc_ret_3s", "btc_ret_5s", "btc_ret_10s", "btc_ret_30s",
    "btc_rv_5s", "btc_rv_10s", "btc_rv_30s",
    "btc_trade_count_1s", "btc_trade_count_5s", "btc_trade_count_30s",
    "btc_volume_1s", "btc_volume_5s", "btc_volume_30s",
    "btc_signed_flow_1s", "btc_signed_flow_5s", "btc_signed_flow_30s",
    "btc_pre30_coverage_pct", "book_pre10_coverage_pct",
    "opp_mid", "opp_spread", "opp_obi_1", "opp_bid_depth_1", "opp_ask_depth_1",
    "pm_mid_change_1s", "pm_mid_change_5s", "pm_obi_change_1s",
    "book_update_count_1s", "book_update_count_5s",
})
META_ALLOWLIST = frozenset({
    "condition_id", "decision_cutoff_ts_ms", "btc_observed_ts_ms", "book_observed_ts_ms",
})


@dataclass(frozen=True)
class ShadowContext:
    condition_id: str
    decision_cutoff_ts_ms: int
    btc_observed_ts_ms: int
    book_observed_ts_ms: int
    features: dict[str, int | float | None]
    input_feature_sha256: str
    context_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "decision_cutoff_ts_ms": self.decision_cutoff_ts_ms,
            "btc_observed_ts_ms": self.btc_observed_ts_ms,
            "book_observed_ts_ms": self.book_observed_ts_ms,
            "features": dict(self.features),
            "input_feature_sha256": self.input_feature_sha256,
            "context_sha256": self.context_sha256,
        }


def build_shadow_context(row: Mapping[str, Any]) -> ShadowContext:
    if not isinstance(row, Mapping) or any(not isinstance(k, str) for k in row):
        raise ValueError("PIT row must be a string-keyed mapping")
    unknown = set(row) - FEATURE_ALLOWLIST - META_ALLOWLIST
    if unknown:
        raise ValueError(f"PIT row contains forbidden or unknown fields: {sorted(unknown)}")
    cid = row.get("condition_id")
    if not isinstance(cid, str) or not cid.strip():
        raise ValueError("condition_id missing")
    cutoff = row.get("decision_cutoff_ts_ms")
    if type(cutoff) is not int or cutoff <= 0:
        raise ValueError("decision cutoff must be a positive integer timestamp")
    for name in ("btc_observed_ts_ms", "book_observed_ts_ms"):
        observed = row.get(name)
        if type(observed) is not int or observed <= 0 or observed > cutoff:
            raise ValueError(f"{name} is missing or post-cutoff")
    features: dict[str, int | float | None] = {}
    for name in sorted(FEATURE_ALLOWLIST & row.keys()):
        value = row[name]
        if value is not None:
            if type(value) not in (int, float):
                raise ValueError(f"{name} must be finite numeric or null")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite numeric or null")
        features[name] = value
    if not features:
        raise ValueError("PIT row has no allowlisted public features")
    feature_hash = sha256_json(features)
    content = {
        "condition_id": cid,
        "decision_cutoff_ts_ms": cutoff,
        "btc_observed_ts_ms": row["btc_observed_ts_ms"],
        "book_observed_ts_ms": row["book_observed_ts_ms"],
        "input_feature_sha256": feature_hash,
    }
    return ShadowContext(cid, cutoff, row["btc_observed_ts_ms"], row["book_observed_ts_ms"],
                         features, feature_hash, sha256_json(content))
