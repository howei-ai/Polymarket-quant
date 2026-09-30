"""Frozen project-specific Jev typed question contract for PIT shadow research."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Mapping


JEV_QUESTION_SCHEMA_VERSION = "jev-shadow-q-v1"

_RAW_SPECS = {
    "directional_evidence_consistent": {
        "type": "noul",
        "instructions": (
            "Judge whether the analyst, bull, bear, and manager claims are "
            "internally consistent with their cited pre-cutoff public evidence. "
            "This is not a trade recommendation."
        ),
        "criteria": {
            "true": "Directional claims agree with cited PIT BTC and book features without invented evidence.",
            "false": "Claims conflict with one another or their cited PIT evidence.",
        },
    },
    "book_state_anomalous": {
        "type": "noul",
        "instructions": (
            "Judge whether pre-cutoff Polymarket book features show an obvious "
            "anomalous, unstable, or self-contradictory state; do not forecast future book states."
        ),
        "criteria": {
            "true": "PIT mid, spread, OBI, depth, updates, or coverage reveal material inconsistency.",
            "false": "Available PIT book features are internally coherent without obvious instability.",
        },
    },
    "execution_risk": {
        "type": "choice",
        "instructions": (
            "Classify research-only execution-condition risk from current PIT liquidity, "
            "spread, depth, and update activity; do not authorize an order."
        ),
        "criteria": {
            "low": "PIT spread and depth are comparatively stable and liquid.",
            "medium": "PIT spread, depth, or updates show material but bounded fragility.",
            "high": "PIT liquidity is thin, spread is unstable, or updates are unreliable.",
        },
    },
    "regime": {
        "type": "choice",
        "instructions": (
            "Classify only the current PIT cross-feature regime as momentum, "
            "mean reversion, or unclear; never use future outcome data."
        ),
        "criteria": {
            "momentum": "PIT BTC and PM directional signals align with persistence.",
            "mean_reversion": "PIT directional moves oppose or revert across the observed short horizon.",
            "unclear": "PIT signals conflict, are missing, or do not support either regime.",
        },
    },
    "candidate_quality": {
        "type": "score",
        "instructions": (
            "Score only the quality of this shadow research candidate from "
            "validated pre-cutoff evidence; this is not trading or formal cohort eligibility."
        ),
        "criteria": {
            "0": "No coherent, sufficiently supported PIT research candidate.",
            "1": "Coherent, well-supported PIT shadow research candidate with explicit risks.",
        },
    },
}
JEV_QUESTION_SPECS = MappingProxyType({
    name: MappingProxyType({
        "type": spec["type"],
        "instructions": spec["instructions"],
        "criteria": MappingProxyType(dict(spec["criteria"])),
    })
    for name, spec in _RAW_SPECS.items()
})
del _RAW_SPECS


def question_specs_payload() -> dict[str, dict[str, Any]]:
    """A detached JSON-serializable copy of the frozen specs."""
    return {
        name: {
            "type": spec["type"],
            "instructions": spec["instructions"],
            "criteria": dict(spec["criteria"]),
        }
        for name, spec in JEV_QUESTION_SPECS.items()
    }


def build_jev_request(model_id: str, state: Mapping[str, Any], *,
                      specs: Mapping[str, Any] | None = None,
                      schema_version: str = JEV_QUESTION_SCHEMA_VERSION) -> dict[str, Any]:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Jev model_id required for request")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("Jev question schema version required")
    if not isinstance(state, Mapping):
        raise TypeError("Jev state must be a mapping")
    question_specs = deepcopy(specs) if specs is not None else question_specs_payload()
    if not isinstance(question_specs, dict) or set(question_specs) != set(JEV_QUESTION_SPECS):
        raise ValueError("Jev question specs must contain the frozen five question names")
    for name, spec in question_specs.items():
        if not isinstance(spec, dict) or set(spec) != {"type", "instructions", "criteria"}:
            raise ValueError(f"malformed Jev question spec: {name}")
        frozen = JEV_QUESTION_SPECS[name]
        if spec["type"] != frozen["type"]:
            raise ValueError(f"Jev question type splice: {name}")
        if not isinstance(spec["instructions"], str) or not spec["instructions"].strip():
            raise ValueError(f"missing Jev instructions: {name}")
        if not isinstance(spec["criteria"], dict) or set(spec["criteria"]) != set(frozen["criteria"]):
            raise ValueError(f"missing Jev criteria: {name}")
        if any(not isinstance(text, str) or not text.strip() for text in spec["criteria"].values()):
            raise ValueError(f"empty Jev criterion: {name}")
    return {
        "state": deepcopy(dict(state)),
        "questions": {
            "model_id": model_id,
            "schema_version": schema_version,
            "specs": question_specs,
        },
    }
