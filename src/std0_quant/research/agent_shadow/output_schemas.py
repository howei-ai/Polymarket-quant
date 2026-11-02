"""JSON schemas for provider structured outputs; the typed parsers stay authoritative."""

from __future__ import annotations

from typing import Any

from .schemas import ANALYST_ROLES, DEBATE_ROLES


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _strings(enum: list[str] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"type": "string"}
    if enum is not None:
        item["enum"] = enum
    return {"type": "array", "items": item}


def k3_output_schema(role: str) -> dict[str, Any]:
    if role == "RESEARCH_MANAGER":
        cited = list(ANALYST_ROLES + DEBATE_ROLES)
        return _object({
            "direction": {"type": "string", "enum": ["UP", "DOWN", "ABSTAIN"]},
            "confidence": {"type": "number"},
            "supporting_roles": _strings(cited),
            "contradicting_roles": _strings(cited),
            "key_risks": _strings(),
        })
    if role == "BULL_RESEARCHER":
        directions = ["UP", "UNCLEAR"]
    elif role == "BEAR_RESEARCHER":
        directions = ["DOWN", "UNCLEAR"]
    elif role in ANALYST_ROLES:
        directions = ["UP", "DOWN", "NEUTRAL", "UNCLEAR"]
    else:
        raise ValueError("unknown K3 role")
    return _object({
        "role": {"type": "string", "enum": [role]},
        "direction": {"type": "string", "enum": directions},
        "confidence": {"type": "number"},
        "evidence": _strings(),
        "risks": _strings(),
        "abstain": {"type": "boolean"},
    })


def jev_output_schema() -> dict[str, Any]:
    def boolean_judgment() -> dict[str, Any]:
        return _object({"answer": {"type": "boolean"}, "probability": {"type": "number"}})

    return _object({
        "directional_evidence_consistent": boolean_judgment(),
        "book_state_anomalous": boolean_judgment(),
        "execution_risk": _object({
            "level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
            "confidence": {"type": "number"},
        }),
        "regime": _object({
            "label": {"type": "string", "enum": ["MOMENTUM", "MEAN_REVERSION", "UNCLEAR"]},
            "confidence": {"type": "number"},
        }),
        "candidate_quality": _object({"score": {"type": "number"}}),
    })
