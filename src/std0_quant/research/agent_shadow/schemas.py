"""Strict, immutable model contracts and deterministic shadow audit artifact."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .audit import sha256_json


ANALYST_ROLES = ("MICROSTRUCTURE", "MOMENTUM", "LIQUIDITY", "REGIME")
DEBATE_ROLES = ("BULL_RESEARCHER", "BEAR_RESEARCHER")
K3_ROLES = ANALYST_ROLES + DEBATE_ROLES + ("RESEARCH_MANAGER",)
COMPARISON_ARMS = ("BASELINE", "K3_ONLY", "K3_PLUS_JEV")


def _mapping(raw: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise ValueError(f"{name} must have exactly {sorted(keys)}")
    return raw


def _prob(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return float(value)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{name} must be a list of nonempty strings")
    return tuple(value)


@dataclass(frozen=True)
class GateEvidence:
    coverage_valid: bool | None
    provenance_valid: bool | None
    sanity_valid: bool | None

    def __post_init__(self) -> None:
        for name in ("coverage_valid", "provenance_valid", "sanity_valid"):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, False)
            elif type(value) is not bool:
                raise TypeError(f"{name} must be boolean or unknown")


@dataclass(frozen=True)
class AnalystReport:
    role: str
    direction: str
    confidence: float
    evidence: tuple[str, ...]
    risks: tuple[str, ...]
    abstain: bool

    @classmethod
    def parse(cls, raw: Any, expected_role: str) -> "AnalystReport":
        value = _mapping(raw, {"role", "direction", "confidence", "evidence", "risks", "abstain"}, "K3 report")
        if expected_role not in ANALYST_ROLES + DEBATE_ROLES or value["role"] != expected_role:
            raise ValueError("K3 role mismatch")
        if value["direction"] not in ("UP", "DOWN", "NEUTRAL", "UNCLEAR"):
            raise ValueError("invalid K3 direction")
        if type(value["abstain"]) is not bool:
            raise ValueError("K3 abstain must be boolean")
        if expected_role in DEBATE_ROLES:
            required_direction = "UP" if expected_role == "BULL_RESEARCHER" else "DOWN"
            allowed_direction = "UNCLEAR" if value["abstain"] else required_direction
            if value["direction"] != allowed_direction:
                raise ValueError("debate role, direction, and abstain are inconsistent")
        return cls(expected_role, value["direction"], _prob(value["confidence"], "K3 confidence"),
                   _strings(value["evidence"], "K3 evidence"), _strings(value["risks"], "K3 risks"), value["abstain"])


@dataclass(frozen=True)
class ManagerReport:
    direction: str
    confidence: float
    supporting_roles: tuple[str, ...]
    contradicting_roles: tuple[str, ...]
    key_risks: tuple[str, ...]

    @classmethod
    def parse(cls, raw: Any) -> "ManagerReport":
        value = _mapping(raw, {"direction", "confidence", "supporting_roles", "contradicting_roles", "key_risks"}, "manager report")
        if value["direction"] not in ("UP", "DOWN", "ABSTAIN"):
            raise ValueError("invalid manager direction")
        supporting = _strings(value["supporting_roles"], "supporting_roles")
        contradicting = _strings(value["contradicting_roles"], "contradicting_roles")
        if any(role not in ANALYST_ROLES + DEBATE_ROLES for role in supporting + contradicting):
            raise ValueError("manager referenced unknown role")
        if len(set(supporting + contradicting)) != len(supporting + contradicting):
            raise ValueError("manager role appears more than once")
        return cls(value["direction"], _prob(value["confidence"], "manager confidence"), supporting,
                   contradicting, _strings(value["key_risks"], "key_risks"))


@dataclass(frozen=True)
class JevJudgments:
    directional_evidence_consistent: dict[str, Any]
    book_state_anomalous: dict[str, Any]
    execution_risk: dict[str, Any]
    regime: dict[str, Any]
    candidate_quality: dict[str, Any]

    @classmethod
    def parse(cls, raw: Any) -> "JevJudgments":
        value = _mapping(raw, {"directional_evidence_consistent", "book_state_anomalous",
                               "execution_risk", "regime", "candidate_quality"}, "Jev judgments")
        bool_items = []
        for name in ("directional_evidence_consistent", "book_state_anomalous"):
            item = _mapping(value[name], {"answer", "probability"}, name)
            if type(item["answer"]) is not bool:
                raise ValueError(f"{name} must be an available boolean judgment")
            bool_items.append({"answer": item["answer"], "probability": _prob(item["probability"], name)})
        risk = _mapping(value["execution_risk"], {"level", "confidence"}, "execution_risk")
        if risk["level"] not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError("invalid execution risk")
        regime = _mapping(value["regime"], {"label", "confidence"}, "regime")
        if regime["label"] not in ("MOMENTUM", "MEAN_REVERSION", "UNCLEAR"):
            raise ValueError("invalid regime")
        quality = _mapping(value["candidate_quality"], {"score"}, "candidate_quality")
        return cls(*bool_items,
                   {"level": risk["level"], "confidence": _prob(risk["confidence"], "risk confidence")},
                   {"label": regime["label"], "confidence": _prob(regime["confidence"], "regime confidence")},
                   {"score": _prob(quality["score"], "candidate score")})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowResult:
    schema_version: str
    condition_id: str
    decision_cutoff_ts: int
    input_feature_sha256: str
    context_sha256: str
    gate_evidence: GateEvidence
    k3_model_id: str
    k3_prompt_schema_version: str
    k3_request_hashes: dict[str, str]
    k3_response_hashes: dict[str, str]
    jev_model_id: str
    jev_question_schema_version: str
    jev_request_hash: str
    jev_response_hash: str
    analyst_reports: tuple[AnalystReport, ...]
    bull_report: AnalystReport | None
    bear_report: AnalystReport | None
    manager_report: ManagerReport | None
    jev_judgments: JevJudgments | None
    gate_status: str
    gate_reasons: tuple[str, ...]
    execution_allowed: bool = False
    formal_cohort_effect: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.gate_evidence, GateEvidence):
            raise TypeError("gate_evidence must be the supplied GateEvidence")
        if self.gate_status not in ("SHADOW_ACCEPT", "SHADOW_REJECT", "BLOCKED"):
            raise ValueError("invalid SHADOW gate status")
        if self.gate_status == "SHADOW_ACCEPT" and not all(
            getattr(self.gate_evidence, name) is True
            for name in ("coverage_valid", "provenance_valid", "sanity_valid")
        ):
            raise ValueError("SHADOW_ACCEPT requires all hard evidence true")
        if self.execution_allowed is not False or self.formal_cohort_effect is not False:
            raise ValueError("SHADOW artifact cannot authorize execution or cohort mutation")
        hashes = (self.input_feature_sha256, self.context_sha256, self.jev_request_hash,
                  self.jev_response_hash, *self.k3_request_hashes.values(),
                  *self.k3_response_hashes.values())
        if any(not isinstance(value, str) or len(value) != 64 or
               any(character not in "0123456789abcdef" for character in value) for value in hashes):
            raise ValueError("audit hashes must be lowercase SHA256 hex")

    @property
    def artifact_sha256(self) -> str:
        return sha256_json(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifact_sha256"] = self.artifact_sha256
        return value
