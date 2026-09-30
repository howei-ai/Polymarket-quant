"""Offline fixtures for the additive research shadow layer."""

from __future__ import annotations

from copy import deepcopy


def pit_row(**changes):
    row = {
        "condition_id": "condition-1",
        "decision_cutoff_ts_ms": 1_700_000_000_000,
        "btc_observed_ts_ms": 1_699_999_999_900,
        "book_observed_ts_ms": 1_699_999_999_800,
        "btc_ret_5s": 0.001,
        "btc_pre30_coverage_pct": 1.0,
        "book_pre10_coverage_pct": 1.0,
        "opp_mid": 0.51,
        "opp_spread": 0.02,
    }
    row.update(changes)
    return row


def judge_response(**changes):
    response = {
        "directional_evidence_consistent": {"answer": True, "probability": 0.8},
        "book_state_anomalous": {"answer": False, "probability": 0.1},
        "execution_risk": {"level": "LOW", "confidence": 0.9},
        "regime": {"label": "MOMENTUM", "confidence": 0.8},
        "candidate_quality": {"score": 0.8},
    }
    response.update(changes)
    return response


class FakeReasoner:
    model_id = "kimi-k3"

    def __init__(self, responses=None, error=None):
        self.requests = []
        self.responses = responses or {}
        self.error = error

    def analyze(self, request):
        self.requests.append(deepcopy(request))
        if self.error:
            raise self.error
        role = request["role"]
        if role == "RESEARCH_MANAGER":
            return deepcopy(self.responses.get(role, {
                "direction": "UP", "confidence": 0.8,
                "supporting_roles": ["MICROSTRUCTURE", "MOMENTUM"],
                "contradicting_roles": ["BEAR_RESEARCHER"],
                "key_risks": ["spread"],
            }))
        default = {
            "role": role,
            "direction": "DOWN" if role == "BEAR_RESEARCHER" else "UP",
            "confidence": 0.75,
            "evidence": ["public pre-cutoff feature"],
            "risks": ["uncertainty"],
            "abstain": False,
        }
        return deepcopy(self.responses.get(role, default))


class FakeJudge:
    model_id = "fake-jev-1.0.0"

    def __init__(self, response=None, error=None):
        self.requests = []
        self.response = response or judge_response()
        self.error = error

    def judge(self, state, questions):
        self.requests.append((deepcopy(state), deepcopy(questions)))
        if self.error:
            raise self.error
        return deepcopy(self.response)
