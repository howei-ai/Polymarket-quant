"""Provider-independent model interfaces; no network implementation here."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class ModelUnavailable(RuntimeError):
    """Model explicitly unavailable, including timeout or disabled network."""


class ReasoningModel(Protocol):
    model_id: str

    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ProbabilisticJudge(Protocol):
    model_id: str

    def judge(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Mapping[str, Any]: ...
