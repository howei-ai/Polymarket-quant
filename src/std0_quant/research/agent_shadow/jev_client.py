"""Jev adapter for fixed, typed questions only; network disabled by default."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from .interfaces import ModelUnavailable


class JevClient:
    def __init__(self, *, model_id: str,
                 transport: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
                 network_enabled: bool = False) -> None:
        if not isinstance(model_id, str) or re.fullmatch(r"jev-\d+\.\d+\.\d+", model_id) is None:
            raise ValueError("Jev requires an explicit versioned model_id such as jev-1.13.0")
        self.model_id = model_id
        self._transport = transport
        self._enabled = network_enabled

    def judge(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._enabled or self._transport is None:
            raise ModelUnavailable("Jev transport is disabled")
        return self._transport(state, questions)
