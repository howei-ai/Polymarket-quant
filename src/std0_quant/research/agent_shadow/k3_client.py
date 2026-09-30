"""K3 adapter; caller must explicitly enable and supply a transport."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .interfaces import ModelUnavailable


class K3Client:
    def __init__(self, transport: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
                 *, model_id: str = "kimi-k3", network_enabled: bool = False) -> None:
        if not model_id.strip():
            raise ValueError("K3 model_id required")
        self.model_id = model_id
        self._transport = transport
        self._enabled = network_enabled

    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._enabled or self._transport is None:
            raise ModelUnavailable("K3 transport is disabled")
        return self._transport(request)
