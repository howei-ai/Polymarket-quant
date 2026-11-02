"""Jev adapter for fixed, typed questions only; bound to the registered judge model, network disabled by default."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .interfaces import ModelUnavailable
from .model_registry import require_registered_model
from .openai_jev_transport import OpenAIJevTransport


class JevClient:
    def __init__(self, *, model_id: str,
                 transport: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
                 network_enabled: bool = False) -> None:
        binding = require_registered_model("JEV", model_id)
        if getattr(getattr(transport, "binding", None), "role", "JEV") != "JEV":
            raise ValueError("Jev transport is bound to a different role")
        registered = isinstance(transport, OpenAIJevTransport)
        if network_enabled and not registered:
            raise ValueError("a network-enabled Jev client needs the registered OpenAI transport")
        # Only the registered transport class reports provider and version; other callables are "unspecified".
        self.model_id = binding.model_id
        self.provider = binding.provider if registered else "unspecified"
        self.transport_version = transport.transport_version if registered else "unspecified"
        self._transport = transport
        self._enabled = network_enabled

    def judge(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._enabled or self._transport is None:
            raise ModelUnavailable("Jev transport is disabled")
        return self._transport(state, questions)

    def drain_call_records(self) -> tuple[Any, ...]:
        drain = getattr(self._transport, "drain_call_records", None)
        return tuple(drain()) if callable(drain) else ()
