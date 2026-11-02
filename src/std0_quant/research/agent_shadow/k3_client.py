"""K3 adapter bound to the registered reasoner model; caller must explicitly enable a transport."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .anthropic_k3_transport import AnthropicK3Transport
from .interfaces import ModelUnavailable
from .model_registry import binding_for, require_registered_model


class K3Client:
    def __init__(self, transport: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
                 *, model_id: str | None = None, network_enabled: bool = False) -> None:
        binding = require_registered_model("K3", binding_for("K3").model_id if model_id is None else model_id)
        if getattr(getattr(transport, "binding", None), "role", "K3") != "K3":
            raise ValueError("K3 transport is bound to a different role")
        registered = isinstance(transport, AnthropicK3Transport)
        if network_enabled and not registered:
            raise ValueError("a network-enabled K3 client needs the registered Anthropic transport")
        # Only the registered transport class reports provider and version; other callables are "unspecified".
        self.model_id = binding.model_id
        self.provider = binding.provider if registered else "unspecified"
        self.transport_version = transport.transport_version if registered else "unspecified"
        self._transport = transport
        self._enabled = network_enabled

    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._enabled or self._transport is None:
            raise ModelUnavailable("K3 transport is disabled")
        return self._transport(request)

    def drain_call_records(self) -> tuple[Any, ...]:
        drain = getattr(self._transport, "drain_call_records", None)
        return tuple(drain()) if callable(drain) else ()
