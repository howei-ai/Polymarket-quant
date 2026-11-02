"""Explicit SDK client construction and registered K3/Jev wiring.

Credentials come only from the environment and are never logged or echoed.
Routing is explicit: base URLs are pinned (ANTHROPIC_BASE_URL / OPENAI_BASE_URL are
ignored) and each client gets its own transport, so neither environment nor macOS
system proxies apply; pass proxy_url to use a proxy. Anthropic tracing and trace-header
propagation are off. Environment variables that log request bodies, inject headers, or
re-target OpenAI billing are refused. Constructing a client sends no request; network
use still requires network_enabled=True.
"""

from __future__ import annotations

import os
import socket
import sys
from typing import Any, Mapping

from .anthropic_k3_transport import AnthropicK3Transport
from .interfaces import ModelUnavailable
from .jev_client import JevClient
from .k3_client import K3Client
from .model_registry import CostBudget, binding_for
from .openai_jev_transport import OpenAIJevTransport
from .provider_common import ANTHROPIC_BASE_URL, OPENAI_BASE_URL, register_trusted_client


REFUSED_ENVIRONMENT = ("ANTHROPIC_LOG", "OPENAI_LOG", "ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS",
                       "OPENAI_ORG_ID", "OPENAI_PROJECT_ID")


def _api_key(environ: Mapping[str, str] | None, name: str) -> str:
    value = (os.environ if environ is None else environ).get(name)
    if not isinstance(value, str) or not value.strip():
        raise ModelUnavailable(f"missing credentials: {name} is not set")
    return value


def _optional_text(value: str | None, name: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be a nonempty string or None")
    return value


def _keepalive_socket_options() -> list[tuple[int, int, int | bool]]:
    """TCP keepalive as anthropic 2.3.0's default transport sets it (idle 60 s, interval 60 s, 5 probes)."""
    options: list[tuple[int, int, int | bool]] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)]
    interval = getattr(socket, "TCP_KEEPINTVL", None)
    if interval is not None:
        options.append((socket.IPPROTO_TCP, interval, 60))
    elif sys.platform == "darwin":
        options.append((socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPALIVE", 0x10), 60))
    count = getattr(socket, "TCP_KEEPCNT", None)
    if count is not None:
        options.append((socket.IPPROTO_TCP, count, 5))
    idle = getattr(socket, "TCP_KEEPIDLE", None)
    if idle is not None:
        options.append((socket.IPPROTO_TCP, idle, 60))
    return options


def _http_client(sdk_http_client: Any, proxy_url: str | None) -> Any:
    _optional_text(proxy_url, "proxy_url")
    for name in REFUSED_ENVIRONMENT:
        if name in os.environ:
            raise ModelUnavailable(f"refusing to build a provider client while {name} is set")
    import httpx2

    # An explicit transport replaces the SDK's environment/system proxy mounts, so it must also
    # carry the SDKs' default keepalive and connection limits (max 1000, 100 kept alive).
    transport = httpx2.HTTPTransport(
        proxy=proxy_url, trust_env=False, socket_options=_keepalive_socket_options(),
        limits=httpx2.Limits(max_connections=1000, max_keepalive_connections=100),
    )
    return sdk_http_client(trust_env=False, transport=transport)


def build_anthropic_client(*, environ: Mapping[str, str] | None = None, proxy_url: str | None = None) -> Any:
    key = _api_key(environ, "ANTHROPIC_API_KEY")
    import anthropic

    http_client = _http_client(anthropic.DefaultHttpxClient, proxy_url)
    return register_trusted_client(anthropic.Anthropic(
        api_key=key, base_url=ANTHROPIC_BASE_URL, http_client=http_client,
        open_telemetry={"traces": False, "propagation": False},
    ))


def build_openai_client(*, environ: Mapping[str, str] | None = None, proxy_url: str | None = None,
                        organization: str | None = None, project: str | None = None) -> Any:
    key = _api_key(environ, "OPENAI_API_KEY")
    organization = _optional_text(organization, "organization")
    project = _optional_text(project, "project")
    import openai

    http_client = _http_client(openai.DefaultHttpx2Client, proxy_url)
    return register_trusted_client(openai.OpenAI(
        api_key=key, base_url=OPENAI_BASE_URL, organization=organization, project=project,
        http_client=http_client,
    ))


def build_shadow_models(anthropic_client: Any, openai_client: Any, *, budget: CostBudget,
                        network_enabled: bool, allow_test_clients: bool = False) -> tuple[K3Client, JevClient]:
    """K3 on the registered Anthropic model, Jev on the registered OpenAI model, one shared budget.

    allow_test_clients=True accepts offline stand-ins (never unregistered real SDK clients);
    their transport_version carries a "-test-client" suffix in every artifact.
    """
    if type(network_enabled) is not bool:
        raise TypeError("network_enabled must be an explicit boolean")
    k3 = K3Client(AnthropicK3Transport(anthropic_client, budget=budget, allow_test_client=allow_test_clients),
                  network_enabled=network_enabled)
    jev = JevClient(model_id=binding_for("JEV").model_id,
                    transport=OpenAIJevTransport(openai_client, budget=budget, allow_test_client=allow_test_clients),
                    network_enabled=network_enabled)
    return k3, jev
