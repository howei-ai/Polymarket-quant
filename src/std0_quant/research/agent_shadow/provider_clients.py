"""Explicit SDK client construction and registered K3/Jev wiring.

Credentials come only from the environment and are never logged or echoed.
Routing is explicit: base URLs and key variables are pinned per registered route ("official",
or the "relay_mtu" relay whose artifacts are UNVERIFIED_RELAY; ANTHROPIC_BASE_URL / OPENAI_BASE_URL
are ignored) and each client gets its own transport, so neither environment nor macOS
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

from .interfaces import ModelUnavailable
from .jev_client import JevClient
from .k3_client import K3Client
from .model_registry import OFFICIAL_KEY_ENV, CostBudget, binding_for, route_for

# Official credentials a relay client must never see, including ones the SDKs pick up by themselves.
OFFICIAL_CREDENTIAL_ENV = OFFICIAL_KEY_ENV + ("OPENAI_ADMIN_KEY", "OPENAI_WEBHOOK_SECRET", "ANTHROPIC_AUTH_TOKEN")
from .openai_jev_transport import OpenAIJevTransport
from .openai_k3_transport import OpenAIK3Transport
from .provider_common import register_trusted_client


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
    """TCP keepalive exactly as anthropic 2.3.0's default transport builds it, applied to both SDKs
    (openai 3.42.1 sets no keepalive by default): SO_KEEPALIVE, a 60 s probe interval, 5 probes and,
    only where TCP_KEEPIDLE exists (not on macOS), a 60 s idle time. On macOS with TCP_KEEPINTVL
    (this host) no idle time is set, so the OS default applies; only when TCP_KEEPINTVL is missing
    on macOS does the 60 s go to TCP_KEEPALIVE, which there is the idle time."""
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


def _route_key(environ: Mapping[str, str] | None, route: str, provider: str) -> tuple[str, str]:
    """(api key, base URL) for a route. A relay route refuses to build while an official key is
    present, so a mixed-up environment cannot send official credentials or traffic to a relay."""
    selected = route_for(route)
    if selected.provenance != "OFFICIAL":
        # Both the given mapping and the process environment: the SDKs read official credentials
        # (OPENAI_ADMIN_KEY, webhook secrets) from os.environ on their own.
        sources = (os.environ,) if environ is None else (environ, os.environ)
        present = sorted({name for env in sources for name in OFFICIAL_CREDENTIAL_ENV if name in env})
        if present:
            raise ModelUnavailable(f"refusing to build a {route} client while {present} is set")
    return _api_key(environ, selected.key_env[provider]), selected.base_urls[provider]


def build_anthropic_client(*, environ: Mapping[str, str] | None = None, proxy_url: str | None = None,
                           route: str = "official") -> Any:
    key, base_url = _route_key(environ, route, "anthropic")
    import anthropic

    http_client = _http_client(anthropic.DefaultHttpxClient, proxy_url)
    return register_trusted_client(anthropic.Anthropic(
        api_key=key, base_url=base_url, http_client=http_client,
        open_telemetry={"traces": False, "propagation": False},
    ), route)


def build_openai_client(*, environ: Mapping[str, str] | None = None, proxy_url: str | None = None,
                        organization: str | None = None, project: str | None = None,
                        route: str = "official") -> Any:
    key, base_url = _route_key(environ, route, "openai")
    organization = _optional_text(organization, "organization")
    project = _optional_text(project, "project")
    if route_for(route).provenance != "OFFICIAL" and (organization is not None or project is not None):
        raise ValueError("organization and project apply only to the official OpenAI route")
    import openai

    http_client = _http_client(openai.DefaultHttpx2Client, proxy_url)
    return register_trusted_client(openai.OpenAI(
        api_key=key, base_url=base_url, organization=organization, project=project,
        http_client=http_client,
    ), route)


def build_shadow_models(k3_client: Any, jev_client: Any, *, budget: CostBudget,
                        network_enabled: bool, allow_test_clients: bool = False,
                        route: str = "official") -> tuple[K3Client, JevClient]:
    """K3 on GPT-6 Astra, Jev on GPT-6.1 Sol (both OpenAI Responses), one shared budget.

    allow_test_clients=True accepts offline stand-ins (never unregistered real SDK clients);
    their transport_version carries a "-test-client" suffix in every artifact.
    """
    if type(network_enabled) is not bool:
        raise TypeError("network_enabled must be an explicit boolean")
    k3 = K3Client(OpenAIK3Transport(k3_client, budget=budget, allow_test_client=allow_test_clients,
                                     route=route),
                  network_enabled=network_enabled)
    jev = JevClient(model_id=binding_for("JEV").model_id,
                    transport=OpenAIJevTransport(jev_client, budget=budget, allow_test_client=allow_test_clients,
                                               route=route),
                    network_enabled=network_enabled)
    return k3, jev
