"""Shared fail-closed helpers for provider transports; no SDK is imported at module load."""

from __future__ import annotations

import contextlib
import contextvars
import json
import sys
import threading
import types
import weakref
from types import SimpleNamespace
from typing import Any, Iterator

from .audit import canonical_json
from .interfaces import ModelUnavailable
from .model_registry import ROUTES, CostBudget, ModelBinding, Reservation, route_for
from .schemas import ModelCallRecord


ANTHROPIC_BASE_URL = ROUTES["official"].base_urls["anthropic"]
OPENAI_BASE_URL = ROUTES["official"].base_urls["openai"]
# Bytes added to every request's size estimate for message framing and SDK overhead.
REQUEST_OVERHEAD_BYTES = 2_048

# id(client) -> (weak reference, route name); looked up by identity, never by the client's __hash__/__eq__.
_TRUSTED_CLIENTS: dict[int, tuple["weakref.ref[Any]", str]] = {}


def _trusted_route(client: Any) -> str | None:
    """Route a factory-built client was registered for, by identity; None for anything else."""
    entry = _TRUSTED_CLIENTS.get(id(client))
    return entry[1] if entry is not None and entry[0]() is client else None


def _is_trusted(client: Any) -> bool:
    return _trusted_route(client) is not None


def register_trusted_client(client: Any, route: str = "official") -> Any:
    """Called only by provider_clients factories (route-pinned base URL, trust_env off)."""
    route_for(route)
    key = id(client)

    def forget(ref: Any, key: int = key) -> None:
        if _TRUSTED_CLIENTS.get(key, (None,))[0] is ref:  # ids can be reused; drop only this entry
            del _TRUSTED_CLIENTS[key]

    _TRUSTED_CLIENTS[key] = (weakref.ref(client, forget), route)
    return client


_SDK_CLIENT_CLASSES = (("anthropic", ("Anthropic", "AsyncAnthropic")), ("openai", ("OpenAI", "AsyncOpenAI")))


def _is_sdk_client(client: Any) -> bool:
    """SDK objects, including user subclasses of the SDK client classes."""
    if type(client).__module__.split(".", 1)[0] in ("anthropic", "openai"):
        return True
    for module_name, class_names in _SDK_CLIENT_CLASSES:
        module = sys.modules.get(module_name)
        for name in class_names:
            sdk_class = getattr(module, name, None)
            if isinstance(sdk_class, type) and isinstance(client, sdk_class):
                return True
    return False


# Runtime guard for offline test stand-ins. Once a test-client transport exists, a process-wide
# audit hook refuses every socket, DNS, subprocess and thread-start event raised on a thread that
# is inside a test-client provider call, wherever the stand-in keeps (or builds) a real client.
_GUARDED_EVENTS = frozenset({
    "socket.__new__", "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg",
    "socket.getaddrinfo", "socket.getnameinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
    "socket.gethostbyaddr", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
    "os.fork", "os.forkpty", "_thread.start_new_thread", "_thread.start_joinable_thread",
    "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function",
})
_OFFLINE = threading.local()
_HOOK_LOCK = threading.Lock()
_hook_installed = False


class OfflineNetworkBlocked(RuntimeError):
    """Raised inside a test-client provider call that tried to reach a network or a process."""


def _offline_audit_hook(event: str, _args: Any) -> None:
    if event in _GUARDED_EVENTS:
        state = getattr(_OFFLINE, "state", None)
        if state is not None:
            state.blocked = True
            raise OfflineNetworkBlocked(f"{event} is blocked during an offline test-client call")


def install_offline_guard() -> None:
    """Idempotent; audit hooks cannot be removed, and the hook is inert outside guarded calls."""
    global _hook_installed
    with _HOOK_LOCK:
        if not _hook_installed:
            sys.addaudithook(_offline_audit_hook)
            _hook_installed = True


@contextlib.contextmanager
def offline_call(enabled: bool) -> Iterator[SimpleNamespace]:
    """Yields a state whose .blocked is True if the guarded call attempted any guarded event,
    even when the stand-in swallowed the exception."""
    state = SimpleNamespace(blocked=False)
    if not enabled:
        yield state
        return
    install_offline_guard()
    previous = getattr(_OFFLINE, "state", None)
    _OFFLINE.state = state
    try:
        yield state
    finally:
        _OFFLINE.state = previous


def _network_class_names() -> list[type]:
    names = (("httpx", ("Client", "AsyncClient", "BaseTransport", "AsyncBaseTransport")),
             ("httpx2", ("Client", "AsyncClient", "BaseTransport", "AsyncBaseTransport")),
             ("socket", ("socket",)))
    classes = []
    for module_name, class_names in names:
        module = sys.modules.get(module_name)
        classes.extend(cls for name in class_names if isinstance(cls := getattr(module, name, None), type))
    return classes


def _wraps_network_object(root: Any) -> bool:
    """Shallow early refusal of a stand-in that plainly holds an SDK client, an HTTP client or
    transport, or a socket (its attributes, containers and bound methods, two levels deep).
    Best effort only, for a clear error at construction: the runtime guard above is what keeps
    a stand-in from reaching the network, however it holds or builds a client."""
    network_classes = tuple(_network_class_names())
    stack: list[tuple[Any, int]] = [(root, 0)]
    seen: set[int] = set()
    while stack:
        value, depth = stack.pop()
        if id(value) in seen or isinstance(value, (str, bytes, int, float, bool, type(None), type, BaseException)):
            continue  # fakes legitimately hold SDK errors to raise
        seen.add(id(value))
        module = getattr(type(value), "__module__", None) or ""
        if module.startswith(("anthropic.types", "openai.types")):
            continue  # canned SDK response data
        if value is not root and (_is_trusted(value) or _is_sdk_client(value)
                                  or (network_classes and isinstance(value, network_classes))):
            return True
        if depth >= 2:
            continue
        if isinstance(value, dict):
            children = list(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = list(value)
        elif isinstance(value, types.MethodType):
            children = [value.__self__]
        else:
            attributes = getattr(value, "__dict__", None)
            children = list(attributes.values()) if isinstance(attributes, dict) else []
        stack.extend((child, depth + 1) for child in children)
    return False


def _stand_in_acceptable(client: Any) -> bool:
    """Every check that can run stand-in code (isinstance via __class__, attribute access) runs
    inside the offline guard; any guarded attempt or error there refuses the stand-in."""
    if _is_trusted(client):
        return False
    with offline_call(True) as state:
        try:
            acceptable = not _is_sdk_client(client) and not _wraps_network_object(client)
        except Exception:
            acceptable = False
    return acceptable and not state.blocked


def require_trusted_client(client: Any, *, expected_base_url: str, allow_test_client: bool,
                           route: str = "official") -> bool:
    """Returns True when an offline test stand-in is accepted (recorded in the transport version).

    Real SDK clients are accepted only when built by provider_clients, even in tests. A stand-in
    must not be, or plainly hold, an SDK client (factory-built ones included), an HTTP client or
    a socket; transports then run every stand-in call under offline_call(). See the threat model
    in the README: this guards against accidental misuse, not deliberate in-process tampering.
    """
    if type(allow_test_client) is not bool:
        raise TypeError("allow_test_client must be an explicit boolean")
    if _trusted_route(client) == route and str(client.base_url).rstrip("/") == expected_base_url:
        return False
    if allow_test_client:
        if not _stand_in_acceptable(client):
            raise ValueError("test client must not be or hold a real SDK or HTTP client")
        return True
    raise ValueError("provider client must come from provider_clients with the pinned base URL")


def client_matches(client: Any, *, expected_base_url: str, test_client: bool, route: str = "official") -> bool:
    """Re-check, after construction, that a transport's client still has the trust level its
    version and records claim: a factory-built client on the pinned base URL for real runs, or
    an acceptable stand-in (checked inside the offline guard) for test runs."""
    if test_client is True:
        return _stand_in_acceptable(client)
    return (test_client is False and _trusted_route(client) == route
            and str(client.base_url).rstrip("/") == expected_base_url)


class RoutePolicy:
    """Route rules for the provider calls made in one scope (for ShadowRunner: one row)."""

    __slots__ = ("route", "allow_unverified_relay", "conflict", "parent")

    def __init__(self, route: str | None, allow_unverified_relay: bool, parent: "RoutePolicy | None") -> None:
        self.route = route  # None: pinned by the first call in the scope
        self.allow_unverified_relay = allow_unverified_relay
        self.conflict = False  # the scope asked for a different route than an enclosing scope pins
        self.parent = parent

    def pin(self, route: str) -> None:
        """Pins this scope and every unpinned enclosing scope to route."""
        policy: RoutePolicy | None = self
        while policy is not None:
            if policy.route is None:
                policy.route = route
            policy = policy.parent


_ROUTE_POLICY: contextvars.ContextVar[RoutePolicy | None] = contextvars.ContextVar(
    "agent_shadow_route_policy", default=None)


@contextlib.contextmanager
def route_policy(route: str | None = None, *, allow_unverified_relay: bool = False) -> Iterator[RoutePolicy]:
    """Scope in which provider calls may run. Transports check it on every call on the calling
    thread (context variables do not reach threads started inside the scope), so however a
    transport is wrapped: a relay call needs allow_unverified_relay=True, and every call in the
    scope must use one route (the given one, or the first one used). A nested scope can only
    narrow its enclosing scope: it needs the enclosing opt-in too, and shares its route."""
    if route is not None:
        route = route_for(route).name
    if type(allow_unverified_relay) is not bool:
        raise TypeError("allow_unverified_relay must be an explicit boolean")
    parent = _ROUTE_POLICY.get()
    if parent is not None:
        allow_unverified_relay = allow_unverified_relay and parent.allow_unverified_relay
    policy = RoutePolicy(route, allow_unverified_relay, parent)
    if parent is not None and parent.route is not None:
        policy.conflict = route is not None and route != parent.route
        policy.route = parent.route
    elif route is not None:
        policy.pin(route)
    token = _ROUTE_POLICY.set(policy)
    try:
        yield policy
    finally:
        _ROUTE_POLICY.reset(token)


def route_refusal(route: str) -> str | None:
    """Call outcome the current route policy imposes on a call through `route`, or None.
    Outside any scope, official calls are allowed and relay calls are refused."""
    policy = _ROUTE_POLICY.get()
    if route_for(route).provenance != "OFFICIAL" and not (policy is not None and policy.allow_unverified_relay):
        return "RELAY_NOT_ALLOWED"
    if policy is not None:
        if policy.conflict or (policy.route is not None and policy.route != route):
            return "ROUTE_MIXED"
        policy.pin(route)
    return None


def transport_version_for(base_version: str, route: str, test_client: bool) -> str:
    """The only version a transport may report: base, route suffix, then the test-client marker."""
    return base_version + route_for(route).version_suffix + ("-test-client" if test_client is True else "")


def request_size_bytes(*parts: Any) -> int:
    """Conservative input size: every text part and the output schema, plus fixed overhead."""
    total = REQUEST_OVERHEAD_BYTES
    for part in parts:
        text = part if isinstance(part, str) else canonical_json(part)
        total += len(text.encode("utf-8"))
    return total


def is_timeout(exc: BaseException) -> bool:
    """True for builtin timeouts and the SDKs' APITimeoutError (not a TimeoutError subclass)."""
    if isinstance(exc, TimeoutError):
        return True
    for module_name in ("anthropic", "openai"):
        timeout_type = getattr(sys.modules.get(module_name), "APITimeoutError", None)
        if isinstance(timeout_type, type) and isinstance(exc, timeout_type):
            return True
    return False


def error_detail(exc: BaseException) -> str:
    """Exception class and HTTP status only; messages may echo request content."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    return f"{name}:{status}" if type(status) is int else name


def short_text(value: Any) -> str | None:
    return value[:200] if isinstance(value, str) and value else None


def token_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def parse_json_object(text: str) -> dict[str, Any]:
    value = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("structured output must be a JSON object")
    return value


class CallLedger:
    """Settles each reservation exactly once and collects audit records.

    Not thread-safe: use one transport (and runner) per thread; CostBudget may be shared.
    """

    def __init__(self, binding: ModelBinding, budget: CostBudget, *, test_client: bool = False,
                 route: str = "official") -> None:
        if not isinstance(budget, CostBudget):
            raise TypeError("an explicit CostBudget is required")
        self.binding = binding
        self.budget = budget
        self.test_client = test_client
        self.route = route_for(route).name
        self._records: list[ModelCallRecord] = []

    def record(self, role: str, outcome: str, *, reservation: Reservation | None = None, served: Any = None,
               request_id: Any = None, detail: Any = None, input_tokens: Any = None,
               output_tokens: Any = None, billable_input_tokens: Any = None,
               budget_locked: bool = False) -> None:
        """reservation is the hold for this call (None if nothing was held).
        input_tokens is the raw count kept in the record; billable_input_tokens (default:
        input_tokens) prices the call. Known usage is charged at list price; otherwise the full
        reservation is charged, because a failed or timed-out request may still have been billed.
        A charge above the reservation locks the budget; that record, and any refusal because the
        budget is locked (budget_locked=True), carries budget_overrun=True."""
        if self.route not in ROUTES:
            raise ModelUnavailable("call ledger route is no longer a registered route")
        in_tokens, out_tokens = token_count(input_tokens), token_count(output_tokens)
        billable = in_tokens if billable_input_tokens is None else token_count(billable_input_tokens)
        cost, upper_bound, overrun = None, False, budget_locked is True
        if reservation is not None:
            if billable is not None and out_tokens is not None:
                cost = self.binding.call_cost_usd(billable, out_tokens)
            else:
                cost, upper_bound = reservation.amount, True
            overrun = self.budget.settle(reservation, cost) or overrun
        self._records.append(ModelCallRecord(
            role=role, provider=self.binding.provider, requested_model_id=self.binding.model_id,
            served_model_id=short_text(served), request_id=short_text(request_id), outcome=outcome,
            detail=short_text(detail), input_tokens=in_tokens, output_tokens=out_tokens,
            cost_usd=cost, cost_is_upper_bound=upper_bound, test_client=self.test_client,
            billable_input_tokens=billable if in_tokens is not None and billable is not None else None,
            budget_overrun=overrun, route=self.route,
        ))

    def drain(self) -> tuple[ModelCallRecord, ...]:
        records, self._records = tuple(self._records), []
        return records
