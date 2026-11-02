"""Shared fail-closed helpers for provider transports; no SDK is imported at module load."""

from __future__ import annotations

import json
import sys
import weakref
from typing import Any

from .audit import canonical_json
from .model_registry import CostBudget, ModelBinding, Reservation
from .schemas import ModelCallRecord


ANTHROPIC_BASE_URL = "https://api.anthropic.com"
OPENAI_BASE_URL = "https://api.openai.com/v1"
# Bytes added to every request's size estimate for message framing and SDK overhead.
REQUEST_OVERHEAD_BYTES = 2_048

_TRUSTED_CLIENTS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def register_trusted_client(client: Any) -> Any:
    """Called only by provider_clients factories (pinned base URL, trust_env off)."""
    _TRUSTED_CLIENTS.add(client)
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


def require_trusted_client(client: Any, *, expected_base_url: str, allow_test_client: bool) -> bool:
    """Returns True when an offline test stand-in is accepted (recorded in the transport version).

    Real SDK clients are accepted only when built by provider_clients, even in tests.
    """
    if type(allow_test_client) is not bool:
        raise TypeError("allow_test_client must be an explicit boolean")
    if client in _TRUSTED_CLIENTS and str(getattr(client, "base_url", "")).rstrip("/") == expected_base_url:
        return False
    if allow_test_client and not _is_sdk_client(client):
        return True
    raise ValueError("provider client must come from provider_clients with the pinned base URL")


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

    def __init__(self, binding: ModelBinding, budget: CostBudget, *, test_client: bool = False) -> None:
        if not isinstance(budget, CostBudget):
            raise TypeError("an explicit CostBudget is required")
        self.binding = binding
        self.budget = budget
        self.test_client = test_client
        self._records: list[ModelCallRecord] = []

    def record(self, role: str, outcome: str, *, reservation: Reservation | None = None, served: Any = None,
               request_id: Any = None, detail: Any = None, input_tokens: Any = None,
               output_tokens: Any = None) -> None:
        """reservation is the hold for this call (None if nothing was held).
        Known usage is charged at list price; otherwise the full reservation is charged,
        because a failed or timed-out request may still have been billed."""
        in_tokens, out_tokens = token_count(input_tokens), token_count(output_tokens)
        cost, upper_bound = None, False
        if reservation is not None:
            if in_tokens is not None and out_tokens is not None:
                cost = self.binding.call_cost_usd(in_tokens, out_tokens)
            else:
                cost, upper_bound = reservation.amount, True
            self.budget.settle(reservation, cost)
        self._records.append(ModelCallRecord(
            role=role, provider=self.binding.provider, requested_model_id=self.binding.model_id,
            served_model_id=short_text(served), request_id=short_text(request_id), outcome=outcome,
            detail=short_text(detail), input_tokens=in_tokens, output_tokens=out_tokens,
            cost_usd=cost, cost_is_upper_bound=upper_bound, test_client=self.test_client,
        ))

    def drain(self) -> tuple[ModelCallRecord, ...]:
        records, self._records = tuple(self._records), []
        return records
