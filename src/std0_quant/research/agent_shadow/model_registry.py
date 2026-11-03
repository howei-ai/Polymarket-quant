"""Controlled provider bindings for the K3 reasoner and Jev judge roles.

K3 runs only on Anthropic and Jev only on OpenAI; there is no cross-provider
fallback. Model IDs, effort, limits and prices change only by editing this file.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType

from .interfaces import ModelUnavailable


ROLE_PROVIDERS = MappingProxyType({"K3": "anthropic", "JEV": "openai"})
PROVIDER_EFFORTS = MappingProxyType({
    "anthropic": frozenset({"low", "medium", "high", "xhigh", "max"}),
    "openai": frozenset({"low", "medium", "high", "xhigh", "max"}),
})
MAX_OUTPUT_TOKENS_LIMIT = 128_000


def _model_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(c.isspace() for c in value):
        raise ValueError(f"{name} must be a nonempty model id without whitespace")
    if "latest" in value.lower():
        raise ValueError(f"{name} must be pinned, not a latest alias")
    return value


def _positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _nonnegative(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class ModelBinding:
    role: str
    provider: str
    model_id: str
    effort: str
    max_output_tokens: int
    max_input_bytes: int
    timeout_seconds: float
    max_retries: int
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    accepted_served_model_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in ROLE_PROVIDERS:
            raise ValueError("unknown model role")
        if self.provider != ROLE_PROVIDERS[self.role]:
            raise ValueError(f"{self.role} must use provider {ROLE_PROVIDERS[self.role]}")
        _model_name(self.model_id, "model_id")
        if self.effort not in PROVIDER_EFFORTS[self.provider]:
            raise ValueError("unsupported effort for provider")
        for name, upper in (("max_output_tokens", MAX_OUTPUT_TOKENS_LIMIT), ("max_input_bytes", 1_000_000)):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= upper:
                raise ValueError(f"{name} out of range")
        if _positive(self.timeout_seconds, "timeout_seconds") > 600:
            raise ValueError("timeout_seconds out of range")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 3:
            raise ValueError("max_retries out of range")
        _positive(self.input_usd_per_mtok, "input_usd_per_mtok")
        _positive(self.output_usd_per_mtok, "output_usd_per_mtok")
        accepted = self.accepted_served_model_ids
        if not isinstance(accepted, tuple) or self.model_id not in accepted:
            raise ValueError("accepted_served_model_ids must be a tuple containing model_id")
        for served in accepted:
            _model_name(served, "accepted served model id")

    def max_call_cost_usd(self, input_bytes: int) -> float:
        """Upper bound for one call including SDK retries: every input byte counted
        as a token and the full output budget, for each possible attempt."""
        per_attempt = (input_bytes * self.input_usd_per_mtok
                       + self.max_output_tokens * self.output_usd_per_mtok) / 1_000_000
        return per_attempt * (self.max_retries + 1)

    def call_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_usd_per_mtok
                + output_tokens * self.output_usd_per_mtok) / 1_000_000


# Facts verified 2026-11-02: claude-fable-5-1 $10/$50 per MTok, 128K output;
# gpt-6.1-sol $2/$10 per MTok (input over 272K tokens is billed higher, so
# max_input_bytes keeps requests far below that), 128K output, alias only.
# max_retries=0: the SDKs must not retry silently (a timed-out attempt may still
# be billed); a failed call ends BLOCKED.
MODEL_BINDINGS = MappingProxyType({
    "K3": ModelBinding(
        role="K3", provider="anthropic", model_id="claude-fable-5-1", effort="high",
        max_output_tokens=16_000, max_input_bytes=200_000, timeout_seconds=300.0,
        max_retries=0, input_usd_per_mtok=10.0, output_usd_per_mtok=50.0,
        accepted_served_model_ids=("claude-fable-5-1",),
    ),
    "JEV": ModelBinding(
        role="JEV", provider="openai", model_id="gpt-6.1-sol", effort="high",
        max_output_tokens=16_000, max_input_bytes=200_000, timeout_seconds=300.0,
        max_retries=0, input_usd_per_mtok=2.0, output_usd_per_mtok=10.0,
        accepted_served_model_ids=("gpt-6.1-sol",),
    ),
})

_IDENTITY_FIELDS = ("role", "provider", "model_id", "effort", "input_usd_per_mtok",
                    "output_usd_per_mtok", "accepted_served_model_ids")
_TIGHTEN_ONLY_FIELDS = ("max_output_tokens", "max_input_bytes", "timeout_seconds", "max_retries")


def binding_for(role: str) -> ModelBinding:
    try:
        return MODEL_BINDINGS[role]
    except KeyError as exc:
        raise ValueError("unknown model role") from exc


def require_registered_model(role: str, model_id: object) -> ModelBinding:
    binding = binding_for(role)
    _model_name(model_id, f"{role} model_id")
    if model_id != binding.model_id:
        raise ValueError(f"{role} model_id must be the registered {binding.provider} model {binding.model_id}")
    return binding


def require_registered_identity(binding: object, role: str) -> ModelBinding:
    """Model identity, effort and prices are fixed; operational limits may only be tightened."""
    registered = binding_for(role)
    if type(binding) is not ModelBinding or any(  # exact type: a subclass could reprice calls
        getattr(binding, name) != getattr(registered, name) for name in _IDENTITY_FIELDS
    ):
        raise ValueError(f"{role} binding must keep the registered model identity, effort and prices")
    if any(getattr(binding, name) > getattr(registered, name) for name in _TIGHTEN_ONLY_FIELDS):
        raise ValueError(f"{role} binding may tighten but never loosen registered limits")
    return binding


class Reservation:
    """Single-use hold returned by CostBudget.reserve and consumed by CostBudget.settle."""

    __slots__ = ("_amount", "settled", "_budget")

    def __init__(self, budget: "CostBudget", amount: float) -> None:
        self._amount = amount
        self.settled = False
        self._budget = budget

    @property
    def amount(self) -> float:
        return self._amount


class CostBudget:
    """Spend guard shared by every call that uses it (for a runner: all rows it processes).

    reserve() holds a conservative upper bound before a call and returns a Reservation;
    settle() consumes it exactly once, releasing the hold and charging the actual (or,
    when unknown, the reserved) amount. A charge above its reservation means the upper
    bound failed: the true amount is still charged and the budget locks, so every later
    reserve() is refused. Thread-safe.
    """

    def __init__(self, limit_usd: float) -> None:
        self._limit = _positive(limit_usd, "limit_usd")
        self._spent = 0.0
        self._held = 0.0
        self._locked = False
        self._lock = threading.Lock()

    @property
    def limit_usd(self) -> float:
        return self._limit

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent

    @property
    def held_usd(self) -> float:
        with self._lock:
            return self._held

    @property
    def locked(self) -> bool:
        with self._lock:
            return self._locked

    def reserve(self, max_cost_usd: float) -> Reservation:
        amount = _nonnegative(max_cost_usd, "max_cost_usd")
        with self._lock:
            if self._locked:
                raise ModelUnavailable("model call budget locked after a charge exceeded its reservation")
            if self._spent + self._held + amount > self._limit:
                raise ModelUnavailable("model call budget exceeded")
            self._held += amount
        return Reservation(self, amount)

    def settle(self, reservation: Reservation, charged_usd: float) -> bool:
        """Returns True when the charge exceeded the reservation (the budget is then locked)."""
        if not isinstance(reservation, Reservation) or reservation._budget is not self:
            raise ValueError("reservation does not belong to this budget")
        charged = _nonnegative(charged_usd, "charged_usd")
        with self._lock:
            if reservation.settled:
                raise ValueError("reservation already settled")
            reservation.settled = True
            self._held = max(0.0, self._held - reservation.amount)
            self._spent += charged
            overrun = charged > reservation.amount
            if overrun:
                self._locked = True
            return overrun
