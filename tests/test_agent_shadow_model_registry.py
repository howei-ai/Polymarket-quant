import dataclasses
import math
import threading

import pytest

from std0_quant.research.agent_shadow.interfaces import ModelTimeout, ModelUnavailable
from std0_quant.research.agent_shadow.model_registry import (
    MODEL_BINDINGS, CostBudget, binding_for, require_registered_identity, require_registered_model,
)
from std0_quant.research.agent_shadow.schemas import ModelCallRecord


def test_registry_binds_k3_to_astra_and_jev_to_gpt61sol():
    assert set(MODEL_BINDINGS) == {"K3", "JEV"}
    k3, jev = binding_for("K3"), binding_for("JEV")
    assert (k3.provider, k3.model_id, k3.accepted_served_model_ids) == (
        "openai", "gpt-6-astra", ("gpt-6-astra",))
    assert (jev.provider, jev.model_id, jev.accepted_served_model_ids) == (
        "openai", "gpt-6.1-sol", ("gpt-6.1-sol",))
    assert (k3.input_usd_per_mtok, k3.output_usd_per_mtok) == (10.0, 50.0)
    assert (jev.input_usd_per_mtok, jev.output_usd_per_mtok) == (2.0, 10.0)


@pytest.mark.parametrize("changes", [
    {"provider": "anthropic"},
    {"role": "JUDGE"},
    {"model_id": "claude-fable-latest", "accepted_served_model_ids": ("claude-fable-latest",)},
    {"model_id": " claude-fable-5-1"},
    {"effort": "none"},
    {"effort": "minimal"},
    {"max_output_tokens": 0},
    {"max_output_tokens": 128_001},
    {"max_output_tokens": True},
    {"max_input_bytes": 0},
    {"timeout_seconds": 0},
    {"timeout_seconds": float("nan")},
    {"timeout_seconds": 601},
    {"max_retries": -1},
    {"max_retries": 4},
    {"input_usd_per_mtok": 0},
    {"output_usd_per_mtok": float("inf")},
    {"accepted_served_model_ids": ("claude-opus-5-5",)},
    {"accepted_served_model_ids": ["gpt-6-astra"]},
])
def test_binding_validation_rejects(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(binding_for("K3"), **changes)


def test_jev_binding_cannot_move_to_anthropic():
    with pytest.raises(ValueError):
        dataclasses.replace(binding_for("JEV"), provider="anthropic")


def test_registry_is_immutable():
    with pytest.raises(TypeError):
        MODEL_BINDINGS["K3"] = binding_for("JEV")
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding_for("K3").model_id = "claude-opus-5-5"


@pytest.mark.parametrize("role,model_id", [
    ("K3", "gpt-6.1-sol"), ("JEV", "claude-fable-5-1"), ("K3", "kimi-k3"),
    ("K3", "claude-fable-5-1-latest"), ("JEV", "jev-1.13.0"), ("OTHER", "claude-fable-5-1"),
])
def test_require_registered_model_rejects(role, model_id):
    with pytest.raises(ValueError):
        require_registered_model(role, model_id)


def test_identity_is_fixed_but_limits_may_tighten():
    tighter = dataclasses.replace(binding_for("K3"), max_output_tokens=1_000, max_input_bytes=10_000,
                                  timeout_seconds=60.0)
    assert require_registered_identity(tighter, "K3") is tighter
    for changes in ({"input_usd_per_mtok": 1.0}, {"effort": "low"}, {"effort": "max"},
                    {"model_id": "claude-opus-5-5", "accepted_served_model_ids": ("claude-opus-5-5",)},
                    {"accepted_served_model_ids": ("claude-fable-5-1", "claude-fable-5")}):
        with pytest.raises(ValueError):
            require_registered_identity(dataclasses.replace(binding_for("K3"), **changes), "K3")
    with pytest.raises(ValueError):
        require_registered_identity(binding_for("JEV"), "K3")


@pytest.mark.parametrize("role", ["K3", "JEV"])
@pytest.mark.parametrize("changes", [
    {"max_input_bytes": 1_000_000}, {"max_output_tokens": 128_000},
    {"timeout_seconds": 600.0}, {"max_retries": 1},
])
def test_registered_limits_cannot_be_loosened(role, changes):
    with pytest.raises(ValueError):
        require_registered_identity(dataclasses.replace(binding_for(role), **changes), role)


def test_registry_disables_sdk_retries():
    assert binding_for("K3").max_retries == 0 and binding_for("JEV").max_retries == 0


def test_cost_math_counts_every_possible_attempt():
    k3 = binding_for("K3")
    assert math.isclose(k3.call_cost_usd(1_000_000, 0), 10.0)
    assert math.isclose(k3.call_cost_usd(0, 1_000_000), 50.0)
    one_attempt = (1_000 * 2.5 * 10 + 16_000 * 50) / 1_000_000
    assert math.isclose(k3.max_call_cost_usd(1_000), one_attempt)
    assert math.isclose(dataclasses.replace(k3, max_retries=2).max_call_cost_usd(1_000), 3 * one_attempt)


def test_budget_holds_reservations_until_settled():
    budget = CostBudget(1.0)
    first = budget.reserve(0.6)
    assert first.amount == 0.6 and first.settled is False and budget.held_usd == 0.6
    with pytest.raises(ModelUnavailable):
        budget.reserve(0.5)
    budget.settle(first, 0.2)
    assert first.settled is True and (budget.spent_usd, budget.held_usd) == (0.2, 0.0)
    second = budget.reserve(0.8)
    with pytest.raises(ModelUnavailable):
        budget.reserve(0.01)
    for bad in (float("nan"), -0.1):
        with pytest.raises(ValueError):
            budget.settle(second, bad)
    assert second.settled is False and math.isclose(budget.held_usd, 0.8)
    with pytest.raises(ValueError):
        CostBudget(1.0).settle(second, 0.0)
    with pytest.raises(ValueError):
        budget.settle(0.8, 0.0)
    budget.settle(second, 0.0)
    assert budget.limit_usd == 1.0 and budget.held_usd == 0.0
    for bad in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            CostBudget(bad)


def test_reservations_are_single_use_even_with_other_holds_open():
    budget = CostBudget(1.0)
    first, second = budget.reserve(0.4), budget.reserve(0.4)
    budget.settle(first, 0.1)
    with pytest.raises(ValueError):
        budget.settle(first, 0.1)
    assert math.isclose(budget.held_usd, 0.4) and math.isclose(budget.spent_usd, 0.1)
    budget.settle(second, 0.1)
    assert budget.held_usd == 0.0 and math.isclose(budget.spent_usd, 0.2)


def test_budget_reservations_are_thread_safe():
    budget, barrier, granted = CostBudget(1.0), threading.Barrier(8), []

    def worker():
        barrier.wait()
        try:
            budget.reserve(0.4)
            granted.append(True)
        except ModelUnavailable:
            pass

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(granted) == 2 and math.isclose(budget.held_usd, 0.8)


def test_model_timeout_is_both_unavailable_and_timeout():
    assert issubclass(ModelTimeout, ModelUnavailable) and issubclass(ModelTimeout, TimeoutError)


def _record(**changes):
    base = dict(role="MOMENTUM", provider="anthropic", requested_model_id="claude-fable-5-1",
                served_model_id="claude-fable-5-1", request_id="req_1", outcome="OK", detail=None,
                input_tokens=1, output_tokens=2, cost_usd=0.1, cost_is_upper_bound=False, test_client=False, route="official")
    base.update(changes)
    return ModelCallRecord(**base)


@pytest.mark.parametrize("changes", [
    {"role": "TRADER"}, {"outcome": "ORDER_PLACED"}, {"provider": ""}, {"requested_model_id": " "},
    {"input_tokens": -1}, {"output_tokens": 1.5}, {"cost_usd": float("nan")}, {"cost_usd": -1.0},
    {"detail": "x" * 201}, {"request_id": 7}, {"cost_is_upper_bound": 1}, {"test_client": "no"},
    {"cost_is_upper_bound": True, "cost_usd": None},
])
def test_model_call_record_validation(changes):
    with pytest.raises(ValueError):
        _record(**changes)


def test_model_call_record_accepts_jev_and_unknowns():
    record = _record(role="JEV", provider="openai", requested_model_id="gpt-6.1-sol",
                     served_model_id=None, request_id=None, outcome="TIMEOUT",
                     input_tokens=None, output_tokens=None, cost_usd=None)
    assert record.outcome == "TIMEOUT"
