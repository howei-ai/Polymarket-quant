"""Exercise the pinned OpenAI SDK with local HTTP responses, without network I/O."""

import json
from decimal import Decimal
import math

import httpx2
import pytest
from openai.types.responses import Response

from agent_shadow_fakes import FakeReasoner, judge_response, pit_row
from std0_quant.research.agent_shadow import provider_clients
from std0_quant.research.agent_shadow.model_registry import CostBudget, binding_for, route_for
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import GateEvidence, K3_ROLES


EVIDENCE = GateEvidence(True, True, True)


def _wire_response(body, number):
    request = json.loads(body["input"])
    report = FakeReasoner().analyze(request) if "role" in request else judge_response()
    return {
        "id": f"resp_{number}", "object": "response", "created_at": 1_000_000_000,
        "model": body["model"], "status": "completed", "error": None, "incomplete_details": None,
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "output": [{"id": f"msg_{number}", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps(report), "annotations": []}]}],
        "usage": {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500,
                  "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100},
                  "output_tokens_details": {"reasoning_tokens": 100}},
    }


def _sdk_models(monkeypatch, route, *, alter=None, http_status=200):
    seen = []

    def handle(request):
        body = json.loads(request.content)
        seen.append((request, body))
        response = _wire_response(body, len(seen))
        if alter is not None:
            alter(response)
        if http_status == 200 and alter is None:
            Response.model_validate(response)  # the fixture follows the SDK's complete response schema
        if http_status != 200:
            response = {"error": {"message": "local test failure", "type": "api_error", "code": None}}
        return httpx2.Response(http_status, json=response, headers={"x-request-id": f"req_{len(seen)}"})

    def local_http(sdk_http_client, proxy_url):
        return sdk_http_client(trust_env=False, transport=httpx2.MockTransport(handle))

    monkeypatch.setattr(provider_clients, "_http_client", local_http)
    key_name = "STD0_RELAY_API_KEY" if route == "relay_mtu" else "OPENAI_API_KEY"
    clients = [provider_clients.build_openai_client(environ={key_name: "sk-sdk-test"}, route=route)
               for _ in range(2)]
    budget = CostBudget(10.0)
    models = provider_clients.build_shadow_models(*clients, budget=budget, network_enabled=True, route=route)
    return models, clients, seen, budget


@pytest.mark.parametrize("route", ["official", "relay_mtu"])
def test_full_pipeline_uses_the_real_sdk_without_network(monkeypatch, route):
    models, clients, seen, budget = _sdk_models(monkeypatch, route)
    try:
        result = ShadowRunner(*models, allow_unverified_relay=route == "relay_mtu").run(pit_row(), EVIDENCE)
        assert result.gate_status == "SHADOW_ACCEPT"
        assert (result.route, result.provenance) == (route, route_for(route).provenance)
        assert [call.role for call in result.model_calls] == list(K3_ROLES) + ["JEV"]
        assert len(seen) == 8 and budget.held_usd == 0 and not budget.locked
        for number, ((request, body), record) in enumerate(zip(seen, result.model_calls), 1):
            expected_model = "gpt-6-astra" if number < 8 else "gpt-6.1-sol"
            assert str(request.url) == route_for(route).base_urls["openai"] + "/responses"
            assert request.headers["authorization"] == "Bearer sk-sdk-test"
            assert request.method == "POST" and body["model"] == record.served_model_id == expected_model
            assert body["store"] is False and body["reasoning"] == {"effort": "high"}
            assert body["max_output_tokens"] == 16_000 and body["text"]["format"]["strict"] is True
            assert not {"temperature", "top_p", "top_logprobs", "messages", "output_config", "tools"} & body.keys()
            assert (record.input_tokens, record.billable_input_tokens, record.request_id) == (1200, 1350, f"req_{number}")
            assert record.outcome == "OK" and record.test_client is False
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("http_status", [429, 500])
def test_sdk_does_not_retry_or_fall_back_on_http_errors(monkeypatch, http_status):
    models, clients, seen, budget = _sdk_models(monkeypatch, "relay_mtu", http_status=http_status)
    try:
        result = ShadowRunner(*models, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
        assert result.gate_status == "BLOCKED" and len(seen) == 1
        (record,) = result.model_calls
        assert record.outcome == "API_ERROR" and record.request_id == "req_1"
        assert record.cost_is_upper_bound and budget.spent_usd == record.cost_usd and budget.held_usd == 0
        assert "local test failure" not in json.dumps(result.to_dict())
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("missing", ["all", "writes", "reads"])
def test_missing_sdk_cache_counters_preserve_raw_usage_and_charge_reservation(monkeypatch, missing):
    def remove(response):
        if missing == "all":
            response["usage"].pop("input_tokens_details")
        else:
            name = "cache_write_tokens" if missing == "writes" else "cached_tokens"
            response["usage"]["input_tokens_details"].pop(name)

    models, clients, seen, budget = _sdk_models(monkeypatch, "relay_mtu", alter=remove)
    try:
        result = ShadowRunner(*models, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
        assert result.gate_status == "SHADOW_ACCEPT" and len(seen) == 8
        assert all(call.input_tokens == 1200 and call.output_tokens == 300
                   and call.billable_input_tokens is None and call.cost_is_upper_bound for call in result.model_calls)
        assert budget.spent_usd == pytest.approx(sum(call.cost_usd for call in result.model_calls))
        assert budget.held_usd == 0 and not budget.locked
    finally:
        for client in clients:
            client.close()


def test_sdk_clients_send_nothing_without_relay_opt_in(monkeypatch):
    models, clients, seen, budget = _sdk_models(monkeypatch, "relay_mtu")
    try:
        result = ShadowRunner(*models).run(pit_row(), EVIDENCE)
        assert result.gate_reasons == ("RELAY_NOT_ALLOWED",) and not seen
        assert budget.spent_usd == budget.held_usd == 0
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("role", ["K3", "JEV"])
@pytest.mark.parametrize("bad_shape", ["output", "content"])
@pytest.mark.parametrize("billing", ["known", "missing_cache", "unknown_input"])
def test_malformed_sdk_output_preserves_usage_and_locks_on_overrun(monkeypatch, role, bad_shape, billing):
    binding = binding_for(role)

    def alter(response):
        if response["model"] != binding.model_id:
            return
        response["usage"]["input_tokens"] = 900_000 if billing != "unknown_input" else None
        response["usage"]["output_tokens"] = 300 if billing != "unknown_input" else 900_000
        if billing != "known":
            response["usage"]["input_tokens_details"].pop("cache_write_tokens")
        if bad_shape == "output":
            response["output"] = 7
        else:
            response["output"][0]["content"] = 7

    models, clients, seen, budget = _sdk_models(monkeypatch, "relay_mtu", alter=alter)
    try:
        runner = ShadowRunner(*models, allow_unverified_relay=True)
        result = runner.run(pit_row(), EVIDENCE)
        assert result.gate_status == "BLOCKED" and "BUDGET_OVERRUN" in result.gate_reasons
        record = result.model_calls[-1]
        assert record.role == ("MICROSTRUCTURE" if role == "K3" else "JEV")
        assert record.outcome == "INVALID_OUTPUT" and record.detail == "unexpected_response_shape"
        assert record.request_id == f"req_{len(seen)}" and record.served_model_id == binding.model_id
        assert (record.input_tokens, record.output_tokens) == ((900_000, 300) if billing != "unknown_input"
                                                             else (None, 900_000))
        assert record.cost_is_upper_bound is (billing != "known")
        if billing == "known":
            assert record.billable_input_tokens == 900_150
            assert record.cost_usd == pytest.approx(binding.call_cost_usd(900_150, 300))
        elif billing == "missing_cache":
            assert record.billable_input_tokens is None
            assert record.cost_usd == pytest.approx(binding.call_cost_usd(2_250_000, 300))
        else:
            assert record.billable_input_tokens is None
            assert record.cost_usd > binding.call_cost_usd(0, 900_000 - binding.max_output_tokens)
        assert record.budget_overrun and budget.locked and budget.held_usd == 0
        assert budget.spent_usd == pytest.approx(sum(call.cost_usd for call in result.model_calls))
        requests = len(seen)
        again = runner.run(pit_row(condition_id="after-malformed-response"), EVIDENCE)
        assert again.gate_status == "BLOCKED" and "BUDGET_OVERRUN" in again.gate_reasons
        assert len(seen) == requests
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("role", ["K3", "JEV"])
@pytest.mark.parametrize("token_field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("exponent", [309, 315])
@pytest.mark.parametrize("known_cache", [True, False])
def test_extreme_sdk_usage_releases_hold_and_stops_further_requests(monkeypatch, role, token_field,
                                                                   exponent, known_cache):
    binding = binding_for(role)
    huge = 10 ** exponent

    def alter(response):
        if response["model"] != binding.model_id:
            return
        response["usage"][token_field] = huge
        if not known_cache:
            response["usage"]["input_tokens_details"].pop("cache_write_tokens")
            if token_field == "output_tokens":
                response["usage"]["input_tokens"] = None
        response["output"] = 7

    models, clients, seen, budget = _sdk_models(monkeypatch, "relay_mtu", alter=alter)
    try:
        runner = ShadowRunner(*models, allow_unverified_relay=True)
        result = runner.run(pit_row(), EVIDENCE)
        assert result.gate_status == "BLOCKED" and "BUDGET_OVERRUN" in result.gate_reasons
        record = result.model_calls[-1]
        assert getattr(record, token_field) == huge
        assert record.outcome == "INVALID_OUTPUT" and record.budget_overrun
        assert budget.locked and budget.held_usd == 0
        if exponent == 309:
            assert math.isfinite(record.cost_usd)
            input_rate, output_rate = Decimal(str(binding.input_usd_per_mtok)), Decimal(str(binding.output_usd_per_mtok))
            if known_cache:
                expected = (Decimal(record.billable_input_tokens) * input_rate
                            + Decimal(record.output_tokens) * output_rate) / 1_000_000
            elif token_field == "input_tokens":
                expected = (Decimal(huge) * Decimal("2.5") * input_rate + 300 * output_rate) / 1_000_000
            else:
                expected = Decimal(huge - binding.max_output_tokens) * output_rate / 1_000_000
            assert record.cost_usd == pytest.approx(float(expected))
            assert budget.spent_usd == pytest.approx(sum(call.cost_usd for call in result.model_calls))
        else:
            assert record.cost_usd is None and record.cost_is_upper_bound is False
            assert "cost_not_representable" in record.detail
            assert 0 < budget.spent_usd < budget.limit_usd  # only the reservation is charged; full cost is unknown
        count = len(seen)
        again = runner.run(pit_row(condition_id="after-extreme-usage"), EVIDENCE)
        assert again.gate_status == "BLOCKED" and "BUDGET_OVERRUN" in again.gate_reasons
        assert len(seen) == count
    finally:
        for client in clients:
            client.close()
