import pytest

from agent_shadow_fakes import FakeReasoner, pit_row
from agent_shadow_provider_fakes import FakeAnthropicClient, FakeOpenAIClient
from std0_quant.research.agent_shadow import provider_clients
from std0_quant.research.agent_shadow.anthropic_k3_transport import AnthropicK3Transport
from std0_quant.research.agent_shadow.interfaces import ModelUnavailable
from std0_quant.research.agent_shadow.jev_client import JevClient
from std0_quant.research.agent_shadow.k3_client import K3Client
from std0_quant.research.agent_shadow.model_registry import ROUTES, CostBudget, route_for
from std0_quant.research.agent_shadow.openai_jev_transport import OpenAIJevTransport
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import ROUTE_NAMES, GateEvidence, ModelCallRecord


EVIDENCE = GateEvidence(True, True, True)
RELAY_KEY = {"STD0_RELAY_API_KEY": "sk-relay-test"}
OFFICIAL_KEYS = {"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": "sk-test"}


def _relay_clients(monkeypatch, anthropic_fake=None, openai_fake=None):
    """Factory-built relay clients whose request methods answer from offline fakes."""
    anthropic_client = provider_clients.build_anthropic_client(environ=RELAY_KEY, route="relay_mtu")
    openai_client = provider_clients.build_openai_client(environ=RELAY_KEY, route="relay_mtu")
    anthropic_fake = anthropic_fake or FakeAnthropicClient()
    openai_fake = openai_fake or FakeOpenAIClient()
    monkeypatch.setattr(anthropic_client, "with_options", lambda **options: anthropic_fake)
    monkeypatch.setattr(openai_client, "with_options", lambda **options: openai_fake)
    return anthropic_client, openai_client


def _official_clients(monkeypatch, anthropic_fake=None, openai_fake=None):
    anthropic_client = provider_clients.build_anthropic_client(environ=OFFICIAL_KEYS)
    openai_client = provider_clients.build_openai_client(environ=OFFICIAL_KEYS)
    monkeypatch.setattr(anthropic_client, "with_options", lambda **options: anthropic_fake or FakeAnthropicClient())
    monkeypatch.setattr(openai_client, "with_options", lambda **options: openai_fake or FakeOpenAIClient())
    return anthropic_client, openai_client


def _relay_models(monkeypatch, **fakes):
    anthropic_client, openai_client = _relay_clients(monkeypatch, **fakes)
    return provider_clients.build_shadow_models(anthropic_client, openai_client, budget=CostBudget(5.0),
                                                network_enabled=True, route="relay_mtu")


# Registry

def test_route_names_match_the_registry_and_urls_are_pinned_https():
    assert tuple(ROUTES) == ROUTE_NAMES
    relay = route_for("relay_mtu")
    assert dict(relay.base_urls) == {"anthropic": "https://mtu.mtuopenai.xyz", "openai": "https://mtu.mtuopenai.xyz/v1"}
    assert relay.provenance == "UNVERIFIED_RELAY" and route_for("official").provenance == "OFFICIAL"
    for route in ROUTES.values():
        assert all(url.startswith("https://") for url in route.base_urls.values())
    for bad in ("relay", "RELAY_MTU", None, 1):
        with pytest.raises(ValueError):
            route_for(bad)


# Client factories: keys and URLs never cross routes

def test_relay_clients_use_only_the_relay_key_and_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://evil.example/v1")
    anthropic_client = provider_clients.build_anthropic_client(environ=RELAY_KEY, route="relay_mtu")
    openai_client = provider_clients.build_openai_client(environ=RELAY_KEY, route="relay_mtu")
    assert str(anthropic_client.base_url).rstrip("/") == "https://mtu.mtuopenai.xyz"
    assert str(openai_client.base_url).rstrip("/") == "https://mtu.mtuopenai.xyz/v1"
    assert anthropic_client.api_key == openai_client.api_key == "sk-relay-test"


def test_relay_clients_refuse_while_official_keys_are_present():
    for name in OFFICIAL_KEYS:
        with pytest.raises(ModelUnavailable):
            provider_clients.build_anthropic_client(environ={**RELAY_KEY, name: "x"}, route="relay_mtu")
        with pytest.raises(ModelUnavailable):
            provider_clients.build_openai_client(environ={**RELAY_KEY, name: "x"}, route="relay_mtu")


def test_official_clients_never_read_the_relay_key():
    with pytest.raises(ModelUnavailable):
        provider_clients.build_anthropic_client(environ=RELAY_KEY)
    with pytest.raises(ModelUnavailable):
        provider_clients.build_openai_client(environ=RELAY_KEY)
    client = provider_clients.build_anthropic_client(environ={**OFFICIAL_KEYS, **RELAY_KEY})
    assert client.api_key == "sk-ant-test" and str(client.base_url).rstrip("/") == "https://api.anthropic.com"


def test_relay_needs_its_key_and_rejects_openai_org_and_project():
    with pytest.raises(ModelUnavailable):
        provider_clients.build_anthropic_client(environ={}, route="relay_mtu")
    for extra in ({"organization": "org"}, {"project": "proj"}):
        with pytest.raises(ValueError):
            provider_clients.build_openai_client(environ=RELAY_KEY, route="relay_mtu", **extra)
    with pytest.raises(ValueError):
        provider_clients.build_anthropic_client(environ=RELAY_KEY, route="nope")


# Transports: a client is trusted only on the route it was built for

def test_clients_cannot_cross_routes_in_transports():
    relay = provider_clients.build_anthropic_client(environ=RELAY_KEY, route="relay_mtu")
    official = provider_clients.build_anthropic_client(environ=OFFICIAL_KEYS)
    with pytest.raises(ValueError):
        AnthropicK3Transport(relay, budget=CostBudget(1.0))
    with pytest.raises(ValueError):
        AnthropicK3Transport(official, budget=CostBudget(1.0), route="relay_mtu")
    relay_openai = provider_clients.build_openai_client(environ=RELAY_KEY, route="relay_mtu")
    with pytest.raises(ValueError):
        OpenAIJevTransport(relay_openai, budget=CostBudget(1.0))
    assert AnthropicK3Transport(relay, budget=CostBudget(1.0), route="relay_mtu").transport_version \
        == "anthropic-k3-transport-v4-relay-unverified"
    assert OpenAIJevTransport(relay_openai, budget=CostBudget(1.0), route="relay_mtu").transport_version \
        == "openai-jev-transport-v4-relay-unverified"


def test_relay_test_client_version_marks_both():
    transport = AnthropicK3Transport(FakeAnthropicClient(), budget=CostBudget(1.0), allow_test_client=True,
                                     route="relay_mtu")
    assert transport.transport_version == "anthropic-k3-transport-v4-relay-unverified-test-client"


# Runner: explicit opt-in, provenance, mixing

def test_relay_rows_are_blocked_before_any_call_without_opt_in(monkeypatch):
    anthropic_fake, openai_fake = FakeAnthropicClient(), FakeOpenAIClient()
    k3, jev = _relay_models(monkeypatch, anthropic_fake=anthropic_fake, openai_fake=openai_fake)
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and result.gate_reasons == ("RELAY_NOT_ALLOWED",)
    assert not anthropic_fake.calls and not openai_fake.calls and result.model_calls == ()
    assert (result.route, result.provenance) == ("relay_mtu", "UNVERIFIED_RELAY")


def test_opted_in_relay_rows_run_and_are_labelled_unverified(monkeypatch):
    k3, jev = _relay_models(monkeypatch)
    result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert result.gate_status == "SHADOW_ACCEPT"
    assert (result.route, result.provenance) == ("relay_mtu", "UNVERIFIED_RELAY")
    assert result.k3_transport_version == "anthropic-k3-transport-v4-relay-unverified"
    assert len(result.model_calls) == 8
    assert all(call.route == "relay_mtu" and call.test_client is False for call in result.model_calls)


def test_official_rows_stay_official(monkeypatch):
    anthropic_client, openai_client = _official_clients(monkeypatch)
    k3, jev = provider_clients.build_shadow_models(anthropic_client, openai_client, budget=CostBudget(5.0),
                                                   network_enabled=True)
    result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert (result.route, result.provenance) == ("official", "OFFICIAL")
    assert all(call.route == "official" for call in result.model_calls)


def test_relay_test_clients_are_labelled_test_client():
    k3, jev = provider_clients.build_shadow_models(FakeAnthropicClient(), FakeOpenAIClient(), budget=CostBudget(5.0),
                                                   network_enabled=True, allow_test_clients=True, route="relay_mtu")
    result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert (result.route, result.provenance) == ("relay_mtu", "TEST_CLIENT")


def test_mixed_routes_are_blocked_before_any_call(monkeypatch):
    anthropic_fake = FakeAnthropicClient()
    relay_anthropic, _ = _relay_clients(monkeypatch, anthropic_fake=anthropic_fake)
    _, official_openai = _official_clients(monkeypatch)
    k3 = K3Client(AnthropicK3Transport(relay_anthropic, budget=CostBudget(5.0), route="relay_mtu"),
                  network_enabled=True)
    jev = JevClient(model_id="gpt-6.1-sol", transport=OpenAIJevTransport(official_openai, budget=CostBudget(5.0)),
                    network_enabled=True)
    result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and result.gate_reasons == ("MODEL_ROUTE_MIXED",)
    assert not anthropic_fake.calls and result.provenance == "UNSPECIFIED"


def test_relabelling_a_relay_transport_as_official_is_refused_and_blocks(monkeypatch):
    k3, jev = _relay_models(monkeypatch)
    k3._transport._ledger.route = "official"
    k3.transport_version = k3._transport.transport_version = "anthropic-k3-transport-v4"
    with pytest.raises(ModelUnavailable):
        k3.analyze({"role": "MICROSTRUCTURE"})
    assert k3._transport.drain_call_records()[0].outcome == "CLIENT_UNTRUSTED"
    result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons
    assert result.provenance == "UNSPECIFIED"


def test_dropping_the_relay_suffix_blocks(monkeypatch):
    k3, jev = _relay_models(monkeypatch)
    jev.transport_version = jev._transport.transport_version = "openai-jev-transport-v4"
    result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons


def test_unregistered_models_are_unspecified():
    _, jev = provider_clients.build_shadow_models(FakeAnthropicClient(), FakeOpenAIClient(), budget=CostBudget(5.0),
                                                  network_enabled=True, allow_test_clients=True)
    result = ShadowRunner(FakeReasoner(), jev).run(pit_row(), EVIDENCE)
    assert (result.route, result.provenance) == ("unspecified", "UNSPECIFIED")


def test_opt_in_must_be_a_real_boolean():
    k3, jev = provider_clients.build_shadow_models(FakeAnthropicClient(), FakeOpenAIClient(), budget=CostBudget(5.0),
                                                   network_enabled=True, allow_test_clients=True)
    with pytest.raises(TypeError):
        ShadowRunner(k3, jev, allow_unverified_relay="yes")


def test_records_reject_unknown_routes():
    with pytest.raises(ValueError):
        ModelCallRecord(role="JEV", provider="openai", requested_model_id="gpt-6.1-sol", served_model_id=None,
                        request_id=None, outcome="OK", detail=None, input_tokens=1, output_tokens=1, cost_usd=0.0,
                        cost_is_upper_bound=False, test_client=False, route="elsewhere")


# Review fixes: the route policy is enforced at call time, however models are wrapped (F1).

class _WrappedK3(K3Client):
    """An ordinary subclass: unregistered for the runner, but around a real relay transport."""


def _official_jev(monkeypatch, openai_fake=None):
    _, official_openai = _official_clients(monkeypatch, openai_fake=openai_fake)
    return JevClient(model_id="gpt-6.1-sol", transport=OpenAIJevTransport(official_openai, budget=CostBudget(5.0)),
                     network_enabled=True)


def _wrapped_relay_k3(monkeypatch, anthropic_fake):
    relay_anthropic, _ = _relay_clients(monkeypatch, anthropic_fake=anthropic_fake)
    return _WrappedK3(AnthropicK3Transport(relay_anthropic, budget=CostBudget(5.0), route="relay_mtu"),
                      network_enabled=True)


def test_wrapped_relay_model_cannot_call_without_opt_in(monkeypatch):
    anthropic_fake = FakeAnthropicClient()
    k3 = _wrapped_relay_k3(monkeypatch, anthropic_fake)
    result = ShadowRunner(k3, _official_jev(monkeypatch)).run(pit_row(), EVIDENCE)
    assert not anthropic_fake.calls
    assert result.gate_status == "BLOCKED"


def test_wrapped_relay_model_cannot_mix_with_an_official_model_even_with_opt_in(monkeypatch):
    anthropic_fake = FakeAnthropicClient()
    k3 = _wrapped_relay_k3(monkeypatch, anthropic_fake)
    result = ShadowRunner(k3, _official_jev(monkeypatch), allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert not anthropic_fake.calls
    assert result.gate_status == "BLOCKED"


def test_relay_model_next_to_an_unregistered_model_is_blocked_before_any_call(monkeypatch):
    openai_fake = FakeOpenAIClient()
    _, relay_openai = _relay_clients(monkeypatch, openai_fake=openai_fake)
    jev = JevClient(model_id="gpt-6.1-sol", transport=OpenAIJevTransport(relay_openai, budget=CostBudget(5.0),
                                                                        route="relay_mtu"), network_enabled=True)
    result = ShadowRunner(FakeReasoner(), jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert result.gate_reasons == ("MODEL_ROUTE_MIXED",) and not openai_fake.calls


def test_relay_calls_outside_any_scope_are_refused_and_opt_in_scopes_allow_them(monkeypatch):
    from std0_quant.research.agent_shadow.analysts import analyst_request
    from std0_quant.research.agent_shadow.context import build_shadow_context
    from std0_quant.research.agent_shadow.provider_common import route_policy

    anthropic_fake = FakeAnthropicClient()
    k3 = _wrapped_relay_k3(monkeypatch, anthropic_fake)
    request = analyst_request("MICROSTRUCTURE", build_shadow_context(pit_row()))
    with pytest.raises(ModelUnavailable):
        k3.analyze(request)
    assert k3._transport.drain_call_records()[0].outcome == "RELAY_NOT_ALLOWED" and not anthropic_fake.calls
    with route_policy(allow_unverified_relay=True):
        k3.analyze(request)
    assert len(anthropic_fake.calls) == 1


def test_a_scope_pins_the_first_route_used(monkeypatch):
    from std0_quant.research.agent_shadow.analysts import analyst_request
    from std0_quant.research.agent_shadow.context import build_shadow_context
    from std0_quant.research.agent_shadow.provider_common import route_policy

    official_anthropic, _ = _official_clients(monkeypatch)
    official = K3Client(AnthropicK3Transport(official_anthropic, budget=CostBudget(5.0)), network_enabled=True)
    relay = _wrapped_relay_k3(monkeypatch, FakeAnthropicClient())
    request = analyst_request("MICROSTRUCTURE", build_shadow_context(pit_row()))
    with route_policy(allow_unverified_relay=True):
        official.analyze(request)
        with pytest.raises(ModelUnavailable):
            relay.analyze(request)
    assert relay._transport.drain_call_records()[0].outcome == "ROUTE_MIXED"


def test_unknown_ledger_route_yields_a_blocked_artifact(monkeypatch):
    k3, jev = _relay_models(monkeypatch)
    k3._transport._ledger.route = "relay"
    result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons


def test_relay_clients_check_the_process_environment_too(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_WEBHOOK_SECRET"):
        monkeypatch.setenv(name, "x")
        with pytest.raises(ModelUnavailable):
            provider_clients.build_openai_client(environ=RELAY_KEY, route="relay_mtu")
        with pytest.raises(ModelUnavailable):
            provider_clients.build_anthropic_client(environ=RELAY_KEY, route="relay_mtu")
        monkeypatch.delenv(name)


def test_test_and_real_clients_cannot_share_a_row_and_real_relay_is_never_hidden(monkeypatch):
    openai_fake = FakeOpenAIClient()
    _, relay_openai = _relay_clients(monkeypatch, openai_fake=openai_fake)
    k3 = K3Client(AnthropicK3Transport(FakeAnthropicClient(), budget=CostBudget(5.0), allow_test_client=True,
                                       route="relay_mtu"), network_enabled=True)
    jev = JevClient(model_id="gpt-6.1-sol", transport=OpenAIJevTransport(relay_openai, budget=CostBudget(5.0),
                                                                        route="relay_mtu"), network_enabled=True)
    result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert result.gate_reasons == ("TEST_CLIENT_MIXED",) and not openai_fake.calls
    assert result.provenance == "UNVERIFIED_RELAY"


def test_artifact_schema_ties_provenance_and_call_routes_to_the_route(monkeypatch):
    import dataclasses

    k3, jev = _relay_models(monkeypatch)
    result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    for changes in ({"provenance": "OFFICIAL"}, {"route": "official"},
                    {"route": "unspecified"}, {"route": "unspecified", "provenance": "UNVERIFIED_RELAY"}):
        with pytest.raises(ValueError):
            dataclasses.replace(result, **changes)
    from std0_quant.research.agent_shadow.schemas import ROUTE_PROVENANCE
    assert ROUTE_PROVENANCE == {name: route.provenance for name, route in ROUTES.items()}


def test_trust_requires_the_registered_route_not_just_the_url():
    from std0_quant.research.agent_shadow.provider_common import client_matches, register_trusted_client

    class _Client:  # weak-referenceable; same URL as the official route, registered for the relay
        base_url = "https://api.anthropic.com"

    stand_in = register_trusted_client(_Client(), "relay_mtu")
    official_url = "https://api.anthropic.com"
    assert not client_matches(stand_in, expected_base_url=official_url, test_client=False, route="official")
    assert client_matches(stand_in, expected_base_url=official_url, test_client=False, route="relay_mtu")


def test_nested_scopes_only_narrow(monkeypatch):
    from std0_quant.research.agent_shadow.provider_common import route_policy

    anthropic_fake, openai_fake = FakeAnthropicClient(), FakeOpenAIClient()
    k3, jev = _relay_models(monkeypatch, anthropic_fake=anthropic_fake, openai_fake=openai_fake)
    with route_policy("official"):
        result = ShadowRunner(k3, jev, allow_unverified_relay=True).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and not anthropic_fake.calls and not openai_fake.calls
    assert all(call.outcome in ("RELAY_NOT_ALLOWED", "ROUTE_MIXED") for call in result.model_calls)
    with route_policy(allow_unverified_relay=False):
        with route_policy(allow_unverified_relay=True) as inner:
            assert inner.allow_unverified_relay is False
    with route_policy(allow_unverified_relay=True) as outer:
        with route_policy("relay_mtu"):
            pass
        assert outer.route == "relay_mtu"  # an inner pin also pins the enclosing scope
