import collections
import dataclasses
import functools
import math
import sys
from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pytest

from agent_shadow_fakes import FakeReasoner, pit_row
from agent_shadow_provider_fakes import FakeAnthropicClient, FakeOpenAIClient
from std0_quant.research.agent_shadow import provider_clients
from std0_quant.research.agent_shadow.analysts import analyst_request
from std0_quant.research.agent_shadow.anthropic_k3_transport import AnthropicK3Transport
from std0_quant.research.agent_shadow.context import build_shadow_context
from std0_quant.research.agent_shadow.interfaces import ModelUnavailable
from std0_quant.research.agent_shadow.k3_client import K3Client
from std0_quant.research.agent_shadow.model_registry import CostBudget
from std0_quant.research.agent_shadow.openai_jev_transport import OpenAIJevTransport
from std0_quant.research.agent_shadow.provider_common import (
    OfflineNetworkBlocked, install_offline_guard, offline_call,
)
from std0_quant.research.agent_shadow.runner import ShadowRunner
from std0_quant.research.agent_shadow.schemas import GateEvidence, ModelCallRecord


EVIDENCE = GateEvidence(True, True, True)


def _models(budget=None):
    return provider_clients.build_shadow_models(
        FakeAnthropicClient(), FakeOpenAIClient(), budget=budget or CostBudget(5.0),
        network_enabled=True, allow_test_clients=True)


def _first_k3_request():
    return analyst_request("MICROSTRUCTURE", build_shadow_context(pit_row()))


def _record(**changes):
    values = dict(role="MICROSTRUCTURE", provider="anthropic", requested_model_id="claude-fable-5-1",
                  served_model_id="claude-fable-5-1", request_id="req_forged", outcome="OK", detail=None,
                  input_tokens=1, output_tokens=1, cost_usd=0.0, cost_is_upper_bound=False, test_client=False)
    values.update(changes)
    return ModelCallRecord(**values)


class _ForgedReasoner(FakeReasoner):
    """Claims the registered labels and reports a real-looking call record (K1)."""
    model_id = "claude-fable-5-1"
    provider = "anthropic"
    transport_version = "anthropic-k3-transport-v3"

    def drain_call_records(self):
        return (_record(),)


# K1: labels and records come only from the exact registered client/transport pair.

def test_registered_pair_keeps_labels_and_records():
    result = ShadowRunner(*_models()).run(pit_row(), EVIDENCE)
    assert result.gate_status == "SHADOW_ACCEPT"
    assert result.k3_transport_version == "anthropic-k3-transport-v3-test-client"
    assert result.jev_transport_version == "openai-jev-transport-v3-test-client"
    assert len(result.model_calls) == 8 and all(call.test_client for call in result.model_calls)


def test_forged_model_object_cannot_claim_labels_or_records():
    _, jev = _models()
    result = ShadowRunner(_ForgedReasoner(), jev).run(pit_row(), EVIDENCE)
    assert (result.k3_provider, result.k3_transport_version) == ("unspecified", "unspecified")
    assert all(call.role == "JEV" for call in result.model_calls)
    assert "MODEL_PROVENANCE_INVALID" not in result.gate_reasons


def test_tampered_label_on_registered_pair_blocks():
    k3, jev = _models()
    k3.transport_version = "anthropic-k3-transport-v3"  # drops the -test-client marker
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons
    assert (result.k3_provider, result.k3_transport_version) == ("unspecified", "unspecified")
    assert all(call.role == "JEV" for call in result.model_calls)


def test_tampered_records_on_registered_pair_block():
    k3, jev = _models()
    k3._transport._ledger.test_client = False  # records now disagree with the transport version
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons


def test_tampered_jev_label_blocks():
    k3, jev = _models()
    jev.provider = "anthropic"
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons


def test_transport_subclass_is_not_registered():
    class LocalTransport(AnthropicK3Transport):
        pass

    transport = LocalTransport(FakeAnthropicClient(), budget=CostBudget(1.0), allow_test_client=True)
    offline = K3Client(transport)
    assert (offline.provider, offline.transport_version) == ("unspecified", "unspecified")
    with pytest.raises(ValueError):
        K3Client(transport, network_enabled=True)


# K2: raw input tokens are recorded; pricing stays conservative.

def test_anthropic_records_raw_input_but_prices_cache_writes_at_double():
    transport = AnthropicK3Transport(FakeAnthropicClient(cache_tokens=(100, 0)), budget=CostBudget(5.0),
                                     allow_test_client=True)
    transport(_first_k3_request())
    (call,) = transport.drain_call_records()
    assert call.input_tokens == 1300
    assert math.isclose(call.cost_usd, (1400 * 10 + 300 * 50) / 1_000_000)


# K3: a charge above its reservation is charged in full and locks the budget.

def test_settle_overrun_locks_the_budget():
    budget = CostBudget(1.0)
    first = budget.reserve(0.1)
    assert budget.settle(first, 0.2) is True
    assert budget.locked and math.isclose(budget.spent_usd, 0.2) and budget.held_usd == 0
    with pytest.raises(ModelUnavailable):
        budget.reserve(0.0)


def test_settle_within_reservation_does_not_lock():
    budget = CostBudget(1.0)
    assert budget.settle(budget.reserve(0.1), 0.1) is False
    assert not budget.locked


def test_usage_above_reservation_is_noted_and_blocks_the_next_call():
    budget = CostBudget(50.0)
    transport = AnthropicK3Transport(FakeAnthropicClient(usage=(900_000, 300)), budget=budget,
                                     allow_test_client=True)
    transport(_first_k3_request())
    with pytest.raises(ModelUnavailable):
        transport(_first_k3_request())
    first, second = transport.drain_call_records()
    assert first.outcome == "OK" and first.budget_overrun and first.detail is None
    assert math.isclose(first.cost_usd, (900_000 * 10 + 300 * 50) / 1_000_000)
    assert second.outcome == "BUDGET_EXCEEDED" and second.budget_overrun and budget.locked


# K5: a test stand-in must not wrap a real SDK or HTTP client.

def _real_anthropic():
    return anthropic.Anthropic(api_key="sk-ant-test", base_url="https://api.anthropic.com")


def _real_openai():
    return openai.OpenAI(api_key="sk-test", base_url="https://api.openai.com/v1")


class _Wrapper:
    def __init__(self, inner):
        self.inner = inner

    def with_options(self, **options):
        return self.inner.with_options(**options)


@pytest.mark.parametrize("wrap", [
    lambda real: _Wrapper(real),
    lambda real: SimpleNamespace(with_options=real.with_options, messages=real.messages),
])
def test_stand_in_plainly_wrapping_real_anthropic_client_is_refused_early(wrap):
    with pytest.raises(ValueError):
        AnthropicK3Transport(wrap(_real_anthropic()), budget=CostBudget(1.0), allow_test_client=True)


def test_stand_in_wrapping_real_openai_client_is_refused():
    with pytest.raises(ValueError):
        OpenAIJevTransport(_Wrapper(_real_openai()), budget=CostBudget(1.0), allow_test_client=True)


def test_stand_in_wrapping_raw_http_client_is_refused():
    with pytest.raises(ValueError):
        AnthropicK3Transport(SimpleNamespace(http=httpx2.Client()), budget=CostBudget(1.0),
                             allow_test_client=True)


def test_fake_holding_an_sdk_error_is_still_a_valid_stand_in():
    error = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    transport = AnthropicK3Transport(FakeAnthropicClient(error=error), budget=CostBudget(1.0),
                                     allow_test_client=True)
    assert transport.transport_version.endswith("-test-client")


# Review round 1 fixes.

ANTHROPIC_KEY = {"ANTHROPIC_API_KEY": "sk-ant-test"}
_GLOBAL_CLIENT = None


def _real_trusted_anthropic():
    return provider_clients.build_anthropic_client(environ=ANTHROPIC_KEY)


def test_fake_swapped_into_real_transport_is_refused_and_blocks():
    budget = CostBudget(5.0)
    transport = AnthropicK3Transport(_real_trusted_anthropic(), budget=budget)
    transport._client = FakeAnthropicClient()
    with pytest.raises(ModelUnavailable):
        transport(_first_k3_request())
    (call,) = transport.drain_call_records()
    assert call.outcome == "CLIENT_UNTRUSTED" and budget.spent_usd == 0
    _, jev = _models()
    result = ShadowRunner(K3Client(transport, network_enabled=True), jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons
    assert (result.k3_provider, result.k3_transport_version) == ("unspecified", "unspecified")


def test_real_client_swapped_into_test_transport_is_refused_before_any_request():
    transport = AnthropicK3Transport(FakeAnthropicClient(), budget=CostBudget(5.0), allow_test_client=True)
    transport._client = _real_trusted_anthropic()
    with pytest.raises(ModelUnavailable):
        transport(_first_k3_request())
    assert transport.drain_call_records()[0].outcome == "CLIENT_UNTRUSTED"


def _forge_real_labels(k3):
    """Every label and the ledger flag say 'real run', but the client is still a fake."""
    k3.transport_version = k3._transport.transport_version = "anthropic-k3-transport-v3"
    k3._transport._ledger.test_client = False


def test_fully_consistent_label_forgery_around_a_fake_blocks():
    k3, jev = _models()
    _forge_real_labels(k3)
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons
    assert not any(call.role != "JEV" for call in result.model_calls)


def test_label_forgery_blocks_even_when_no_call_was_made():
    k3, jev = _models()
    _forge_real_labels(k3)
    result = ShadowRunner(k3, jev).run(pit_row(decision_cutoff_ts_ms=-1), EVIDENCE)
    assert result.gate_reasons == ("CONTEXT_PIT_INVALID", "MODEL_PROVENANCE_INVALID")
    assert (result.k3_provider, result.k3_transport_version) == ("unspecified", "unspecified")


def test_broken_registered_model_yields_blocked_artifact_not_an_exception():
    k3, jev = _models()
    k3._transport._ledger = SimpleNamespace(test_client=True, drain=lambda: ("not a record",))
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons


def test_instance_overrides_of_trust_check_and_drain_are_ignored():
    k3, jev = _models()
    k3._transport._client_still_trusted = lambda: True
    k3._transport.drain_call_records = lambda: ()
    k3.transport_version = k3._transport.transport_version = "anthropic-k3-transport-v3"
    k3._transport._ledger.test_client = False
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons
    assert (result.k3_provider, result.k3_transport_version) == ("unspecified", "unspecified")


def test_genuine_records_survive_an_instance_drain_override():
    k3, jev = _models()
    k3._transport.drain_call_records = lambda: ()
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "SHADOW_ACCEPT"
    assert sum(call.role != "JEV" for call in result.model_calls) == 7


def test_overrun_on_the_last_call_blocks_the_row():
    budget = CostBudget(50.0)
    k3, jev = provider_clients.build_shadow_models(
        FakeAnthropicClient(), FakeOpenAIClient(usage=(9_000_000, 400)), budget=budget,
        network_enabled=True, allow_test_clients=True)
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "BUDGET_OVERRUN" in result.gate_reasons
    assert result.model_calls[-1].budget_overrun and budget.locked


def test_locked_budget_refusal_is_marked_and_blocks():
    budget = CostBudget(50.0)
    budget.settle(budget.reserve(0.0), 0.01)
    result = ShadowRunner(*_models(budget)).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "BUDGET_OVERRUN" in result.gate_reasons
    assert result.model_calls[0].outcome == "BUDGET_EXCEEDED" and result.model_calls[0].budget_overrun


def test_records_carry_billable_input_tokens():
    transport = AnthropicK3Transport(FakeAnthropicClient(cache_tokens=(100, 0)), budget=CostBudget(5.0),
                                     allow_test_client=True)
    transport(_first_k3_request())
    (call,) = transport.drain_call_records()
    assert (call.input_tokens, call.billable_input_tokens) == (1300, 1400)
    with pytest.raises(ValueError):
        _record(input_tokens=10, billable_input_tokens=9)


def test_reservation_amount_is_read_only():
    reservation = CostBudget(1.0).reserve(0.5)
    with pytest.raises(AttributeError):
        reservation.amount = 0.0


# K5 runtime guard: however a stand-in holds or builds a real client, its call cannot reach a network.

class _ClassAttributeFake:
    client = None

    def with_options(self, **options):
        return self.client.with_options(**options)


class _GlobalFake:
    def with_options(self, **options):
        return _GLOBAL_CLIENT.with_options(**options)


class _PropertyFake:
    @property
    def client(self):
        return _GLOBAL_CLIENT

    def with_options(self, **options):
        return self.client.with_options(**options)


class _LazyFake:
    def with_options(self, **options):
        return anthropic.Anthropic(api_key="sk-ant-test", base_url="https://api.anthropic.com").with_options(**options)


class _SwallowingFake(FakeAnthropicClient):
    """Tries a socket, swallows the refusal, then answers like a normal fake."""

    def with_options(self, **options):
        import socket

        try:
            socket.create_connection(("api.anthropic.com", 443), timeout=1)
        except Exception:
            pass
        return super().with_options(**options)


def _hidden_stand_ins(real, monkeypatch):
    monkeypatch.setattr(_ClassAttributeFake, "client", real)
    monkeypatch.setattr(sys.modules[__name__], "_GLOBAL_CLIENT", real)
    holder = {"clients": [real]}
    queue = collections.deque([real])
    return [
        SimpleNamespace(with_options=lambda **options: real.with_options(**options)),
        SimpleNamespace(with_options=functools.partial(real.with_options)),
        SimpleNamespace(with_options=lambda **options: holder["clients"][0].with_options(**options)),
        SimpleNamespace(with_options=lambda **options: queue[0].with_options(**options)),
        _ClassAttributeFake(), _GlobalFake(), _PropertyFake(), _LazyFake(), _SwallowingFake(),
    ]


def test_hidden_or_lazily_built_clients_are_blocked_at_call_time(monkeypatch):
    for stand_in in _hidden_stand_ins(_real_anthropic(), monkeypatch):
        budget = CostBudget(5.0)
        transport = AnthropicK3Transport(stand_in, budget=budget, allow_test_client=True)
        with pytest.raises(ModelUnavailable):
            transport(_first_k3_request())
        (call,) = transport.drain_call_records()
        assert (call.outcome, call.detail) == ("CLIENT_UNTRUSTED", "network_blocked"), type(stand_in).__name__


def test_hidden_client_in_jev_stand_in_is_blocked_at_call_time():
    real = _real_openai()
    transport = OpenAIJevTransport(SimpleNamespace(with_options=lambda **options: real.with_options(**options)),
                                   budget=CostBudget(5.0), allow_test_client=True)
    request = {"state": {"context": {}}, "questions": {"model_id": "gpt-6.1-sol"}}
    with pytest.raises(ModelUnavailable):
        transport(request["state"], request["questions"])
    (call,) = transport.drain_call_records()
    assert (call.outcome, call.detail) == ("CLIENT_UNTRUSTED", "network_blocked")


def test_hidden_client_blocks_the_whole_row(monkeypatch):
    real = _real_anthropic()
    k3 = K3Client(AnthropicK3Transport(SimpleNamespace(with_options=lambda **options: real.with_options(**options)),
                                       budget=CostBudget(5.0), allow_test_client=True), network_enabled=True)
    _, jev = _models()
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and result.gate_reasons == ("K3_OUTPUTS_INVALID",)
    assert result.model_calls[0].outcome == "CLIENT_UNTRUSTED"


def test_guard_is_inert_outside_test_client_calls():
    install_offline_guard()
    import socket

    sock = socket.socket()  # socket.__new__ is allowed when no guarded call is running
    sock.close()


def test_guard_blocks_threads_and_subprocesses_inside_a_guarded_call():
    import subprocess
    import threading

    with offline_call(True) as state:
        with pytest.raises(OfflineNetworkBlocked):
            threading.Thread(target=lambda: None).start()
        with pytest.raises(OfflineNetworkBlocked):
            subprocess.run(["true"], check=False)
    assert state.blocked


def test_swapped_binding_is_refused_and_blocks():
    k3, jev = _models()
    cheap = dataclasses.replace(k3._transport.binding, input_usd_per_mtok=0.0001,
                                accepted_served_model_ids=("claude-fable-5-1", "claude-other"))
    k3._transport.binding = k3._transport._ledger.binding = cheap
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons


def test_fake_holding_sdk_response_types_is_accepted():
    canned = anthropic.types.Message.model_construct(id="msg_1", type="message", role="assistant",
                                                     content=[], model="claude-fable-5-1")
    transport = AnthropicK3Transport(SimpleNamespace(canned=canned, request=httpx2.Request("GET", "https://x.test")),
                                     budget=CostBudget(1.0), allow_test_client=True)
    assert transport.transport_version.endswith("-test-client")


def test_agent_shadow_tests_cannot_connect_sockets():
    import socket

    sock = socket.socket()
    try:
        with pytest.raises(OSError, match="network access is blocked"):
            sock.connect(("127.0.0.1", 7897))
        with pytest.raises(OSError, match="network access is blocked"):
            sock.connect_ex(("127.0.0.1", 7897))
    finally:
        sock.close()


# Review round 3: stand-in code runs only inside the guard; guard state handling; threat-model edges.

class _HashProbeFake(FakeAnthropicClient):
    """Trust checks must never run its __hash__/__eq__ (identity only)."""
    hashed = False

    def __hash__(self):
        type(self).hashed = True
        return 1

    def __eq__(self, other):
        type(self).hashed = True
        return self is other


class _ClassProbeFake(FakeAnthropicClient):
    """isinstance() consults __class__; that stand-in code must run inside the guard."""

    @property
    def __class__(self):
        import socket

        socket.socket().close()
        return FakeAnthropicClient


def test_trust_checks_never_run_stand_in_hash_or_eq():
    transport = AnthropicK3Transport(_HashProbeFake(), budget=CostBudget(5.0), allow_test_client=True)
    transport(_first_k3_request())
    assert transport.drain_call_records()[0].outcome == "OK"
    assert _HashProbeFake.hashed is False


def test_stand_in_code_run_by_checks_is_inside_the_guard():
    with pytest.raises(ValueError):
        AnthropicK3Transport(_ClassProbeFake(), budget=CostBudget(1.0), allow_test_client=True)


def test_stand_in_holding_a_factory_built_client_is_refused():
    trusted = _real_trusted_anthropic()
    with pytest.raises(ValueError):
        AnthropicK3Transport(SimpleNamespace(spare=trusted), budget=CostBudget(1.0), allow_test_client=True)


def test_offline_call_nests_and_restores_state_after_errors():
    import socket

    with offline_call(True) as outer:
        with offline_call(False):
            pass
        with pytest.raises(RuntimeError):
            with offline_call(True):
                raise RuntimeError("inner failure")
        with pytest.raises(OfflineNetworkBlocked):
            socket.socket()
    assert outer.blocked
    socket.socket().close()  # guard inactive again after the outer call


def test_other_threads_are_outside_the_guard_by_design():
    import socket
    import threading

    created = []
    worker = threading.Thread(target=lambda: created.append(socket.socket()))
    with offline_call(True) as state:
        pass
    worker.start()
    worker.join()
    created[0].close()
    assert not state.blocked  # documented limit: only the calling thread is guarded


def test_real_client_calls_are_unaffected_after_the_guard_is_installed(monkeypatch):
    install_offline_guard()
    real = _real_trusted_anthropic()
    monkeypatch.setattr(real, "with_options", lambda **options: FakeAnthropicClient())
    transport = AnthropicK3Transport(real, budget=CostBudget(5.0))
    transport(_first_k3_request())
    (call,) = transport.drain_call_records()
    assert call.outcome == "OK" and call.test_client is False


def test_ctypes_calls_are_blocked_inside_a_guarded_call():
    import ctypes

    with offline_call(True) as state:
        with pytest.raises(OfflineNetworkBlocked):
            ctypes.CDLL(None)
    assert state.blocked


def test_blocked_attempt_then_base_exception_is_recorded_as_untrusted():
    class _Interrupting(FakeAnthropicClient):
        def with_options(self, **options):
            import socket

            try:
                socket.socket()
            except Exception:
                pass
            raise KeyboardInterrupt

    transport = AnthropicK3Transport(_Interrupting(), budget=CostBudget(5.0), allow_test_client=True)
    with pytest.raises(KeyboardInterrupt):
        transport(_first_k3_request())
    (call,) = transport.drain_call_records()
    assert (call.outcome, call.detail) == ("CLIENT_UNTRUSTED", "network_blocked")


# Final review fixes: outputs must come through the registered path; bindings are exact.

def _real_k3(monkeypatch):
    real = _real_trusted_anthropic()
    monkeypatch.setattr(real, "with_options", lambda **options: FakeAnthropicClient())
    return K3Client(AnthropicK3Transport(real, budget=CostBudget(5.0)), network_enabled=True)


def test_instance_analyze_override_is_ignored(monkeypatch):
    k3 = _real_k3(monkeypatch)
    forged = FakeReasoner(responses={"RESEARCH_MANAGER": {"direction": "DOWN", "confidence": 0.99,
                                                          "supporting_roles": [], "contradicting_roles": [],
                                                          "key_risks": []}})
    k3.analyze = forged.analyze
    _, jev = _models()
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert not forged.requests  # the override never ran
    assert sum(call.role != "JEV" for call in result.model_calls) == 7


def test_instance_interpret_override_is_ignored(monkeypatch):
    k3 = _real_k3(monkeypatch)
    k3._transport._interpret = lambda *args: {"forged": True}
    _, jev = _models()
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert sum(call.role != "JEV" for call in result.model_calls) == 7
    assert result.gate_status != "BLOCKED" or "MODEL_PROVENANCE_INVALID" not in result.gate_reasons


def test_calls_without_records_block_the_row(monkeypatch):
    k3 = _real_k3(monkeypatch)
    _, jev = _models()
    runner = ShadowRunner(k3, jev)
    original = AnthropicK3Transport.__call__

    def unrecorded(self, request):
        data = original(self, request)
        self._ledger._records.clear()
        return data

    monkeypatch.setattr(AnthropicK3Transport, "__call__", unrecorded)
    result = runner.run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons


def test_binding_subclass_is_rejected():
    from std0_quant.research.agent_shadow.model_registry import ModelBinding, binding_for

    class FreeBinding(ModelBinding):
        def call_cost_usd(self, input_tokens, output_tokens):
            return 0.0

        def max_call_cost_usd(self, input_bytes):
            return 0.0

    free = FreeBinding(**{field.name: getattr(binding_for("K3"), field.name)
                          for field in dataclasses.fields(ModelBinding)})
    with pytest.raises(ValueError):
        AnthropicK3Transport(FakeAnthropicClient(), budget=CostBudget(1.0), binding=free, allow_test_client=True)
    k3, jev = _models()
    k3._transport.binding = k3._transport._ledger.binding = free
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons


def test_swapped_transport_on_enabled_client_is_refused_and_blocks():
    k3, jev = _models()
    calls = []
    k3._transport = lambda request: calls.append(request) or FakeReasoner().analyze(request)
    with pytest.raises(ModelUnavailable):
        k3.analyze(_first_k3_request())
    result = ShadowRunner(k3, jev).run(pit_row(), EVIDENCE)
    assert not calls
    assert result.gate_status == "BLOCKED" and "MODEL_PROVENANCE_INVALID" in result.gate_reasons


def test_model_id_read_once_at_construction():
    class Flaky(FakeReasoner):
        def __init__(self):
            super().__init__()
            self.reads = 0

        @property
        def model_id(self):
            self.reads += 1
            if self.reads > 1:
                raise RuntimeError("model_id unavailable")
            return "kimi-k3"

    _, jev = _models()
    result = ShadowRunner(Flaky(), jev).run(pit_row(), EVIDENCE)
    assert result.k3_model_id == "kimi-k3"
