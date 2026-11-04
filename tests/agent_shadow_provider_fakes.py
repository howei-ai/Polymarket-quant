"""Offline stand-ins for the openai 3.x Responses SDK client; no network."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

from agent_shadow_fakes import FakeReasoner, judge_response


class FakeOpenAIClient:
    """Mimics client.with_options(...).responses.create(...)."""

    def __init__(self, *, reply=None, error=None, status="completed", incomplete_reason=None,
                 model="gpt-6.1-sol", usage=(3000, 400), output=None, request_id="req_openai_1",
                 replies=None, cache_tokens=(0, 0)):
        self.calls = []
        self.closed = False
        self.options = []
        self.reply = reply
        self.replies = replies or {}
        self.cache_tokens = cache_tokens
        self.error = error
        self.status = status
        self.incomplete_reason = incomplete_reason
        self.model = model
        self.usage = usage
        self.output = output
        self.request_id = request_id
        self.responses = SimpleNamespace(create=self._create)

    def with_options(self, **options):
        self.options.append(options)
        return self

    def close(self):
        self.closed = True

    def _create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.error is not None:
            raise self.error
        request = json.loads(kwargs["input"])
        if "role" in request:
            text = self.replies.get(request["role"], self.reply)
            if text is None:
                text = json.dumps(FakeReasoner().analyze(request))
        else:
            text = self.reply if self.reply is not None else json.dumps(judge_response())
        output = self.output if self.output is not None else [
            SimpleNamespace(type="reasoning", summary=[]),
            SimpleNamespace(type="message", role="assistant",
                            content=[SimpleNamespace(type="output_text", text=text)]),
        ]
        incomplete = SimpleNamespace(reason=self.incomplete_reason) if self.incomplete_reason else None
        usage = None if self.usage is None else SimpleNamespace(
            input_tokens=self.usage[0], output_tokens=self.usage[1])
        if usage is not None and self.cache_tokens is not None:
            usage.input_tokens_details = SimpleNamespace(
                cache_write_tokens=self.cache_tokens[0], cached_tokens=self.cache_tokens[1])
        return SimpleNamespace(model=self.model, status=self.status, incomplete_details=incomplete,
                               output=output, usage=usage, _request_id=self.request_id)


class FakeOpenAIK3Client(FakeOpenAIClient):
    """Responses stand-in for all seven GPT-6 Astra research calls."""

    def __init__(self, **kwargs):
        defaults = dict(model="gpt-6-astra", usage=(1200, 300), request_id="req_k3_1")
        defaults.update(kwargs)
        super().__init__(**defaults)


def message_with(*parts):
    return SimpleNamespace(type="message", role="assistant", content=list(parts))


def output_text(text=None):
    return SimpleNamespace(type="output_text", text=json.dumps(judge_response()) if text is None else text)


def refusal(text="I can't help with that."):
    return SimpleNamespace(type="refusal", refusal=text)
