"""Offline stand-ins for the anthropic 2.x and openai 3.x SDK clients; no network."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

from agent_shadow_fakes import FakeReasoner, judge_response


class FakeAnthropicClient:
    """Mimics client.with_options(...).messages.create(...)."""

    def __init__(self, *, replies=None, error=None, stop_reason="end_turn", stop_category=None,
                 model="claude-fable-5-1", usage=(1200, 300), cache_tokens=None, content=None,
                 request_id="req_anthropic_1"):
        self.calls = []
        self.options = []
        self.replies = replies or {}
        self.error = error
        self.stop_reason = stop_reason
        self.stop_category = stop_category
        self.model = model
        self.usage = usage
        self.cache_tokens = cache_tokens
        self.content = content
        self.request_id = request_id
        self.messages = SimpleNamespace(create=self._create)

    def with_options(self, **options):
        self.options.append(options)
        return self

    def _create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.error is not None:
            raise self.error
        request = json.loads(kwargs["messages"][0]["content"])
        role = request["role"]
        text = self.replies.get(role)
        if text is None:
            text = json.dumps(FakeReasoner().analyze(request))
        content = self.content if self.content is not None else [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ]
        stop_details = None
        if self.stop_reason == "refusal":
            stop_details = SimpleNamespace(type="refusal", category=self.stop_category, explanation=None)
        usage = None if self.usage is None else SimpleNamespace(
            input_tokens=self.usage[0], output_tokens=self.usage[1])
        if usage is not None and self.cache_tokens is not None:
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens = self.cache_tokens
        return SimpleNamespace(model=self.model, stop_reason=self.stop_reason, stop_details=stop_details,
                               content=content, usage=usage, _request_id=self.request_id)


class FakeOpenAIClient:
    """Mimics client.with_options(...).responses.create(...)."""

    def __init__(self, *, reply=None, error=None, status="completed", incomplete_reason=None,
                 model="gpt-6.1-sol", usage=(3000, 400), output=None, request_id="req_openai_1"):
        self.calls = []
        self.options = []
        self.reply = reply
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

    def _create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.error is not None:
            raise self.error
        text = self.reply if self.reply is not None else json.dumps(judge_response())
        output = self.output if self.output is not None else [
            SimpleNamespace(type="reasoning", summary=[]),
            SimpleNamespace(type="message", role="assistant",
                            content=[SimpleNamespace(type="output_text", text=text)]),
        ]
        incomplete = SimpleNamespace(reason=self.incomplete_reason) if self.incomplete_reason else None
        usage = None if self.usage is None else SimpleNamespace(
            input_tokens=self.usage[0], output_tokens=self.usage[1])
        return SimpleNamespace(model=self.model, status=self.status, incomplete_details=incomplete,
                               output=output, usage=usage, _request_id=self.request_id)


def message_with(*parts):
    return SimpleNamespace(type="message", role="assistant", content=list(parts))


def output_text(text=None):
    return SimpleNamespace(type="output_text", text=json.dumps(judge_response()) if text is None else text)


def refusal(text="I can't help with that."):
    return SimpleNamespace(type="refusal", refusal=text)
