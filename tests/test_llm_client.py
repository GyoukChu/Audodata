from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading
import time

import httpx
import openai
import pytest

from autodata.config import ModelEndpoint
from autodata.llm.client import ChatResult, LLMClient, ToolCall
from tests.fake_openai_server import (
    ScriptedResponder, make_transport, text_response, tool_call_response,
)


def endpoint(**overrides) -> ModelEndpoint:
    return ModelEndpoint(base_url="http://fake.invalid/v1", model="glm-test", **overrides)


async def test_sampling_and_extra_body_merge(monkeypatch):
    responder = ScriptedResponder([text_response("answer")])
    config = endpoint(
        temperature=0.8, top_p=0.9, top_k=20, min_p=0.1,
        presence_penalty=0.2, repetition_penalty=1.1, max_tokens=123,
        chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"},
        extra_body={"top_k": 30, "chat_template_kwargs": {"reasoning_effort": "medium"}, "seed": 4},
    )
    original = deepcopy(config.model_dump())
    call_extras = {"top_k": 40, "chat_template_kwargs": {"reasoning_effort": "high"}}
    client = LLMClient(config, transport=make_transport(responder))
    sdk_requests = []
    create = openai.resources.chat.completions.AsyncCompletions.create

    async def spy(self, **kwargs):
        sdk_requests.append(kwargs)
        return await create(self, **kwargs)

    monkeypatch.setattr(openai.resources.chat.completions.AsyncCompletions, "create", spy)
    tool = {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
    result = await client.chat(
        [{"role": "user", "content": "hello"}], tools=[tool], tool_choice="auto",
        max_tokens=42, temperature=0, response_format={"type": "json_object"}, extra_body=call_extras,
    )
    sdk = sdk_requests[0]
    assert sdk["extra_body"] == {
        "top_k": 40, "min_p": 0.1, "repetition_penalty": 1.1, "seed": 4,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"},
    }
    assert "top_k" not in sdk and "chat_template_kwargs" not in sdk
    wire = responder.requests[0]
    assert wire["max_tokens"] == 42 and wire["temperature"] == 0
    assert wire["top_p"] == 0.9 and wire["presence_penalty"] == 0.2
    assert wire["tools"] == [tool] and wire["tool_choice"] == "auto"
    assert wire["response_format"] == {"type": "json_object"}
    assert wire["top_k"] == 40 and wire["chat_template_kwargs"]["enable_thinking"]
    assert result.content == "answer" and result.latency_s >= 0
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert config.model_dump() == original
    assert call_extras == {"top_k": 40, "chat_template_kwargs": {"reasoning_effort": "high"}}
    await client.chat([])
    assert responder.requests[1]["temperature"] == 0.8
    assert responder.requests[1]["max_tokens"] == 123
    assert responder.requests[1]["top_k"] == 30


@pytest.mark.parametrize("arguments", ['{"filePath": "paper.txt"}', '{bad json', '[1, 2]', 'null', ''])
async def test_tool_arguments_and_assistant_round_trip(arguments):
    response = tool_call_response([("read", {})], reasoning="Read the paper first.")
    response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    responder = ScriptedResponder([response, text_response("finished")])
    client = LLMClient(endpoint(), transport=make_transport(responder))
    result = await client.chat([])
    call = result.tool_calls[0]
    assert call.arguments == ({"filePath": "paper.txt"} if arguments.startswith('{"') else {})
    assert call.raw_arguments == arguments
    assert call.name == "read" and call.id
    history = client.assistant_message(result)
    assert history == {
        "role": "assistant", "content": "", "reasoning": "Read the paper first.",
        "reasoning_content": "Read the paper first.",
        "tool_calls": [{"id": call.id, "type": "function", "function": {
            "name": "read", "arguments": arguments or "{}",
        }}],
    }
    await client.chat([history, {"role": "tool", "tool_call_id": call.id, "content": "paper"}])
    assert responder.requests[1]["messages"][0] == history


@pytest.mark.parametrize("reasoning_content,reasoning,expected", [
    ("first", "second", "first"), (None, "second", "second"), ("", "second", "second"),
    (None, None, None), ("", "", ""),
])
async def test_reasoning_variants(reasoning_content, reasoning, expected):
    response = text_response("answer")
    response["choices"][0]["message"].update(reasoning_content=reasoning_content, reasoning=reasoning)
    result = await LLMClient(endpoint(), transport=make_transport(lambda *_: response)).chat([])
    assert result.reasoning == expected
    message = LLMClient.assistant_message(result)
    assert message["role"] == "assistant" and message["content"] == "answer"
    # both field names are sent: GLM's template reads `reasoning`, Qwen's reads `reasoning_content`
    assert ("reasoning_content" in message) == bool(expected)
    assert ("reasoning" in message) == bool(expected)
    if expected:
        assert message["reasoning"] == message["reasoning_content"] == expected
    assert "tool_calls" not in message


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 599])
async def test_retryable_status_then_success(status, monkeypatch, capsys):
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("autodata.llm.client.asyncio.sleep", sleep)
    responder = ScriptedResponder([(status, {"error": {"message": "try again"}}), text_response("ok")])
    client = LLMClient(endpoint(max_retries=1), name="weak-solver", transport=make_transport(responder))
    assert (await client.chat([])).content == "ok"
    assert len(responder.requests) == 2 and len(sleeps) == 1
    assert 1 <= sleeps[0] <= 2
    assert client.usage_totals == {"calls": 2, "errors": 1, "prompt_tokens": 10, "completion_tokens": 5}
    stderr = capsys.readouterr().err
    assert "weak-solver" in stderr and str(status) in stderr


async def test_429_then_500_then_success_uses_exponential_delays(monkeypatch):
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("autodata.llm.client.asyncio.sleep", sleep)
    responder = ScriptedResponder([(429, {}), (500, {}), text_response("ok")])
    client = LLMClient(endpoint(max_retries=2), transport=make_transport(responder))
    assert (await client.chat([])).content == "ok"
    assert 1 <= sleeps[0] <= 2 and 2 <= sleeps[1] <= 3
    assert 0 < client._backoff(100) <= 60


@pytest.mark.parametrize("exception", [httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout])
async def test_connection_errors_are_retried(exception, monkeypatch):
    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda _: 0))

    def respond(body, index):
        if index == 0:
            raise exception("offline")
        return text_response("ok")

    client = LLMClient(endpoint(max_retries=1), transport=make_transport(respond))
    assert (await client.chat([])).content == "ok"
    assert client.usage_totals["calls"] == 2


@pytest.mark.parametrize("status,retries,expected_calls", [(400, 3, 1), (401, 3, 1), (422, 3, 1), (429, 2, 3), (500, 0, 1)])
async def test_nonretryable_status_and_retry_exhaustion(status, retries, expected_calls, monkeypatch):
    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda _: 0))
    responder = ScriptedResponder([(status, {"error": {"message": "failure"}})])
    client = LLMClient(endpoint(max_retries=retries), transport=make_transport(responder))
    with pytest.raises(openai.APIStatusError):
        await client.chat([])
    assert len(responder.requests) == expected_calls
    assert client.usage_totals["errors"] == expected_calls


@pytest.mark.parametrize("seed", [None, 0, 12345])
async def test_chat_seed_is_optional_and_forwarded(seed):
    responder = ScriptedResponder([text_response("ok")])
    client = LLMClient(endpoint(), transport=make_transport(responder))
    await client.chat([], seed=seed)
    assert responder.requests[0].get("seed") == seed
    assert ("seed" in responder.requests[0]) == (seed is not None)


def test_chat_sync_explicit_seed_overrides_extras_without_mutating_endpoint():
    responder = ScriptedResponder([text_response("ok")])
    config = endpoint(extra_body={"seed": 42})
    client = LLMClient(config, transport=make_transport(responder))
    client.chat_sync([], seed=0, extra_body={"seed": 17})
    client.chat_sync([])
    assert [request["seed"] for request in responder.requests] == [0, 42]
    assert config.extra_body == {"seed": 42}


@pytest.mark.parametrize("empty", [None, "", "  \n"])
async def test_empty_response_retries_even_with_reasoning(empty, monkeypatch):
    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda _: 0))
    responder = ScriptedResponder([text_response(empty, reasoning="thinking"), text_response("answer")])
    client = LLMClient(endpoint(max_retries=1), transport=make_transport(responder))
    assert (await client.chat([])).content == "answer"
    assert client.usage_totals == {"calls": 2, "errors": 1, "prompt_tokens": 20, "completion_tokens": 10}


async def test_empty_choices_and_exhaustion(monkeypatch):
    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda _: 0))
    response = text_response("")
    response["choices"] = []
    responder = ScriptedResponder([response])
    with pytest.raises(RuntimeError, match="neither content nor tool_calls"):
        await LLMClient(endpoint(max_retries=1), transport=make_transport(responder)).chat([])
    assert len(responder.requests) == 2


async def test_logs_response_without_api_key(tmp_path):
    client = LLMClient(
        endpoint(api_key="secret-test-key"), name="main/agent", log_dir=tmp_path,
        transport=make_transport(lambda *_: text_response("answer")),
    )
    await client.chat([])
    logfile = next(tmp_path.glob("*.jsonl"))
    record = json.loads(logfile.read_text())
    assert record["request"]["model"] == "glm-test"
    assert record["response"]["choices"][0]["message"]["content"] == "answer"
    assert "secret-test-key" not in logfile.read_text()


def test_chat_sync_thread_safety_and_concurrency():
    lock = threading.Lock()
    active = peak = 0

    def respond(body, index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return text_response(body["messages"][0]["content"])

    client = LLMClient(endpoint(max_concurrency=2), transport=make_transport(respond))
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(
            lambda i: client.chat_sync([{"role": "user", "content": str(i)}]).content, range(12),
        ))
    assert results == [str(i) for i in range(12)]
    assert peak == 2
    assert client.usage_totals == {"calls": 12, "errors": 0, "prompt_tokens": 120, "completion_tokens": 60}
    assert client.chat_sync([{"role": "user", "content": "another loop"}]).content == "another loop"


async def test_async_concurrency_and_cancellation_release_slots():
    active = peak = 0
    entered = asyncio.Event()

    async def respond(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        try:
            await asyncio.sleep(0.02)
            return httpx.Response(200, json=text_response("ok"))
        finally:
            active -= 1

    client = LLMClient(endpoint(max_concurrency=2), transport=httpx.MockTransport(respond))
    cancelled = asyncio.create_task(client.chat([]))
    await entered.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    results = await asyncio.wait_for(asyncio.gather(*(client.chat([]) for _ in range(5))), timeout=5)
    assert all(result.content == "ok" for result in results)
    assert peak == 2


def test_assistant_message_serializes_arguments_if_raw_is_absent():
    result = ChatResult(None, "", [ToolCall("c1", "read", {"filePath": "paper.txt"}, "")],
                        "tool_calls", {}, {}, 0)
    message = LLMClient.assistant_message(result)
    assert message["content"] == "" and "reasoning_content" not in message
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"filePath": "paper.txt"}


@pytest.mark.parametrize("content", [None, "", "  "])
async def test_empty_truncation_is_returned_without_retry(content):
    responder = ScriptedResponder([text_response(content, reasoning="thinking", finish_reason="length")])
    client = LLMClient(endpoint(max_retries=5), transport=make_transport(responder))
    result = await client.chat([])
    assert result.finish_reason == "length" and result.content == content
    assert len(responder.requests) == 1
    assert client.usage_totals == {"calls": 1, "errors": 0, "prompt_tokens": 10, "completion_tokens": 5}


async def test_per_call_usage_includes_all_discarded_completions(monkeypatch):
    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda _: 0))
    responder = ScriptedResponder([text_response(""), (503, {}), text_response(""), text_response("answer")])
    client = LLMClient(endpoint(max_retries=3), transport=make_transport(responder))
    result = await client.chat([])
    assert result.usage == {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}
    assert result.raw["usage"]["prompt_tokens"] == 10
    assert client.usage_totals == {"calls": 4, "errors": 3, "prompt_tokens": 30, "completion_tokens": 15}
    assert (await client.chat([])).usage["prompt_tokens"] == 10


async def test_endpoint_seed_defaults_and_explicit_call_override():
    responder = ScriptedResponder([text_response("ok")])
    client = LLMClient(endpoint(seed=42), transport=make_transport(responder))
    await client.chat([])
    await client.chat([], seed=0)
    assert [body["seed"] for body in responder.requests] == [42, 0]
