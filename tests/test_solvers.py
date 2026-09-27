from copy import deepcopy

import httpx
import pytest

from autodata.config import ModelEndpoint
from autodata.cs import solvers
from autodata.cs.solvers import SyncClient, run_solver
from autodata.llm.client import LLMClient, backoff, build_request_kwargs, retryable
from tests.fake_openai_server import ScriptedResponder, make_transport, text_response


@pytest.mark.parametrize("failure", [408, 409, 429, 500, 599, "connection", "timeout"])
def test_solver_shared_transport_retries(failure, monkeypatch):
    monkeypatch.setattr(solvers, "retry_backoff", lambda _: None)

    calls = []

    def response(body, index):
        calls.append(body)
        if index == 0:
            if failure == "connection":
                raise httpx.ConnectError("offline")
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout")
            return failure, {}
        return text_response("answer")

    with SyncClient(ModelEndpoint(base_url="http://fake/v1", model="m"),
                    transport=make_transport(response)) as client:
        result = run_solver(client, "c", "q", retries=1)
    assert result.error is None and result.response_text == "answer"
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_solver_permanent_error_does_not_retry(status):
    responder = ScriptedResponder([(status, {})])
    with SyncClient(ModelEndpoint(base_url="http://fake/v1", model="m"),
                    transport=make_transport(responder)) as client:
        result = run_solver(client, "c", "q", retries=3)
    assert result.error and len(responder.requests) == 1


@pytest.mark.parametrize("content", ["", " ", "<think>unfinished"])
def test_truncated_solver_never_retries(content):
    responder = ScriptedResponder([text_response(content, finish_reason="length")])
    with SyncClient(ModelEndpoint(base_url="http://fake/v1", model="m"),
                    transport=make_transport(responder)) as client:
        result = run_solver(client, "c", "q", retries=3)
    assert result.error is None and result.response_text == ""
    assert len(responder.requests) == 1 and result.usage["prompt_tokens"] == 10


async def test_sync_and_async_share_sampling_requests_and_seed_precedence():
    endpoint = ModelEndpoint(
        base_url="http://fake/v1", model="m", seed=13, max_tokens=42, temperature=0,
        top_p=0.8, top_k=20, min_p=0.1, presence_penalty=0.2, repetition_penalty=1.1,
        chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "max"},
        extra_body={"seed": 7, "top_k": 30, "chat_template_kwargs": {"reasoning_effort": "high"}},
    )
    before = deepcopy(endpoint.model_dump())
    responder = ScriptedResponder([text_response("ok")])
    async_client = LLMClient(endpoint, transport=make_transport(responder))
    messages = [{"role": "user", "content": "question"}]
    await async_client.chat(messages)
    with SyncClient(endpoint, transport=make_transport(responder)) as client:
        sdk_client = client._client
        assert client.request_seed == 7
        client.chat(messages)
        client.chat(messages, seed=0, chat_template_kwargs={"reasoning_effort": "low"})
        assert client._client is sdk_client
    assert responder.requests[0] == responder.requests[1]
    assert responder.requests[2]["seed"] == 0
    assert responder.requests[2]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
    kwargs = build_request_kwargs(endpoint, seed=0, chat_template_kwargs={"reasoning_effort": "low"})
    kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] = False
    assert endpoint.model_dump() == before


def test_backoff_and_retry_policy_are_shared(monkeypatch):
    error = httpx.ConnectError("offline")
    assert solvers.retryable(error=error) is retryable(error) is True
    monkeypatch.setattr(solvers, "backoff", lambda _: 1.234)
    delays = []
    monkeypatch.setattr(solvers.time, "sleep", delays.append)
    solvers.retry_backoff(0)
    assert delays == [1.234]
    assert 1 <= backoff(0) <= 2 and 2 <= backoff(1) <= 3 and backoff(100) == 60
