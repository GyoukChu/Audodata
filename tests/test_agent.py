from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from autodata.config import ModelEndpoint
from autodata.harness.agent import Agent
from autodata.harness.tools import Tool, Workspace, make_read_tool, make_task_tool, make_write_tool
from autodata.llm.client import LLMClient
from tests.fake_openai_server import ScriptedResponder, make_transport, text_response, tool_call_response


def client_for(responses, **endpoint_options):
    responder = ScriptedResponder(responses)
    client = LLMClient(
        ModelEndpoint(base_url="http://fake.invalid/v1", model="glm-test", max_retries=0, **endpoint_options),
        transport=make_transport(responder),
    )
    return client, responder


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


async def test_multistep_sequential_tools_final_and_event_hooks(tmp_path):
    ws = Workspace(tmp_path)
    responses = [
        tool_call_response([
            ("write", {"filePath": "output.txt", "content": "draft"}),
            ("read", {"filePath": "output.txt"}),
        ], reasoning="Write and inspect."),
        tool_call_response([("task", {
            "subagent_type": "challenger", "description": "improve", "prompt": "Check draft.",
        })]),
        text_response("Final answer", reasoning="Done."),
    ]
    client, responder = client_for(responses)
    events = []
    runner = AsyncMock(return_value="challenger answer")
    path = tmp_path / "trajectory/main.jsonl"
    agent = Agent(
        name="main", system_prompt="system", llm=client, max_steps=5,
        tools=[make_read_tool(ws), make_write_tool(ws), make_task_tool(runner, ["challenger"])],
        transcript_path=path, event_hook=events.append,
    )
    result = await agent.run("task prompt")
    assert (result.final_text, result.stop_reason, result.steps_used) == ("Final answer", "final", 3)
    assert result.error is None and result.transcript_path == path
    assert result.usage["prompt_tokens"] == 30 and result.usage["completion_tokens"] == 15
    assert result.usage["calls"] == 3
    second_history = responder.requests[1]["messages"]
    assert [message["role"] for message in second_history] == ["system", "user", "assistant", "tool", "tool"]
    assert second_history[2]["content"] == "" and second_history[2]["reasoning_content"] == "Write and inspect."
    assert second_history[-1]["content"] == "draft"
    calls = second_history[2]["tool_calls"]
    assert second_history[3]["tool_call_id"] == calls[0]["id"]
    assert second_history[4]["tool_call_id"] == calls[1]["id"]
    runner.assert_awaited_once_with("challenger", "improve", "Check draft.")
    tool_events = [event for event in events if event.kind in ("tool_call", "tool_result")]
    assert [event.kind for event in tool_events] == ["tool_call", "tool_result"] * 3
    assert [event.data["name"] for event in tool_events] == ["write", "write", "read", "read", "task", "task"]
    assert tool_events[0].data["arguments"] == {"filePath": "output.txt", "content": "draft"}
    assert tool_events[3].data["result"] == "draft"
    assert all(event.ts > 0 for event in events)
    transcript = records(path)
    assert len([row for row in transcript if row["type"] == "event"]) == len(events)
    messages = [row["message"] for row in transcript if row["type"] == "message"]
    assert len(messages) == 8 and messages[-1]["content"] == "Final answer"


async def test_transcript_is_written_before_waiting_tool_finishes(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(arguments):
        entered.set()
        await release.wait()
        return "tool done"

    client, _ = client_for([tool_call_response([("wait", {})]), text_response("done")])
    path = tmp_path / "trajectory.jsonl"
    events = []

    def hook(event):
        # A hook can already inspect the event just delivered to it on disk.
        assert records(path)[-1]["kind"] == event.kind
        events.append(event)

    agent = Agent(name="main", system_prompt="system", llm=client, max_steps=3,
                  tools=[Tool("wait", "wait", {"type": "object"}, handler)],
                  transcript_path=path, event_hook=hook)
    running = asyncio.create_task(agent.run("task"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not running.done()
        partial = records(path)
        assert [row["message"]["role"] for row in partial if row["type"] == "message"] == [
            "system", "user", "assistant",
        ]
        assert partial[-1]["kind"] == "tool_call"
        assert not any(event.kind == "tool_result" for event in events)
    finally:
        release.set()
        await running
    assert records(path)[-1]["kind"] == "final"


async def test_max_steps_uses_last_assistant_content():
    handler = AsyncMock(return_value="tool result")
    client, responder = client_for([tool_call_response([("again", {})], content="still working")])
    result = await Agent(
        name="main", system_prompt="system", llm=client, max_steps=2,
        tools=[Tool("again", "again", {}, handler)],
    ).run("task")
    assert result.stop_reason == "max_steps" and result.steps_used == 2
    assert result.final_text == "still working" and result.transcript_path is None
    assert len(responder.requests) == 2 and handler.await_count == 2


async def test_zero_steps_does_not_call_llm():
    client, responder = client_for([text_response("unused")])
    result = await Agent(name="main", system_prompt="system", llm=client, max_steps=0, tools=[]).run("task")
    assert (result.stop_reason, result.steps_used, result.final_text) == ("max_steps", 0, "")
    assert responder.requests == []


async def test_length_stops_without_tools():
    client, responder = client_for([text_response("partial", finish_reason="length"), text_response("unused")])
    result = await Agent(name="main", system_prompt="system", llm=client, max_steps=5, tools=[]).run("task")
    assert (result.stop_reason, result.steps_used, result.final_text) == ("length", 1, "partial")
    assert len(responder.requests) == 1


async def test_length_with_tools_continues():
    response = tool_call_response([("tool", {})])
    response["choices"][0]["finish_reason"] = "length"
    client, _ = client_for([response, text_response("final")])
    handler = AsyncMock(return_value="output")
    result = await Agent(name="main", system_prompt="system", llm=client, max_steps=2,
                         tools=[Tool("tool", "tool", {}, handler)]).run("task")
    assert result.stop_reason == "final" and result.steps_used == 2
    handler.assert_awaited_once()


async def test_handler_exception_and_unknown_tool_become_tool_messages():
    client, responder = client_for([
        tool_call_response([("broken", {}), ("missing", {})]), text_response("recovered"),
    ])
    handler = AsyncMock(side_effect=ValueError("bad arguments"))
    events = []
    result = await Agent(name="main", system_prompt="system", llm=client, max_steps=3,
                         tools=[Tool("broken", "broken", {}, handler)], event_hook=events.append).run("task")
    assert result.final_text == "recovered" and result.stop_reason == "final"
    outputs = [message["content"] for message in responder.requests[1]["messages"] if message["role"] == "tool"]
    assert outputs[0] == "Error: ValueError: bad arguments"
    assert "unknown tool" in outputs[1] and "missing" in outputs[1]
    assert [event.data["result"] for event in events if event.kind == "tool_result"] == outputs


async def test_tool_result_truncation_matches_history_event_and_transcript(tmp_path):
    client, responder = client_for([tool_call_response([("paper", {})]), text_response("done")])
    handler = AsyncMock(return_value="p" * 150_000)
    path = tmp_path / "transcript.jsonl"
    events = []
    await Agent(name="main", system_prompt="system", llm=client, max_steps=2,
                tools=[Tool("paper", "paper", {}, handler)], tool_result_max_chars=1000,
                transcript_path=path, event_hook=events.append).run("task")
    expected = "p" * 1000 + "\n[truncated 149000 chars]"
    assert responder.requests[1]["messages"][-1]["content"] == expected
    assert next(event.data["result"] for event in events if event.kind == "tool_result") == expected
    assert next(row["message"]["content"] for row in records(path)
                if row["type"] == "message" and row["message"]["role"] == "tool") == expected


async def test_llm_failure_returns_error_and_keeps_transcript(tmp_path):
    client, _ = client_for([(400, {"error": {"message": "bad request"}})])
    path = tmp_path / "transcript.jsonl"
    result = await Agent(name="main", system_prompt="system", llm=client, max_steps=3,
                         tools=[], transcript_path=path).run("task")
    assert result.stop_reason == "error" and result.steps_used == 1
    assert "BadRequestError" in result.error
    assert result.usage["errors"] == 1
    assert records(path)[-1]["kind"] == "error"


@pytest.mark.parametrize("keep_recent", [None, 0, 2, 20])
@pytest.mark.parametrize("budget", [None, 1, 4500])
async def test_context_elision_preserves_sequence_and_recent_results(tmp_path, keep_recent, budget):
    responses = [
        tool_call_response([("read", {"index": i}) for i in range(8)],
                           content="assistant content", reasoning="assistant reasoning"),
        tool_call_response([("read", {"index": i}) for i in range(8, 10)]),
        text_response("done"),
    ]
    client, responder = client_for(responses)
    path = tmp_path / "transcript.jsonl"
    events = []

    async def read(arguments):
        return str(arguments["index"]) * 1000

    options = {} if keep_recent is None else {"keep_recent_tool_results": keep_recent}
    result = await Agent(
        name="main", system_prompt="s" * 200, llm=client, max_steps=3,
        tools=[Tool("read", "read", {}, read)], transcript_path=path, event_hook=events.append,
        context_budget_chars=budget, **options,
    ).run("u" * 300)
    assert result.stop_reason == "final"
    keep = 6 if keep_recent is None else keep_recent
    transcript_messages = [row["message"] for row in records(path) if row["type"] == "message"]
    elision_events = [event for event in events if event.kind == "elide"]
    assert len({event.data["message_index"] for event in elision_events}) == len(elision_events)
    assert [event.data["message_index"] for event in elision_events] == sorted(
        event.data["message_index"] for event in elision_events
    )
    for request in responder.requests:
        history = request["messages"]
        pending = []
        tool_messages = [message for message in history if message["role"] == "tool"]
        for index, message in enumerate(history):
            original = transcript_messages[index]
            if message["role"] == "tool":
                assert message["tool_call_id"] == pending.pop(0)
                assert {k: v for k, v in message.items() if k != "content"} == {
                    k: v for k, v in original.items() if k != "content"
                }
                assert len(original["content"]) == 1000  # Full results stay on disk.
            else:
                assert not pending
                if message["role"] == "assistant" and "reasoning" not in message:
                    assert message == {key: value for key, value in original.items()
                                       if key not in ("reasoning", "reasoning_content")}
                else:
                    assert message == original
                pending = [call["id"] for call in message.get("tool_calls", [])]
        assert not pending
        contents = [message["content"] for message in tool_messages]
        for i, content in enumerate(contents):
            if budget is None or i >= len(contents) - keep:
                assert content == str(i) * 1000
            elif content != str(i) * 1000:
                assert content == (
                    "[tool result elided by the harness to fit the context window: "
                    "1000 chars; the full text is in the transcript]"
                )
        if budget is not None and sum(len(message["content"]) for message in history) > budget:
            assert all(content.startswith("[tool result elided")
                       for content in contents[:max(0, len(contents) - keep)])
    if budget is None or keep >= 10:
        assert not elision_events
    else:
        assert elision_events
        assert all(event.data["original_chars"] == 1000 for event in elision_events
                   if "tool_call_id" in event.data)
    assert len([row for row in records(path) if row.get("kind") == "elide"]) == len(elision_events)


async def test_context_400_stops_immediately_despite_retry_and_step_budgets(tmp_path, monkeypatch):
    def no_retry(_):
        pytest.fail("context errors must not retry")

    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(no_retry))
    responder = ScriptedResponder([(400, {"error": {"message": "maximum context length exceeded"}})])
    client = LLMClient(ModelEndpoint(base_url="http://fake/v1", model="main", max_retries=20),
                       transport=make_transport(responder))
    path = tmp_path / "transcript.jsonl"
    result = await Agent(name="main", system_prompt="system", tools=[], llm=client,
                         max_steps=50, transcript_path=path).run("task")
    assert result.stop_reason == "error" and result.steps_used == 1
    assert "maximum context length exceeded" in result.error
    assert len(responder.requests) == 1 and client.usage_totals["calls"] == 1
    assert records(path)[-1]["data"]["error"] == result.error


async def test_context_elides_short_results_only_once_when_budget_cannot_fit():
    client, responder = client_for([
        tool_call_response([("tool", {})]), tool_call_response([("tool", {})]), text_response("done"),
    ])
    events = []
    await Agent(name="main", system_prompt="system", llm=client, max_steps=3,
                tools=[Tool("tool", "tool", {}, AsyncMock(return_value="tiny"))],
                context_budget_chars=0, keep_recent_tool_results=0, event_hook=events.append).run("task")
    elisions = [event for event in events if event.kind == "elide"]
    assert len(elisions) == 2 and [event.data["original_chars"] for event in elisions] == [4, 4]
    assert all("4 chars;" in message["content"] for message in responder.requests[-1]["messages"]
               if message["role"] == "tool")


async def test_usage_is_per_run_when_client_is_shared():
    client, _ = client_for([text_response("first"), text_response("second")])
    agent = Agent(name="main", system_prompt="system", llm=client, max_steps=3, tools=[])
    first = await agent.run("task")
    second = await agent.run("other task")
    assert first.usage == second.usage
    assert second.usage["prompt_tokens"] == 10
    assert client.usage_totals["prompt_tokens"] == 20


async def test_subagent_can_use_same_client_during_main_tool_call():
    client, _ = client_for([
        tool_call_response([("task", {"subagent_type": "challenger", "description": "check", "prompt": "paper"})]),
        text_response("subagent answer"), text_response("main answer"),
    ], max_concurrency=1)

    async def runner(subagent_type, description, prompt):
        result = await Agent(name=subagent_type, system_prompt=description, llm=client,
                             max_steps=2, tools=[]).run(prompt)
        return result.final_text

    result = await asyncio.wait_for(Agent(
        name="main", system_prompt="system", llm=client, max_steps=3,
        tools=[make_task_tool(runner, ["challenger"])],
    ).run("task"), timeout=5)
    assert result.final_text == "main answer" and result.steps_used == 2


async def test_agent_spend_includes_discarded_completions(monkeypatch):
    responder = ScriptedResponder([text_response(""), text_response("answer")])
    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda _: 0))
    client = LLMClient(ModelEndpoint(base_url="http://fake/v1", model="m", max_retries=1),
                       transport=make_transport(responder))
    result = await Agent(name="main", system_prompt="s", tools=[], llm=client, max_steps=1).run("u")
    assert result.usage["prompt_tokens"] == 20 and result.usage["completion_tokens"] == 10
    assert result.stop_reason == "final"


async def test_empty_truncation_produces_length_agent_result():
    client, responder = client_for([text_response("", reasoning="thinking", finish_reason="length")])
    result = await Agent(name="main", system_prompt="s", tools=[], llm=client, max_steps=3).run("u")
    assert result.stop_reason == "length" and result.error is None
    assert result.final_text == "" and len(responder.requests) == 1


def test_context_counts_both_reasoning_variants_once_and_json_arguments():
    from autodata.harness.agent import _message_chars

    message = {"role": "assistant", "content": "answer", "reasoning": "thought",
               "reasoning_content": "thought", "tool_calls": [{"function": {"arguments": {"q": "雪"}}}]}
    assert _message_chars(message) == len("answerthought") + len(json.dumps({"q": "雪"}, ensure_ascii=False))
    message["reasoning_content"] = "different"
    assert _message_chars(message) == len("answerthoughtdifferent") + len(json.dumps({"q": "雪"}, ensure_ascii=False))
    assert _message_chars({"role": "tool", "content": "tool text"}) == 9


async def test_old_reasoning_elided_before_tool_results_and_recent_turn_kept(tmp_path):
    responses = [tool_call_response([("read", {})], reasoning="r" * 4000),
                 tool_call_response([("read", {})], reasoning="recent"), text_response("done")]
    client, responder = client_for(responses)
    events = []
    path = tmp_path / "transcript.jsonl"
    result = await Agent(name="main", system_prompt="s", llm=client, max_steps=3,
                         tools=[Tool("read", "read", {}, AsyncMock(return_value="t" * 300))],
                         context_budget_chars=1000, keep_recent_tool_results=1,
                         event_hook=events.append, transcript_path=path).run("u")
    assert result.stop_reason == "final"
    history = responder.requests[-1]["messages"]
    old, recent = [message for message in history if message["role"] == "assistant"]
    assert "reasoning" not in old and "reasoning_content" not in old
    assert recent["reasoning"] == recent["reasoning_content"] == "recent"
    assert all(message["content"] == "t" * 300 for message in history if message["role"] == "tool")
    elisions = [event for event in events if event.kind == "elide"]
    assert len(elisions) == 1 and elisions[0].data["original_chars"] == 4000
    assert elisions[0].data["fields"] == ["reasoning", "reasoning_content"]
    assert any(row.get("message", {}).get("reasoning") == "r" * 4000 for row in records(path))


async def test_tool_arguments_trigger_context_elision():
    client, responder = client_for([tool_call_response([("read", {"argument": "x" * 2000})]), text_response("done")])
    await Agent(name="main", system_prompt="s", llm=client, max_steps=2,
                tools=[Tool("read", "read", {}, AsyncMock(return_value="t" * 500))],
                context_budget_chars=1000, keep_recent_tool_results=0).run("u")
    history = responder.requests[-1]["messages"]
    assert history[-1]["content"].startswith("[tool result elided")
    assert json.loads(history[-2]["tool_calls"][0]["function"]["arguments"])["argument"] == "x" * 2000


@pytest.mark.parametrize("prompt_tokens,tool_chars,budget,should_elide", [
    (900, 500, 800, True),  # Server-measured prompt overhead exceeds the character estimate.
    (100, 1600, 400, True),  # Newly appended text must be added to prompt_tokens.
    (100, 500, 400, False),
])
async def test_token_budget_uses_prompt_usage_and_new_text(prompt_tokens, tool_chars, budget, should_elide):
    first = tool_call_response([("read", {})])
    first["usage"]["prompt_tokens"] = prompt_tokens
    client, responder = client_for([first, text_response("done")])
    events = []
    await Agent(name="main", system_prompt="s", llm=client, max_steps=2,
                tools=[Tool("read", "read", {}, AsyncMock(return_value="t" * tool_chars))],
                context_budget_chars=0, context_budget_tokens=budget, keep_recent_tool_results=0,
                event_hook=events.append).run("u")
    elisions = [event for event in events if event.kind == "elide"]
    assert bool(elisions) == should_elide
    assert responder.requests[-1]["messages"][-1]["content"].startswith("[tool result elided") == should_elide
    if elisions:
        assert elisions[-1].data["context_budget_tokens"] == budget
        replacement = responder.requests[-1]["messages"][-1]["content"]
        assert elisions[-1].data["context_tokens"] == prompt_tokens + (len(replacement) + 2) / 4


def test_negative_token_budget_is_rejected():
    client, _ = client_for([text_response("unused")])
    with pytest.raises(ValueError, match="context_budget_tokens"):
        Agent(name="main", system_prompt="s", llm=client, max_steps=1, tools=[], context_budget_tokens=-1)


async def test_agent_reports_spend_when_empty_retries_are_exhausted(monkeypatch):
    responder = ScriptedResponder([text_response("")])
    monkeypatch.setattr(LLMClient, "_backoff", staticmethod(lambda _: 0))
    client = LLMClient(ModelEndpoint(base_url="http://fake/v1", model="m", max_retries=1),
                       transport=make_transport(responder))
    result = await Agent(name="main", system_prompt="s", tools=[], llm=client, max_steps=1).run("u")
    assert result.stop_reason == "error"
    assert result.usage["prompt_tokens"] == client.usage_totals["prompt_tokens"] == 20
    assert result.usage["completion_tokens"] == client.usage_totals["completion_tokens"] == 10
