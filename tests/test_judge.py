from __future__ import annotations

import json

import httpx
import pytest

from autodata.config import ModelEndpoint
from autodata.cs import solvers
from autodata.cs.judge import JudgeError, build_judge_messages, parse_judgments, run_judge
from autodata.cs.rubric import parse_rubric
from autodata.cs.solvers import PROMPTS_DIR, SyncClient
from tests.fake_openai_server import ScriptedResponder, make_transport, text_response

RUBRIC = parse_rubric([{"criterion": "Insight", "weight": 8}, {"criterion": "Mistake", "weight": -5}])
VALID = {"criteria": [{"index": 1, "satisfied": True, "evidence": "insight"},
                      {"index": 2, "satisfied": False, "evidence": ""}]}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(solvers, "retry_backoff", lambda _: None)


def test_exact_judge_messages():
    messages = build_judge_messages("context", "question", RUBRIC, "answer")
    assert messages == [
        {"role": "system", "content": (PROMPTS_DIR / "judge.md").read_text()},
        {"role": "user", "content": "## Context\ncontext\n\n## Question\nquestion\n\n"
         "## Rubric\n1. [+8 positive] Insight\n2. [-5 negative] Mistake\n\n## Response\nanswer"},
    ]


@pytest.mark.parametrize("prefix,suffix", [("", ""), ("```json\n", "\n```"),
                                            ("Assessment follows:\n", "\nDone."),
                                            ("preamble {not JSON}\n", "")])
def test_json_fences_preamble_and_order(prefix, suffix):
    raw = prefix + json.dumps({"criteria": list(reversed(VALID["criteria"]))}) + suffix
    assert parse_judgments(raw, 2) == ([True, False], ["insight", ""])


@pytest.mark.parametrize("criteria", [
    None, {}, [], VALID["criteria"][:1], VALID["criteria"] * 2,
    [VALID["criteria"][0]] * 2,
    [{"index": 0, "satisfied": True, "evidence": "x"}, VALID["criteria"][1]],
    [{"index": 3, "satisfied": True, "evidence": "x"}, VALID["criteria"][1]],
    [{"index": "1", "satisfied": True, "evidence": "x"}, VALID["criteria"][1]],
    [{"index": True, "satisfied": True, "evidence": "x"}, VALID["criteria"][1]],
    [{"index": 1, "satisfied": 1, "evidence": "x"}, VALID["criteria"][1]],
    [{"index": 1, "satisfied": "false", "evidence": "x"}, VALID["criteria"][1]],
    [{"index": 1, "satisfied": True}, VALID["criteria"][1]],
    [{"index": 1, "satisfied": True, "evidence": None}, VALID["criteria"][1]],
    [None, VALID["criteria"][1]],
])
def test_strict_criterion_validation(criteria):
    with pytest.raises(JudgeError):
        parse_judgments(json.dumps({"criteria": criteria}), 2)


def test_only_first_json_object_is_graded():
    with pytest.raises(JudgeError):
        parse_judgments('{}\n' + json.dumps(VALID), 2)


def test_invalid_judgments_retry_identical_messages_and_track_usage():
    responder = ScriptedResponder([text_response("not JSON"), text_response(json.dumps(VALID))])
    with SyncClient(ModelEndpoint(base_url="http://fake/v1", model="judge"),
                    transport=make_transport(responder)) as client:
        result = run_judge(client, "context", "question", RUBRIC, "answer", retries=1)
    assert result.satisfied == [True, False]
    assert result.evidence == ["insight", ""]
    assert result.raw == json.dumps(VALID)
    assert result.usage == {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    assert result.latency_s >= 0
    assert len(result.requests) == 2
    assert responder.requests[0] == responder.requests[1]
    assert responder.requests[0]["messages"] == build_judge_messages("context", "question", RUBRIC, "answer")


def test_judge_error_after_initial_call_plus_configured_retries():
    responder = ScriptedResponder([text_response('{"criteria": []}')])
    with SyncClient(ModelEndpoint(base_url="http://fake/v1", model="judge", max_retries=20),
                    transport=make_transport(responder)) as client:
        with pytest.raises(JudgeError) as caught:
            run_judge(client, "c", "q", RUBRIC, "a", retries=2)
    assert len(responder.requests) == 3
    assert len(caught.value.requests) == 3
    assert caught.value.messages == build_judge_messages("c", "q", RUBRIC, "a")


@pytest.mark.parametrize("failure", [408, 409, 429, 500, 503, 599, "connection", "timeout"])
def test_judge_transient_request_retry(failure):
    calls = []

    def responder(body, index):
        calls.append(body)
        if index == 0:
            if failure == "connection":
                raise httpx.ConnectError("offline")
            if failure == "timeout":
                raise httpx.ReadTimeout("slow")
            return failure, {"error": {"message": "temporary"}}
        return text_response(json.dumps(VALID))

    with SyncClient(ModelEndpoint(base_url="http://fake/v1", model="judge"),
                    transport=make_transport(responder)) as client:
        result = run_judge(client, "c", "q", RUBRIC, "a", retries=1)
    assert len(calls) == 2
    assert result.satisfied == [True, False]


def test_permanent_api_errors_are_not_retried_or_logged_verbatim():
    calls = []

    def responder(body, index):
        calls.append(body)
        return 401, {"error": {"message": "do not log my-secret-key"}}

    with SyncClient(ModelEndpoint(base_url="http://fake/v1", model="judge", api_key="my-secret-key"),
                    transport=make_transport(responder)) as client:
        with pytest.raises(JudgeError) as caught:
            run_judge(client, "c", "q", RUBRIC, "a", retries=3)
    assert len(calls) == 1
    assert "HTTP 401" in str(caught.value)
    assert "my-secret-key" not in str(caught.value) + str(caught.value.requests)


@pytest.mark.parametrize("failure", ["invalid_json", 503, 400])
def test_judge_records_request_seeds_on_success_retry_and_error(failure):
    first = text_response("invalid JSON") if failure == "invalid_json" else (
        failure, {"error": {"message": "request failed"}}
    )
    responder = ScriptedResponder([first, text_response(json.dumps(VALID))])
    endpoint = ModelEndpoint(base_url="http://fake/v1", model="judge", extra_body={"seed": 0})
    with SyncClient(endpoint, transport=make_transport(responder)) as client:
        if failure == 400:
            with pytest.raises(JudgeError) as caught:
                run_judge(client, "c", "q", RUBRIC, "a", retries=2)
            requests = caught.value.requests
            assert len(requests) == 1
        else:
            requests = run_judge(client, "c", "q", RUBRIC, "a", retries=1).requests
            assert len(requests) == 2
    assert all(body["seed"] == 0 for body in responder.requests)
    assert all(request["seed"] == 0 for request in requests)


def test_truncation_fallback_lowers_reasoning_effort():
    """A judge reply cut off by max_tokens (finish_reason=length, no JSON) retries at a lower reasoning effort."""
    from autodata.config import ModelEndpoint
    from autodata.cs.judge import lower_effort, run_judge
    from autodata.cs.rubric import parse_rubric
    from autodata.cs.solvers import SyncClient
    from tests.fake_openai_server import make_transport, text_response

    assert lower_effort("max") == "high" and lower_effort("high") == "low" and lower_effort("low") == "low"
    assert lower_effort(None) == "high" and lower_effort("weird") == "high"
    rubric = parse_rubric([{"criterion": "a", "weight": 3, "category": "positive"},
                           {"criterion": "b", "weight": -2, "category": "negative"}])
    seen: list[dict] = []

    def responder(req: dict, idx: int) -> dict:
        seen.append(req)
        if idx == 0:
            return text_response("", reasoning="x" * 100, finish_reason="length")
        return text_response('{"criteria": [{"index": 1, "satisfied": true, "evidence": "q"}, {"index": 2, "satisfied": false, "evidence": ""}]}')

    ep = ModelEndpoint(base_url="http://fake/v1", model="glm-5.3", max_tokens=64, timeout_s=5,
                       chat_template_kwargs={"reasoning_effort": "max"})
    with SyncClient(ep, transport=make_transport(responder)) as client:
        res = run_judge(client, "ctx", "q?", rubric, "resp", retries=3)
    assert res.satisfied == [True, False]
    assert len(seen) == 2
    assert seen[0]["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "max" if "extra_body" in seen[0] else seen[0]["chat_template_kwargs"]["reasoning_effort"] == "max"
    second = seen[1].get("chat_template_kwargs") or seen[1].get("extra_body", {}).get("chat_template_kwargs")
    assert second["reasoning_effort"] == "high"
    assert res.requests[0]["reasoning_effort_override"] is None and res.requests[1]["reasoning_effort_override"] == "high"


@pytest.mark.parametrize("fallback", [True, False])
def test_effort_fallback_is_optional_and_uses_effective_template(fallback):
    endpoint = ModelEndpoint(base_url="http://fake/v1", model="judge",
                             chat_template_kwargs={"reasoning_effort": "max"},
                             extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}})
    responder = ScriptedResponder([text_response("", finish_reason="length"), text_response(json.dumps(VALID))])
    with SyncClient(endpoint, transport=make_transport(responder)) as client:
        result = run_judge(client, "c", "q", RUBRIC, "answer", retries=1, effort_fallback=fallback)
    assert result.satisfied == [True, False]
    assert [body["chat_template_kwargs"]["reasoning_effort"] for body in responder.requests] == [
        "high", "low" if fallback else "high",
    ]
    assert result.usage["prompt_tokens"] == 20
