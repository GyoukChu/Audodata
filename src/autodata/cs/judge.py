"""Rubric-only judging; the reference answer never enters the judge messages."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai

from autodata.cs import solvers
from autodata.cs.rubric import RubricItem
from autodata.cs.solvers import PROMPTS_DIR, SyncClient
from autodata.llm.client import build_request_kwargs, retryable


class JudgeError(ValueError):
    def __init__(self, message: str, *, messages: list[dict[str, str]] | None = None,
                 requests: list[dict[str, Any]] | None = None, latency_s: float = 0.0):
        super().__init__(message)
        self.messages = messages or []
        self.requests = requests or []
        self.latency_s = latency_s


@dataclass
class JudgeResult:
    satisfied: list[bool]
    evidence: list[str]
    raw: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    messages: list[dict[str, str]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)


def build_judge_messages(
    context: str, question: str, rubric: list[RubricItem], response: str,
    system_prompt_path: str | Path = PROMPTS_DIR / "judge.md",
) -> list[dict[str, str]]:
    lines = "\n".join(
        f"{i}. [{item.weight:+d} {item.category}] {item.criterion}"
        for i, item in enumerate(rubric, 1)
    )
    return [
        {"role": "system", "content": Path(system_prompt_path).read_text(encoding="utf-8")},
        {"role": "user", "content": (
            f"## Context\n{context}\n\n## Question\n{question}\n\n"
            f"## Rubric\n{lines}\n\n## Response\n{response}"
        )},
    ]


from autodata.cs.parsing import repair_json_escapes  # noqa: E402  (shared lenient JSON repair)


def _first_json_object(raw: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def parse_judgments(raw: str, n_criteria: int) -> tuple[list[bool], list[str]]:
    obj = _first_json_object(raw or "")
    if obj is None and raw:
        obj = _first_json_object(repair_json_escapes(raw))  # lenient pass: invalid escapes from LaTeX quotes
    if obj is None:
        raise JudgeError("judge output contains no JSON object")
    criteria = obj.get("criteria")
    if not isinstance(criteria, list) or len(criteria) != n_criteria:
        raise JudgeError(f"judge criteria must contain exactly {n_criteria} entries")
    entries: dict[int, tuple[bool, str]] = {}
    for entry in criteria:
        if not isinstance(entry, dict):
            raise JudgeError("each judge criterion must be an object")
        index = entry.get("index")
        if type(index) is not int or not 1 <= index <= n_criteria or index in entries:
            raise JudgeError(f"judge indexes must contain each of 1..{n_criteria} exactly once")
        satisfied, evidence = entry.get("satisfied"), entry.get("evidence")
        if type(satisfied) is not bool or not isinstance(evidence, str):
            raise JudgeError("judge entries require boolean satisfied and string evidence")
        entries[index] = satisfied, evidence
    return ([entries[i][0] for i in range(1, n_criteria + 1)],
            [entries[i][1] for i in range(1, n_criteria + 1)])


_EFFORT_LADDER = ("max", "high", "low")


def lower_effort(current: str | None) -> str:
    """Next lower GLM reasoning effort (max -> high -> low -> low)."""
    cur = (current or "max").lower()
    if cur not in _EFFORT_LADDER:
        cur = "max"
    idx = _EFFORT_LADDER.index(cur)
    return _EFFORT_LADDER[min(idx + 1, len(_EFFORT_LADDER) - 1)]


def _endpoint_effort(client: SyncClient) -> str | None:
    endpoint = getattr(client, "endpoint", None)
    kwargs = (build_request_kwargs(endpoint).get("extra_body", {}).get("chat_template_kwargs") or {}
              if endpoint is not None else {})
    return kwargs.get("reasoning_effort")


def run_judge(
    client: SyncClient, context: str, question: str, rubric: list[RubricItem], response: str,
    system_prompt_path: str | Path = PROMPTS_DIR / "judge.md", *, retries: int = 3,
    effort_fallback: bool = True,
) -> JudgeResult:
    if retries < 0:
        raise ValueError("retries must be nonnegative")
    messages = build_judge_messages(context, question, rubric, response, system_prompt_path)
    requests: list[dict[str, Any]] = []
    started = time.perf_counter()
    error = "judge failed"
    # Truncation fallback: when the judge exhausts max_tokens while thinking (finish_reason == "length", no verdict),
    # retrying at the same reasoning effort is futile (observed 3x32k-token retries at effort=max); step the effort
    # down for the retry instead (max -> high -> low). Only triggers on truncation, so other configs are untouched.
    effort_override: str | None = None
    for attempt in range(retries + 1):
        request_started = time.perf_counter()
        try:
            reply = client.chat(messages, chat_template_kwargs={"reasoning_effort": effort_override} if effort_override else None)
            record = asdict(reply)
            record["seed"] = client.request_seed
            record["reasoning_effort_override"] = effort_override
            requests.append(record)
            try:
                satisfied, evidence = parse_judgments(reply.content, len(rubric))
            except JudgeError as exc:
                error = str(exc)
                record["error"] = error
                can_retry = True
                if effort_fallback and reply.finish_reason == "length":
                    effort_override = lower_effort(effort_override or _endpoint_effort(client))
            else:
                return JudgeResult(satisfied, evidence, reply.content,
                                   solvers.total_usage(requests), time.perf_counter() - started,
                                   messages, requests)
        except (openai.OpenAIError, httpx.TransportError, ValueError, TypeError, AttributeError) as exc:
            error = solvers.error_text(exc)
            requests.append({"error": error, "usage": {}, "seed": client.request_seed,
                             "latency_s": time.perf_counter() - request_started})
            can_retry = retryable(exc)
        if attempt == retries or not can_retry:
            break
        solvers.retry_backoff(attempt)
    raise JudgeError(error, messages=messages, requests=requests,
                     latency_s=time.perf_counter() - started)
