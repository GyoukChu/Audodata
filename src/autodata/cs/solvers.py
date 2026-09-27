"""Synchronous OpenAI client with shared request/retry policy and final-answer solver calls.

Evaluation owns its retry budget: one initial call plus ``eval.solver_retries``
retries. The SDK's automatic retries are disabled to avoid multiplying it.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import openai

from autodata.config import ModelEndpoint
from autodata.llm.client import backoff, build_request_kwargs, retryable as _retryable

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts" / "cs"


@dataclass
class ChatReply:
    content: str
    reasoning: str | None
    finish_reason: str | None
    usage: dict[str, Any]
    latency_s: float


class SyncClient:
    """Thin synchronous client; one connection pool shared between attempts."""

    def __init__(self, endpoint: ModelEndpoint, *, transport: httpx.BaseTransport | None = None):
        self.endpoint = endpoint
        self._client = openai.OpenAI(
            base_url=endpoint.base_url, api_key=endpoint.api_key,
            timeout=endpoint.timeout_s, max_retries=0,
            http_client=httpx.Client(transport=transport),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SyncClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def request_seed(self) -> int | None:
        """Effective wire seed, including the existing extra_body override."""
        request = build_request_kwargs(self.endpoint)
        return request.get("extra_body", {}).get("seed", request.get("seed"))

    def chat(self, messages: list[dict[str, str]], *, timeout: float | None = None,
             chat_template_kwargs: dict[str, Any] | None = None,
             seed: int | None = None) -> ChatReply:
        """`chat_template_kwargs` (keyword-only, optional) overrides individual template options for this call
        only, e.g. {"reasoning_effort": "low"} for the judge's truncation fallback."""
        endpoint = self.endpoint
        request = build_request_kwargs(endpoint, chat_template_kwargs=chat_template_kwargs, seed=seed)
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            messages=messages, timeout=endpoint.timeout_s if timeout is None else timeout, **request,
        )
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        reasoning = (
            getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        )
        return ChatReply(
            content=(message.content or "") if message else "",
            reasoning=reasoning, finish_reason=choice.finish_reason if choice else None,
            usage=response.usage.model_dump(exclude_none=True) if response.usage else {},
            latency_s=time.perf_counter() - started,
        )


def retryable(error: Exception) -> bool:
    """Compatibility export retaining the original `error` keyword."""
    return _retryable(error)


def error_text(error: Exception) -> str:
    """Keep API bodies/credentials out of reports, including authentication errors."""
    if isinstance(error, openai.APIStatusError):
        return f"API request failed (HTTP {error.status_code})"
    if isinstance(error, openai.APIConnectionError):
        return f"API request failed ({type(error).__name__})"
    return f"invalid API response ({type(error).__name__})"


def retry_backoff(retry_index: int) -> None:
    time.sleep(backoff(retry_index))


def total_usage(requests: list[dict[str, Any]]) -> dict[str, int]:
    """Include all returned token counts, including empty/invalid responses retried."""
    return {
        key: sum(request.get("usage", {}).get(key, 0) or 0 for request in requests)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def build_solver_messages(
    context: str, question: str, template_path: str | Path = PROMPTS_DIR / "solver_user.md",
) -> list[dict[str, str]]:
    template = Path(template_path).read_text(encoding="utf-8")
    return [{"role": "user", "content": template.format(context=context, question=question)}]


def strip_reasoning(content: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def remove(match: re.Match[str]) -> str:
        blocks.append(match.group(1).strip())
        return ""

    # Also remove an unclosed block when generation was truncated during thinking.
    content = re.sub(r"<think\b[^>]*>(.*?)(?:</think\s*>|\Z)", remove, content,
                     flags=re.DOTALL | re.IGNORECASE)
    return content.strip(), blocks


@dataclass
class SolverAttempt:
    response_text: str
    reasoning: str | None
    finish_reason: str | None
    usage: dict[str, int]
    latency_s: float
    error: str | None
    messages: list[dict[str, str]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    inline_reasoning: list[str] = field(default_factory=list)


def run_solver(
    client: SyncClient, context: str, question: str,
    template_path: str | Path = PROMPTS_DIR / "solver_user.md", *,
    retries: int = 3, timeout: float | None = None,
) -> SolverAttempt:
    if retries < 0:
        raise ValueError("retries must be nonnegative")
    messages = build_solver_messages(context, question, template_path)
    requests: list[dict[str, Any]] = []
    started = time.perf_counter()
    answer, reasoning, finish, inline = "", None, None, []
    error: str | None = None
    for attempt in range(retries + 1):
        request_started = time.perf_counter()
        try:
            reply = client.chat(messages, timeout=timeout)
            record = asdict(reply)
            record["seed"] = client.request_seed
            answer, inline = strip_reasoning(reply.content)
            reasoning = reply.reasoning
            finish = reply.finish_reason
            # A response cut off by max_tokens (finish_reason == "length") with no final answer is a legitimate outcome
            # for thinking models: the paper grades truncated answers as-is (a truncated/absent answer scores 0),
            # so it is NOT retried. Only an empty response that stopped normally is treated as a transient error.
            truncated_no_answer = (not answer) and finish == "length"
            error = None if (answer or truncated_no_answer) else "empty response"
            record["error"] = error
            record["truncated_no_answer"] = truncated_no_answer
            requests.append(record)
            if error is None:
                break
            can_retry = True
        except (openai.OpenAIError, httpx.TransportError, ValueError, TypeError, AttributeError) as exc:
            error = error_text(exc)
            requests.append({"error": error, "usage": {}, "seed": client.request_seed,
                             "latency_s": time.perf_counter() - request_started})
            can_retry = retryable(exc)
        if attempt == retries or not can_retry:
            break
        retry_backoff(attempt)
    return SolverAttempt(
        answer, reasoning, finish, total_usage(requests), time.perf_counter() - started,
        error, messages, requests, inline,
    )
