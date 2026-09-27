"""OpenAI-compatible chat calls, including vLLM sampling and reasoning fields."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any, AsyncIterator
from weakref import WeakKeyDictionary

import httpx
import openai

from autodata.config import ModelEndpoint


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str


@dataclass
class ChatResult:
    content: str | None
    reasoning: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: dict[str, int]
    raw: dict[str, Any]
    latency_s: float
    # Usage of the returned request, separate from aggregate retry accounting.
    last_request_usage: dict[str, int] = field(default_factory=dict)


class _EmptyResponseError(RuntimeError):
    """An empty, non-truncated completion is retryable."""


def retryable(exc: Exception) -> bool:
    """Shared transport/status policy; malformed requests are never retried."""
    if isinstance(exc, (_EmptyResponseError, httpx.TransportError,
                        openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return isinstance(exc, openai.APIStatusError) and (
        exc.status_code in (408, 409, 429) or 500 <= exc.status_code <= 599
    )


def backoff(attempt: int) -> float:
    """Exponential retry delay in seconds, capped at 60s with jitter."""
    return min(60.0, 2.0 ** min(attempt, 6) + random.uniform(0.0, 1.0))


def _merge_extra_body(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if key == "chat_template_kwargs" and isinstance(value, dict):
            previous = merged.get(key)
            merged[key] = {**(previous if isinstance(previous, dict) else {}), **deepcopy(value)}
        else:
            merged[key] = deepcopy(value)
    return merged


def build_request_kwargs(
    endpoint: ModelEndpoint, *, chat_template_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build sampling kwargs without mutating the endpoint or creating a client.

    Explicit call seeds win; otherwise legacy extra_body seeds override endpoint
    defaults. Sampling reproducibility remains best-effort on the model server.
    """
    request = {"model": endpoint.model, "max_tokens": endpoint.max_tokens}
    for key in ("temperature", "top_p", "presence_penalty", "seed"):
        value = getattr(endpoint, key, None)
        if value is not None:
            request[key] = value
    extra = {key: value for key in ("top_k", "min_p", "repetition_penalty")
             if (value := getattr(endpoint, key)) is not None}
    if endpoint.chat_template_kwargs:
        extra["chat_template_kwargs"] = deepcopy(endpoint.chat_template_kwargs)
    extra = _merge_extra_body(extra, endpoint.extra_body)
    if chat_template_kwargs:
        extra = _merge_extra_body(extra, {"chat_template_kwargs": chat_template_kwargs})
    if seed is not None:
        request["seed"] = seed
        extra.pop("seed", None)
    if extra:
        request["extra_body"] = extra
    return request


class LLMClient:
    def __init__(
        self, endpoint: ModelEndpoint, *, name: str = "", log_dir: Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if endpoint.max_concurrency < 1 or endpoint.max_retries < 0:
            raise ValueError("max_concurrency must be positive and max_retries nonnegative")
        self.endpoint = endpoint
        self.name = name or endpoint.model
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._transport = transport
        self.usage_totals: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "errors": 0,
        }
        self._lock = threading.Lock()
        self._semaphores: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            WeakKeyDictionary()
        )
        self._permits = threading.BoundedSemaphore(endpoint.max_concurrency)

    @asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        # asyncio primitives cannot be shared by chat_sync's private event loops.
        loop = asyncio.get_running_loop()
        with self._lock:
            semaphore = self._semaphores.setdefault(
                loop, asyncio.Semaphore(self.endpoint.max_concurrency)
            )
        async with semaphore:
            # The thread-safe pool also bounds concurrent calls across those loops.
            # Nonblocking acquisition avoids stranding a permit on cancellation.
            while not self._permits.acquire(blocking=False):
                await asyncio.sleep(0.01)
            try:
                yield
            finally:
                self._permits.release()

    def _request(
        self, messages: list[dict], *, tools: list[dict] | None,
        tool_choice: str | None, max_tokens: int | None, temperature: float | None,
        response_format: dict | None, extra_body: dict | None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        request = build_request_kwargs(self.endpoint, seed=seed)
        request["messages"] = messages
        overrides = {"tools": tools, "tool_choice": tool_choice, "max_tokens": max_tokens,
                     "temperature": temperature, "response_format": response_format}
        request.update({key: value for key, value in overrides.items() if value is not None})
        merged = _merge_extra_body(request.get("extra_body", {}), extra_body or {})
        if seed is not None:
            merged.pop("seed", None)
        if merged:
            request["extra_body"] = merged
        return request

    @staticmethod
    def _parse(raw: dict[str, Any], latency_s: float) -> ChatResult:
        choices = raw.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw_arguments = function.get("arguments") or ""
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, ValueError):
                arguments = {}
            calls.append(ToolCall(
                id=call.get("id", ""), name=function.get("name", ""),
                arguments=arguments if isinstance(arguments, dict) else {},
                raw_arguments=raw_arguments,
            ))
        usage = {
            key: value for key, value in (raw.get("usage") or {}).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        return ChatResult(
            content=message.get("content"),
            reasoning=message.get("reasoning_content") or message.get("reasoning"),
            tool_calls=calls, finish_reason=choice.get("finish_reason"),
            usage=usage, raw=raw, latency_s=latency_s, last_request_usage=dict(usage),
        )

    # Compatibility aliases for existing callers and retry-delay test hooks.
    _retryable = staticmethod(retryable)
    _backoff = staticmethod(backoff)

    def _log(self, record: dict[str, Any]) -> None:
        if self.log_dir is None:
            return
        # Request metadata deliberately excludes credentials, URLs and arbitrary
        # extra_body values. Never serialize the ModelEndpoint or HTTP headers.
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", self.name) or "llm"
        with self._lock:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / f"{filename}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None,
        tool_choice: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None, response_format: dict | None = None,
        extra_body: dict | None = None,
        seed: int | None = None,
    ) -> ChatResult:
        request = self._request(
            messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens,
            temperature=temperature, response_format=response_format, extra_body=extra_body,
            seed=seed,
        )
        request_meta = {
            "model": self.endpoint.model, "message_count": len(messages),
            "max_tokens": request["max_tokens"], "temperature": request.get("temperature"),
        }
        async with self._slot():
            # Each call owns its connections, so neither HTTP connections nor SDK
            # clients leak across the private loops used by worker threads.
            async with openai.AsyncOpenAI(
                base_url=self.endpoint.base_url, api_key=self.endpoint.api_key,
                timeout=self.endpoint.timeout_s, max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=self._transport, timeout=self.endpoint.timeout_s,
                ),
            ) as client:
                started = time.perf_counter()
                call_usage: dict[str, int] = {}
                for attempt in range(self.endpoint.max_retries + 1):
                    with self._lock:
                        self.usage_totals["calls"] += 1
                    raw: dict[str, Any] | None = None
                    try:
                        response = await client.chat.completions.create(**request)
                        raw = response.model_dump(mode="json")
                        result = self._parse(raw, time.perf_counter() - started)
                        for key, value in result.usage.items():
                            call_usage[key] = call_usage.get(key, 0) + value
                        with self._lock:
                            for key in ("prompt_tokens", "completion_tokens"):
                                self.usage_totals[key] += result.usage.get(key, 0)
                        if (not (result.content or "").strip() and not result.tool_calls
                                and result.finish_reason != "length"):
                            raise _EmptyResponseError("response has neither content nor tool_calls")
                    except Exception as exc:
                        with self._lock:
                            self.usage_totals["errors"] += 1
                        label = type(exc).__name__
                        if isinstance(exc, openai.APIStatusError):
                            label += f" (HTTP {exc.status_code})"
                        self._log({
                            "ts": time.time(), "request": request_meta, "attempt": attempt + 1,
                            "error": label, "response": raw,
                        })
                        retry = self._retryable(exc) and attempt < self.endpoint.max_retries
                        delay = self._backoff(attempt) if retry else 0.0
                        suffix = f"; retrying in {delay:.2f}s" if retry else "; giving up"
                        print(f"[{self.name}] {label}{suffix}", file=sys.stderr)
                        if not retry:
                            # Preserve spend when no ChatResult can be returned.
                            exc._autodata_usage = dict(call_usage)
                            raise
                        await asyncio.sleep(delay)
                    else:
                        self._log({
                            "ts": time.time(), "request": request_meta, "attempt": attempt + 1,
                            "response": raw, "latency_s": result.latency_s,
                        })
                        result.usage = dict(call_usage)
                        return result
        raise AssertionError("retry loop exited without a result")

    def chat_sync(
        self, messages: list[dict], *, tools: list[dict] | None = None,
        tool_choice: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None, response_format: dict | None = None,
        extra_body: dict | None = None,
        seed: int | None = None,
    ) -> ChatResult:
        """Run one call on a private event loop, including from concurrent threads."""
        return asyncio.run(self.chat(
            messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens,
            temperature=temperature, response_format=response_format, extra_body=extra_body,
            seed=seed,
        ))

    @staticmethod
    def assistant_message(result: ChatResult) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": result.content or ""}
        if result.reasoning:
            # vLLM 0.30 returns `reasoning`; GLM's chat template renders history thinking ONLY from `reasoning`
            # (verified with /tokenize: `reasoning_content` is silently dropped), while Qwen templates read
            # `reasoning_content`. Send both so every template keeps the earlier turns' thinking.
            message["reasoning"] = result.reasoning
            message["reasoning_content"] = result.reasoning
        if result.tool_calls:
            message["tool_calls"] = [{
                "id": call.id, "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments or json.dumps(call.arguments, ensure_ascii=False),
                },
            } for call in result.tool_calls]
        return message
