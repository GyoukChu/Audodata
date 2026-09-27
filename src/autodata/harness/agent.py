"""Sequential tool-calling agents with an append-only, inspectable transcript."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Callable, Literal

from autodata.harness.tools import Tool, _truncate
from autodata.llm.client import LLMClient


@dataclass
class AgentEvent:
    step: int
    kind: Literal["llm", "tool_call", "tool_result", "elide", "final", "error"]
    data: dict
    ts: float


@dataclass
class AgentResult:
    final_text: str
    steps_used: int
    stop_reason: Literal["final", "max_steps", "length", "error"]
    usage: dict[str, int]
    transcript_path: Path | None
    error: str | None = None


def _message_chars(message: dict) -> int:
    def size(value) -> int:
        if value is None:
            return 0
        return len(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))

    total = size(message.get("content"))
    reasoning = message.get("reasoning")
    reasoning_content = message.get("reasoning_content")
    total += size(reasoning)
    if reasoning_content != reasoning:
        total += size(reasoning_content)
    total += sum(size(call.get("function", {}).get("arguments"))
                 for call in message.get("tool_calls") or [])
    return total


class Agent:
    def __init__(
        self, *, name: str, system_prompt: str, tools: list[Tool], llm: LLMClient,
        max_steps: int, transcript_path: Path | None = None,
        event_hook: Callable[[AgentEvent], None] | None = None,
        tool_result_max_chars: int = 400_000,
        context_budget_chars: int | None = None,
        context_budget_tokens: int | None = None,
        keep_recent_tool_results: int = 6,
        max_model_len: int | None = None,
    ):
        if max_steps < 0 or tool_result_max_chars < 0:
            raise ValueError("max_steps and tool_result_max_chars must be nonnegative")
        if context_budget_chars is not None and context_budget_chars < 0:
            raise ValueError("context_budget_chars must be nonnegative")
        if context_budget_tokens is not None and context_budget_tokens < 0:
            raise ValueError("context_budget_tokens must be nonnegative")
        if keep_recent_tool_results < 0:
            raise ValueError("keep_recent_tool_results must be nonnegative")
        if max_model_len is not None and max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps
        self.transcript_path = Path(transcript_path) if transcript_path is not None else None
        self.event_hook = event_hook
        self.tool_result_max_chars = tool_result_max_chars
        self.context_budget_chars = context_budget_chars
        self.context_budget_tokens = context_budget_tokens
        self.keep_recent_tool_results = keep_recent_tool_results
        self.max_model_len = max_model_len

    def _record(self, record: dict) -> None:
        if self.transcript_path is not None:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            # Closing after each line makes partial runs visible immediately.
            with self.transcript_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _event(
        self, step: int, kind: Literal["llm", "tool_call", "tool_result", "elide", "final", "error"], data: dict,
    ) -> None:
        event = AgentEvent(step=step, kind=kind, data=data, ts=time.time())
        self._record({"type": "event", "agent": self.name, **asdict(event)})
        if self.event_hook is not None:
            self.event_hook(event)

    async def run(self, task_prompt: str) -> AgentResult:
        messages: list[dict] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "errors": 0}
        steps_used = 0
        final_text = ""
        tool_map = {tool.name: tool for tool in self.tools}
        tool_specs = [tool.to_openai() for tool in self.tools]
        tool_schema_chars = len(json.dumps(tool_specs, ensure_ascii=False)) if tool_specs else 0
        elided: set[int] = set()
        last_prompt_tokens: int | None = None
        last_prompt_chars = 0

        def append(message: dict, step: int) -> None:
            messages.append(message)
            self._record({
                "type": "message", "agent": self.name, "step": step,
                "ts": time.time(), "message": message,
            })

        def result(
            stop_reason: Literal["final", "max_steps", "length", "error"], error: str | None = None,
        ) -> AgentResult:
            return AgentResult(
                final_text=final_text, steps_used=steps_used, stop_reason=stop_reason,
                usage=usage, transcript_path=self.transcript_path, error=error,
            )

        def fit_context(step: int) -> None:
            token_budget = self.context_budget_tokens
            budget = token_budget if token_budget is not None else self.context_budget_chars
            if budget is None:
                return
            total = sum(_message_chars(message) for message in messages)

            def measure() -> float:
                if token_budget is None:
                    return total
                # Calibrate with the last billed prompt, then estimate additions
                # and removals since that request at four characters per token.
                if last_prompt_tokens is not None:
                    return max(0.0, last_prompt_tokens + (total - last_prompt_chars) / 4)
                return total / 4

            def context_data() -> dict:
                data = {"context_chars": total, "context_budget_chars": self.context_budget_chars}
                if token_budget is not None:
                    data.update(context_tokens=measure(), context_budget_tokens=token_budget)
                return data

            turns = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
            old_turns = turns[:max(0, len(turns) - self.keep_recent_tool_results)]
            for index in old_turns:
                if measure() <= budget:
                    break
                message = messages[index]
                fields = [key for key in ("reasoning", "reasoning_content") if key in message]
                if not fields:
                    continue
                before = _message_chars(message)
                for key in fields:
                    del message[key]
                removed = before - _message_chars(message)
                total -= removed
                self._event(step, "elide", {
                    "message_index": index, "fields": fields, "original_chars": removed,
                    **context_data(),
                })
            tool_indexes = [i for i, message in enumerate(messages) if message["role"] == "tool"]
            eligible = tool_indexes[:max(0, len(tool_indexes) - self.keep_recent_tool_results)]
            for index in eligible:
                if measure() <= budget:
                    break
                if index in elided:
                    continue
                message = messages[index]
                original_chars = len(message["content"])
                replacement = (
                    "[tool result elided by the harness to fit the context window: "
                    f"{original_chars} chars; the full text is in the transcript]"
                )
                message["content"] = replacement
                total -= original_chars - len(replacement)
                elided.add(index)
                self._event(step, "elide", {
                    "message_index": index, "tool_call_id": message["tool_call_id"],
                    "original_chars": original_chars, "content": replacement,
                    **context_data(),
                })

        def hard_preflight(step: int) -> int | None:
            if self.max_model_len is None:
                return None
            max_tokens = self.llm.endpoint.max_tokens

            def estimate() -> float:
                total = sum(_message_chars(message) for message in messages)
                if last_prompt_tokens is not None:
                    return max(0.0, last_prompt_tokens + (total - last_prompt_chars) / 4)
                return (total + tool_schema_chars) / 4

            def fits() -> bool:
                return estimate() + max_tokens <= self.max_model_len

            def fits_soft_budget() -> bool:
                if self.context_budget_tokens is not None:
                    return estimate() <= self.context_budget_tokens
                if self.context_budget_chars is not None:
                    return sum(_message_chars(message) for message in messages) <= self.context_budget_chars
                return True

            def log(action: str, **data) -> None:
                self._event(step, "elide", {
                    "action": action, "context_tokens": estimate(),
                    "max_tokens": max_tokens, "max_model_len": self.max_model_len, **data,
                })

            def compact(indexes: list[int], *, reasoning: bool = False, soft_budget: bool = False) -> None:
                for index in indexes:
                    if fits() and (not soft_budget or fits_soft_budget()):
                        break
                    message = messages[index]
                    before = _message_chars(message)
                    if reasoning:
                        fields = [key for key in ("reasoning", "reasoning_content") if key in message]
                        if not fields:
                            continue
                        for key in fields:
                            del message[key]
                        log("drop_reasoning", message_index=index, fields=fields,
                            original_chars=before - _message_chars(message))
                    elif index not in elided:
                        replacement = "[tool result elided; full text in transcript]"
                        # Compaction must never make a short result larger.
                        if len(replacement) >= len(message["content"]):
                            continue
                        message["content"] = replacement
                        elided.add(index)
                        log("elide_tool_result", message_index=index, original_chars=before,
                            tool_call_id=message["tool_call_id"], content=replacement)

            tools = [i for i, message in enumerate(messages) if message["role"] == "tool"]
            turns = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
            keep = self.keep_recent_tool_results
            compact(tools[:max(0, len(tools) - keep)], soft_budget=True)
            compact(turns[:max(0, len(turns) - keep)], reasoning=True, soft_budget=True)
            protected = tools[max(0, len(tools) - keep):] + turns[max(0, len(turns) - keep):]
            window = max(2, len(messages) - min(protected, default=len(messages)))
            while not fits() and window > 2:
                window -= 1
                log("shrink_protected_window", protected_messages=window)
                cutoff = len(messages) - window
                compact([i for i in tools if i < cutoff])
                compact([i for i in turns if i < cutoff], reasoning=True)
            if not fits():
                available = math.floor(self.max_model_len - estimate())
                if available >= 8192:
                    previous_max_tokens, max_tokens = max_tokens, min(max_tokens, available)
                    log("lower_max_tokens", previous_max_tokens=previous_max_tokens)
                else:
                    log("context_limit_exceeded", available_completion_tokens=available)
                    raise ValueError(
                        f"context preflight: estimated prompt {estimate():.0f} tokens plus "
                        f"completion allowance cannot fit max_model_len={self.max_model_len} "
                        "(minimum reduced max_tokens is 8192)"
                    )
            return max_tokens

        try:
            append({"role": "system", "content": self.system_prompt}, 0)
            append({"role": "user", "content": task_prompt}, 0)
            for step in range(1, self.max_steps + 1):
                steps_used = step
                if self.max_model_len is None:
                    fit_context(step)
                call_max_tokens = hard_preflight(step)
                usage["calls"] += 1
                sent_chars = sum(_message_chars(message) for message in messages)
                completion = await self.llm.chat(
                    messages, tools=tool_specs,
                    **({"max_tokens": call_max_tokens} if call_max_tokens is not None else {}),
                )
                last_prompt_tokens = completion.usage.get("prompt_tokens")
                last_prompt_chars = sent_chars
                for key, value in completion.usage.items():
                    usage[key] = usage.get(key, 0) + value
                final_text = completion.content or ""
                append(LLMClient.assistant_message(completion), step)
                self._event(step, "llm", {
                    "content": completion.content, "reasoning": completion.reasoning,
                    "tool_calls": [asdict(call) for call in completion.tool_calls],
                    "finish_reason": completion.finish_reason, "usage": completion.usage,
                    "latency_s": completion.latency_s,
                })
                if not completion.tool_calls:
                    stop_reason: Literal["length", "final"] = (
                        "length" if completion.finish_reason == "length" else "final"
                    )
                    self._event(step, "final", {"text": final_text, "stop_reason": stop_reason})
                    return result(stop_reason)
                for call in completion.tool_calls:
                    call_data = {"id": call.id, "name": call.name, "arguments": call.arguments}
                    self._event(step, "tool_call", call_data)
                    try:
                        tool = tool_map.get(call.name)
                        if tool is None:
                            output = f"Error: unknown tool {call.name!r}. Allowed: {', '.join(tool_map)}"
                        else:
                            output = str(await tool.handler(call.arguments))
                    except Exception as exc:
                        output = f"Error: {type(exc).__name__}: {exc}"
                    output = _truncate(output, self.tool_result_max_chars)
                    append({"role": "tool", "tool_call_id": call.id, "content": output}, step)
                    self._event(step, "tool_result", {**call_data, "result": output})
            self._event(steps_used, "final", {"text": final_text, "stop_reason": "max_steps"})
            return result("max_steps")
        except Exception as exc:
            for key, value in getattr(exc, "_autodata_usage", {}).items():
                usage[key] = usage.get(key, 0) + value
            usage["errors"] += 1
            error = f"{type(exc).__name__}: {exc}"
            try:
                self._event(steps_used, "error", {"error": error})
            except Exception:
                # Preserve the original failure if transcript I/O or a hook failed.
                pass
            return result("error", error)
