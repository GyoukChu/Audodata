#!/usr/bin/env python3
"""Smoke checks for the GLM-5.3 server (serving/serve_glm53.sh), OpenAI python client.

(a) plain chat completion: prints the reasoning field name vLLM returns (reasoning_content or reasoning) + content
(b) tool call: tools=[one function], a prompt that forces a call -> message.tool_calls with JSON-parseable arguments
(c) two-turn tool round trip: assistant message (with tool_calls) + tool result -> final answer
(d) extra_body={"chat_template_kwargs": {"reasoning_effort": "max"}} is accepted

Usage: python serving/check_glm53.py [--base-url http://127.0.0.1:8000/v1] [--model glm-5.3]
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from openai import OpenAI

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}
TOOL_PROMPT = (
    "What is the weather in Paris right now, in celsius? You have no weather knowledge of your own: "
    "you must call the get_weather tool."
)
REASONING_FIELDS = ("reasoning_content", "reasoning")


def reasoning_of(message: Any) -> tuple[str | None, str | None]:
    """Return (field_name, text) for whichever reasoning field vLLM filled in."""
    dumped = message.model_dump()
    for field in REASONING_FIELDS:
        value = dumped.get(field)
        if value:
            return field, value
    return None, None


def short(text: str | None, n: int = 300) -> str:
    if text is None:
        return "None"
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[:n] + f"... [{len(text)} chars]"


def run(client: OpenAI, model: str, max_tokens: int) -> dict[str, bool]:
    results: dict[str, bool] = {}

    # (a) plain chat completion
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 17 * 23? Reply with just the number."}],
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    field, reasoning = reasoning_of(msg)
    present = [k for k, v in msg.model_dump().items() if v not in (None, [], "")]
    print(f"[a] plain chat ({time.time() - t0:.1f}s) finish_reason={resp.choices[0].finish_reason} usage={resp.usage.model_dump()}")
    print(f"    non-empty message fields: {present}")
    print(f"    reasoning field name: {field!r}")
    print(f"    reasoning: {short(reasoning)}")
    print(f"    content:   {short(msg.content)}")
    results["a_plain_chat"] = bool(field) and bool(msg.content) and "391" in (msg.content or "")

    # (b) tool call
    t0 = time.time()
    messages: list[dict[str, Any]] = [{"role": "user", "content": TOOL_PROMPT}]
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=[WEATHER_TOOL], tool_choice="auto", max_tokens=max_tokens
    )
    choice = resp.choices[0]
    msg = choice.message
    calls = msg.tool_calls or []
    print(f"[b] tool call ({time.time() - t0:.1f}s) finish_reason={choice.finish_reason} n_tool_calls={len(calls)}")
    print(f"    content: {short(msg.content)}")
    parsed_ok = False
    for call in calls:
        try:
            args = json.loads(call.function.arguments)
            parsed_ok = isinstance(args, dict) and "city" in args
        except json.JSONDecodeError as exc:
            args = f"<invalid JSON: {exc}>"
        print(f"    tool_call id={call.id} name={call.function.name} raw_arguments={call.function.arguments!r} parsed={args}")
    results["b_tool_call"] = bool(calls) and calls[0].function.name == "get_weather" and parsed_ok

    # (c) two-turn tool round trip
    if calls:
        t0 = time.time()
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ],
        }
        rfield, rtext = reasoning_of(msg)
        if rtext:
            assistant["reasoning_content"] = rtext
        messages.append(assistant)
        for c in calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": c.id,
                    "content": json.dumps({"city": "Paris", "temperature": 18, "unit": "celsius", "condition": "light rain"}),
                }
            )
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=[WEATHER_TOOL], tool_choice="auto", max_tokens=max_tokens
        )
        choice = resp.choices[0]
        print(f"[c] tool round trip ({time.time() - t0:.1f}s) finish_reason={choice.finish_reason} "
              f"n_tool_calls={len(choice.message.tool_calls or [])}")
        print(f"    final content: {short(choice.message.content)}")
        results["c_tool_round_trip"] = bool(choice.message.content) and "18" in choice.message.content
    else:
        print("[c] skipped: no tool call in (b)")
        results["c_tool_round_trip"] = False

    # (d) reasoning_effort=max through chat_template_kwargs
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Name the largest planet in the solar system in one word."}],
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"reasoning_effort": "max"}},
        )
        msg = resp.choices[0].message
        field, reasoning = reasoning_of(msg)
        print(f"[d] reasoning_effort=max ({time.time() - t0:.1f}s) finish_reason={resp.choices[0].finish_reason} "
              f"usage={resp.usage.model_dump()}")
        print(f"    reasoning ({field}): {short(reasoning, 150)}")
        print(f"    content: {short(msg.content)}")
        results["d_reasoning_effort_max"] = bool(msg.content)
    except Exception as exc:  # noqa: BLE001 - report any server-side rejection
        print(f"[d] reasoning_effort=max FAILED: {exc!r}")
        results["d_reasoning_effort_max"] = False

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout, max_retries=0)
    results = run(client, args.model, args.max_tokens)
    print("\nSUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
