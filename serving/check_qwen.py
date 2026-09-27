#!/usr/bin/env python3
"""Thinking-mode smoke check for the Qwen solver servers (serving/serve_qwen27b.sh, serving/serve_qwen4b.sh).

Sends one request with the Qwen thinking-mode sampling settings (temperature 1.0, top_p 0.95, top_k 20,
max_tokens 2048, chat_template_kwargs.enable_thinking=true). It must come back with both reasoning
(reasoning_content or reasoning) and content. Prints finish_reason and usage.

Usage: python serving/check_qwen.py [27b|4b|all]   (default: all)
       python serving/check_qwen.py --base-url http://127.0.0.1:8001/v1 --model qwen3.8-27b
Exit code 0 only if every checked server passes.
"""

from __future__ import annotations

import argparse
import sys
import time

from openai import OpenAI

SERVERS = {
    "27b": ("http://127.0.0.1:8001/v1", "qwen3.8-27b"),
    "4b": ("http://127.0.0.1:8002/v1", "qwen3.5-4b"),
}
PROMPT = "How many prime numbers are there between 1 and 30? Answer with the number and list them."
REASONING_FIELDS = ("reasoning_content", "reasoning")


def short(text: str | None, n: int = 300) -> str:
    if text is None:
        return "None"
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[:n] + f"... [{len(text)} chars]"


def check(base_url: str, model: str, max_tokens: int, timeout: float) -> bool:
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=0)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=1.0,
            top_p=0.95,
            max_tokens=max_tokens,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
        )
    except Exception as exc:  # noqa: BLE001 - report connection/HTTP errors as a failed check
        print(f"[{model}] {base_url} request FAILED: {exc!r}")
        return False
    choice = resp.choices[0]
    dumped = choice.message.model_dump()
    field = next((f for f in REASONING_FIELDS if dumped.get(f)), None)
    reasoning = dumped.get(field) if field else None
    print(f"[{model}] {base_url} ({time.time() - t0:.1f}s)")
    print(f"    finish_reason: {choice.finish_reason}")
    print(f"    usage: {resp.usage.model_dump() if resp.usage else None}")
    print(f"    reasoning field name: {field!r}")
    print(f"    reasoning: {short(reasoning)}")
    print(f"    content:   {short(choice.message.content)}")
    ok = bool(reasoning) and bool(choice.message.content)
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default="all", choices=["27b", "4b", "all"])
    ap.add_argument("--base-url", help="override: check this endpoint only (requires --model)")
    ap.add_argument("--model")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    if args.base_url:
        if not args.model:
            ap.error("--base-url requires --model")
        targets = [(args.base_url, args.model)]
    else:
        targets = list(SERVERS.values()) if args.target == "all" else [SERVERS[args.target]]
    results = [check(url, model, args.max_tokens, args.timeout) for url, model in targets]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
