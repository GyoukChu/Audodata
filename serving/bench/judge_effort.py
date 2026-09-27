#!/usr/bin/env python3
"""GLM-5.3 judge cost vs reasoning effort: resend one recorded judge request with reasoning_effort low/high/max.

Takes judge.messages (system = prompts/cs/judge.md, user = context + question + rubric + response) from an
eval-attempt file and sends it --repeats times per effort, all requests concurrently so that every effort sees the
same load (temperature 1.0, top_p 0.95, max_tokens 32768 as for the pipeline's judge role). For each response:
completion/reasoning tokens, latency, whether the JSON verdict parses with one entry per criterion, the rubric
score, and agreement with the other runs and with the verdict recorded in the attempt file.

Usage: python serving/bench/judge_effort.py [--attempt PATH] [--repeats 3] [--out serving/bench/results/judge_effort.json]
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTEMPT = ROOT / "runs/smoke_probe/s2_259138764/eval_attempts/run_001_weak-only/attempt_weak_2.json"
EFFORTS = ("low", "high", "max")


def rubric_weights(user_msg: str) -> list[int]:
    """'N. [+9 positive] ...' / 'N. [-8 negative] ...' lines of the '## Rubric' section, in order."""
    section = user_msg.split("## Rubric", 1)[1].split("\n## ", 1)[0]
    return [int(w) for w in re.findall(r"^\s*\d+\.\s*\[([+-]?\d+)\s+(?:positive|negative)\]", section, re.M)]


def parse_verdict(text: str | None, n: int) -> list[bool] | None:
    if not text:
        return None
    text = text.strip()
    candidates = [text] + re.findall(r"\{.*\}", text, re.S)
    for c in candidates:
        try:
            obj = json.loads(c)
        except ValueError:
            continue
        crit = obj.get("criteria") if isinstance(obj, dict) else None
        if isinstance(crit, list) and len(crit) == n and all(isinstance(c.get("satisfied"), bool) for c in crit):
            return [c["satisfied"] for c in sorted(crit, key=lambda c: c.get("index", 0))]
    return None


def score(sat: list[bool], weights: list[int]) -> float:
    pos = sum(w for w in weights if w > 0)
    earned = sum(w for s, w in zip(sat, weights) if s and w > 0)
    penalty = sum(-w for s, w in zip(sat, weights) if s and w < 0)
    return max(0.0, (earned - penalty) / pos) if pos else 0.0


def agreement(a: list[bool], b: list[bool]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


async def call(client: AsyncOpenAI, messages: list[dict], effort: str, rep: int, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.chat.completions.create(
            model="glm-5.3", messages=messages, temperature=1.0, top_p=0.95, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"reasoning_effort": effort}})
    except Exception as exc:  # noqa: BLE001
        return {"effort": effort, "rep": rep, "error": f"{type(exc).__name__}: {exc}"[:300],
                "latency_s": round(time.perf_counter() - t0, 2)}
    ch = r.choices[0]
    details = r.usage.completion_tokens_details
    return {"effort": effort, "rep": rep, "latency_s": round(time.perf_counter() - t0, 2),
            "finish_reason": ch.finish_reason, "prompt_tokens": r.usage.prompt_tokens,
            "completion_tokens": r.usage.completion_tokens,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
            "content": ch.message.content}


async def main_async(args) -> None:
    attempt = json.loads(Path(args.attempt).read_text())
    messages = attempt["judge"]["messages"]
    weights = rubric_weights(messages[1]["content"])
    n = len(weights)
    recorded = attempt["judge"]["satisfied"]
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0, timeout=3600)
    t0 = time.perf_counter()
    runs = await asyncio.gather(*(call(client, messages, e, k, args.max_tokens)
                                  for e in EFFORTS for k in range(args.repeats)))
    wall = time.perf_counter() - t0
    await client.close()
    for r in runs:
        sat = parse_verdict(r.get("content"), n)
        r["verdict_parses"] = sat is not None
        r["satisfied"] = sat
        r["score"] = round(score(sat, weights), 4) if sat else None
        r["agree_with_recorded"] = round(agreement(sat, recorded), 4) if sat else None
    summary = {}
    for e in EFFORTS:
        rs = [r for r in runs if r["effort"] == e]
        ok = [r for r in rs if r.get("verdict_parses")]
        sats = [r["satisfied"] for r in ok]
        summary[e] = {
            "n": len(rs), "errors": sum(1 for r in rs if "error" in r), "parsed": len(ok),
            "finish_reasons": [r.get("finish_reason") for r in rs],
            "completion_tokens": [r.get("completion_tokens") for r in rs],
            "mean_completion_tokens": round(statistics.fmean(r["completion_tokens"] for r in rs if r.get("completion_tokens"))) if any(r.get("completion_tokens") for r in rs) else None,
            "latency_s": [r.get("latency_s") for r in rs],
            "mean_latency_s": round(statistics.fmean(r["latency_s"] for r in rs), 1),
            "scores": [r["score"] for r in ok],
            "within_effort_pairwise_agreement": round(statistics.fmean(agreement(a, b) for a, b in itertools.combinations(sats, 2)), 4) if len(sats) > 1 else None,
            "agree_with_recorded_mean": round(statistics.fmean(r["agree_with_recorded"] for r in ok), 4) if ok else None,
        }
    # Cross-effort agreement: every pair of parsed runs from two different efforts.
    cross = {}
    for e1, e2 in itertools.combinations(EFFORTS, 2):
        a = [r["satisfied"] for r in runs if r["effort"] == e1 and r.get("verdict_parses")]
        b = [r["satisfied"] for r in runs if r["effort"] == e2 and r.get("verdict_parses")]
        cross[f"{e1}_vs_{e2}"] = round(statistics.fmean(agreement(x, y) for x in a for y in b), 4) if a and b else None
    out = {
        "attempt": str(args.attempt), "n_criteria": n, "weights": weights,
        "recorded_verdict": {"satisfied": recorded, "score": round(score(recorded, weights), 4),
                             "completion_tokens": attempt["judge"]["usage"]["completion_tokens"],
                             "latency_s": round(attempt["judge"]["latency_s"], 1), "effort": "max (pipeline default)"},
        "settings": {"temperature": 1.0, "top_p": 0.95, "max_tokens": args.max_tokens, "repeats": args.repeats,
                     "all_requests_concurrent": True, "wall_s": round(wall, 1)},
        "summary": summary, "cross_effort_agreement": cross, "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"recorded verdict (max effort, smoke run): score {out['recorded_verdict']['score']}, "
          f"{out['recorded_verdict']['completion_tokens']} completion tokens, {out['recorded_verdict']['latency_s']} s")
    for e, s in summary.items():
        print(f"{e:>4}: tokens {s['completion_tokens']} latency {s['latency_s']} finish {s['finish_reasons']} "
              f"parsed {s['parsed']}/{s['n']} scores {s['scores']} within-agree {s['within_effort_pairwise_agreement']} "
              f"agree-recorded {s['agree_with_recorded_mean']}")
    print(f"cross-effort agreement: {cross}; wall {wall:.1f} s; results: {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt", default=str(DEFAULT_ATTEMPT))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", default=str(ROOT / "serving/bench/results/judge_effort.json"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
