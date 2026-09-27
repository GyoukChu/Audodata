#!/usr/bin/env python3
"""Throughput / latency benchmark for the Autodata vLLM servers (serving/serve_*.sh).

For every server and concurrency level C it runs a closed loop: C workers, each sending its next chat request as
soon as the previous one returns, until N requests are done. Reported per (server, C): requests/s, aggregate output
tokens/s, prompt tokens/s, p50/p95/mean latency, finish reasons, errors, server-side deltas from /metrics
(prompt/generation tokens, prefix-cache hits, preemptions, speculative-decoding acceptance, mean TTFT / inter-token
latency / queue time), peak running/waiting/KV usage sampled every 2 s, and sample outputs for spot checks.

Prompts (sampling mirrors configs/cs_default.yaml; only max_tokens is shortened):
  glm   glm-5.3 :8000      papers from --corpus concatenated to 30k-60k tokens (the target paper first, then
                           related papers as background) + a short question-writing instruction; thinking on
                           (reasoning_effort max = model default), temperature 1.0, top_p 0.95, max_tokens 2048.
  27b   qwen3.8-27b :8001  solver format "Context: ... Question: ..." of 1k-2k tokens (paper opening as context and
  4b    qwen3.5-4b :8002   a reasoning question); enable_thinking, temperature 1.0, top_p 0.95, top_k 20, min_p 0,
                           presence_penalty 0 (27b) / 1.5 (4b, as in the pipeline config), max_tokens 4096.
Each prompt starts with a unique tag, so nothing is served from the prefix cache (cold prefill; vLLM's block hashes
chain from the first token). --allow-prefix-cache drops the tag; --prefix-check measures a cold vs. a repeated prompt.

Examples:
  python serving/bench/bench.py --servers glm,27b,4b --levels 1,8,32 --label baseline
  python serving/bench/bench.py --servers 4b --levels 4 --requests 4 --out serving/bench/results/x.json
  python serving/bench/bench.py --servers glm --prefix-check
See serving/bench/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data/corpus/cs2022_smoke.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Sampling mirrors the endpoints in configs/cs_default.yaml; max_tokens is the benchmark's (brief E).
SERVERS: dict[str, dict[str, Any]] = {
    "glm": {
        "base_url": "http://127.0.0.1:8000/v1", "model": "glm-5.3", "tokenizer": "nvidia/GLM-5.3-NVFP4",
        "kind": "paper", "prompt_tokens": (30000, 60000), "max_tokens": 2048,
        "sampling": {"temperature": 1.0, "top_p": 0.95},
        "extra_body": {"chat_template_kwargs": {"reasoning_effort": "max"}},
        "requests": {1: 2, 8: 8, 32: 32},
    },
    "27b": {
        "base_url": "http://127.0.0.1:8001/v1", "model": "qwen3.8-27b", "tokenizer": "Qwen/Qwen3.8-27B-FP8",
        "kind": "question", "prompt_tokens": (1000, 2000), "max_tokens": 4096,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
        "extra_body": {"top_k": 20, "min_p": 0.0, "chat_template_kwargs": {"enable_thinking": True}},
        "requests": {1: 2, 8: 8, 32: 32},
    },
    "4b": {
        "base_url": "http://127.0.0.1:8002/v1", "model": "qwen3.5-4b", "tokenizer": "Qwen/Qwen3.5-4B",
        "kind": "question", "prompt_tokens": (1000, 2000), "max_tokens": 4096,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5},
        "extra_body": {"top_k": 20, "min_p": 0.0, "chat_template_kwargs": {"enable_thinking": True}},
        "requests": {1: 3, 8: 16, 32: 64},
    },
}
ALIASES = {"glm53": "glm", "glm-5.3": "glm", "qwen27b": "27b", "qwen3.8-27b": "27b", "qwen4b": "4b",
           "qwen3.5-4b": "4b", "all": "all"}

GLM_HEADER = ("Generate a challenging research question-answer pair from the following CS paper. The first document "
              "is the target paper; the documents after it are related CS papers included only as background.\n\n")
GLM_INSTRUCTION = ("\nRead the target paper carefully. Then write ONE challenging research question that tests deep "
                   "reasoning about the target paper's method or findings (predicting an outcome, choosing a design "
                   "under constraints, or resolving an apparent contradiction; not recall), followed by a concise "
                   "reference answer of at most 200 words grounded in the paper.")
QUESTIONS = [
    "Suppose the key assumption behind the approach described in the context is violated at deployment time (for "
    "example, the input distribution shifts substantially away from the training or calibration data). Predict how "
    "the method's main reported metric would change relative to the strongest baseline, and justify the direction "
    "and rough magnitude of the change step by step.",
    "Identify the single design choice in the described approach that most limits its scalability. Estimate how its "
    "compute and memory cost grow with problem size under this choice, propose one modification that improves the "
    "scaling, and analyse what the modification would trade off.",
    "A reviewer argues that the reported improvement could be explained entirely by a confounding factor rather than "
    "by the proposed method. Construct the most plausible confounder for this setting, then design a minimal "
    "experiment that separates the two explanations and predict its outcome under each hypothesis.",
    "Two components of the described system interact. Predict what happens to end-to-end performance if the "
    "second component is made twice as accurate while the first stays unchanged, and explain under which conditions "
    "the improvement would be much smaller than a naive estimate suggests.",
]
SPEC_KEYS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
HIST_MEANS = {  # server-side histograms reported as delta(sum) / delta(count)
    "ttft_s": "vllm:time_to_first_token_seconds",
    "itl_s": "vllm:inter_token_latency_seconds",
    "queue_s": "vllm:request_queue_time_seconds",
    "prefill_s": "vllm:request_prefill_time_seconds",
    "decode_s": "vllm:request_decode_time_seconds",
    "e2e_s": "vllm:e2e_request_latency_seconds",
}


# ----------------------------------------------------------------------------------------------------- prompts
class Tok:
    """Token counting/truncation with the served model's tokenizer (HF cache, offline); 4.8 chars/token fallback."""

    def __init__(self, repo: str):
        self.tok = None
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoTokenizer  # noqa: PLC0415 - optional, slow import
            self.tok = AutoTokenizer.from_pretrained(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] tokenizer {repo} unavailable ({type(exc).__name__}); estimating 4.8 chars/token")
        self._cache: dict[str, list[int]] = {}

    def ids(self, text: str) -> list[int]:
        if text not in self._cache:
            self._cache[text] = self.tok.encode(text, add_special_tokens=False)
        return self._cache[text]

    def count(self, text: str) -> int:
        return len(self.ids(text)) if self.tok else max(1, round(len(text) / 4.8))

    def head(self, text: str, n: int) -> str:
        if n <= 0:
            return ""
        if self.tok:
            ids = self.ids(text)
            return text if len(ids) <= n else self.tok.decode(ids[:n])
        return text[: int(n * 4.8)]


def load_papers(path: Path) -> list[dict[str, str]]:
    papers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                # Same layout as autodata.data.paper_text(rec).
                text = f"Title: {r.get('title') or ''}\n\nAbstract: {r.get('abstract') or ''}\n\n{r.get('body_text') or ''}"
                papers.append({"id": r.get("paper_id") or "", "title": r.get("title") or "", "text": text})
    if not papers:
        sys.exit(f"no papers in {path}")
    return papers


def paper_prompt(papers: list[dict], tok: Tok, target: int, start: int) -> tuple[str, list[str]]:
    budget = target - tok.count(GLM_HEADER + GLM_INSTRUCTION)
    parts, used = [GLM_HEADER], []
    k = 0
    while budget > 200:
        p = papers[(start + k) % len(papers)]
        label = "TARGET PAPER" if k == 0 else f"RELATED PAPER {k}"
        head, tail = f"=== {label}: {p['title']} ===\n", f"\n=== END OF {label} ===\n\n"
        room = budget - tok.count(head + tail)
        body = tok.head(p["text"], room)
        parts += [head, body, tail]
        used.append(p["id"])
        budget -= tok.count(head + tail) + tok.count(body)
        k += 1
    parts.append(GLM_INSTRUCTION)
    return "".join(parts), used


def question_prompt(papers: list[dict], tok: Tok, target: int, idx: int) -> tuple[str, list[str]]:
    p = papers[idx % len(papers)]
    q = QUESTIONS[idx % len(QUESTIONS)]
    ctx = tok.head(p["text"], target - tok.count(q) - 8)
    return f"Context:\n{ctx}\n\nQuestion:\n{q}", [p["id"]]


def build_prompts(srv: dict, papers: list[dict], tok: Tok, n: int, seed: int, tag: str | None) -> list[dict]:
    rng = random.Random(f"{seed}-{srv['model']}-{n}")
    lo, hi = srv["prompt_tokens"]
    out = []
    for i in range(n):
        target = rng.randint(lo, hi)
        start = rng.randrange(len(papers))
        if srv["kind"] == "paper":
            text, used = paper_prompt(papers, tok, target, start)
        else:
            text, used = question_prompt(papers, tok, target, start + i)
        if tag is not None:
            text = f"[benchmark request {tag}-{i}]\n\n{text}"
        out.append({"i": i, "text": text, "papers": used, "target_tokens": target})
    return out


# ----------------------------------------------------------------------------------------------------- metrics
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


async def scrape(http: httpx.AsyncClient, root: str) -> dict[str, float] | None:
    """Prometheus text -> {metric[/position=k]: value summed over engines/labels}; None if unreachable."""
    try:
        r = await http.get(f"{root}/metrics", timeout=10)
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, float] = {}
    for line in r.text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = _LINE.match(line)
        if not m or m.group(1).endswith(("_created", "_bucket")):
            continue
        name, labels, value = m.group(1), m.group(2) or "", m.group(3)
        pos = re.search(r'position="(\d+)"', labels)
        if pos:
            name = f"{name}/position={pos.group(1)}"
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            pass
    return out


def summarize_metrics(before: dict | None, after: dict | None) -> dict[str, Any]:
    if not before or not after:
        return {"available": False}
    d = {k: after[k] - before.get(k, 0.0) for k in after
         if (k.endswith(("_total", "_sum", "_count")) or "_total/position=" in k)
         and abs(after[k] - before.get(k, 0.0)) > 1e-9}
    s: dict[str, Any] = {
        "available": True,
        "prompt_tokens": d.get("vllm:prompt_tokens_total", 0.0),
        "generation_tokens": d.get("vllm:generation_tokens_total", 0.0),
        "prompt_tokens_cached": d.get("vllm:prompt_tokens_cached_total", 0.0),
        "prefix_cache_queries": d.get("vllm:prefix_cache_queries_total", 0.0),
        "prefix_cache_hits": d.get("vllm:prefix_cache_hits_total", 0.0),
        "preemptions": d.get("vllm:num_preemptions_total", 0.0),
    }
    for key, base in HIST_MEANS.items():
        cnt = d.get(f"{base}_count", 0.0)
        s[f"mean_{key}"] = (d.get(f"{base}_sum", 0.0) / cnt) if cnt else None
    spec = {k: d.get(f"vllm:spec_decode_{k}_total", 0.0) for k in SPEC_KEYS}
    if spec["num_drafts"]:
        per_pos = sorted((int(k.split("=")[1]), v) for k, v in d.items()
                         if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total/position="))
        spec["acceptance_rate"] = spec["num_accepted_tokens"] / max(spec["num_draft_tokens"], 1)
        spec["mean_acceptance_length"] = 1 + spec["num_accepted_tokens"] / spec["num_drafts"]
        spec["accepted_rate_per_pos"] = [round(v / spec["num_drafts"], 4) for _, v in per_pos]
        s["spec_decode"] = spec
    return s


def gauges(m: dict | None) -> dict[str, float | None]:
    m = m or {}
    return {"running": m.get("vllm:num_requests_running"), "waiting": m.get("vllm:num_requests_waiting"),
            "kv_usage": m.get("vllm:kv_cache_usage_perc")}


async def sample_gauges(http: httpx.AsyncClient, root: str, store: list, stop: asyncio.Event, period: float = 2.0):
    while not stop.is_set():
        g = gauges(await scrape(http, root))
        if g["running"] is not None:
            store.append(g)
        try:
            await asyncio.wait_for(stop.wait(), timeout=period)
        except asyncio.TimeoutError:
            pass


# ----------------------------------------------------------------------------------------------------- requests
def distinct_4gram_ratio(text: str) -> float | None:
    """Share of distinct word 4-grams (degenerate loops score low; normal prose ~0.9+)."""
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return round(len(set(grams)) / len(grams), 4) if len(grams) >= 20 else None


async def one_request(client: AsyncOpenAI, srv: dict, prompt: dict, t0: float, stream: bool, sink: list) -> None:
    """Send one request; its record is appended to `sink` even when the request is cancelled."""
    rec: dict[str, Any] = {"i": prompt["i"], "start": time.perf_counter() - t0}
    kwargs = dict(model=srv["model"], messages=[{"role": "user", "content": prompt["text"]}],
                  max_tokens=srv["max_tokens"], extra_body=srv["extra_body"], **srv["sampling"])
    reasoning = content = None
    try:
        if stream:
            first = None
            rs, cs, usage, finish = [], [], None, None
            resp = await client.chat.completions.create(**kwargs, stream=True, stream_options={"include_usage": True})
            async for chunk in resp:
                if chunk.usage:
                    usage = chunk.usage
                for ch in chunk.choices:
                    extra = ch.delta.model_extra or {}
                    r = extra.get("reasoning") or extra.get("reasoning_content")
                    if first is None and (r or ch.delta.content):
                        first = time.perf_counter() - t0
                    if r:
                        rs.append(r)
                    if ch.delta.content:
                        cs.append(ch.delta.content)
                    if ch.finish_reason:
                        finish = ch.finish_reason
            reasoning, content = "".join(rs), "".join(cs)
            rec["ttft"] = None if first is None else first - rec["start"]
        else:
            resp = await client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            extra = choice.message.model_extra or {}
            reasoning = extra.get("reasoning") or extra.get("reasoning_content")
            content, usage, finish = choice.message.content, resp.usage, choice.finish_reason
        rec["end"] = time.perf_counter() - t0
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        rec.update(ok=True, finish=finish, prompt_tokens=usage.prompt_tokens if usage else None,
                   completion_tokens=usage.completion_tokens if usage else None,
                   reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
                   reasoning_chars=len(reasoning or ""), content_chars=len(content or ""),
                   distinct4=distinct_4gram_ratio(f"{reasoning or ''}\n{content or ''}"))
    except asyncio.CancelledError:
        rec.update(ok=False, cancelled=True, end=time.perf_counter() - t0, error="cancelled (--max-seconds)")
        raise
    except Exception as exc:  # noqa: BLE001 - errors are part of the measurement
        rec.update(ok=False, end=time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}"[:400])
    finally:
        rec["latency"] = rec["end"] - rec["start"]
        rec["_text"] = (reasoning, content)
        sink.append(rec)


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs: list[float]) -> dict[str, float | None]:
    r = lambda v: None if v is None else round(v, 3)  # noqa: E731
    return {"p50": r(pct(xs, 0.5)), "p95": r(pct(xs, 0.95)), "mean": r(statistics.fmean(xs)) if xs else None,
            "min": r(min(xs)) if xs else None, "max": r(max(xs)) if xs else None}


def sample(rec: dict) -> dict:
    reasoning, content = rec["_text"]
    reasoning, content = reasoning or "", content or ""
    head = reasoning if len(reasoning) <= 1100 else reasoning[:700] + " [...] " + reasoning[-400:]
    return {"i": rec["i"], "finish": rec.get("finish"), "completion_tokens": rec.get("completion_tokens"),
            "distinct4": rec.get("distinct4"), "reasoning": head, "content": content[:800]}


async def run_level(name: str, srv: dict, prompts: list[dict], conc: int, args, http: httpx.AsyncClient) -> dict:
    root = srv["base_url"].rsplit("/v1", 1)[0]
    client = AsyncOpenAI(base_url=srv["base_url"], api_key="EMPTY", max_retries=0,
                         timeout=httpx.Timeout(args.request_timeout, connect=10.0))
    before = await scrape(http, root)
    load_before = gauges(before)
    samples_g: list[dict] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_gauges(http, root, samples_g, stop))
    queue = list(prompts)
    records: list[dict] = []
    t0 = time.perf_counter()

    async def worker():
        while queue:
            await one_request(client, srv, queue.pop(0), t0, args.stream, records)

    tasks = [asyncio.create_task(worker()) for _ in range(min(conc, len(prompts)))]
    timed_out = False
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.max_seconds)
    except asyncio.TimeoutError:
        timed_out = True
    wall = time.perf_counter() - t0
    stop.set()
    await sampler
    await client.close()
    after = await scrape(http, root)

    ok = [r for r in records if r.get("ok")]
    errs = [r for r in records if not r.get("ok")]
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    in_tok = sum(r["prompt_tokens"] or 0 for r in ok)
    # Wall time = first dispatch -> last completion (or the --max-seconds cut-off).
    if records and not timed_out:
        wall = max(r["end"] for r in records)
    finishes: dict[str, int] = {}
    for r in ok:
        finishes[str(r["finish"])] = finishes.get(str(r["finish"]), 0) + 1
    srv_m = summarize_metrics(before, after)
    d4 = [r["distinct4"] for r in ok if r.get("distinct4") is not None]
    res = {
        "concurrency": conc, "n_requests": len(prompts), "n_ok": len(ok), "n_errors": len(errs),
        "n_cancelled": sum(1 for r in records if r.get("cancelled")),
        "n_not_started": len(prompts) - len(records), "timed_out": timed_out,
        "errors": sorted({r["error"] for r in errs})[:5],
        "wall_s": round(wall, 2),
        "req_per_s": round(len(ok) / wall, 4) if wall else None,
        "output_tok_per_s": round(out_tok / wall, 1) if wall else None,
        "prompt_tok_per_s": round(in_tok / wall, 1) if wall else None,
        "latency_s": dist([r["latency"] for r in ok]),
        "per_request_output_tok_per_s": dist([r["completion_tokens"] / r["latency"] for r in ok
                                             if r["completion_tokens"] and r["latency"] > 0]),
        "mean_prompt_tokens": round(in_tok / len(ok)) if ok else None,
        "mean_completion_tokens": round(out_tok / len(ok)) if ok else None,
        "total_completion_tokens": out_tok, "total_prompt_tokens": in_tok,
        "finish_reasons": finishes,
        "distinct4": {"mean": round(statistics.fmean(d4), 4) if d4 else None, "min": min(d4) if d4 else None},
        "empty_outputs": sum(1 for r in ok if not r["reasoning_chars"] and not r["content_chars"]),
        "server": srv_m,
        # Other clients' traffic during the run (server-side generated tokens not produced for this benchmark).
        "external_generation_tokens": (round(srv_m["generation_tokens"] - out_tok) if srv_m.get("available")
                                       else None),
        "load_before": load_before,
        "load_during": {k: {"max": max(v), "mean": round(statistics.fmean(v), 3)}
                        for k in ("running", "waiting", "kv_usage")
                        if (v := [g[k] for g in samples_g if g.get(k) is not None])},
    }
    if args.stream:
        res["ttft_s"] = dist([r["ttft"] for r in ok if r.get("ttft") is not None])
        res["decode_tok_per_s_per_request"] = dist(
            [(r["completion_tokens"] - 1) / (r["latency"] - r["ttft"]) for r in ok
             if r.get("ttft") is not None and r["completion_tokens"] and r["latency"] > r["ttft"]])
    res["requests"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "_text"}
                       for r in sorted(records, key=lambda r: r["i"])]
    res["samples"] = [sample(r) for r in sorted(ok, key=lambda r: r["i"])[: args.samples]]
    return res


async def prefix_check(name: str, srv: dict, papers: list[dict], tok: Tok, http: httpx.AsyncClient, args) -> dict:
    """Send the same fresh prompt twice (max_tokens 1): the second prefill should come from the prefix cache."""
    root = srv["base_url"].rsplit("/v1", 1)[0]
    client = AsyncOpenAI(base_url=srv["base_url"], api_key="EMPTY", max_retries=0, timeout=args.request_timeout)
    prompt = build_prompts(srv, papers, tok, 1, args.seed, uuid.uuid4().hex[:8])[0]
    out = {}
    for label in ("cold", "repeat"):
        before = await scrape(http, root)
        t = time.perf_counter()
        r = await client.chat.completions.create(
            model=srv["model"], messages=[{"role": "user", "content": prompt["text"]}], max_tokens=1,
            extra_body=srv["extra_body"], **srv["sampling"])
        lat = time.perf_counter() - t
        m = summarize_metrics(before, await scrape(http, root))
        out[label] = {"latency_s": round(lat, 3), "prompt_tokens": r.usage.prompt_tokens,
                      "prefix_cache_hits": m.get("prefix_cache_hits"), "prefix_cache_queries": m.get("prefix_cache_queries"),
                      "prompt_tokens_cached": m.get("prompt_tokens_cached")}
    await client.close()
    return out


# ----------------------------------------------------------------------------------------------------- driver
def server_cmdline(url: str) -> str | None:
    port = re.search(r":(\d+)", url.split("//", 1)[-1])
    if not port:
        return None
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in ps.splitlines():
        if "vllm serve" in line and re.search(rf"--port {port.group(1)}(\s|$)", line):
            return line.split("vllm serve", 1)[1].strip()
    return None


def gpu_memory() -> list[str] | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=10).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        return None


def parse_requests(spec: str | None, levels: list[int], table: dict[int, int]) -> dict[int, int]:
    if spec is None:
        return {c: table.get(c, max(4, 2 * c)) for c in levels}
    if ":" not in spec:
        return {c: int(spec) for c in levels}
    m = {int(a): int(b) for a, b in (kv.split(":") for kv in spec.split(","))}
    return {c: m.get(c, table.get(c, max(4, 2 * c))) for c in levels}


def print_table(results: dict) -> None:
    hdr = (f"{'server':<12}{'C':>4}{'N':>5}{'ok':>4}{'err':>4}{'wall_s':>8}{'req/s':>8}{'out_tok/s':>10}"
           f"{'in_tok/s':>10}{'p50_s':>8}{'p95_s':>8}{'out/req':>8}{'ttft*':>7}{'itl_ms*':>8}{'accept':>7}{'ext_tok':>8}")
    print(hdr)
    for srv in results["servers"].values():
        for lv in srv.get("levels", []):
            s = lv["server"]
            acc = s.get("spec_decode", {}).get("mean_acceptance_length")
            itl = s.get("mean_itl_s")
            print(f"{srv['model']:<12}{lv['concurrency']:>4}{lv['n_requests']:>5}{lv['n_ok']:>4}{lv['n_errors']:>4}"
                  f"{lv['wall_s']:>8.1f}{lv['req_per_s'] or 0:>8.3f}{lv['output_tok_per_s'] or 0:>10.1f}"
                  f"{lv['prompt_tok_per_s'] or 0:>10.1f}{lv['latency_s']['p50'] or 0:>8.1f}{lv['latency_s']['p95'] or 0:>8.1f}"
                  f"{lv['mean_completion_tokens'] or 0:>8}{(s.get('mean_ttft_s') or 0):>7.2f}{(itl or 0) * 1000:>8.1f}"
                  f"{(acc or 0):>7.2f}{lv['external_generation_tokens'] if lv['external_generation_tokens'] is not None else '-':>8}")
    print("* ttft/itl are server-side means from /metrics (include any other traffic); accept = mean spec-decode "
          "acceptance length (0 = off); ext_tok = generated tokens from other clients during the run")


async def main_async(args) -> int:
    names = []
    for s in args.servers.split(","):
        s = ALIASES.get(s.strip(), s.strip())
        names += list(SERVERS) if s == "all" else [s]
    for n in names:
        if n not in SERVERS:
            sys.exit(f"unknown server {n!r}; choose from {list(SERVERS)}")
    overrides = dict(kv.split("=", 1) for kv in args.url)
    levels = [int(c) for c in args.levels.split(",")]
    papers = load_papers(Path(args.corpus))
    tag_base = None if args.allow_prefix_cache else uuid.uuid4().hex[:8]
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{args.label}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "label": args.label, "notes": args.notes, "host": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv[1:],
        "mode": "stream" if args.stream else "non-stream",
        "prefix_cache": "allowed" if args.allow_prefix_cache else "busted (unique tag per request)",
        "gpu_memory_start": gpu_memory(), "servers": {},
    }

    def save():
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        tmp.replace(out_path)

    async with httpx.AsyncClient() as http:
        for name in names:
            srv = dict(SERVERS[name])
            srv["base_url"] = overrides.get(name, srv["base_url"])
            if args.max_tokens:
                srv["max_tokens"] = args.max_tokens
            try:
                models = (await http.get(f"{srv['base_url']}/models", timeout=10)).json()
                info = {m["id"]: m.get("max_model_len") for m in models.get("data", [])}
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] {srv['base_url']} unreachable ({exc}); skipped")
                results["servers"][name] = {"model": srv["model"], "error": f"unreachable: {exc}"}
                save()
                continue
            tok = Tok(srv["tokenizer"])
            entry = results["servers"][name] = {
                "model": srv["model"], "base_url": srv["base_url"], "max_model_len": info.get(srv["model"]),
                "server_args": server_cmdline(srv["base_url"]), "max_tokens": srv["max_tokens"],
                "sampling": {**srv["sampling"], **srv["extra_body"]}, "prompt_tokens_target": srv["prompt_tokens"],
                "levels": [],
            }
            if args.prefix_check:
                entry["prefix_check"] = await prefix_check(name, srv, papers, tok, http, args)
                print(f"[{name}] prefix check: {json.dumps(entry['prefix_check'])}")
                save()
                continue
            # Untimed warm-up so the first measured request does not pay one-off costs.
            try:
                c = AsyncOpenAI(base_url=srv["base_url"], api_key="EMPTY", max_retries=0, timeout=120)
                await c.chat.completions.create(model=srv["model"], messages=[{"role": "user", "content": "Hi"}],
                                                max_tokens=8, extra_body=srv["extra_body"], **srv["sampling"])
                await c.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] warm-up failed: {exc}")
            n_by_level = parse_requests(args.requests, levels, srv["requests"])
            for conc in levels:
                n = n_by_level[conc]
                prompts = build_prompts(srv, papers, tok, n, args.seed, None if tag_base is None else f"{tag_base}-c{conc}")
                print(f"[{time.strftime('%H:%M:%S')}] {name} ({srv['model']}): concurrency {conc}, {n} requests, "
                      f"prompt ~{statistics.fmean(p['target_tokens'] for p in prompts):.0f} tokens, "
                      f"max_tokens {srv['max_tokens']}", flush=True)
                lv = await run_level(name, srv, prompts, conc, args, http)
                entry["levels"].append(lv)
                print(f"    -> ok {lv['n_ok']}/{n}, wall {lv['wall_s']} s, {lv['output_tok_per_s']} out tok/s, "
                      f"{lv['prompt_tok_per_s']} in tok/s, p50 {lv['latency_s']['p50']} s, p95 {lv['latency_s']['p95']} s, "
                      f"finish {lv['finish_reasons']}, errors {lv['errors'][:2]}", flush=True)
                save()
                if args.pause and conc != levels[-1]:
                    await asyncio.sleep(args.pause)
    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    if not args.prefix_check:
        print_table(results)
    print(f"results: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servers", default="all", help="comma list of glm,27b,4b or all (default all)")
    ap.add_argument("--levels", default="1,8,32", help="concurrency levels (default 1,8,32)")
    ap.add_argument("--requests", help="requests per level: N for all levels or C:N,C:N (default: per-server table)")
    ap.add_argument("--max-tokens", type=int, help="override max_tokens (default glm 2048, qwen 4096)")
    ap.add_argument("--url", action="append", default=[], metavar="NAME=URL",
                    help="override a base URL, e.g. 4b=http://127.0.0.1:8003/v1")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--seed", type=int, default=0, help="prompt selection seed (same seed = same prompts)")
    ap.add_argument("--stream", action="store_true", help="stream responses to measure TTFT (adds client CPU load)")
    ap.add_argument("--allow-prefix-cache", action="store_true", help="omit the unique per-request tag")
    ap.add_argument("--prefix-check", action="store_true", help="only check prefix caching (cold vs repeated prompt)")
    ap.add_argument("--max-seconds", type=float, default=420.0,
                    help="hard cap per level; unfinished requests are cancelled (default 420)")
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--pause", type=float, default=5.0, help="seconds between levels (default 5)")
    ap.add_argument("--samples", type=int, default=3, help="sample outputs saved per level (default 3)")
    ap.add_argument("--label", default="bench")
    ap.add_argument("--notes", default="", help="free text stored in the results (e.g. the server config)")
    ap.add_argument("--out", help="results JSON (default serving/bench/results/<label>_<time>.json)")
    args = ap.parse_args()
    def on_term(signum, frame):  # noqa: ARG001 - treat SIGTERM like Ctrl-C (partial results are already saved)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("interrupted; partial results were saved after each level")
        return 130


if __name__ == "__main__":
    sys.exit(main())
