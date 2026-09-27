#!/usr/bin/env python3
"""Run ONE challenger (or quality_verifier) subagent call against the real servers on one corpus paper, to inspect GLM's
tool use and output before launching the full loop.

    python scripts/probe_subagent.py --config configs/cs_default.yaml --corpus data/corpus/cs2022_smoke.jsonl \
        --workdir runs/probe --subagent challenger
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from autodata.config import load_config
from autodata.cs.corpus_io import load_papers
from autodata.cs.parsing import parse_challenger_output, parse_qv_output
from autodata.cs.prompts import CHALLENGER_ROUND1_PROMPT, PromptSet
from autodata.cs.run_paper import prepare_workspace
from autodata.harness.agent import Agent
from autodata.harness.tools import make_bash_tool, make_read_tool
from autodata.llm.client import LLMClient


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--workdir", default="runs/probe")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--subagent", default="challenger", choices=["challenger", "quality_verifier"])
    ap.add_argument("--eval-input", help="for quality_verifier: an eval_input.json to verify")
    a = ap.parse_args()
    cfg = load_config(a.config)
    prompts_dir = Path(cfg.prompts_dir).resolve()
    prompts = PromptSet(prompts_dir)
    paper = load_papers(a.corpus, offset=a.index, limit=1)[0]
    wd = Path(a.workdir) / paper.paper_id
    ws = prepare_workspace(cfg, paper, wd, prompts_dir)
    client = LLMClient(cfg.endpoint(a.subagent), name=a.subagent)
    system = prompts.challenger_system() if a.subagent == "challenger" else prompts.quality_verifier_system()
    if a.subagent == "challenger":
        prompt = CHALLENGER_ROUND1_PROMPT
    else:
        ev = json.loads(Path(a.eval_input).read_text())
        prompt = prompts.qv_request(ev.get("question_type", ""), ev["context"], ev["question"], json.dumps(ev["rubric"], indent=1))
    agent = Agent(name=f"probe_{a.subagent}", system_prompt=system, tools=[make_bash_tool(ws), make_read_tool(ws)],
                  llm=client, max_steps=cfg.run.subagent_max_steps, transcript_path=wd / "trajectory" / f"probe_{a.subagent}.jsonl")
    t0 = time.time()
    res = await agent.run(prompt)
    print(f"stop={res.stop_reason} steps={res.steps_used} usage={res.usage} wall={time.time()-t0:.0f}s")
    print("=== FINAL TEXT (tail) ===")
    print((res.final_text or "")[-3000:])
    if a.subagent == "challenger":
        cj = parse_challenger_output(res.final_text or "")
        print("=== PARSED ===", "OK" if cj else "FAILED")
        if cj:
            print(json.dumps({k: (v if k != "rubric" else f"{len(v)} items") for k, v in cj.items()}, indent=1, ensure_ascii=False)[:2500])
            (wd / "eval_input.json").write_text(json.dumps({k: cj.get(k) for k in ("question_type", "context", "question", "rubric")}, indent=1))
            print("wrote", wd / "eval_input.json")
    else:
        print("=== PARSED ===", parse_qv_output(res.final_text or ""))


if __name__ == "__main__":
    asyncio.run(main())
