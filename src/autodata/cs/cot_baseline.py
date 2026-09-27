"""CoT Self-Instruct baseline: single-shot challenger (verbatim round-1 prompt), same quality verifier, then
weak x3 and strong x3 solver attempts graded by the same judge — no agentic loop (paper Sec 3.1, Table 1 left column).

    autodata-cot-baseline --config configs/cs_default.yaml --corpus data/corpus/cs2022_smoke.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from functools import partial
from pathlib import Path

from autodata.config import AppConfig, load_config
from autodata.cs.corpus_io import load_papers
from autodata.cs.parsing import parse_challenger_output, parse_qv_output
from autodata.cs.pipeline import run_papers
from autodata.cs.prompts import CHALLENGER_ROUND1_PROMPT, PromptSet
from autodata.cs.run_paper import PaperInput, evaluator_deadline_s, final_filter, prepare_workspace, run_evaluator
from autodata.harness.agent import Agent
from autodata.harness.tools import make_bash_tool, make_read_tool
from autodata.llm.client import LLMClient


async def _eval(cfg: AppConfig, workdir: Path, mode: str) -> tuple[str, dict | None, int | None]:
    """Run the evaluator CLI as an isolated subprocess with a deadline and cancellation-safe cleanup."""
    argv = ["--input", str((workdir / "eval_input.json").resolve()), f"--{mode}", "--output-dir",
            str((workdir / "eval_attempts").resolve()), "--config", str((workdir / ".opencode/tools/api_config.json").resolve()),
            "--timeout", str(cfg.eval.timeout_s)]
    deadline = evaluator_deadline_s(cfg, mode, cfg.eval.timeout_s)
    try:
        stdout, stderr, rc = await run_evaluator(workdir, argv, deadline_s=deadline)
    except asyncio.TimeoutError:
        return f"SOLVER_ERROR: evaluator exceeded the deadline of {deadline:.0f}s", None, None
    report = None
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH:"):
            try:
                report = json.loads(Path(line.split(":", 1)[1].strip()).read_text(encoding="utf-8"))
            except Exception:
                report = None
    if not stdout.strip():
        stdout = f"[evaluate_rubric exited {rc}] {stderr[-1500:]}"
    return stdout, report, rc


async def run_one(cfg: AppConfig, paper: PaperInput, workdir: Path, clients: dict[str, LLMClient],
                  prompts: PromptSet, prompts_dir_abs: Path, log=print) -> dict:
    t0 = time.time()
    ws = prepare_workspace(cfg, paper, workdir, prompts_dir_abs)
    out: dict = {"paper_id": paper.paper_id, "title": paper.title, "meta": paper.meta, "errors": []}
    # 1) challenger, single shot, verbatim round-1 prompt
    ch = Agent(name="challenger_01", system_prompt=prompts.challenger_system(),
               tools=[make_bash_tool(ws), make_read_tool(ws)], llm=clients["challenger"],
               max_steps=cfg.run.subagent_max_steps, transcript_path=workdir / "trajectory" / "challenger_01.jsonl")
    res = await ch.run(CHALLENGER_ROUND1_PROMPT)
    cj = parse_challenger_output(res.final_text or "")
    out["challenger_stop_reason"] = res.stop_reason
    out["challenger_parsed"] = cj is not None
    out["usage"] = {"challenger": res.usage}
    if not cj:
        out["errors"].append("challenger output could not be parsed")
        out["wall_time_s"] = round(time.time() - t0, 1)
        return out
    out.update({k: cj.get(k) for k in ("question_type", "reasoning_skills", "context", "question", "reference_answer", "rubric")})
    out["question_chars"] = len(cj.get("question") or "")
    out["n_rubric_items"] = len(cj.get("rubric") or []) if isinstance(cj.get("rubric"), list) else None
    eval_input = {"question_type": cj.get("question_type"), "context": cj.get("context"), "question": cj.get("question"),
                  "rubric": cj.get("rubric")}
    (workdir / "eval_input.json").write_text(json.dumps(eval_input, ensure_ascii=False, indent=1), encoding="utf-8")
    # 2) quality verifier (same prompt as the agentic pipeline)
    qv = Agent(name="quality_verifier_01", system_prompt=prompts.quality_verifier_system(),
               tools=[make_bash_tool(ws), make_read_tool(ws)], llm=clients["quality_verifier"],
               max_steps=cfg.run.subagent_max_steps, transcript_path=workdir / "trajectory" / "quality_verifier_01.jsonl")
    qres = await qv.run(prompts.qv_request(str(cj.get("question_type") or ""), cj.get("context") or "",
                                           cj.get("question") or "", json.dumps(cj.get("rubric"), ensure_ascii=False, indent=1)))
    qparsed = parse_qv_output(qres.final_text or "")
    if qres.stop_reason != "final":
        qparsed = {**qparsed, "overall": False, "incomplete": True}
    out["qv_passed"] = qparsed.get("overall")
    out["qv_checks"] = qparsed.get("checks")
    out["qv_contradiction"] = qparsed.get("contradiction")
    out["qv_feedback"] = qparsed.get("feedback")
    out["usage"]["quality_verifier"] = qres.usage
    out["final_filter"] = final_filter(cfg, cj.get("context") or "", cj.get("rubric"))
    out["final_filter_passed"] = bool(out["qv_passed"]) and out["final_filter"]["context_ok"] and out["final_filter"]["rubric_ok"]
    # 3) both solvers, always (statistics need weak AND strong for every item)
    w_out, w_rep, w_rc = await _eval(cfg, workdir, "weak-only")
    s_out, s_rep, s_rc = await _eval(cfg, workdir, "strong-only")
    for name, rep, rc, txt in (("weak", w_rep, w_rc, w_out), ("strong", s_rep, s_rc, s_out)):
        if rep is not None and (rep.get("error") or rc not in (0, None)):
            out["errors"].append(f"{name} eval error: {rep.get('error') or rc}: {txt[:200]}")
    out["weak_report"] = w_rep
    out["strong_report"] = s_rep
    out["weak_avg"] = w_rep.get("weak_avg") if w_rep else None
    out["strong_avg"] = s_rep.get("strong_avg") if s_rep else None
    out["gap"] = s_rep.get("gap") if s_rep else None
    out["weak_passed"] = w_rep.get("weak_passed") if w_rep else None
    out["strong_passed"] = s_rep.get("strong_passed") if s_rep else None
    out["gap_passed"] = s_rep.get("gap_passed") if s_rep else None
    out["all_solver_criteria_passed"] = s_rep.get("all_passed") if s_rep else None
    out["would_be_accepted"] = bool(out["qv_passed"]) and bool(out["all_solver_criteria_passed"])
    if not w_rep:
        out["errors"].append(f"weak eval failed: {w_out[:300]}")
    if not s_rep:
        out["errors"].append(f"strong eval failed: {s_out[:300]}")
    out["wall_time_s"] = round(time.time() - t0, 1)
    log(f"[{paper.paper_id}] CoT: qv={out['qv_passed']} weak={out['weak_avg']} strong={out['strong_avg']} gap={out['gap']}")
    return out


async def run_corpus(cfg: AppConfig, papers: list[PaperInput], root: Path, *, concurrency: int, resume: bool = True,
                     prompts_dir: Path | None = None) -> list[dict]:
    def is_done(summary: dict) -> bool:
        return (not summary.get("errors") and summary.get("weak_avg") is not None
                and summary.get("strong_avg") is not None)

    return await run_papers(cfg, papers, root, concurrency=concurrency, resume=resume, retry_errors=True,
                            summary_filename="cot_summary.json", is_done=is_done, run_one=partial(run_one, cfg),
                            roles=("challenger", "quality_verifier"), prompts_dir=prompts_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CoT Self-Instruct baseline over a corpus")
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--paper-ids")
    ap.add_argument("--workdir-root", help="default: <run.workdir_root>_cot")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--prompts-dir")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ids = set(args.paper_ids.split(",")) if args.paper_ids else None
    papers = load_papers(args.corpus, limit=args.limit, offset=args.offset, paper_ids=ids, min_chars=cfg.run.paper_text_min_chars)
    if not papers:
        print("no papers selected", file=sys.stderr)
        return 1
    root = Path(args.workdir_root or (cfg.run.workdir_root.rstrip("/") + "_cot"))
    conc = args.concurrency or cfg.run.paper_concurrency
    print(f"CoT baseline: {len(papers)} papers -> {root} (concurrency {conc})", flush=True)
    results = asyncio.run(run_corpus(cfg, papers, root, concurrency=conc, resume=not args.no_resume,
                                     prompts_dir=Path(args.prompts_dir) if args.prompts_dir else None))
    ok = [r for r in results if r.get("weak_avg") is not None and r.get("strong_avg") is not None and not r.get("errors")]
    print(f"done: {len(results)} papers, {len(ok)} fully evaluated", flush=True)
    try:
        from autodata.cs.stats import print_report, summarize_cot

        print_report(summarize_cot(root))
    except Exception as e:
        print(f"(stats unavailable: {e!r})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
