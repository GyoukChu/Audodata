"""End-to-end tests of one paper through the Agentic Self-Instruct loop against a scripted fake OpenAI server.

Exercises: main agent tool loop (task/bash/write), challenger + QV subagents (cat ./paper.txt), the evaluate_rubric.py
CLI as a real isolated subprocess (solvers x3 + judge), harness round tracking, harness-verified acceptance, guardrails
(write allow-list, planted-module isolation, QV binding/contradictions, strong-without-weak) and the end-of-loop QV.
No GPU, no network (localhost only).
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path

import pytest

from autodata.config import AppConfig, ModelEndpoint
from autodata.cs.prompts import PromptSet
from autodata.cs.run_paper import PaperInput, PaperRun, write_allowed
from autodata.llm.client import LLMClient

from .fake_openai_server import FakeServer, text_response, tool_call_response

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "cs"
SENTINEL_REF = "GATING-SATURATES-FIRST-7731"

CHALLENGER_JSON = {
    "question_type": "failure mode prediction",
    "reasoning_skills": ["causal_reasoning", "design_tradeoff"],
    "context": "The paper studies a retrieval-augmented decoder under distribution shift. " * 6,
    "question": "Predict which component fails first when the retriever index is stale, and justify why.",
    "reference_answer": f"The answer is {SENTINEL_REF} because the gate saturates before the retriever degrades.",
    "rubric": [{"criterion": f"positive insight {i}", "weight": 5, "category": "positive"} for i in range(8)]
              + [{"criterion": f"negative error {i}", "weight": -4, "category": "negative"} for i in range(3)],
}
EVAL_INPUT = {k: CHALLENGER_JSON[k] for k in ("question_type", "context", "question", "rubric")}
QV_PASS = ("CHECK_1_VERDICT: NO_LEAKAGE\nCHECK_2_VERDICT: GOOD\nCHECK_3_VERDICT: PASS\n"
           "CHECK_3_ISSUES: none (Positive: 8, Negative: 3, Total: 11)\nCHECK_4_VERDICT: CONSISTENT\nOVERALL: PASS\nFEEDBACK: none")
QV_CONTRADICTION = QV_PASS.replace("CHECK_1_VERDICT: NO_LEAKAGE", "CHECK_1_VERDICT: LEAKS_ANSWER")
WEAK_CMD = ("cd /workspace/project && uv run python3 .opencode/tools/evaluate_rubric.py --input ./eval_input.json "
            "--weak-only --output-dir ./eval_attempts --config .opencode/tools/api_config.json --timeout 600")
STRONG_CMD = WEAK_CMD.replace("--weak-only", "--strong-only")
QV_PROMPT_FULL = ("Verify this QA package.\n\nquestion_type: failure mode prediction\n\ncontext:\n" + CHALLENGER_JSON["context"]
                  + "\n\nquestion:\n" + CHALLENGER_JSON["question"] + "\n\nrubric:\n" + json.dumps(CHALLENGER_JSON["rubric"]))
RESULT_JSON = {"paper_title": "T", "question_type": "failure mode prediction", "reasoning_skills": ["causal_reasoning"],
               "rounds": [{"refinement_round": 1, "question": CHALLENGER_JSON["question"], "context": CHALLENGER_JSON["context"],
                           "reference_answer": CHALLENGER_JSON["reference_answer"], "rubric": CHALLENGER_JSON["rubric"], "accepted": True,
                           "quality_verifier_kimi_passed": True, "quality_verifier_kimi_feedback": "PASS", "weak_solver_avg": "15%",
                           "strong_solver_avg": "100%", "gap": "85%", "eval_report": "...", "eval_output_dir": "./eval_attempts"}],
               "final_accepted_round": 1, "total_rounds": 1}


def _role(req: dict) -> str:
    msgs = req.get("messages", [])
    sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
    if sys_msg is None:
        return "solver"
    head = (sys_msg.get("content") or "")[:40]
    for prefix, role in (("# Main Agent", "main"), ("# Challenger", "challenger"), ("# Quality Verifier", "qv"), ("# Rubric Judge", "judge")):
        if head.startswith(prefix):
            return role
    return "unknown"


def happy_main_steps() -> list[dict]:
    return [
        tool_call_response([("task", {"description": "challenger round 1", "prompt": "Generate a challenging research question-answer pair with grading rubrics. The paper is available at ./paper.txt — read it first.", "subagent_type": "challenger"})]),
        tool_call_response([("task", {"description": "QV round 1", "prompt": QV_PROMPT_FULL, "subagent_type": "quality_verifier"})]),
        tool_call_response([("write", {"filePath": "eval_input.json", "content": json.dumps(EVAL_INPUT)})]),
        tool_call_response([("bash", {"command": WEAK_CMD})]),
        tool_call_response([("bash", {"command": STRONG_CMD})]),
        tool_call_response([("write", {"filePath": "output/result.json", "content": json.dumps(RESULT_JSON)})]),
        text_response("Round 1 ACCEPTED. Final result.json written."),
    ]


class Scenario:
    """Stateful responder: decides each role's next action from how many tool results it has already seen."""

    def __init__(self, main_steps: list[dict] | None = None, qv_text: str = QV_PASS):
        self.lock = threading.Lock()
        self.calls: list[tuple[str, dict]] = []
        self.main_steps = main_steps or happy_main_steps()
        self.qv_text = qv_text

    def __call__(self, req: dict, idx: int) -> dict:
        role = _role(req)
        with self.lock:
            self.calls.append((role, req))
        msgs = req["messages"]
        n_tool_results = sum(1 for m in msgs if m.get("role") == "tool")
        model = req.get("model", "")
        if role == "solver":
            text = "STRONG ANSWER: the gate saturates first because ..." if "27b" in model else "WEAK ANSWER: the retriever."
            return text_response(text, reasoning="thinking about it", model=model)
        if role == "judge":
            user = next(m["content"] for m in msgs if m["role"] == "user")
            n = len(re.findall(r"^\s*\d+\.\s", user, re.M))
            strong = "STRONG ANSWER" in user
            marks = []
            for i in range(1, n + 1):
                positive = i <= 8
                sat = (positive and (strong or i <= 2)) or ((not positive) and (not strong) and i == 9)
                marks.append({"index": i, "satisfied": bool(sat), "evidence": "quoted" if sat else ""})
            return text_response(json.dumps({"criteria": marks}), model=model)
        if role == "challenger":
            if n_tool_results == 0:
                return tool_call_response([("bash", {"command": "cat ./paper.txt"})])
            return text_response("## Scratchpad Analysis\n...\n```json\n" + json.dumps(CHALLENGER_JSON) + "\n```")
        if role == "qv":
            if n_tool_results == 0:
                return tool_call_response([("bash", {"command": "cat ./paper.txt"})])
            return text_response("Analysis...\n" + self.qv_text)
        if role == "main":
            return self.main_steps[min(n_tool_results, len(self.main_steps) - 1)]
        return text_response("unknown role")


def _config(base_url: str, tmp: Path) -> AppConfig:
    ep = lambda model, **kw: ModelEndpoint(base_url=base_url, model=model, max_tokens=512, timeout_s=30, max_retries=2, **kw)  # noqa: E731
    return AppConfig.model_validate({
        "run": {"name": "e2e", "max_rounds": 3, "main_agent_max_steps": 20, "subagent_max_steps": 12,
                "workdir_root": str(tmp / "runs"), "paper_text_min_chars": 10, "final_qv": True},
        "acceptance_preset": "prose_s31",
        "eval": {"n_attempts": 3, "timeout_s": 30, "judge_retries": 2, "solver_retries": 1},
        "prompts_dir": str(PROMPTS),
        "models": {
            "main_agent": ep("glm-5.3").model_dump(), "challenger": ep("glm-5.3").model_dump(),
            "quality_verifier": ep("glm-5.3").model_dump(), "judge": ep("glm-5.3").model_dump(),
            "weak_solver": ep("qwen3.5-4b", chat_template_kwargs={"enable_thinking": True}).model_dump(),
            "strong_solver": ep("qwen3.8-27b").model_dump(),
        },
    })


def _run(tmp_path: Path, scenario: Scenario, paper_id: str = "p1") -> tuple[dict, Path]:
    with FakeServer(scenario) as srv:
        cfg = _config(srv.base_url, tmp_path)
        prompts = PromptSet(PROMPTS)
        clients = {r: LLMClient(cfg.endpoint(r), name=r) for r in ("main_agent", "challenger", "quality_verifier")}
        paper = PaperInput(paper_id=paper_id, title="A retrieval paper", text="Title: A retrieval paper\n\nAbstract: x\n\n" + "body " * 500)
        wd = tmp_path / "runs" / paper_id
        run = PaperRun(cfg, paper, wd, clients, prompts, PROMPTS, log=lambda m: None)
        summary = asyncio.run(run.run())
    return summary, wd


def test_one_paper_accepted_end_to_end(tmp_path: Path):
    scenario = Scenario()
    summary, wd = _run(tmp_path, scenario)
    assert (wd / "paper.txt").exists() and (wd / ".opencode" / "tools" / "api_config.json").exists()
    assert summary["accepted"] is True and summary["accepted_round"] == 1, summary
    assert summary["n_rounds"] == 1 and summary["completed"] is True
    r1 = summary["rounds"][0]
    assert r1["failure_mode"] == "ACCEPTED" and r1["qv_passed"] is True and r1["eval_verified"] is True and r1["eval_problems"] == []
    # weak: 2 positives (10) - 1 negative (4) => 6/40 = 0.15 ; strong: 8 positives => 40/40 = 1.0 ; gap 0.85
    assert abs(r1["weak_avg"] - 0.15) < 1e-9 and abs(r1["strong_avg"] - 1.0) < 1e-9 and abs(r1["gap"] - 0.85) < 1e-9
    assert r1["weak_scores"] == [0.15, 0.15, 0.15] and r1["strong_scores"] == [1.0, 1.0, 1.0]
    assert r1["n_rubric_items"] == 11 and r1["question_type"] == "failure mode prediction"
    assert r1["reference_answer"] == CHALLENGER_JSON["reference_answer"]
    assert summary["final_qv"]["passed"] is True and summary["final_qv"]["qv_completed"] is True and summary["final_accepted"] is True
    assert summary["agent_claimed_accepted_round"] == 1 and summary["agent_claim_matches_harness"] is True
    assert summary["agent_stop_reason"] == "final" and summary["guardrail_events"] == []
    runs = sorted((wd / "eval_attempts").glob("run_*"))
    assert len(runs) == 2 and all((r / "report.json").exists() for r in runs)
    assert json.loads((wd / "output" / "result.json").read_text())["final_accepted_round"] == 1
    assert (wd / "trajectory" / "main_agent.jsonl").exists() and (wd / "trajectory" / "challenger_01.jsonl").exists()
    assert (wd / "trajectory" / "final_qv_02.jsonl").exists()
    assert sorted(p.name for p in (wd / "subagent_view").iterdir()) == ["paper.txt"]
    # solvers: user-only messages with context + question, never the reference answer
    solver_reqs = [req for role, req in scenario.calls if role == "solver"]
    assert len(solver_reqs) == 6
    assert all(len(req["messages"]) == 1 and req["messages"][0]["role"] == "user" for req in solver_reqs)
    assert all(CHALLENGER_JSON["question"] in req["messages"][0]["content"] for req in solver_reqs)
    assert all(SENTINEL_REF not in json.dumps(req) for req in solver_reqs)
    # the judge sees rubric + response but never the reference answer; 6 judge calls
    judge_reqs = [req for role, req in scenario.calls if role == "judge"]
    assert len(judge_reqs) == 6 and all(SENTINEL_REF not in json.dumps(req) for req in judge_reqs)
    assert all("positive insight 0" in json.dumps(req) for req in judge_reqs)
    # the initial QV received the complete package
    qv_reqs = [req for role, req in scenario.calls if role == "qv"]
    assert any(CHALLENGER_JSON["question"] in json.dumps(req) and "positive insight 7" in json.dumps(req) for req in qv_reqs)


def test_write_allowlist():
    assert write_allowed("eval_input.json") and write_allowed("./eval_input.json") and write_allowed("output/result.json")
    assert write_allowed("/workspace/project/output/result.json") and write_allowed("notes/scratch.md")
    for bad in (".opencode/tools/api_config.json", ".opencode/tools/evaluate_rubric.py", "eval_attempts/run_001_weak-only/report.json",
                "paper.txt", "harness_summary.json", "autodata.py", "autodata/__init__.py", "output/x.py", "sitecustomize.py",
                "../escape.json", "trajectory/main_agent.jsonl", ".hidden", "x.pth",
                # spellings that a naive prefix check would let through (found by the review)
                "/workspace/project/./paper.txt", "paper.txt/", "harness_summary.json/", "/workspace/project/./.opencode/tools/api_config.json",
                "/workspace/project/.//eval_attempts/run_001_both/report.json", "output/../paper.txt", "subagent_view/paper.txt",
                "/workspace/project", "/etc/passwd", "autodata.py/", ".hidden/"):
        assert not write_allowed(bad), bad


def test_guardrails_reject_config_tampering_and_planted_module(tmp_path: Path):
    """The agent tries to (1) relax api_config.json, (2) plant autodata.py in the workspace (would shadow the package
    without -I), (3) drop a forged report into eval_attempts; none of it may take effect."""
    planted = "print('PLANTED-MODULE-EXECUTED')\nraise SystemExit(0)\n"
    tampered = {"acceptance": {"name": "prose_s31", "weak_avg_max": 1.0, "strong_avg_min": 0.0, "gap_min": -1.0}}
    steps = [
        tool_call_response([("write", {"filePath": ".opencode/tools/api_config.json", "content": json.dumps(tampered)})]),
        tool_call_response([("write", {"filePath": "autodata.py", "content": planted})]),
        tool_call_response([("write", {"filePath": "eval_attempts/run_000_weak-only/report.json", "content": "{}"})]),
        tool_call_response([("write", {"filePath": "notes/autodata.py", "content": planted})]),
        *happy_main_steps(),
    ]
    scenario = Scenario(main_steps=steps)
    summary, wd = _run(tmp_path, scenario)
    denied = [e for e in summary["guardrail_events"] if e["kind"] == "write_denied"]
    assert len(denied) == 4, summary["guardrail_events"]
    assert not (wd / "autodata.py").exists() and not (wd / "notes" / "autodata.py").exists()
    assert json.loads((wd / ".opencode" / "tools" / "api_config.json").read_text())["acceptance"]["weak_avg_max"] == 0.5
    assert summary["accepted"] is True and summary["rounds"][0]["eval_verified"] is True
    # even if a module were planted, the isolated interpreter must not execute it
    (wd / "autodata.py").write_text(planted)
    run_dirs_before = set((wd / "eval_attempts").glob("run_*"))
    import subprocess, sys
    out = subprocess.run([sys.executable, "-I", "-m", "autodata.cs.evaluate_rubric", "--help"], cwd=wd, capture_output=True, text=True)
    assert "PLANTED-MODULE-EXECUTED" not in out.stdout + out.stderr and "usage" in (out.stdout + out.stderr).lower()
    assert set((wd / "eval_attempts").glob("run_*")) == run_dirs_before


def test_strong_only_without_weak_is_not_accepted(tmp_path: Path):
    steps = happy_main_steps()
    del steps[3]  # skip the --weak-only run
    steps[3] = tool_call_response([("bash", {"command": STRONG_CMD})])
    scenario = Scenario(main_steps=steps)
    summary, _ = _run(tmp_path, scenario)
    assert summary["accepted"] is False
    r1 = summary["rounds"][0]
    assert r1["strong_avg"] is None and r1["weak_avg"] is None and r1["gap"] is None
    assert not any(role in ("solver", "judge") for role, _ in scenario.calls)
    assert r1["failure_mode"] == "FAILED_ON_STRONG"
    assert summary["agent_claim_matches_harness"] is False  # the agent claimed round 1


def test_qv_contradiction_blocks_acceptance(tmp_path: Path):
    scenario = Scenario(qv_text=QV_CONTRADICTION)
    summary, _ = _run(tmp_path, scenario)
    assert summary["accepted"] is False
    r1 = summary["rounds"][0]
    assert r1["qv_passed"] is False and r1["qv_contradiction"] is True and r1["failure_mode"] == "FAILED_QV"
    assert any(e["kind"] == "accept_without_qv" for e in summary["guardrail_events"])


def test_qv_must_review_the_evaluated_question(tmp_path: Path):
    steps = happy_main_steps()
    steps[1] = tool_call_response([("task", {"description": "QV", "prompt": "Please verify the package (see eval_input.json).", "subagent_type": "quality_verifier"})])
    scenario = Scenario(main_steps=steps)
    summary, _ = _run(tmp_path, scenario)
    assert summary["accepted"] is False
    assert any(e["kind"] == "qv_not_bound" for e in summary["guardrail_events"])


def test_round_budget_and_challenger_refusal(tmp_path: Path):
    call = tool_call_response([("task", {"description": "c", "prompt": "Generate ...", "subagent_type": "challenger"})])
    steps = [call, call, call, call, call, text_response("giving up")]
    scenario = Scenario(main_steps=steps)
    summary, _ = _run(tmp_path, scenario)
    assert summary["n_rounds"] == 3 and summary["accepted"] is False
    assert sum(1 for e in summary["guardrail_events"] if e["kind"] == "round_budget") == 2
    assert all(r["failure_mode"] == "NO_QV" for r in summary["rounds"])


def test_subagents_cannot_read_outside_the_paper_view(tmp_path: Path):
    """Challenger/QV subagents only see subagent_view/paper.txt: reading the agent's result.json, transcripts or
    eval_attempts must fail (paper: the challenger and verifier read the paper, nothing else)."""
    probes = ["cat ../output/result.json", "cat /workspace/project/output/result.json", "ls ..", "cat ../trajectory/main_agent.jsonl",
              "cat ../eval_attempts/run_001_weak-only/report.json", "cat ../.opencode/tools/api_config.json"]
    seen: dict[str, str] = {}

    class Probing(Scenario):
        def __call__(self, req: dict, idx: int) -> dict:
            role = _role(req)
            if role == "challenger":
                msgs = req["messages"]
                tool_msgs = [m for m in msgs if m.get("role") == "tool"]
                n = len(tool_msgs)
                if n < len(probes):
                    if n > 0:
                        seen[probes[n - 1]] = tool_msgs[-1].get("content", "")
                    return tool_call_response([("bash", {"command": probes[n]})])
                if n == len(probes):
                    seen[probes[n - 1]] = tool_msgs[-1].get("content", "")
                    return tool_call_response([("read", {"filePath": "/workspace/project/output/result.json"})])
                seen["read"] = tool_msgs[-1].get("content", "")
                return text_response("## Scratchpad Analysis\n...\n```json\n" + json.dumps(CHALLENGER_JSON) + "\n```")
            return super().__call__(req, idx)

    steps = happy_main_steps()
    steps.insert(0, tool_call_response([("write", {"filePath": "output/result.json", "content": json.dumps({"secret": SENTINEL_REF})})]))
    scenario = Probing(main_steps=steps)
    scenario.main_steps = steps
    summary, wd = _run(tmp_path, scenario, paper_id="p2")
    assert len(seen) == len(probes) + 1, seen.keys()
    for cmd, out in seen.items():
        assert SENTINEL_REF not in out, (cmd, out[:200])
        assert "Error" in out or out.strip() == "" or "not permitted" in out or "outside" in out.lower() or "No such" in out, (cmd, out[:200])
    assert summary["accepted"] is True  # the round itself still completes normally


def test_state_machine_blocks_qv_shopping_reevaluation_and_weight_tampering(tmp_path: Path):
    """Within a round: a second QV on the same question is refused, a completed evaluation is served from cache (no new
    evaluator run), a question that is not the challenger's is refused, rubric weights changed after QV break the binding,
    and nothing can be evaluated after acceptance."""
    tampered = json.loads(json.dumps(EVAL_INPUT))
    tampered["rubric"][0]["weight"] = 10  # was 5: same criterion text, different weight -> QV binding must fail
    foreign = json.loads(json.dumps(EVAL_INPUT))
    foreign["question"] = "A question the challenger never wrote?"
    steps = [
        tool_call_response([("task", {"description": "challenger round 1", "prompt": "Generate ... ./paper.txt", "subagent_type": "challenger"})]),
        tool_call_response([("task", {"description": "QV round 1", "prompt": QV_PROMPT_FULL, "subagent_type": "quality_verifier"})]),
        tool_call_response([("task", {"description": "QV again", "prompt": QV_PROMPT_FULL, "subagent_type": "quality_verifier"})]),   # refused
        tool_call_response([("write", {"filePath": "eval_input.json", "content": json.dumps(tampered)})]),
        tool_call_response([("bash", {"command": WEAK_CMD})]),      # runs, but the candidate is not QV-bound (weight changed)
        tool_call_response([("bash", {"command": STRONG_CMD})]),    # runs; acceptance refused (qv_not_bound)
        tool_call_response([("write", {"filePath": "eval_input.json", "content": json.dumps(foreign)})]),
        tool_call_response([("bash", {"command": WEAK_CMD})]),      # refused: foreign question
        tool_call_response([("write", {"filePath": "eval_input.json", "content": json.dumps(EVAL_INPUT)})]),
        tool_call_response([("bash", {"command": WEAK_CMD})]),      # real evaluation of the QV-bound candidate
        tool_call_response([("bash", {"command": WEAK_CMD})]),      # cached (no new run dir)
        tool_call_response([("bash", {"command": STRONG_CMD})]),    # accepted
        tool_call_response([("bash", {"command": STRONG_CMD})]),    # refused: after acceptance
        tool_call_response([("write", {"filePath": "output/result.json", "content": json.dumps(RESULT_JSON)})]),
        text_response("done"),
    ]
    scenario = Scenario(main_steps=steps)
    summary, wd = _run(tmp_path, scenario, paper_id="p3")
    kinds = [e["kind"] for e in summary["guardrail_events"]]
    assert "qv_repeat" in kinds and "qv_not_bound" in kinds and "eval_foreign_candidate" in kinds
    assert "eval_repeat" in kinds and "eval_after_accept" in kinds, kinds
    assert summary["accepted"] is True and summary["accepted_round"] == 1
    runs = sorted((wd / "eval_attempts").glob("run_*"))
    # tampered weak + tampered strong + real weak + real strong = 4 evaluator runs; cached/refused calls create none
    assert len(runs) == 4, [r.name for r in runs]
    r1 = summary["rounds"][0]
    assert r1["failure_mode"] == "ACCEPTED" and r1["rubric"][0]["weight"] == 5 and abs(r1["weak_avg"] - 0.15) < 1e-9
    # only one QV subagent transcript exists for the round (the repeat was refused before spawning)
    assert len(list((wd / "trajectory").glob("quality_verifier_*.jsonl"))) == 1
