"""One paper through the Agentic Self-Instruct loop (Sec 3.1 / App. C.1).

The LLM main agent drives the loop through tools (task / bash / write / read); this module only
(1) prepares the workspace exactly as the verbatim prompts expect it (./paper.txt, .opencode/tools/evaluate_rubric.py,
    .opencode/tools/api_config.json, ./eval_input.json, ./eval_attempts/, output/result.json),
(2) runs the subagents and the evaluation CLI on the agent's behalf,
(3) records every round from the real tool events and derives the harness-verified acceptance (guardrail: the agent
    cannot accept by claiming; evaluation provenance, the acceptance preset and the scores are re-verified here), and
(4) runs the end-of-loop quality verifier on the accepted (frozen) item.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from autodata.config import AcceptancePreset, AppConfig
from autodata.cs.evaluate_rubric import compute_question_hash
from autodata.cs.parsing import parse_challenger_output, parse_qv_output
from autodata.cs.rubric import parse_rubric, score_response
from autodata.cs.prompts import PromptSet
from autodata.harness.agent import Agent, AgentEvent, AgentResult
from autodata.harness.tools import Workspace, make_bash_tool, make_read_tool, make_task_tool, make_write_tool
from autodata.llm.client import LLMClient

HARNESS_VERSION = "0.2.0"

EVALUATE_RUBRIC_SHIM = '''#!/usr/bin/env python3
"""Shim: lets the verbatim command `uv run python3 .opencode/tools/evaluate_rubric.py ...` run the real evaluator."""
import sys
from autodata.cs.evaluate_rubric import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''

SUBAGENT_TYPES = ("challenger", "quality_verifier")
API_CONFIG_REL = ".opencode/tools/api_config.json"
EVAL_DIR_REL = "eval_attempts"
SUBAGENT_VIEW_REL = "subagent_view"   # subagents (challenger / QV) only ever see ./paper.txt, as in the paper
VIRTUAL_ROOT = "/workspace/project"
_REPORT_PATH_RE = re.compile(r"^REPORT_PATH:\s*(.+?)\s*$", re.M)


# ----------------------------------------------------------------------------- data classes
@dataclass
class PaperInput:
    paper_id: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundRecord:
    index: int
    started_at: float
    challenger_prompt: str = ""
    challenger_output: str = ""
    challenger_json: dict[str, Any] | None = None
    qv_calls: list[dict[str, Any]] = field(default_factory=list)
    evals: list[dict[str, Any]] = field(default_factory=list)

    @property
    def qv_passed(self) -> bool | None:
        if not self.qv_calls:
            return None
        return self.qv_calls[-1]["parsed"].get("overall")

    def last_eval(self, mode: str) -> dict[str, Any] | None:
        for ev in reversed(self.evals):
            if ev.get("report") and ev["mode"] == mode:
                return ev
        return None

    def last_candidate(self) -> dict[str, Any] | None:
        for ev in reversed(self.evals):
            if ev.get("candidate"):
                return ev["candidate"]
        return None


# ----------------------------------------------------------------------------- helpers
def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _alnum(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def canonical_rel(path: str) -> str | None:
    """Canonical workspace-relative POSIX path for an agent-supplied path, or None if it cannot be inside the workspace.
    Handles the virtual root prefix, ./ segments, doubled and trailing slashes exactly like Workspace.resolve does."""
    p = str(path).replace("\\", "/").strip()
    if p == VIRTUAL_ROOT:
        return None
    if p.startswith(VIRTUAL_ROOT + "/"):
        p = p[len(VIRTUAL_ROOT) + 1:]
    if not p or p.startswith("/"):
        return None
    p = posixpath.normpath(p)
    if p in (".", "..") or p.startswith("../"):
        return None
    return p


def write_allowed(rel_path: str) -> bool:
    """Which files the AGENT may write. Only data files: eval_input.json, output/*, notes, top-level .md/.txt.
    Never the evaluator, its config, its outputs, the paper, transcripts, the subagent view, or anything importable."""
    p = canonical_rel(rel_path)
    if p is None:
        return False
    if p.startswith((".opencode/", EVAL_DIR_REL + "/", "trajectory/", SUBAGENT_VIEW_REL + "/")):
        return False
    if p in ("paper.txt", "harness_summary.json", API_CONFIG_REL, SUBAGENT_VIEW_REL, ".opencode", EVAL_DIR_REL, "trajectory"):
        return False
    base = p.split("/")[-1]
    if base.startswith((".", "autodata", "sitecustomize", "usercustomize")):
        return False
    if base.endswith((".py", ".pyc", ".pth", ".so", ".sh", ".egg-link")):
        return False
    return True


def prepare_workspace(cfg: AppConfig, paper: PaperInput, workdir: Path, prompts_dir_abs: Path) -> Workspace:
    workdir.mkdir(parents=True, exist_ok=True)
    text = paper.text
    if len(text) > cfg.run.paper_text_max_chars:
        text = text[: cfg.run.paper_text_max_chars] + "\n\n[paper text truncated by the harness]\n"
    (workdir / "paper.txt").write_text(text, encoding="utf-8")
    tools_dir = workdir / ".opencode" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "evaluate_rubric.py").write_text(EVALUATE_RUBRIC_SHIM, encoding="utf-8")
    api_config = {
        "weak_solver": cfg.endpoint("weak_solver").model_dump(),
        "strong_solver": cfg.endpoint("strong_solver").model_dump(),
        "judge": cfg.endpoint("judge").model_dump(),
        "acceptance": cfg.acceptance.model_dump(),
        "eval": cfg.eval.model_dump(),
        "prompts_dir": str(prompts_dir_abs),
    }
    (tools_dir / "api_config.json").write_text(json.dumps(api_config, indent=2), encoding="utf-8")
    for d in ("output", EVAL_DIR_REL, "trajectory", SUBAGENT_VIEW_REL):
        (workdir / d).mkdir(exist_ok=True)
    # Subagents may read nothing but the paper (the prompts say `cat ./paper.txt`); they get their own root.
    shutil.copyfile(workdir / "paper.txt", workdir / SUBAGENT_VIEW_REL / "paper.txt")
    return Workspace(workdir)


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def config_fingerprint(cfg: AppConfig) -> str:
    return _sha1_bytes(cfg.model_dump_json().encode("utf-8"))[:16]


async def run_evaluator(workdir: Path, argv_abs: list[str], *, deadline_s: float) -> tuple[str, str, int | None]:
    """Run the isolated evaluator, killing and reaping it on timeout or cancellation.

    Arguments must already contain validated absolute paths. Timeouts and launch
    errors propagate so each caller can retain its own diagnostics.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONUNBUFFERED"] = "1"
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-m", "autodata.cs.evaluate_rubric", *argv_abs,
            cwd=str(workdir), env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=deadline_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
    except asyncio.CancelledError:
        if proc is not None and proc.returncode is None:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
        raise
    return out_b.decode("utf-8", errors="replace"), err_b.decode("utf-8", errors="replace"), proc.returncode


def evaluator_deadline_s(cfg: AppConfig, mode: str, solver_timeout_s: float) -> float:
    judge_t = float(cfg.endpoint("judge").timeout_s)
    stage = solver_timeout_s * (cfg.eval.solver_retries + 1) + judge_t * (cfg.eval.judge_retries + 1)
    return stage * (2 if mode == "both" else 1) + 300.0


def final_filter(cfg: AppConfig, context: str, rubric: Any) -> dict:
    """Programmatic final checks shared by the agentic and CoT pipelines (Sec. 3.1)."""
    context = context or ""
    out: dict = {"context_chars": len(context), "n_rubric_items": len(rubric) if isinstance(rubric, list) else None,
                 "context_ok": len(context) >= cfg.run.final_min_context_chars, "rubric_ok": False, "rubric_error": None}
    try:
        items = parse_rubric(rubric)
        n_pos = sum(1 for it in items if it.weight > 0)
        n_neg = sum(1 for it in items if it.weight < 0)
        weights_ok = all(1 <= abs(it.weight) <= 10 for it in items)  # challenger spec (Fig. 8): +1..+10 / -1..-10
        # Defaults also support RunConfig versions predating these shape knobs.
        ok = weights_ok and (getattr(cfg.run, "final_rubric_min_items", 10) <= len(items)
              <= getattr(cfg.run, "final_rubric_max_items", 20)
              and n_pos >= getattr(cfg.run, "final_rubric_min_positive", 4)
              and n_neg >= getattr(cfg.run, "final_rubric_min_negative", 3))
        out.update({"n_positive": n_pos, "n_negative": n_neg, "rubric_ok": ok})
    except Exception as e:
        out["rubric_error"] = repr(e)
    return out


def harness_predicate(preset: AcceptancePreset, weak: list[Fraction] | None,
                      strong: list[Fraction] | None) -> dict[str, Any]:
    """Independent implementation of the acceptance predicate (exact arithmetic), mirroring evaluate_rubric.py."""
    out: dict[str, Any] = {"weak_avg": None, "strong_avg": None, "gap": None, "weak_passed": None,
                           "strong_passed": None, "gap_passed": None, "all_passed": False, "reasons": []}
    if weak:
        avg = sum(weak) / len(weak)
        out["weak_avg"] = float(avg)
        lim = Fraction(str(preset.weak_avg_max))
        ok = (avg <= lim) if preset.weak_avg_max_inclusive else (avg < lim)
        if not ok:
            out["reasons"].append("TOO EASY")
        if preset.weak_attempt_max is not None and max(weak) > Fraction(str(preset.weak_attempt_max)):
            ok = False
            out["reasons"].append("max_weak")
        if preset.weak_no_zero and min(weak) == 0:
            ok = False
            out["reasons"].append("zero weak attempt")
        out["weak_passed"] = ok
    if strong:
        avg = sum(strong) / len(strong)
        out["strong_avg"] = float(avg)
        ok = avg >= Fraction(str(preset.strong_avg_min))
        if not ok:
            out["reasons"].append("strong too low")
        if preset.strong_avg_max is not None and avg >= Fraction(str(preset.strong_avg_max)):
            ok = False
            out["reasons"].append("strong saturated")
        if preset.strong_no_zero and min(strong) == 0:
            ok = False
            out["reasons"].append("zero strong attempt")
        out["strong_passed"] = ok
    if weak and strong:
        gap = sum(strong) / len(strong) - sum(weak) / len(weak)
        out["gap"] = float(gap)
        out["gap_passed"] = gap >= Fraction(str(preset.gap_min))
        if not out["gap_passed"]:
            out["reasons"].append("gap too small")
    out["all_passed"] = bool(out["weak_passed"] and out["strong_passed"] and out["gap_passed"])
    return out


def _attempt_scores(rubric_items: list[Any], attempts: list[dict[str, Any]] | None) -> list[Fraction] | None:
    """Recompute each attempt's score from its `satisfied` list (never trust the stored score)."""
    if not attempts:
        return None
    out: list[Fraction] = []
    for a in attempts:
        sat = a.get("satisfied")
        if a.get("error") or not isinstance(sat, list) or len(sat) != len(rubric_items) or not all(isinstance(x, bool) for x in sat):
            raise ValueError("attempt without a complete boolean satisfied[] list")
        b = score_response(rubric_items, list(sat))
        out.append(max(Fraction(0), min(Fraction(1), Fraction(b.earned - b.penalty, b.max_positive))))
    return out


# ----------------------------------------------------------------------------- the run
class PaperRun:
    """Runs the main agent for one paper and records the loop."""

    def __init__(self, cfg: AppConfig, paper: PaperInput, workdir: Path, clients: dict[str, LLMClient],
                 prompts: PromptSet, prompts_dir_abs: Path, *, log=print):
        self.cfg = cfg
        self.paper = paper
        self.workdir = Path(workdir)
        self.clients = clients
        self.prompts = prompts
        self.prompts_dir_abs = Path(prompts_dir_abs)
        self.log = log
        self.ws: Workspace | None = None
        self.rounds: list[RoundRecord] = []
        self.subagent_counter: dict[str, int] = {t: 0 for t in SUBAGENT_TYPES}
        self.subagent_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.accepted_round_idx: int | None = None  # harness-verified
        self.accepted: dict[str, Any] | None = None  # frozen candidate + verdict
        self.guardrail_events: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.n_eval_calls = 0
        self.api_config_sha1: str | None = None

    # ------------------------------------------------------------------ helpers
    def _current_round(self) -> RoundRecord:
        if self.rounds:
            return self.rounds[-1]
        rec = RoundRecord(index=1, started_at=time.time())
        rec.challenger_output = "[no challenger call recorded before this event]"
        self.rounds.append(rec)
        return rec

    def _make_subagent(self, subagent_type: str, n: int, *, name: str | None = None) -> Agent:
        assert self.ws is not None
        system = self.prompts.challenger_system() if subagent_type == "challenger" else self.prompts.quality_verifier_system()
        view = Workspace(self.workdir / SUBAGENT_VIEW_REL)  # only ./paper.txt is visible to subagents
        tools = [make_bash_tool(view), make_read_tool(view)]
        name = name or f"{subagent_type}_{n:02d}"
        return Agent(
            name=name,
            system_prompt=system,
            tools=tools,
            llm=self.clients[subagent_type],
            max_steps=self.cfg.run.subagent_max_steps,
            transcript_path=self.workdir / "trajectory" / f"{name}.jsonl",
            max_model_len=getattr(self.cfg.run, "max_model_len", None),
        )

    def _account(self, result: AgentResult) -> None:
        u = result.usage or {}
        self.subagent_usage["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        self.subagent_usage["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
        self.subagent_usage["calls"] += int(u.get("calls", result.steps_used) or 0)

    def _guard(self, kind: str, detail: str, **extra: Any) -> None:
        ev = {"kind": kind, "detail": detail, "round": self.rounds[-1].index if self.rounds else None,
              "ts": time.time(), **extra}
        self.guardrail_events.append(ev)
        self.log(f"[{self.paper.paper_id}] guardrail {kind}: {detail}")

    # ------------------------------------------------------------------ tool backends
    async def run_subagent(self, subagent_type: str, description: str, prompt: str) -> str:
        if subagent_type not in SUBAGENT_TYPES:
            return f"Error: unknown subagent_type {subagent_type!r}; allowed: {list(SUBAGENT_TYPES)}"
        if subagent_type == "challenger":
            if self.accepted_round_idx is not None:
                self._guard("challenger_after_accept", "challenger call refused after acceptance")
                return (f"A question was already ACCEPTED in round {self.accepted_round_idx}. Do not call the challenger "
                        "again: write the final output/result.json (with final_accepted_round set) and stop.")
            if len(self.rounds) >= self.cfg.run.max_rounds:
                self._guard("round_budget", f"challenger call refused: {self.cfg.run.max_rounds} rounds used")
                return (f"ROUND BUDGET EXHAUSTED: all {self.cfg.run.max_rounds} challenger rounds for this paper have been "
                        "used. Do not call the challenger again. Write the final output/result.json now (accepted=false "
                        "for every round unless an earlier round was accepted; include ALL rounds attempted) and stop.")
            rec = RoundRecord(index=len(self.rounds) + 1, started_at=time.time(), challenger_prompt=prompt)
            self.rounds.append(rec)
            self.log(f"[{self.paper.paper_id}] round {rec.index}: challenger")
        if subagent_type == "quality_verifier" and self.rounds:
            rec0 = self.rounds[-1]
            q_alnum = _alnum((rec0.challenger_json or {}).get("question"))
            for prev in rec0.qv_calls:
                if prev.get("stop_reason") == "final" and q_alnum and q_alnum in prev.get("prompt_alnum", "") and q_alnum in _alnum(prompt):
                    verdict = "PASS" if prev["parsed"].get("overall") else "FAIL"
                    self._guard("qv_repeat", f"repeated QV on the same question refused (previous verdict {verdict})")
                    return (f"Quality verification for this question was already completed in this round with OVERALL: {verdict}. "
                            "Do not repeat it: " + ("proceed to evaluation." if verdict == "PASS" else
                            "add it to the failed quality check list and ask the challenger for an ENTIRELY NEW question.")
                            + "\n\n[previous verifier output]\n" + prev["output"][-4000:])
        self.subagent_counter[subagent_type] += 1
        n = self.subagent_counter[subagent_type]
        agent = self._make_subagent(subagent_type, n)
        try:
            result = await agent.run(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # never propagate into the main agent loop
            self.errors.append(f"{subagent_type}_{n}: {e!r}")
            return f"Error: subagent {subagent_type} failed: {e!r}"
        self._account(result)
        text = result.final_text or ""
        if result.stop_reason != "final":
            text += f"\n\n[subagent stopped early: {result.stop_reason}" + (f" ({result.error})" if result.error else "") + "]"
        if subagent_type == "challenger":
            rec = self.rounds[-1]
            rec.challenger_output = text
            rec.challenger_json = parse_challenger_output(text)
        else:
            rec = self._current_round()
            parsed = parse_qv_output(text)
            if result.stop_reason != "final":
                parsed = {**parsed, "overall": False, "incomplete": True}
            rec.qv_calls.append({"prompt": prompt, "output": text, "parsed": parsed, "description": description,
                                 "stop_reason": result.stop_reason, "prompt_alnum": _alnum(prompt)})
            self.log(f"[{self.paper.paper_id}] round {rec.index}: QV -> {parsed.get('overall')}"
                     + (" (contradiction)" if parsed.get("contradiction") else ""))
        return text

    # ---- evaluator argv handling ----
    def _parse_eval_argv(self, argv: list[str]) -> tuple[dict[str, str], str, list[str]]:
        assert self.ws is not None
        opts: dict[str, str] = {}
        flags: list[str] = []
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok in ("--input", "--output-dir", "--config", "--timeout"):
                if i + 1 >= len(argv):
                    raise ValueError(f"missing value for {tok}")
                opts[tok] = argv[i + 1]
                i += 2
            elif tok in ("--weak-only", "--strong-only"):
                flags.append(tok)
                i += 1
            elif tok.startswith("--input=") or tok.startswith("--output-dir=") or tok.startswith("--config=") or tok.startswith("--timeout="):
                k, v = tok.split("=", 1)
                opts[k] = v
                i += 1
            else:
                raise ValueError(f"unexpected evaluator argument {tok!r}")
        mode = "weak-only" if "--weak-only" in flags else "strong-only" if "--strong-only" in flags else "both"
        if len(flags) > 1:
            raise ValueError("use either --weak-only or --strong-only")
        input_rel = opts.get("--input", "./eval_input.json")
        out_rel = opts.get("--output-dir", f"./{EVAL_DIR_REL}")
        cfg_rel = opts.get("--config", API_CONFIG_REL)
        input_path = self.ws.resolve(input_rel)
        out_path = self.ws.resolve(out_rel)
        cfg_path = self.ws.resolve(cfg_rel)
        if cfg_path != (self.workdir / API_CONFIG_REL).resolve():
            raise ValueError(f"--config must be {API_CONFIG_REL}")
        if out_path != (self.workdir / EVAL_DIR_REL).resolve():
            raise ValueError(f"--output-dir must be ./{EVAL_DIR_REL}")
        if not input_path.is_file():
            raise ValueError(f"input file not found: {input_rel} (write it with the write tool first)")
        timeout = opts.get("--timeout", str(self.cfg.eval.timeout_s))
        try:
            timeout_s = float(timeout)
        except ValueError as e:
            raise ValueError(f"invalid --timeout {timeout!r}") from e
        if not (0 < timeout_s <= float(self.cfg.eval.timeout_s)):
            # the agent may not extend the per-request timeout beyond the run config (it drives the harness deadline)
            timeout_s = float(self.cfg.eval.timeout_s)
        argv_abs = ["--input", str(input_path), "--output-dir", str(out_path), "--config", str(cfg_path),
                    "--timeout", str(timeout_s), *flags]
        return {"input": str(input_path), "timeout_s": str(timeout_s)}, mode, argv_abs

    def _verify_report(self, report: dict[str, Any], candidate: dict[str, Any], input_sha1: str, rc: int | None,
                       report_path: Path, new_dirs: set[Path], *, mode: str = "both",
                       rec: RoundRecord | None = None) -> tuple[dict[str, Any], list[str]]:
        """Provenance + independent recomputation. Returns (harness_verdict, problems)."""
        problems: list[str] = []
        expected_models = {role: self.cfg.endpoint(role).model for role in ("weak_solver", "strong_solver", "judge")}
        models = report.get("models") or {}
        for role, name in expected_models.items():
            if models.get(role) not in (None, name):
                problems.append(f"{role} model in report ({models.get(role)!r}) differs from the run config ({name!r})")
        if report.get("prompts_dir") not in (None, str(self.prompts_dir_abs)):
            problems.append("prompts_dir in report differs from the run config")
        if self.api_config_sha1 and report.get("config_sha1") not in (None, self.api_config_sha1):
            problems.append("evaluator config differs from the harness-written api_config.json")
        if mode == "strong-only":
            src = report.get("weak_source_report")
            qh = report.get("question_hash")
            verified_weak = []
            for ev in (rec.evals if rec is not None else []):
                if ev.get("mode") == "weak-only" and ev.get("verified") and ev.get("question_hash") == qh and ev.get("report_path"):
                    verified_weak.append(Path(ev["report_path"]).resolve())
            if not src or Path(src).resolve() not in verified_weak:
                problems.append("weak result provenance: strong-only report does not reuse a weak run verified in this round")
        try:
            rp = report_path.resolve()
            if not rp.is_relative_to((self.workdir / EVAL_DIR_REL).resolve()):
                problems.append("report outside eval_attempts")
            if rp.parent not in new_dirs:
                problems.append("report dir was not created by this evaluator call")
        except Exception as e:
            problems.append(f"report path check failed: {e!r}")
        if rc != 0:
            problems.append(f"evaluator exit code {rc}")
        if report.get("error"):
            problems.append(f"evaluator error: {report['error']}")
        if report.get("question_hash") != compute_question_hash(candidate):
            problems.append("question_hash mismatch")
        if report.get("input_sha1") not in (None, input_sha1):
            problems.append("input_sha1 mismatch")
        mine = self.cfg.acceptance.model_dump()
        theirs = report.get("acceptance") or {}
        if any(theirs.get(k) != v for k, v in mine.items()):
            problems.append("acceptance preset in report differs from the run config")
        n_req = self.cfg.eval.n_attempts
        if (report.get("eval") or {}).get("n_attempts") != n_req:
            problems.append("n_attempts in report differs from the run config")
        if not self.rubric_weights_in_bounds(candidate.get("rubric")):
            problems.append("rubric weights outside the challenger spec (|weight| must be 1..10)")
        verdict: dict[str, Any] = harness_predicate(self.cfg.acceptance, None, None)
        try:
            items = parse_rubric(candidate.get("rubric"))
            weak = _attempt_scores(items, report.get("weak_attempts"))
            strong = _attempt_scores(items, report.get("strong_attempts"))
            for role, sc in (("weak", weak), ("strong", strong)):
                if sc is not None and len(sc) != n_req:
                    problems.append(f"{role}: {len(sc)} attempts, {n_req} required")
            verdict = harness_predicate(self.cfg.acceptance, weak, strong)
            verdict["weak_scores"] = [float(x) for x in weak] if weak else None
            verdict["strong_scores"] = [float(x) for x in strong] if strong else None
            for k in ("weak_passed", "strong_passed", "gap_passed", "all_passed"):
                if report.get(k) is not None and bool(report.get(k)) != bool(verdict.get(k)):
                    problems.append(f"verdict mismatch on {k}: evaluator={report.get(k)} harness={verdict.get(k)}")
        except Exception as e:
            problems.append(f"recomputation failed: {e!r}")
        return verdict, problems

    def _qv_bound(self, rec: RoundRecord, candidate: dict[str, Any]) -> bool:
        """The round's (last) QV call must have reviewed THIS candidate: the question, the head of the context and every
        rubric criterion WITH ITS WEIGHT must appear in the QV prompt (compared on alphanumerics only, so JSON escaping
        does not matter; the weight must sit within 40 characters of its criterion, as in a JSON item or a bullet)."""
        if not rec.qv_calls:
            return False
        prompt = _alnum(rec.qv_calls[-1]["prompt"])
        q = _alnum(candidate.get("question"))
        if not q or q not in prompt:
            return False
        ctx = _alnum(candidate.get("context"))[:200]
        if ctx and ctx not in prompt:
            return False
        for item in candidate.get("rubric") or []:
            if not isinstance(item, dict):
                return False
            crit = _alnum(item.get("criterion"))[:80]
            if not crit:
                return False
            pos = prompt.find(crit)
            if pos < 0:
                return False
            weight = str(abs(int(item.get("weight", 0)))) if str(item.get("weight", "")).lstrip("+-").isdigit() else ""
            window = prompt[max(0, pos - 40):pos] + prompt[pos + len(crit):pos + len(crit) + 40]
            if not weight or weight not in window:
                return False
        return True

    @staticmethod
    def rubric_weights_in_bounds(rubric: Any, lo: int = 1, hi: int = 10) -> bool:
        """Challenger spec (Fig. 8): positive weights +1..+10, negative weights -1..-10."""
        if not isinstance(rubric, list) or not rubric:
            return False
        for item in rubric:
            try:
                w = int(item.get("weight"))
            except Exception:
                return False
            if not (lo <= abs(w) <= hi):
                return False
        return True

    async def run_evaluate_rubric(self, argv: list[str]) -> str:
        assert self.ws is not None
        self.n_eval_calls += 1
        rec = self._current_round()
        try:
            opts, mode, argv_abs = self._parse_eval_argv(argv)
        except (ValueError, PermissionError) as e:
            self._guard("bad_eval_args", str(e), argv=argv)
            return f"Error: {e}. Usage: evaluate_rubric.py --input ./eval_input.json [--weak-only|--strong-only] --output-dir ./{EVAL_DIR_REL} --config {API_CONFIG_REL} --timeout 600"
        input_path = Path(opts["input"])
        raw = input_path.read_bytes()
        input_sha1 = _sha1_bytes(raw)
        try:
            candidate = json.loads(raw.decode("utf-8"))
            if not isinstance(candidate, dict):
                raise ValueError("input is not a JSON object")
        except Exception as e:
            self._guard("bad_eval_input", f"{e!r}")
            rec.evals.append({"mode": mode, "argv": argv, "stdout": f"INPUT_ERROR: {e}", "report": None,
                              "candidate": None, "input_sha1": input_sha1, "returncode": None, "elapsed_s": 0.0})
            return f"INPUT_ERROR: eval_input.json is not valid JSON ({e}). Rewrite it with the write tool."
        qh = compute_question_hash(candidate)
        if self.accepted_round_idx is not None:
            self._guard("eval_after_accept", "evaluation refused after acceptance")
            return (f"A question was already ACCEPTED in round {self.accepted_round_idx}. No further evaluations: write the final "
                    "output/result.json (with final_accepted_round set) and stop.")
        # the evaluated question must be the challenger's question for this round (the main agent never writes questions)
        cj = rec.challenger_json or {}
        cand_q = _alnum(candidate.get("question"))
        origin_ok = bool(cand_q) and (
            (cj.get("question") and _alnum(cj.get("question")) == cand_q) or
            (not cj.get("question") and cand_q in _alnum(rec.challenger_output)))
        if not origin_ok:
            self._guard("eval_foreign_candidate", "evaluation refused: the question is not this round's challenger output")
            return ("Error: the question in eval_input.json is not the challenger's question for this round. Only the "
                    "challenger writes questions: call the challenger (a new round) and evaluate its output unchanged.")
        for prev in reversed(rec.evals):
            if prev.get("question_hash") == qh and prev.get("mode") == mode and prev.get("verified"):
                self._guard("eval_repeat", f"repeated {mode} evaluation of the same question served from cache")
                return prev["stdout"] + "\n[cached: this question was already evaluated in this mode in this round; a completed " \
                                        "result is final - to change the outcome, ask the challenger for an ENTIRELY NEW question]"
        eval_dir = (self.workdir / EVAL_DIR_REL).resolve()
        before = {p.resolve() for p in eval_dir.glob("run_*")}
        deadline = evaluator_deadline_s(self.cfg, mode, float(opts["timeout_s"]))
        t0 = time.time()
        try:
            stdout, stderr, rc = await run_evaluator(self.workdir, argv_abs, deadline_s=deadline)
        except asyncio.TimeoutError:
            msg = f"SOLVER_ERROR: evaluate_rubric.py exceeded the harness deadline of {deadline:.0f}s (infrastructure issue; retry the evaluation)"
            rec.evals.append({"mode": mode, "argv": argv, "stdout": msg, "report": None, "candidate": candidate,
                              "input_sha1": input_sha1, "elapsed_s": time.time() - t0, "returncode": None})
            return msg
        except Exception as e:
            msg = f"SOLVER_ERROR: could not run evaluate_rubric.py ({e!r}); retry the evaluation"
            self.errors.append(msg)
            rec.evals.append({"mode": mode, "argv": argv, "stdout": msg, "report": None, "candidate": candidate,
                              "input_sha1": input_sha1, "elapsed_s": time.time() - t0, "returncode": None})
            return msg
        after = {p.resolve() for p in eval_dir.glob("run_*")}
        new_dirs = after - before
        report = None
        report_path = None
        m = _REPORT_PATH_RE.search(stdout)
        if m:
            report_path = Path(m.group(1))
            report = _read_json_file(report_path)
        if not stdout.strip():
            stdout = (f"SOLVER_ERROR: evaluate_rubric.py exited with code {rc} and produced no report "
                      f"(stderr tail: {stderr[-800:]!r}). Retry the evaluation.")
        entry: dict[str, Any] = {"mode": mode, "argv": argv, "stdout": stdout[-20000:], "stderr_tail": stderr[-2000:],
                                 "report": report, "report_path": str(report_path) if report_path else None,
                                 "candidate": candidate, "input_sha1": input_sha1, "question_hash": compute_question_hash(candidate),
                                 "elapsed_s": time.time() - t0, "returncode": rc, "verified": False, "problems": [], "verdict": None}
        rec.evals.append(entry)
        if report and report_path:
            verdict, problems = self._verify_report(report, candidate, input_sha1, rc, report_path, new_dirs, mode=mode, rec=rec)
            entry["verdict"] = verdict
            entry["problems"] = problems
            entry["verified"] = not problems
            self.log(f"[{self.paper.paper_id}] round {rec.index}: eval {mode} -> weak_avg={verdict.get('weak_avg')} "
                     f"strong_avg={verdict.get('strong_avg')} gap={verdict.get('gap')} all_passed={verdict.get('all_passed')}"
                     + (f" PROBLEMS={problems}" if problems else ""))
            if problems:
                self._guard("unverified_report", "; ".join(problems), mode=mode)
            if verdict.get("all_passed") and not problems and self.accepted_round_idx is None:
                if not rec.qv_passed:
                    self._guard("accept_without_qv", "solver criteria passed but the round has no passing QV; not accepted")
                elif not self._qv_bound(rec, candidate):
                    self._guard("qv_not_bound", "solver criteria passed but the QV call did not review this question; not accepted")
                else:
                    self.accepted_round_idx = rec.index
                    ref = None
                    cj = rec.challenger_json or {}
                    if _norm(cj.get("question"))[:120] == _norm(candidate.get("question"))[:120]:
                        ref = cj.get("reference_answer")
                    self.accepted = {"round": rec.index, "candidate": candidate, "reference_answer": ref,
                                     "verdict": verdict, "report_path": str(report_path), "input_sha1": input_sha1,
                                     "question_hash": entry["question_hash"], "qv": rec.qv_calls[-1]["parsed"]}
                    self.log(f"[{self.paper.paper_id}] round {rec.index}: ACCEPTED (harness-verified)")
        return stdout

    def _on_event(self, ev: AgentEvent) -> None:  # hook for the main agent (kept light; transcripts hold the details)
        pass

    # ------------------------------------------------------------------ end-of-loop QV
    async def run_final_qv(self) -> dict[str, Any] | None:
        if self.accepted is None:
            return None
        cand = self.accepted["candidate"]
        context = cand.get("context") or ""
        question = cand.get("question") or ""
        rubric = cand.get("rubric") or []
        qtype = str(cand.get("question_type") or (self.rounds[self.accepted["round"] - 1].challenger_json or {}).get("question_type") or "")
        programmatic = final_filter(self.cfg, context, rubric)
        prompt = self.prompts.qv_request(qtype, context, question, json.dumps(rubric, ensure_ascii=False, indent=1))
        self.subagent_counter["quality_verifier"] += 1
        n = self.subagent_counter["quality_verifier"]
        agent = self._make_subagent("quality_verifier", n, name=f"final_qv_{n:02d}")
        try:
            result = await agent.run(prompt)
            self._account(result)
            parsed = parse_qv_output(result.final_text or "")
            text = result.final_text or ""
            completed = result.stop_reason == "final"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.errors.append(f"final_qv: {e!r}")
            parsed, text, completed = {"overall": None, "checks": {}, "feedback": None}, f"error: {e!r}", False
        passed = bool(parsed.get("overall")) and completed and programmatic["context_ok"] and programmatic["rubric_ok"]
        return {"passed": passed, "qv": parsed, "qv_completed": completed, "qv_output": text[-6000:], "programmatic": programmatic}

    async def run_final_qv_only(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Repair path (resume): the paper was accepted but its end-of-loop QV never completed (infrastructure failure).
        Re-runs ONLY the final QV on the frozen accepted candidate recorded in the summary; returns the fields to merge."""
        t0 = time.time()
        idx = summary.get("accepted_round")
        rounds = summary.get("rounds") or []
        if not summary.get("accepted") or not idx or idx > len(rounds):
            raise ValueError("summary has no accepted round to repair")
        rd = rounds[idx - 1]
        self.ws = prepare_workspace(self.cfg, self.paper, self.workdir, self.prompts_dir_abs)
        self.subagent_counter["quality_verifier"] = sum(1 for _ in (self.workdir / "trajectory").glob("*qv*.jsonl")) + 1
        candidate = {k: rd.get(k) for k in ("question_type", "context", "question", "rubric")}
        self.accepted_round_idx = idx
        self.accepted = {"round": idx, "candidate": candidate, "reference_answer": rd.get("reference_answer"),
                         "verdict": {k: rd.get(k) for k in ("weak_avg", "strong_avg", "gap", "weak_passed", "strong_passed",
                                                             "gap_passed", "weak_scores", "strong_scores")} | {"all_passed": True}}
        final_qv = await self.run_final_qv()
        errors = list(summary.get("errors") or []) + self.errors
        return {"final_qv": final_qv, "final_accepted": bool(final_qv and final_qv.get("passed")),
                "final_qv_repaired_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "errors": errors,
                "wall_time_s": round(float(summary.get("wall_time_s") or 0) + time.time() - t0, 1)}

    # ------------------------------------------------------------------ summary
    def _round_summary(self, rec: RoundRecord) -> dict[str, Any]:
        strong_ev = rec.last_eval("strong-only") or rec.last_eval("both")
        weak_ev = None
        if strong_ev is not None:  # the weak evaluation of the SAME candidate, if any
            for ev in reversed(rec.evals):
                if ev.get("report") and ev["mode"] in ("weak-only", "both") and ev.get("question_hash") == strong_ev.get("question_hash"):
                    weak_ev = ev
                    break
        if weak_ev is None:
            weak_ev = rec.last_eval("weak-only") or rec.last_eval("both")
        weak = (weak_ev or {}).get("verdict") or {}
        strong = (strong_ev or {}).get("verdict") or {}
        cj = rec.challenger_json or {}
        if rec.index == self.accepted_round_idx and self.accepted:
            cand = self.accepted["candidate"]
            ref = self.accepted.get("reference_answer")
            frozen = self.accepted.get("verdict") or {}
            weak, strong = frozen, frozen  # scores of the accepted candidate, never a later evaluation
        else:
            cand = rec.last_candidate() or {}
            ref = cj.get("reference_answer") if _norm(cj.get("question"))[:120] == _norm(cand.get("question"))[:120] else None
        question = cand.get("question") or cj.get("question")
        context = cand.get("context") or cj.get("context")
        rubric = cand.get("rubric") or cj.get("rubric")
        if rec.index == self.accepted_round_idx:
            mode = "ACCEPTED"
        elif rec.qv_passed is False:
            mode = "FAILED_QV"
        elif weak and weak.get("weak_passed") is False:
            mode = "TOO_EASY"
        elif strong and strong.get("all_passed") is False:
            mode = "FAILED_ON_STRONG"
        elif rec.qv_passed is None and not rec.evals:
            mode = "NO_QV"
        else:
            mode = "NO_EVAL"
        return {
            "index": rec.index,
            "failure_mode": mode,
            "qv_passed": rec.qv_passed,
            "qv_contradiction": bool(rec.qv_calls and rec.qv_calls[-1]["parsed"].get("contradiction")),
            "n_qv_calls": len(rec.qv_calls),
            "n_eval_calls": len(rec.evals),
            "eval_verified": bool((strong_ev or weak_ev or {}).get("verified")),
            "eval_problems": ((strong_ev or weak_ev or {}).get("problems") or []),
            "weak_avg": weak.get("weak_avg"),
            "weak_scores": weak.get("weak_scores"),
            "strong_avg": strong.get("strong_avg"),
            "strong_scores": strong.get("strong_scores"),
            "gap": strong.get("gap"),
            "weak_passed": weak.get("weak_passed"),
            "strong_passed": strong.get("strong_passed"),
            "gap_passed": strong.get("gap_passed"),
            "question_type": cand.get("question_type") or cj.get("question_type"),
            "reasoning_skills": cj.get("reasoning_skills"),
            "question": question,
            "context": context,
            "reference_answer": ref,
            "rubric": rubric,
            "n_rubric_items": len(rubric) if isinstance(rubric, list) else None,
            "question_chars": len(question) if isinstance(question, str) else None,
            "challenger_parsed": rec.challenger_json is not None,
            "qv_feedback": rec.qv_calls[-1]["parsed"].get("feedback") if rec.qv_calls else None,
            "started_at": rec.started_at,
        }

    def summary(self, result: AgentResult | None, final_qv: dict[str, Any] | None, t0: float) -> dict[str, Any]:
        agent_json = _read_json_file(self.workdir / "output" / "result.json")
        claimed = None
        if agent_json:
            claimed = agent_json.get("final_accepted_round")
            try:
                claimed = int(claimed) if claimed is not None else None
            except (TypeError, ValueError):
                pass
        stop = result.stop_reason if result else "crashed"
        return {
            "paper_id": self.paper.paper_id,
            "title": self.paper.title,
            "meta": self.paper.meta,
            "accepted": self.accepted_round_idx is not None,
            "accepted_round": self.accepted_round_idx,
            "final_qv": final_qv,
            "final_accepted": (self.accepted_round_idx is not None) and (final_qv is None or bool(final_qv.get("passed"))),
            "n_rounds": len(self.rounds),
            "rounds": [self._round_summary(r) for r in self.rounds],
            "agent_stop_reason": stop,
            "agent_steps_used": result.steps_used if result else None,
            "agent_error": result.error if result else None,
            "agent_result_json_present": agent_json is not None,
            "agent_claimed_accepted_round": claimed,
            "agent_claim_matches_harness": (claimed == self.accepted_round_idx) if agent_json else None,
            "guardrail_events": self.guardrail_events,
            "usage": {"main_agent": (result.usage if result else {}), "subagents": self.subagent_usage},
            "n_eval_calls": self.n_eval_calls,
            "errors": self.errors,
            "completed": stop in ("final", "max_steps", "length"),
            "wall_time_s": round(time.time() - t0, 1),
            "acceptance_preset": self.cfg.acceptance.name,
            "config_name": self.cfg.run.name,
            "config_fingerprint": config_fingerprint(self.cfg),
            "harness_version": HARNESS_VERSION,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # ------------------------------------------------------------------ main entry
    async def run(self) -> dict[str, Any]:
        t0 = time.time()
        self.ws = prepare_workspace(self.cfg, self.paper, self.workdir, self.prompts_dir_abs)
        paper_text = (self.workdir / "paper.txt").read_text(encoding="utf-8")
        self.api_config_sha1 = _sha1_bytes((self.workdir / API_CONFIG_REL).read_bytes())

        def allow_write(rel_path: str) -> bool:  # receives the canonical workspace-relative path from the write tool
            ok = write_allowed(rel_path)
            if not ok:
                self._guard("write_denied", f"write to {rel_path!r} refused")
            return ok

        write_tool = make_write_tool(self.ws, allow=allow_write)
        tools = [
            make_task_tool(self.run_subagent, list(SUBAGENT_TYPES)),
            make_bash_tool(self.ws, evaluate_rubric_runner=self.run_evaluate_rubric),
            write_tool,
            make_read_tool(self.ws),
        ]
        agent = Agent(
            name="main_agent",
            system_prompt=self.prompts.main_agent_system(self.cfg.acceptance),
            tools=tools,
            llm=self.clients["main_agent"],
            max_steps=self.cfg.run.main_agent_max_steps,
            transcript_path=self.workdir / "trajectory" / "main_agent.jsonl",
            event_hook=self._on_event,
            context_budget_chars=self.cfg.run.main_agent_context_budget_chars,
            context_budget_tokens=getattr(self.cfg.run, "main_agent_context_budget_tokens", None),
            max_model_len=getattr(self.cfg.run, "max_model_len", None),
        )
        result: AgentResult | None = None
        try:
            result = await agent.run(self.prompts.task_prompt(self.paper.title, paper_text))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.errors.append(f"main agent crashed: {e!r}")
            self.log(f"[{self.paper.paper_id}] main agent crashed: {e!r}")
        final_qv = None
        if self.accepted is not None and self.cfg.run.final_qv:
            final_qv = await self.run_final_qv()
        summary = self.summary(result, final_qv, t0)
        atomic_write_json(self.workdir / "harness_summary.json", summary)
        self.log(f"[{self.paper.paper_id}] done: accepted={summary['accepted']} rounds={summary['n_rounds']} "
                 f"stop={summary['agent_stop_reason']} wall={summary['wall_time_s']}s")
        return summary
