"""Prompt loading/rendering for the CS pipeline.

All prompt text lives in prompts/cs/*.md (see prompts/cs/README.md for provenance). The only substitution applied to
the verbatim main-agent prompt is the acceptance-threshold wording, rendered from the active AcceptancePreset so that the
prompt, evaluate_rubric.py and the harness always agree.
"""
from __future__ import annotations

from pathlib import Path

from autodata.config import AcceptancePreset

# Verbatim round-1 message the main agent sends to the challenger (RAM README / Fig. 7). Used by the CoT baseline.
CHALLENGER_ROUND1_PROMPT = (
    "Generate a challenging research question-answer pair with grading rubrics. "
    "The paper is available at ./paper.txt — read it first."
)
# Verbatim refinement template (RAM README). Filled by the main agent LLM itself; kept here for tests/docs.
CHALLENGER_REFINEMENT_TEMPLATE = """The paper is available at ./paper.txt — read it first.

REFINEMENT: The following questions were previously generated for this
paper but did not meet our criteria:

Questions that were TOO EASY (weak model scored too high):
{too_easy}

Questions that FAILED ON STRONG (weak was low but strong also
struggled or scored worse):
{failed_strong}

Questions that FAILED QUALITY CHECK (quality verifier rejected):
{failed_qv}

Generate an ENTIRELY NEW question from a DIFFERENT angle that
requires deeper reasoning."""


class PromptSet:
    FILES = {
        "main_agent": "main_agent.md",
        "task_prompt": "task_prompt.md",
        "challenger": "challenger.md",
        "quality_verifier": "quality_verifier.md",
        "judge": "judge.md",
        "solver_user": "solver_user.md",
    }

    def __init__(self, prompts_dir: str | Path):
        self.dir = Path(prompts_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"prompts dir not found: {self.dir}")
        self._cache: dict[str, str] = {}

    def raw(self, name: str) -> str:
        if name not in self._cache:
            self._cache[name] = (self.dir / self.FILES[name]).read_text(encoding="utf-8")
        return self._cache[name]

    # ---- renderings ----
    def main_agent_system(self, preset: AcceptancePreset) -> str:
        text = self.raw("main_agent")
        subs = {
            "{{WEAK_CRITERIA_SHORT}}": preset.weak_criteria_short(),
            "{{STRONG_CRITERIA_SHORT}}": preset.strong_criteria_short(),
            "{{GAP_CRITERIA_SHORT}}": preset.gap_criteria_short(),
            "{{STRONG_CHECKLIST}}": preset.strong_checklist(),
        }
        for k, v in subs.items():
            if k not in text:
                raise ValueError(f"placeholder {k} missing from main_agent.md")
            text = text.replace(k, v)
        return text

    def task_prompt(self, title: str, paper_text: str) -> str:
        return self.raw("task_prompt").format(title=title, paper_text=paper_text)

    def challenger_system(self) -> str:
        return self.raw("challenger")

    def quality_verifier_system(self) -> str:
        return self.raw("quality_verifier")

    def judge_system(self) -> str:
        return self.raw("judge")

    def solver_user(self, context: str, question: str) -> str:
        return self.raw("solver_user").format(context=context, question=question)

    @staticmethod
    def qv_request(question_type: str, context: str, question: str, rubric_json: str) -> str:
        """Programmatic version of the main agent's QV message ("Send: context + question + rubric + question_type").
        Used by the CoT baseline and the end-of-loop QV pass."""
        return (
            "Verify the following research QA package. The paper is available at ./paper.txt — read it first.\n\n"
            f"question_type: {question_type}\n\n"
            f"context:\n{context}\n\n"
            f"question:\n{question}\n\n"
            f"rubric (JSON):\n{rubric_json}\n"
        )
