"""Configuration models.

Every number that the paper leaves implicit (thresholds, budgets, sampling) lives here so that
the prompts and the evaluate_rubric.py tool are driven by ONE source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AcceptancePreset(BaseModel):
    """Acceptance predicate for the CS pipeline (see ambiguities.md A1/A2).

    All scores are fractions in [0, 1]; averages are over the n solver attempts.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    weak_avg_max: float = Field(default=0.50, ge=0, le=1)            # weak_avg must be below (or at most, if inclusive) this
    weak_avg_max_inclusive: bool = False  # False -> weak_avg <  max ; True -> weak_avg <= max
    weak_attempt_max: float | None = Field(default=None, ge=0, le=1)  # max over weak attempts <= this (deployed form only)
    weak_no_zero: bool = False            # reject if any weak attempt scored exactly 0 (deployed form)
    strong_avg_min: float = Field(default=0.65, ge=0, le=1)          # strong_avg >= this
    strong_avg_max: float | None = Field(default=None, ge=0, le=1)   # strong_avg <  this (deployed form: 0.95)
    strong_no_zero: bool = False          # reject if any strong attempt scored exactly 0 (deployed form)
    gap_min: float = Field(default=0.20, ge=0, le=1)                 # strong_avg - weak_avg >= this

    # ---- human-readable renderings used inside the (verbatim) main-agent prompt ----
    def weak_criteria_short(self) -> str:
        op = "≤" if self.weak_avg_max_inclusive else "<"
        parts = [f"weak_avg {op} {self.weak_avg_max*100:.0f}%"]
        if self.weak_attempt_max is not None:
            parts.append(f"max_weak ≤ {self.weak_attempt_max*100:.0f}%")
        if self.weak_no_zero:
            parts.append("no zeros")
        return ", ".join(parts)

    def strong_criteria_short(self) -> str:
        s = f"strong_avg ≥ {self.strong_avg_min*100:.0f}%"
        if self.strong_avg_max is not None:
            s += f" AND strong_avg < {self.strong_avg_max*100:.0f}%"
        return s

    def gap_criteria_short(self) -> str:
        return f"Gap (strong_avg - weak_avg) ≥ {self.gap_min*100:.0f}%"

    def strong_checklist(self) -> str:
        lines = [f"  - strong_avg ≥ {self.strong_avg_min*100:.0f}%? (too low = question is hard for everyone)"]
        if self.strong_avg_max is not None:
            lines.append(f"  - strong_avg < {self.strong_avg_max*100:.0f}%? (too high = question is trivial)")
        if self.strong_no_zero:
            lines.append("  - No individual strong = 0%? (suspicious)")
        lines.append(f"  - gap (strong_avg - weak_avg) ≥ {self.gap_min*100:.0f}%?")
        return "\n".join(lines)


PRESETS: dict[str, AcceptancePreset] = {
    # Sec. 3.1 prose: "accepted only if the strong solver averages >= 0.65, the weak solver < 0.5,
    # and the strong-weak gap >= 20 percentage points across the solver attempts"
    "prose_s31": AcceptancePreset(name="prose_s31", weak_avg_max=0.50, weak_avg_max_inclusive=False,
                                  strong_avg_min=0.65, gap_min=0.20),
    # Fig. 7 / RAM README main-agent prompt (deployed form)
    "deployed_c1": AcceptancePreset(name="deployed_c1", weak_avg_max=0.65, weak_avg_max_inclusive=True,
                                    weak_attempt_max=0.75, weak_no_zero=True, strong_avg_min=0.60,
                                    strong_avg_max=0.95, strong_no_zero=True, gap_min=0.20),
}


class ModelEndpoint(BaseModel):
    """One OpenAI-compatible chat endpoint + sampling parameters."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    seed: int | None = None  # per-request sampling reproducibility is best-effort
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None                 # passed through extra_body (vLLM)
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None  # extra_body (vLLM)
    max_tokens: int = 32768
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)  # e.g. enable_thinking / reasoning_effort
    extra_body: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 3600.0
    max_concurrency: int = 32
    max_retries: int = 6


class EvalConfig(BaseModel):
    n_attempts: int = 3          # "each invoked 3 times to reduce variance"
    timeout_s: int = 600         # --timeout 600 in the verbatim command
    judge_retries: int = 3
    solver_retries: int = 3
    judge_effort_fallback: bool = True  # after a verdict-less truncation (finish=length) retry at a lower reasoning effort


class RunConfig(BaseModel):
    name: str = "cs_default"
    max_rounds: int = 15                 # per-paper challenger rounds (ruled: 15)
    main_agent_max_steps: int = 120      # LLM turns for the orchestrator
    main_agent_context_budget_chars: int = 1_000_000  # ~285k tokens; older tool results are elided beyond this (OpenCode-style compaction)
    main_agent_context_budget_tokens: int | None = 290_000  # measured prompt tokens (vLLM usage) + 81,920 output must stay < 400k max-model-len
    subagent_max_steps: int = 12
    paper_concurrency: int = 8
    workdir_root: str = "runs/cs"
    paper_text_max_chars: int = 200_000  # safety cap for ./paper.txt (papers longer than this are truncated)
    paper_text_min_chars: int = 8_000    # skip papers with less body text than this
    final_qv: bool = True                # end-of-loop quality verifier pass on accepted items
    final_min_context_chars: int = 200   # end-of-loop filter: "short contexts" are removed (Sec 3.1)
    final_rubric_min_items: int = 10     # end-of-loop "malformed rubric" filter: QV Check 3 bounds (Fig. 9)
    final_rubric_max_items: int = 20
    final_rubric_min_positive: int = 4
    final_rubric_min_negative: int = 3
    seed: int = 0  # default for endpoints without an explicit seed


class AppConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    acceptance_preset: str = "prose_s31"
    acceptance_overrides: dict[str, Any] = Field(default_factory=dict)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    models: dict[str, ModelEndpoint]
    prompts_dir: str = "prompts/cs"

    @property
    def acceptance(self) -> AcceptancePreset:
        if self.acceptance_preset not in PRESETS:
            raise ValueError(f"unknown acceptance preset: {self.acceptance_preset!r}")
        base = PRESETS[self.acceptance_preset]
        return AcceptancePreset.model_validate({**base.model_dump(), **self.acceptance_overrides})

    @model_validator(mode="after")
    def validate_acceptance(self) -> AppConfig:
        # Fail during config loading, before prompts or model requests use it.
        self.acceptance
        return self

    def endpoint(self, role: str) -> ModelEndpoint:
        try:
            endpoint = self.models[role]
            return (endpoint if endpoint.seed is not None
                    else endpoint.model_copy(update={"seed": self.run.seed}))
        except KeyError as e:
            raise KeyError(f"no model endpoint configured for role {role!r}; have {list(self.models)}") from e


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)
    # a config may reference shared endpoint definitions via "roles: {role: endpoint_name}"
    if "endpoints" in raw and "roles" in raw:
        eps = raw.pop("endpoints")
        roles = raw.pop("roles")
        models = {}
        for role, spec in roles.items():
            if isinstance(spec, str):
                models[role] = eps[spec]
            else:  # dict: {"use": name, ...overrides}
                base = dict(eps[spec.pop("use")])
                base.update(spec)
                models[role] = base
        raw["models"] = models
    return AppConfig.model_validate(raw)
