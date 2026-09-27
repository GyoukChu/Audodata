"""Evaluate research questions, with weak-first compute saving (paper §3.1)."""
from __future__ import annotations

import os
import sys


def _die_with_parent() -> None:
    """Arm Linux parent-death handling before evaluator dependencies are imported."""
    if not sys.platform.startswith("linux"):
        return
    expected_parent = os.environ.get("AUTODATA_PARENT_PID", str(os.getppid()))
    try:
        expected_parent = int(expected_parent)
        import ctypes
        import signal

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
        if os.getppid() != expected_parent:
            raise RuntimeError("parent pid mismatch after arming parent-death signal")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"EVALUATOR_PARENT_ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(5)


# Library imports in the harness must not arm a signal on the harness itself.
# Both the module CLI and harness-spawned imports arm before heavy dependencies.
if __name__ == "__main__" or "AUTODATA_PARENT_PID" in os.environ:
    _die_with_parent()

import argparse
import hashlib
import json
import math
import re
import unicodedata
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

# Direct source-file execution under python -I has no script directory on
# sys.path. Resolve the package from this file, never from the working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autodata.config import AcceptancePreset, EvalConfig, ModelEndpoint, PRESETS
from autodata.cs.judge import JudgeError, run_judge
from autodata.cs.rubric import RubricError, RubricItem, parse_rubric, score_response
from autodata.cs.solvers import PROMPTS_DIR, SyncClient, run_solver, total_usage


class InputError(ValueError):
    """Invalid input, configuration, or filesystem arguments."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise InputError(message)


class _EvaluationEndpoint(ModelEndpoint):
    # Preserve optional top-level API-config seeds alongside shared sampling fields.
    seed: int | None = None


class _ApiConfig(BaseModel):
    weak_solver: _EvaluationEndpoint
    strong_solver: _EvaluationEndpoint
    judge: _EvaluationEndpoint
    acceptance: AcceptancePreset = Field(default_factory=lambda: PRESETS["prose_s31"].model_copy())
    eval: EvalConfig = Field(default_factory=EvalConfig)
    prompts_dir: str = str(PROMPTS_DIR)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path, *, raw_bytes: bytes | None = None) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_bytes() if raw_bytes is None else raw_bytes)
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read JSON from {path}: {type(exc).__name__}") from exc
    if not isinstance(obj, dict):
        raise InputError(f"{path} must contain a JSON object")
    return obj


def _load_config(path: Path, *, raw_bytes: bytes | None = None) -> tuple[_ApiConfig, Path]:
    try:
        config = _ApiConfig.model_validate(_read_object(path, raw_bytes=raw_bytes))
    except ValidationError as exc:
        # Pydantic's default error includes input values, which can include API keys.
        fields = ", ".join(".".join(map(str, e["loc"])) for e in exc.errors())
        raise InputError(f"invalid configuration fields: {fields}") from exc
    if config.eval.n_attempts < 1:
        raise InputError("eval.n_attempts must be at least 1")
    if config.eval.solver_retries < 0 or config.eval.judge_retries < 0:
        raise InputError("eval retry counts must be nonnegative")
    for name in ("weak_solver", "strong_solver", "judge"):
        endpoint = getattr(config, name)
        if not math.isfinite(endpoint.timeout_s) or endpoint.timeout_s <= 0:
            raise InputError(f"{name}.timeout_s must be positive and finite")
    for name, value in config.acceptance.model_dump().items():
        if name != "name" and type(value) is not bool and value is not None:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise InputError(f"acceptance.{name} must be a fraction in [0, 1]")
    prompts = Path(config.prompts_dir)
    if not prompts.is_absolute():
        # Explicit relative paths are config-relative, with repository-relative
        # fallback for the standard "prompts/cs" path in workspace API configs.
        local = path.parent / prompts
        prompts = local if local.is_dir() else PROMPTS_DIR.parents[1] / prompts
    try:
        (prompts / "judge.md").read_text(encoding="utf-8")
        (prompts / "solver_user.md").read_text(encoding="utf-8").format(context="", question="")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise InputError(f"cannot load prompts from {prompts}: {type(exc).__name__}") from exc
    return config, prompts.resolve()


def _new_run(output_dir: Path, mode: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    numbers = [int(match[1]) for child in output_dir.iterdir()
               if (match := re.match(r"^run_(\d+)_", child.name))]
    number = max(numbers, default=0) + 1
    while True:
        run_dir = output_dir / f"run_{number:03d}_{mode}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            number += 1


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def attempt_seed(base_seed: int | None, data: dict[str, Any], role: str, index: int) -> int | None:
    """Distinct, reproducible seed per (question, solver role, attempt index); None when sampling is unseeded.
    Identical seeds across the 3 attempts would collapse them into one sample (found in review)."""
    if base_seed is None:
        return None
    digest = hashlib.sha1(f"{base_seed}|{compute_question_hash(data)}|{role}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _run_attempt(
    index: int, role: str, solver: SyncClient, judge: SyncClient,
    data: dict[str, Any], rubric: list[RubricItem], config: _ApiConfig,
    prompts: Path, timeout: float, run_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _utc_now()
    attempt = run_solver(solver, data["context"], data["question"], prompts / "solver_user.md",
                         retries=config.eval.solver_retries, timeout=timeout,
                         seed=attempt_seed(solver.request_seed, data, role, index))
    record = asdict(attempt)
    record.update(index=index, solver=role, model=solver.endpoint.model, started_at=started_at,
                  score=None, satisfied=None, evidence=None, breakdown=None, judge=None,
                  error_type="SOLVER_ERROR" if attempt.error else None)
    truncated_no_answer = (not attempt.response_text.strip()) and attempt.finish_reason == "length"
    if not attempt.error and truncated_no_answer:
        # Truncated with no final answer (paper: counts as-is, i.e. wrong): every criterion unsatisfied, no judge call.
        satisfied = [False] * len(rubric)
        breakdown = score_response(rubric, satisfied)
        record.update(judge={"skipped": "truncated response without a final answer (finish_reason=length)",
                             "usage": {}, "latency_s": 0.0, "raw": ""},
                      satisfied=satisfied, evidence=[""] * len(rubric), breakdown=asdict(breakdown), score=breakdown.score,
                      truncated_no_answer=True)
    elif not attempt.error:
        try:
            judgment = run_judge(judge, data["context"], data["question"], rubric,
                                 attempt.response_text, prompts / "judge.md",
                                 retries=config.eval.judge_retries,
                                 effort_fallback=getattr(config.eval, "judge_effort_fallback", True))
        except JudgeError as exc:
            record.update(error=str(exc), error_type="JUDGE_ERROR")
            record["judge"] = {
                "messages": exc.messages, "requests": exc.requests,
                "raw": exc.requests[-1].get("content", "") if exc.requests else "",
                "usage": total_usage(exc.requests), "latency_s": exc.latency_s,
                "error": str(exc),
            }
        else:
            breakdown = score_response(rubric, judgment.satisfied)
            record.update(judge=asdict(judgment), satisfied=judgment.satisfied,
                          evidence=judgment.evidence, breakdown=asdict(breakdown), score=breakdown.score)
    record.update(completed_at=_utc_now(), total_latency_s=time.perf_counter() - started)
    record["attempt_path"] = str(run_dir / f"attempt_{role}_{index}.json")
    _write_json(Path(record["attempt_path"]), record)
    return record


def _run_stage(
    role: str, data: dict[str, Any], rubric: list[RubricItem], config: _ApiConfig,
    prompts: Path, timeout: float, run_dir: Path, transport: httpx.BaseTransport | None,
) -> list[dict[str, Any]]:
    endpoint = getattr(config, f"{role}_solver")
    with SyncClient(endpoint, transport=transport) as solver, SyncClient(config.judge, transport=transport) as judge:
        with ThreadPoolExecutor(max_workers=config.eval.n_attempts) as pool:
            futures = [pool.submit(_run_attempt, i, role, solver, judge, data, rubric,
                                   config, prompts, timeout, run_dir)
                       for i in range(1, config.eval.n_attempts + 1)]
            return [future.result() for future in futures]


def _latest_weak(
    output_dir: Path, question_hash: str, rubric: list[RubricItem],
    *, config: _ApiConfig | None = None, config_sha1: str | None = None,
    prompts_dir: str | Path | None = None, legacy_hash: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, Path | None]:
    # Without the current configuration there is no safe provenance comparison.
    if config is None or config_sha1 is None or prompts_dir is None:
        return [], None, None
    models = {role: getattr(config, role).model for role in ("weak_solver", "strong_solver", "judge")}
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("run_*/report.json"):
        if match := re.fullmatch(r"run_(\d+)_(?:weak-only|both)", path.parent.name):
            candidates.append((int(match[1]), path))
    for _, path in sorted(candidates, reverse=True):
        try:
            report = _read_object(path)
            # Reports retain the legacy field for running harnesses; use their
            # canonical field too when only JSON layout or trailing whitespace changed.
            identities = {report.get("question_hash"), report.get("question_hash_canonical")} - {None}
            if not identities.intersection({question_hash, legacy_hash} - {None}):
                continue
            if (report.get("acceptance") != config.acceptance.model_dump()
                    or report.get("eval") != config.eval.model_dump()):
                continue
            if (report.get("models") != models or report.get("prompts_dir") != str(prompts_dir)
                    or report.get("config_sha1") != config_sha1):
                continue
            attempts = report["weak_attempts"]
            if not isinstance(attempts, list) or not attempts:
                continue
            if len(attempts) != config.eval.n_attempts:
                continue
            for attempt in attempts:
                if attempt.get("error") or attempt.get("error_type"):
                    raise ValueError("incomplete weak run")
                # Stored scores are derived data; recompute from the judgments.
                breakdown = score_response(rubric, attempt["satisfied"])
                attempt.update(score=breakdown.score, breakdown=asdict(breakdown))
            return attempts, report, path.resolve()
        except (InputError, KeyError, TypeError, ValueError, AttributeError):
            continue
    return [], None, None


def legacy_question_hash(data: dict) -> str:
    """Identity used by existing evaluator reports before canonicalisation."""
    question = {key: data[key] for key in ("context", "question", "rubric")}
    return hashlib.sha1(json.dumps(question).encode("utf-8")).hexdigest()


def compute_question_hash(data: dict) -> str:
    """Hash canonical question content, independent of JSON layout and metadata."""
    def normalise(text: str) -> str:
        return "\n".join(line.rstrip() for line in
                         unicodedata.normalize("NFC", text).replace("\r\n", "\n").split("\n"))

    question = {key: normalise(data[key]) for key in ("context", "question")}
    question["rubric"] = [dict(criterion=normalise(item.criterion), weight=item.weight,
                               category=normalise(item.category)) for item in parse_rubric(data["rubric"])]
    canonical = json.dumps(question, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def assess_attempts(
    rubric: list[RubricItem] | list[dict[str, Any]],
    weak_attempts: list[dict[str, Any]], strong_attempts: list[dict[str, Any]],
    preset: AcceptancePreset | dict[str, Any],
) -> dict:
    """Recompute acceptance from satisfied lists without trusting cached scores.

    Inputs are never modified. Missing stages and attempts with errors cannot
    pass; malformed judgments raise RubricError instead of producing a score.
    Attempt-count and configuration provenance must be checked by the caller.
    """
    items = (rubric if rubric and all(isinstance(item, RubricItem) for item in rubric)
             else parse_rubric(rubric))
    if isinstance(preset, dict):
        preset = AcceptancePreset.model_validate(preset)

    def recompute(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for attempt in attempts:
            error = attempt.get("error") or attempt.get("error_type")
            record: dict[str, Any] = {"error": error}
            if not error:
                satisfied = attempt.get("satisfied")
                if not isinstance(satisfied, list):
                    raise RubricError("satisfied must contain exactly one boolean per criterion")
                record["breakdown"] = asdict(score_response(items, satisfied))
            records.append(record)
        return records

    assessment: dict[str, Any] = {
        "weak_attempts": recompute(weak_attempts), "strong_attempts": recompute(strong_attempts),
    }
    error_kinds = [attempt["error_type"] for attempt in weak_attempts + strong_attempts
                   if attempt.get("error_type")]
    if error_kinds:
        # Match the CLI's solver-error precedence for a stage with mixed failures.
        assessment["error"] = {"kind": "SOLVER_ERROR" if "SOLVER_ERROR" in error_kinds else error_kinds[0]}
    _assess(assessment, preset)
    return {key: value for key, value in assessment.items()
            if key not in ("weak_attempts", "strong_attempts", "error")}


def _scores(attempts: list[dict[str, Any]]) -> list[Fraction]:
    if not attempts or any(attempt.get("error") for attempt in attempts):
        return []
    # Use exact rubric ratios to avoid rejecting a gap equal to 0.20 due to
    # binary floating-point subtraction; thresholds still operate on fractions.
    return [max(Fraction(0), min(Fraction(1), Fraction(
        attempt["breakdown"]["earned"] - attempt["breakdown"]["penalty"],
        attempt["breakdown"]["max_positive"],
    ))) for attempt in attempts]


def _pct(value: float | Fraction) -> str:
    return f"{float(value) * 100:.1f}%"


def _assess(report: dict[str, Any], preset: AcceptancePreset) -> None:
    weak, strong = _scores(report["weak_attempts"]), _scores(report["strong_attempts"])
    weak_reasons: list[str] = []
    strong_reasons: list[str] = []
    gap_reasons: list[str] = []
    for role, scores in (("weak", weak), ("strong", strong)):
        report[f"{role}_avg"] = float(sum(scores) / len(scores)) if scores else None
        report[f"max_{role}"] = float(max(scores)) if scores else None
        report[f"min_{role}"] = float(min(scores)) if scores else None
        report[f"{role}_passed"] = None
    report.update(gap=None, gap_passed=None)
    if weak:
        avg = sum(weak) / len(weak)
        limit = Fraction(str(preset.weak_avg_max))
        if avg > limit or (avg == limit and not preset.weak_avg_max_inclusive):
            op = ">" if preset.weak_avg_max_inclusive else ">="
            weak_reasons.append(f"TOO EASY (weak_avg {_pct(avg)} {op} {_pct(limit)})")
        if preset.weak_attempt_max is not None and max(weak) > Fraction(str(preset.weak_attempt_max)):
            weak_reasons.append(f"max_weak {_pct(max(weak))} > {_pct(preset.weak_attempt_max)}")
        if preset.weak_no_zero and min(weak) == 0:
            weak_reasons.append("zero weak attempt")
        report["weak_passed"] = not weak_reasons
    if strong:
        avg = sum(strong) / len(strong)
        if avg < Fraction(str(preset.strong_avg_min)):
            strong_reasons.append(f"strong_avg {_pct(avg)} < {_pct(preset.strong_avg_min)}")
        if preset.strong_avg_max is not None and avg >= Fraction(str(preset.strong_avg_max)):
            strong_reasons.append(f"strong_avg {_pct(avg)} >= {_pct(preset.strong_avg_max)}")
        if preset.strong_no_zero and min(strong) == 0:
            strong_reasons.append("zero strong attempt")
        report["strong_passed"] = not strong_reasons
    if weak and strong:
        gap = sum(strong) / len(strong) - sum(weak) / len(weak)
        report.update(gap=float(gap), gap_passed=gap >= Fraction(str(preset.gap_min)))
        if not report["gap_passed"]:
            gap_reasons.append(f"gap {_pct(gap)} < {_pct(preset.gap_min)}")
    report.update(weak_failure_reasons=weak_reasons, strong_failure_reasons=strong_reasons,
                  gap_failure_reasons=gap_reasons)
    reasons = ([f"weak: {reason}" for reason in weak_reasons]
               + [f"failed on strong: {reason}" for reason in strong_reasons] + gap_reasons)
    if not weak:
        reasons.append("no complete weak result")
    if not strong:
        reasons.append("strong not evaluated" if not report["strong_attempts"] else "no complete strong result")
    if report.get("error"):
        reasons.append(report["error"]["kind"])
    report["failure_reasons"] = reasons
    report["all_passed"] = bool(report["weak_passed"] and report["strong_passed"]
                                and report["gap_passed"] and not report.get("error"))


def _render_report(report: dict[str, Any], preset: AcceptancePreset, rubric: list[RubricItem]) -> str:
    lines: list[str] = []
    if report.get("error"):
        lines.append(f"{report['error']['kind']}: {report['error']['message']}")
    lines.extend([f"=== EVALUATE_RUBRIC REPORT ({report['mode']}) ===",
                  f"question_hash: {report['question_hash']}"])
    n_pos = sum(item.weight > 0 for item in rubric)
    n_neg = len(rubric) - n_pos
    lines.append(f"rubric: {len(rubric)} criteria ({n_pos} positive, {n_neg} negative), "
                 f"positive weight total {sum(item.weight for item in rubric if item.weight > 0)}")
    for role in ("weak", "strong"):
        attempts = report[f"{role}_attempts"]
        if not attempts:
            if role == "strong" and report["mode"] == "both":
                lines.append("STRONG_SKIPPED: weak evaluation did not pass")
            elif role == "strong" and report["mode"] == "strong-only":
                lines.append("STRONG_SKIPPED: no passing weak result for this question (run --weak-only first)")
            continue
        lines.append(f"{role} solver: {report['models'][f'{role}_solver']} x{len(attempts)}")
        for attempt in attempts:
            if attempt.get("error"):
                lines.append(f"  attempt {attempt['index']}: {attempt['error_type']}: {attempt['error']}")
                continue
            breakdown = attempt["breakdown"]
            lines.append(
                f"  attempt {attempt['index']}: {_pct(attempt['score'])}  "
                f"(positive satisfied {breakdown['n_pos_satisfied']}/{n_pos}, "
                f"negative triggered {breakdown['n_neg_triggered']}/{n_neg}, "
                f"finish={attempt['finish_reason']}, {attempt['usage'].get('completion_tokens', 0)} completion tokens)"
            )
        if report[f"{role}_avg"] is None:
            continue
        lines.append(f"{role}_avg: {_pct(report[f'{role}_avg'])}  "
                     f"max_{role}: {_pct(report[f'max_{role}'])}  min_{role}: {_pct(report[f'min_{role}'])}")
        if report[f"{role}_passed"]:
            if role == "weak":
                op = "<=" if preset.weak_avg_max_inclusive else "<"
                criteria = [f"weak_avg {_pct(report['weak_avg'])} {op} {_pct(preset.weak_avg_max)}"]
                if preset.weak_attempt_max is not None:
                    criteria.append(f"max_weak <= {_pct(preset.weak_attempt_max)}")
                if preset.weak_no_zero:
                    criteria.append("no zero weak attempt")
                lines.append(f"WEAK_PASSED ({', '.join(criteria)})")
            else:
                lines.append("STRONG_PASSED")
        else:
            lines.append(f"{role.upper()}_FAILED: {'; '.join(report[f'{role}_failure_reasons'])}")
    if report["mode"] == "strong-only" and not report["weak_attempts"]:
        lines.append("NO_WEAK_RESULT: run --weak-only first")
    if report["gap"] is not None:
        lines.append(f"gap: {_pct(report['gap'])}")
        lines.append("GAP_PASSED" if report["gap_passed"]
                     else f"GAP_FAILED: {'; '.join(report['gap_failure_reasons'])}")
    if report["mode"] != "weak-only" or report.get("error"):
        lines.append("ACCEPTANCE: ALL_SOLVER_CRITERIA_PASSED" if report["all_passed"]
                     else f"ACCEPTANCE: FAILED ({'; '.join(report['failure_reasons'])})")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    if transport is None:
        _die_with_parent()
    parser = _Parser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout", type=float, help="timeout in seconds per solver request")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--weak-only", action="store_true")
    modes.add_argument("--strong-only", action="store_true")
    parser.add_argument("--force-strong", action="store_true",
                        help="strong-only: evaluate the strong solver even without a passing weak result (CoT baseline "
                             "statistics need both solvers on every item; the agent sandbox does not allow this flag)")
    try:
        args = parser.parse_args(argv)
        input_path, config_path = args.input.resolve(), args.config.resolve()
        input_bytes = input_path.read_bytes()
        data = _read_object(input_path, raw_bytes=input_bytes)
        for key in ("context", "question", "rubric"):
            if key not in data:
                raise InputError(f"input is missing {key}")
        for key in ("context", "question"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise InputError(f"input {key} must be a non-empty string")
        rubric = parse_rubric(data["rubric"])
        config_bytes = config_path.read_bytes()
        config, prompts = _load_config(config_path, raw_bytes=config_bytes)
        timeout = config.eval.timeout_s if args.timeout is None else args.timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise InputError("timeout must be positive and finite")
        mode = "weak-only" if args.weak_only else "strong-only" if args.strong_only else "both"
        output_dir = args.output_dir.resolve()
        run_dir = _new_run(output_dir, mode)
        question = {key: data[key] for key in ("context", "question", "rubric")}
        # Report field `question_hash` keeps the legacy identity so that harness processes started before the
        # canonical hash existed still verify new reports; `question_hash_canonical` carries the new identity.
        # Harness and statistics accept either value.
        question_hash = legacy_question_hash(data)
        question_hash_canonical = compute_question_hash(data)
        started = time.perf_counter()
        report: dict[str, Any] = {
            "mode": mode, "question_hash": question_hash, "question_hash_canonical": question_hash_canonical, **question,
            "input_path": str(input_path), "input_sha1": hashlib.sha1(input_bytes).hexdigest(),
            "config_path": str(config_path), "config_sha1": hashlib.sha1(config_bytes).hexdigest(),
            "n_attempts_required": config.eval.n_attempts,
            "parsed_rubric": [asdict(item) for item in rubric],
            "models": {role: getattr(config, role).model for role in ("weak_solver", "strong_solver", "judge")},
            "eval": config.eval.model_dump(), "acceptance": config.acceptance.model_dump(),
            "timeout_s": timeout, "prompts_dir": str(prompts), "run_dir": str(run_dir), "force_strong": bool(args.force_strong),
            "started_at": _utc_now(), "weak_attempts": [], "strong_attempts": [],
            "weak_source_report": None, "weak_source_run_dir": None, "error": None,
        }
        if mode == "strong-only":
            attempts, previous, source = _latest_weak(
                output_dir, question_hash, rubric, config=config,
                config_sha1=report["config_sha1"], prompts_dir=prompts, legacy_hash=question_hash_canonical,
            )
            report.update(weak_attempts=attempts, weak_source_report=str(source) if source else None,
                          weak_source_run_dir=str(source.parent) if source else None)
            if previous:
                report["models"]["weak_solver"] = previous["models"]["weak_solver"]
        for role in (("strong",) if mode == "strong-only" else ("weak",) if mode == "weak-only"
                     else ("weak", "strong")):
            report.update(assess_attempts(rubric, report["weak_attempts"], report["strong_attempts"], config.acceptance))
            if role == "strong" and not report["weak_passed"] and not args.force_strong:
                # Sec 3.1: the strong solver runs only when the weak solver passes. The CoT baseline needs both
                # columns on every item (Table 1) and passes --force-strong, which bypasses this gate too.
                break
            if mode == "strong-only" and not args.force_strong and (not previous or previous.get("weak_passed") is not True):
                break
            attempts = _run_stage(role, data, rubric, config, prompts, timeout, run_dir, transport)
            report[f"{role}_attempts"] = attempts
            errors = [attempt for attempt in attempts if attempt.get("error")]
            if errors:
                # Any exhausted solver failure takes priority, independently of thread order.
                error = next((e for e in errors if e["error_type"] == "SOLVER_ERROR"), errors[0])
                code = 2 if error["error_type"] == "SOLVER_ERROR" else 3
                report["error"] = {"kind": error["error_type"], "exit_code": code,
                                   "message": f"{role} attempt {error['index']}: {error['error']}"}
                break
        report.update(assess_attempts(rubric, report["weak_attempts"], report["strong_attempts"], config.acceptance))
        report.update(completed_at=_utc_now(), latency_s=time.perf_counter() - started)
        rendered = _render_report(report, config.acceptance, rubric)
        _write_json(run_dir / "report.json", report)
        (run_dir / "report.txt").write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        print(f"REPORT_PATH: {run_dir / 'report.json'}")
        return report["error"]["exit_code"] if report["error"] else 0
    except RubricError as exc:
        print(f"RUBRIC_ERROR: {exc}")
        return 4
    except (InputError, OSError, UnicodeError) as exc:
        print(f"INPUT_ERROR: {exc}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
