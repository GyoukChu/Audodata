"""Table-1-style statistics for a run directory (agentic) and a CoT baseline directory.

    autodata-stats --run-root runs/cs_default [--cot-root runs/cs_default_cot] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _load_summaries(root: Path, filename: str) -> list[dict[str, Any]]:
    newest: dict[str, dict[str, Any]] = {}

    def finished_at(summary: dict) -> float:
        value = summary.get("finished_at")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (TypeError, ValueError, OverflowError):
            return float("-inf")

    for p in sorted(Path(root).glob(f"*/{filename}")):
        if p.parent.name.startswith("_") or ".old." in p.parent.name:
            continue
        try:
            summary = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(summary, dict) or not isinstance(summary.get("paper_id"), str) or not summary["paper_id"]:
            continue
        paper_id = summary["paper_id"]
        if paper_id not in newest or finished_at(summary) >= finished_at(newest[paper_id]):
            newest[paper_id] = summary
    return list(newest.values())


def _skipped_locked(root: Path, runs: list[dict[str, Any]]) -> int:
    # A runner must not write a summary into a workspace owned by another
    # process. Its skipped records live only in the corpus-level append log.
    latest: dict[str, dict[str, Any]] = {}
    try:
        with (root / "summary.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("paper_id"), str):
                    latest[record["paper_id"]] = record
    except OSError:
        pass
    ids = {paper_id for paper_id, record in latest.items() if record.get("skipped") == "locked"}
    ids.update(run["paper_id"] for run in runs if run.get("skipped") == "locked")
    return len(ids)


def _mean(xs: Iterable[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return round(st.fmean(vals), 4) if vals else None


def _median(xs: Iterable[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return round(st.median(vals), 3) if vals else None


def _table1(items: list[dict], rounds: list[dict]) -> dict[str, Any]:
    return {
        "n": len(items),
        "weak_solver_avg": _mean(rd.get("weak_avg") for rd in rounds),
        "strong_solver_avg": _mean(rd.get("strong_avg") for rd in rounds),
        "gap": _mean(rd.get("gap") for rd in rounds),
        "agentic_rounds_mean": _mean(r.get("accepted_round") for r in items),
        "agentic_rounds_median": _median(r.get("accepted_round") for r in items),
        "agentic_rounds_max": max((r.get("accepted_round") or 0) for r in items) if items else None,
        "question_length_chars": _mean(rd.get("question_chars") for rd in rounds),
        "rubric_items": _mean(rd.get("n_rubric_items") for rd in rounds),
    }


def summarize_agentic(root: str | Path) -> dict[str, Any]:
    runs = _load_summaries(Path(root), "harness_summary.json")
    accepted = [r for r in runs if r.get("accepted")]
    final = [r for r in runs if r.get("final_accepted")]

    def _acc_round(r: dict) -> dict:
        return r["rounds"][r["accepted_round"] - 1]

    acc_rounds = [_acc_round(r) for r in accepted]
    final_rounds = [_acc_round(r) for r in final]
    all_rounds = [rd for r in runs for rd in r.get("rounds", [])]
    failed_rounds = [rd for rd in all_rounds if rd.get("failure_mode") != "ACCEPTED"]
    modes = Counter(rd.get("failure_mode") for rd in failed_rounds)
    n_failed = len(failed_rounds) or 1
    stop = Counter(r.get("agent_stop_reason") for r in runs)
    usage_prompt = sum((r.get("usage", {}).get("main_agent", {}) or {}).get("prompt_tokens", 0) or 0 for r in runs)
    usage_comp = sum((r.get("usage", {}).get("main_agent", {}) or {}).get("completion_tokens", 0) or 0 for r in runs)
    sub_prompt = sum((r.get("usage", {}).get("subagents", {}) or {}).get("prompt_tokens", 0) or 0 for r in runs)
    sub_comp = sum((r.get("usage", {}).get("subagents", {}) or {}).get("completion_tokens", 0) or 0 for r in runs)
    return {
        "kind": "agentic",
        "root": str(root),
        "n_papers": len(runs),
        "n_accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / len(runs), 4) if runs else None,
        "n_final_accepted": len(final),
        "table1": _table1(accepted, acc_rounds),
        "table1_after_final_qv": _table1(final, final_rounds),
        "guardrail_events": sum(len(r.get("guardrail_events") or []) for r in runs),
        "incomplete_runs": sum(1 for r in runs if not r.get("completed", True)),
        "n_incomplete": sum(1 for r in runs if not r.get("completed", True)),
        "n_skipped_locked": _skipped_locked(Path(root), runs),
        "rounds_per_paper_mean": _mean(r.get("n_rounds") for r in runs),
        "failed_round_modes": {k: {"count": v, "share": round(v / n_failed, 3)} for k, v in modes.most_common()},
        "agent_stop_reasons": dict(stop),
        "agent_claim_mismatches": sum(1 for r in runs if r.get("agent_claim_matches_harness") is False),
        "papers_with_errors": sum(1 for r in runs if r.get("errors")),
        "wall_time_mean_s": _mean(r.get("wall_time_s") for r in runs),
        "tokens": {"main_agent_prompt": usage_prompt, "main_agent_completion": usage_comp,
                   "subagents_prompt": sub_prompt, "subagents_completion": sub_comp},
    }


def summarize_cot(root: str | Path) -> dict[str, Any]:
    runs = _load_summaries(Path(root), "cot_summary.json")
    ok = [r for r in runs if r.get("weak_avg") is not None and r.get("strong_avg") is not None and not r.get("errors")]
    qv_ok = [r for r in ok if r.get("qv_passed")]
    final_ok = [r for r in ok if r.get("final_filter_passed")]

    def block(rs: list[dict]) -> dict[str, Any]:
        return {
            "n": len(rs),
            "weak_solver_avg": _mean(r.get("weak_avg") for r in rs),
            "strong_solver_avg": _mean(r.get("strong_avg") for r in rs),
            "gap": _mean(r.get("gap") for r in rs),
            "agentic_rounds": 1.0 if rs else None,
            "question_length_chars": _mean(r.get("question_chars") for r in rs),
            "rubric_items": _mean(r.get("n_rubric_items") for r in rs),
            "would_pass_solver_criteria": round(sum(1 for r in rs if r.get("all_solver_criteria_passed")) / len(rs), 3) if rs else None,
        }

    return {
        "kind": "cot",
        "root": str(root),
        "n_papers": len(runs),
        "n_evaluated": len(ok),
        "n_challenger_parse_failures": sum(1 for r in runs if r.get("challenger_parsed") is False),
        "qv_pass_rate": round(len(qv_ok) / len(ok), 3) if ok else None,
        "table1_all_evaluated": block(ok),
        "table1_qv_passed": block(qv_ok),
        "table1_final_filter_passed": block(final_ok),
        "papers_with_errors": sum(1 for r in runs if r.get("errors")),
        "wall_time_mean_s": _mean(r.get("wall_time_s") for r in runs),
    }


def print_report(s: dict[str, Any]) -> None:
    if s.get("kind") == "agentic":
        t = s["table1"]
        print(f"\n== Agentic Self-Instruct: {s['root']} ==")
        print(f"papers {s['n_papers']}  accepted {s['n_accepted']} ({s['acceptance_rate']})  after final QV {s['n_final_accepted']}")
        print("| Metric | Agentic (accepted) |\n|---|---|")
        print(f"| Weak solver avg | {t['weak_solver_avg']} |\n| Strong solver avg | {t['strong_solver_avg']} |\n| Gap (strong-weak) | {t['gap']} |")
        print(f"| Agentic rounds (mean/median/max) | {t['agentic_rounds_mean']} / {t['agentic_rounds_median']} / {t['agentic_rounds_max']} |")
        print(f"| Question length (chars) | {t['question_length_chars']} |\n| Rubric items | {t['rubric_items']} |")
        t2 = s["table1_after_final_qv"]
        print(f"| (after final QV, n={t2['n']}) weak/strong/gap | {t2['weak_solver_avg']} / {t2['strong_solver_avg']} / {t2['gap']} |")
        print(f"failed-round modes: {s['failed_round_modes']}  guardrail events: {s['guardrail_events']}  incomplete: {s['incomplete_runs']}")
        print(f"stop reasons: {s['agent_stop_reasons']}  claim mismatches: {s['agent_claim_mismatches']}  errors: {s['papers_with_errors']}")
        print(f"tokens: {s['tokens']}")
    else:
        print(f"\n== CoT Self-Instruct baseline: {s['root']} ==")
        print(f"papers {s['n_papers']}  evaluated {s['n_evaluated']}  QV pass rate {s['qv_pass_rate']}  parse failures {s['n_challenger_parse_failures']}")
        for name in ("table1_all_evaluated", "table1_qv_passed", "table1_final_filter_passed"):
            t = s[name]
            print(f"[{name}] n={t['n']} weak {t['weak_solver_avg']} strong {t['strong_solver_avg']} gap {t['gap']} "
                  f"qlen {t['question_length_chars']} rubric {t['rubric_items']} pass-solver-criteria {t['would_pass_solver_criteria']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root")
    ap.add_argument("--cot-root")
    ap.add_argument("--json", help="write both summaries to this file")
    args = ap.parse_args(argv)
    out: dict[str, Any] = {}
    if args.run_root:
        out["agentic"] = summarize_agentic(args.run_root)
        print_report(out["agentic"])
    if args.cot_root:
        out["cot"] = summarize_cot(args.cot_root)
        print_report(out["cot"])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
