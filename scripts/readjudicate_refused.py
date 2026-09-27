#!/usr/bin/env python3
"""Re-adjudicate agentic papers whose acceptance was refused only by harness checks that were later withdrawn.

Round-4 decisions (docs/knowledge-base/review-decisions.md) removed two refusal causes that the paper's own pipeline
never had: (a) the blocking in-loop QV binding (`qv_not_bound`), now informational; (b) report-verification problems
caused by version skew between a running harness and an updated evaluator (`question_hash mismatch` and the cascading
`weak result provenance` problem), plus the withdrawn weight-range check. A paper qualifies only if, in its FIRST
qualifying round r (and no later round exists): the in-loop QV passed; the harness verdict passed all three solver
criteria; every recorded problem is one of the withdrawn ones; an evaluator report in eval_attempts carries exactly this
round's candidate with `all_passed`, the configured models and attempt counts; its weak source report carries the same
candidate; and an independent exact recomputation from the per-criterion judgments passes the acceptance predicate.

The summary is rewritten as accepted at round r with `final_qv` cleared (a backup is kept); the next resume of the
pipeline runs the end-of-loop quality verifier on the frozen candidate. Papers held by a runner are skipped.
Usage: python scripts/readjudicate_refused.py --config configs/cs_pilot.yaml --root runs/pilot [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from autodata.config import load_config
from autodata.cs.pipeline import PaperLock, _atomic_write_json
from autodata.cs.rubric import parse_rubric
from autodata.cs.run_paper import _attempt_scores, harness_predicate

WITHDRAWN = {
    "question_hash mismatch",
    "weak result provenance: strong-only report does not reuse a weak run verified in this round",
    "rubric weights outside the challenger spec (|weight| must be 1..10)",
}
FIELDS = ("context", "question", "rubric")


def _same_candidate(a: dict, b: dict) -> bool:
    return all(a.get(key) == b.get(key) for key in FIELDS)


def qualifying_round(cfg, workdir: Path, summary: dict) -> tuple[int, Path, str] | None:
    rounds = summary.get("rounds") or []
    events = summary.get("guardrail_events") or []
    for rd in rounds:
        r = rd.get("index")
        if not (rd.get("qv_passed") is True and rd.get("weak_passed") is True and rd.get("strong_passed") is True
                and rd.get("gap_passed") is True):
            continue
        problems = set(rd.get("eval_problems") or [])
        if not problems <= WITHDRAWN:
            continue
        binding = any(e.get("kind") == "qv_not_bound" and e.get("round") == r and not e.get("informational")
                      for e in events)
        if not problems and not binding:
            continue  # verified and passing but refused for another reason: leave it alone
        if r != len(rounds):
            print(f"  {workdir.name}: round {r} qualifies but later rounds exist; manual review", flush=True)
            return None
        items = parse_rubric(rd["rubric"])
        n_req = cfg.eval.n_attempts
        models = {role: cfg.endpoint(role).model for role in ("weak_solver", "strong_solver", "judge")}
        for path in sorted(workdir.glob("eval_attempts/run_*/report.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("all_passed") is not True or report.get("error") or not _same_candidate(report, rd):
                continue
            if (report.get("models") or {}) != models:
                continue
            src = report.get("weak_source_report")
            if report.get("mode") == "strong-only":
                if not src or not Path(src).exists():
                    continue
                weak_report = json.loads(Path(src).read_text(encoding="utf-8"))
                if not _same_candidate(weak_report, rd) or weak_report.get("error"):
                    continue
            weak = _attempt_scores(items, report.get("weak_attempts"))
            strong = _attempt_scores(items, report.get("strong_attempts"))
            if not weak or not strong or len(weak) != n_req or len(strong) != n_req:
                continue
            verdict = harness_predicate(cfg.acceptance, weak, strong)
            if verdict.get("all_passed"):
                reason = "qv_binding_withdrawn" if binding else "withdrawn_report_checks:" + ",".join(sorted(problems))
                return r, path, reason
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(args.root)
    changed = 0
    for path in sorted(root.glob("*/harness_summary.json")):
        workdir = path.parent
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("accepted") or summary.get("readjudicated"):
            continue
        lock = PaperLock(workdir)
        if not lock.acquire():
            print(f"  {workdir.name}: held by a runner, skipped", flush=True)
            continue
        try:
            found = qualifying_round(cfg, workdir, summary)
            if found is None:
                continue
            r, report_path, reason = found
            print(f"  {workdir.name}: re-adjudicated as accepted at round {r} ({reason}; {report_path.parent.name})",
                  flush=True)
            if args.dry_run:
                continue
            backup = workdir / "harness_summary.pre_readjudication.json"
            if not backup.exists():
                backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            summary["rounds"][r - 1]["failure_mode"] = "ACCEPTED"
            summary.update(accepted=True, accepted_round=r, final_qv=None, final_accepted=False,
                           agent_claim_matches_harness=summary.get("agent_claimed_accepted_round") == r,
                           readjudicated={"round": r, "reason": reason, "report": str(report_path.relative_to(workdir)),
                                          "at": datetime.now(timezone.utc).isoformat()})
            _atomic_write_json(path, summary)
            changed += 1
        finally:
            lock.release()
    print(f"re-adjudicated {changed} paper(s){' (dry run)' if args.dry_run else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
