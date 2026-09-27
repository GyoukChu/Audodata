#!/usr/bin/env python3
"""Compact view of one paper run: rounds, verdicts, scores, and the main agent's tool-call sequence.

    python scripts/inspect_run.py runs/smoke3/s2_123456 [--tools] [--subagent challenger_01] [--chars 600]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def short(s: str | None, n: int) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[:n] + f"... [+{len(s)-n} chars]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--tools", action="store_true", help="print the main agent's tool calls/results")
    ap.add_argument("--subagent", help="print a subagent transcript, e.g. challenger_01")
    ap.add_argument("--chars", type=int, default=400)
    a = ap.parse_args()
    wd = Path(a.workdir)
    hs = wd / "harness_summary.json"
    if hs.exists():
        s = json.loads(hs.read_text())
        print(f"== {s['paper_id']} | {short(s['title'], 90)}")
        print(f"accepted={s['accepted']} round={s['accepted_round']} final_accepted={s.get('final_accepted')} "
              f"rounds={s['n_rounds']} stop={s['agent_stop_reason']} steps={s['agent_steps_used']} wall={s['wall_time_s']}s "
              f"claim_match={s.get('agent_claim_matches_harness')} errors={len(s.get('errors') or [])}")
        for r in s["rounds"]:
            print(f"  R{r['index']:>2} {r['failure_mode']:<16} qv={str(r['qv_passed']):<5} weak={r['weak_avg']} strong={r['strong_avg']} "
                  f"gap={r['gap']} type={short(r.get('question_type'), 40)}")
            print(f"       Q: {short(r.get('question'), a.chars)}")
            if r.get("qv_feedback") and r["failure_mode"] == "FAILED_QV":
                print(f"       QV: {short(r['qv_feedback'], a.chars)}")
        if s.get("final_qv"):
            fq = s["final_qv"]
            print(f"final QV: passed={fq['passed']} checks={fq['qv'].get('checks')} programmatic={fq.get('programmatic')}")
        for e in s.get("errors") or []:
            print("  ERROR:", short(e, 300))
        print("usage:", s.get("usage"))
    else:
        print("(no harness_summary.json yet)")
    if a.tools:
        tp = wd / "trajectory" / "main_agent.jsonl"
        if tp.exists():
            print("\n== main agent tool calls ==")
            for line in tp.read_text().splitlines():
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                kind = ev.get("kind") or ev.get("type") or ev.get("role")
                if kind in ("tool_call",):
                    d = ev.get("data", ev)
                    print(f"  -> {d.get('name')} {short(json.dumps(d.get('arguments', d.get('args')), ensure_ascii=False), a.chars)}")
                elif kind in ("tool_result",):
                    d = ev.get("data", ev)
                    print(f"  <- {short(str(d.get('result', d.get('content'))), a.chars)}")
                elif kind == "final":
                    print(f"  FINAL: {short(json.dumps(ev.get('data')), a.chars)}")
    if a.subagent:
        tp = wd / "trajectory" / f"{a.subagent}.jsonl"
        print(f"\n== {a.subagent} ==")
        for line in tp.read_text().splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            role = ev.get("role") or ev.get("kind")
            content = ev.get("content") if isinstance(ev.get("content"), str) else json.dumps(ev.get("data") or ev.get("content"))
            print(f"  [{role}] {short(content, a.chars)}")


if __name__ == "__main__":
    main()
