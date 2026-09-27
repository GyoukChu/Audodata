"""Tolerant parsers for subagent outputs (challenger JSON block, quality-verifier verdict lines)."""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_VALID_ESCAPES = set('\\"/bfnrtu')


def repair_json_escapes(raw: str) -> str:
    """Double every backslash that does not start a valid JSON escape (LaTeX inside JSON strings, e.g. `$\\alpha$`).
    Scans left to right so that already-valid escapes (`\\\\`, `\\n`, `\\"`) are skipped as a unit and never re-escaped."""
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\":
            nxt = raw[i + 1] if i + 1 < n else ""
            if nxt in _VALID_ESCAPES and nxt:
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            out.append("\\\\")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the last top-level well-formed JSON object in `text`, regardless of fencing.
    Objects nested inside another well-formed object are never returned on their own."""
    if not text:
        return None
    fenced = [(m.start(1), m.end(1), m.group(1)) for m in _FENCE_RE.finditer(text)]
    spans: list[tuple[int, int, str]] = fenced[:]
    for m in re.finditer(r"\{", text):
        s = m.start()
        depth = 0
        in_str = False
        esc = False
        for i in range(s, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    spans.append((s, i + 1, text[s:i + 1]))
                    break
    valid: list[tuple[int, int, dict[str, Any]]] = []
    # Fenced spans are collected first; dedupe equal spans so they cannot
    # eliminate each other in the top-level containment check.
    unique = {(start, end): cand for start, end, cand in reversed(spans)}
    for (s, e), cand in sorted(unique.items()):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            try:  # lenient pass: LaTeX backslashes inside strings are not valid JSON escapes
                obj = json.loads(repair_json_escapes(cand))
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            valid.append((s, e, obj))
    # drop candidates nested inside another valid candidate
    top = [v for v in valid if not any(o is not v and o[0] <= v[0] and v[1] <= o[1] for o in valid)]
    return top[-1][2] if top else None


def parse_challenger_output(text: str) -> dict[str, Any] | None:
    """Challenger JSON: question_type, reasoning_skills, context, question, reference_answer, rubric."""
    obj = extract_json_object(text)
    if not obj:
        return None
    if not all(k in obj for k in ("context", "question", "rubric")):
        return None
    return obj


_VERDICT_RE = re.compile(
    r"^[ \t]*(?:(?:[-+*]|\d+[.)])[ \t]+)?(?:\*\*|__)?[ \t]*"
    r"(CHECK_[1-4]_VERDICT|OVERALL)(?:\*\*|__)?[ \t]*:[ \t]*(?:(?:\*\*|__)[ \t]*)*([A-Z_]+)",
    re.M,
)


REQUIRED_QV_CHECKS = {
    "CHECK_1_VERDICT": "NO_LEAKAGE",
    "CHECK_2_VERDICT": "GOOD",
    "CHECK_3_VERDICT": "PASS",
    "CHECK_4_VERDICT": "CONSISTENT",
}


def parse_qv_output(text: str) -> dict[str, Any]:
    """Extract CHECK_n verdicts and OVERALL from a quality-verifier message.

    `overall` is True only when OVERALL is exactly PASS AND all four checks are present with their passing value
    (NO_LEAKAGE / GOOD / PASS / CONSISTENT); a PASS that contradicts a failing or missing check is False and flagged
    in `contradiction`. `overall_stated` is what the verifier literally wrote (True/False/None)."""
    out: dict[str, Any] = {"overall": None, "overall_stated": None, "checks": {}, "feedback": None,
                           "contradiction": False, "missing_checks": []}
    if not text:
        return out
    for m in _VERDICT_RE.finditer(text):
        key, val = m.group(1), m.group(2).upper()
        if key == "OVERALL":
            out["overall_stated"] = True if val == "PASS" else False if val == "FAIL" else None
        else:
            out["checks"][key] = val
    fb = re.search(r"^\s*FEEDBACK\s*:\s*(.*)$", text, re.M | re.S)
    if fb:
        out["feedback"] = fb.group(1).strip()[:4000]
    if out["overall_stated"] is None:  # fall back to the last standalone PASS/FAIL token after OVERALL
        m = re.findall(r"\bOVERALL[^A-Z]*(PASS|FAIL)\b", text.upper())
        if m:
            out["overall_stated"] = m[-1] == "PASS"
    out["missing_checks"] = [k for k in REQUIRED_QV_CHECKS if k not in out["checks"]]
    checks_ok = all(out["checks"].get(k) == v for k, v in REQUIRED_QV_CHECKS.items())
    if out["overall_stated"] is True:
        out["overall"] = checks_ok
        out["contradiction"] = not checks_ok
    elif out["overall_stated"] is False:
        out["overall"] = False
    else:
        out["overall"] = None
    return out
