import json
from pathlib import Path

import pytest

from autodata.config import PRESETS
from autodata.cs.parsing import extract_json_object, parse_challenger_output, parse_qv_output
from autodata.cs.prompts import CHALLENGER_ROUND1_PROMPT, PromptSet

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "cs"


def test_main_agent_prompt_renders_prose_preset():
    ps = PromptSet(PROMPTS)
    text = ps.main_agent_system(PRESETS["prose_s31"])
    assert "{{" not in text
    assert "(weak_avg < 50%)" in text
    assert "strong_avg ≥ 65%" in text
    assert "Gap (strong_avg - weak_avg) ≥ 20%" in text
    # verbatim command lines must survive untouched
    assert "cd /workspace/project && uv run python3" in text
    assert ".opencode/tools/evaluate_rubric.py" in text and "--weak-only" in text and "--strong-only" in text
    assert "Write output/result.json using the write tool (not bash)" in text


def test_main_agent_prompt_renders_deployed_preset():
    text = PromptSet(PROMPTS).main_agent_system(PRESETS["deployed_c1"])
    assert "weak_avg ≤ 65%, max_weak ≤ 75%, no zeros" in text
    assert "strong_avg ≥ 60% AND strong_avg < 95%" in text
    assert "No individual strong = 0%? (suspicious)" in text


def test_task_prompt_and_round1_prompt():
    ps = PromptSet(PROMPTS)
    t = ps.task_prompt("A Title", "BODY")
    assert "A Title" in t and "BODY" in t and "./paper.txt" in t
    assert CHALLENGER_ROUND1_PROMPT.startswith("Generate a challenging research question-answer pair")


def test_extract_json_prefers_fenced_block():
    text = 'preamble {"a": 1}\n```json\n{"question_type": "x", "context": "c", "question": "q", "rubric": [{"criterion": "k", "weight": 3, "category": "positive"}]}\n```\ntrailer'
    obj = parse_challenger_output(text)
    assert obj and obj["question"] == "q" and obj["rubric"][0]["weight"] == 3


def test_extract_json_without_fences_and_nested_braces():
    text = 'Scratchpad {not json}\nFinal: {"context": "a {b} c", "question": "q?", "rubric": [], "extra": {"x": "}"}}'
    obj = extract_json_object(text)
    assert obj and obj["context"] == "a {b} c" and obj["extra"]["x"] == "}"
    assert parse_challenger_output("no json here") is None


def test_parse_qv_output():
    text = """CHECK_1_VERDICT: NO_LEAKAGE
CHECK_2_VERDICT: GOOD
CHECK_3_VERDICT: PASS
CHECK_3_ISSUES: none (Positive: 8, Negative: 4, Total: 12)
CHECK_4_VERDICT: CONSISTENT
OVERALL: PASS
FEEDBACK: none"""
    p = parse_qv_output(text)
    assert p["overall"] is True and p["checks"]["CHECK_3_VERDICT"] == "PASS" and p["feedback"] == "none"
    p2 = parse_qv_output("...\n**OVERALL:** FAIL\nFEEDBACK: context leaks the answer\nmore")
    assert p2["overall"] is False and "leaks" in p2["feedback"]
    assert parse_qv_output("") ["overall"] is None


def test_lenient_json_repairs_latex_backslashes():
    from autodata.cs.parsing import repair_json_escapes
    from autodata.cs.judge import parse_judgments

    raw = '```json\n{"context": "c", "question": "q", "rubric": [{"criterion": "uses $\\alpha_d/\\alpha_z$ and \\gamma", "weight": 3, "category": "positive"}]}\n```'
    obj = parse_challenger_output(raw)
    # \\b, \\f, \\n, \\r, \\t, \\u are legal JSON escapes and stay ambiguous (\\beta -> backspace + "eta"); everything else is repaired
    assert obj and "alpha_d" in obj["rubric"][0]["criterion"] and "\\gamma" in obj["rubric"][0]["criterion"]
    # valid escapes are preserved by the repair
    assert repair_json_escapes('{"a": "line\\nbreak \\"quoted\\" \\\\ ok"}') == '{"a": "line\\nbreak \\"quoted\\" \\\\ ok"}'
    sat, ev = parse_judgments('{"criteria": [{"index": 1, "satisfied": true, "evidence": "$\\alpha$ holds"}]}', 1)
    assert sat == [True] and "alpha" in ev[0]


@pytest.mark.parametrize("prefix,suffix", [
    ("```json\n{invalid}\n```\n", ""),
    ('```json\n{"example": true}\n```\n', ""),
    ('{"example": true}\n```json\n', "\n```"),
    ('```json\n{"example": true}\n```\n```json\n', "\n```"),
])
def test_last_valid_top_level_json_wins_regardless_of_fencing(prefix, suffix):
    final = {"context": "c", "question": "q", "rubric": [], "nested": {"last": "nested"}}
    assert extract_json_object(prefix + json.dumps(final) + suffix) == final
    assert parse_challenger_output(prefix + json.dumps(final) + suffix) == final


def test_invalid_trailing_object_does_not_hide_last_valid_object():
    assert extract_json_object('```json\n{"final": {"nested": 1}}\n```\n{not json}') == {"final": {"nested": 1}}


@pytest.mark.parametrize("template", [
    "**{key}:** {value}", "**{key}**: **{value}**", "{key}: **{value}**",
    "- {key}: {value}", "* **{key}:** **{value}**", "+ {key}: {value}",
    "1. {key}: {value}", "2) **{key}:** {value}",
])
def test_qv_markdown_verdicts_keep_strict_four_check_rule(template):
    checks = [("CHECK_1_VERDICT", "NO_LEAKAGE"), ("CHECK_2_VERDICT", "GOOD"),
              ("CHECK_3_VERDICT", "PASS"), ("CHECK_4_VERDICT", "CONSISTENT"), ("OVERALL", "PASS")]
    lines = [template.format(key=key, value=value) for key, value in checks]
    assert parse_qv_output("\n".join(lines))["overall"] is True
    missing = parse_qv_output("\n".join(lines[1:]))
    assert missing["overall"] is False and missing["missing_checks"] == ["CHECK_1_VERDICT"]
    failed = parse_qv_output("\n".join(lines).replace("NO_LEAKAGE", "LEAKAGE"))
    assert failed["overall"] is False and failed["contradiction"] is True
