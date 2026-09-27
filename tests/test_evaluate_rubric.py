from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
import threading
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from autodata.config import AcceptancePreset, ModelEndpoint, PRESETS
from autodata.cs import solvers
from autodata.cs.evaluate_rubric import _assess, assess_attempts, compute_question_hash, legacy_question_hash, main
from autodata.cs.rubric import parse_rubric, score_response
from autodata.cs.solvers import PROMPTS_DIR, SyncClient, build_solver_messages, run_solver
from tests.fake_openai_server import FakeServer, ScriptedResponder, make_transport, text_response

RUBRIC = [{"criterion": f"criterion {i}", "weight": weight}
          for i, weight in enumerate([40, 25, 15, 20, -10], 1)]
DATA = {"context": "Research context with {literal braces}.", "question": "Why does it work?",
        "rubric": RUBRIC, "reference_answer": "SECRET_REFERENCE_NEVER_SENT", "extra": "ignored"}
MARKS = {"weak": [True, False, False, False, False],
         "strong": [True, True, True, False, False],
         "quarter": [False, True, False, False, False],
         "zero": [False] * 5}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(solvers, "retry_backoff", lambda _: None)


def setup_cli(tmp_path, *, n_attempts=3, preset="prose_s31", base_url="http://fake/v1"):
    input_path, config_path, output_dir = tmp_path / "input.json", tmp_path / "api.json", tmp_path / "attempts"
    input_path.write_text(json.dumps(DATA))
    config = {
        role: {"base_url": base_url, "model": model, "api_key": "private-key",
               "timeout_s": 73, "max_retries": 9}
        for role, model in (("weak_solver", "weak"), ("strong_solver", "strong"), ("judge", "judge"))
    }
    config.update(acceptance=PRESETS[preset].model_dump(),
                  eval={"n_attempts": n_attempts, "solver_retries": 1, "judge_retries": 1},
                  prompts_dir="prompts/cs")
    config_path.write_text(json.dumps(config))
    return ["--input", str(input_path), "--config", str(config_path),
            "--output-dir", str(output_dir)], config, output_dir


def save_config(argv, config):
    Path(argv[argv.index("--config") + 1]).write_text(json.dumps(config))


def judgment(answer):
    return text_response(json.dumps({"criteria": [
        {"index": index, "satisfied": flag, "evidence": "quote" if flag else ""}
        for index, flag in enumerate(MARKS[answer], 1)
    ]}))


def responder(body, _):
    if body["model"] == "judge":
        return judgment(body["messages"][-1]["content"].split("## Response\n")[1])
    return text_response(body["model"], reasoning="separate chain", finish_reason="length")


def read_report(stdout):
    last_line = stdout.splitlines()[-1]
    assert last_line.startswith("REPORT_PATH: ")
    path = Path(last_line.removeprefix("REPORT_PATH: "))
    assert path.is_absolute()
    assert path.with_name("report.txt").read_text() == stdout[:stdout.rfind("REPORT_PATH: ")]
    return json.loads(path.read_text()), path


def test_both_modes_complete_artifacts_and_no_answer_leakage(tmp_path, capsys):
    argv, _, output = setup_cli(tmp_path)
    calls = []

    def record(body, index):
        calls.append(body)
        return responder(body, index)

    assert main(argv, transport=make_transport(record)) == 0
    stdout = capsys.readouterr().out
    report, path = read_report(stdout)
    assert "WEAK_PASSED (weak_avg 40.0% < 50.0%)" in stdout
    assert "STRONG_PASSED" in stdout and "GAP_PASSED" in stdout
    assert "ACCEPTANCE: ALL_SOLVER_CRITERIA_PASSED" in stdout
    assert report["all_passed"] is True
    assert report["weak_avg"] == 0.4 and report["strong_avg"] == 0.8 and report["gap"] == 0.4
    assert report["failure_reasons"] == []
    assert len(list(output.glob("run_*/attempt_*.json"))) == 6
    assert len(calls) == 12
    assert "SECRET_REFERENCE_NEVER_SENT" not in json.dumps(calls)
    assert "private-key" not in path.read_text()
    assert report["started_at"] <= report["completed_at"] and report["latency_s"] >= 0
    assert report["question_hash"] == legacy_question_hash(DATA)
    assert report["question_hash_canonical"] == compute_question_hash(DATA)
    for name in ("input", "config"):
        source = Path(argv[argv.index(f"--{name}") + 1]).resolve()
        assert report[f"{name}_path"] == str(source)
        assert report[f"{name}_sha1"] == hashlib.sha1(source.read_bytes()).hexdigest()
    assert report["n_attempts_required"] == 3
    for role in ("weak", "strong"):
        assert [a["index"] for a in report[f"{role}_attempts"]] == [1, 2, 3]
        for attempt in report[f"{role}_attempts"]:
            assert json.loads(Path(attempt["attempt_path"]).read_text()) == attempt
            assert attempt["messages"] == build_solver_messages(DATA["context"], DATA["question"])
            assert attempt["response_text"] == role and attempt["reasoning"] == "separate chain"
            assert attempt["finish_reason"] == "length"  # Truncated answers still get grades.
            assert attempt["usage"]["completion_tokens"] == 5
            assert attempt["judge"]["usage"]["completion_tokens"] == 5
            assert attempt["latency_s"] >= 0 and attempt["judge"]["latency_s"] >= 0
            assert attempt["total_latency_s"] >= attempt["latency_s"]
            assert json.loads(attempt["judge"]["raw"])["criteria"]
            assert attempt["breakdown"] == asdict(score_response(parse_rubric(RUBRIC), attempt["satisfied"]))


def test_solver_messages_verbatim_and_sampling_timeout_reasoning(tmp_path):
    endpoint = ModelEndpoint(
        base_url="http://fake/v1", model="weak", temperature=1.0, top_p=0.95,
        max_tokens=123, presence_penalty=0.2, top_k=20, min_p=0.01, repetition_penalty=1.1,
        chat_template_kwargs={"enable_thinking": True},
        extra_body={"top_k": 30, "chat_template_kwargs": {"mode": "x"}, "custom": True},
    )
    response = text_response("<think>inline one</think>weak<think>inline two</think>", finish_reason="length")
    response["choices"][0]["message"]["reasoning"] = "reasoning variant"
    scripted = ScriptedResponder([response])
    transport = make_transport(scripted)
    original_handler = transport.handle_request
    timeouts = []

    def capture(request):
        timeouts.append(request.extensions["timeout"])
        return original_handler(request)

    transport.handle_request = capture
    with SyncClient(endpoint, transport=transport) as client:
        attempt = run_solver(client, "literal {question}", "q", timeout=12.5)
    body = scripted.requests[0]
    assert body["messages"] == [{"role": "user", "content": (PROMPTS_DIR / "solver_user.md").read_text().format(
        context="literal {question}", question="q")}]
    for key in ("temperature", "top_p", "max_tokens", "presence_penalty", "min_p", "repetition_penalty"):
        assert body[key] == getattr(endpoint, key)
    assert body["top_k"] == 30 and body["custom"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "mode": "x"}
    assert timeouts[0]["read"] == 12.5
    assert attempt.response_text == "weak" and attempt.reasoning == "reasoning variant"
    assert attempt.inline_reasoning == ["inline one", "inline two"]
    assert attempt.finish_reason == "length" and attempt.error is None


# "unfinished_think" (finish_reason=length, no answer) is no longer a retryable failure: it is graded as-is (score 0),
# see test_truncated_solver_response_scores_zero_without_judge.
@pytest.mark.parametrize("failure", ["empty", "think_only", 429, 503, "connection", "timeout"])
def test_solver_retries_then_grades_and_counts_all_usage(tmp_path, capsys, failure):
    argv, _, _ = setup_cli(tmp_path, n_attempts=1)
    calls = []

    def flaky(body, index):
        calls.append(body)
        if index == 0:
            if failure == "empty":
                return text_response(" \n")
            if failure == "think_only":
                return text_response("<think>not an answer</think>")
            if failure == "unfinished_think":
                return text_response("<think>still reasoning", finish_reason="length")
            if failure == "connection":
                raise httpx.ConnectError("offline")
            if failure == "timeout":
                raise httpx.ReadTimeout("slow")
            return failure, {"error": {"message": "temporary"}}
        return responder(body, index)

    assert main(argv + ["--weak-only"], transport=make_transport(flaky)) == 0
    report, _ = read_report(capsys.readouterr().out)
    assert len(calls) == 3 and calls[0] == calls[1]
    assert report["weak_passed"] is True
    assert len(report["weak_attempts"][0]["requests"]) == 2
    expected_tokens = 10 if failure in ("empty", "think_only") else 5
    assert report["weak_attempts"][0]["usage"]["completion_tokens"] == expected_tokens


def test_threads_overlap_and_weak_judging_finishes_before_strong(tmp_path, capsys):
    argv, _, _ = setup_cli(tmp_path)
    barrier = threading.Barrier(3, timeout=5)
    lock = threading.Lock()
    weak_judged = []
    tids = set()

    def concurrent(body, index):
        if body["model"] != "judge":
            with lock:
                tids.add(threading.get_ident())
                if body["model"] == "strong":
                    assert len(weak_judged) == 3
            barrier.wait()
        elif body["messages"][-1]["content"].endswith("## Response\nweak"):
            with lock:
                weak_judged.append(True)
        return responder(body, index)

    assert main(argv, transport=make_transport(concurrent)) == 0
    assert len(tids) >= 3
    assert read_report(capsys.readouterr().out)[0]["all_passed"] is True


def test_weak_failure_skips_strong_and_returns_zero(tmp_path, capsys):
    argv, _, _ = setup_cli(tmp_path)
    calls = []

    def easy(body, index):
        calls.append(body)
        return text_response("strong") if body["model"] == "weak" else responder(body, index)

    assert main(argv, transport=make_transport(easy)) == 0
    stdout = capsys.readouterr().out
    report, _ = read_report(stdout)
    assert "WEAK_FAILED: TOO EASY" in stdout and "STRONG_SKIPPED" in stdout
    assert "ACCEPTANCE: FAILED" in stdout
    assert not any(body["model"] == "strong" for body in calls)
    assert report["strong_avg"] is None and report["strong_attempts"] == []


def test_strong_failure_and_gap_failure_are_normal_reports(tmp_path, capsys):
    argv, _, _ = setup_cli(tmp_path)

    def hard(body, index):
        return text_response("weak") if body["model"] == "strong" else responder(body, index)

    assert main(argv, transport=make_transport(hard)) == 0
    stdout = capsys.readouterr().out
    report, _ = read_report(stdout)
    assert "STRONG_FAILED" in stdout and "GAP_FAILED" in stdout and "failed on strong" in stdout
    assert report["all_passed"] is False and report["strong_avg"] == 0.4


def test_latest_matching_weak_run_is_loaded_and_rechecked(tmp_path, capsys):
    argv, config, output = setup_cli(tmp_path, n_attempts=1)
    assert main(argv + ["--weak-only"], transport=make_transport(responder)) == 0
    first, _ = read_report(capsys.readouterr().out)

    def quarter(body, index):
        return text_response("quarter") if body["model"] == "weak" else responder(body, index)

    assert main(argv + ["--weak-only"], transport=make_transport(quarter)) == 0
    latest, latest_path = read_report(capsys.readouterr().out)
    input_path = Path(argv[1])
    input_path.write_text(json.dumps({**DATA, "question": "A different question"}))
    assert main(argv + ["--weak-only"], transport=make_transport(responder)) == 0
    capsys.readouterr()
    input_path.write_text(json.dumps({**DATA, "reference_answer": "changed but irrelevant"}))
    # A partial report from an interrupted later run is ignored.
    interrupted = output / "run_004_weak-only"
    interrupted.mkdir()
    (interrupted / "report.json").write_text('{"question_hash":')
    # Derived scores cannot override the stored judgments.
    latest["weak_attempts"][0]["score"] = 0.99
    latest["weak_attempts"][0]["breakdown"]["earned"] = 99
    latest_path.write_text(json.dumps(latest))
    assert main(argv + ["--strong-only"], transport=make_transport(responder)) == 0
    report, path = read_report(capsys.readouterr().out)
    assert report["question_hash"] == first["question_hash"] == latest["question_hash"]
    assert report["weak_source_report"] == str(latest_path)
    assert report["weak_avg"] == 0.25 and report["gap"] == 0.55
    assert report["weak_passed"] is True
    assert path.parent.name == "run_005_strong-only"


@pytest.mark.parametrize("mismatch", [
    "current_attempt_count", "stored_attempt_count", "preset", "threshold", "eval_setting",
    "missing_acceptance", "missing_eval", "attempt_error", "attempt_error_type",
])
def test_latest_weak_rejects_incompatible_or_incomplete_reports(tmp_path, capsys, mismatch):
    argv, config, _ = setup_cli(tmp_path, n_attempts=1)
    assert main(argv + ["--weak-only"], transport=make_transport(responder)) == 0
    previous, previous_path = read_report(capsys.readouterr().out)
    if mismatch == "current_attempt_count":
        config["eval"]["n_attempts"] = 3
    elif mismatch == "stored_attempt_count":
        previous["weak_attempts"] *= 2
    elif mismatch == "preset":
        config["acceptance"] = PRESETS["deployed_c1"].model_dump()
    elif mismatch == "threshold":
        config["acceptance"]["weak_avg_max"] = 0.45
    elif mismatch == "eval_setting":
        config["eval"]["judge_retries"] = 2
    elif mismatch.startswith("missing_"):
        previous.pop(mismatch.removeprefix("missing_"))
    elif mismatch == "attempt_error":
        previous["weak_attempts"][0]["error"] = "failed"
    elif mismatch == "attempt_error_type":
        previous["weak_attempts"][0]["error_type"] = "JUDGE_ERROR"
    previous_path.write_text(json.dumps(previous))
    save_config(argv, config)
    assert main(argv + ["--strong-only"], transport=make_transport(responder)) == 0
    stdout = capsys.readouterr().out
    report, _ = read_report(stdout)
    assert "NO_WEAK_RESULT" in stdout
    assert report["weak_attempts"] == [] and report["weak_source_report"] is None
    assert report["weak_avg"] is None and report["gap"] is None
    assert report["all_passed"] is False


def test_latest_weak_skips_newer_incompatible_report(tmp_path, capsys):
    argv, _, _ = setup_cli(tmp_path, n_attempts=1)
    assert main(argv + ["--weak-only"], transport=make_transport(responder)) == 0
    _, first_path = read_report(capsys.readouterr().out)
    assert main(argv + ["--weak-only"], transport=make_transport(responder)) == 0
    latest, latest_path = read_report(capsys.readouterr().out)
    latest["acceptance"] = PRESETS["deployed_c1"].model_dump()
    latest_path.write_text(json.dumps(latest))
    assert main(argv + ["--strong-only"], transport=make_transport(responder)) == 0
    report, _ = read_report(capsys.readouterr().out)
    assert report["weak_source_report"] == str(first_path)
    assert report["all_passed"] is True


def test_compute_question_hash_is_stable_and_ignores_extra_fields():
    before = deepcopy(DATA)
    reordered = {"reference_answer": "changed", "rubric": deepcopy(RUBRIC),
                 "question": DATA["question"], "context": DATA["context"]}
    expected = compute_question_hash(DATA)
    assert compute_question_hash(DATA) == compute_question_hash(reordered) == expected
    assert DATA == before
    for key, value in (("context", "different"), ("question", "different"), ("rubric", RUBRIC[:-1])):
        assert compute_question_hash({**DATA, key: value}) != expected


@pytest.mark.parametrize("mode,weak_answer,strong_answer", [
    ([], "weak", "strong"), ([], "strong", "strong"), ([], "weak", "weak"),
    (["--weak-only"], "weak", "strong"), (["--strong-only"], "weak", "strong"),
])
def test_assess_attempts_matches_cli_and_ignores_derived_values(tmp_path, capsys, mode, weak_answer, strong_answer):
    argv, config, _ = setup_cli(tmp_path, n_attempts=1)

    def answer(body, index):
        if body["model"] == "judge":
            return responder(body, index)
        return text_response(weak_answer if body["model"] == "weak" else strong_answer)

    assert main(argv + mode, transport=make_transport(answer)) == 0
    report, _ = read_report(capsys.readouterr().out)
    weak, strong = deepcopy(report["weak_attempts"]), deepcopy(report["strong_attempts"])
    for attempt in weak + strong:
        attempt.update(score=12345, breakdown={"earned": 12345}, all_passed=True)
    snapshot = deepcopy((weak, strong, RUBRIC, config["acceptance"]))
    assessment = assess_attempts(RUBRIC, weak, strong, config["acceptance"])
    required = {"weak_avg", "strong_avg", "gap", "weak_passed", "strong_passed",
                "gap_passed", "all_passed", "failure_reasons"}
    assert required <= assessment.keys()
    assert assessment == {key: report[key] for key in assessment}
    assert (weak, strong, RUBRIC, config["acceptance"]) == snapshot
    assert assess_attempts(parse_rubric(RUBRIC), weak, strong, PRESETS["prose_s31"]) == assessment


@pytest.mark.parametrize("bad_satisfied", [None, [True], [1] * 5, ["true"] * 5])
def test_assess_attempts_rejects_malformed_judgments(bad_satisfied):
    with pytest.raises(ValueError, match="one boolean per criterion"):
        assess_attempts(RUBRIC, [{"satisfied": bad_satisfied}], [], PRESETS["prose_s31"])


def test_assess_attempts_does_not_accept_attempt_errors():
    weak = [{"satisfied": MARKS["weak"]}, {"error": "failed", "satisfied": MARKS["weak"]}]
    assessment = assess_attempts(RUBRIC, weak, [{"satisfied": MARKS["strong"]}], PRESETS["prose_s31"])
    assert assessment["weak_avg"] is None and assessment["weak_passed"] is None
    assert assessment["all_passed"] is False and "no complete weak result" in assessment["failure_reasons"]


@pytest.mark.parametrize("seed_source", ["endpoint", "extra_body", "unset"])
def test_solver_judge_seeds_reach_wire_and_attempt_files(tmp_path, capsys, seed_source):
    argv, config, _ = setup_cli(tmp_path, n_attempts=1)
    seeds = {"weak_solver": 0, "strong_solver": 1234, "judge": 5678}
    for role, seed in seeds.items():
        if seed_source == "endpoint":
            config[role]["seed"] = seed
        elif seed_source == "extra_body":
            config[role]["extra_body"] = {"seed": seed}
    save_config(argv, config)
    calls = []

    def record(body, index):
        calls.append(body)
        return responder(body, index)

    assert main(argv, transport=make_transport(record)) == 0
    report, _ = read_report(capsys.readouterr().out)
    for body in calls:
        role = body["model"] + ("_solver" if body["model"] != "judge" else "")
        expected = seeds[role] if seed_source != "unset" else None
        if role in ("weak_solver", "strong_solver") and expected is not None:
            # solver attempts get DISTINCT derived seeds (review: identical seeds collapse the 3 attempts into one sample)
            assert body.get("seed") is not None and body.get("seed") != expected
        else:
            assert body.get("seed") == expected
            assert ("seed" in body) == (expected is not None)
    for role in ("weak", "strong"):
        attempt = json.loads(Path(report[f"{role}_attempts"][0]["attempt_path"]).read_text())
        if seed_source != "unset":
            from autodata.cs.evaluate_rubric import attempt_seed
            data = json.loads(Path(argv[argv.index("--input") + 1]).read_text())
            assert attempt["requests"][0]["seed"] == attempt_seed(seeds[f"{role}_solver"], data, role, attempt["index"])
        else:
            assert attempt["requests"][0]["seed"] is None
        assert attempt["judge"]["requests"][0]["seed"] == (seeds["judge"] if seed_source != "unset" else None)


def test_solver_records_seeds_for_failed_and_retried_requests(tmp_path, capsys):
    argv, config, _ = setup_cli(tmp_path, n_attempts=1)
    config["weak_solver"]["seed"] = 0
    save_config(argv, config)

    def retry(body, index):
        return (503, {"error": {"message": "temporary"}}) if index == 0 else responder(body, index)

    assert main(argv + ["--weak-only"], transport=make_transport(retry)) == 0
    report, _ = read_report(capsys.readouterr().out)
    attempt = json.loads(Path(report["weak_attempts"][0]["attempt_path"]).read_text())
    derived = attempt["requests"][0]["seed"]
    assert derived is not None and [request["seed"] for request in attempt["requests"]] == [derived, derived]  # retries keep the attempt seed


def test_strong_without_matching_weak_omits_gap(tmp_path, capsys):
    argv, _, _ = setup_cli(tmp_path)
    assert main(argv + ["--strong-only"], transport=make_transport(responder)) == 0
    stdout = capsys.readouterr().out
    report, _ = read_report(stdout)
    assert "NO_WEAK_RESULT: run --weak-only first" in stdout
    assert "GAP_" not in stdout and "\ngap:" not in stdout
    assert report["gap"] is None and report["all_passed"] is False


def assessment(preset, weak, strong):
    def attempts(values):
        return [{"breakdown": {"earned": value, "penalty": 0, "max_positive": 100}, "error": None}
                for value in values]

    report = {"weak_attempts": attempts(weak), "strong_attempts": attempts(strong), "error": None}
    _assess(report, preset)
    return report


@pytest.mark.parametrize("preset,weak,strong,expected", [
    ("prose_s31", [50, 50, 50], [80, 80, 80], (False, True, True)),
    ("prose_s31", [49, 49, 49], [69, 69, 69], (True, True, True)),
    ("prose_s31", [45, 45, 45], [65, 65, 65], (True, True, True)),
    ("prose_s31", [44, 44, 44], [64, 64, 64], (True, False, True)),
    ("prose_s31", [49, 49, 49], [68, 68, 68], (True, True, False)),
    ("prose_s31", [0, 0, 90], [0, 100, 100], (True, True, True)),
    ("prose_s31", [40, 40, 40], [100, 100, 100], (True, True, True)),
    ("deployed_c1", [65, 65, 65], [85, 85, 85], (True, True, True)),
    ("deployed_c1", [66, 66, 66], [90, 90, 90], (False, True, True)),
    ("deployed_c1", [75, 30, 30], [80, 80, 80], (True, True, True)),
    ("deployed_c1", [76, 30, 30], [80, 80, 80], (False, True, True)),
    ("deployed_c1", [0, 30, 30], [80, 80, 80], (False, True, True)),
    ("deployed_c1", [40, 40, 40], [60, 60, 60], (True, True, True)),
    ("deployed_c1", [39, 39, 39], [59, 59, 59], (True, False, True)),
    ("deployed_c1", [40, 40, 40], [95, 95, 95], (True, False, True)),
    ("deployed_c1", [40, 40, 40], [94, 94, 94], (True, True, True)),
    ("deployed_c1", [40, 40, 40], [0, 100, 100], (True, False, True)),
])
def test_all_acceptance_boundaries(preset, weak, strong, expected):
    report = assessment(PRESETS[preset], weak, strong)
    assert tuple(report[f"{part}_passed"] for part in ("weak", "strong", "gap")) == expected
    assert report["all_passed"] == all(expected)


def test_acceptance_uses_fields_even_for_custom_preset():
    preset = AcceptancePreset(name="custom", weak_avg_max=0.3, weak_avg_max_inclusive=True,
                              weak_attempt_max=0.35, strong_avg_min=0.5, strong_avg_max=0.9,
                              gap_min=0.3)
    assert assessment(preset, [30] * 3, [60] * 3)["all_passed"] is True
    assert assessment(preset, [30] * 3, [59] * 3)["gap_passed"] is False


@pytest.mark.parametrize("kind,code", [("SOLVER_ERROR", 2), ("JUDGE_ERROR", 3)])
def test_exhausted_errors_are_first_stdout_line_with_audit_files(tmp_path, capsys, kind, code):
    argv, _, output = setup_cli(tmp_path, n_attempts=1)
    calls = []

    def broken(body, index):
        calls.append(body)
        if kind == "SOLVER_ERROR" or body["model"] == "judge":
            return text_response("")
        return responder(body, index)

    assert main(argv, transport=make_transport(broken)) == code
    stdout = capsys.readouterr().out
    assert stdout.splitlines()[0].startswith(kind + ":")
    report, _ = read_report(stdout)
    assert report["error"]["kind"] == kind and report["all_passed"] is False
    assert len(calls) == (2 if kind == "SOLVER_ERROR" else 3)
    assert len(list(output.glob("run_*/attempt_*.json"))) == 1
    attempt = report["weak_attempts"][0]
    assert attempt["error_type"] == kind
    requests = attempt["requests"] if kind == "SOLVER_ERROR" else attempt["judge"]["requests"]
    assert len(requests) == 2
    assessment = assess_attempts(RUBRIC, report["weak_attempts"], report["strong_attempts"], report["acceptance"])
    assert assessment == {key: report[key] for key in assessment}


def test_any_single_solver_failure_fails_stage(tmp_path, capsys):
    argv, _, _ = setup_cli(tmp_path)

    def one_bad(body, index):
        if index == 0:
            return 400, {"error": {"message": "invalid request"}}
        return responder(body, index)

    assert main(argv, transport=make_transport(one_bad)) == 2
    stdout = capsys.readouterr().out
    assert stdout.startswith("SOLVER_ERROR:")
    report, _ = read_report(stdout)
    assert len(report["weak_attempts"]) == 3
    assert sum(attempt["error"] is not None for attempt in report["weak_attempts"]) == 1
    assert report["weak_avg"] is None and report["strong_attempts"] == []


@pytest.mark.parametrize("data,code,label", [
    ({**DATA, "rubric": []}, 4, "RUBRIC_ERROR"),
    ({**DATA, "rubric": [{"criterion": "x", "weight": 0}]}, 4, "RUBRIC_ERROR"),
    ({"question": "q", "rubric": RUBRIC}, 5, "INPUT_ERROR"),
    ({**DATA, "context": None}, 5, "INPUT_ERROR"),
    ([], 5, "INPUT_ERROR"),
])
def test_invalid_input_codes_and_first_stdout_line(tmp_path, capsys, data, code, label):
    argv, _, output = setup_cli(tmp_path)
    Path(argv[1]).write_text(json.dumps(data))
    assert main(argv, transport=make_transport(responder)) == code
    assert capsys.readouterr().out.startswith(label + ":")
    assert not output.exists()


@pytest.mark.parametrize("extra", [["--timeout", "0"], ["--timeout", "nan"], ["--timeout", "inf"],
                                    ["--timeout", "-1"], ["--timeout", "bad"],
                                    ["--weak-only", "--strong-only"]])
def test_bad_cli_arguments(tmp_path, capsys, extra):
    argv, _, _ = setup_cli(tmp_path)
    assert main(argv + extra) == 5
    assert capsys.readouterr().out.startswith("INPUT_ERROR:")


def test_missing_cli_arguments(capsys):
    assert main([]) == 5
    assert capsys.readouterr().out.startswith("INPUT_ERROR:")


@pytest.mark.parametrize("problem", ["missing", "bad_json", "bad_config", "bad_prompt", "zero_attempts", "negative_retries"])
def test_input_and_configuration_errors(tmp_path, capsys, problem):
    argv, config, _ = setup_cli(tmp_path)
    if problem == "missing":
        Path(argv[1]).unlink()
    elif problem == "bad_json":
        Path(argv[1]).write_text("not JSON")
    elif problem == "bad_config":
        config["judge"]["max_tokens"] = {"secret": "private-key"}
    elif problem == "bad_prompt":
        config["prompts_dir"] = "missing-prompts"
    elif problem == "zero_attempts":
        config["eval"]["n_attempts"] = 0
    else:
        config["eval"]["judge_retries"] = -1
    save_config(argv, config)
    assert main(argv) == 5
    stdout = capsys.readouterr().out
    assert stdout.startswith("INPUT_ERROR:") and "private-key" not in stdout


def test_timeout_is_per_solver_request_and_does_not_override_judge(tmp_path, capsys):
    argv, _, _ = setup_cli(tmp_path, n_attempts=1)
    transport = make_transport(responder)
    original_handler = transport.handle_request
    timeouts = []

    def capture(request):
        timeouts.append((json.loads(request.content)["model"], request.extensions["timeout"]["read"]))
        return original_handler(request)

    transport.handle_request = capture
    assert main(argv + ["--weak-only", "--timeout", "0.25"], transport=transport) == 0
    capsys.readouterr()
    assert timeouts == [("weak", 0.25), ("judge", 73)]


@pytest.mark.parametrize("invocation", ["module", "isolated_module", "isolated_script"])
def test_cli_subprocess_from_arbitrary_working_directory(tmp_path, invocation):
    try:
        server = FakeServer(responder)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM, errno.EAFNOSUPPORT):
            pytest.skip(f"sandbox cannot bind localhost sockets: {exc}")
        raise
    with server:
        argv, _, _ = setup_cli(tmp_path, base_url=server.base_url)
        other_cwd = tmp_path / "unrelated-directory"
        other_cwd.mkdir()
        if invocation == "isolated_script":
            entrypoint = ["-I", str(Path(__file__).resolve().parents[1] / "src/autodata/cs/evaluate_rubric.py")]
        else:
            entrypoint = (["-I"] if invocation == "isolated_module" else []) + ["-m", "autodata.cs.evaluate_rubric"]
        result = subprocess.run([sys.executable, *entrypoint, *argv],
                                cwd=other_cwd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report, _ = read_report(result.stdout)
    assert report["all_passed"] is True
    assert result.stderr == ""


def test_truncated_solver_response_scores_zero_without_judge(tmp_path):
    """finish_reason=length with no final answer is graded as-is (score 0) and not retried or judged."""
    import json
    from pathlib import Path
    from autodata.cs import evaluate_rubric as er
    from tests.fake_openai_server import make_transport, text_response

    calls = {"solver": 0, "judge": 0}

    def responder(req, idx):
        sys_msg = next((m for m in req["messages"] if m["role"] == "system"), None)
        if sys_msg is None:
            calls["solver"] += 1
            return text_response("", reasoning="thinking " * 50, finish_reason="length")
        calls["judge"] += 1
        return text_response('{"criteria": []}')

    rubric = [{"criterion": "a", "weight": 3, "category": "positive"}, {"criterion": "b", "weight": -1, "category": "negative"}]
    inp = tmp_path / "eval_input.json"
    inp.write_text(json.dumps({"context": "c", "question": "q", "rubric": rubric}))
    ep = {"base_url": "http://fake/v1", "model": "m", "max_tokens": 32, "timeout_s": 5, "chat_template_kwargs": {}}
    cfg = tmp_path / "api_config.json"
    cfg.write_text(json.dumps({"weak_solver": {**ep, "model": "weak"}, "strong_solver": {**ep, "model": "strong"}, "judge": {**ep, "model": "judge"},
                               "acceptance": {"name": "prose_s31", "weak_avg_max": 0.5, "weak_avg_max_inclusive": False, "weak_attempt_max": None,
                                              "weak_no_zero": False, "strong_avg_min": 0.65, "strong_avg_max": None, "strong_no_zero": False, "gap_min": 0.2},
                               "eval": {"n_attempts": 2, "timeout_s": 5, "judge_retries": 1, "solver_retries": 2},
                               "prompts_dir": str(Path(__file__).resolve().parents[1] / "prompts" / "cs")}))
    rc = er.main(["--input", str(inp), "--weak-only", "--output-dir", str(tmp_path / "out"), "--config", str(cfg), "--timeout", "5"],
                 transport=make_transport(responder))
    assert rc == 0
    assert calls["solver"] == 2 and calls["judge"] == 0
    report = json.loads(next((tmp_path / "out").glob("run_*/report.json")).read_text())
    assert report["weak_avg"] == 0.0 and report["weak_passed"] is True
    assert all(a["truncated_no_answer"] and a["score"] == 0.0 and a["error"] is None for a in report["weak_attempts"])


@pytest.mark.parametrize("weak_result", ["missing", "failed", "marked_failed"])
def test_strong_only_requires_passing_weak_without_any_requests(tmp_path, capsys, weak_result):
    argv, _, _ = setup_cli(tmp_path, n_attempts=1)
    source = None
    if weak_result != "missing":
        def weak(body, index):
            if body["model"] == "weak":
                return text_response("strong" if weak_result == "failed" else "weak")
            return responder(body, index)

        assert main(argv + ["--weak-only"], transport=make_transport(weak)) == 0
        previous, source = read_report(capsys.readouterr().out)
        if weak_result == "marked_failed":
            previous["weak_passed"] = False
            source.write_text(json.dumps(previous))
    calls = []

    def record(body, index):
        calls.append(body)
        return responder(body, index)

    assert main(argv + ["--strong-only"], transport=make_transport(record)) == 0
    stdout = capsys.readouterr().out
    report, path = read_report(stdout)
    assert calls == []
    assert "STRONG_SKIPPED: no passing weak result for this question (run --weak-only first)" in stdout
    assert "ACCEPTANCE: FAILED (" in stdout
    assert ("NO_WEAK_RESULT" in stdout) == (weak_result == "missing")
    assert report["strong_attempts"] == [] and report["all_passed"] is False
    assert report["weak_source_report"] == (str(source) if source else None)
    assert report["weak_source_run_dir"] == (str(source.parent) if source else None)
    assert not list(path.parent.glob("attempt_strong_*.json"))


@pytest.mark.parametrize("field", ["models", "prompts_dir", "config_sha1"])
@pytest.mark.parametrize("missing", [False, True])
def test_latest_weak_requires_matching_provenance(tmp_path, capsys, field, missing):
    argv, _, _ = setup_cli(tmp_path, n_attempts=1)
    assert main(argv + ["--weak-only"], transport=make_transport(responder)) == 0
    previous, path = read_report(capsys.readouterr().out)
    if missing:
        previous.pop(field)
    elif field == "models":
        previous[field]["judge"] = "different judge"
    else:
        previous[field] = "different"
    path.write_text(json.dumps(previous))
    calls = []

    def record(body, index):
        calls.append(body)
        return responder(body, index)

    assert main(argv + ["--strong-only"], transport=make_transport(record)) == 0
    stdout = capsys.readouterr().out
    report, _ = read_report(stdout)
    assert "NO_WEAK_RESULT" in stdout and calls == []
    assert report["weak_source_report"] is None and report["weak_source_run_dir"] is None


@pytest.mark.parametrize("mode", [[], ["--weak-only"], ["--strong-only"]])
def test_every_report_records_input_and_config_provenance(tmp_path, capsys, mode):
    argv, config, _ = setup_cli(tmp_path, n_attempts=1)
    assert main(argv + mode, transport=make_transport(responder)) == 0
    report, _ = read_report(capsys.readouterr().out)
    assert report["config_sha1"] == hashlib.sha1(Path(argv[3]).read_bytes()).hexdigest()
    assert report["input_sha1"] == hashlib.sha1(Path(argv[1]).read_bytes()).hexdigest()
    assert report["input_path"] == str(Path(argv[1]).resolve())
    assert report["prompts_dir"] == str(PROMPTS_DIR.resolve())
    assert report["models"] == {role: config[role]["model"] for role in ("weak_solver", "strong_solver", "judge")}


def test_three_attempts_get_distinct_seeds_when_seeded(tmp_path):
    from autodata.cs.evaluate_rubric import attempt_seed
    data = {"context": "c", "question": "q", "rubric": [{"criterion": "a", "weight": 2, "category": "positive"}]}
    seeds = {attempt_seed(7, data, "weak", i) for i in (1, 2, 3)}
    assert len(seeds) == 3 and all(0 <= x < 2**31 for x in seeds)
    assert attempt_seed(7, data, "weak", 1) == attempt_seed(7, data, "weak", 1)  # reproducible
    assert attempt_seed(7, data, "strong", 1) != attempt_seed(7, data, "weak", 1)
    assert attempt_seed(None, data, "weak", 1) is None


@pytest.mark.parametrize('force', [False, True])
def test_force_strong_runs_after_failing_weak(tmp_path, capsys, force):
    argv, _, output = setup_cli(tmp_path, n_attempts=1)
    calls = []

    def easy(body, index):
        calls.append(body['model'])
        if body['model'] == 'judge':
            return judgment('strong')  # weak_avg = 0.8, so weak-first gating fails
        return responder(body, index)

    transport = make_transport(easy)
    assert main(argv + ['--weak-only'], transport=transport) == 0
    weak, _ = read_report(capsys.readouterr().out)
    assert weak['weak_avg'] == 0.8 and weak['weak_passed'] is False
    calls.clear()
    assert main(argv + ['--strong-only'] + (['--force-strong'] if force else []), transport=transport) == 0
    strong, _ = read_report(capsys.readouterr().out)
    assert bool(strong['strong_attempts']) is force
    assert ('strong' in calls) is force
    assert strong['all_passed'] is False
    calls.clear()
    assert main(argv, transport=transport) == 0
    both, _ = read_report(capsys.readouterr().out)
    assert both['weak_avg'] == 0.8 and not both['strong_attempts']
    assert 'strong' not in calls


def test_canonical_hash_normalises_strings_and_rubric_metadata():
    original = {'context': 'Café\nnext', 'question': 'Why?\nThen?',
                'rubric': [{'criterion': 'A café\ncriterion', 'weight': 8, 'category': 'positive'}]}
    variant = {'question': 'Why?  \r\nThen?\t', 'context': 'Cafe\u0301 \r\nnext\t', 'extra': 'ignored',
               'rubric': [{'ignored': 1, 'weight': '+8', 'criterion': 'A cafe\u0301\t\r\ncriterion  '}]}
    canonical = json.dumps(original, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    assert compute_question_hash(original) == compute_question_hash(variant) == hashlib.sha1(canonical.encode()).hexdigest()
    legacy = json.dumps({key: variant[key] for key in ('context', 'question', 'rubric')})
    assert legacy_question_hash(variant) == hashlib.sha1(legacy.encode()).hexdigest()
    assert legacy_question_hash(variant) != compute_question_hash(variant)


def test_strong_stage_reuses_legacy_weak_report(tmp_path, capsys):
    argv, _, _ = setup_cli(tmp_path, n_attempts=1)
    assert main(argv + ['--weak-only'], transport=make_transport(responder)) == 0
    weak, path = read_report(capsys.readouterr().out)
    weak['question_hash'] = legacy_question_hash(DATA)
    weak.pop('question_hash_canonical', None)  # Reports from before canonical hashing have only the legacy field.
    path.write_text(json.dumps(weak))
    assert main(argv + ['--strong-only'], transport=make_transport(responder)) == 0
    strong, _ = read_report(capsys.readouterr().out)
    assert strong['weak_source_report'] == str(path)
    assert strong['all_passed'] is True
    assert strong['question_hash'] == legacy_question_hash(DATA)
    assert strong['question_hash_canonical'] == compute_question_hash(DATA)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux parent-death signal')
@pytest.mark.parametrize('invocation', [
    ['-m', 'autodata.cs.evaluate_rubric', '--help'],
    ['-c', 'import autodata.cs.evaluate_rubric'],
])
def test_evaluator_exits_at_import_on_parent_pid_mismatch(tmp_path, invocation):
    env = {**os.environ, 'AUTODATA_PARENT_PID': '0'}
    result = subprocess.run([sys.executable, '-I', *invocation], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 5
    assert 'parent pid mismatch' in result.stderr
    assert result.stdout == ''
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux prctl')
def test_parent_death_signal_checks_prctl_result(monkeypatch, capsys):
    import ctypes
    from types import SimpleNamespace
    from autodata.cs.evaluate_rubric import _die_with_parent
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **kw: SimpleNamespace(prctl=lambda *a: -1))
    with pytest.raises(SystemExit) as exc:
        _die_with_parent()
    assert exc.value.code == 5 and 'PR_SET_PDEATHSIG failed' in capsys.readouterr().err
