"""CoT integration tests using the agentic suite's fake challenger, QV, solvers and judge."""
import asyncio
import json
from itertools import groupby

import pytest

from autodata.cs import cot_baseline
from autodata.cs.run_paper import PaperInput
from autodata.cs.stats import summarize_cot

from .fake_openai_server import FakeServer, text_response
from .test_e2e_fake import CHALLENGER_JSON, PROMPTS, QV_PASS, SENTINEL_REF, Scenario, _config, _role


def _paper():
    return PaperInput(paper_id="paper", title="A retrieval paper", text="body " * 500)


@pytest.mark.parametrize("qv_passed", [True, False])
def test_single_shot_cot_evaluates_both_solvers_even_when_qv_fails(tmp_path, qv_passed):
    qv_text = QV_PASS if qv_passed else QV_PASS.replace("CHECK_2_VERDICT: GOOD", "CHECK_2_VERDICT: BAD").replace(
        "OVERALL: PASS", "OVERALL: FAIL")
    scenario = Scenario(qv_text=qv_text)
    root = tmp_path / "cot"
    with FakeServer(scenario) as server:
        cfg = _config(server.base_url, tmp_path)
        del cfg.models["main_agent"]  # The shared driver must create only the requested clients.
        results = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))

    summary, = results
    assert summary["errors"] == [] and summary["challenger_parsed"]
    assert summary["qv_passed"] is qv_passed
    assert summary["final_filter_passed"] is qv_passed
    assert summary["would_be_accepted"] is qv_passed
    assert summary["weak_avg"] == pytest.approx(0.15)
    assert summary["strong_avg"] == pytest.approx(1.0)
    assert summary["gap"] == pytest.approx(0.85)
    assert summary["final_filter"] == {
        "context_chars": len(CHALLENGER_JSON["context"]), "n_rubric_items": 11,
        "n_positive": 8, "n_negative": 3, "context_ok": True, "rubric_ok": True, "rubric_error": None,
        "weights_in_spec": True,
    }
    assert json.loads((root / "paper/cot_summary.json").read_text()) == summary
    assert [json.loads(line) for line in (root / "summary.jsonl").read_text().splitlines()] == results
    usage = json.loads((root / "usage_agents.json").read_text())
    assert set(usage) == {"challenger", "quality_verifier"}
    assert all(value["calls"] == 2 for value in usage.values())
    assert sorted(p.name for p in (root / "paper/trajectory").iterdir()) == [
        "challenger_01.jsonl", "quality_verifier_01.jsonl",
    ]
    stages = [req["model"] if role == "solver" else role for role, req in scenario.calls if role != "judge"]
    assert [(stage, len(list(calls))) for stage, calls in groupby(stages)] == [
        ("challenger", 2), ("qv", 2), ("qwen3.5-4b", 3), ("qwen3.8-27b", 3),
    ]
    assert all(SENTINEL_REF not in json.dumps(req) for role, req in scenario.calls if role in ("solver", "judge"))
    stats = summarize_cot(root)
    assert stats["n_papers"] == stats["n_evaluated"] == 1
    assert stats["table1_all_evaluated"]["weak_solver_avg"] == 0.15
    assert stats["table1_all_evaluated"]["strong_solver_avg"] == 1.0
    assert stats["table1_qv_passed"]["n"] == stats["table1_final_filter_passed"]["n"] == int(qv_passed)


def test_resume_reruns_error_archives_and_then_skips_completed_paper(tmp_path):
    class ParseFailure(Scenario):
        fail_challenger = True

        def __call__(self, req, idx):
            response = super().__call__(req, idx)
            if self.fail_challenger and _role(req) == "challenger":
                return text_response("Could not produce a QA package.")
            return response

    scenario = ParseFailure()
    root = tmp_path / "cot"
    with FakeServer(scenario) as server:
        cfg = _config(server.base_url, tmp_path)
        first, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))
        (root / "paper/marker").write_text("old attempt")
        scenario.fail_challenger = False
        second, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))
        n_calls = len(scenario.calls)
        resumed, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))

    assert first["errors"] == ["challenger output could not be parsed"]
    assert second["errors"] == [] and second["strong_avg"] == 1.0
    assert resumed == second and len(scenario.calls) == n_calls
    archived, = (root / "_archive").iterdir()
    assert archived.name.startswith("paper.")
    assert json.loads((archived / "cot_summary.json").read_text()) == first
    assert (archived / "marker").read_text() == "old attempt"
    assert not (root / "paper/marker").exists()
    assert [json.loads(line) for line in (root / "summary.jsonl").read_text().splitlines()] == [first, second]
    assert summarize_cot(root)["n_papers"] == 1
    assert not list(root.glob("*.old.*"))


@pytest.mark.parametrize("errors,strong_avg", [([], None), (["previous evaluator error"], 1.0)])
def test_resume_requires_both_averages_and_no_errors(tmp_path, errors, strong_avg):
    root = tmp_path / "cot"
    workdir = root / "paper"
    workdir.mkdir(parents=True)
    previous = {"paper_id": "paper", "errors": errors, "weak_avg": 0.15, "strong_avg": strong_avg}
    (workdir / "cot_summary.json").write_text(json.dumps(previous))
    with FakeServer(Scenario()) as server:
        cfg = _config(server.base_url, tmp_path)
        result, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))

    assert result["strong_avg"] == 1.0 and result["errors"] == []
    archived, = (root / "_archive").glob("paper.*/cot_summary.json")
    assert json.loads(archived.read_text()) == previous


def test_cli_no_resume_archives_completed_workdir(tmp_path):
    scenario = Scenario()
    root = tmp_path / "cot"
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"paper_id": "paper", "title": "A retrieval paper", "body_text": "body " * 500}) + "\n")
    config_path = tmp_path / "config.json"
    with FakeServer(scenario) as server:
        cfg = _config(server.base_url, tmp_path)
        config_path.write_text(cfg.model_dump_json())
        first, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))
        (root / "paper/marker").write_text("completed attempt")
        n_calls = len(scenario.calls)
        rc = cot_baseline.main([
            "--config", str(config_path), "--corpus", str(corpus), "--workdir-root", str(root),
            "--concurrency", "1", "--no-resume", "--prompts-dir", str(PROMPTS),
        ])

    assert rc == 0 and len(scenario.calls) == 2 * n_calls
    archived, = (root / "_archive").iterdir()
    assert json.loads((archived / "cot_summary.json").read_text()) == first
    assert (archived / "marker").read_text() == "completed attempt"
    assert not (root / "paper/marker").exists()
    current = json.loads((root / "paper/cot_summary.json").read_text())
    assert current["weak_avg"] == 0.15 and current["strong_avg"] == 1.0 and current["errors"] == []
    assert summarize_cot(root)["n_papers"] == 1


@pytest.mark.parametrize("incomplete", ["unparseable", "length"])
def test_incomplete_qv_is_an_error_and_resume_repairs_only_qv(tmp_path, incomplete):
    class IncompleteQV(Scenario):
        broken = True

        def __call__(self, req, idx):
            response = super().__call__(req, idx)
            if self.broken and _role(req) == "qv":
                return text_response("unfinished" if incomplete == "unparseable" else QV_PASS,
                                     finish_reason="stop" if incomplete == "unparseable" else "length")
            return response

    scenario = IncompleteQV()
    root = tmp_path / "cot"
    with FakeServer(scenario) as server:
        cfg = _config(server.base_url, tmp_path)
        first, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))
        assert first["errors"] == ["quality verifier incomplete"]
        assert first["qv_incomplete"] is True and first["qv_passed"] is False
        assert first["weak_avg"] == 0.15 and first["strong_avg"] == 1.0
        candidate = (root / "paper/eval_input.json").read_bytes()
        challenger = (root / "paper/trajectory/challenger_01.jsonl").read_bytes()
        before = len(scenario.calls)
        scenario.broken = False
        result, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))
        after = len(scenario.calls)
        assert asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1)) == [result]
        assert len(scenario.calls) == after
    assert result["errors"] == [] and result["qv_passed"] is True and result["qv_incomplete"] is False
    assert result["repaired"] == ["quality_verifier"]
    assert result["weak_report"] == first["weak_report"] and result["strong_report"] == first["strong_report"]
    assert all(role == "qv" for role, _ in scenario.calls[before:])
    assert (root / "paper/eval_input.json").read_bytes() == candidate
    assert (root / "paper/trajectory/challenger_01.jsonl").read_bytes() == challenger
    assert not (root / "_archive").exists()


@pytest.mark.parametrize("stage,errored", [("weak", False), ("weak", True), ("strong", False), ("strong", True)])
def test_resume_repairs_only_missing_or_errored_solver_on_frozen_candidate(tmp_path, monkeypatch, stage, errored):
    scenario = Scenario()
    root = tmp_path / "cot"
    real_eval = cot_baseline._eval
    fail = True
    modes = []

    async def evaluate(cfg, workdir, mode):
        modes.append(mode)
        if fail and mode == f"{stage}-only":
            return "incomplete stage", {"error": "solver failed"} if errored else None, 1
        return await real_eval(cfg, workdir, mode)

    monkeypatch.setattr(cot_baseline, "_eval", evaluate)
    with FakeServer(scenario) as server:
        cfg = _config(server.base_url, tmp_path)
        first, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))
        assert first["errors"]
        candidate = (root / "paper/eval_input.json").read_bytes()
        before = len(scenario.calls)
        fail = False
        modes.clear()
        result, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))
        after = len(scenario.calls)
        assert asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1)) == [result]
        assert len(scenario.calls) == after
    assert modes == [f"{stage}-only"]
    assert result["repaired"] == [stage] and result["errors"] == []
    assert result["weak_avg"] == 0.15 and result["strong_avg"] == 1.0
    assert result["gap"] == pytest.approx(0.85) and result["would_be_accepted"]
    assert not any(role in ("challenger", "qv") for role, _ in scenario.calls[before:])
    expected_model = "qwen3.8-27b" if stage == "strong" else "qwen3.5-4b"
    assert [req["model"] for role, req in scenario.calls[before:] if role == "solver"] == [expected_model] * 3
    assert (root / "paper/eval_input.json").read_bytes() == candidate
    assert not (root / "_archive").exists()
    other = "weak" if stage == "strong" else "strong"
    assert result[f"{other}_report"] == first[f"{other}_report"]


@pytest.mark.parametrize("field", ["context", "question", "rubric"])
def test_resume_does_not_repair_when_eval_input_differs(tmp_path, field):
    scenario = Scenario()
    root = tmp_path / "cot"
    with FakeServer(scenario) as server:
        cfg = _config(server.base_url, tmp_path)
        first, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))
        first.update(strong_avg=None, strong_report=None)
        (root / "paper/cot_summary.json").write_text(json.dumps(first))
        path = root / "paper/eval_input.json"
        candidate = json.loads(path.read_text())
        candidate[field] = [] if field == "rubric" else "different"
        path.write_text(json.dumps(candidate))
        before = len(scenario.calls)
        result, = asyncio.run(cot_baseline.run_corpus(cfg, [_paper()], root, concurrency=1))
    assert result["errors"] == [] and "repaired" not in result
    assert any(role == "challenger" for role, _ in scenario.calls[before:])
    archived, = (root / "_archive").glob("paper.*/eval_input.json")
    assert json.loads(archived.read_text()) == candidate
