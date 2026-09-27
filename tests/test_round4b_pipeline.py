"""Resume provenance and repair regressions; all workspaces are temporary."""
import json
from pathlib import Path

import pytest

from autodata.config import AppConfig
from autodata.cs import pipeline
from autodata.cs.run_paper import PaperInput, prepare_workspace

from .test_pipeline import _shared_run, fake_clients


@pytest.mark.parametrize("change", ["corpus_sha1", "corpus_path"])
async def test_corpus_mismatch_requires_override_and_records_history(tmp_path, fake_clients, change):
    cfg = AppConfig(models={})
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("original corpus")
    root = tmp_path / "run"
    await _shared_run(cfg, [], root, corpus_path=corpus)
    original = json.loads((root / "cohort.json").read_text())
    if change == "corpus_sha1":
        corpus.write_text("changed corpus")
    else:
        corpus = tmp_path / "other.jsonl"
        corpus.write_text("original corpus")
    with pytest.raises(SystemExit, match=change):
        await _shared_run(cfg, [], root, corpus_path=corpus)
    assert json.loads((root / "cohort.json").read_text()) == original
    await _shared_run(cfg, [], root, corpus_path=corpus, allow_config_mismatch=True)
    updated = json.loads((root / "cohort.json").read_text())
    assert updated["history"][0][change] == original[change]
    assert updated["history"][-1][change] == updated[change] != original[change]
    await _shared_run(cfg, [], root, corpus_path=corpus)
    assert json.loads((root / "cohort.json").read_text()) == updated


async def test_legacy_history_retains_previous_corpus_on_next_revision(tmp_path, fake_clients):
    cfg = AppConfig(models={})
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("original corpus")
    root = tmp_path / "run"
    await _shared_run(cfg, [], root, corpus_path=corpus)
    path = root / "cohort.json"
    previous = json.loads(path.read_text())
    previous["history"] = [{"config_fingerprint": previous["config_fingerprint"],
                            "prompt_hashes": previous["prompt_hashes"]}]
    path.write_text(json.dumps(previous))
    cfg.run.seed = 42
    await _shared_run(cfg, [], root, allow_config_mismatch=True)
    updated = json.loads(path.read_text())
    assert updated["corpus_sha1"] == previous["corpus_sha1"]
    assert updated["history"][0] == previous["history"][0]
    assert all(entry["corpus_sha1"] == previous["corpus_sha1"] for entry in updated["history"][1:])


async def test_imported_fingerprints_accumulate_ids_once_under_cohort_lock(tmp_path, fake_clients, capsys):
    cfg = AppConfig(models={})
    pipeline._cohort_manifest(cfg, [], tmp_path, Path(cfg.prompts_dir))
    cfg.run.seed = 42
    manifest = pipeline._cohort_manifest(cfg, [], tmp_path, Path(cfg.prompts_dir), allow_config_mismatch=True)
    known = manifest["history"][0]["config_fingerprint"]
    papers = [PaperInput(pid, "", "") for pid in ("import1", "import2", "known")]
    for paper in papers:
        workdir = tmp_path / paper.paper_id
        workdir.mkdir()
        summary = {"paper_id": paper.paper_id, "completed": True, "harness_version": "old-harness",
                   "config_fingerprint": known if paper.paper_id == "known" else "imported-config",
                   "prompt_hashes_sha1": "imported-prompts"}
        (workdir / "cot_summary.json").write_text(json.dumps(summary))
    for _ in range(2):
        await _shared_run(cfg, papers, tmp_path)
    updated = json.loads((tmp_path / "cohort.json").read_text())
    assert updated["history"] == manifest["history"]
    assert updated["imports"] == [{"config_fingerprint": "imported-config", "harness_version": "old-harness",
                                   "prompt_hashes_sha1": "imported-prompts", "paper_ids": ["import1", "import2"]}]
    output = capsys.readouterr().out
    assert "WARNING previous run used a different config" in output
    assert "cohort: extending by 3 paper ids" in output


@pytest.mark.parametrize("changed", [False, True])
async def test_resume_compares_exact_workspace_text_including_truncation(tmp_path, fake_clients, changed, capsys):
    cfg = AppConfig(models={role: {"base_url": "http://fake", "model": role}
                            for role in ("weak_solver", "strong_solver", "judge")},
                    run={"paper_text_max_chars": 20})
    paper = PaperInput("paper", "Title", "αβγ " * 100)
    workdir = tmp_path / paper.paper_id
    prepare_workspace(cfg, paper, workdir, Path(cfg.prompts_dir).resolve())
    # No endpoint probes are needed for a skipped summary; reruns use a stub too.
    cfg.models.clear()
    previous = {"paper_id": paper.paper_id, "completed": True, "sentinel": "previous"}
    (workdir / "cot_summary.json").write_text(json.dumps(previous))
    if changed:
        paper.text = "different corpus text"
    result, = await _shared_run(cfg, [paper], tmp_path)
    if changed:
        assert "sentinel" not in result
        archived, = (tmp_path / "_archive").glob("paper.*/cot_summary.json")
        assert json.loads(archived.read_text()) == previous
        assert "resume: paper text changed, rerunning" in capsys.readouterr().out
    else:
        assert result == previous
        assert not (tmp_path / "_archive").exists()


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_repairs_preserve_candidate_fingerprints_even_on_failure(tmp_path, fake_clients, agentic, fails):
    cfg = AppConfig(models={})
    paper = PaperInput("paper", "", "text")
    workdir = tmp_path / paper.paper_id
    workdir.mkdir()
    filename = "harness_summary.json" if agentic else "cot_summary.json"
    previous = {"paper_id": "paper", "accepted": True, "final_accepted": False, "completed": False,
                "agent_stop_reason": "error", "errors": ["historical error"], "final_qv": {"qv_completed": False},
                "config_fingerprint": "original-config", "prompt_hashes_sha1": "original-prompts"}
    (workdir / filename).write_text(json.dumps(previous))

    async def run(*args, repair=False, prev=None, **kwargs):
        assert repair and prev == previous
        assert kwargs == ({"repair_final_qv": True} if agentic else {})
        if fails:
            raise RuntimeError("repair failed")
        return {"final_accepted": True, "final_qv": {"qv_completed": True, "passed": True},
                "config_fingerprint": "must-not-overwrite", "prompt_hashes_sha1": "must-not-overwrite"}

    result, = await pipeline.run_papers(
        cfg, [paper], tmp_path, concurrency=1, resume=True, retry_errors=True, summary_filename=filename,
        is_done=pipeline._agentic_done, run_one=run, roles=(),
        needs_repair=None if agentic else lambda cfg, prev: True,
    )
    assert result["config_fingerprint"] == "original-config"
    assert result["prompt_hashes_sha1"] == "original-prompts"
    assert result["repair_config_fingerprint"] == pipeline.config_fingerprint(cfg)
    assert result["repair_prompt_hashes_sha1"] != "original-prompts"
    assert result["repaired_at"]
    assert json.loads((workdir / filename).read_text()) == result
    assert json.loads((tmp_path / "summary.jsonl").read_text()) == result
    assert not (tmp_path / "_archive").exists()
    if agentic and not fails:
        assert await pipeline.run_corpus(cfg, [paper], tmp_path, concurrency=1) == [result]
        assert not (tmp_path / "_archive").exists()


@pytest.mark.parametrize("passed", [False, True])
async def test_completed_final_qv_is_terminal_after_error_stop(tmp_path, fake_clients, passed, capsys):
    workdir = tmp_path / "paper"
    workdir.mkdir()
    previous = {"paper_id": "paper", "accepted": True, "final_accepted": passed, "completed": False,
                "agent_stop_reason": "error", "errors": ["historical error"],
                "final_qv": {"qv_completed": True, "passed": passed}}
    (workdir / "harness_summary.json").write_text(json.dumps(previous))
    result = await pipeline.run_corpus(AppConfig(models={}), [PaperInput("paper", "", "")], tmp_path, concurrency=1)
    assert result == [previous]
    assert "already done" in capsys.readouterr().out
    assert not (tmp_path / "_archive").exists()


def test_drain_file_stops_new_papers(tmp_path):
    """A DRAIN file in the run root makes the driver skip papers it has not started (no workspace, no lock)."""
    import asyncio

    root = tmp_path / "run"
    root.mkdir()
    (root / "DRAIN").write_text("")
    cfg = AppConfig(models={})
    papers = [PaperInput("p1", "t", "Title: t\n\nbody " * 50)]

    async def run_one(*args, **kwargs):  # pragma: no cover - must never be called while draining
        raise AssertionError("run_one must not start while DRAIN is present")

    results = asyncio.run(pipeline.run_papers(
        cfg, papers, root, concurrency=1, resume=True, retry_errors=True, summary_filename="harness_summary.json",
        is_done=pipeline._agentic_done, run_one=run_one, roles=(), health_wait_s=0,
    ))
    assert results[0]["skipped"] == "drained"
    assert not (root / "p1").exists()
    lines = [json.loads(line) for line in (root / "summary.jsonl").read_text().splitlines()]
    assert lines[0]["skipped"] == "drained"
