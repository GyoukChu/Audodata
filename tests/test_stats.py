import json

from autodata.cs.stats import _load_summaries, summarize_agentic, summarize_cot


def write_summary(root, directory, summary, filename="harness_summary.json"):
    path = root / directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary))


def test_only_direct_paper_workdirs_are_loaded_and_newest_finished_wins(tmp_path):
    older = {"paper_id": "duplicate", "finished_at": "2026-09-25T12:00:00", "completed": False}
    newest = {"paper_id": "duplicate", "finished_at": "2026-09-27T12:00:00", "completed": True}
    write_summary(tmp_path, "a_latest", newest)
    write_summary(tmp_path, "z_older", older)
    write_summary(tmp_path, "_archive", {"paper_id": "archive", "completed": True})
    write_summary(tmp_path, "_archive/duplicate.20260927-120000", newest)
    write_summary(tmp_path, "duplicate.old.20260927-120000", newest)
    write_summary(tmp_path, "_metadata", {"paper_id": "metadata"})
    write_summary(tmp_path, "nested/paper", {"paper_id": "nested"})
    write_summary(tmp_path, "not_a_paper", {"notes": "metadata"})
    write_summary(tmp_path, "array", [])
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken/harness_summary.json").write_text("bad JSON")
    assert _load_summaries(tmp_path, "harness_summary.json") == [newest]


def test_stats_exclude_archives_dedupe_and_count_incomplete_and_locked(tmp_path):
    accepted = {"paper_id": "good", "finished_at": "2026-09-27T12:00:00", "completed": True,
                "accepted": True, "final_accepted": True, "accepted_round": 1, "n_rounds": 1,
                "rounds": [{"failure_mode": "ACCEPTED", "weak_avg": 0.3, "strong_avg": 0.8, "gap": 0.5,
                            "question_chars": 120, "n_rubric_items": 10}],
                "usage": {"main_agent": {"prompt_tokens": 100, "completion_tokens": 50}}}
    write_summary(tmp_path, "good", accepted)
    write_summary(tmp_path, "older", {**accepted, "finished_at": "2026-09-26T12:00:00", "accepted": False,
                                     "final_accepted": False})
    write_summary(tmp_path, "_archive/good.20260927-120000", accepted)
    write_summary(tmp_path, "good.old.20260927-120000", accepted)
    write_summary(tmp_path, "bad", {"paper_id": "bad", "completed": False, "errors": ["crash"]})
    write_summary(tmp_path, "locked", {"paper_id": "locked", "completed": False, "skipped": "locked"})
    result = summarize_agentic(tmp_path)
    assert result["n_papers"] == 3 and result["n_accepted"] == 1 and result["n_final_accepted"] == 1
    assert result["n_incomplete"] == result["incomplete_runs"] == 2
    assert result["n_skipped_locked"] == 1 and result["papers_with_errors"] == 1
    assert result["table1"]["weak_solver_avg"] == 0.3 and result["tokens"]["main_agent_prompt"] == 100


def test_locked_skips_in_corpus_log_count_without_touching_other_runners_workspace(tmp_path):
    events = [{"paper_id": "skipped", "skipped": "locked"},
              {"paper_id": "skipped", "skipped": "locked"},
              {"paper_id": "finished", "skipped": "locked"},
              {"paper_id": "finished", "completed": True}]
    (tmp_path / "summary.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\nbroken tail")
    assert summarize_agentic(tmp_path)["n_skipped_locked"] == 1


def test_cot_stats_also_ignore_archives_and_duplicates(tmp_path):
    record = {"paper_id": "paper", "finished_at": "2026-09-27T12:00:00", "weak_avg": 0.2,
              "strong_avg": 0.7, "qv_passed": True}
    for directory in ("paper", "_archive/paper.20260927-120000", "paper.old.20260927-120000", "duplicate"):
        write_summary(tmp_path, directory, record, filename="cot_summary.json")
    assert summarize_cot(tmp_path)["n_papers"] == 1
