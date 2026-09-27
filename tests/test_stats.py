import json

import pytest

from autodata.cs.stats import _load_summaries, print_report, summarize_agentic, summarize_cot


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


@pytest.mark.parametrize("kind", ["agentic", "cot"])
def test_manifest_counts_pending_and_limits_rates_to_requested_cohort(tmp_path, capsys, kind):
    filename = "harness_summary.json" if kind == "agentic" else "cot_summary.json"
    summarize = summarize_agentic if kind == "agentic" else summarize_cot
    (tmp_path / "cohort.json").write_text(json.dumps({"paper_ids": ["accepted", "rejected", "incomplete", "pending"]}))
    good = {"paper_id": "accepted", "completed": True, "accepted": True, "would_be_accepted": True,
            "weak_avg": 0.2, "strong_avg": 0.8}
    write_summary(tmp_path, "accepted", good, filename)
    write_summary(tmp_path, "duplicate", good, filename)
    write_summary(tmp_path, "rejected", {**good, "paper_id": "rejected", "accepted": False,
                                        "would_be_accepted": False}, filename)
    write_summary(tmp_path, "incomplete", {"paper_id": "incomplete", "completed": False}, filename)
    write_summary(tmp_path, "unrequested", {**good, "paper_id": "unrequested"}, filename)
    write_summary(tmp_path, "_archive/pending.old", {**good, "paper_id": "pending"}, filename)
    result = summarize(tmp_path)
    assert result["n_papers"] == 3 and result["n_requested"] == 4
    assert result["n_completed"] == 2 and result["n_incomplete"] == result["n_pending"] == 1
    assert result["acceptance_rate_completed"] == 0.5
    assert result["acceptance_rate_cohort"] == 0.25
    print_report(result)
    output = capsys.readouterr().out
    assert "requested 4  completed 2  incomplete 1  pending 1" in output
    assert "acceptance rate (completed): 0.5" in output and "acceptance rate (cohort): 0.25" in output


@pytest.mark.parametrize("kind", ["agentic", "cot"])
def test_without_manifest_counts_paper_directories_without_summary_as_pending(tmp_path, kind):
    filename = "harness_summary.json" if kind == "agentic" else "cot_summary.json"
    summarize = summarize_agentic if kind == "agentic" else summarize_cot
    write_summary(tmp_path, "done", {"paper_id": "done", "accepted": True, "would_be_accepted": True,
                                     "weak_avg": 0.1, "strong_avg": 0.8}, filename)
    for directory in ("pending", "_locks", "_archive", "done.old.20260927", ".internal"):
        (tmp_path / directory).mkdir()
    result = summarize(tmp_path)
    assert result["n_papers"] == result["n_completed"] == result["n_pending"] == 1
    assert result["n_incomplete"] == 0 and result["n_requested"] == 2
    assert result["acceptance_rate_completed"] == 1 and result["acceptance_rate_cohort"] == 0.5
    if kind == "agentic":
        assert result["acceptance_rate"] == 1  # Keep the legacy field's denominator.


@pytest.mark.parametrize("requested", [[], ["pending"]])
def test_empty_or_entirely_pending_cohort_has_defined_rates(tmp_path, requested):
    (tmp_path / "cohort.json").write_text(json.dumps({"paper_ids": requested}))
    result = summarize_agentic(tmp_path)
    assert result["n_requested"] == result["n_pending"] == len(requested)
    assert result["n_completed"] == result["n_incomplete"] == 0
    assert result["acceptance_rate_completed"] is None
    assert result["acceptance_rate_cohort"] == (0 if requested else None)


def test_completed_acceptance_rate_excludes_incomplete_accepted_paper(tmp_path):
    (tmp_path / "cohort.json").write_text(json.dumps({"paper_ids": ["accepted", "rejected"]}))
    write_summary(tmp_path, "accepted", {"paper_id": "accepted", "accepted": True, "completed": False})
    write_summary(tmp_path, "rejected", {"paper_id": "rejected", "accepted": False, "completed": True})
    result = summarize_agentic(tmp_path)
    assert result["acceptance_rate_completed"] == 0 and result["acceptance_rate_cohort"] == 0.5


def test_unreadable_requested_summary_remains_pending(tmp_path):
    (tmp_path / "cohort.json").write_text(json.dumps({"paper_ids": ["paper"]}))
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper/harness_summary.json").write_text("interrupted JSON")
    assert summarize_agentic(tmp_path)["n_pending"] == 1


@pytest.mark.parametrize('legacy', [False, True])
def test_stats_use_earliest_accepting_candidate_report(tmp_path, legacy):
    from autodata.cs.evaluate_rubric import compute_question_hash, legacy_question_hash
    candidate = {'context': 'Context', 'question': 'Why?',
                 'rubric': [{'criterion': 'Explain.', 'weight': 5, 'category': 'positive'}]}
    summary = {'paper_id': 'accepted', 'accepted': True, 'final_accepted': True, 'accepted_round': 1,
               'rounds': [{**candidate, 'weak_avg': 0.1696, 'strong_avg': 0.8, 'gap': 0.6304,
                           'weak_scores': [0.1696], 'strong_scores': [0.8],
                           'question_chars': 4, 'n_rubric_items': 1}]}
    # The directory need not equal paper_id; use the selected summary's actual workspace.
    write_summary(tmp_path, 'workspace', summary)
    report = {'question_hash': (legacy_question_hash if legacy else compute_question_hash)(candidate),
              'all_passed': True, 'started_at': '2026-01-02T00:00:00+00:00',
              'weak_avg': 0.0994, 'strong_avg': 0.9, 'gap': 0.8006,
              'weak_attempts': [{'score': 0.0994}], 'strong_attempts': [{'score': 0.9}]}
    reports = {
        'run_007_strong-only': report,
        'run_001_both': {**report, 'started_at': '2026-01-03T00:00:00+00:00', 'weak_avg': 0.2},
        'run_003_both': {**report, 'started_at': '2026-01-01T00:00:00+00:00', 'question_hash': 'foreign', 'weak_avg': 0.3},
        'run_004_weak-only': {**report, 'started_at': '2026-01-01T00:00:00+00:00', 'all_passed': False, 'weak_avg': 0.4},
    }
    for directory, data in reports.items():
        path = tmp_path / 'workspace' / 'eval_attempts' / directory / 'report.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data))
    archived = {**summary, 'paper_id': 'archived', 'final_accepted': False}
    write_summary(tmp_path, 'archived', archived)
    result = summarize_agentic(tmp_path)
    assert result['n_report_backed'] == result['n_summary_backed'] == 1
    assert result['table1_after_final_qv']['weak_solver_avg'] == 0.0994
    assert result['table1_after_final_qv']['strong_solver_avg'] == 0.9
    assert result['table1_after_final_qv']['gap'] == 0.8006
    assert result['table1']['weak_solver_avg'] == 0.1345
    item = next(item for item in result['items'] if item['paper_id'] == 'accepted')
    assert item['scores_source'] == 'report' and 'run_007' in item['scores_report']
    assert item['weak_scores'] == [0.0994] and item['strong_scores'] == [0.9]
    fallback = next(item for item in result['items'] if item['paper_id'] == 'archived')
    assert fallback['scores_source'] == 'summary' and fallback['weak_scores'] == [0.1696]
    assert json.loads((tmp_path / 'workspace' / 'harness_summary.json').read_text()) == summary


def test_qv_unbound_counts_only_explicit_false_on_accepted_round(tmp_path):
    for paper_id, accepted, recorded in [
        ('unbound', True, {'qv_bound': False}),
        ('bound', True, {'qv_bound': True}),
        ('legacy', True, {'qv_missing': ['question']}),
        ('unknown', True, {'qv_bound': None}),
        ('rejected', False, {'qv_bound': False}),
    ]:
        write_summary(tmp_path, paper_id, {
            'paper_id': paper_id, 'accepted': accepted, 'accepted_round': 2 if accepted else None,
            'rounds': [{'qv_bound': False}, recorded],
        })
    result = summarize_agentic(tmp_path)
    assert result['n_accepted'] == 4 and result['n_qv_unbound'] == 1
