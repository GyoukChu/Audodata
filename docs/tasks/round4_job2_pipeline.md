# Round-4 fix job 2 — pipeline resume/repair, CoT baseline, cohort provenance, corpus builder, limiter, docs

Owner files (ONLY these may be modified by this job): `src/autodata/cs/pipeline.py`, `src/autodata/cs/cot_baseline.py`,
`src/autodata/data/build_corpus.py`, `src/autodata/data/s2_client.py`, `src/autodata/cs/corpus_io.py`, docs
(`README.md`, `docs/RUNBOOK.md`, `docs/REPORT.md`, `docs/IMPLEMENTATION_SPEC.md`), and tests `tests/test_pipeline.py`,
`tests/test_cot_baseline.py`, `tests/test_data.py` (plus new test files named `tests/test_round4b_*.py`). Another job
edits `run_paper.py`, `evaluate_rubric.py`, `parsing.py`, `stats.py`, `rubric.py` concurrently: do not touch those, and
rely only on their current public interfaces (`PaperRun.run_final_qv_only(summary)`, `run_evaluator`,
`evaluator_deadline_s`, `final_filter`, `prepare_workspace`, `parse_qv_output`, `parse_challenger_output`). Never write
into `runs/`, `logs/`, `data/`; never call the Semantic Scholar API; never start model servers or the pipeline; unit
tests only (`.venv/bin/python -m pytest -q`).

Context: paper-exact reproduction of Agentic Self-Instruct (Autodata, arXiv 2606.25996). A production pilot is running
with the CURRENT code on `runs/pilot` (agentic) and `runs/pilot_cot` (CoT baseline); the new code must resume those
roots (cohort.json exists with a history of two config fingerprints; 22 summaries imported from an older run carry a
third fingerprint). Committed files must contain no absolute machine paths and no institution names other than
"mlilab".

## F. CoT baseline: incomplete QV is an error; repair completes the frozen candidate instead of regenerating
1. In `cot_baseline.run_one`, when the quality verifier does not finish (`stop_reason != "final"`) or its verdict is
   unparseable (`overall is None`), append `"quality verifier incomplete"` to `errors` (keep `qv_passed` False and
   `qv_incomplete: True`) so `is_done` is False.
2. Repair path. Generalise the pipeline's repair hook: `run_papers` gets an optional `needs_repair(cfg, prev) -> bool`
   callable (the agentic default stays `_needs_final_qv_repair`); when it returns True the driver does NOT archive the
   workdir and calls `run_one(..., repair=True, prev=prev)` (keep passing `repair_final_qv=True` for the agentic
   runner exactly as today). For the CoT baseline `needs_repair` is True when the previous summary holds the frozen
   candidate (`context`, `question`, `rubric` present) and `<workdir>/eval_input.json` exists with the same
   question/context/rubric, and at least one of: QV incomplete, weak report missing/errored, strong report
   missing/errored. `run_one(repair=True, prev=...)` then skips the challenger, re-runs ONLY the missing pieces on the
   frozen candidate: the quality verifier if incomplete, `--weak-only` if the weak report is missing or errored,
   `--strong-only --force-strong` if the strong report is missing or errored (the evaluator reuses the earlier weak run
   from `eval_attempts` for the same question hash), merges the previous fields, and records `repaired: [...]`.
   Motivation: items whose strong stage was skipped by a former gate bug must be completed, not regenerated (regeneration
   conditioned on a weak-solver outcome selects the single-shot baseline). Tests with the fake server: strong-missing
   repair, QV-incomplete repair, and no-repair when eval_input.json differs.
3. `cot_summary` `is_done` stays "no errors and both averages present".

## I. Cohort provenance (`pipeline.py`)
1. `_cohort_manifest`: also compare `corpus_sha1` (and `corpus_path` when both known); a change is a mismatch handled
   exactly like config/prompt changes (refused unless `--allow-config-mismatch`, then recorded in `history` with the
   corpus fields). New paper ids are still merged (cohort extension) and logged.
2. Imported results: when a resumed summary's `config_fingerprint` is not the manifest's current one and not in its
   `history`, record it once in `cohort.json["imports"]` as {config_fingerprint, harness_version, prompt_hashes_sha1 (if
   present in the summary), paper_ids: [...]} (append paper ids; atomic rewrite under the cohort lock). Keep the WARNING
   log line.
3. Per-paper text provenance on resume: if `<workdir>/paper.txt` exists and its sha1 differs from the current corpus
   text for that paper id (compare the text `prepare_workspace` would write), do not reuse the previous summary: log
   `resume: paper text changed, rerunning`, archive, rerun. Tests.

## I2. Repair provenance and terminal state (`pipeline.py`)
1. On a repair (agentic final QV or the new CoT repair), the merged summary currently receives the CURRENT config and
   prompt fingerprints (`s = {**s, **stamps}`), erasing the fingerprints under which the candidate was generated and
   evaluated. Keep the original `config_fingerprint` / `prompt_hashes_sha1` on the summary and record the repair's
   fingerprints separately as `repair_config_fingerprint`, `repair_prompt_hashes_sha1`, `repaired_at` (also when the
   repair fails). Tests.
2. `_agentic_done(summary)`: an accepted paper whose end-of-loop QV completed (`final_qv` is a dict with
   `qv_completed` True) is terminal regardless of `completed` / `agent_stop_reason` / historical errors (the other job
   moves resolved `final_qv:` errors into `resolved_errors`); otherwise keep the current rule. A resume after a
   successful repair must log `already done` and never archive the accepted workspace. Tests (real summaries shaped like
   the repair output: accepted True, final_accepted True, completed False, agent_stop_reason "error").

## J. Corpus builder resume validation (`build_corpus.py`)
On resume with an existing output file, validate every existing record against the active filters (year, field/CS
filter, text length bounds, release/shard set) before counting it toward completion; if any record violates them,
exit non-zero with a clear message unless `--allow-filter-mismatch` is passed (then log and keep them). Write the
build parameters to `<output>.meta.json` (created on first run, compared on resume). Tests: changed `min_year` resume
is refused; same parameters resume proceeds. Do not change the sampling/filter semantics themselves.

## K. Semantic Scholar limiter (`s2_client.py`)
1. The production floor for the minimum request interval is 3.0 s (the key allows 1 req/s cumulative across all
   endpoints; the user requires >= 3 s spacing): `MIN_INTERVAL_FLOOR = 3.0`, reject non-finite/negative values with a
   clear error. Tests that used a 1.0 s floor must be updated; keep fake-clock tests fast by injecting the clock, not by
   lowering the production floor.
2. The shared-file gate must honour valid long backoff deadlines up to the maximum backoff (900 s): sleep in bounded
   increments (e.g. 30 s) re-reading the gate, instead of ignoring deadlines beyond 600 s. Test the interaction of a
   Retry-After deadline written by one limiter instance and a fresh instance.

## L. Docs
1. README.md / docs/REPORT.md: replace the hard-coded "484 offline tests" with the current total from the suite you run.
2. docs/REPORT.md §6, the line "36% of the single-shot CoT questions would already satisfy the solver criteria
   (paper: 2%)": the paper's 2% is "only ~2% [of accepted questions] use a single round" (Sec 4 statistics), not a CoT
   acceptance rate. Reword to: "... satisfy the solver criteria; in the paper only ~2% of accepted questions needed a
   single agentic round (ours: 3 of 22)". Also fix the throughput line in §7 ("~1-2 days" for the 320-paper pilot):
   measured GLM-5.3 throughput is ~490 generated tokens/s aggregate at 8+2 concurrent papers, ~10-16 rounds/hour, so
   the 320-paper pilot needs roughly 3-4 days.
3. docs/RUNBOOK.md: document `--allow-filter-mismatch`, the CoT repair behaviour, the corpus provenance check, and the
   3 s limiter floor. docs/IMPLEMENTATION_SPEC.md: one paragraph on the CoT repair path and cohort imports.

## Finish
Run the FULL suite `.venv/bin/python -m pytest -q` (the other job's files may change under you; only fix failures in
your owner files and re-run at the end). Final message: what changed per item, test totals, and anything you
deliberately did not do.
