REFACTOR task (write-capable, gpt-6-astra). Repository: this directory (Autodata reproduction). Read docs/IMPLEMENTATION_SPEC.md,
then src/autodata/cs/run_paper.py, pipeline.py, cot_baseline.py, stats.py and tests/test_e2e_fake.py. Goal: remove the
duplication between the CoT baseline and the agentic pipeline WITHOUT changing behaviour or any public CLI flag, keeping
`python -m pytest -q` green. You own: src/autodata/cs/cot_baseline.py, src/autodata/cs/pipeline.py, src/autodata/cs/run_paper.py
(only the extractions listed below), src/autodata/cs/stats.py (only the summarize_cot reader if needed), tests/test_cot_baseline.py
(new), tests/test_pipeline.py (extend). No package installs. A long run is in progress (it imported the old code at start; the
evaluator subprocesses use evaluate_rubric.py which you must not touch).

1. run_paper.py: extract `async def run_evaluator(workdir: Path, argv_abs: list[str], *, deadline_s: float) -> tuple[str, str, int | None]`
   (stdout, stderr, returncode; isolated `python -I -m autodata.cs.evaluate_rubric`, PYTHON*-scrubbed env, wait_for + kill,
   CancelledError cleanup) and `def evaluator_deadline_s(cfg, mode, solver_timeout_s) -> float`, and
   `def final_filter(cfg, context, rubric) -> dict` (context length + rubric shape: total in [10,20], >=4 positive, >=3 negative,
   returning the same dict shape run_final_qv uses today, with n_rubric_items). Put the rubric-shape thresholds into RunConfig
   (final_rubric_min_items=10, final_rubric_max_items=20, final_rubric_min_positive=4, final_rubric_min_negative=3) — config.py is
   owned by another task in flight; if config.py already has these fields use them, otherwise read them with getattr defaults and
   note it in the report. PaperRun.run_evaluate_rubric and run_final_qv must call the extracted helpers.
2. pipeline.py: extract a generic corpus driver `async def run_papers(cfg, papers, root, *, concurrency, resume, retry_errors,
   summary_filename, is_done: Callable[[dict], bool], run_one: Callable[[PaperInput, Path, dict[str, LLMClient], PromptSet, Path, log], Awaitable[dict]],
   roles: tuple[str, ...], prompts_dir=None) -> list[dict]` implementing: per-paper lock (whatever PaperLock is by then — flock),
   archive-on-rerun into <root>/_archive, atomic summary_filename write, summary.jsonl append, usage_agents.json, crash-to-summary-line.
   run_corpus (agentic) becomes a thin call; cot_baseline.run_corpus calls it with is_done = "no errors and weak_avg/strong_avg present".
3. cot_baseline.py: use run_evaluator/evaluator_deadline_s/final_filter and the shared driver; keep `run_one` (the CoT per-paper
   logic) and `main()` with identical flags/behaviour; delete the duplicated code. Keep the resume semantics (rerun errored papers).
4. tests/test_cot_baseline.py: e2e with the fake server (reuse tests/test_e2e_fake.py's Scenario/fake responses: single-shot
   challenger -> QV -> weak -> strong, both stats fields populated; a QV-fail case; a resume case that reruns an errored paper and
   archives the old workdir under _archive; --no-resume archives). tests/test_pipeline.py: the shared driver (lock, archive, crash line).
FINAL REPORT: what was extracted, the exact diff summary per file, test results, anything left duplicated on purpose.
