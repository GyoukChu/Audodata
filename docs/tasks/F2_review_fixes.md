FIX task (write-capable, gpt-6-astra). Repository: this directory (Autodata / Agentic Self-Instruct reproduction; read
docs/IMPLEMENTATION_SPEC.md first, then the files you own). A code review produced the findings below. You OWN these files
(and may create/extend their tests): src/autodata/llm/client.py, src/autodata/harness/agent.py, src/autodata/harness/tools.py,
src/autodata/cs/parsing.py, src/autodata/cs/pipeline.py, src/autodata/cs/stats.py, src/autodata/config.py,
src/autodata/cs/evaluate_rubric.py, src/autodata/cs/solvers.py, src/autodata/cs/judge.py, src/autodata/cs/corpus_io.py,
tests/*. Do NOT edit src/autodata/cs/run_paper.py, src/autodata/cs/cot_baseline.py, src/autodata/cs/prompts.py or prompts/*
(being edited in parallel). Keep public signatures; new parameters keyword-only with defaults. No package installs.
IMPORTANT: a long run is in progress that spawns `python -I -m autodata.cs.evaluate_rubric` subprocesses continuously, so
evaluate_rubric.py / solvers.py / judge.py / parsing.py must stay import-safe and backward compatible at every save (edit
atomically: write the full file in one go, run the tests immediately).

FIXES (all with regression tests; run `.venv/bin/python -m pytest -q` at the end and paste the summary):
1. llm/client.py — a completion with empty content, no tool_calls and finish_reason == "length" is NOT a transient error:
   return the ChatResult (no retry). Only retry empty responses whose finish_reason is not "length". Count the tokens of
   every attempt (including discarded ones) in usage_totals AND expose per-call `usage` summed over attempts so
   AgentResult.usage reflects real spend (agent.py adds completion.usage).
2. harness/agent.py — context budget: count content + reasoning (+reasoning_content, counted once) + tool_call arguments
   (json) for every message; when over budget, first drop `reasoning`/`reasoning_content` from assistant messages older
   than the `keep_recent_tool_results` most recent turns, then elide old tool results as today. Also accept
   `context_budget_tokens: int | None = None` (keyword-only): when set, use the last completion.usage["prompt_tokens"]
   (plus chars/4 of what was appended since) as the measure instead of chars. Keep the sequence valid; log "elide" events.
   Update docs/IMPLEMENTATION_SPEC.md §2.3 to list the "elide" event kind.
3. harness/tools.py — nothing required beyond keeping `make_write_tool(ws, allow=...)` (the harness owner will use it).
   Optional: reject `--weak-only` together with `--strong-only` in _validate_evaluator_args.
4. cs/parsing.py — parse_qv_output must accept markdown-bold, bulleted and numbered verdict lines
   ("**CHECK_1_VERDICT:** NO_LEAKAGE", "- CHECK_2_VERDICT: GOOD", "1. OVERALL: PASS", "OVERALL: **PASS**"); keep the strict
   all-four-checks rule. extract_json_object must consider BOTH fenced and unfenced candidate spans (an invalid or
   illustrative fenced block must not hide the real unfenced object); still prefer the last top-level valid object, fenced
   first when both are valid.
5. config.py — AppConfig.acceptance must validate overrides: AcceptancePreset.model_validate({**base.model_dump(),
   **overrides}) with `extra="forbid"` and numeric constraints (fractions in [0,1], gap in [0,1]); a bad override must raise
   at config load (load_config), not at prompt rendering. Remove `RunConfig.seed` (dead) OR wire it: add
   `seed: int | None = None` to ModelEndpoint that flows through LLMClient (chat(seed=...) default) and SyncClient;
   choose "wire it" and document that per-request sampling reproducibility is best-effort.
6. cs/pipeline.py — PaperLock: use fcntl.flock(LOCK_EX | LOCK_NB) on the lock file (auto-released on process death;
   see data/s2_client.py for the pattern), acquire it INSIDE the concurrency semaphore (queued papers must stay available
   to other runners), release in finally; a stale lock file with no live flock holder must be acquirable. Archiving: on
   ANY rerun of an existing workdir (resume-after-error or --no-resume), move it to `<root>/_archive/<paper>.<YYYYmmdd-HHMMSS>`
   (create _archive/) BEFORE PaperRun.run; expose `archive_workdir(workdir, root)` for reuse by cot_baseline (the harness
   owner will call it). Wrap the per-paper coroutine so an exception escaping PaperRun.run produces a summary line with
   errors=[...] and completed=False instead of aborting the corpus (still write usage_agents.json).
7. cs/stats.py — `_load_summaries` must only read direct children of the run root that are paper workdirs: skip names
   starting with "_" and names containing ".old."; dedupe by paper_id keeping the newest finished_at. Add
   `summarize_agentic` fields: n_incomplete, n_skipped_locked. Tests with a fixture tree including archives.
8. cs/evaluate_rubric.py — in --strong-only mode, if _latest_weak finds no matching weak run OR the matching weak run did
   not pass (weak_passed False), do NOT run the strong stage: print the report with `STRONG_SKIPPED: no passing weak result
   for this question (run --weak-only first)`, `NO_WEAK_RESULT` when none, `ACCEPTANCE: FAILED (...)`, exit 0, and write
   report.json with strong_attempts []. Add `weak_source_report` and `weak_source_run_dir` to report.json for every
   strong-only report (already partly there) and keep `models.weak_solver` from the source report. Also verify in
   _latest_weak that the source report's `models`, `prompts_dir` and `config_sha1` equal the current run's values (skip
   otherwise) and record `config_sha1`, `models`, `prompts_dir`, `input_path`, `input_sha1` in every report (some exist).
9. cs/corpus_io.py — import `paper_text` and `iter_corpus` from autodata.data directly (hard dependency); delete the
   divergent fallbacks. Add tests/test_corpus_io.py (safe_paper_id, dedupe, offset/limit/paper_ids, min_chars).
10. cs/solvers.py + cs/judge.py — share the retry policy and request-building with llm/client.py: move
    `retryable(exc)`, `backoff(attempt)` and a `build_request_kwargs(endpoint, *, chat_template_kwargs=None, seed=None)`
    helper into autodata/llm/client.py (module-level, pure) and use them from SyncClient/run_solver/run_judge; keep
    SyncClient as a thin wrapper (sync OpenAI client) — do NOT introduce a per-call client construction. Retry set for the
    evaluator = the client's (408/409/429/5xx/connection). Keep the truncation semantics: run_solver never retries a
    finish_reason=="length" no-answer response; run_judge steps the reasoning effort down on truncation only when
    `effort_fallback` is True.
11. Tests: add tests/test_pipeline.py (lock with flock + stale lock, archive on rerun, previous_summary rules,
    summary line on crash) and tests/test_stats.py, tests/test_corpus_io.py; extend tests for items 1, 2, 4, 5, 8, 10.
FINAL REPORT: per-file changes, test summary, any signature additions, and anything you deliberately left out.
