FIX task (write-capable, gpt-6-astra). Repository: this directory (Autodata reproduction; read docs/IMPLEMENTATION_SPEC.md and
docs/knowledge-base/review-decisions.md). The final review (docs/tasks/R2_final_review.md) produced findings; the orchestrator
accepted the ones below for YOUR files. You OWN ONLY: src/autodata/harness/agent.py, src/autodata/cs/pipeline.py,
src/autodata/cs/stats.py, serving/check_glm53.py, README.md, docs/RUNBOOK.md, tests/test_agent.py, tests/test_pipeline.py,
tests/test_stats.py (+ new test files you create). Do NOT edit run_paper.py, evaluate_rubric.py, rubric.py, judge.py, solvers.py,
cot_baseline.py, config.py, tools.py, prompts/ (edited in parallel). Keep signatures; keyword-only defaults. No installs.
Run `.venv/bin/python -m pytest -q` at the end (all tests, including files you do not own) and paste the summary.

1. agent.py (review #7) — hard context preflight: before each LLM call compute an estimate of the prompt tokens
   (last completion.usage["prompt_tokens"] + chars/4 of everything appended since, or chars/4 for the first call) and, if
   estimate + max_tokens_for_call > `max_model_len` (new keyword-only Agent param, default None = disabled), progressively:
   (a) elide older tool results, (b) drop reasoning from older assistant messages, (c) shrink the protected window down to
   the last 2 messages, (d) as a last resort lower this call's max_tokens (never below 8192) via the client's per-call
   max_tokens. Never send a request whose estimate exceeds the limit. Log an "elide" event with what was done. Also apply the
   budget to any Agent (subagents included; run_paper passes budgets — keep the params optional).
2. pipeline.py (review #9, #10, #11):
   a. Cohort manifest: at launch write `<root>/cohort.json` {run_name, config_fingerprint, prompt_hashes (sha1 of each
      prompts/cs/*.md), corpus_path, corpus_sha1, paper_ids[], created_at, harness_version}; if a manifest exists and its
      config_fingerprint or prompt_hashes differ from the current run -> raise SystemExit with a clear message unless
      `--allow-config-mismatch` (new CLI flag, also a run_papers kwarg) is given; with the flag, append the new fingerprint to
      the manifest's `history`. Stamp CoT summaries too: run_papers already writes summaries — add `config_fingerprint` and
      `prompt_hashes_sha1` to every summary line it writes (merge into the dict) so both pipelines are consistent.
   b. Final-QV repair: a previous harness_summary with accepted=True and final_qv.qv_completed == False (or final_qv None
      while cfg.run.final_qv is True) must NOT be skipped on resume: the driver calls `run_one` with a keyword
      `repair_final_qv=True` (run_paper's PaperRun exposes `run_final_qv_only(summary)` — assume it exists; if it does not
      at test time, monkeypatch in tests) and merges the result into the summary. Keep everything else skipped.
   c. Endpoint health gate: before a paper starts (inside the semaphore), wait until every endpoint used by the run
      answers GET {base_url}/models (poll every 15 s, up to `health_wait_s` = 1800, then log and proceed); log the wait.
3. stats.py (review #10): read `<root>/cohort.json` when present: report n_requested, n_completed, n_incomplete,
   n_pending (requested minus present), acceptance_rate_completed and acceptance_rate_cohort; without a manifest fall back
   to the current behaviour but also count paper directories without a summary as pending. Print both rates.
4. serving/check_glm53.py (review #13): build the round-trip check's assistant message with BOTH `reasoning` and
   `reasoning_content` (as src/autodata/llm/client.py's assistant_message does) and assert via /tokenize + /detokenize that
   the rendered history contains the reasoning text.
5. README.md + docs/RUNBOOK.md (review #14): the quick start must wait for readiness between server launches
   (`bash scripts/wait_servers.sh`) and before running the pipeline; mention cohort.json and --allow-config-mismatch.
6. Tests for each item; keep the whole suite green.
FINAL REPORT: per-file changes, test summary, deviations.
