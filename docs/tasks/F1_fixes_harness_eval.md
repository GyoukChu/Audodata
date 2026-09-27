Follow-up FIX task (write-capable). Repository: this directory (Autodata / Agentic Self-Instruct reproduction).
A read-only review found the issues below. You own ONLY these files (others are being edited in parallel by someone else —
do not touch src/autodata/cs/run_paper.py, pipeline.py, cot_baseline.py, corpus_io.py, parsing.py, prompts.py, stats.py,
config.py, prompts/, tests/test_e2e_fake.py, tests/test_prompts_and_parsing.py):
  src/autodata/harness/tools.py, src/autodata/harness/agent.py, src/autodata/llm/client.py,
  src/autodata/cs/evaluate_rubric.py, src/autodata/cs/judge.py, src/autodata/cs/solvers.py, src/autodata/cs/rubric.py,
  tests/test_tools.py, tests/test_agent.py, tests/test_llm_client.py, tests/test_evaluate_rubric.py, tests/test_judge.py,
  tests/test_rubric.py, tests/fake_openai_server.py (extend only).
Keep every public signature from docs/IMPLEMENTATION_SPEC.md; new parameters must be keyword-only with defaults.
Do not install packages. Run: .venv/bin/python -m pytest -q tests/test_tools.py tests/test_agent.py tests/test_llm_client.py tests/test_evaluate_rubric.py tests/test_judge.py tests/test_rubric.py

FIXES:
1. tools.py — the verbatim evaluator commands in prompts/cs/main_agent.md are written with backslash-newline continuations
   ("... \\\n    --input ./eval_input.json \\\n ..."). Normalize shell line continuations (backslash followed by \r?\n and
   optional indentation -> single space) BEFORE rejecting newlines/control operators, so the verbatim multi-line command routes to
   evaluate_rubric_runner exactly like the single-line form. Also: when forwarding argv to evaluate_rubric_runner, keep the
   original tokens (the runner resolves paths itself) but reject any argv token that resolves outside the workspace.
2. tools.py — make_write_tool(ws, *, allow: Callable[[str], bool] | None = None): when `allow` is given and returns False for the
   requested path (the relative, normalized path string), return "Error: writing <path> is not permitted in this workspace"
   without writing. make_read_tool unchanged. Document in the tool description that only data files may be written.
3. agent.py — context-window budget. Add keyword-only `context_budget_chars: int | None = None` and `keep_recent_tool_results:
   int = 6`. Before each LLM call, if the total character length of all message contents exceeds the budget, elide the CONTENT
   of the oldest tool-result messages (never the system prompt, never the first user task prompt, never the most recent
   `keep_recent_tool_results` tool results, never assistant messages/tool_calls) by replacing their content with
   "[tool result elided by the harness to fit the context window: <N> chars; the full text is in the transcript]", until under
   budget or nothing left to elide. Elisions must keep the assistant/tool message sequence valid for the vLLM chat template.
   Log an "elide" event to the transcript. If the LLM returns an HTTP 400 mentioning context/length, stop with stop_reason
   "error" and error text (do not retry forever).
4. client.py — retry policy: treat HTTP 400/422 as non-retryable (surface immediately); keep 408/409/429/5xx retryable.
   Add `seed: int | None` keyword-only param to chat()/chat_sync() passed through to the request when given (vLLM supports it).
5. evaluate_rubric.py — provenance hardening: (a) `_latest_weak` must require len(weak_attempts) == the CURRENT config's
   eval.n_attempts and no attempt errors, and must also verify the stored report's `acceptance` and `eval` sections equal the
   current config (otherwise skip that report); (b) expose `compute_question_hash(data: dict) -> str` (the sha1 used for
   question_hash) and `assess_attempts(rubric, weak_attempts, strong_attempts, preset) -> dict` (pure function returning
   weak_avg/strong_avg/gap/weak_passed/strong_passed/gap_passed/all_passed/failure_reasons from the satisfied lists) so the
   harness can recompute acceptance independently; (c) include in report.json: `input_path` (absolute), `input_sha1` (sha1 of
   the raw input file bytes), `config_path`, `config_sha1`, `n_attempts_required`; (d) when run with `python -I` the module
   must still import (no reliance on cwd being on sys.path).
6. solvers.py/judge.py — pass `seed` through when the endpoint dict contains "seed" (optional) and record request seeds in the
   attempt files. Keep everything else.
7. Tests for each fix (multi-line verbatim command routing; write allow-list; elision keeps sequence valid and respects
   keep_recent; 400 non-retryable; _latest_weak rejects wrong n_attempts / wrong preset; compute_question_hash stable;
   assess_attempts matches the CLI's own verdicts on a sample).
FINAL REPORT: what changed per file, test results, any signature additions.
