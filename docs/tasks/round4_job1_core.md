# Round-4 fix job 1 — core runtime, evaluator, parsing, statistics

Owner files (ONLY these may be modified by this job): `src/autodata/cs/run_paper.py`, `src/autodata/cs/evaluate_rubric.py`,
`src/autodata/cs/parsing.py`, `src/autodata/cs/stats.py`, `src/autodata/cs/rubric.py`, `src/autodata/harness/agent.py`,
`src/autodata/llm/client.py`, and tests `tests/test_evaluate_rubric.py`, `tests/test_prompts_and_parsing.py`,
`tests/test_stats.py`, `tests/test_e2e_fake.py`, `tests/test_agent.py`, `tests/test_llm_client.py` (plus new test files named `tests/test_round4_*.py`). Another job edits `pipeline.py`,
`cot_baseline.py`, the data package and the docs concurrently: do not touch those. Never write into `runs/`, `logs/`,
`data/`; never start model servers or the pipeline; unit tests only (`.venv/bin/python -m pytest -q`).
Keep the public interfaces used by `pipeline.py`/`cot_baseline.py` unchanged: `PaperRun.run_final_qv_only(summary)`
returns the fields to merge; `run_evaluator`, `evaluator_deadline_s`, `final_filter`, `prepare_workspace`,
`compute_question_hash` keep their signatures.

Context: paper-exact reproduction of Agentic Self-Instruct (Autodata, arXiv 2606.25996). The harness guardrails are
additions on top of the paper's loop and must never reject a candidate the paper's own harness would accept, except
where documented in docs/knowledge-base/review-decisions.md. A production pilot is running with the CURRENT code:
resume compatibility with existing `runs/*/s2_*/eval_attempts/run_*/report.json` files (legacy question hashes) is
required.

## A. `--force-strong` regression test (the code fix is already applied)
`evaluate_rubric.py` stage loop: `if role == "strong" and not report["weak_passed"] and not args.force_strong: break`.
Add tests: strong-only with a FAILING weak result (weak_avg >= 0.50) and `--force-strong` runs the strong stage;
without `--force-strong` it does not; the agentic (no-flag / weak-only+strong-only) gating is unchanged.

## B. Canonical question identity and stage-aware evaluation cache
1. `compute_question_hash(data)`: hash the canonical form of {context, question, rubric}: `json.dumps(obj, sort_keys=True,
   separators=(",", ":"), ensure_ascii=False)` after normalising strings (NFC, `\r\n`->`\n`, strip trailing whitespace
   per line) and rubric items reduced to {criterion, weight(int), category}. Keep `legacy_question_hash(data)` = the old
   `json.dumps(question)` sha1. `_latest_weak(...)` (strong-only reuse of a weak run) must accept a report whose
   `question_hash` equals either the canonical or the legacy hash of the current input. The harness `_verify_report`
   must accept either as well for reports produced by the current evaluator.
2. `PaperRun.run_evaluate_rubric` cache: key on the canonical hash, stage-aware. Requested `weak-only` or `both`: if a
   VERIFIED prior evaluation in this round with the same canonical hash contains a weak stage -> return its stdout with
   the existing "[cached ...]" note (no new sampling). Requested `strong-only`: same, when a verified strong stage exists.
   Requested `both`: if a verified strong stage exists -> return that evaluation's stdout (cached); elif a verified weak
   stage exists and FAILED -> return the weak evaluation's stdout (cached); elif a verified weak stage exists and PASSED
   and no strong stage exists yet -> execute as `--strong-only` (the evaluator reuses the weak run) and say so in the
   returned text. A candidate that completed a failing evaluation never gets a second sampling in the same round. Keys reordered or whitespace
   changed inside eval_input.json must not create a new identity. Tests for each branch.

## C. QV binding (`PaperRun._qv_bound`)
Current check compares only the first 80 alphanumerics of each criterion and looks for the UNSIGNED weight within 40
characters of that prefix, so it fails for long criteria when the weight follows the full text (JSON order) and it
ignores the weight sign. In production 85% of accepted rubric criteria exceed 120 alphanumerics. New rule, format
agnostic: normalise with `_signed_alnum` = lowercase, keep `[a-z0-9-]`, drop everything else (so "+8" -> "8",
"-5" -> "-5"). For every rubric item: the FULL normalised criterion must occur in the normalised prompt; the signed
weight token (`-5` for negatives, `8` for positives, matched with digit boundaries and, for positives, not preceded by
`-`) must occur within 60 normalised characters immediately BEFORE the criterion start or AFTER the criterion end.
Question: full normalised text must occur; context: first 200 normalised characters must occur. Category is not
checked (the parser already enforces category == sign). Validate the new rule (read-only) against the real
quality-verifier prompts of all accepted candidates in `runs/smoke24/*/trajectory/main_agent.jsonl` (task tool calls
whose arguments mention the quality verifier) and their accepted candidates in `harness_summary.json`: every accepted
candidate must bind; report the count in your final message. Add unit tests with JSON-after, JSON-before and bullet
formats, long criteria, sign flips (+8 -> -8 must NOT bind), and tail edits of a criterion (must NOT bind).

## C2. Challenger-origin check (`run_evaluate_rubric`)
The origin check compares `_alnum` forms, which discard operators, case and non-ASCII symbols (`x < 0` vs `x > 0`,
`alpha` vs `beta` in Greek letters both pass). Replace with `_text_norm`: NFC normalisation, `\r\n` -> `\n`, collapse
runs of whitespace to one space, strip; case preserved. The evaluated question must equal the parsed challenger
question under `_text_norm`; when the challenger JSON could not be parsed, the `_text_norm` question must occur as a
substring of the `_text_norm` challenger output (the main agent may repair malformed JSON but never edit the question).
Tests: operator flip and Greek-letter swap are refused; whitespace/escaping differences are accepted.

## C3. Quality-verifier repeat rule
The QV-repeat refusal (same question in the same round) must apply only when the previous QV call was BOUND to the
candidate under the new rule; an unbound QV call may be repeated once with the complete candidate (the refusal message
must say what was missing: question / context head / criterion N / weight of criterion N). This prevents a deadlock
where a QV that omitted the rubric can neither be repeated nor lead to acceptance.

## D. Quality-verifier parser (`parse_qv_output`)
An echoed template line such as `OVERALL: PASS | FAIL`, `OVERALL: PASS or FAIL`, `OVERALL: PASS/FAIL` or
`CHECK_1_VERDICT: NO_LEAKAGE | LEAKAGE` currently parses as the first token. Treat a verdict followed (on the same line)
by `|`, `/`, ` or ` and another verdict-like token as UNRESOLVED: `overall_stated` None / the check missing. Keep the
formatted-line tolerance (bold, bullets) and the fallback for the last standalone PASS/FAIL after OVERALL, but the
fallback must not resolve `PASS | FAIL` either. Tests.

## E. Rubric weight range: non-blocking
`rubric_weights_in_bounds` (Fig. 8 says +1..+10 / -1..-10) currently makes the evaluation report unverifiable
(`problems`) and fails `final_filter`. The paper's harness does not reject rubrics by weight range and the quality
verifier only checks counts, so out-of-range weights emitted by the challenger must not block acceptance. Change:
`_verify_report` records the message in a new `warnings` list (entry["warnings"], also in the round summary as
`eval_warnings`), NOT in `problems`; `final_filter` keeps reporting `weights_in_spec: bool` but `rubric_ok` no longer
depends on it. Log a `guardrail`-style event `rubric_weights_out_of_spec` (informational). Update tests and the
docstrings.

## G. Final-QV repair (`run_final_qv`, `run_final_qv_only`)
1. `run_final_qv` question_type fallback must not index `self.rounds` when it is empty (repair path): use the
   candidate's `question_type`, else the summary round's, else "".
2. On a SUCCESSFUL repair (final QV completed), previously recorded `final_qv:` errors are resolved: move them to
   `resolved_errors` and return `errors` without them, so the next resume treats the paper as done. On a failed repair
   keep the errors as now. Tests with the real `run_final_qv_only` (fake server), including a candidate without
   `question_type` and a resume-after-repair check on the returned fields.

## G2. Repair transcript naming
`run_final_qv_only` sets `subagent_counter["quality_verifier"]` from the COUNT of `*qv*.jsonl` files, so a repair can
reuse an existing transcript name (`final_qv_03.jsonl` appended twice). Use the maximum existing numeric suffix + 1
across `quality_verifier_NN.jsonl` and `final_qv_NN.jsonl`.

## G3. Evaluator parent-death race (`evaluate_rubric.py`)
`_die_with_parent()` arms PR_SET_PDEATHSIG only inside `main()` after imports; if the harness dies before that, the
orphan keeps running and issuing model requests after another runner takes the paper lock. Arm the signal at the very
top of module import (before heavy imports), check prctl's return value, take the expected parent pid from an
environment variable set by the harness (`AUTODATA_PARENT_PID`; fall back to `os.getppid()` at spawn time), and exit
immediately (code 5, message on stderr) if `os.getppid()` differs from it after arming. The harness (`run_evaluator`
in run_paper.py) sets that variable. Tests for the pid-mismatch exit using a subprocess.

## G4. Context preflight must use the last successful request's prompt tokens (`agent.py`, `client.py`)
`LLMClient` aggregates usage across retries into `result.usage`; the agent's preflight reads
`completion.usage["prompt_tokens"]` from that aggregate, so after an empty-response retry the estimate doubles and a
prompt that fits is rejected as too long. Expose the last successful request's usage separately
(`completion.last_request_usage`) and use it for the context calibration; keep the aggregate for accounting. Tests
with the fake server (two billed attempts of 7,000 prompt tokens, 10,000-token context, next turn must proceed).

## H. Statistics from acceptance-time reports (`stats.py`)
Legacy summaries (smoke run) can carry a weak score from a weak-only evaluation run AFTER acceptance (observed:
runs/smoke24/s2_288742856, summary weak 0.1696 vs accepting report 0.0994). For every accepted item, locate the
accepting report: `<workdir>/eval_attempts/run_*/report.json` with `all_passed` true and `question_hash` equal to the
canonical OR legacy hash of the accepted round's candidate, earliest by `started_at`; use its `weak_avg`, `strong_avg`,
`gap`, `weak_scores`/`strong_scores` for the Table-1 columns and record per item `scores_source: "report"|"summary"`.
Fall back to the summary when no such report exists (archived/missing). Add cohort counters `n_report_backed`,
`n_summary_backed`. Tests with a synthetic workdir.

## Finish
Run the FULL suite `.venv/bin/python -m pytest -q` (the other job's files may change under you; only fix failures in
your owner files and re-run at the end). Final message: what changed per item, the binding validation count, test
totals, and anything you deliberately did not do.
