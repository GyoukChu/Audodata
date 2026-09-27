You are implementing part of a research-code reproduction. Work ONLY inside this repository (git root = this directory).

READ FIRST (in this order):
1. docs/IMPLEMENTATION_SPEC.md — fixed interfaces. You own section 2.4 (rubric.py, judge.py, solvers.py, evaluate_rubric.py).
2. src/autodata/config.py — AcceptancePreset / ModelEndpoint / EvalConfig (do not modify).
3. prompts/cs/judge.md and prompts/cs/solver_user.md — the prompts you must use verbatim.
4. prompts/cs/main_agent.md — shows how the main agent calls evaluate_rubric.py and what report lines it expects
   (WEAK_PASSED, SOLVER_ERROR, "failed on strong", etc.).
5. docs/knowledge-base/paper-analysis.md sections 3.3-3.7 (the paper's
   description of the evaluation: each solver invoked 3 times, judge scores per criterion, weak evaluated first, strong only if weak
   passed, scoring formula assumption A4 in ambiguities.md).

DELIVERABLES (create these files; do not touch other source files):
- src/autodata/cs/rubric.py, src/autodata/cs/judge.py, src/autodata/cs/solvers.py, src/autodata/cs/evaluate_rubric.py
  (module with `main(argv=None)`; console script entry `autodata-evaluate-rubric` already declared in pyproject.toml).
- tests/test_rubric.py, tests/test_judge.py, tests/test_evaluate_rubric.py — pytest, no GPU/network. Use tests/fake_openai_server.py:
  prefer httpx.MockTransport (make_transport) injected into the client. The spec's LLMClient (src/autodata/llm/client.py) is being
  written IN PARALLEL by someone else and may not exist yet: do NOT depend on it. Implement your own thin sync client in solvers.py /
  judge.py using `openai.OpenAI(base_url, api_key, timeout, max_retries=0, http_client=httpx.Client(transport=...))` with a small
  retry helper (backoff on 429/5xx/connection errors), and accept an optional `transport` argument for tests. Sampling params come
  from ModelEndpoint (temperature/top_p/max_tokens/presence_penalty natively; top_k/min_p/repetition_penalty/chat_template_kwargs/
  extra_body merged into extra_body).
- The CLI must be runnable as a subprocess from an arbitrary working directory:  python -m autodata.cs.evaluate_rubric --input ...
  Tests should exercise the CLI both in-process (main(argv)) and as a subprocess using the real-socket FakeServer if sockets work in
  your sandbox (skip that test with a clear reason if they do not).

BEHAVIOUR DETAILS (in addition to spec 2.4):
- Solver messages: exactly [{"role":"user","content": template}] with prompts/cs/solver_user.md filled ({context}, {question}); no system prompt.
- Strip reasoning: the answer text is message.content; if the endpoint returned reasoning in reasoning_content/reasoning, keep it
  separately in the attempt file; also strip any inline <think>...</think> blocks from content. finish_reason "length" is recorded
  but the (truncated) answer is still graded (paper: truncated answers count as-is).
- Judge messages: [{"role":"system", judge.md}, {"role":"user", "## Context\n...\n\n## Question\n...\n\n## Rubric\n1. [+8 positive] ...\n2. [-5 negative] ...\n\n## Response\n..."}].
  Parse the first JSON object in the output (tolerate ```json fences and preamble). Validate exactly len(rubric) entries with indexes 1..n
  (order-insensitive but all present); otherwise retry (eval.judge_retries) with the same messages; then raise JudgeError.
- Score: clip((sum of weights of satisfied positive criteria - sum of |weights| of triggered negative criteria) / sum of positive weights, 0, 1).
- Percentages in reports are formatted with one decimal ("48.1%"). Comparisons use fractions. The acceptance preset can be
  prose_s31 (weak_avg < 0.50 strict, strong_avg >= 0.65, gap >= 0.20, no other rules) or deployed_c1 (weak_avg <= 0.65, max_weak <= 0.75,
  no zero weak attempt, strong_avg >= 0.60 and < 0.95, no zero strong attempt, gap >= 0.20) — implement generically from the
  AcceptancePreset fields (weak_avg_max_inclusive, weak_attempt_max, weak_no_zero, strong_avg_max, strong_no_zero).
- Attempts run concurrently with ThreadPoolExecutor(max_workers=n_attempts); each attempt = solver call (retry up to eval.solver_retries
  on error/empty) then judge call. --timeout applies per solver request (pass as the request timeout).
- Both modes in one invocation (no --weak-only/--strong-only): run weak first; run strong only if weak passed (paper: compute saving),
  print both blocks and the ACCEPTANCE line.
- Exit codes: 0 normal (even when criteria fail), 2 SOLVER_ERROR, 3 JUDGE_ERROR, 4 RUBRIC_ERROR, 5 INPUT_ERROR. Always print
  the error line as the first stdout line in those cases.
- report.json must be complete enough to recompute everything (per-attempt scores, satisfied lists, breakdowns, token usage, latencies).

CONSTRAINTS: Python 3.12, type hints; no new dependencies; do NOT run pip/uv installs. Run tests with
  .venv/bin/python -m pytest tests/test_rubric.py tests/test_judge.py tests/test_evaluate_rubric.py -q
FINAL REPORT: files created, test results (pasted), exact CLI usage, any deviation from the spec.
