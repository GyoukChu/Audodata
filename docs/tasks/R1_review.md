Read-only review task (do NOT edit files). Repository: this directory. Context: a faithful reproduction of the CS pipeline of
"Autodata: An agentic data scientist to create high quality synthetic data" (arXiv 2606.25996, Agentic Self-Instruct).
Read first: docs/IMPLEMENTATION_SPEC.md, prompts/cs/README.md + prompts/cs/*.md, and the paper notes in
docs/knowledge-base/paper-analysis.md (section 3) and ambiguities.md.

Review these modules for (1) correctness bugs, (2) deviations from the spec/paper semantics, (3) failure modes under real
vLLM serving (GLM-5.3 with reasoning_content + tool calls via the OpenAI-compatible API; Qwen thinking models; long 150k-char
paper texts; subprocess CLI runs), (4) concurrency/resume issues when 6-8 papers run at once:
- src/autodata/llm/client.py, src/autodata/harness/tools.py, src/autodata/harness/agent.py
- src/autodata/cs/rubric.py, judge.py, solvers.py, evaluate_rubric.py
- src/autodata/cs/run_paper.py, pipeline.py, cot_baseline.py, stats.py, corpus_io.py, prompts.py, parsing.py
- tests/test_e2e_fake.py (does it prove what it claims?)
Specific questions to answer explicitly:
a. Does the main-agent loop preserve GLM's reasoning_content across tool turns correctly, and would a vLLM 4xx from a malformed
   assistant/tool message be retried forever or surfaced?
b. In run_paper.PaperRun.run_evaluate_rubric, is harness-verified acceptance derived correctly from report.json for both
   `--strong-only` and `both` modes, including the case where the agent re-runs an eval after SOLVER_ERROR?
c. Can the agent "hack" acceptance (e.g. by writing output/result.json claiming acceptance, by editing eval_input.json between
   weak and strong runs, or by running --strong-only without --weak-only)? What does the harness do in each case?
d. Is the rubric scoring exactly clip((earned - penalty)/max_positive, 0, 1) with binary judgments, and do report percentages
   and threshold comparisons use consistent units?
e. Any place where the reference answer could leak to the solvers or the judge?
f. Path/sandbox escapes in the bash/read/write tools; robustness of shlex parsing to the verbatim commands in prompts/cs/main_agent.md.
g. Anything that would break `python -m pytest -q` in CI or make results non-reproducible (timestamps in run_dir names, etc.).
Output: a prioritized list (P0 blocker / P1 should fix / P2 nice-to-have) with file:line references and a concrete fix for each.
Be concrete and skeptical; do not restate the code.
