# Ambiguities and required rulings

> Source: Deep Interview 2026-09-27. Each item: what the paper says, options, and the ruling
> (filled in as the user answers). Items marked DEFAULT are decided by me unless overridden.

## A1. CS acceptance thresholds (prose vs deployed prompt)
- Sec 3.1 prose: strong_avg >= 0.65, weak_avg < 0.50, gap >= 20 pp.
- Fig. 7 / README / Sec 4: weak_avg <= 65%, max_weak <= 75%, no zeros; strong_avg >= 60% and
  < 95%; no individual strong 0%; gap >= 20%.
- Plan: implement BOTH as named presets (`prose_s31`, `deployed_c1`), both prompt text and
  evaluate_rubric.py thresholds driven by the same config.
- RULED (2026-09-27): default = `prose_s31` (strong_avg >= 0.65, weak_avg < 0.50, gap >= 0.20).

## A2. "no zeros" semantics (deployed form only)
- Reject if ANY weak attempt scored 0 (per-attempt reading, matches "No individual strong =
  0%") vs reject only if ALL are 0. DEFAULT: per-attempt (min > 0), configurable.

## A3. Max rounds / step budget per paper (CS)
- Not stated. Evidence: mean 6.59, tail > 10, Fig. 4 accepted at round 17, legal hard cap 15.
- RULED (2026-09-27): max 15 rounds per paper, plus a main-agent tool-call step cap (~80).

## A4. Rubric score formula
- Not stated for CS. DEFAULT: PRBench clipped score = clip(sum w_i*I_i / sum_{w>0} w_i, 0, 1),
  binary per-criterion judgments, judge does not see the reference answer.

## A5. Judge prompt wording
- Not given anywhere. DEFAULT: write a judge prompt implementing the stated semantics (binary
  per criterion, strict, default 0 when in doubt, negative criteria "asserted in the
  affirmative"), output JSON {criterion_index: 0/1}.

## A6. Solver inference settings
- temperature 1.0 given. Thinking mode, max_tokens, top_p not given. DEFAULT: thinking ON for
  both Qwen solvers, Qwen-recommended thinking sampling (temp 1.0, top_p 0.95, top_k 20),
  max_tokens = 32,768 (RULED 2026-09-27), plain user message = context + question, no system
  prompt. Truncated answers are graded as-is: a response cut off by max_tokens with NO final answer (thinking
  models, finish_reason=length) scores 0 for every criterion without a judge call and is not retried (matches the
  paper's treatment of truncation as failure, App. A); only an empty response that stopped normally is retried.
  Implemented 2026-09-27 06:30 after observing 27B/4B responses exhausting 32k tokens in thinking (previously such
  attempts were retried up to 3x and then raised SOLVER_ERROR).

## A7. Orchestrator / challenger / QV / judge sampling
- Not given. RULED (round 3): GLM-5.3 model default (reasoning_effort=max) for all roles; temperature 1.0 / top_p 0.95.
- WITHDRAWN deviation: the orchestrator briefly set the judge to reasoning_effort=low (judge_effort.json: low
  ~0.9k tokens/29 s with verdicts identical to max; max 9-17k tokens/317 s and sometimes verdict-less at 32k tokens).
  USER RULING 2026-09-27 03:5x: keep max effort for ALL GLM roles; the 32k limit applied to the SOLVERS only; GLM roles
  get max_tokens 81,920 (user offered 81,920 or 128,000; 81,920 chosen so a 285k-token main-agent prompt + output fits
  the 400k max-model-len). Judge timeout raised to 3600 s. Error-handling only: after a verdict-less truncation at the
  full budget, the judge retry steps the effort down (max->high->low) instead of repeating a futile 82k-token call.

- USER RULING 2026-09-27 11:2x: GLM-5.3 must run with interleaved thinking ON and preserved thinking OFF. Verified on the
  live server with /tokenize+/detokenize: within one turn (our agent loops: one user message, then assistant/tool steps)
  each step's reasoning is rendered back as <think>...</think> before its tool call regardless of clear_thinking
  (interleaved = on, because the client sends `reasoning`/`reasoning_content` back); `clear_thinking: true` removes the
  thinking of assistant messages from earlier user turns (preserved = off). Configs now use
  chat_template_kwargs {reasoning_effort: max, clear_thinking: true} for all GLM roles.

## A8. Paper text preparation
- `./paper.txt` = full paper text. DEFAULT: pdftotext of the PDF (or S2ORC parsed text),
  truncated to a max char budget to respect context windows; skip papers that are too short.

## A9. End-of-loop quality verifier
- "removes questions with paper-specific reference leakage, short contexts, or malformed
  rubrics". DEFAULT: re-run the same QV prompt on the accepted item + programmatic checks
  (context length floor, rubric JSON schema/count validity).

## A10. CoT Self-Instruct baseline prompt
- Not given. DEFAULT: identical challenger prompt, round-1 message only, single shot; same QV
  filter; same 3+3 solver attempts for statistics.

## A11. Model substitutions and access -- RULED (2026-09-27): full GLM-5.3 served locally as NVFP4
(nvidia/GLM-5.3-NVFP4 / vLLM recipe nvfp4 variant) with offload enabled; strong = Qwen3.8-27B; weak = Qwen3.5-4B.
## A12. Harness style -- RULED: custom Python agent harness reproducing OpenCode semantics; verbatim prompts.
## A13. Corpus source -- RULED: S2ORC via Semantic Scholar API (key in .env), 1 req/s global limit.
## A14. Scope -- RULED: CS pipeline + CoT baseline; smoke 10-20 papers -> pilot ~200 papers.
