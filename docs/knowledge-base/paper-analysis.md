# Paper analysis: Autodata / Agentic Self-Instruct (arXiv 2606.25996v3, 2026-07-04)

> Source: Deep Interview 2026-09-27. Extracted from the v3 PDF (28 pp.), arXiv HTML v1/v3,
> and the Meta RAM README (`facebookresearch/RAM/projects/autodata/README.md`, May-2026
> snapshot, used ONLY for verbatim prompt text). There is NO official code release
> (checked 2026-09-27: RAM dir = README + 7 images). Two unofficial reconstructions exist
> (hossainpazooki/agentic-self-instruct; JAE-HUN-CHO/autodata-agentic-self-instruct) and were
> used only to cross-check ambiguity findings, never as a source of truth.

## 0. One-paragraph summary
Autodata = an LLM agent acting as a data scientist: create data -> analyze it (example- and
dataset-level) -> update the recipe -> repeat -> emit training/eval data. The agent itself can
be meta-optimized (outer loop) with the same criterion (Sec 4). The concrete instantiation used
in all experiments is **Agentic Self-Instruct**: a main orchestrator agent drives (i) a
challenger that writes an example from a grounding document, (ii) a weak solver, (iii) a strong
solver, (iv) a verifier/judge, and accepts an example only when the strong solver clearly beats
the weak solver (or, for legal, when a loop-judge says it is GRPO-suitable). Rejections are fed
back to the challenger as grouped failure lists, and the challenger writes an ENTIRELY NEW
question from a different angle. Loop until accepted or budget exhausted.

## 1. Framework (Sec 2)
- Data Creation: agent grounds on provided documents, may use tools/skills/inference compute.
- Data Analysis: per-example (correct? high quality? challenging enough?) or per-dataset
  (diverse? improves a model?) learnings fed back into creation.
- Loop until satisfied; "specific guardrails in the outer loop to prevent hacking".
- Meta-optimization: optimize the agent harness (autoresearch / meta-harness style) using the
  same inner-loop criterion.

## 2. Agentic Self-Instruct (Sec 2.1, Fig. 2)
Main orchestrator + 4 LLM subagents: Challenger, Weak solver, Strong solver, Verifier/judge.
- Main agent sends initial prompt + grounding context to the challenger -> example
  (context/input, reference answer, evaluation criteria depending on task).
- Main agent sends the input to weak and strong solvers, assigns a reward based on the verifier.
- Judge also checks the example itself (question, reference answer, rubric).
- Verifiable tasks: majority vote over strong correct AND majority vote over weak wrong.
- Non-verifiable tasks: quality gap measured by the judge on challenger-written rubrics; task
  "neither too easy nor too hard for the weak solver, while the strong solver helps guarantee
  correctness".
- If criterion unmet, main agent modifies the prompt to the challenger using the new learnings.
- Weak and strong may be the same LLM in different modes (more compute, aggregation,
  privileged information).

## 3. CS research-paper pipeline (Sec 3.1, App. C.1, RAM README) -- PRIMARY TARGET

### 3.1 Roles and models
| Role | Paper | This reproduction |
|---|---|---|
| Main orchestrator agent | Kimi-K2.6 (OpenCode harness) | GLM-5.3 |
| Challenger (subagent) | Kimi-K2.6 | GLM-5.3 |
| Quality verifier (subagent) | Kimi-K2.6 | GLM-5.3 |
| Rubric judge inside evaluate_rubric.py | Kimi-K2.6 | GLM-5.3 |
| Strong solver | Qwen3.5-397B-A17B | Qwen3.8-27B |
| Weak solver | Qwen3.5-4B | Qwen3.5-4B |
RAM README (May 2026) used Kimi-K2.5 everywhere; the paper (June/July 2026) uses Kimi-K2.6.

### 3.2 Harness facts (from the prompts themselves)
- Harness = **OpenCode**. Prompts live in `.opencode/prompts/`; the eval script at
  `.opencode/tools/evaluate_rubric.py` with `.opencode/tools/api_config.json`; run via
  `cd /workspace/project && uv run python3 .opencode/tools/evaluate_rubric.py --input
  ./eval_input.json --weak-only|--strong-only --output-dir ./eval_attempts --config
  .opencode/tools/api_config.json --timeout 600`.
- Per-paper working dir contains `./paper.txt` (full paper text). Challenger and QV read it
  with `cat ./paper.txt` themselves; the main agent must NOT interpret the paper.
- Main agent tools used: task (call challenger / quality verifier subagents), bash
  (run evaluate_rubric.py), write (write `eval_input.json`, `output/result.json`).
- "Steps" = the OpenCode agent step budget ("or you run out of steps"); the numeric budget is
  NOT given for CS (legal: hard cap 15 IMPROVE rounds).

### 3.3 Per-paper loop (exact order, from Main Agent prompt)
1. Call challenger -> QA + rubric. Challenger output: (1) question_type (short phrase),
   (2) 2-3 reasoning-skill tags, (3) context (situates solver, no answer leakage),
   (4) question (single, reasoning not recall), (5) reference answer (from paper's findings),
   (6) rubric: FLAT JSON array of 10-15 items, each exactly {criterion: str, weight: int,
   category: "positive"|"negative"}; 7-10 positive (+1..+10), 3-5 negative (-1..-10).
   Challenger first "scratchpad-analyses" key insights / likely weak-model errors.
2. Call quality verifier with context + question + rubric + question_type (QV reads paper).
   Checks: (1) context+question answer leakage -> NO_LEAKAGE / LEAKS_ANSWER;
   (2) question quality -> GOOD / TOO_EASY / RECALL ("explain why X works/how X fails" = too
   easy); (3) rubric quality STRICT: positives >= 4, negatives >= 3, total in [10, 20], each
   positive requires reasoning beyond context, each negative catches a specific reasoning
   error, criteria test reasoning not format; must report "Positive: X, Negative: Y, Total: Z";
   (4) question_type consistency -> CONSISTENT / INCONSISTENT. Then OVERALL: PASS/FAIL +
   FEEDBACK.
3. QV FAIL -> add to "FAILED QUALITY CHECK" list -> step 1 with feedback.
4. Write `eval_input.json`; run `evaluate_rubric.py --weak-only` (weak solver x3 on
   context+question; judge scores each answer per criterion). Report says WEAK_PASSED or not.
5. Weak fails (too easy) -> "TOO EASY" list (with weak avg) -> step 1.
6. Run `evaluate_rubric.py --strong-only` (strong x3).
7. Check strong criteria and gap; fail -> "FAILED ON STRONG" list (weak avg, strong avg,
   gap) -> step 1.
8. ALL pass -> ACCEPTED -> write final `output/result.json`, stop.
- Compute saving: strong solver is evaluated ONLY if weak passed (Sec 3.1 + prompt).
- Error handling: SOLVER_ERROR (all solver API calls failed) / timeout / empty result ->
  infrastructure issue -> RETRY the evaluation, do not refine. QV failure IS a quality issue.
- `output/result.json` rewritten after EVERY round (write tool), all rounds kept (accepted and
  rejected) so data survives step exhaustion. Schema (verbatim in prompts-verbatim.md):
  paper_title, question_type, reasoning_skills, rounds[{refinement_round, question, context,
  reference_answer, rubric, accepted, quality_verifier_kimi_passed,
  quality_verifier_kimi_feedback, weak_solver_avg, strong_solver_avg, gap, eval_report,
  eval_output_dir}], final_accepted_round, total_rounds.
- Final step of the loop: quality verifier runs again on the accepted example ("and also as a
  final step at the end of the loop"); the end-of-loop QV "removes questions with
  paper-specific reference leakage, short contexts, or malformed rubrics" -> 2.8k accepted
  -> 1.3k retained for RL.

### 3.4 Challenger refinement prompt (verbatim template, README)
```
The paper is available at ./paper.txt — read it first.

REFINEMENT: The following questions were previously generated for this
paper but did not meet our criteria:

Questions that were TOO EASY (weak model scored too high):
1. [<question_type>] "<question text>" — weak avg: <X>%

Questions that FAILED ON STRONG (weak was low but strong also
struggled or scored worse):
2. [<question_type>] "<question text>" — weak avg: <X>%,
   strong avg: <Y>%, gap: <Z>%

Questions that FAILED QUALITY CHECK (quality verifier rejected):
3. [<question_type>] "<question text>" — QV reason: <feedback>

Generate an ENTIRELY NEW question from a DIFFERENT angle that
requires deeper reasoning.
```
Only categories with entries are included. Round-1 prompt: "Generate a challenging research
question-answer pair with grading rubrics. The paper is available at ./paper.txt — read it
first."

### 3.5 Acceptance criteria -- TWO versions in the paper (AMBIGUITY A1, see ambiguities.md)
| Source | weak | strong | gap |
|---|---|---|---|
| Sec 3.1 prose ("we therefore define...") | weak_avg < 0.50 | strong_avg >= 0.65 | >= 20 pp |
| Fig. 7 / RAM README main-agent prompt (deployed) | weak_avg <= 65%, max_weak <= 75%, "no zeros" | strong_avg >= 60% AND < 95%, "No individual strong = 0%? (suspicious)" | >= 20% |
| Sec 4 meta-opt setup | weak <= 65%, best weak attempt <= 75% | strong >= 60% and <= 95% | >= 20 pp |
All averages are over the 3 solver attempts. The blog (RAM README) restates the deployed form.

### 3.6 Rubric scoring inside evaluate_rubric.py (formula NOT stated for CS; A4)
- Judge scores each criterion BINARY (0/1, "satisfied completely and unambiguously", default 0
  when in doubt); negative criteria scored 1 when the response exhibited the bad behaviour and
  polarity inverted at aggregation (stated in the legal loop-judge prompt; same evaluator).
- Judge sees the response + rubric, NOT the reference answer (Sec 3.1: "score any response
  without access to the reference answer").
- Most consistent formula (PRBench "clipped" score, which the paper uses for legal eval):
  score = clip( sum_i w_i * I_i / sum_{w_i>0} w_i , 0, 1 ), w_i signed integer weights.
- Report per attempt: score; aggregate: weak_avg, max_weak, strong_avg, min/individual scores,
  gap = strong_avg - weak_avg; textual report "WEAK_PASSED" etc.; per-attempt files under
  `./eval_attempts/run_*/`.

### 3.7 Solver settings
- Each solver invoked 3 times "to reduce variance". Solvers receive context + question only
  (never the paper, never the rubric). Temperature 1.0 (Sec 4: "solvers run at temperature
  1.0"). Max tokens / thinking mode not stated for CS (Principia used a 65,536-token reasoning
  budget). Qwen3.5-4B is described as "a very capable reasoning model" -> thinking mode on.

### 3.8 Corpus and scale
- S2ORC CS papers, 2022+ (Lo et al. 2020). >10k papers processed -> 2.8k accepted (agentic)
  -> 1.3k after end-of-loop QV filter. CoT Self-Instruct baseline: same QV filter, 1.3k sampled.
- Loop stats: mean 6.59 rounds per accepted item (README: median 3-5), long tail > 10 rounds
  (Fig. 4 shows an item accepted at round 17; RAM trajectory accepted at round 6).
  Of 880 pre-acceptance failed rounds: 80% too easy (weak too high), 13% strong failed.
- Table 1 (graded by Kimi-K2.6, 4B-weak / 397B-strong): CoT vs Agentic: weak avg 0.677 vs
  0.458; strong avg 0.696 vs 0.772; gap 0.019 vs 0.314; rounds 1.00 vs 6.59; question length
  723 vs 619 chars; rubric items 13.2 vs 13.1.

### 3.9 CoT Self-Instruct baseline (needed for Table 1 / Table 2)
"standard prompted generation" = single-shot challenger output (no loop), then the same
quality verifier filter, then 3+3 solver attempts for statistics. Exact baseline prompt not
given (A10) -> use the same challenger prompt, round-1 message only, no refinement.

### 3.10 Downstream RL (OUT OF SCOPE for now, recorded for later)
Qwen3.5-4B, GRPO, batch size 16 (README: 32, ~1 epoch), lr 1e-6, Kimi-K2.6 rubric reward,
100 held-out test examples per data source, metrics mean@3 / best@3, ~240 steps, eval every
20 steps. Table 2 numbers: base 0.630/0.758 (CoT test), 0.366/0.484 (Agentic test); CoT-trained
0.727/0.853, 0.500/0.631; Agentic-trained 0.774/0.894, 0.632/0.768.

## 4. Legal pipeline (Sec 3.2, App. C.2) -- SECONDARY
- Source: Pile of Law court opinions etc.; eval PRBench-Legal (500) / Legal-Hard (250).
- 4 subagents: main agent, extractor, question-and-rubric writer, loop-judge (all Kimi-K2.6);
  strong Qwen3.5-397B-A17B, weak Qwen3.5-4B. 5 weak + 3 strong rollouts per candidate.
- Round 1: extractor -> ./extract.json {suitable_for_synthetic_question, suitability_note,
  document_type, topic_keywords, issues, facts, holdings}; early-exit if unsuitable
  (skipped: true, skip_reason: doc_not_suitable). Writer -> {target_capabilities, question,
  rubric} (15-25 criteria, six keys each: number, criterion, category, capability,
  weight_class, weight; positive +2..+10 in three bands, negative -1..-10, >= 1 negative).
  evaluate_rubric.py once (5 weak + 3 strong in parallel, per-criterion scored by Kimi) ->
  diagnostic packet (aggregate weak/strong/gap, per-rollout scores, per-capability table,
  post-filter flags) -> loop_judge -> two-layer decision: Layer 1 post-filter overrides
  (body_text > 1000 chars OR case-recap/exam-prompt regex: "court of appeals", "supreme court
  of", "in the case of", "in re ", "you are an", "act as", "as a law clerk", "draft the
  arguments", "draft the brief", "the trial court held", "on appeal", or a pre-1940 year
  18xx/1900-1939) -> IMPROVE regardless; Layer 2 defer to loop-judge (accept / improve).
  Hard cap 15 IMPROVE rounds. IMPROVE rounds reuse extract.json, call writer with MODE:
  IMPROVE + packet + verdict_reason + suggestion_for_writer verbatim.
- Loop-judge output JSON: weak_pattern, strong_pattern, gap_interpretation, rubric_concerns,
  grpo_suitability (high/medium/low), verdict (accept/improve), verdict_reason,
  suggestion_for_writer (required on improve). No fixed thresholds; defaults to improve when
  uncertain. PRBench anchors: 4B = 24.5%, 397B = 35.8%.
- Stats (Table 3): CoT vs Agentic weak 0.159 vs 0.283; strong 0.717 vs 0.698; gap 0.558 vs
  0.415; rounds 1.00 vs 4.98 (median 4, max 19, ~2% single round); q length 1569 vs 900;
  rubric items 18.6 vs 17.3; weak rollout std 7.93 vs 12.63; grpo_suitability high/med/low
  CoT 4.8/41/45%, Agentic 52/43/2%. 7.8k docs -> 5.7k CoT usable, 2.8k agentic accepted.
- RL: 2.8k prompts each, GRPO n=8, Kimi rubric reward; PRBench clipped + normalized scores,
  GPT-5 and Kimi graders (Tables 4, 12).

## 5. Scientific reasoning pipeline (Sec 3.3, App. C.3) -- SECONDARY
- Grounding: Principia collection (Aggarwal et al. 2026; MSC2020/PHYS curricula); eval Principia
  bench (ARB, Physics, RealMath, SuperGPQA; 2113 items). Weak Qwen3.5-4B, strong 397B, main
  agent + challenger Kimi K2.6. Verifiable: judge compares answer to reference -> binary.
- Main agent spawns 3-5 challengers in parallel with varied angles, screens candidates itself,
  evaluates promising ones with `evaluate.py --runs-per-solver 4 --solvers weak,strong`;
  accept iff weak correct <= 1/4 AND strong correct >= 3/4 (exact counts). Atomic questions,
  one-sentence answers, avoid bare integers / precision-sensitive reals.
  `sample_examples.py` samples grounding examples from principia_data/. Output
  ./question_answer.json {question, answer}. Dependencies via pixi. "2+2" API sanity check on
  INCONCLUSIVE reports.
- RL: 9k train + 1k held-out per source; GRPO group 8, batch 64; 65,536-token reasoning budget;
  Kimi K2.6 binary judge vs reference. Tables 5-6, 8-11.

## 6. Meta-optimization (Sec 4) -- OUT OF SCOPE for now
Population of prompt diffs; Boltzmann parent selection T=0.1; evaluate parent on minibatch of
training papers; analyzer LLM writes root-cause analysis; implementer (code-editing agent)
produces diff; re-evaluate parent+mutant on held-out validation papers; accept iff strictly
better; history log; concurrent iterations; re-evaluations averaged. 50 train / 25 val papers;
success = weak <= 65%, best weak <= 75%, strong >= 60% and <= 95%, gap >= 20 pp. 62.1% -> 79.6%
(iter 124 of 233, 6h/session). Discovered fixes: paper-specific insight enforcement, context-leak
self-test, positive-only rubric with weights capped at 7, strict JSON integer weights.

## 7. Differences between RAM README (May 2026) and paper v3
Kimi-K2.5 vs K2.6; 2,117 vs 2.8k accepted; median 3-5 vs mean 6.59 rounds; CoT/Agentic weak
71.4/43.7 vs 67.7/45.8, strong 73.3/77.8 vs 69.6/77.2; RL batch 32 vs 16; meta-opt 12.8->42.4
vs 62.1->79.6. README prompt states the deployed acceptance thresholds (Fig. 7 form).
