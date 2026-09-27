# Reproduction report — Agentic Self-Instruct (Autodata, arXiv 2606.25996v3), CS pipeline

_Status: draft written during the first real runs (2026-09-27). Numbers in §6 are refreshed by `autodata-stats`._

## 1. What was reproduced
The Sec. 3.1 / App. C.1 pipeline: an LLM main agent (verbatim RAM-README prompt) orchestrating a challenger and a quality
verifier (prompts expanded verbatim from Fig. 8/9), an `evaluate_rubric.py` tool that runs the weak solver x3 and the strong
solver x3 and grades every answer per criterion with an LLM judge, the acceptance predicate of Sec. 3.1 (strong_avg >= 0.65,
weak_avg < 0.50, gap >= 20 pp), grouped TOO EASY / FAILED ON STRONG / FAILED QUALITY CHECK feedback, "ENTIRELY NEW question
from a DIFFERENT angle" refinement, the end-of-loop quality verifier, and the CoT Self-Instruct baseline (single shot +
same QV + 3+3 solver attempts). Table-1 statistics are produced by `autodata-stats`.

## 2. Substitutions (user rulings) and their consequences
| Paper | Here | Consequence observed |
|---|---|---|
| Kimi-K2.6 (orchestrator, challenger, QV, judge) | GLM-5.3 (nvidia/GLM-5.3-NVFP4, reasoning_effort=max, max_tokens 81,920) | challenger calls generate 40-65k reasoning tokens (~9 min single-stream); judge calls 2-30k tokens |
| Qwen3.5-397B-A17B strong solver | Qwen3.8-27B-FP8 (thinking, 32,768 max tokens) | often fails hard questions or exhausts the 32k budget while thinking -> FAILED ON STRONG dominates round rejections (paper: TOO EASY dominated, 80%) |
| Qwen3.5-4B weak solver | same (thinking, 32,768) | scores 0-0.4 on round-1 questions |
| S2ORC CS 2022+ | s2orc_v2 shard 0, CS label from S2, 2022+ | 38% of "CS" papers are CS only by the S2 classifier (medical/physics ML included) |

## 3. Fidelity decisions (see the interview knowledge base, ambiguities.md)
- Acceptance thresholds: Sec. 3.1 prose (default) — the appendix/README "deployed" form is a config preset.
- 15 rounds per paper (paper: unspecified; mean 6.59, legal cap 15). Solvers: temperature 1.0, thinking on, 32k tokens.
- Rubric score = clip((earned − penalty) / max_positive, 0, 1), binary per-criterion judgments, judge never sees the
  reference answer, solvers see context + question only.
- Truncated solver responses (finish_reason=length, no final answer) are graded as-is (score 0), not retried.
- Judge: reasoning_effort max (user ruling); on a verdict-less truncation the retry steps the effort down (error path only).
- Guardrails (paper §6 "agents trying to cheat"): isolated evaluator subprocess, write allow-list, provenance-verified
  reports, harness-recomputed verdicts, QV bound to the evaluated question, frozen accepted candidate, round budget.

## 4. Serving (4x B200)
GLM-5.3-NVFP4 TP4+EP, fp8 KV, MTP(1) — 491k-token KV; Qwen3.8-27B-FP8 TP2 (GPU0-1) + MTP(2); Qwen3.5-4B DP2 (GPU2-3) + MTP(2).
No weight offload; KV offload impossible (/dev/shm 200 GB). Measured: GLM ~110 tok/s single stream, ~450-500 tok/s aggregate
at 8-32 concurrent; solvers slow to 40-50% while GLM is busy. GPUs 0-1 keep only ~3.4 GB free (watch for OOM).

## 5. Verification
- 669 offline tests (fake OpenAI server; e2e loop, guardrails, evaluator, corpus builder, pipeline driver, CoT baseline, stats).
- Review rounds: (1) Codex gpt-6-astra xhigh read-only review -> P0/P1 fixes (isolated evaluator, write allow-list,
  provenance-verified reports, QV binding, resume/lock/deadline fixes); (2) code-review skill (max effort; some angles hit
  the session rate limit) -> canonical-path allow-list, subagent paper-only view, models/prompts/config provenance,
  strict QV binding (question + context head + every criterion), timeout clamp, truncation-aware client retries,
  reasoning-aware context budget (+ token budget 290k), formatted-verdict parsing, last-valid-JSON extraction,
  validated acceptance overrides, flock locks, archive-on-rerun, archive-aware stats, strong-only gated on a passing weak
  run, shared retry/request helpers, CoT/pipeline dedupe (Codex F2/F3).
- Round 3 (final Codex gpt-6-astra review, 15 findings; decisions in docs/knowledge-base/review-decisions.md): two
  P0s confirmed and fixed — the public checkout was missing `src/autodata/data/` (unanchored ignore rule) and all solver
  attempts carried the same seed (two attempts with identical responses observed), so the 3 attempts were not independent;
  the 22 accepted smoke-run items were evaluated before the seed change and are unaffected, the pilot evaluations made
  under the shared seed were discarded and the pilot restarted at 13:44. Also fixed: the CoT baseline now evaluates the
  strong solver unconditionally (`--force-strong`, unavailable to the agent), QV binding includes each criterion's weight
  and |weight| is bounded to 1..10, repeated QV / repeated evaluation / non-challenger questions / evaluation after
  acceptance are refused, the accepted round reports its frozen verdict, the evaluator dies with its parent, and a NOTICE
  file credits the third-party prompt text. Delegated: hard context preflight, final-QV repair on resume, endpoint health
  gate, cohort manifest + cohort-level statistics, config/prompt fingerprint enforcement.
- GLM-5.3 thinking settings verified on the live server: interleaved thinking on (within-turn reasoning rendered back
  before each tool call), preserved thinking off (`clear_thinking: true` clears earlier user turns' thinking).
- Real-server probes: challenger JSON (14-15 criteria), QV 7-line verdict, judge JSON, evaluator reports.
- First accepted items verified end to end: harness verdict == agent's result.json claim; final QV passed.

## 6. Results (smoke, 24 papers) — runs/smoke24_stats.round4.json (recomputed 2026-09-27 from acceptance-time reports)
| Metric (Table 1 analogue) | Paper: CoT | Paper: Agentic | Ours: CoT (n=22) | Ours: Agentic accepted (n=22) | Ours: after final QV (n=20) |
|---|---|---|---|---|---|
| Weak solver avg | 0.677 | 0.458 | 0.234 | 0.254 | 0.245 |
| Strong solver avg | 0.696 | 0.772 | 0.507 | 0.848 | 0.848 |
| Gap (strong − weak) | 0.019 | 0.314 | 0.272 | 0.595 | 0.603 |
| Agentic rounds (mean / median / max) | 1 | 6.59 | 1 | 3.41 / 3 / 7 | — |
| Question length (chars) | 723 | 619 | 659 | 1011 | — |
| Rubric items | 13.2 | 13.1 | 14.5 | 14.4 | — |

Scores are read from the evaluator report that produced each acceptance (22/22 report-backed). The earlier table
(weak 0.257) had picked up, for one paper, a weak-only evaluation run after acceptance; that is now refused by the
harness and excluded by the statistics.

* Acceptance: 22 of 24 papers (92%) within ≤ 7 rounds; 20 of the 22 also pass the end-of-loop quality verifier.
  The 2 non-accepted papers ended in a context-length error of the main agent (prompt 318k tokens + 81,920 output >
  400k max-model-len after 4 and 8 rounds); the reasoning-aware token budget added afterwards prevents this and the two
  papers are being rerun.
* Failed rounds (65): FAILED ON STRONG 45%, TOO EASY 41%, FAILED QV 12% (paper: 80% too easy, 13% strong failed).
* Cost: ~75.6M main-agent prompt tokens (prefix-cached), 1.7M main-agent completion tokens, 3.9M subagent completion
  tokens for 24 papers; ~8.5 h wall clock at 8-way concurrency (≈ 3 papers/hour).
* 36% of the single-shot CoT questions would already satisfy the solver criteria; in the paper only ~2% of accepted
  questions needed a single agentic round (ours: 3 of 22). Five papers show a bookkeeping mismatch between the agent's
  result.json and the harness verdict (the harness verdict is authoritative).
* Reading: with a 27B strong solver, single-shot GLM questions are already hard for the 4B (weak 0.23); the loop's main
  effect is to find questions the strong solver can actually answer (strong 0.51 → 0.85) while keeping the weak solver
  low — the paper's discriminative objective reached from the opposite starting point (paper: CoT questions too easy).

## 6b. Pilot (320 papers) — interim snapshot 2026-09-28 07:06, paused for a container restart (runs/pilot_stats.interim.json)
| Metric (Table 1 analogue) | Paper: CoT | Paper: Agentic | Ours: CoT (n=86) | Ours: Agentic accepted (n=60) | Ours: after final QV (n=55) |
|---|---|---|---|---|---|
| Weak solver avg | 0.677 | 0.458 | 0.253 | 0.253 | 0.246 |
| Strong solver avg | 0.696 | 0.772 | 0.459 | 0.831 | 0.832 |
| Gap (strong − weak) | 0.019 | 0.314 | 0.206 | 0.579 | 0.585 |
| Agentic rounds (mean / median / max) | 1 | 6.59 | 1 | 3.85 / 4 / 9 | — |
| Question length (chars) | 723 | 619 | 663 | 1083 | — |
| Rubric items | 13.2 | 13.1 | 14.5 | 13.9 | — |

* Progress: 60 of 320 agentic papers completed (22 reused from the smoke run), all accepted within 9 rounds; 55 also
  pass the end-of-loop quality verifier. CoT: 86 of 320 evaluated; 20% of single-shot questions would already satisfy
  the solver criteria. All accepted items are scored from their acceptance-time evaluator report.
* Failed rounds (171): FAILED ON STRONG 51%, TOO EASY 34%, FAILED QV 9%, no QV/eval 5% (paper: 80% too easy, 13%
  strong failed) — as in the smoke run, the 27B strong solver, not the weak solver, is the binding constraint.
* Nine pilot papers were refused acceptance only by harness checks withdrawn in review round 4 (blocking in-loop QV
  binding; evaluator/harness version skew; rubric weight range). They were re-adjudicated from the recorded judgments
  with an exact recomputation (scripts/readjudicate_refused.py) and all nine then passed the end-of-loop verifier.
* Throughput: ~2.2 papers/hour at 8 agentic + 2 CoT slots (GLM-5.3 is KV-bound by orchestrator contexts of up to
  ~220k tokens; the 290k context budget is kept for fidelity by user decision); ~5 days remain for 260 papers.
* Paused 2026-09-28 07:06 for a container restart; `scripts/resume_pilot.sh` resumes it (interrupted papers rerun).

## 7. Known limitations / next steps
- Throughput: measured GLM-5.3 generation is ~490 tokens/s aggregate at 8+2 concurrent papers, or ~10-16 rounds/hour;
  a 320-paper pilot needs roughly 3-4 days. Resumable (`autodata-run-cs ... --workdir-root runs/pilot`).
- The strong solver is far weaker than the paper's; expect a lower acceptance rate and a different failure-mode mix.
- Legal (App. C.2), scientific (App. C.3), meta-optimization (Sec. 4) and RL are not implemented yet.
