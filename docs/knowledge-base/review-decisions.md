# Review findings and orchestrator decisions

Each finding from the review rounds is recorded with the decision (ACCEPT / REJECT / DEFER) and the reason. Findings are not
applied automatically; fidelity to the paper wins over reviewer preference when they conflict.

## Round 1 — Codex gpt-6-astra read-only review (2026-09-27 ~03:30)
| # | Finding | Decision | Reason |
|---|---|---|---|
| 1 | Evaluator could import a planted `autodata.py` from the workspace | ACCEPT | real guardrail gap; `python -I` + write allow-list |
| 2 | Acceptance trusted agent-writable config/reports | ACCEPT | provenance checks + harness recomputation |
| 3 | Acceptance not bound to an immutable candidate | ACCEPT | QV binding + frozen candidate |
| 4 | Contradictory QV output could pass | ACCEPT | strict four-check rule; the paper's QV lists the four verdicts then OVERALL |
| 5 | Verbatim multi-line evaluator command rejected by the sandbox | ACCEPT | the prompt is verbatim, the sandbox must accept it |
| 6 | Paper ids could escape the run root / collide | ACCEPT | sanitized ids, locks |
| 7 | Resume treated errors as completed | ACCEPT | rerun errored papers, archive old workdirs |
| 8 | Evaluator deadline / cancellation | ACCEPT | deadline from config, cancellation-safe kill |
| 9 | CoT lacked the final filter parity | ACCEPT | same programmatic checks for the baseline cohort |
| 10 | No context-window budget | ACCEPT | char budget, later token budget |
| 11 | E2E test proved too little | ACCEPT | rewritten with guardrail scenarios |
| 12 | Endpoint concurrency not global across evaluator subprocesses | DEFER | vLLM queues requests; measured no failures at 8 papers |
| 13 | run.seed unused | ACCEPT (later, F2) | wired as ModelEndpoint.seed, best-effort |

## Round 2 — code-review skill angles (2026-09-27 ~09:00; several angles rate-limited)
| # | Finding | Decision | Reason |
|---|---|---|---|
| 1 | write allow-list bypass via `/workspace/project/./x`, trailing slash | ACCEPT | verified; now canonical path via the tool hook |
| 2 | Duplicated predicate / evaluator client / argv parsers | ACCEPT partially | shared retry/request helpers (F2); kept the harness-side independent recomputation on purpose (a second implementation is the guardrail) |
| 3 | Subagents could read result.json / transcripts | ACCEPT | paper-only view; matches the paper (challenger/QV read the paper) |
| 4 | QV bound only by the first 120 chars of the question | ACCEPT | question + context head + every criterion |
| 5 | Agent-supplied --timeout drives the deadline | ACCEPT | clamped to config |
| 6 | Empty-content/length completions retried 7x by the client | ACCEPT | truncation is a result, not a transient error |
| 7 | fit_context counted only content | ACCEPT | reasoning + arguments counted; token budget from vLLM usage |
| 8 | Stale O_EXCL locks after SIGKILL | ACCEPT | flock, acquired inside the semaphore |
| 9 | Stats counted archived workdirs | ACCEPT | `_archive/` + archive-aware stats |
| 10 | QV markdown-bold verdict lines not parsed | ACCEPT | verified on real outputs |
| 11 | acceptance overrides not validated | ACCEPT | validated at load |
| 12 | strong-only without a passing weak result still ran the strong stage | ACCEPT | pure compute saving consistent with Sec 3.1 ("evaluate the strong solver only if the weak solver passes") |
| 13 | "Prompts must be verbatim; expanded challenger/QV prompts need a knob" | REJECT | the paper prints condensed figures (Fig. 8-9), not the original files; an expansion is unavoidable and is documented in prompts/cs/README.md; a "knob" would only switch between two non-original texts |
| 14 | Replace the shell sandbox with Python implementations of cat/ls/... | REJECT | the verbatim prompts say `cat ./paper.txt` via bash; a real (allow-listed) subprocess keeps the agent experience faithful |
| 15 | Remove chat_sync / per-call client construction | DEFER | not on the hot path (evaluator uses SyncClient); revisit if profiling shows cost |
| 16 | EVALUATE_RUBRIC_SHIM never executed | REJECT (keep) | the file mirrors the paper's `.opencode/tools/evaluate_rubric.py` layout that the prompt references |
| 17 | Untested modules (pipeline, cot, stats, corpus_io) | ACCEPT | tests added (F2/F3) |
| 18 | GLM `reasoning` vs `reasoning_content` in history | ACCEPT | send both; verified with /tokenize |

## Round 3 — final Codex gpt-6-astra read-only review (2026-09-27 13:35)
| # | Finding | Decision | Reason / evidence |
|---|---|---|---|
| 1 | `src/autodata/data/` missing from git (unanchored `data/` ignore) | ACCEPT (P0) | verified with `git ls-files`; rules anchored, package committed |
| 2 | CoT strong stage gated by weak-first (my F2 gate) -> biased CoT column, regeneration on resume | ACCEPT (P0) | real: the paper's Table 1 CoT column has both solvers on every item; added `--force-strong` for the baseline only (sandbox rejects it for the agent) |
| 3 | All solver attempts used seed 0 -> identical samples | ACCEPT (P0) | verified: two attempts with identical response hashes; default now unseeded; when a run seed is set, seeds are derived per (question, role, attempt); pilot evaluations made under seed 0 were discarded (smoke24 accepted results were all pre-seed and unaffected) |
| 4 | QV binding ignored weights -> post-QV weight inflation | ACCEPT (P1) | probe reproduced; binding now requires each criterion's weight next to its text; |weight| bounded to 1..10 (Fig. 8 challenger spec) in verification and final filter |
| 5 | QV/evaluation shopping within a round | ACCEPT (P1) | repeated QV on the same question refused; completed evaluations cached; evaluation of a non-challenger question refused; no evaluation after acceptance — matches the paper's loop (fail -> ENTIRELY NEW question) |
| 6 | Accepted summary mixed frozen text with later scores | ACCEPT (P1) | accepted round now reports the frozen verdict |
| 7 | Context budget advisory, no hard preflight | ACCEPT (P1) | delegated (Codex F4): hard preflight against max-model-len incl. shrinking the protected window and lowering max_tokens |
| 8 | SIGKILL'd harness leaves an evaluator writing | ACCEPT (P1, cheap) | evaluator sets PR_SET_PDEATHSIG (SIGTERM on parent death) |
| 9 | Final-QV infrastructure failure = permanent rejection; outage burns queued papers | ACCEPT (P1) | delegated (F4): resume repairs an incomplete final QV; endpoint health gate before each paper |
| 10 | Stats hide papers without summaries | ACCEPT (P1) | delegated (F4): cohort manifest + cohort/completed rates |
| 11 | Resume mixes incompatible runs silently | ACCEPT (P1) | delegated (F4): manifest with config + prompt hashes; mismatch is an error unless --allow-config-mismatch (used deliberately to reuse smoke results after non-semantic config changes) |
| 12 | Third-party text licensing | ACCEPT (P1) | NOTICE added (RAM MIT notice; paper prompt quotes attributed); code license left to the repository owner |
| 13 | GLM canary replays only reasoning_content | ACCEPT (P2) | delegated (F4): canary uses both fields + /tokenize check |
| 14 | README lacks readiness barriers | ACCEPT (P2) | delegated (F4) |
| 15 | Storage duplication | DEFER | not a correctness issue; revisit before the 10k run |
| verdict | "not yet ready to claim a Table-1 comparison" | PARTIALLY ACCEPT | smoke24 numbers are valid w.r.t. seeds (all accepted items evaluated unseeded) and the CoT column was computed ungated (before the F2 gate existed); they remain provisional because of #4/#5 (no exploitation observed in transcripts, but not prevented at the time). The pilot is rerun from scratch under the fixed harness. |
