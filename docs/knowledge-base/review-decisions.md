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

## Round 4 — three Codex gpt-6-astra (xhigh) read-only reviews on a5f72f3 (2026-09-27 14:30-15:00): paper fidelity, guardrail correctness, pipeline/stats/docs
Triage by the orchestrator; every ACCEPT was re-verified against the code, the paper and (where possible) run data before delegation (docs/tasks/round4_job1_core.md, round4_job2_pipeline.md).
| # | Finding | Decision | Reason / evidence |
|---|---|---|---|
| 1 | `--force-strong` never bypassed the first weak gate in the evaluator stage loop -> CoT items with weak >= 0.50 had no strong score and were regenerated on resume (selection into the baseline) | ACCEPT (P0) | confirmed in code and in runs/smoke24_cot (s2_247188082, s2_248227281 had strong=None); one-line fix applied immediately (evaluator subprocess picks it up); regression test + CoT repair path (complete the frozen candidate instead of regenerating) |
| 2 | QV binding: first-80-alphanumerics window misses weights after long criteria; unsigned weight ("1" in "10", +5 -> -5 unbound) | ACCEPT (P1) | 269 of 317 accepted smoke criteria exceed 120 alphanumerics; the check passed only because of the agent's prompt format; new rule = full criterion + signed weight within 60 chars before/after, validated against real QV prompts |
| 3 | Evaluation cache bypass via JSON key order / whitespace (hash of json.dumps) and via mode switch (weak-only -> both) | ACCEPT (P0/P1) | canonical hash (sort_keys, NFC) + stage-aware cache; legacy hashes still matched for the running pilot |
| 4 | Challenger-origin check compares alphanumerics only (operator/Greek-letter edits pass) | ACCEPT (P0) | NFC + whitespace-collapsed exact comparison; unparseable challenger JSON -> exact substring of its output (the main agent may repair JSON, never edit the question) |
| 5 | QV template echo (`OVERALL: PASS | FAIL`) parses as PASS | ACCEPT (P1) | unresolved alternatives -> None / missing check |
| 6 | Rubric weight range 1..10 blocks acceptance (my round-3 addition) | ACCEPT (fidelity fix) | the paper's harness never rejects rubrics by weight range and Fig. 9 checks only counts; pilot paper s2_168844958 lost two rounds to it; now a non-blocking warning |
| 7 | Smoke summary of s2_288742856 carries a weak score from a weak-only run made AFTER acceptance (0.1696 vs accepting report 0.0994) | ACCEPT (P0 for reporting) | eval-after-accept is refused since round 3; statistics now read the acceptance-time report; smoke numbers to be regenerated |
| 8 | REPORT.md "paper: 2%" misattributed | ACCEPT (docs) | the paper's 2% is "only ~2% use a single round" (Sec 4), reworded |
| 9 | Final-QV repair keeps resolved `final_qv:` errors / `completed=False` -> next resume regenerates an accepted paper; `question_type` missing -> IndexError; repair transcript name reuse; repaired summaries re-stamped with the current fingerprints | ACCEPT (P1) | confirmed by reading the repair path; terminal state = accepted + final QV completed; original fingerprints preserved, repair fingerprints recorded separately |
| 10 | Context preflight uses cumulative retry usage as the prompt size | ACCEPT (P1) | last successful request's usage exposed separately |
| 11 | PR_SET_PDEATHSIG armed late (after imports) -> orphan on early parent death | ACCEPT (P1, cheap) | armed at import, expected parent pid checked |
| 12 | Cohort manifest ignores corpus sha1 and imported legacy fingerprints; per-paper text changes reuse old results | ACCEPT (P1) | corpus fields in the mismatch check + `imports` record + paper.txt sha1 check on resume |
| 13 | Corpus builder resume ignores changed filters | ACCEPT (P2) | validate existing records + `<out>.meta.json` |
| 14 | S2 limiter floor 1.0 s; shared-file gate drops deadlines > 600 s | ACCEPT (P1) | floor 3.0 s (user rule), gate honours up to 900 s |
| 15 | CoT QV infrastructure failure = permanent rejection | ACCEPT (P1) | error + repair on the frozen candidate |
| 16 | Stale test count in docs | ACCEPT (docs) | |
| 17 | "Reject an unparseable challenger candidate outright" | REJECT (partial) | the paper's main agent may repair malformed challenger JSON; keeping an exact-substring fallback preserves that behaviour while blocking edits |
| 18 | "Construct the QV request from the immutable candidate in the harness" | REJECT | in the paper the main agent composes the quality-verifier prompt (RAM README); the harness only verifies that the prompt contained the evaluated candidate |

### Round 4 addendum — in-loop QV binding made informational (orchestrator decision, 2026-09-27 15:40)
Codex's strict binding (full criterion text or 80-char prefix + signed weight adjacent) bound only 14 of the 22
accepted smoke candidates. Checking the accepted candidates against their rounds' challenger outputs showed why: the
main agent (GLM-5.3) rewrote the candidate before evaluation in the smoke run (rubric differs from the challenger's in
11/22 accepted items, context in 5/22, question in 2/22) and paraphrased or relabelled the rubric in its
quality-verifier prompts (P/N labels, magnitudes without signs, shortened criteria). The paper's pipeline has no
harness-side binding at all, and its real quality filter is the end-of-loop verifier, which this harness runs on the
exact accepted candidate with a harness-built prompt. A blocking in-loop binding would therefore have refused or
delayed ~36% of paper-faithful acceptances. Decision: `qv_bound` / `qv_missing` are recorded on every round and on the
accepted record (event `qv_not_bound`, informational; statistic `n_qv_unbound`), acceptance is never refused for it.
The question-origin check stays blocking (the verbatim prompt says the main agent must not write questions), the
QV-repeat refusal stays keyed on the question (the prompt says a failed QV goes back to the challenger with feedback).

### Round 4 incident — evaluator/harness version skew during the pilot (2026-09-27 15:09-15:39)
The evaluator runs as a fresh subprocess and imports the working tree, while runner processes keep the code they
started with. A Codex edit at 15:09 switched the evaluator's report `question_hash` to a new canonical identity; the
running pilot harness (old code) then marked every new report unverified ("question_hash mismatch", cascading into
"weak result provenance"). One genuine acceptance was lost (runs/pilot/s2_281674067, round 2: weak 0.348, strong
0.788). Fix at 15:39: `question_hash` keeps the legacy identity, `question_hash_canonical` carries the new one; harness,
evaluator reuse and statistics accept either. Affected papers are re-adjudicated by scripts/readjudicate_refused.py
(exact recomputation from the recorded judgments; qualifying only when every refusal reason is a withdrawn check), which
also covers runs/pilot/s2_269282862 (refused only by the formerly blocking QV binding). Rule added to the runbook: any
evaluator report change must stay readable by already-running harness processes.
