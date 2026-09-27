FINAL READ-ONLY CODE REVIEW (do NOT edit files; report only). Repository: this directory = the public GitHub repo
GyoukChu/Audodata (branch main, 2 commits). It is a from-the-paper reproduction of Agentic Self-Instruct, the practical
instantiation of "Autodata: An agentic data scientist to create high quality synthetic data" (arXiv 2606.25996v3, Meta FAIR),
CS-paper pipeline (Sec 3.1 / App C.1), run on 4x B200 with GLM-5.3-NVFP4 (orchestrator/challenger/QV/judge), Qwen3.8-27B-FP8
(strong solver) and Qwen3.5-4B (weak solver) served by vLLM 0.30.

Read first: README.md, docs/REPORT.md, docs/IMPLEMENTATION_SPEC.md, docs/knowledge-base/summary.md and ambiguities.md
(every ruling and deviation), prompts/cs/*.md (main_agent.md is the verbatim Meta RAM README prompt with thresholds templated;
challenger.md / quality_verifier.md are expanded from the paper's condensed Figures 8-9), configs/*.yaml. Then review ALL of
src/autodata/**, tests/**, serving/** (shell + python), scripts/**. Two earlier review rounds were already applied (see
REPORT.md section 5), so focus on what is still wrong, not on what is documented.

Review dimensions, in priority order:
1. Correctness bugs (crashes, wrong scores/thresholds, wrong resume/lock/archive behaviour, race conditions between the 8
   concurrent papers + evaluator subprocesses + 2 concurrent CoT papers, asyncio cancellation, subprocess handling).
2. Fidelity to the paper: anything in the loop that differs from Sec 3.1 / App C.1 / the RAM README prompt beyond the
   documented rulings (acceptance thresholds, 3 attempts, weak-first gating, feedback grouping, "ENTIRELY NEW question",
   end-of-loop QV, CoT baseline definition, Table-1 statistics definitions: are our stats computed over the same population
   as the paper's Table 1?). Check the prompt files against docs/knowledge-base/prompts-verbatim.md.
3. Guardrails (paper §6 "agents trying to cheat"): can the main agent or a subagent still influence acceptance other than by
   producing a genuinely discriminative question? (write allow-list on canonical paths, subagent paper-only view,
   evaluator provenance checks: models/prompts_dir/config sha/weak_source_report, QV binding, timeout clamp, round budget.)
   Try to construct concrete bypasses; verify with small Python probes if useful (you may run python read-only).
4. Robustness for a multi-day 320-paper run: context budget (token-based) vs GLM max-model-len 400k with 81,920 output tokens,
   retries/backoff, server restarts mid-run, disk growth (transcripts), resume after SIGKILL, stats over partial runs.
5. Tests: what is untested or tested only against the fake server in a way that could hide a real-vLLM difference
   (e.g. reasoning field names `reasoning` vs `reasoning_content`, tool-call argument JSON strings, finish_reason=length).
6. Public-repo hygiene: secrets, absolute paths, institution names (only "mlilab" is acceptable), licensing of included
   third-party text (the RAM README prompt, arXiv prompt figures), README accuracy vs the code, dead files.
7. Efficiency (GPU-time): avoidable LLM calls or tokens (e.g. judge calls on empty answers, duplicate QV, re-evaluations).

Output: a prioritized list (P0 blocker / P1 should fix / P2 nice-to-have) with file:line references, a concrete failure
scenario for each, and a concrete fix (code-level). Finish with a short verdict: is the reproduction faithful enough to
report Table-1-style numbers, and what would you change before scaling to 10k papers? Be concrete and skeptical; do not
restate the code or the docs.
