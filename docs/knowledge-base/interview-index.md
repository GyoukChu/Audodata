# Interview: Autodata / Agentic Self-Instruct reproduction (arXiv 2606.25996v3)
**Date:** 2026-09-27
**Status:** Interview complete (3 rounds) -> implementation phase
**Depth:** deep (implementation-grade spec)

## Goal
Reproduce the Agentic Self-Instruct data-creation loop from "Autodata: An agentic data
scientist to create high quality synthetic data" (Kulikov et al., FAIR at Meta, v3 2026-07-04)
as faithfully as possible, with these substitutions:
- Kimi-K2.6  -> GLM-5.3          (orchestrator / challenger / quality verifier / judge)
- Qwen3.5-397B-A17B -> Qwen3.8-27B (strong solver)
- Qwen3.5-4B stays (weak solver)
Environment: 4x NVIDIA B200 (183 GB each), CUDA 13.1 toolkit / driver 580 (CUDA 13.0), uv 0.9.18,
Python 3.12, 72 CPUs, 2.2 TB RAM. HF cache must live at
$AUTODATA_CACHE_DIR. RL training is out of scope for now.

## Themes Discovered
1. Paper method spec (what the loop does, exactly)          -> paper-analysis.md
2. Verbatim prompts (paper App. C + Meta RAM README)         -> prompts-verbatim.md
3. Infrastructure / model serving decisions                  -> infra-decisions.md
4. Method ambiguities needing a ruling                        -> ambiguities.md
5. Scope, scale, data source                                  -> scope-and-data.md

## Files Created
- _interview-index.md (this file)
- paper-analysis.md
- prompts-verbatim.md
- ambiguities.md
- sources/ (paper PDF+txt, RAM README, RAM images)
- infra-decisions.md
- scope-and-data.md
- _summary.md
- _open-questions.md

## Implementation status (2026-09-27)
- Project: <repo> (spec: docs/IMPLEMENTATION_SPEC.md, runbook: docs/RUNBOOK.md)
- Delegated: harness+client (Codex A), evaluate_rubric CLI (Codex B), S2ORC corpus (Claude C), vLLM serving (Claude D)
- Orchestrator-owned: prompts/cs/*, cs/{prompts,parsing,run_paper,pipeline,cot_baseline,stats,corpus_io}.py, tests
- New user constraints: GPU idle-deletion policy (>1.5% memory must stay used) -> scripts/gpu_guard.sh; no keepalive during
  runs; serving must be tuned (language-model-only, DP, MTP spec-dec) -> docs/tasks/E_serving_optimization.md
- 03:00 status: all modules implemented (340 tests pass), guardrails added after a Codex xhigh review, all three vLLM
  servers up (GLM-5.3-NVFP4 TP4 no offload: 649k-token KV; Qwen3.8-27B-FP8 TP2; Qwen3.5-4B), corpus built
  (smoke 24 / pilot 320 papers, s2orc_v2 release 2026-09-22), real-server smoke runs in progress; serving optimization
  (MTP spec-dec, DP for 4B) queued behind the smoke tests.
- 11:05 status: 24-paper smoke run at 14/24 (all accepted, mean 2.7 rounds; interim Table-1 numbers in docs/REPORT.md §6);
  first code-review pass (code-review skill, max effort; several angles hit the session rate limit but 4 delivered) found
  a write allow-list bypass, unbounded context accounting, stale locks, stats counting archives, QV markdown parsing, and
  duplication between CoT/pipeline; harness-owned fixes applied (run_paper), shared-module fixes delegated to Codex F2, dedupe to F3.
- 11:20: Codex F2 (gpt-6-astra xhigh) applied the shared-module review fixes: truncation-aware client retries, context
  budget counting reasoning/tool-args (+ token budget option), formatted QV verdict parsing, last-valid-JSON extraction,
  validated acceptance overrides + endpoint seeds, flock-based per-paper locks under _locks/, archive-on-rerun to _archive/,
  archive-aware stats, strong-only gated on a passing weak run with provenance checks, shared retry/request helpers;
  461 tests pass. F3 (CoT/pipeline dedupe + tests) launched.
