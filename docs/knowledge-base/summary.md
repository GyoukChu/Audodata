# Summary: decisions for the Agentic Self-Instruct (CS) reproduction

> Deep Interview 2026-09-27, three rounds. All rulings below are final unless the user changes them.

## What is being built
A faithful reimplementation of the Sec 3.1 / App. C.1 CS-paper pipeline of Autodata's Agentic
Self-Instruct, plus the CoT Self-Instruct baseline and the Table-1 statistics, run for real on 4x B200
with a smoke test (10-20 papers) and a ~200-paper pilot. RL training, legal, scientific and
meta-optimization are later phases.

## Rulings
| # | Topic | Ruling |
|---|---|---|
| 1 | Kimi-K2.6 replacement | full GLM-5.3, local, NVFP4 (nvidia/GLM-5.3-NVFP4), vLLM 0.30, TP=4+EP on all four B200s, fp8 KV, MTP (1 draft token), no weight/KV offload (KV offload impossible: /dev/shm 200 GB) |
| 2 | Strong solver | Qwen/Qwen3.8-27B-FP8 (TP=2, GPU0-1, co-located, MTP spec-dec 2 tokens after the optimization pass) |
| 3 | Weak solver | Qwen/Qwen3.5-4B (data-parallel 2 on GPU2-3 + MTP after the optimization pass) |
| 4 | Harness | custom Python agent harness reproducing OpenCode semantics: main-agent LLM with task/bash/write tools, challenger + quality-verifier subagents with bash(cat)/write, a real evaluate_rubric.py CLI, verbatim prompts |
| 5 | Acceptance preset | default `prose_s31`: strong_avg >= 0.65, weak_avg < 0.50, gap >= 0.20 over 3 attempts; `deployed_c1` preset also implemented |
| 6 | Round budget | 15 rounds per paper + main-agent step cap |
| 7 | Solvers | thinking ON, temp 1.0 / top_p 0.95 / top_k 20, max_tokens 32,768, input = context + question only |
| 8 | GLM-5.3 reasoning | model default (reasoning_effort max) for orchestrator, challenger, QV, judge; max_tokens 81,920 for all GLM roles (user ruling 03:5x; the 32,768 cap is for the solvers only) |
| 9 | Corpus | S2ORC via Semantic Scholar API key: `s2orc_v2` shards streamed + `/paper/batch` metadata filter (CS, 2022+); 1 req / >= 3 s global limiter with backoff |
| 10 | Project root | <repo> (git, uv venv); HF cache $AUTODATA_CACHE_DIR |
| 11 | Scope | CS pipeline + CoT baseline; smoke -> pilot; 10k run decided later |
| 12 | Review | after completion, one code-review subagent pass (fable-5.1 xhigh); Codex gpt-6-astra xhigh may be used for implementation help |

## Defaults I set (documented assumptions, see ambiguities.md)
Rubric score = clip(sum w_i I_i / sum_{w>0} w_i, 0, 1) with binary per-criterion judgments and no reference
answer shown to the judge; judge prompt written from the paper's stated semantics; "no zeros" = per-attempt
(deployed preset only); end-of-loop QV = same QV prompt + programmatic checks; CoT baseline = same challenger
prompt, single shot; paper.txt = title + abstract + body text.
