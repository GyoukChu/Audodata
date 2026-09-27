# Autodata — Agentic Self-Instruct reproduction (CS research-paper pipeline)

A from-the-paper reimplementation of **Agentic Self-Instruct**, the practical instantiation of
*Autodata: An agentic data scientist to create high quality synthetic data* (Kulikov et al., FAIR at Meta,
[arXiv:2606.25996](https://arxiv.org/abs/2606.25996)), run end to end on a single 4× B200 node with open weights.

The main orchestrator agent (verbatim prompt from Meta's RAM README) drives a **challenger** and a **quality verifier**
subagent, tests every candidate question with an `evaluate_rubric.py` tool (weak solver ×3, strong solver ×3, an LLM judge
scoring every rubric criterion), accepts a question only when the strong solver clearly beats the weak solver, and otherwise
feeds grouped `TOO EASY / FAILED ON STRONG / FAILED QUALITY CHECK` feedback back to the challenger for an *entirely new
question from a different angle*. The CoT Self-Instruct baseline and Table-1-style statistics are included.

| Role (paper) | Paper model | This repo (default config) |
|---|---|---|
| Orchestrator / challenger / quality verifier / judge | Kimi-K2.6 | GLM-5.3 (`nvidia/GLM-5.3-NVFP4`, vLLM, reasoning effort max) |
| Strong solver | Qwen3.5-397B-A17B | `Qwen/Qwen3.8-27B-FP8` (thinking, 32k tokens) |
| Weak solver | Qwen3.5-4B | `Qwen/Qwen3.5-4B` (thinking, 32k tokens) |
| Corpus | S2ORC CS papers, 2022+ | Semantic Scholar `s2orc_v2` shards, CS, 2022+ |

Status (2026-09-27): pipeline implemented and running; 24-paper smoke run in progress, 320-paper pilot corpus built.
Legal (App. C.2), scientific-reasoning (App. C.3), meta-optimization (Sec. 4) and RL training are not implemented yet.

## Layout
```
prompts/cs/          verbatim / expanded prompts (main_agent.md is the RAM-README prompt with the thresholds templated)
configs/             cs_default.yaml (paper-faithful), cs_pilot.yaml (same + concurrency 8)
src/autodata/
  config.py          every threshold / budget / sampling knob (acceptance presets prose_s31 and deployed_c1)
  llm/client.py      OpenAI-compatible client (vLLM), retries, reasoning/tool-call handling
  harness/           generic tool-calling Agent, sandboxed task/bash/write/read tools, context budget
  cs/                run_paper.py (one paper through the loop, guardrails), evaluate_rubric.py (the tool),
                     judge.py, solvers.py, rubric.py, pipeline.py, cot_baseline.py, stats.py, parsing.py
  data/              Semantic Scholar client (strict rate limiter) and s2orc_v2 corpus builder
serving/             vLLM launch scripts, health checks, benchmark harness and notes (4× B200 layout)
scripts/             smoke / launch / inspection helpers, GPU guard
docs/                IMPLEMENTATION_SPEC.md, RUNBOOK.md, REPORT.md, knowledge-base/ (paper analysis, prompts, rulings)
tests/               669 offline tests against a fake OpenAI-compatible server (no GPU, no network)
```

## Quick start
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]" vllm==0.30.0          # vLLM only for serving
cp .env.example .env                              # HF_TOKEN, S2_API_KEY
echo 'export AUTODATA_CACHE_DIR=/path/to/cache' > env.local.sh   # optional, default ./.cache
source env.sh
python -m pytest -q                               # offline tests
bash serving/serve_glm53.sh                         # see serving/NOTES.md
until curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null; do sleep 15; done
bash serving/serve_qwen27b.sh
until curl -fsS --max-time 3 http://127.0.0.1:8001/v1/models >/dev/null; do sleep 15; done
bash serving/serve_qwen4b.sh
bash scripts/wait_servers.sh                       # all three must be ready
autodata-build-corpus --out data/corpus/cs2022_smoke.jsonl --n-papers 24 --shard-indices 0
bash scripts/wait_servers.sh                       # recheck before the pipeline
autodata-run-cs --config configs/cs_pilot.yaml --corpus data/corpus/cs2022_smoke.jsonl --workdir-root runs/smoke24
bash scripts/wait_servers.sh
autodata-cot-baseline --config configs/cs_pilot.yaml --corpus data/corpus/cs2022_smoke.jsonl --workdir-root runs/smoke24_cot
autodata-stats --run-root runs/smoke24 --cot-root runs/smoke24_cot
```
See `docs/RUNBOOK.md` for the full sequence and `docs/REPORT.md` for results and deviations.
Each run root contains `cohort.json` with the requested paper IDs and config/prompt provenance. Resuming with changed
config or prompts stops unless `autodata-run-cs --allow-config-mismatch` is supplied; overrides are recorded in manifest
history. Statistics report acceptance over completed papers and over the full requested cohort, including pending papers.

## Fidelity notes
* Acceptance thresholds default to the Sec. 3.1 prose (strong ≥ 0.65, weak < 0.50, gap ≥ 20 pp); the appendix/README
  "deployed" thresholds are the `deployed_c1` preset. Every other implicit number (15 rounds, 3 attempts, solver sampling,
  rubric score = clip((earned − penalty)/max_positive, 0, 1)) is documented in `docs/knowledge-base/ambiguities.md`.
* The harness re-verifies every evaluation (provenance, thresholds recomputed from per-criterion judgments) and binds
  acceptance to the quality-verified candidate; the agent cannot accept by claiming (paper §6 on agents "cheating").
* Truncated solver answers are graded as-is; subagents can read only the paper.

## Acknowledgements
Method and prompts: Kulikov et al., *Autodata* (arXiv:2606.25996) and the accompanying
[RAM project page](https://github.com/facebookresearch/RAM/tree/main/projects/autodata). Corpus: Semantic Scholar S2ORC
(ODC-BY). Models: Z.ai GLM-5.3 (NVIDIA NVFP4 build), Alibaba Qwen3.5 / Qwen3.8.
