# Infrastructure decisions

> Source: Deep Interview 2026-09-27 (round 1 answers)

## Key Points
- **GLM-5.3 role model (orchestrator / challenger / QV / judge): serve the FULL GLM-5.3 locally
  in NVFP4** (user: "nvidia/GLM-5.3-NVFP4 나 https://recipes.vllm.ai/zai-org/GLM-5.3?variant=nvfp4
  참고, offload 켜서 로컬 서빙"). No API. Offloading must be enabled (host has 2.2 TB RAM).
- **Harness: custom Python agent harness** that reproduces the OpenCode semantics (main-agent LLM
  with task/bash/write-equivalent tools, subagents = challenger + quality verifier, evaluation via
  an evaluate_rubric.py-equivalent), verbatim prompts. Not the OpenCode CLI.
- Environment: 4x B200 183 GB, CUDA 13.1 toolkit, driver 580.95.05 (CUDA 13.0), uv 0.9.18, Python
  3.12.3, 72 CPUs, 2.2 TB RAM. Lustre workspace 49 TB free; /home is an overlay (799 GB free).
- HF cache: $AUTODATA_CACHE_DIR (HF_HOME / HF_HUB_CACHE).
- Serving stack candidates: vLLM 0.30.0 (PyPI 2026-09-22; wheels for cu128/cu129/cu130; Blackwell
  needs >= 12.8) or SGLang 0.5.20. transformers 5.17.0.
- Solvers: Qwen3.8-27B (dense, 55.6 GB BF16, FP8 variant exists), Qwen3.5-4B (Apache 2.0).
  Qwen thinking-mode sampling defaults: temp 1.0, top_p 0.95, top_k 20, min_p 0; 4B card:
  presence_penalty 1.5; 27B card: presence_penalty 0.0. Both native context 262,144.

- **Project root: <repo>** (Lustre, persistent; git repo;
  uv venv inside; data + results inside). RULED 2026-09-27.

## Round-3 rulings (2026-09-27)
- GPU layout: **no weight offload; co-locate.** GLM-5.3-NVFP4 TP=4 + EP on all four GPUs (gpu-memory-utilization
  ~0.72), Qwen3.8-27B-FP8 TP=2 on GPU0-1, Qwen3.5-4B on GPU2 (optionally a replica on GPU3), small memory
  fractions each. KV-cache offload to CPU (`--kv-offloading-size`) allowed as an option; prefetch weight offload
  only as a fallback if it does not fit.
- Strong solver checkpoint: **Qwen/Qwen3.8-27B-FP8** (official FP8).
- GLM-5.3 reasoning: **model default (reasoning_effort=max) for all four roles**, configurable.
- Downloads started 2026-09-27 into the cache: nvidia/GLM-5.3-NVFP4, Qwen/Qwen3.5-4B, Qwen/Qwen3.8-27B-FP8,
  Qwen/Qwen3.8-27B (BF16 kept as backup).
- vLLM 0.30.0 default PyPI wheel (cu129 runtime) is fine on the CUDA 13.0 driver; no cu130 asset exists in the
  v0.30.0 GitHub release.

## Open Questions
- Exact GPU layout: GLM-5.3-NVFP4 TP=4 across all four GPUs + co-located solvers, vs. dedicated
  GPUs; what "offload" means concretely in the vLLM recipe (weights via --cpu-offload-gb vs KV
  cache offload). Throughput implications for thousands of long generations.
- Codex (gpt-6-astra-xhigh) subagent to be used for heavy implementation/diagnosis passes.

## Details (verified 2026-09-27)
### Model checkpoints (HF API; none gated; HF_TOKEN present in .env)
| Repo | Size | Notes |
|---|---|---|
| nvidia/GLM-5.3-NVFP4 | 464.2 GB (50 shards) | GlmMoeDsaForCausalLM, 78 layers, 256 routed experts, 8 active, first 3 dense layers kept FP8, MoE expert linears NVFP4 (ModelOpt); license: NVIDIA Open Model Agreement + GLM-5.3 license; card shows SGLang cmd (--quantization modelopt_fp4, --reasoning-parser glm45, --tool-call-parser glm47) |
| Inferact/GLM-5.3-NVFP4 | 464.9 GB | the checkpoint used by recipes.vllm.ai nvfp4 variant: `vllm serve Inferact/GLM-5.3-NVFP4 --tensor-parallel-size 8 --enable-expert-parallel --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice --kv-cache-dtype fp8_e4m3` (vLLM >= 0.29) |
| zai-org/GLM-5.3 (FP8) | 755.7 GB | does not fit 4x B200 |
| Qwen/Qwen3.8-27B | 55.6 GB BF16 | arch Qwen3_5ForConditionalGeneration (same code path as Qwen3.5) |
| Qwen/Qwen3.8-27B-FP8 | 30.9 GB | official FP8 |
| Qwen/Qwen3.5-4B | 9.3 GB | Qwen3_5ForConditionalGeneration |
### vLLM 0.30.0 (PyPI 2026-09-22)
- Registry has GlmMoeDsaForCausalLM (via deepseek_v32 code), Qwen3_5ForConditionalGeneration (+ Qwen3_5MTP).
- Reasoning parsers: `glm47` (GLM-5.x), `qwen3` family. Weight offload flags exist: `--offload-backend`
  (uva | prefetch), `--cpu-offload-gb`, `--cpu-offload-params`, `--offload-group-size`,
  `--offload-num-in-group`, `--offload-prefetch-step`, `--offload-params`; KV offload:
  `--kv-offloading-size`, `--kv-offloading-backend`.
- Memory math for GLM-5.3-NVFP4 on 4x B200: 464 GB / TP4 = 116 GB weights per GPU; ~67 GB per GPU left for
  KV/activations/solvers. MLA-style latent KV is small (order 45 KB/token in fp8), so the model fits WITHOUT
  weight offload; weight offload would stream ~100+ GB per decode step over PCIe and cut throughput badly.
### Semantic Scholar API (verified with the key)
- Even at ~1.3 s spacing the API returned 429 twice in four calls -> limiter must be >= 3 s spacing with
  exponential backoff on 429 (both measured OK).
- Datasets (release 2026-09-22): `s2orc` (722 gz shards; schema externalIds, content.source.pdfUrls/oaInfo,
  content.text, content.annotations; NO year / field-of-study inside), `s2orc_v2` (329 shards; "replacement
  for s2orc"; title, authors, body.text, bibliography.text, openaccessinfo), `papers` (60 x ~1.5 GB;
  year, s2fieldsofstudy, ...). Shard URLs are pre-signed S3 links from
  `GET /datasets/v1/release/latest/dataset/<name>` (header `x-api-key`).
- `GET /graph/v1/paper/search/bulk?query=...&fieldsOfStudy=Computer Science&year=2022-&openAccessPdf&fields=...`
  works (1000/page with continuation token); `/graph/v1/paper/batch` gives metadata for up to 500 ids per call.

## Serving optimization pass (2026-09-27 03:49-04:47, serving/NOTES.md "Optimization pass")
Final configuration (serve_*.sh): GLM-5.3-NVFP4 TP4+EP, fp8 KV, no offload, + MTP speculative decoding (1 draft token,
~76% acceptance; KV 491k tokens); Qwen3.8-27B-FP8 TP2 on GPUs 0-1 at 0.13 + MTP (2 tokens): +49-60% per-request decode
under load; Qwen3.5-4B data-parallel 2 on GPUs 2-3 at 0.12 + MTP (2 tokens): +23%/+39% at 8/32 concurrent and 2x KV.
Sampling unchanged. Measured: solvers run at ~40-50% of idle speed while GLM is busy (shared GPUs); GLM aggregate
~450-500 output tok/s at 8-32 concurrent, 111 tok/s single stream idle. Risk: GPUs 0-1 have ~4 GB free; if GLM grows,
restart the 27B at 0.12. Judge-effort measurement (low/high/max) recorded but NOT applied: user ruled max effort for all
GLM roles with max_tokens 81,920.
