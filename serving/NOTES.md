# Serving notes (vLLM 0.30.0, torch 2.13 cu130, 4x B200 183 GB)

State on 2026-09-27 02:55 KST: all three servers up, all checks pass. Scripts are in serving/ and logs in logs/serve_*.log.

**Superseded by the optimization pass (2026-09-27 03:49-05:00).** The serve_*.sh scripts now start the optimized
configuration: 4B with DP=2 on GPUs 2,3 + MTP, 27B with MTP at 0.13, and GLM as described under "Final configuration".
The table and the "Exact commands" below describe the original bring-up; see "Optimization pass" at the end of this file.

| server | script | GPUs | port / served name | GPU KV cache (vLLM log) | ready after |
|---|---|---|---|---|---|
| GLM-5.3 (nvidia/GLM-5.3-NVFP4) | serve_glm53.sh | 0,1,2,3 (TP=4 + EP) | 8000 / `glm-5.3` | 28.84 GiB/GPU = **649,152 tokens**; max concurrency at 400,000 tokens/request: **1.62x** | 9 min (02:43:15 -> 02:52:15) |
| Qwen3.8-27B-FP8 (strong) | serve_qwen27b.sh | 0,1 (TP=2) | 8001 / `qwen3.8-27b` | 5.71 GiB/GPU = **173,306 tokens**; at 65,536 tokens/request: **2.64x** | 2.5 min (02:52:30 -> 02:54:58) |
| Qwen3.5-4B (weak) | serve_qwen4b.sh | 2 | 8002 / `qwen3.5-4b` | 11.58 GiB = **359,197 tokens**; at 65,536 tokens/request: **5.48x** | 1.6 min (02:52:30 -> 02:54:08) |

Start order: `serving/serve_glm53.sh`, wait for "Application startup complete" in logs/serve_glm53.log, then
`serving/serve_qwen27b.sh` and `serving/serve_qwen4b.sh` (these two can start together). Check with
`python serving/healthcheck.py [--wait 900]`, `python serving/check_glm53.py`, `python serving/check_qwen.py`.
Stop with `serving/stop_all.sh [glm53|qwen27b|qwen4b]`.

## Changes from docs/IMPLEMENTATION_SPEC.md section 2.7
1. **GLM: `--kv-offloading-size 400` dropped (no CPU KV offload).** The spec's 400 GiB failed (attempt 1): vLLM's
   native offload buffer is a single mmap file in /dev/shm, which is 200 GiB in this container. The coordinator then
   chose 120 GiB (attempt 2). The user then decided no offloading at all, since the model fits in GPU memory, so it
   was restarted without the flag (attempt 3, running). To re-enable later, append e.g. `--kv-offloading-size 32` to
   serve_glm53.sh (the value is the total across the 4 TP ranks and must fit in /dev/shm).
2. `VLLM_ENGINE_READY_TIMEOUT_S=3600` is exported for every server (serving/common.sh). vLLM's default of 600 s
   covers weight load, autotuning and CUDA-graph capture; GLM's first start needed about 19 min.
3. Nothing else changed. The Qwen fractions (0.14 / 0.16) started without complaint; no --trust-remote-code or
   --quantization flags were needed (GLM's ModelOpt mixed FP8/NVFP4 checkpoint is auto-detected as `modelopt_mixed`).

## Exact commands running now
All are launched via serving/common.sh: `cd` to the project, `source env.sh`, activate .venv,
`export VLLM_ENGINE_READY_TIMEOUT_S=3600`, then `setsid nohup <cmd> >> logs/serve_<name>.log 2>&1 < /dev/null &`.
```
env CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve nvidia/GLM-5.3-NVFP4 --served-model-name glm-5.3 --port 8000 \
  --tensor-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.80 \
  --max-model-len 400000 --max-num-seqs 64 --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice
env CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b --port 8001 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.14 --max-model-len 65536 --max-num-seqs 32 \
  --reasoning-parser qwen3 --language-model-only
env CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3.5-4B --served-model-name qwen3.5-4b --port 8002 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.16 --max-model-len 65536 --max-num-seqs 48 \
  --reasoning-parser qwen3 --language-model-only
```
The servers bind 0.0.0.0 (vLLM default, as in the spec) with no API key; clients use 127.0.0.1.

## GPU memory with all three up (nvidia-smi, 02:55, GLM already serving one request)
```
GPU  used / total (MiB)   processes
0    173068 / 183359      GLM Worker_TP0_EP0 148854 + Qwen27B Worker_TP0 24188     free  9564
1    173068 / 183359      GLM Worker_TP1_EP1 148854 + Qwen27B Worker_TP1 24188     free  9564
2    171528 / 183359      GLM Worker_TP2_EP2 148854 + Qwen4B EngineCore 22648      free 11104
3    148868 / 183359      GLM Worker_TP3_EP3 148854                                 free 33764
```
- GLM's budget is 0.80 x 183359 = 146,687 MiB. Right after startup each worker used 145,944 MiB. After the first
  requests it settled at 148,854 MiB, about 2.1 GiB over the nominal fraction (allocator and workspace growth
  outside the profiled run). Leave about 10 GB of slack per GPU for this.
- Qwen27B: budget 24.97 GiB/GPU, actual 23.6 GiB (weights 14.09 GiB/GPU, KV 5.71 GiB).
- Qwen4B: budget 28.54 GiB, actual 22.1 GiB (weights 7.99 GiB, KV 11.58 GiB). vLLM's CUDA-graph memory estimate was
  7.12 GiB but actual use is 0.07 GiB, so about 7 GiB of its fraction is reserved and unused. Possible future tweak,
  not applied: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` for the 4B server gives about +7 GiB of KV at the same fraction.
- Host: /dev/shm 564 KiB used (no offload buffer); host RAM 360 GiB used out of 2.2 TB.

## Verification (outputs from the scripts)
- `python serving/healthcheck.py`: OK for glm-5.3 (max_model_len 400000), qwen3.8-27b (65536), qwen3.5-4b (65536).
- `python serving/check_glm53.py`: all 4 PASS.
  - (a) plain chat: message fields `['content', 'role', 'reasoning']`. **vLLM 0.30.0 returns the reasoning as
    `message.reasoning`, not `reasoning_content`.** Content "391" for 17*23; usage includes
    completion_tokens_details.reasoning_tokens.
  - (b) tool call: finish_reason=`tool_calls`, 1 call `get_weather`, raw arguments `'{"city": "Paris", "unit": "celsius"}'`,
    which json.loads cleanly; content None; id `chatcmpl-tool-...`.
  - (c) round trip (assistant message with tool_calls, then a tool message): finish_reason=stop, final answer uses
    the tool result ("18°C with light rain").
  - (d) `extra_body={"chat_template_kwargs": {"reasoning_effort": "max"}}` accepted, finish_reason=stop.
    The template only knows 'low' and 'high'; any other value, or none, renders "Reasoning Effort: Max",
    so max is also the default.
- `python serving/check_qwen.py`: both PASS with temperature 1.0, top_p 0.95, top_k 20, max_tokens 2048, enable_thinking=true.
  - qwen3.8-27b: finish_reason=stop, usage prompt 74 / completion 155 (reasoning 99), field `reasoning`, 3.0 s.
  - qwen3.5-4b: finish_reason=stop, usage prompt 32 / completion 945 (reasoning 885), field `reasoning`, 6.5 s.
  - Both servers log that generation_config.json sets default sampling to temperature 1.0 / top_k 20 / top_p 0.95
    when a request leaves them unset.

### Important for autodata/llm/client.py (owned by subagent A; not changed here)
- Responses: read `message.reasoning` first (0.30.0 fills only this); `reasoning_content` is not returned.
- **History: send the assistant's reasoning back as `reasoning`, not `reasoning_content`.** Checked with /tokenize and
  /detokenize on GLM: an assistant message with `reasoning` is rendered as `<think>...</think>`. The same text in
  `reasoning_content` is dropped, and the template emits `<think></think>`. GLM-5.3's template keeps the thinking of
  every earlier assistant turn (clear_thinking defaults to false), so with `reasoning_content` the agent silently
  loses its prior reasoning. Spec section 2.2 (`assistant_message` uses reasoning_content) should switch to
  `reasoning` (or send both).

## Kernels, warnings, load times (from the logs)
- GLM-5.3: attention FLASHINFER_MLA_SPARSE (the only candidate), MLA prefill TRTLLM_RAGGED, NVFP4 MoE backend
  FLASHINFER_TRTLLM (candidates: FLASHINFER_CUTEDSL, CUTLASS, MARLIN, ...), FP8 dense layers FlashInferFP8ScaledMMLinearKernel,
  DSA indexer KV block size 64, KV cache fp8_e4m3 in the "standard fp8" layout (vLLM notes that fp8_ds_mla would need
  `--attention-backend FLASHMLA_SPARSE`). All-reduce: FlashInfer (mnnvl) + custom. torch.compile is off
  (compilation mode NONE); CUDA graphs FULL_AND_PIECEWISE up to batch 128; chunked prefill with max_num_batched_tokens=16384.
- GLM load time (attempt 3): weights 52.9 s (432 GiB, Lustre, page cache warm), model load about 55 s
  (106.57 GiB/GPU); init engine 442.5 s, mostly FlashInfer autotuning (02:44:56-02:51:12, 66 configs saved to
  $VLLM_CACHE_ROOT/flashinfer_autotune_cache) plus CUDA-graph capture. Cold first start (attempt 1) also spent about 13 min
  JIT-compiling FlashInfer kernels, now cached in ~/.cache/flashinfer/0.6.18.post1/100a.
- GLM runtime warning (performance only): `[AutoTuner]: No tuned config covers fp8_gemm input_shapes=((1, 8708, 6144), ...);
  falling back to runner=CutlassFp8GemmRunner`. That prefill chunk shape was not autotuned; the result is still correct.
- No other NVFP4/FlashInfer warnings. There is an info-level note that CUDA-graph memory profiling (default since v0.21)
  makes 0.80 behave like 0.7901. Actual CUDA-graph pool: GLM 0.42 GiB vs 1.77 GiB estimated.
- Qwen3.8-27B-FP8: weights 5.0 s, model load 11.5 s, init 87.6 s (torch.compile 57 s). FLASHINFER attention,
  FlashInfer GDN prefill kernel, TRTLLM prefill attention. WARNING: "Auto-disabled DeepGemm for model_type=qwen3_5_text
  on Blackwell (E8M0 scale accuracy degradation); falling back to CUTLASS". This is expected and safe.
- Qwen3.5-4B: weights 3.0 s, model load 7.8 s, init 45.3 s (compile 10.7 s). No warnings beyond "Using TRTLLM prefill attention".

## Operational notes
- **GPU keepalive / scripts/gpu_guard.sh**: the guard polls every 60 s, kills the 4.3 GB/GPU keepalive once any
  `vllm serve` exists, and restarts it when no server runs and the GPUs are idle. vLLM sizes its KV cache as
  requested - (free at baseline - free after profiling). If the keepalive dies after a server's baseline snapshot
  (about 40 s after launch for GLM) but before profiling, that server's KV budget is inflated by about 4.2 GiB/GPU
  (attempt 1: 33.04 GiB vs 28.84 GiB). To start clean, launch GLM when the keepalive is already gone, or right after
  seeing the guard's `sleep 60` child reach about 57 s, so the guard kills the keepalive within about 3 s of the launch.
  When restarting, start the new GLM within the same guard cycle as the stop, so the keepalive is not brought back in between.
- stop_all.sh sends SIGTERM to the server's process group and waits GRACE_S=60 s, then sends SIGKILL. A server still in
  FlashInfer autotuning ignores SIGTERM (attempt 2 needed SIGKILL). After a SIGKILL, check /dev/shm for leftover
  `psm_*` segments (about 160 MB each) that no process maps. It never touches gpu_guard.sh or the keepalive.
- serve_*.sh refuses to start if the port is listening or a `vllm serve ... --port N` process is still loading
  (vLLM binds its port only after the engine is ready). An existing log is kept as logs/serve_<name>.<timestamp>.log.
- A GLM restart now takes about 9 min: weights about 1 min (page cache), autotuning about 6 min, graphs about 1 min.

## Follow-ups asked by the coordinator (not implemented)
### (1) MTP weights and speculative-decoding recipes
| checkpoint | MTP weights in the checkpoint | model-card recipe |
|---|---|---|
| nvidia/GLM-5.3-NVFP4 | yes: `num_nextn_predict_layers: 1`, `model.layers.78.*` = 791 tensors, **18.54 GiB, BF16** (ModelOpt `ignore: model.layers.78*`, so not quantized) | none. The card only covers SGLang (`--tp-size 8 --quantization modelopt_fp4 --tool-call-parser glm47 --reasoning-parser glm45`), with no speculative flags |
| Qwen/Qwen3.8-27B-FP8 | yes: `mtp_num_hidden_layers: 1`, 22 `mtp.*` tensors in mtp.safetensors, 0.44 GiB (FP8 + BF16) | none locally; the card links https://recipes.vllm.ai/Qwen/Qwen3.8-27B |
| Qwen/Qwen3.5-4B | yes: `mtp_num_hidden_layers: 1`, 15 `mtp.*` tensors, 0.22 GiB BF16 | vLLM: `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`; SGLang: `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |

In vLLM 0.30.0 (vllm/config/speculative.py) `glm_moe_dsa` uses the DeepSeek-V3.2 MTP path (model_type -> deepseek_mtp),
and Qwen3.5-family checkpoints map to `qwen3_5_mtp`. Every MTP method name is normalised to `"mtp"`. Untested
candidates: GLM `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`; Qwen3.8-27B the same form as the
Qwen3.5 card (both are Qwen3_5ForConditionalGeneration). Cost for GLM: the BF16 MTP layer is about 18.5 GiB in total,
roughly 4.5-5 GiB per GPU with EP=4. At a fixed 0.80 that comes out of the KV cache (about -100k tokens of the 649k).

### (2) `--data-parallel-size 2` for the 4B solver on GPUs 2,3
It fits. GPU 3 holds only GLM (148,854 MiB) and has 33,764 MiB free. A second 0.16 rank needs 0.16 x 179.06 GiB =
28.65 GiB (29,337 MiB) free at startup and actually uses about 22.6 GB, which is the same footprint as GPU 2 today
(about 11 GB free afterwards). GPU 2 is unchanged. vLLM checks free memory on each rank's GPU, so either way works:
one server with `CUDA_VISIBLE_DEVICES=2,3 --data-parallel-size 2`, or a second server on GPU 3 port 8003.
Keep the ~10 GB/GPU slack for GLM's post-startup growth.

## GLM-5.3 start attempts (full record)

### Attempt 1 (02:16:54) -- spec command verbatim -- FAILED at KV-cache init (after ~19 min)
`vllm serve nvidia/GLM-5.3-NVFP4 --served-model-name glm-5.3 --port 8000 --tensor-parallel-size 4 --enable-expert-parallel
--kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.80 --max-model-len 400000 --max-num-seqs 64 --reasoning-parser glm47
--tool-call-parser glm47 --enable-auto-tool-choice --kv-offloading-size 400`  (log: logs/serve_glm53.attempt1.log)

- All flags accepted. Architecture resolved to GlmMoeDsaForCausalLM, quantization auto-detected as `modelopt_mixed`
  (FP8 dense layers 0-2, NVFP4 routed experts, MTP layer 78 unquantized); no --trust-remote-code or --quantization needed.
- Weights: 55 s to read 432 GiB from Lustre, model load 103 s, 106.57 GiB of weights per GPU.
- FlashInfer JIT-compiled the TRT-LLM fused-MoE routing kernel (~11 min, 02:20-02:31) and the TRT-LLM FMHA kernels
  during CUDA-graph capture (~2 min). Both are cached in ~/.cache/flashinfer/0.6.18.post1/100a, so later starts skip this.
- Available KV cache memory 33.04 GiB/GPU; `GPU KV cache size: 743,680 tokens`, `Maximum concurrency for 400,000 tokens
  per request: 1.86x`. This is inflated by about 4.2 GiB/GPU: the GPU keepalive (4.3 GB/GPU) was running at the
  worker's baseline memory snapshot and was killed by scripts/gpu_guard.sh before profiling (see Operational notes).
- Failure: `RuntimeError: Insufficient space in /dev/shm: 409594 MiB required, 204800 MiB free`. The native CPU
  offloading backend keeps its whole buffer (total across TP ranks) in one mmap file,
  /dev/shm/vllm_offload_<engine_id>.mmap (vllm/v1/kv_offload/cpu/shared_offload_region.py), and /dev/shm is 200 GiB
  in this container. Host RAM (2.2 TB) is not the limit.
- Fix: --kv-offloading-size 120 (coordinator's choice; leaves ~80 GiB of /dev/shm for NCCL/vLLM IPC and the other servers).

### Attempt 2 (02:39:11) -- `--kv-offloading-size 120` -- stopped on purpose at 02:42 (log: logs/serve_glm53.attempt2.log)
- Launched just before a gpu_guard.sh check, so the guard killed the keepalive 3 s after launch, before the workers'
  baseline memory snapshot.
- Weights reloaded in 61-67 s (page cache warm). JIT kernels came from the cache; it was in FlashInfer autotuning
  (`[AutoTuner]: Tuning fp8_gemm`) when stopped.
- Stopped because of a new user instruction: no CPU KV offloading, since GLM fits in GPU memory without it.
  SIGTERM during autotuning was ignored for 60 s, and stop_all.sh fell back to SIGKILL. That left a 160 MB
  /dev/shm/psm_* segment behind, which was removed after checking nothing mapped it.
- While attempt 2 was loading, a test of launch() (run before the port guard was added) renamed its live log.
  It was renamed straight back; nothing was lost, and the complete attempt-2 output is in logs/serve_glm53.attempt2.log.

### Attempt 3 (02:43:15) -- spec command minus `--kv-offloading-size` -- UP at 02:52:15 (log: logs/serve_glm53.log)
- Started within the same guard cycle as the stop of attempt 2; the keepalive was not running at the baseline snapshot.
- Weights 52.9 s, model load 51-57 s, init engine 442.5 s (autotuning 6.3 min).
- Available KV cache memory 28.84 GiB/GPU, `GPU KV cache size: 649,152 tokens`, `Maximum concurrency for 400,000 tokens
  per request: 1.62x`. All four check_glm53.py checks pass.

## Optimization pass (docs/tasks/E_serving_optimization.md)

### Benchmark: serving/bench/bench.py
Usage and metric definitions are in serving/bench/README.md; results are in serving/bench/results/*.json and logs in logs/bench_*.log.
It runs a closed loop of C concurrent chat requests per server with the pipeline's sampling (configs/cs_default.yaml) and
realistic prompts. GLM gets 30k-60k tokens (mean ~44k): smoke-corpus papers concatenated, the target paper first, plus a
question-writing instruction, with thinking at max effort and **max_tokens 2048**. The Qwen servers get 1k-2k tokens in
the solver format (paper opening as context plus a reasoning question), with thinking and **max_tokens 4096**. Every
prompt starts with a unique tag, so every request is a cold prefill. Reported per level: requests/s, aggregate output
and prompt tokens/s, p50/p95 latency, server-side /metrics deltas (TTFT, ITL, preemptions, prefix-cache hits,
spec-decode acceptance), running/waiting/KV usage, and 3 sample outputs.

### Phase 1: light baseline (03:09-03:12, concurrency 4, 4 requests per server; baseline configs above)
Run while the harness smoke test (runs/smoke_one) was using the servers. GLM served 3-7 requests in total, and
6.2k of the tokens it generated during the run were the smoke test's. These numbers are a sanity check, not an
idle-server baseline. File: serving/bench/results/baseline_light.json.

| server | C | ok | wall s | out tok/s | prompt tok/s | p50 / p95 latency s | out tokens/req | per-request tok/s | server TTFT / ITL |
|---|---|---|---|---|---|---|---|---|---|
| glm-5.3 | 4 | 4/4 | 31.1 | 263.6 | 5,339 | 31.0 / 31.1 | 2048 (all `length`) | 66.0 | 4.09 s / 14.0 ms |
| qwen3.8-27b | 4 | 4/4 | 71.1 | 230.3 | 88.6 | 71.1 / 71.1 | 4096 (all `length`) | 57.6 | 0.84 s / 17.2 ms |
| qwen3.5-4b | 4 | 4/4 | 30.4 | 500.1 | 170.5 | 30.4 / 30.4 | 3797 (1 `stop`) | 134.9 | 0.16 s / 7.4 ms |

Spot checks: all outputs are coherent reasoning on the given paper and question (distinct word-4-gram ratio 0.88-0.99).
Given the budgets, every GLM answer and 7 of 8 Qwen answers were still in the thinking phase at max_tokens.

### Facts that shape the candidates
- **Async scheduling is already on for all three servers.** In vLLM 0.30, `async_scheduling=None` resolves to True
  unless something incompatible is configured (vllm/config/vllm.py). MTP is compatible, since MTP methods count as
  EagleModelTypes. So `--async-scheduling` is a no-op here and not a candidate.
- **GLM's max_num_batched_tokens is already 16384** (log: "Chunked prefill is enabled with max_num_batched_tokens=16384"),
  so the batched-tokens candidate is 32768.
- **GLM's KV cache binds before `--max-num-seqs 64`**: 649,152 tokens / (~44k prompt + 2k output) is about 14
  concurrent benchmark requests. The pipeline caps GLM at max_concurrency 48 (configs/cs_default.yaml), so
  `--max-num-seqs 128` cannot raise concurrency at C <= 32. The same applies to the Qwen servers: their client caps are
  24 (27B) and 48 (4B), against max-num-seqs 32 and 48.
- **In the real pipeline the Qwen solvers are KV-bound; the 4096-token benchmark is not.** With 32k-token thinking
  outputs (about 34k tokens per request), the 27B's 173k-token cache holds about 5 concurrent requests and the 4B's
  359k about 10, while 6 papers x 3 attempts want about 18. So KV capacity (DP=2 doubles the 4B's) and per-request
  speed (MTP) matter more than batch-32 throughput.
- The GPUs are shared (GLM on 0-3, 27B on 0-1, 4B on 2). Each server is benchmarked with the others idle, so the
  numbers are upper bounds for the pipeline, where they compete for SMs.
- Penalties are applied inside vLLM's rejection sampler (vllm/v1/sample/rejection_sampler.py), so MTP keeps the
  4B's presence_penalty 1.5 semantics. `qwen3_next_mtp` / `qwen3_5_mtp` are deprecated aliases that vLLM rewrites to
  `mtp`. With a single MTP layer, num_speculative_tokens > 1 reuses that layer. GLM-5.3's config sets
  `index_share_for_mtp_iteration: true`.
- The vLLM recipe for Qwen3.8-27B (https://recipes.vllm.ai/Qwen/Qwen3.8-27B) uses
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. The Qwen3.5-4B model card uses
  `{"method":"qwen3_next_mtp","num_speculative_tokens":2}`.

### Plan (one change at a time; benchmark at C = 1, 8, 32; keep a change only if output tok/s improves at C = 8 and 32 with no errors and sane outputs)
0. Full baseline sweep of all three servers with the current configs.
1. Qwen3.5-4B: (a) MTP n=2; (b) plus `--data-parallel-size 2` on GPUs 2,3 (one endpoint, so the harness keeps one
   base_url); (c) n=3 if n=2 wins clearly.
2. Qwen3.8-27B-FP8: (a) MTP n=2, then n=3 (recipe). `--max-num-seqs` stays 32, because the client cap is 24 and KV
   binds first. `--kv-cache-dtype fp8` would double its KV but changes solver numerics, so it is not applied.
3. GLM-5.3, last: (a) MTP n=1; (b) n=2; (c) `--max-num-batched-tokens 32768` on top of the winner. It keeps
   fp8_e4m3 KV, no weight offload and no KV offload. At a fixed 0.80, the BF16 MTP layer (~4.6 GiB/GPU) comes out of
   the KV cache, so C=32 (KV-bound) is the deciding level.
4. Update serve_*.sh to the winners, restart one at a time (GLM last), verify with healthcheck.py, check_glm53.py and
   check_qwen.py, and record the results below.

### Phase 1 addendum: GLM judge cost vs reasoning effort (03:26-03:33, no restart)
Script serving/bench/judge_effort.py; results in serving/bench/results/judge_effort.json; log in logs/bench_judge_effort.log.
The recorded judge request (judge.messages) from runs/smoke_probe/s2_259138764/eval_attempts/run_001_weak-only/attempt_weak_2.json
was used: system = judge.md, user = context + question + 14-criterion rubric + the 4B's answer, 2,565 prompt tokens.
It was sent 3x at each `chat_template_kwargs.reasoning_effort` of low, high and max, with temperature 1.0, top_p 0.95
and max_tokens 32768. All 9 requests ran concurrently, so they saw the same load; the smoke run was also using GLM.
The verdict was parsed like the harness does (one entry per criterion) and scored as max(0, (earned - penalty) / max_positive).

| effort | completion tokens (3 runs) | latency s | finish | verdict parses | score | agreement with recorded verdict |
|---|---|---|---|---|---|---|
| recorded (max, smoke run, lighter load) | 10,290 | 127.0 | stop | yes | 0.0545 | - |
| low | 812 / 916 / 895 (mean 874) | 27.2 / 30.4 / 29.7 | stop x3 | 3/3 | 0.0545 x3 | 100% (identical 14/14) |
| high | 2,466 / 1,785 / 2,268 (mean 2,173) | 75.4 / 56.5 / 69.8 | stop x3 | 3/3 | 0.109 / 0.0545 / 0.0545 | 97.6% (one run also marked criterion 9) |
| max | 9,052 / 13,973 / 17,104 (mean 13,376) | 255.7 / 332.9 / 362.9 | stop x3 | **2/3** | 0.0545 x2 | 100% for the 2 parsed |

- Cross-effort agreement per criterion (all pairs of parsed runs): low vs max 1.00, low vs high 0.976, high vs max 0.976.
  On this example every effort reaches the same verdict. Max costs about 15x the tokens of low and 11x the latency.
  That is one example; more are needed before changing the judge setting.
- The unparsed max run failed on JSON syntax, not content: its evidence quote contains LaTeX `$\alpha_d/\alpha_z$`
  with single backslashes, which is an invalid JSON escape (`\a`). `parse_judgments` in src/autodata/cs/judge.py
  uses strict json.raw_decode, so it would raise JudgeError and retry the whole judge call (another 10k+ tokens at
  max effort). A lenient pre-pass that doubles backslashes not starting a valid JSON escape
  (`re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', raw)`) would recover such verdicts. That code belongs to the harness owner,
  not serving/.

### Phase 2 (RESTART_ALLOWED at 03:49; time-boxed to 05:00 by the coordinator)
All numbers are aggregate output tokens/s at concurrency 1 / 8 / 32 (bench.py defaults: glm N = 2/8/32,
27b N = 2/8/32, 4b N = 3/16/64). Results are in serving/bench/results/<label>.json and logs in logs/bench_<label>.log
and logs/bench_candidate_<label>.log. Every candidate passed healthcheck.py plus check_qwen.py or check_glm53.py
before its benchmark. **Load conditions differ.** `baseline` (03:49-03:58) and `qwen4b_mtp2` (04:01-04:03) ran on
idle servers. From about 04:05 the coordinator's CoT-baseline workload (2 papers; GLM challenger/QV/judge calls plus
solver calls) kept GLM busy with 2-6 requests. That contends for the SMs of every GPU the Qwen servers share with GLM
and cuts solver speed about 2.1-2.4x. Later candidates are therefore compared with a baseline re-measured under the
same load (`*_bgload`), not with the idle baseline.

**Memory headroom (important for the long run).** GLM's workers grow past their nominal 0.80 budget
(0.80 x 178.35 GiB = 142.7 GiB): 145,944 MiB right after start, 148,854 after the first requests, 150,600 at 03:00 and
152,910 MiB at 03:51. They stayed at 152,912 until the restart at 04:29. The Qwen workers also grow ~2.5-3 GB under
load. Two consequences:
1. The 4B at its old 0.16 no longer starts on GPU 2: "Free memory on device cuda:0 (28.4/178.35 GiB) ... less than
   desired GPU memory utilization (0.16, 28.54 GiB)" (logs/bench_candidate_qwen4b_mtp2.attempt1_nomem.log). Fixed
   with `--gpu-memory-utilization 0.12` + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. The default profiler reserved
   7.12 GiB for CUDA graphs that use 0.07 GiB (1.2 GiB with MTP), so 0.12 without the reservation gives about the same
   KV (10.91 GiB) and the same physical footprint (~25.5 GB/GPU under load).
2. A fresh 27B start at 0.14 now profiles 7.48 GiB of KV per GPU, not 5.71 (the first start was deflated, probably
   because GLM was still growing during its profiling). It used 28,150 MiB/GPU at start, so only ~2.3 GB would be
   free on GPUs 0,1, and the ~2.5 GB growth under load could OOM a GPU that GLM shares. That candidate run was aborted
   after its C=1 level, and the 27B now runs at 0.13.
With the final configuration and GLM at ~153 GB/GPU, free memory under load is about 2.4 GB on GPUs 0,1 and about
4.8 GB on GPUs 2,3. If GLM grows further (longer main-agent contexts than any test so far), GPUs 0,1 are the first
at risk. Watch `nvidia-smi` during the first hours of the long run.

#### Qwen3.5-4B (:8002)
| label | config | C=1 | C=8 | C=32 | notes |
|---|---|---|---|---|---|
| `baseline` (idle) | 1 GPU (2), 0.16 | 336.8 | 2,266.1 | 6,842.9 | ITL 3.0/3.4/4.5 ms; KV 359,197 tokens |
| `qwen4b_mtp2` (idle) | + MTP n=2, 0.12 + estimate off | **478.9 (+42%)** | **2,618.9 (+16%)** | **7,428.7 (+9%)** | accept length 2.00/1.97/2.00 (pos1 0.62, pos2 0.38); KV 287,767 tokens (the MTP layer + 3 padding layers raise per-token KV ~17%); 19 preemptions at C=32 |
| `qwen4b_mtp2_dp2` (bg load) | + DP=2 on GPUs 2,3 | 224.9 | 1,510.2 | 4,863.8 | 2 x 287,767 KV tokens; accept length 2.03/1.99/2.00; 0 errors |

- MTP n=2 is kept: better at every level, same output quality (distinct-4-gram 0.92-0.99, coherent reasoning).
- DP=2 was measured after the background load started, so no clean single-GPU run under the same load exists. At
  C=1 only one DP rank is used, so the C=1 drop from 478.9 to 224.9 is the load factor (x0.47). The same factor gives
  ~1,230 (C=8) and ~3,490 (C=32) for single-GPU MTP n=2 under this load; the 27B lost a uniform 2.4x at all levels.
  DP=2 measured 1,510 (+23%) and 4,864 (+39%), so **DP=2 is kept**. It also doubles the KV (575k tokens), which is
  what matters for 32k-token thinking outputs: ~17 concurrent full-length requests vs ~8 on one GPU.
- Caveat: GPU 3 was GLM-only before and now also runs a 4B rank, so GLM's TP rank 3 sees the same contention as
  ranks 0-2. That cost could not be measured in the time box.

#### Qwen3.8-27B-FP8 (:8001)
| label | config | C=1 | C=8 | C=32 | notes |
|---|---|---|---|---|---|
| `baseline` (idle) | TP=2, 0.14 | 150.7 | 1,020.6 | 1,949.1 | KV 173,306 tokens; C=32 KV-bound (38 preemptions) |
| `baseline_27b_bgload` (bg load) | same | 62.1 | 429.7 | 802.8 | same server re-measured under the background load |
| `qwen27b_mtp2_frac014_partial` (bg load) | + MTP n=2, 0.14 | 101.5 | - | - | KV 200,791 tokens, 28,150 MiB/GPU at start: aborted before C=8 (OOM risk, see above) |
| `qwen27b_mtp2_f013` (bg load) | + MTP n=2, **0.13** | **99.5 (+60%)** | **642.4 (+49%)** | 779.6 (-3%) | KV 153,382 tokens; accept length 1.94/1.98/1.98 (pos1 0.61, pos2 0.37); 0 errors |

- **MTP n=2 at 0.13 is kept, although C=32 is -3%.** That level is KV-bound (12.2 running on average vs 20.0 for
  the baseline, 58 vs 38 preemptions), because 0.13 with MTP holds 153k tokens vs the old 173k. The old 0.14 cannot
  be restarted safely any more, since it would now take ~30.5 GB/GPU under load. In the pipeline the 27B serves a few
  long requests (32k-token thinking, client cap 24, KV fits ~4.5 x 34k), where per-request decode speed (+49-60%)
  dominates. Outputs look normal (distinct-4-gram 0.87-0.97 vs 0.87-0.93 at baseline, same telegraphic thinking style).
- Not tried (time box): n=3 (the vLLM recipe's value), --max-num-seqs (client cap 24 < 32, and KV binds first), and
  --kv-cache-dtype fp8 (it would double KV but changes solver numerics).

#### GLM-5.3 (:8000)
| label | config | C=1 | C=8 | C=32 | notes |
|---|---|---|---|---|---|
| `baseline` (idle) | spec command (TP4+EP, fp8_e4m3 KV, 0.80, max-num-seqs 64, mnbt 16384) | 111.4 | 450.3 | 489.9 | KV 649,152 tokens; C=8 p50 36.3 s; C=32 KV-bound (max 15 running, up to 29 waiting, KV 99.9%, TTFT 45.7 s) |
| `glm_mtp1` (bg load) | + `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` | - | **454.0** | **501.0** | KV 22.15 GiB/GPU = 491,263 tokens (-24%; the BF16 MTP layer takes 4.92 GiB/GPU); acceptance 0.764 (mean length 1.76); C=8 p50 35.0 s |

- **MTP n=1 is kept.** It beats the idle baseline at C=8 (+0.8%) and C=32 (+2.3%) while carrying the background
  CoT workload: 4,081 and 16,134 extra tokens generated during those levels. Counting those, the server produced
  ~569 and ~625 tok/s (+26% and +28%). C=32 stays KV-bound (at most 12 running, KV 99.6%, 1 preemption) with 24% less
  KV, and still comes out ahead. Outputs are normal (distinct-4-gram 0.88-0.99, coherent paper-grounded reasoning).
  check_glm53.py passes 4/4: plain chat, tool call, tool round trip, reasoning_effort=max.
- Startup took 12 min (04:30:29 -> 04:42:27). Weight loading took 337 s on GPUs 0,1 vs 64 s on 2,3, under the
  background load. New FlashInfer autotuning ran for this config (87 configs, including trtllm_bf16_moe for the MTP
  layer, whose cubins were fetched on first use). It logged `[Autotuner]: OOM detected, falling back to default
  tactic` once: the memory is tight. The next start with the same flags reuses $VLLM_CACHE_ROOT/flashinfer_autotune_cache/.../5862eedc...
- The old instance needed SIGKILL after 60 s, because in-flight background requests delayed shutdown.
  `/dev/shm` now holds ~2.7 GB of `psm_*` segments (some orphaned by the SIGKILLs); /dev/shm is 200 GiB, so they
  are harmless.
- GLM memory: 151,128-151,132 MiB/GPU from start through the C=32 run. The old instance started at 145,944 and grew
  to 152,910. The MTP layer and a larger CUDA-graph pool (2.66 + 1.78 GiB captured) are part of that.
- Not tested (time box, coordinator's priorities): MTP n=2; --max-num-batched-tokens 32768 (the workload is
  decode-dominated, and larger prefill chunks would add transient memory where there is no headroom); --max-num-seqs
  128 (KV binds at ~12-15 long requests and the client cap is 48); --async-scheduling (already on by default).

### Final configuration (running since 04:42; all checks pass)
| server | GPUs | command change vs bring-up | KV cache | GPU memory per worker at 04:46 |
|---|---|---|---|---|
| GLM-5.3 :8000 | 0,1,2,3 TP4+EP | + `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` | 22.15 GiB/GPU = 491,263 tokens (1.23x at 400k) | 151,130 MiB |
| Qwen3.8-27B-FP8 :8001 | 0,1 TP2 | `--gpu-memory-utilization 0.13` (was 0.14) + `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` | 5.71 GiB/GPU = 153,382 tokens | 28,086 MiB |
| Qwen3.5-4B :8002 | **2,3 DP2** | `--data-parallel-size 2`, `--gpu-memory-utilization 0.12` (was 0.16) with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, + `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` | 10.91 GiB = 287,767 tokens per rank (575,534 total) | 25,604 / 25,386 MiB |

- serving/serve_glm53.sh, serve_qwen27b.sh and serve_qwen4b.sh start exactly these commands (checked by dry-running
  `launch`). Start order is unchanged: GLM first, then the Qwen servers. stop_all.sh works unchanged; for DP the
  API server leads the process group of both engines.
- Sampling is untouched: the same checkpoints and client-side sampling. Speculative decoding uses rejection sampling,
  so output distributions are preserved. GLM KV stays fp8_e4m3. No weight or KV offload.
- GPU memory at 04:46: GPUs 0,1 179,240 / 183,359 MiB (4.1 GB free), GPU 2 176,758 (6.6 GB free), GPU 3 176,540
  (6.8 GB free). GPUs 0,1 have the least headroom. If GLM grows the way the old instance did (+7 GB after start),
  the first fix is `serving/serve_qwen27b.sh --gpu-memory-utilization 0.12`, a restart of the 27B only.
- For the pipeline: whenever GLM is busy, the solvers run at ~40-50% of their idle speed (shared GPUs; 27B 150.7 ->
  62.1 tok/s single-stream, 4B 479 -> 225).
- Judge: the smoke run ended at 03:48 on `JUDGE_ERROR ... judge output contains no JSON object`. The invalid-`\escape`
  failure seen in the judge-effort test is a likely cause; see the Phase 1 addendum.
