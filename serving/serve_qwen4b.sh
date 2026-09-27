#!/usr/bin/env bash
# Weak solver: Qwen3.5-4B on GPUs 2,3 (data parallel: one engine per GPU behind one endpoint), port 8002,
# co-located with GLM-5.3. Start only AFTER GLM-5.3 is up.
# Optimization pass 2026-09-27 (serving/NOTES.md, "Optimization pass"), changes from docs/IMPLEMENTATION_SPEC.md 2.7:
#   - --data-parallel-size 2 over GPUs 2,3: twice the KV cache and decode capacity, still one base_url.
#   - MTP speculative decoding, num_speculative_tokens 2 (Qwen3.5-4B model card; `qwen3_next_mtp` is an alias of `mtp`).
#     vLLM applies temperature/top-p/top-k/presence penalty inside its rejection sampler, so sampling is unchanged.
#   - --gpu-memory-utilization 0.12 with VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0. The default profiler reserved
#     7.1 GiB per GPU for CUDA graphs that actually use ~0.1-1.2 GiB. The old 0.16 no longer starts next to GLM, whose
#     workers grow to ~153 GB/GPU. 0.12 without that reservation gives about the same KV per GPU (10.9 GiB with MTP,
#     287,767 tokens per rank) with the same physical footprint (~25.5 GB per GPU).
# Extra vLLM arguments are appended. QWEN4B_GPUS overrides the GPU list; QWEN4B_DP must then match its length,
# e.g. QWEN4B_GPUS=2 QWEN4B_DP=1 serving/serve_qwen4b.sh
source "$(dirname "$(readlink -f "$0")")/common.sh"

launch qwen4b 8002 env CUDA_VISIBLE_DEVICES="${QWEN4B_GPUS:-2,3}" VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  vllm serve Qwen/Qwen3.5-4B \
    --served-model-name qwen3.5-4b \
    --port 8002 \
    --tensor-parallel-size 1 \
    --data-parallel-size "${QWEN4B_DP:-2}" \
    --gpu-memory-utilization 0.12 \
    --max-model-len 65536 \
    --max-num-seqs 48 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    "$@"
