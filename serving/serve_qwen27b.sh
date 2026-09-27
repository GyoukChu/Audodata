#!/usr/bin/env bash
# Strong solver: Qwen3.8-27B-FP8 on GPUs 0,1 (TP=2), port 8001, co-located with GLM-5.3.
# Start only AFTER GLM-5.3 is up: vLLM sizes its budget as a fraction of total GPU memory and refuses to start if
# that much is not free.
# Optimization pass 2026-09-27 (serving/NOTES.md, "Optimization pass"), changes from docs/IMPLEMENTATION_SPEC.md 2.7:
#   - MTP speculative decoding, num_speculative_tokens 2 (`qwen3_next_mtp` is an alias of `mtp`). vLLM applies
#     temperature/top-p/top-k/penalties inside its rejection sampler, so sampling is unchanged. Under the same
#     background load: +60% output tok/s at 1 request, +49% at 8, -3% at 32 (KV-bound).
#   - --gpu-memory-utilization 0.14 -> 0.13. GLM's workers now hold ~153 GB/GPU. A fresh 0.14 start profiles ~7.5 GiB
#     of KV and settles at ~30.5 GB/GPU under load, which leaves no headroom on GPUs 0,1. At 0.13 with MTP:
#     5.71 GiB KV = 153,382 tokens, ~28.1 GB/GPU under load.
# Extra vLLM arguments are appended: serving/serve_qwen27b.sh --gpu-memory-utilization 0.125
source "$(dirname "$(readlink -f "$0")")/common.sh"

launch qwen27b 8001 env CUDA_VISIBLE_DEVICES=0,1 \
  vllm serve Qwen/Qwen3.8-27B-FP8 \
    --served-model-name qwen3.8-27b \
    --port 8001 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.13 \
    --max-model-len 65536 \
    --max-num-seqs 32 \
    --reasoning-parser qwen3 \
    --language-model-only \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    "$@"
