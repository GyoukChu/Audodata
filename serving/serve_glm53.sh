#!/usr/bin/env bash
# GLM-5.3 NVFP4 (main agent, challenger, quality verifier, judge) on all 4 B200s, port 8000.
# Command from docs/IMPLEMENTATION_SPEC.md section 2.7. Start this BEFORE the Qwen servers (they take the
# memory GLM leaves free). Returns immediately; loading takes ~10-40 min, watch logs/serve_glm53.log.
# Deviation from the spec: no --kv-offloading-size (user decision: the model fits in GPU memory, so no CPU KV
# offload). The spec's 400 GiB also cannot work here: the native offload buffer is one mmap file in /dev/shm,
# which is 200 GiB in this container (see serving/NOTES.md, attempt 1).
# Optimization pass 2026-09-27 (serving/NOTES.md, "Optimization pass"): MTP speculative decoding with the checkpoint's
# own BF16 MTP layer (layer 78), num_speculative_tokens 1. Sampling is unchanged (rejection sampling). At the fixed
# 0.80 fraction the MTP layer's weights come out of the KV cache (see NOTES.md for the KV size).
# Extra vLLM arguments are appended: serving/serve_glm53.sh --some-flag value
source "$(dirname "$(readlink -f "$0")")/common.sh"

launch glm53 8000 env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve nvidia/GLM-5.3-NVFP4 \
    --served-model-name glm-5.3 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 400000 \
    --max-num-seqs 64 \
    --reasoning-parser glm47 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    "$@"
