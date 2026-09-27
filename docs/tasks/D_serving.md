Bring up the three vLLM servers for the Autodata reproduction on this 4x B200 node and verify them. Project:
<repo> (`cd` there, `source env.sh`, `source .venv/bin/activate`; vLLM 0.30.0 +
torch 2.13 cu130 are installed; weights are already in the HF cache: nvidia/GLM-5.3-NVFP4 (433 GB, GlmMoeDsaForCausalLM, ModelOpt
NVFP4), Qwen/Qwen3.8-27B-FP8, Qwen/Qwen3.5-4B). Read docs/IMPLEMENTATION_SPEC.md section 2.7 for the target layout and commands.
Nothing else is using the GPUs. You own ONLY the serving/ directory and logs/serve_*.log; do not modify src/, prompts/, configs/.

Do, in order:
1. serving/serve_glm53.sh — start GLM-5.3 with the spec's command (TP=4, expert parallel, fp8 KV, gpu-memory-utilization 0.80,
   max-model-len 400000, max-num-seqs 64, reasoning-parser glm47, tool-call-parser glm47, enable-auto-tool-choice, KV offloading
   --kv-offloading-size 400 (CPU RAM; the host has 2.2 TB). Run it detached (setsid nohup ... > logs/serve_glm53.log 2>&1 &) and wait for
   "Application startup complete" / GET :8000/v1/models. Loading 433 GB takes a while; poll the log every ~60 s (use a Monitor or a
   sleep loop, up to ~40 min). If a flag is unsupported or the model fails to load, read the traceback, fix the flag set (e.g. drop
   --kv-offloading-size, add --trust-remote-code, lower max-model-len, try --quantization modelopt_fp4 or the Inferact/GLM-5.3-NVFP4
   checkpoint which the vLLM recipe uses) and retry; record every attempt in serving/NOTES.md. Do NOT use weight offload flags
   (--cpu-offload-gb / --offload-*); KV offload is wanted but optional.
2. Verify GLM: (a) plain chat completion with the OpenAI python client (model "glm-5.3"), print the reasoning field name that vLLM
   returns (reasoning_content or reasoning) and the content; (b) a tool-calling request with tools=[one function schema] and a user
   message that forces a call -> must return message.tool_calls with parsed JSON arguments; (c) a two-turn tool round-trip (append the
   assistant message incl. tool_calls and a tool message, get a final answer); (d) pass extra_body={"chat_template_kwargs":
   {"reasoning_effort":"max"}} and confirm no error. Save these checks as serving/check_glm53.py.
3. serving/serve_qwen27b.sh (CUDA_VISIBLE_DEVICES=0,1, port 8001, TP=2, gpu-memory-utilization 0.14, max-model-len 65536,
   --reasoning-parser qwen3 --language-model-only, served name qwen3.8-27b) and serving/serve_qwen4b.sh (CUDA_VISIBLE_DEVICES=2, port 8002,
   gpu-memory-utilization 0.16, served name qwen3.5-4b). Start them AFTER GLM is up (they share GPUs). If vLLM refuses because the
   requested memory is not free, lower the fraction slightly (GLM must keep >= 0.78). Verify each with serving/check_qwen.py: a request
   with extra_body={"chat_template_kwargs":{"enable_thinking":true}}, temperature 1.0, top_p 0.95, extra_body top_k 20, max_tokens 2048,
   must return reasoning (reasoning_content/reasoning) and content; print finish_reason and usage.
4. From the vLLM logs record for each server: the KV cache size in tokens ("GPU KV cache size: N tokens" / "Maximum concurrency for X tokens
   per request"), model load time, and any warnings about NVFP4/FlashInfer kernels; put them in serving/NOTES.md together with
   nvidia-smi output after all three are up.
5. serving/healthcheck.py: checks all three /v1/models and prints OK/FAIL; serving/stop_all.sh (kills the three servers by pattern).
Leave all three servers RUNNING at the end. Final report: exact working commands, memory usage per GPU, KV capacities, the reasoning
field name, tool-call verification result, and anything that had to be changed from the spec.
