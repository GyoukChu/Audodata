Serving optimization pass (run AFTER the baseline servers are verified and the harness smoke test passed).
Project: <repo> (`source env.sh && source .venv/bin/activate`).
Goal: maximize end-to-end throughput of the Agentic Self-Instruct pipeline on 4x B200 without changing model behaviour
(same checkpoints, same sampling). You own serving/ (scripts, NOTES.md, bench/) only. Never run weight offload. Keep
scripts/gpu_guard.sh running (it manages the idle keepalive); no keepalive must be resident while servers run.

Workload profile (per paper, per round): main agent ~10 GLM calls with 60-250k-token prefixes (shared prefix = paper text,
prefix caching matters), challenger/QV 2-3 GLM calls each with the full paper (~50k tokens in) and 5-30k tokens of thinking
out, judge 6 GLM calls (~10k in, 1-5k out), solvers: Qwen3.5-4B 3 calls and Qwen3.8-27B-FP8 3 calls per round with thinking
(up to 32k tokens out each). Concurrency: 6-8 papers in flight.

Candidates to benchmark (one change at a time, keep what wins; report tokens/s and p50/p95 latency at concurrency 1, 8, 32):
1. GLM-5.3-NVFP4 (:8000): MTP speculative decoding if the checkpoint contains MTP weights
   (--speculative-config '{"method":"mtp","num_speculative_tokens":1..3}' or the deepseek_mtp method used for the deepseek_v32
   code path); --max-num-seqs 64 vs 128; --max-num-batched-tokens (8192 vs 16384 vs 32768); --enable-prefix-caching (verify
   it is on and hits for repeated paper prefixes); --kv-cache-dtype fp8 vs auto (quality is fixed by the paper only for solvers;
   GLM is the judge/agent, keep fp8 only if outputs stay sane); async scheduling (--async-scheduling); cudagraph settings;
   DSA/sparse-attention backend selection on Blackwell (check log warnings about FlashMLA / sparse attention kernels).
2. Qwen3.5-4B (:8002): --data-parallel-size 2 over GPUs 2,3 (vs one replica), MTP spec-dec (--speculative-config per the Qwen3.5
   model card, e.g. {"method":"mtp","num_speculative_tokens":2}), --max-num-seqs 64-128, --language-model-only, memory fraction.
3. Qwen3.8-27B-FP8 (:8001): MTP spec-dec (model card recipe), TP=2 vs DP=2 (two single-GPU replicas fit? FP8 weights 31 GB),
   --max-num-seqs, memory fraction, --kv-cache-dtype fp8 if KV becomes the bottleneck (record any quality caveat).
Method: serving/bench/bench.py — an asyncio client that fires N concurrent chat requests with realistic prompt lengths
(reuse data/corpus/cs2022_smoke.jsonl papers as prefixes; thinking on; max_tokens 4096 for the benchmark) and reports
throughput/latency; record every configuration and result in serving/NOTES.md, and update serving/serve_*.sh to the winning
configuration. Restart servers one at a time; GLM last. Leave all three servers running with the chosen configuration.
