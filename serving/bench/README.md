# serving/bench: throughput benchmark for the three vLLM servers

`bench.py` is an asyncio client (openai python client) that measures what the Agentic Self-Instruct pipeline needs
from each server: aggregate token throughput and request latency under concurrency, with realistic prompts and the
pipeline's own sampling settings. It is used for the serving optimization pass (docs/tasks/E_serving_optimization.md);
every configuration and result is recorded in serving/NOTES.md, section "Optimization pass".

```bash
cd "$(git rev-parse --show-toplevel)" && source env.sh && source .venv/bin/activate
python serving/bench/bench.py --servers glm,27b,4b --levels 1,8,32 --label baseline      # full sweep, ~12-15 min
python serving/bench/bench.py --servers 4b --levels 1,8,32 --label qwen4b_mtp2 --notes "MTP n=2"
python serving/bench/bench.py --servers glm --levels 4 --requests 4 --out serving/bench/results/x.json
python serving/bench/bench.py --servers glm --prefix-check          # is the prefix cache on and hitting?
```
Run it while the pipeline is idle: the numbers are only meaningful when the benchmark is the only traffic. The
results record other traffic as `external_generation_tokens` (server-side generated tokens minus the benchmark's own).
All three servers share GPUs (GLM on 0-3, 27B on 0-1, 4B on 2), so a benchmark of one server runs with the other
two idle. That is an upper bound for the pipeline, where they compete for the same SMs.

## What it sends

| server (`--servers`) | endpoint | prompt | sampling (mirrors configs/cs_default.yaml) |
|---|---|---|---|
| `glm` | :8000 `glm-5.3` | 30k-60k tokens (mean ~44k): papers from `--corpus` (default data/corpus/cs2022_smoke.jsonl) in the `paper_text()` layout, the target paper first and further papers as "related papers" until the drawn length is reached, then a short instruction to write one research question and a reference answer | thinking on (`reasoning_effort: max`, the model default), temperature 1.0, top_p 0.95, **max_tokens 2048** |
| `27b` | :8001 `qwen3.8-27b` | 1k-2k tokens in the solver format (`prompts/cs/solver_user.md`): `Context:` = the opening of a paper (title, abstract, introduction), `Question:` = one of four reasoning questions | `enable_thinking: true`, temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, **max_tokens 4096** |
| `4b` | :8002 `qwen3.5-4b` | as for 27b | as for 27b but presence_penalty 1.5 (the pipeline's 4B setting) |

- The smoke papers are only 1.7k-18k tokens each (median 5.6k), so one GLM prompt concatenates about 7-8 papers.
- Prompts are drawn with `--seed` (default 0). The same seed, server and request count give the same prompts, so A/B
  runs compare like with like.
- Every prompt starts with a unique tag (`[benchmark request <run>-c<C>-<i>]`). vLLM chains its prefix-cache block
  hashes from the first token, so nothing is served from the prefix cache: every request is a cold prefill. That is
  the prefill-heavy end of the real workload, where the main agent re-sends a growing conversation and mostly hits
  the cache. `--allow-prefix-cache` drops the tag. `--prefix-check` sends one fresh prompt twice with max_tokens 1
  and reports latency and prefix-cache hits for the cold and the repeated call.
- Token counts use the served model's tokenizer from the HF cache (offline); `usage` in the responses is what is
  reported.

## How it measures

For each server and concurrency level C: a closed loop of C workers, each sending its next request as soon as the
previous one returns, until N requests are done. An untimed warm-up request precedes each server's first level.
N per level defaults to glm {1: 2, 8: 8, 32: 32}, 27b {1: 2, 8: 8, 32: 32} and 4b {1: 3, 8: 16, 32: 64},
otherwise max(4, 2C). A full 1/8/32 sweep of all three servers takes about 12-15 minutes, and each level stays under
about 3 minutes. Outputs almost always run to max_tokens, so one or two waves of C requests are representative.
GLM at C=32 is KV-bound (about 14 requests of ~46k tokens fit), so its 32 requests run in about 3 waves. Override
the counts with `--requests N` or `--requests 1:3,8:16`. `--max-seconds` (default 420) is a hard cap per level; requests still
running at the cap are cancelled and reported as `n_cancelled`.

Per level, in the results JSON and the printed table:
- `wall_s`: first dispatch to last completion. `req_per_s`, `output_tok_per_s`, `prompt_tok_per_s` = the sums over
  successful requests of 1, `usage.completion_tokens` and `usage.prompt_tokens`, each divided by `wall_s`.
  Output tokens include the reasoning tokens.
- `latency_s` {p50, p95, mean, min, max}: end-to-end per request, client side (includes server queueing).
  `per_request_output_tok_per_s`: completion tokens / latency per request.
- `finish_reasons`, `errors` (first 5 distinct), `empty_outputs`, and `distinct4` {mean, min}: the share of distinct
  word 4-grams in reasoning + answer. Normal text scores about 0.9 or more; degenerate repetition scores low.
- `server`: deltas of the server's /metrics counters over the level: prompt/generation tokens, prefix-cache
  queries/hits, cached prompt tokens, preemptions, server-side means of TTFT, inter-token latency, queue, prefill and
  decode time, and with speculative decoding `spec_decode` {num_drafts, num_draft_tokens, num_accepted_tokens,
  acceptance_rate, mean_acceptance_length, accepted_rate_per_pos}. Counters are summed over engines, so DP=2 works.
- `load_before` and `load_during` {max, mean}: running/waiting requests and KV-cache usage, sampled every 2 s.
  A `waiting` > 0 at C=32 means the server's KV cache or `--max-num-seqs` limited concurrency.
- `requests`: one record per request (start/end offsets, tokens, finish reason, error). `samples`: the first 3
  successful outputs (reasoning head and tail, answer head) for spot checks.
- `server_args`: the running `vllm serve` arguments for that port (from `ps`), plus GPU memory at the start.

`--stream` switches to streaming and adds client-side `ttft_s` and `decode_tok_per_s_per_request`. The openai client
parses every chunk, so use it only at low concurrency. The server-side `mean_ttft_s` works in both modes.

The table printed at the end has one row per (server, C). Results are written after every level, so an interrupted
run keeps its finished levels. The default output is `serving/bench/results/<label>_<time>.json`; set it with `--out`.

## Files
- `bench.py`: the benchmark.
- `candidate.sh NAME LABEL [vllm args]`: restart one server with extra flags (env `QWEN4B_GPUS`, `LEVELS`, `NO_BENCH`
  pass through), wait until ready, run healthcheck.py and check_glm53.py / check_qwen.py, then benchmark it into
  `results/LABEL.json`. The previous server log is rotated by serving/common.sh.
- `wait_ready.sh NAME [TIMEOUT_S]`: wait for `Application startup complete` plus /v1/models (exit 1 if the process
  dies first, 2 on timeout), then print the KV-cache lines of the log.
- `judge_effort.py`: GLM judge cost vs `reasoning_effort` (low/high/max) on one recorded judge request; results in
  `results/judge_effort.json`.
- `results/*.json`: one file per run. `baseline_light.json` is the Phase-1 run at concurrency 4 while the harness
  smoke test was running (not an idle-server measurement). Logs of the runs are in `logs/bench_*.log`.
