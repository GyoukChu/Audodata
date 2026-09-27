# Runbook

```bash
cd <repo>
source env.sh && source .venv/bin/activate        # HF cache, S2/HF keys, venv

# 0. GPU keepalive (cluster deletes containers idle < 1.5% GPU memory for 3 h) — must always run
pgrep -af gpu_guard.sh || (setsid nohup bash scripts/gpu_guard.sh > logs/gpu_guard.log 2>&1 &)   # guard starts/stops scripts/gpu_keepalive.py automatically

# 1. Serving (see serving/NOTES.md): GLM-5.3-NVFP4 :8000, Qwen3.8-27B-FP8 :8001, Qwen3.5-4B :8002
bash serving/serve_glm53.sh; bash serving/serve_qwen27b.sh; bash serving/serve_qwen4b.sh; python serving/healthcheck.py

# 2. Corpus (S2ORC v2, CS 2022+; strict 1 request / 3 s limiter)
autodata-build-corpus --out data/corpus/cs2022_pilot.jsonl --n-papers 320 --shard-indices 0,1

# 3. Agentic Self-Instruct (CS) — smoke, then pilot (configs/cs_pilot.yaml = cs_default.yaml + concurrency 8)
#    one-shot 24-paper sequence (agentic -> CoT -> stats): setsid nohup bash scripts/launch_smoke24.sh > logs/smoke24.log 2>&1 &
autodata-run-cs --config configs/cs_default.yaml --corpus data/corpus/cs2022_smoke.jsonl --limit 3 --concurrency 3 \
    --workdir-root runs/smoke3
autodata-run-cs --config configs/cs_pilot.yaml --corpus data/corpus/cs2022_pilot.jsonl --workdir-root runs/pilot

# 4. CoT Self-Instruct baseline on the same papers
autodata-cot-baseline --config configs/cs_pilot.yaml --corpus data/corpus/cs2022_pilot.jsonl --workdir-root runs/pilot_cot

# 5. Table-1 statistics
autodata-stats --run-root runs/pilot --cot-root runs/pilot_cot --json runs/pilot_stats.json

# Tests (no GPU/network)
python -m pytest -q
```

Per-paper artifacts live in `<workdir_root>/<paper_id>/`: `paper.txt`, `eval_input.json`, `eval_attempts/run_NNN_<mode>/`
(per-attempt solver + judge outputs, report.json/txt), `output/result.json` (written by the agent), `trajectory/*.jsonl`
(every LLM message of the main agent and each subagent), `harness_summary.json` (rounds, harness-verified acceptance,
final QV, usage). `summary.jsonl` aggregates papers; `autodata-stats` computes the Table-1 metrics.
