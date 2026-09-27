# Runbook

```bash
cd <repo>
source env.sh && source .venv/bin/activate        # HF cache, S2/HF keys, venv

# 0. GPU keepalive (cluster deletes containers idle < 1.5% GPU memory for 3 h) — must always run
pgrep -af gpu_guard.sh || (setsid nohup bash scripts/gpu_guard.sh > logs/gpu_guard.log 2>&1 &)   # guard starts/stops scripts/gpu_keepalive.py automatically

# 1. Serving (see serving/NOTES.md): GLM-5.3-NVFP4 :8000, Qwen3.8-27B-FP8 :8001, Qwen3.5-4B :8002
bash serving/serve_glm53.sh
until curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null; do sleep 15; done
bash serving/serve_qwen27b.sh
until curl -fsS --max-time 3 http://127.0.0.1:8001/v1/models >/dev/null; do sleep 15; done
bash serving/serve_qwen4b.sh
bash scripts/wait_servers.sh
python serving/healthcheck.py

# 2. Corpus (S2ORC v2, CS 2022+; strict 1 request / 3 s limiter)
autodata-build-corpus --out data/corpus/cs2022_pilot.jsonl --n-papers 320 --shard-indices 0,1

# 3. Agentic Self-Instruct (CS) — smoke, then pilot (configs/cs_pilot.yaml = cs_default.yaml + concurrency 8)
#    one-shot 24-paper sequence (agentic -> CoT -> stats): setsid nohup bash scripts/launch_smoke24.sh > logs/smoke24.log 2>&1 &
bash scripts/wait_servers.sh
autodata-run-cs --config configs/cs_default.yaml --corpus data/corpus/cs2022_smoke.jsonl --limit 3 --concurrency 3 \
    --workdir-root runs/smoke3
bash scripts/wait_servers.sh
autodata-run-cs --config configs/cs_pilot.yaml --corpus data/corpus/cs2022_pilot.jsonl --workdir-root runs/pilot

# 4. CoT Self-Instruct baseline on the same papers
bash scripts/wait_servers.sh
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

Wait for each server before launching the next so their GPU memory profiling happens in sequence. The individual checks
above cover the first two launches; `scripts/wait_servers.sh` checks all three and has a 1800-second timeout.
The pipeline also polls every endpoint it uses before each paper, every 15 seconds for up to 1800 seconds
(`autodata-run-cs --health-wait-s` overrides this deadline), logging any wait and proceeding if the deadline expires.

`<workdir_root>/cohort.json` records the run name, config fingerprint, hashes of every prompt Markdown file, source corpus
path and SHA-1 when supplied, requested paper IDs, creation time, and harness version. Later selections extend the paper
list, so resuming a subset does not remove pending papers from the cohort. Both workflows stamp newly written summaries
with `config_fingerprint` and `prompt_hashes_sha1`. Programmatic callers can supply `corpus_path` to the shared driver;
the agentic CLI supplies it from `--corpus`.

Changed configs or prompts require a fresh run root or an explicit `autodata-run-cs --allow-config-mismatch` override.
The override records the old and new provenance in `cohort.json` history; the shared `run_papers` API accepts
`allow_config_mismatch=True` too. Completed papers stay skipped. Accepted papers with unfinished final QV resume only
that verification, retaining their workspace and solver results. Statistics print requested, completed, incomplete,
and pending counts and both acceptance rates: completed papers and the entire requested cohort.
