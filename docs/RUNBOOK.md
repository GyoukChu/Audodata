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
both CLIs supply it from `--corpus`.

Changed configs, prompts, corpus SHA-1, or known corpus paths require a fresh run root or an explicit
`--allow-config-mismatch` override (available on both workflows).
The override records the old and new provenance in `cohort.json` history; the shared `run_papers` API accepts
`allow_config_mismatch=True` too. Completed papers stay skipped. Accepted papers with unfinished final QV resume only
that verification, retaining their workspace and solver results. Statistics print requested, completed, incomplete,
and pending counts and both acceptance rates: completed papers and the entire requested cohort.

On resume, each existing `paper.txt` is compared with the current corpus text as the workspace would render it,
including truncation. A difference logs `resume: paper text changed, rerunning`, archives the workspace, and starts a
fresh attempt. Imported summary fingerprints absent from both the current cohort and its history are recorded once in
`cohort.json["imports"]`, with harness version, available prompt fingerprint, and accumulating paper IDs.

The CoT baseline repairs a frozen candidate when its summary's question, context, and rubric match `eval_input.json`.
It keeps the workspace and skips the challenger, rerunning only an incomplete quality verifier, a missing/errored weak
report (`--weak-only`), or a missing/errored strong report (`--strong-only --force-strong`). The evaluator can reuse the
earlier weak run for that question hash. The summary lists attempted stages in `repaired`; incomplete QV is an error,
and completion still requires no errors and both averages. A mismatched candidate requires a fresh attempt. Both repair
paths preserve original `config_fingerprint` and `prompt_hashes_sha1`, recording current values separately in
`repair_config_fingerprint`, `repair_prompt_hashes_sha1`, and `repaired_at`, including failed repairs. An accepted agentic
candidate with completed final QV stays terminal even if its old agent stop reason or historical errors remain.

Corpus building validates every existing record against the active year, field, abstract/body length, release, and
shard filters before counting it toward completion. Build parameters are stored in `<output>.meta.json`; filter or
release/shard recipe changes are refused on resume. Increasing `--n-papers` with the same recipe is supported.
`autodata-build-corpus --allow-filter-mismatch` logs mismatches and keeps existing records, recording earlier parameters
in the metadata history. It does not refilter or discard them.

Draining a runner: create an empty `DRAIN` file in the run root (`touch runs/pilot/DRAIN`). Papers already in flight
finish normally; papers not yet started are skipped with `skipped: drained` (recorded only in `summary.jsonl`, never as
a per-paper summary) and the runner exits when the in-flight papers are done. Delete the file before the next launch.
Use this before restarting a runner under new code so that no in-flight work is lost.

Retiring a runner that predates the DRAIN file (one-off, used for the round-4 upgrade of the pilot):
`python scripts/hold_paper_locks.py --root runs/pilot --root runs/pilot_cot` holds the lock of every queued paper, so the
old runner finishes its in-flight papers and skips the rest ("skipped: workspace is locked") at its next free slot.
After that skip burst appears in the old runner's log, `touch runs/.release_locks` releases the locks and new runners
can start on the same roots. Never release before the burst: the old runner would pick up queued papers again.

Re-adjudication after withdrawn checks: `python scripts/readjudicate_refused.py --config configs/cs_pilot.yaml
--root runs/pilot [--dry-run]` rewrites (with a backup) the summaries of papers whose acceptance was refused only by
checks withdrawn in round 4 (blocking in-loop QV binding; version-skew hash/provenance problems; weight range), after an
exact recomputation from the recorded judgments. The next resume runs the end-of-loop quality verifier on the frozen
candidate. Papers held by a runner are skipped.

Upgrading code under a running pipeline: the evaluator runs as a fresh subprocess and imports the CURRENT working tree,
while runner processes keep the code they started with. Any change to the evaluator report format must stay readable by
the running harness (e.g. `question_hash` keeps the legacy identity; the canonical identity is `question_hash_canonical`).

The Semantic Scholar limiter enforces a 3.0-second minimum between requests across all endpoints, including shard
downloads; lower finite `S2_MIN_INTERVAL` values are clamped to that floor, and negative/non-finite values are rejected.
The shared file gate honours backoff deadlines through 900 seconds, rechecking them in increments of at most 30 seconds.
