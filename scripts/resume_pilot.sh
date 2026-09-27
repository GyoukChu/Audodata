#!/usr/bin/env bash
# Resume the pilot after a container restart (all processes gone; repository, run roots and caches on shared storage).
# Starts the GPU guard, the three vLLM servers in sequence, then the agentic (8 slots) and CoT (2 slots) runners.
# Completed papers are skipped, accepted papers with an unfinished final QV are repaired, interrupted papers are archived
# to <root>/_archive/ and rerun. Usage: setsid nohup bash scripts/resume_pilot.sh > logs/resume_pilot.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f env.local.sh ] || { echo "env.local.sh missing (sets AUTODATA_CACHE_DIR); restore it first"; exit 1; }
source env.sh && source .venv/bin/activate
CFG=${CFG:-configs/cs_pilot.yaml}; CORPUS=${CORPUS:-data/corpus/cs2022_pilot.jsonl}
AGENTIC_SLOTS=${AGENTIC_SLOTS:-8}; COT_SLOTS=${COT_SLOTS:-2}
stamp=$(date +%Y%m%d-%H%M%S)
echo "$(date '+%F %T') resume: guard + servers"
ps -eo args | awk '$1 ~ /bash$/ && $2 ~ /gpu_guard\.sh$/ {found=1} END {exit !found}' \
  || { setsid nohup bash scripts/gpu_guard.sh >> logs/gpu_guard.log 2>&1 < /dev/null & }
wait_port() { local t=0; until curl -fsS --max-time 3 "http://127.0.0.1:$1/v1/models" >/dev/null; do
  sleep 15; t=$((t+15)); [ $t -ge 3600 ] && { echo "port $1 not up after 3600 s"; return 1; }; done; }
curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null || bash serving/serve_glm53.sh
wait_port 8000 || exit 1
curl -fsS --max-time 3 http://127.0.0.1:8001/v1/models >/dev/null || bash serving/serve_qwen27b.sh
wait_port 8001 || exit 1
curl -fsS --max-time 3 http://127.0.0.1:8002/v1/models >/dev/null || bash serving/serve_qwen4b.sh
bash scripts/wait_servers.sh 3600 || exit 1
python serving/healthcheck.py || echo "healthcheck reported a problem (continuing; the pipeline health gate waits per paper)"
rm -f runs/pilot/DRAIN runs/pilot_cot/DRAIN
echo "$(date '+%F %T') resume: runners (agentic $AGENTIC_SLOTS, CoT $COT_SLOTS)"
( autodata-run-cs --config $CFG --corpus $CORPUS --concurrency $AGENTIC_SLOTS --allow-config-mismatch \
    --workdir-root runs/pilot > logs/pilot_agentic_$stamp.log 2>&1; echo "AGENTIC_EXIT=$?" >> logs/pilot_agentic_$stamp.log ) &
( autodata-cot-baseline --config $CFG --corpus $CORPUS --concurrency $COT_SLOTS --allow-config-mismatch \
    --workdir-root runs/pilot_cot > logs/pilot_cot_$stamp.log 2>&1; echo "COT_EXIT=$?" >> logs/pilot_cot_$stamp.log ) &
wait
echo "$(date '+%F %T') pilot runners finished"
autodata-stats --run-root runs/pilot --cot-root runs/pilot_cot --json runs/pilot_stats.json
