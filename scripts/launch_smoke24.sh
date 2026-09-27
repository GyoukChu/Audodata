#!/usr/bin/env bash
# 24-paper smoke: Agentic Self-Instruct (concurrency 8) -> CoT Self-Instruct baseline -> Table-1 stats. Detached, resumable.
# Usage: setsid nohup bash scripts/launch_smoke24.sh > logs/smoke24.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh && source .venv/bin/activate
CFG=${CFG:-configs/cs_pilot.yaml}; CORPUS=${CORPUS:-data/corpus/cs2022_smoke.jsonl}; ROOT=${ROOT:-runs/smoke24}
bash scripts/wait_servers.sh 3600 || { echo "SERVERS_NOT_UP"; exit 1; }
echo "=== AGENTIC start $(date '+%F %T') ==="
autodata-run-cs --config $CFG --corpus $CORPUS --concurrency ${CONC:-8} --workdir-root $ROOT; echo "AGENTIC_EXIT=$?"
echo "=== COT start $(date '+%F %T') ==="
autodata-cot-baseline --config $CFG --corpus $CORPUS --concurrency ${CONC:-8} --workdir-root ${ROOT}_cot; echo "COT_EXIT=$?"
echo "=== STATS $(date '+%F %T') ==="
autodata-stats --run-root $ROOT --cot-root ${ROOT}_cot --json ${ROOT}_stats.json; echo "STATS_EXIT=$?"
echo "SMOKE24_DONE $(date '+%F %T')"
