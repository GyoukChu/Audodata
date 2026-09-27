#!/usr/bin/env bash
# Round-4 pilot upgrade: new-code runners next to the retiring pre-DRAIN runner (see docs/RUNBOOK.md).
# Usage: setsid nohup bash scripts/launch_pilot_v4.sh <old_agentic_pid> > logs/pilot_v4_launcher.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh && source .venv/bin/activate
OLD_PID=${1:?old agentic runner pid}
CFG=configs/cs_pilot.yaml; CORPUS=data/corpus/cs2022_pilot.jsonl
echo "$(date '+%F %T') start: agentic A (3 slots) + CoT (2 slots); agentic B (5 slots) after pid $OLD_PID exits"
( autodata-run-cs --config $CFG --corpus $CORPUS --concurrency 3 --allow-config-mismatch --workdir-root runs/pilot \
    > logs/pilot_agentic_v4a.log 2>&1; echo "AGENTIC_V4A_EXIT=$?" >> logs/pilot_agentic_v4a.log ) &
( autodata-cot-baseline --config $CFG --corpus $CORPUS --concurrency 2 --allow-config-mismatch --workdir-root runs/pilot_cot \
    > logs/pilot_cot_v4.log 2>&1; echo "COT_V4_EXIT=$?" >> logs/pilot_cot_v4.log ) &
while kill -0 "$OLD_PID" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') old runner $OLD_PID exited; re-adjudicating its papers, then starting agentic B"
python scripts/readjudicate_refused.py --config $CFG --root runs/pilot
autodata-run-cs --config $CFG --corpus $CORPUS --concurrency 5 --allow-config-mismatch --workdir-root runs/pilot \
    > logs/pilot_agentic_v4b.log 2>&1; echo "AGENTIC_V4B_EXIT=$?" >> logs/pilot_agentic_v4b.log
wait
echo "$(date '+%F %T') all v4 runners finished"
