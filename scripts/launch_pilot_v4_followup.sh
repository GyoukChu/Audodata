#!/usr/bin/env bash
# Round-4 pilot upgrade, part 2: agentic runner B now; after the pre-DRAIN runner exits, re-adjudicate its last paper
# and add runner C (1 slot) so that paper is resumed (runners A and B skipped it while it was locked).
# Usage: setsid nohup bash scripts/launch_pilot_v4_followup.sh <old_agentic_pid> > logs/pilot_v4_followup.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh && source .venv/bin/activate
OLD_PID=${1:?old agentic runner pid}
CFG=configs/cs_pilot.yaml; CORPUS=data/corpus/cs2022_pilot.jsonl
echo "$(date '+%F %T') start: agentic B (4 slots)"
( autodata-run-cs --config $CFG --corpus $CORPUS --concurrency 4 --allow-config-mismatch --workdir-root runs/pilot \
    > logs/pilot_agentic_v4b.log 2>&1; echo "AGENTIC_V4B_EXIT=$?" >> logs/pilot_agentic_v4b.log ) &
while kill -0 "$OLD_PID" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') old runner $OLD_PID exited; re-adjudicating, then starting agentic C (1 slot)"
python scripts/readjudicate_refused.py --config $CFG --root runs/pilot
autodata-run-cs --config $CFG --corpus $CORPUS --concurrency 1 --allow-config-mismatch --workdir-root runs/pilot \
    > logs/pilot_agentic_v4c.log 2>&1; echo "AGENTIC_V4C_EXIT=$?" >> logs/pilot_agentic_v4c.log
wait
echo "$(date '+%F %T') follow-up runners finished"
