#!/usr/bin/env bash
# Smoke sequence against the REAL servers: probe one challenger call -> 1 paper -> 3 papers -> 24 papers (agentic + CoT).
# Usage: bash scripts/smoke.sh [stage]   stage in {probe, one, three, full, cot}; default runs all in order.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh && source .venv/bin/activate
CFG=configs/cs_default.yaml; CORPUS=data/corpus/cs2022_smoke.jsonl
stage=${1:-all}
run_stage() {
  case "$1" in
    probe) python scripts/probe_subagent.py --config $CFG --corpus $CORPUS --workdir runs/smoke_probe --subagent challenger ;;
    one)   autodata-run-cs --config $CFG --corpus $CORPUS --limit 1 --concurrency 1 --workdir-root runs/smoke_one ;;
    three) autodata-run-cs --config $CFG --corpus $CORPUS --offset 1 --limit 3 --concurrency 3 --workdir-root runs/smoke_three ;;
    full)  autodata-run-cs --config $CFG --corpus $CORPUS --concurrency 6 --workdir-root runs/smoke24 ;;
    cot)   autodata-cot-baseline --config $CFG --corpus $CORPUS --concurrency 6 --workdir-root runs/smoke24_cot ;;
    stats) autodata-stats --run-root runs/smoke24 --cot-root runs/smoke24_cot --json runs/smoke24_stats.json ;;
  esac
}
bash scripts/wait_servers.sh 1800 || exit 1
if [ "$stage" = "all" ]; then for s in probe one three full cot stats; do echo "=== STAGE $s $(date '+%T') ==="; run_stage $s || { echo "STAGE $s FAILED"; exit 1; }; done
else run_stage "$stage"; fi
