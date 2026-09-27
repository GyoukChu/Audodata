#!/usr/bin/env bash
# One optimization candidate: restart ONE server with extra vLLM flags, wait until ready, run the checks, then
# benchmark it at concurrency 1/8/32.
# Usage: serving/bench/candidate.sh glm53|qwen27b|qwen4b LABEL [extra vllm serve args...]
#   env QWEN4B_GPUS=2,3 is passed through to serve_qwen4b.sh; LEVELS=1,8,32 (default); NO_BENCH=1 skips the benchmark.
# Output: logs/serve_<name>.log (previous log rotated by common.sh), logs/bench_<LABEL>.log,
#         serving/bench/results/<LABEL>.json. Exit 1 if the server does not come up or a check fails.
set -u
name=$1; label=$2; shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
source env.sh && source .venv/bin/activate
declare -A SHORT=([glm53]=glm [qwen27b]=27b [qwen4b]=4b)
[[ -z ${SHORT[$name]:-} ]] && { echo "unknown server $name" >&2; exit 64; }

echo "[$(date +%T)] $label: stopping $name"
serving/stop_all.sh "$name"
sleep 5
echo "[$(date +%T)] $label: starting $name with: $*"
serving/serve_"$name".sh "$@" || exit 1
if ! serving/bench/wait_ready.sh "$name" 2400; then
  echo "[$(date +%T)] $label: $name FAILED to start"; exit 1
fi
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
python serving/healthcheck.py || { echo "healthcheck FAILED"; exit 1; }
case $name in
  glm53) python serving/check_glm53.py || { echo "check_glm53 FAILED"; exit 1; } ;;
  qwen27b) python serving/check_qwen.py 27b || { echo "check_qwen 27b FAILED"; exit 1; } ;;
  qwen4b) python serving/check_qwen.py 4b || { echo "check_qwen 4b FAILED"; exit 1; } ;;
esac
[[ ${NO_BENCH:-0} == 1 ]] && exit 0
echo "[$(date +%T)] $label: benchmark"
python serving/bench/bench.py --servers "${SHORT[$name]}" --levels "${LEVELS:-1,8,32}" --label "$label" \
  --notes "$name candidate $label: extra args: $*" --out "serving/bench/results/$label.json" \
  > "logs/bench_$label.log" 2>&1
rc=$?
grep -v -i -E "warn|warnings.warn" "logs/bench_$label.log"
echo "[$(date +%T)] $label: done (bench exit $rc)"
exit $rc
