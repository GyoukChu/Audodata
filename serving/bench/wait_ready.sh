#!/usr/bin/env bash
# Wait for a server started by serving/serve_<name>.sh to become ready.
# Usage: serving/bench/wait_ready.sh glm53|qwen27b|qwen4b [TIMEOUT_S=1800]
# Exit 0 when logs/serve_<name>.log shows "Application startup complete" and /v1/models answers,
# 1 when the server process is gone before that (prints the log tail), 2 on timeout.
# Afterwards prints the KV-cache lines of the log (size, concurrency, memory).
name=$1
timeout=${2:-1800}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log=$ROOT/logs/serve_${name}.log
declare -A PATTERNS=(
  [glm53]='bin/vllm serve nvidia/GLM-5\.3-NVFP4( |$)'
  [qwen27b]='bin/vllm serve Qwen/Qwen3\.8-27B-FP8( |$)'
  [qwen4b]='bin/vllm serve Qwen/Qwen3\.5-4B( |$)'
)
declare -A PORTS=([glm53]=8000 [qwen27b]=8001 [qwen4b]=8002)
pattern=${PATTERNS[$name]:-}
[[ -z $pattern ]] && { echo "usage: $0 glm53|qwen27b|qwen4b [timeout_s]" >&2; exit 64; }
start=$(date +%s)
while true; do
  if grep -q "Application startup complete" "$log" 2>/dev/null \
     && curl -sf "http://127.0.0.1:${PORTS[$name]}/v1/models" >/dev/null; then
    echo "[$name] ready after $(( $(date +%s) - start )) s"
    grep -E "GPU KV cache size|Maximum concurrency|Available KV cache memory|Model loading took|init engine .* took|Graph capturing finished|speculative|Speculative|WARNING" "$log" \
      | grep -v -E "AutoTuner|No tuned config" | cut -c1-400 | tail -40
    exit 0
  fi
  if ! pgrep -u "$(id -u)" -f -- "$pattern" >/dev/null; then
    echo "[$name] server process exited before becoming ready; last lines of $log:"
    grep -E "Error|error|Traceback|raise|Exception" "$log" | tail -15 | cut -c1-600
    tail -25 "$log" | cut -c1-600
    exit 1
  fi
  if (( $(date +%s) - start > timeout )); then
    echo "[$name] not ready after ${timeout} s"; tail -5 "$log" | cut -c1-400
    exit 2
  fi
  sleep 10
done
