#!/usr/bin/env bash
# Stop the Autodata vLLM servers (GLM-5.3 :8000, Qwen3.8-27B :8001, Qwen3.5-4B :8002) by command-line pattern.
# Each server was started with setsid by serving/serve_*.sh, so its API server leads its own process group; the
# whole group (API server, engine core, TP workers) gets SIGTERM, then SIGKILL after GRACE_S seconds (a server
# still in startup autotuning can ignore SIGTERM for a while). Does not touch scripts/gpu_guard.sh or the keepalive.
# Usage: serving/stop_all.sh [glm53|qwen27b|qwen4b ...]   (no argument = all three)
GRACE_S=${GRACE_S:-60}
# Matches only the API server process (".../.venv/bin/python3 .../.venv/bin/vllm serve <model> ...").
declare -A PATTERNS=(
  [glm53]='bin/vllm serve nvidia/GLM-5\.3-NVFP4( |$)'
  [qwen27b]='bin/vllm serve Qwen/Qwen3\.8-27B-FP8( |$)'
  [qwen4b]='bin/vllm serve Qwen/Qwen3\.5-4B( |$)'
)
names=("$@")
[[ ${#names[@]} -eq 0 ]] && names=(glm53 qwen27b qwen4b)

# True if the target (-PGID or PID) still has a live (non-zombie) process.
alive() {
  local t=$1 pids p
  if [[ $t == -* ]]; then pids=$(pgrep -g "${t#-}"); else pids=$t; fi
  for p in $pids; do
    [[ $(ps -o stat= -p "$p" 2>/dev/null) =~ ^[^Z] ]] && return 0
  done
  return 1
}

targets=()
for name in "${names[@]}"; do
  pattern=${PATTERNS[$name]:-}
  if [[ -z $pattern ]]; then echo "unknown server '$name' (expected glm53, qwen27b or qwen4b)" >&2; exit 2; fi
  pids=$(pgrep -u "$(id -u)" -f -- "$pattern" || true)
  if [[ -z $pids ]]; then echo "[$name] not running"; continue; fi
  for pid in $pids; do
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [[ -z $pgid ]] && continue
    # Signal the whole group only if the server leads it (setsid launch); otherwise just the pid.
    if [[ $pgid == "$pid" ]]; then target="-$pgid"; else target="$pid"; fi
    echo "[$name] SIGTERM pid=$pid target=$target"
    kill -TERM -- "$target" 2>/dev/null
    targets+=("$target")
  done
done

[[ ${#targets[@]} -eq 0 ]] && exit 0
for _ in $(seq 1 "$GRACE_S"); do
  any=0
  for t in "${targets[@]}"; do alive "$t" && any=1; done
  [[ $any -eq 0 ]] && { echo "all stopped"; exit 0; }
  sleep 1
done
for t in "${targets[@]}"; do
  if alive "$t"; then echo "SIGKILL $t (still running after ${GRACE_S}s)"; kill -KILL -- "$t" 2>/dev/null; fi
done
sleep 2
echo "stopped (some processes needed SIGKILL)"
