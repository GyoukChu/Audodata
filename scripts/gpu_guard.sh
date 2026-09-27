#!/usr/bin/env bash
# GPU guard: keeps the container alive (cluster deletes containers idle < 1.5% GPU memory for 3 h) WITHOUT wasting
# VRAM while inference servers run.
#   - if any `vllm serve` process is running (or GPU memory is already in use)  -> make sure the keepalive is NOT running
#   - if no vLLM server is running and every GPU is (almost) empty for >= IDLE_GRACE_S -> start the keepalive
# Run detached:  setsid nohup bash scripts/gpu_guard.sh > logs/gpu_guard.log 2>&1 &
set -u
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
IDLE_GRACE_S=${IDLE_GRACE_S:-600}
IDLE_MIB=${IDLE_MIB:-6000}          # a GPU with less than this is considered idle (keepalive itself uses ~4.3 GB)
idle_since=""
while true; do
  vllm_up=$(pgrep -f "vllm serve" | wc -l)
  max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  max_used=${max_used:-0}
  ka_pid=$(pgrep -f "scripts/gpu_keepalive.py" | head -1)
  now=$(date +%s)
  if [ "$vllm_up" -gt 0 ] || [ "$max_used" -gt 40000 ]; then
    idle_since=""
    if [ -n "$ka_pid" ]; then echo "$(date '+%F %T') servers/GPU busy (vllm=$vllm_up, max_used=${max_used}MiB): stopping keepalive $ka_pid"; kill "$ka_pid" 2>/dev/null; fi
  else
    if [ -n "$ka_pid" ]; then idle_since=""; else
      if [ -z "$idle_since" ]; then idle_since=$now; echo "$(date '+%F %T') GPUs idle (max_used=${max_used}MiB); grace ${IDLE_GRACE_S}s"; fi
      if [ $((now - idle_since)) -ge "$IDLE_GRACE_S" ] || [ "$max_used" -lt "$IDLE_MIB" ]; then
        echo "$(date '+%F %T') starting keepalive"; setsid nohup "$PROJ/.venv/bin/python" "$PROJ/scripts/gpu_keepalive.py" > "$PROJ/logs/gpu_keepalive.log" 2>&1 < /dev/null &
        idle_since=""; sleep 20
      fi
    fi
  fi
  sleep 60
done
