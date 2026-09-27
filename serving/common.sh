# Shared helpers for serving/serve_*.sh -- source this file, do not execute it.
# Sets up the project environment and provides `launch NAME PORT CMD...`, which starts CMD fully detached
# (setsid nohup, stdin from /dev/null) and logs to logs/serve_NAME.log. An existing non-empty log is kept
# as logs/serve_NAME.<timestamp>.log so every start attempt stays on disk.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
source "$ROOT/env.sh"
source "$ROOT/.venv/bin/activate"
mkdir -p "$ROOT/logs"

# The API server waits VLLM_ENGINE_READY_TIMEOUT_S (vLLM default 600 s) for the engine to finish loading
# weights, allocating KV cache and capturing CUDA graphs; the 433 GB GLM checkpoint on Lustre can exceed that.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}

launch() {
  local name=$1 port=$2
  shift 2
  local log="$ROOT/logs/serve_${name}.log"
  # vLLM binds its port only after the engine is ready, so also look for a server that is still loading.
  if ss -ltn "sport = :${port}" | grep -q LISTEN || pgrep -u "$(id -u)" -f -- "vllm serve .*--port ${port}( |$)" >/dev/null; then
    echo "[${name}] port ${port} is in use or a server for it is still starting; not starting (see serving/stop_all.sh)" >&2
    return 1
  fi
  if [[ -s "$log" ]]; then
    mv "$log" "$ROOT/logs/serve_${name}.$(date +%Y%m%d-%H%M%S).log"
  fi
  echo "[$(date -Is)] launch: $*" > "$log"
  setsid nohup "$@" >> "$log" 2>&1 < /dev/null &
  echo "[${name}] started pid $! on port ${port}; log: ${log}"
  echo "[${name}] wait with: until grep -q 'Application startup complete' ${log}; do sleep 30; done"
}
