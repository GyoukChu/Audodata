# Source this before running anything in the project:  source env.sh
# Machine-specific settings (cache location, secrets) go in env.local.sh (gitignored) or a .env file; see .env.example.
AUTODATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AUTODATA_ROOT
if [ -f "$AUTODATA_ROOT/env.local.sh" ]; then . "$AUTODATA_ROOT/env.local.sh"; fi
export AUTODATA_CACHE_DIR="${AUTODATA_CACHE_DIR:-$AUTODATA_ROOT/.cache}"   # all Hugging Face / kernel caches live here
export HF_HOME="${HF_HOME:-$AUTODATA_CACHE_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$AUTODATA_CACHE_DIR/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$AUTODATA_CACHE_DIR/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$AUTODATA_CACHE_DIR/triton}"
# secrets: HF_TOKEN (Hugging Face), S2_API_KEY (Semantic Scholar) from ./.env or ~/.env
for f in "$AUTODATA_ROOT/.env" "$HOME/.env"; do
  if [ -f "$f" ]; then set -a; . "$f"; set +a; break; fi
done
