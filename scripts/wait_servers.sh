#!/usr/bin/env bash
# Wait until all three vLLM servers answer /v1/models (max ${1:-1800} seconds).
max=${1:-1800}; t=0
while [ $t -lt $max ]; do
  ok=0
  for p in 8000 8001 8002; do curl -s -m 3 http://127.0.0.1:$p/v1/models | grep -q '"id"' && ok=$((ok+1)); done
  [ $ok -eq 3 ] && { echo "all servers up"; exit 0; }
  sleep 15; t=$((t+15))
done
echo "servers not up after ${max}s"; exit 1
