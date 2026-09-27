#!/usr/bin/env bash
# Usage: wait_markers.sh <max_seconds> <tail_lines> file1:MARKER1 [file2:MARKER2 ...]
# Polls every 20 s; when any file contains its marker, prints which ones and the tail of those files, then exits 0.
max=$1; tl=$2; shift 2
t=0
while [ $t -lt $max ]; do
  hit=0
  for spec in "$@"; do f=${spec%%:*}; m=${spec#*:}; if [ -f "$f" ] && grep -q "$m" "$f" 2>/dev/null; then hit=1; fi; done
  if [ $hit -eq 1 ]; then
    for spec in "$@"; do f=${spec%%:*}; m=${spec#*:}; if [ -f "$f" ] && grep -q "$m" "$f" 2>/dev/null; then echo "=== DONE: $f ($m) ==="; grep -vE '^\s*$' "$f" | tail -n "$tl" | cut -c1-500; else echo "=== pending: $f ==="; fi; done
    exit 0
  fi
  sleep 20; t=$((t+20))
done
echo "=== TIMEOUT after ${max}s ==="; for spec in "$@"; do f=${spec%%:*}; echo "--- $f:"; grep -vE '^\s*$' "$f" 2>/dev/null | tail -n 3 | cut -c1-200; done
