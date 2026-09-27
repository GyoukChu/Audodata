#!/usr/bin/env python3
"""Health check for the three Autodata vLLM servers: GET /v1/models on each, print OK/FAIL.

Usage: python serving/healthcheck.py [--wait SECONDS]
  --wait N   keep polling (every 10 s) until all servers are OK or N seconds have passed
Exit code 0 only if all three are OK. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SERVERS = [
    ("glm-5.3", "http://127.0.0.1:8000"),
    ("qwen3.8-27b", "http://127.0.0.1:8001"),
    ("qwen3.5-4b", "http://127.0.0.1:8002"),
]


def probe(name: str, base: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout) as r:
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
        return False, f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
    models = {m.get("id"): m for m in data.get("data", [])}
    if name not in models:
        return False, f"served model ids {sorted(models)} do not include {name!r}"
    return True, f"max_model_len={models[name].get('max_model_len')}"


def check_all() -> bool:
    ok_all = True
    for name, base in SERVERS:
        ok, detail = probe(name, base)
        ok_all &= ok
        print(f"{'OK  ' if ok else 'FAIL'}  {name:<12} {base}  {detail}")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait", type=float, default=0.0)
    args = ap.parse_args()
    deadline = time.time() + args.wait
    while True:
        ok = check_all()
        if ok or time.time() >= deadline:
            return 0 if ok else 1
        print("-- not all up yet; retrying in 10 s")
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
