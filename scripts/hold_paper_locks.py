#!/usr/bin/env python3
"""Retire a runner that predates the DRAIN file: hold the per-paper locks of every paper it has not started.

A running `autodata-run-cs` / `autodata-cot-baseline` process skips any paper whose lock is held by another process
("skipped: workspace is locked by another runner"), so while this helper holds the locks of all queued papers the old
runner finishes its in-flight papers and starts nothing new. Once the old runner's queue is exhausted, release the locks
(create the release file or send SIGTERM) and start the new runner(s) on the same run root.

Usage: python scripts/hold_paper_locks.py --root runs/pilot [--root runs/pilot_cot] [--release-file runs/.release_locks]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", action="append", required=True, help="run root with cohort.json (repeatable)")
    ap.add_argument("--release-file", default="runs/.release_locks", help="release all locks when this file appears")
    args = ap.parse_args()
    held: dict[str, int] = {}
    busy: list[str] = []
    for root in map(Path, args.root):
        ids = json.loads((root / "cohort.json").read_text(encoding="utf-8"))["paper_ids"]
        (root / "_locks").mkdir(exist_ok=True)
        for pid in ids:
            path = root / "_locks" / f"{pid}.lock"
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                busy.append(f"{root}/{pid}")
                continue
            held[f"{root}/{pid}"] = fd
    print(f"{time.strftime('%H:%M:%S')} holding {len(held)} locks; in flight elsewhere: {busy}", flush=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    release = Path(args.release_file)
    while not stop["flag"] and not release.exists():
        time.sleep(5)
    for fd in held.values():
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    print(f"{time.strftime('%H:%M:%S')} released {len(held)} locks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
