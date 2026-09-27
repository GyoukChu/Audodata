"""Run the Agentic Self-Instruct loop over a corpus of papers (concurrently, resumable).

    autodata-run-cs --config configs/cs_default.yaml --corpus data/corpus/cs2022_smoke.jsonl [--limit N] [--concurrency K]
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx

from autodata.config import AppConfig, load_config
from autodata.cs.corpus_io import load_papers
from autodata.cs.prompts import PromptSet
from autodata.cs.run_paper import HARNESS_VERSION, PaperInput, PaperRun, config_fingerprint
from autodata.llm.client import LLMClient

AGENT_ROLES = ("main_agent", "challenger", "quality_verifier")


def _atomic_write_json(path: Path, data: dict) -> None:
    # Different corpus runners can finish together; never share a .tmp filename.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_clients(cfg: AppConfig, roles: tuple[str, ...] = AGENT_ROLES) -> dict[str, LLMClient]:
    return {role: LLMClient(cfg.endpoint(role), name=role) for role in roles}


def _cohort_manifest(
    cfg: AppConfig, papers: list[PaperInput], root: Path, prompts_dir: Path, *,
    corpus_path: Path | None = None, allow_config_mismatch: bool = False,
) -> dict:
    prompt_hashes = {path.name: hashlib.sha1(path.read_bytes()).hexdigest()
                     for path in sorted(prompts_dir.glob("*.md"))}
    corpus_sha1 = None
    if corpus_path is not None:
        corpus_path = Path(corpus_path).resolve()
        with corpus_path.open("rb") as stream:
            corpus_sha1 = hashlib.file_digest(stream, "sha1").hexdigest()
    current = {
        "run_name": cfg.run.name, "config_fingerprint": config_fingerprint(cfg),
        "prompt_hashes": prompt_hashes, "corpus_path": str(corpus_path) if corpus_path else None,
        "corpus_sha1": corpus_sha1, "paper_ids": list(dict.fromkeys(p.paper_id for p in papers)),
        "created_at": datetime.now(timezone.utc).isoformat(), "harness_version": HARNESS_VERSION,
    }
    path = root / "cohort.json"
    # Keep the lock inode separate from the atomically replaced manifest.
    lock_path = root / ".cohort.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            changed = [key for key in ("config_fingerprint", "prompt_hashes")
                       if previous.get(key) != current[key]]
            if changed and not allow_config_mismatch:
                raise SystemExit(
                    f"Cohort mismatch in {path}: {', '.join(changed)} differ from this run. "
                    "Use a new workdir root or --allow-config-mismatch to record the change."
                )
            if changed:
                def revision(manifest: dict) -> dict:
                    return {key: manifest.get(key) for key in (
                        "config_fingerprint", "prompt_hashes", "created_at", "harness_version",
                    )}
                history = previous.setdefault("history", [revision(previous)])
                history.append(revision(current))
                for key in ("config_fingerprint", "prompt_hashes", "harness_version"):
                    previous[key] = current[key]
            previous["paper_ids"] = list(dict.fromkeys(previous.get("paper_ids", []) + current["paper_ids"]))
            if previous.get("corpus_path") is None and corpus_path is not None:
                previous.update(corpus_path=current["corpus_path"], corpus_sha1=corpus_sha1)
            current = previous
        _atomic_write_json(path, current)
    return current


async def wait_for_endpoints(
    cfg: AppConfig, roles: tuple[str, ...], log: Callable[[str], None], *, health_wait_s: float = 1800,
) -> None:
    endpoints = {cfg.endpoint(role).base_url.rstrip("/"): cfg.endpoint(role)
                 for role in (*roles, "weak_solver", "strong_solver", "judge") if role in cfg.models}
    if not endpoints:
        return
    started = time.monotonic()
    deadline = started + health_wait_s
    waiting = False
    async with httpx.AsyncClient() as client:
        async def probe(base_url, endpoint) -> bool:
            try:
                response = await client.get(
                    f"{base_url}/models", headers={"Authorization": f"Bearer {endpoint.api_key}"}, timeout=5,
                )
                response.raise_for_status()
                return True
            except httpx.HTTPError:
                return False

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                ready = await asyncio.wait_for(
                    asyncio.gather(*(probe(url, endpoint) for url, endpoint in endpoints.items())),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                break
            if all(ready):
                if waiting:
                    log(f"endpoints ready after {time.monotonic() - started:.1f}s")
                return
            waiting = True
            unavailable = [url for url, ok in zip(endpoints, ready) if not ok]
            log(f"waiting for endpoints ({time.monotonic() - started:.1f}/{health_wait_s:g}s): "
                + ", ".join(unavailable))
            await asyncio.sleep(min(15, max(0, deadline - time.monotonic())))
    log(f"endpoint health wait timed out after {health_wait_s:g}s; proceeding with the paper")


def _needs_final_qv_repair(cfg: AppConfig, summary: dict) -> bool:
    final_qv = summary.get("final_qv")
    return bool(summary.get("accepted") and (
        (isinstance(final_qv, dict) and final_qv.get("qv_completed") is False)
        or (final_qv is None and cfg.run.final_qv)
    ))


def _agentic_done(summary: dict) -> bool:
    return bool(summary.get("completed") and not summary.get("errors")
                and summary.get("agent_stop_reason") not in ("error", "crashed"))


def previous_summary(workdir: Path, filename: str, *, retry_errors: bool,
                     is_done: Callable[[dict], bool] = _agentic_done) -> dict | None:
    """Return a usable previous summary for resume, or None when the paper must (re)run."""
    p = workdir / filename
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(s, dict):
        return None
    if retry_errors and not is_done(s):
        return None
    return s


def archive_workdir(workdir: Path, *, root: Path | None = None) -> None:
    """Move any previous workspace out of the run root before a fresh attempt.

    Call while holding PaperLock. The optional root defaults to workdir.parent;
    numeric suffixes preserve multiple reruns within the same second.
    """
    if workdir.exists():
        archive = (Path(root) if root is not None else workdir.parent) / "_archive"
        archive.mkdir(parents=True, exist_ok=True)
        name = f"{workdir.name}.{time.strftime('%Y%m%d-%H%M%S')}"
        dest = archive / name
        suffix = 0
        while dest.exists():
            suffix += 1
            dest = archive / f"{name}.{suffix}"
        shutil.move(str(workdir), str(dest))


class PaperLock:
    """Nonblocking flock, automatically released if the owning process dies.

    Keep the inode in _locks, outside workdirs that can be archived. Never unlink
    it: another opener must always lock the same inode, including after release.
    """

    def __init__(self, workdir: Path):
        self.path = workdir.parent / "_locks" / f"{workdir.name}.lock"
        self.fd: int | None = None

    def acquire(self) -> bool:
        if self.fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except BaseException:
            os.close(fd)
            raise
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {time.time()}".encode())
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        return True

    def release(self) -> None:
        if self.fd is not None:
            fd, self.fd = self.fd, None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


async def run_papers(
    cfg: AppConfig, papers: list[PaperInput], root: Path, *, concurrency: int, resume: bool,
    retry_errors: bool, summary_filename: str, is_done: Callable[[dict], bool],
    run_one: Callable[..., Awaitable[dict]],
    roles: tuple[str, ...], prompts_dir: Path | None = None,
    allow_config_mismatch: bool = False, corpus_path: Path | None = None, health_wait_s: float = 1800,
) -> list[dict]:
    """Run either paper workflow with shared locking, resume, archives and bookkeeping."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if health_wait_s < 0:
        raise ValueError("health_wait_s must be nonnegative")
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(cfg)
    prompts_dir = Path(prompts_dir or cfg.prompts_dir).resolve()
    cohort = _cohort_manifest(cfg, papers, root, prompts_dir, corpus_path=corpus_path,
                              allow_config_mismatch=allow_config_mismatch)
    stamps = {"config_fingerprint": fingerprint,
              "prompt_hashes_sha1": hashlib.sha1(json.dumps(
                  cohort["prompt_hashes"], sort_keys=True, ensure_ascii=False,
              ).encode("utf-8")).hexdigest()}
    prompts = PromptSet(prompts_dir)
    clients = build_clients(cfg, roles)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    summary_path = root / "summary.jsonl"
    results: list[dict] = []
    t0 = time.time()
    done = {"n": 0}

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    async def one(p: PaperInput) -> dict:
        workdir = root / p.paper_id
        started = time.time()

        def failed(exc: Exception) -> dict:
            error = f"{type(exc).__name__}: {exc}"
            log(f"[{p.paper_id}] crashed: {error}")
            return {"paper_id": p.paper_id, "title": p.title, "accepted": False,
                    "final_accepted": False, "completed": False, "agent_stop_reason": "crashed",
                    "errors": [error], "rounds": [], "n_rounds": 0,
                    "config_fingerprint": fingerprint, "wall_time_s": round(time.time() - started, 3),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

        # Queued papers remain available to other runners. Recheck resume state
        # after acquiring the lock, because another runner may have just finished.
        async with sem:
            plock = PaperLock(workdir)
            try:
                if not plock.acquire():
                    log(f"[{p.paper_id}] skipped: workspace is locked by another runner")
                    s = {"paper_id": p.paper_id, "title": p.title, "skipped": "locked",
                         "accepted": False, "completed": False,
                         "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                else:
                    try:
                        repair = False
                        prev = None
                        if resume:
                            prev = previous_summary(workdir, summary_filename, retry_errors=False, is_done=is_done)
                            repair = (summary_filename == "harness_summary.json" and prev is not None
                                      and _needs_final_qv_repair(cfg, prev))
                            if prev is not None and retry_errors and not repair and not is_done(prev):
                                prev = None
                            if prev is not None:
                                if prev.get("config_fingerprint") not in (None, fingerprint):
                                    log(f"[{p.paper_id}] resume: WARNING previous run used a different config "
                                        f"({prev.get('config_fingerprint')})")
                                if not repair:
                                    log(f"[{p.paper_id}] resume: already done (accepted={prev.get('accepted')})")
                                    return prev
                                log(f"[{p.paper_id}] resume: repairing final QV only")
                        await wait_for_endpoints(cfg, roles, lambda msg: log(f"[{p.paper_id}] {msg}"),
                                                 health_wait_s=health_wait_s)
                        if not repair:
                            archive_workdir(workdir, root=root)
                        try:
                            update = await run_one(p, workdir, clients, prompts, prompts_dir, log,
                                                   **({"repair_final_qv": True} if repair else {}))
                            s = {**prev, **update} if repair else update
                        except Exception as exc:
                            if repair:
                                log(f"[{p.paper_id}] final QV repair failed: {type(exc).__name__}: {exc}")
                                s = {**prev, "final_accepted": False,
                                     "errors": [*(prev.get("errors") or []), f"final_qv: {type(exc).__name__}: {exc}"]}
                            else:
                                s = failed(exc)
                        s = {**s, **stamps}
                        try:
                            _atomic_write_json(workdir / summary_filename, s)
                        except Exception as write_exc:
                            s.setdefault("errors", []).append(f"writing {summary_filename}: {type(write_exc).__name__}: {write_exc}")
                    finally:
                        plock.release()
            except Exception as exc:
                # Filesystem/constructor failures are per-paper failures too.
                s = failed(exc)
        async with lock:
            s = {**s, **stamps}
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            done["n"] += 1
            done["accepted"] = done.get("accepted", 0) + (1 if s.get("accepted") else 0)
            log(f"progress: {done['n']}/{len(papers)} papers, accepted so far {done['accepted']}, elapsed {time.time()-t0:.0f}s")
        return s

    try:
        results = list(await asyncio.gather(*[one(p) for p in papers]))
    finally:
        usage = {role: c.usage_totals for role, c in clients.items()}
        _atomic_write_json(root / "usage_agents.json", usage)
    return results


async def run_corpus(cfg: AppConfig, papers: list[PaperInput], workdir_root: Path, *, concurrency: int,
                     resume: bool = True, prompts_dir: Path | None = None, retry_errors: bool = True,
                     allow_config_mismatch: bool = False, corpus_path: Path | None = None,
                     health_wait_s: float = 1800) -> list[dict]:
    async def run_one(paper, workdir, clients, prompts, prompts_dir_abs, log, *, repair_final_qv=False):
        runner = PaperRun(cfg, paper, workdir, clients, prompts, prompts_dir_abs, log=log)
        if repair_final_qv:
            summary = json.loads((workdir / "harness_summary.json").read_text(encoding="utf-8"))
            return await runner.run_final_qv_only(summary)
        return await runner.run()

    return await run_papers(cfg, papers, workdir_root, concurrency=concurrency, resume=resume,
                            retry_errors=retry_errors, summary_filename="harness_summary.json",
                            is_done=_agentic_done, run_one=run_one, roles=AGENT_ROLES, prompts_dir=prompts_dir,
                            allow_config_mismatch=allow_config_mismatch, corpus_path=corpus_path,
                            health_wait_s=health_wait_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agentic Self-Instruct (CS) over a corpus")
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--paper-ids", help="comma-separated paper ids to run")
    ap.add_argument("--workdir-root", help="override run.workdir_root")
    ap.add_argument("--concurrency", type=int, help="override run.paper_concurrency")
    ap.add_argument("--no-resume", action="store_true", help="archive existing per-paper workspaces and start fresh")
    ap.add_argument("--keep-errors", action="store_true", help="on resume, do not rerun papers that ended in error")
    ap.add_argument("--prompts-dir")
    ap.add_argument("--allow-config-mismatch", action="store_true", help="record changed config/prompts in cohort history")
    ap.add_argument("--health-wait-s", type=float, default=1800, help="endpoint readiness deadline per paper (seconds)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ids = set(args.paper_ids.split(",")) if args.paper_ids else None
    papers = load_papers(args.corpus, limit=args.limit, offset=args.offset, paper_ids=ids,
                         min_chars=cfg.run.paper_text_min_chars)
    if not papers:
        print("no papers selected", file=sys.stderr)
        return 1
    root = Path(args.workdir_root or cfg.run.workdir_root)
    conc = args.concurrency or cfg.run.paper_concurrency
    print(f"running {len(papers)} papers -> {root} (concurrency {conc}, preset {cfg.acceptance.name}, "
          f"max_rounds {cfg.run.max_rounds})", flush=True)
    results = asyncio.run(run_corpus(cfg, papers, root, concurrency=conc, resume=not args.no_resume,
                                     prompts_dir=Path(args.prompts_dir) if args.prompts_dir else None,
                                     retry_errors=not args.keep_errors, allow_config_mismatch=args.allow_config_mismatch,
                                     corpus_path=Path(args.corpus), health_wait_s=args.health_wait_s))
    n_acc = sum(1 for r in results if r.get("accepted"))
    n_final = sum(1 for r in results if r.get("final_accepted"))
    print(f"done: {len(results)} papers, accepted {n_acc}, accepted+final QV {n_final}", flush=True)
    try:
        from autodata.cs.stats import print_report, summarize_agentic

        print_report(summarize_agentic(root))
    except Exception as e:  # stats are best-effort
        print(f"(stats unavailable: {e!r})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
