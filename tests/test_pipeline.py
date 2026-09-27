import asyncio
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from autodata.config import AppConfig
from autodata.cs import pipeline
from autodata.cs.pipeline import PaperLock, archive_workdir, previous_summary, run_corpus, run_papers
from autodata.cs.run_paper import PaperInput


def test_flock_blocks_live_holder_and_acquires_stale_file(tmp_path):
    lock = PaperLock(tmp_path / "paper")
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    lock.path.write_text("stale PID")
    fd = os.open(lock.path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not lock.acquire() and lock.fd is None
        lock.release()
        assert lock.path.exists()
    finally:
        os.close(fd)
    assert lock.acquire()
    contender = PaperLock(tmp_path / "paper")
    assert not contender.acquire()
    lock.release()
    assert lock.path.exists()  # Unlinking could create a second independently locked inode.
    assert contender.acquire()
    contender.release()


def test_lock_released_on_process_death(tmp_path):
    script = """from pathlib import Path
import sys
from autodata.cs.pipeline import PaperLock
lock = PaperLock(Path(sys.argv[1]))
assert lock.acquire()
print('locked', flush=True)
sys.stdin.read()
"""
    process = subprocess.Popen([sys.executable, "-I", "-c", script, str(tmp_path / "paper")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        contender = PaperLock(tmp_path / "paper")
        assert not contender.acquire()
        process.kill()
        process.wait(timeout=10)
        assert contender.acquire()
        contender.release()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


@pytest.mark.parametrize("summary,retry_errors,usable", [
    ({"completed": True, "agent_stop_reason": "final"}, True, True),
    ({"completed": True, "agent_stop_reason": "max_steps"}, True, True),
    ({"completed": True, "agent_stop_reason": "length"}, True, True),
    ({"completed": False}, True, False),
    ({"completed": True, "agent_stop_reason": "error"}, True, False),
    ({"completed": True, "agent_stop_reason": "crashed"}, True, False),
    ({"completed": True, "errors": ["failure"]}, True, False),
    ({}, True, False),
    ({"completed": False, "agent_stop_reason": "error"}, False, True),
    ([], False, False),
])
def test_previous_summary_rules(tmp_path, summary, retry_errors, usable):
    (tmp_path / "harness_summary.json").write_text(json.dumps(summary))
    assert (previous_summary(tmp_path, "harness_summary.json", retry_errors=retry_errors) is not None) == usable


def test_previous_summary_missing_or_invalid(tmp_path):
    assert previous_summary(tmp_path, "summary.json", retry_errors=True) is None
    (tmp_path / "summary.json").write_text("broken")
    assert previous_summary(tmp_path, "summary.json", retry_errors=False) is None


def test_archive_default_root_and_timestamp_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.time, "strftime", lambda *a: "20260927-120000")
    workdir = tmp_path / "paper"
    for content in ("first", "second"):
        workdir.mkdir()
        (workdir / "marker").write_text(content)
        archive_workdir(workdir)
        assert not workdir.exists()
    archived = sorted((tmp_path / "_archive").iterdir())
    assert [p.name for p in archived] == ["paper.20260927-120000", "paper.20260927-120000.1"]
    assert [(p / "marker").read_text() for p in archived] == ["first", "second"]


@pytest.fixture
def fake_clients(monkeypatch):
    clients = {"main_agent": SimpleNamespace(usage_totals={"calls": 0, "prompt_tokens": 7})}
    monkeypatch.setattr(pipeline, "build_clients", lambda cfg, roles: clients)
    return clients


@pytest.mark.parametrize("resume", [False, True])
async def test_every_rerun_archives_before_paper_run(tmp_path, monkeypatch, fake_clients, resume):
    paper = PaperInput(paper_id="paper", title="Title", text="Text")
    wd = tmp_path / paper.paper_id
    wd.mkdir()
    (wd / "marker").write_text("old")
    (wd / "harness_summary.json").write_text(json.dumps({"completed": False, "errors": ["old error"]}))
    called = []

    class FakeRun:
        def __init__(self, cfg, p, workdir, *args, **kwargs):
            self.paper = p
            assert not workdir.exists()
            assert len(list((tmp_path / "_archive").glob("paper.*/marker"))) == 1
            workdir.mkdir()
            called.append(p.paper_id)
            # Archiving must not move the inode on which the runner holds its lock.
            other = PaperLock(workdir)
            assert not other.acquire()
            other.release()

        async def run(self):
            return {"paper_id": self.paper.paper_id, "accepted": False, "completed": True}

    monkeypatch.setattr(pipeline, "PaperRun", FakeRun)
    results = await run_corpus(AppConfig(models={}), [paper], tmp_path, concurrency=1, resume=resume)
    assert called == ["paper"] and results[0]["completed"]
    assert not (wd / "marker").exists()
    assert json.loads((tmp_path / "usage_agents.json").read_text())["main_agent"]["prompt_tokens"] == 7


async def test_resume_completed_summary_does_not_archive(tmp_path, monkeypatch, fake_clients):
    paper = PaperInput(paper_id="paper", title="Title", text="Text")
    wd = tmp_path / paper.paper_id
    wd.mkdir()
    previous = {"paper_id": "paper", "completed": True, "accepted": False}
    (wd / "harness_summary.json").write_text(json.dumps(previous))
    monkeypatch.setattr(pipeline, "PaperRun", lambda *a, **k: pytest.fail("completed paper reran"))
    assert await run_corpus(AppConfig(models={}), [paper], tmp_path, concurrency=1) == [previous]
    assert not (tmp_path / "_archive").exists()


async def test_crash_records_summary_continues_corpus_and_releases_lock(tmp_path, monkeypatch, fake_clients):
    papers = [PaperInput(paper_id=pid, title=pid, text="Text") for pid in ("bad", "good")]

    class FakeRun:
        def __init__(self, cfg, paper, workdir, *a, **k):
            self.paper = paper

        async def run(self):
            if self.paper.paper_id == "bad":
                raise RuntimeError("paper crashed")
            return {"paper_id": "good", "completed": True, "accepted": False}

    monkeypatch.setattr(pipeline, "PaperRun", FakeRun)
    result = await run_corpus(AppConfig(models={}), papers, tmp_path, concurrency=1)
    assert result[0]["completed"] is False and result[0]["errors"] == ["RuntimeError: paper crashed"]
    assert result[0]["agent_stop_reason"] == "crashed" and result[1]["completed"]
    lines = [json.loads(line) for line in (tmp_path / "summary.jsonl").read_text().splitlines()]
    assert lines == result
    assert json.loads((tmp_path / "bad/harness_summary.json").read_text()) == result[0]
    assert (tmp_path / "usage_agents.json").exists()
    lock = PaperLock(tmp_path / "bad")
    assert lock.acquire()
    lock.release()


async def test_queued_papers_remain_available_to_other_runners(tmp_path, monkeypatch, fake_clients):
    entered, release = asyncio.Event(), asyncio.Event()
    papers = [PaperInput(paper_id=pid, title=pid, text="Text") for pid in ("first", "queued")]

    class FakeRun:
        def __init__(self, cfg, paper, *a, **k):
            self.paper = paper

        async def run(self):
            assert self.paper.paper_id == "first"
            entered.set()
            await release.wait()
            return {"paper_id": "first", "completed": True, "accepted": False}

    monkeypatch.setattr(pipeline, "PaperRun", FakeRun)
    task = asyncio.create_task(run_corpus(AppConfig(models={}), papers, tmp_path, concurrency=1))
    contender = PaperLock(tmp_path / "queued")
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert contender.acquire()
        release.set()
        result = await asyncio.wait_for(task, 5)
        assert result[1]["skipped"] == "locked" and not result[1]["completed"]
        assert not (tmp_path / "queued").exists()
        assert not (tmp_path / "_archive").exists()
        assert '"skipped": "locked"' in (tmp_path / "summary.jsonl").read_text()
    finally:
        release.set()
        contender.release()
        await task


def test_concurrent_runners_can_write_usage_without_temporary_file_collisions(tmp_path, monkeypatch, fake_clients):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    barrier = threading.Barrier(2)
    replace = os.replace

    def synchronized_replace(source, destination):
        if Path(destination).name == "usage_agents.json":
            barrier.wait(timeout=5)
        return replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(asyncio.run, run_corpus(AppConfig(models={}), [], tmp_path, concurrency=1))
                   for _ in range(2)]
        assert [future.result(timeout=10) for future in futures] == [[], []]
    assert json.loads((tmp_path / "usage_agents.json").read_text())["main_agent"]["prompt_tokens"] == 7
    assert not list(tmp_path.glob(".usage_agents.json.*"))


async def test_shared_driver_locks_archives_and_writes_custom_summaries_atomically(tmp_path, monkeypatch, fake_clients):
    papers = [PaperInput(paper_id=pid, title=pid, text="Text") for pid in ("locked", "done", "rerun", "crash")]
    previous = {}
    for paper in papers[:3]:
        wd = tmp_path / paper.paper_id
        wd.mkdir()
        previous[paper.paper_id] = {"paper_id": paper.paper_id, "done": paper.paper_id == "done"}
        (wd / "custom.json").write_text(json.dumps(previous[paper.paper_id]))
    holder = PaperLock(tmp_path / "locked")
    assert holder.acquire()
    called, replaced = [], []
    replace = os.replace

    def checked_replace(source, destination):
        destination = Path(destination)
        assert json.loads(Path(source).read_text()) is not None
        assert Path(source).parent == destination.parent
        if destination.name == "custom.json":
            contender = PaperLock(destination.parent)
            assert not contender.acquire()  # Summary persistence is covered by the paper lock too.
            replaced.append(destination.parent.name)
        return replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", checked_replace)

    async def run_one(paper, workdir, clients, prompts, prompts_dir, log):
        called.append(paper.paper_id)
        assert clients is fake_clients and prompts_dir.is_absolute()
        assert not workdir.exists()
        if paper.paper_id == "crash":
            raise RuntimeError("callback failed before creating workspace")
        archived, = (tmp_path / "_archive").glob("rerun.*/custom.json")
        assert json.loads(archived.read_text()) == previous["rerun"]
        return {"paper_id": paper.paper_id, "done": True}

    try:
        results = await run_papers(
            AppConfig(models={}), papers, tmp_path, concurrency=1, resume=True, retry_errors=True,
            summary_filename="custom.json", is_done=lambda s: bool(s.get("done")), run_one=run_one,
            roles=("main_agent",),
        )
    finally:
        holder.release()
    assert called == replaced == ["rerun", "crash"]
    assert results[0]["skipped"] == "locked" and results[1] == previous["done"]
    assert results[3]["errors"] == ["RuntimeError: callback failed before creating workspace"]
    assert results[3]["completed"] is False and results[3]["agent_stop_reason"] == "crashed"
    for pid in ("locked", "done"):
        assert json.loads((tmp_path / pid / "custom.json").read_text()) == previous[pid]
    for result in results[2:]:
        assert json.loads((tmp_path / result["paper_id"] / "custom.json").read_text()) == result
        contender = PaperLock(tmp_path / result["paper_id"])
        assert contender.acquire()
        contender.release()
    assert [json.loads(line) for line in (tmp_path / "summary.jsonl").read_text().splitlines()] == [
        results[0], *results[2:],
    ]
    assert json.loads((tmp_path / "usage_agents.json").read_text())["main_agent"]["prompt_tokens"] == 7
    assert not list(tmp_path.glob("*/harness_summary.json"))
    assert not list(tmp_path.glob("*/.custom.json.*"))


async def test_shared_driver_cancellation_releases_lock_and_persists_usage(tmp_path, fake_clients):
    entered = asyncio.Event()

    async def run_one(*args):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(run_papers(
        AppConfig(models={}), [PaperInput("paper", "Title", "Text")], tmp_path,
        concurrency=1, resume=True, retry_errors=True, summary_filename="custom.json",
        is_done=lambda s: False, run_one=run_one, roles=("main_agent",),
    ))
    try:
        await asyncio.wait_for(entered.wait(), 5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    contender = PaperLock(tmp_path / "paper")
    assert contender.acquire()
    contender.release()
    assert json.loads((tmp_path / "usage_agents.json").read_text())["main_agent"]["prompt_tokens"] == 7


async def test_run_evaluator_isolates_environment_and_returns_decoded_streams(tmp_path, monkeypatch):
    from autodata.cs.run_paper import run_evaluator

    monkeypatch.setenv("PYTHONPATH", "/untrusted/modules")
    monkeypatch.setenv("PYTHONHOME", "/untrusted/python")
    monkeypatch.setenv("AUTODATA_TEST_ENV", "retained")

    class Process:
        returncode = 2

        async def communicate(self):
            return b"stdout\xff", b"stderr\xff"

    async def spawn(*args, **kwargs):
        assert args == (sys.executable, "-I", "-m", "autodata.cs.evaluate_rubric", "--help")
        assert kwargs["cwd"] == str(tmp_path)
        assert {k: v for k, v in kwargs["env"].items() if k.startswith("PYTHON")} == {"PYTHONUNBUFFERED": "1"}
        assert kwargs["env"]["AUTODATA_TEST_ENV"] == "retained"
        assert kwargs["stdout"] == kwargs["stderr"] == asyncio.subprocess.PIPE
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    assert await run_evaluator(tmp_path, ["--help"], deadline_s=1) == ("stdout\ufffd", "stderr\ufffd", 2)


@pytest.mark.parametrize("cancel", [False, True])
async def test_run_evaluator_kills_and_reaps_on_timeout_or_cancellation(tmp_path, monkeypatch, cancel):
    from autodata.cs.run_paper import run_evaluator

    entered = asyncio.Event()
    events = []

    class Process:
        returncode = None

        async def communicate(self):
            entered.set()
            await asyncio.Event().wait()

        def kill(self):
            events.append("kill")

        async def wait(self):
            self.returncode = -9
            events.append("wait")

    async def spawn(*args, **kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(run_evaluator(tmp_path, [], deadline_s=5 if cancel else 0.01))
    await asyncio.wait_for(entered.wait(), 5)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else asyncio.TimeoutError):
        await task
    assert events == ["kill", "wait"]


def test_evaluator_deadline_accounts_for_mode_and_retry_budgets():
    from autodata.cs.run_paper import evaluator_deadline_s

    cfg = AppConfig(models={"judge": {"base_url": "http://localhost/v1", "model": "judge", "timeout_s": 11}},
                    eval={"solver_retries": 2, "judge_retries": 3})
    assert evaluator_deadline_s(cfg, "weak-only", 7) == 365
    assert evaluator_deadline_s(cfg, "strong-only", 7) == 365
    assert evaluator_deadline_s(cfg, "both", 7) == 430


@pytest.mark.parametrize("n_pos,n_neg,ok", [(7, 3, True), (17, 3, True), (6, 3, False),
                                           (18, 3, False), (3, 7, False), (8, 2, False)])
def test_final_filter_shape_boundaries(n_pos, n_neg, ok):
    from autodata.cs.run_paper import final_filter

    cfg = SimpleNamespace(run=SimpleNamespace(final_min_context_chars=200))
    rubric = [{"criterion": "positive", "weight": 1}] * n_pos + [{"criterion": "negative", "weight": -1}] * n_neg
    result = final_filter(cfg, "x" * 200, rubric)
    assert result["rubric_ok"] is ok and result["context_ok"] is True
    assert result["n_rubric_items"] == n_pos + n_neg
    assert final_filter(cfg, "x" * 199, rubric)["context_ok"] is False


@pytest.mark.parametrize("overrides", [
    {"final_rubric_min_items": 12}, {"final_rubric_max_items": 10},
    {"final_rubric_min_positive": 9}, {"final_rubric_min_negative": 4},
])
def test_final_filter_reads_configured_thresholds(overrides):
    from autodata.cs.run_paper import final_filter

    cfg = SimpleNamespace(run=SimpleNamespace(final_min_context_chars=200, **overrides))
    rubric = [{"criterion": "positive", "weight": 1}] * 8 + [{"criterion": "negative", "weight": -1}] * 3
    assert final_filter(cfg, "x" * 200, rubric)["rubric_ok"] is False


def test_final_filter_invalid_rubric_retains_summary_shape():
    from autodata.cs.run_paper import final_filter

    result = final_filter(AppConfig(models={}), "", {"malformed": "rubric"})
    assert result["n_rubric_items"] is None
    assert result["context_chars"] == 0 and result["context_ok"] is False and result["rubric_ok"] is False
    assert "RubricError" in result["rubric_error"]
    assert "n_positive" not in result and "n_negative" not in result
