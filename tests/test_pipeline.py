import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import httpx

from autodata.config import AppConfig
from autodata.cs import pipeline
from autodata.cs.pipeline import PaperLock, archive_workdir, previous_summary, run_corpus, run_papers
from autodata.cs.run_paper import PaperInput


async def _simple_run(paper, *args):
    return {"paper_id": paper.paper_id, "completed": True, "accepted": False}


async def _shared_run(cfg, papers, root, **kwargs):
    return await run_papers(cfg, papers, root, concurrency=1, resume=True, retry_errors=True,
                            summary_filename="cot_summary.json", is_done=lambda s: bool(s.get("completed")),
                            run_one=_simple_run, roles=("challenger", "quality_verifier"), **kwargs)


async def test_manifest_records_launch_provenance_and_stamps_cot_summaries(tmp_path, fake_clients):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "challenger.md").write_text("challenge")
    (prompts / "quality_verifier.md").write_text("verify")
    (prompts / "ignored.txt").write_text("not a prompt")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"paper_id":"paper"}\n')
    cfg = AppConfig(models={}, prompts_dir=str(prompts), run={"name": "cohort test"})
    papers = [PaperInput("paper", "Title", "Text")]
    root = tmp_path / "run"
    result, = await _shared_run(cfg, papers, root, corpus_path=corpus)
    manifest = json.loads((root / "cohort.json").read_text())
    assert manifest == {
        "run_name": "cohort test", "config_fingerprint": pipeline.config_fingerprint(cfg),
        "prompt_hashes": {path.name: hashlib.sha1(path.read_bytes()).hexdigest() for path in prompts.glob("*.md")},
        "corpus_path": str(corpus.resolve()), "corpus_sha1": hashlib.sha1(corpus.read_bytes()).hexdigest(),
        "paper_ids": ["paper"], "created_at": manifest["created_at"], "harness_version": pipeline.HARNESS_VERSION,
    }
    assert manifest["created_at"].endswith("+00:00")
    assert result["config_fingerprint"] == manifest["config_fingerprint"]
    assert result["prompt_hashes_sha1"] == hashlib.sha1(
        json.dumps(manifest["prompt_hashes"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert json.loads((root / "paper/cot_summary.json").read_text()) == result
    assert json.loads((root / "summary.jsonl").read_text()) == result


@pytest.mark.parametrize("change", ["config_fingerprint", "prompt_hashes"])
async def test_manifest_mismatch_stops_before_run_unless_override_records_history(
    tmp_path, monkeypatch, fake_clients, change,
):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    prompt = prompts / "main_agent.md"
    prompt.write_text("original")
    cfg = AppConfig(models={}, prompts_dir=str(prompts))
    root = tmp_path / "run"
    await _shared_run(cfg, [], root)
    original = json.loads((root / "cohort.json").read_text())
    if change == "config_fingerprint":
        cfg.run.seed = 42
    else:
        prompt.write_text("changed")
    with pytest.raises(SystemExit, match=change + ".*--allow-config-mismatch"):
        await _shared_run(cfg, [], root)
    assert json.loads((root / "cohort.json").read_text()) == original
    await _shared_run(cfg, [], root, allow_config_mismatch=True)
    updated = json.loads((root / "cohort.json").read_text())
    assert updated[change] != original[change]
    assert updated["history"][0][change] == original[change]
    assert updated["history"][-1][change] == updated[change]
    assert updated["history"][-1]["config_fingerprint"] == pipeline.config_fingerprint(cfg)
    await _shared_run(cfg, [], root)
    assert json.loads((root / "cohort.json").read_text()) == updated


async def test_resuming_subset_preserves_pending_cohort_ids(tmp_path, fake_clients):
    cfg = AppConfig(models={})
    papers = [PaperInput(pid, pid, "Text") for pid in ("first", "pending")]
    # Record the requested cohort before any paper has started.
    pipeline._cohort_manifest(cfg, papers, tmp_path, Path(cfg.prompts_dir))
    await _shared_run(cfg, papers[:1], tmp_path)
    assert json.loads((tmp_path / "cohort.json").read_text())["paper_ids"] == ["first", "pending"]
    assert not (tmp_path / "pending").exists()


@pytest.mark.parametrize("final_qv,enabled", [({"qv_completed": False}, True), (None, True),
                                             ({"qv_completed": False}, False)])
@pytest.mark.parametrize("retry_errors", [True, False])
async def test_resume_repairs_only_final_qv_and_merges_summary(
    tmp_path, monkeypatch, fake_clients, final_qv, enabled, retry_errors,
):
    paper = PaperInput("paper", "Title", "Text")
    workdir = tmp_path / "paper"
    workdir.mkdir()
    previous = {"paper_id": "paper", "completed": True, "accepted": True, "final_accepted": False,
                "final_qv": final_qv, "rounds": [{"sentinel": "original solver result"}],
                "accepted_round": 1, "errors": ["final_qv: interrupted"]}
    (workdir / "harness_summary.json").write_text(json.dumps(previous))
    (workdir / "paper.txt").write_text(paper.text)
    calls = []

    class RepairRun:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self):
            pytest.fail("final QV repair must not regenerate the paper")

        async def run_final_qv_only(self, summary):
            assert summary == previous
            assert (workdir / "paper.txt").read_text() == paper.text
            assert not PaperLock(workdir).acquire()
            calls.append("repair")
            return {"final_qv": {"qv_completed": True, "passed": True}, "final_accepted": True, "errors": []}

    monkeypatch.setattr(pipeline, "PaperRun", RepairRun)
    cfg = AppConfig(models={}, run={"final_qv": enabled})
    result, = await run_corpus(cfg, [paper], tmp_path, concurrency=1, retry_errors=retry_errors)
    assert calls == ["repair"] and result["rounds"] == previous["rounds"]
    assert result["accepted"] and result["final_accepted"] and result["errors"] == []
    assert json.loads((workdir / "harness_summary.json").read_text()) == result
    assert json.loads((tmp_path / "summary.jsonl").read_text()) == result
    assert not (tmp_path / "_archive").exists()
    assert await run_corpus(cfg, [paper], tmp_path, concurrency=1) == [result]
    assert calls == ["repair"]


@pytest.mark.parametrize("accepted,final_qv,enabled", [
    (False, None, True), (True, None, False), (True, {"qv_completed": True, "passed": False}, True),
])
async def test_completed_summaries_outside_repair_conditions_stay_skipped(
    tmp_path, monkeypatch, fake_clients, accepted, final_qv, enabled,
):
    workdir = tmp_path / "paper"
    workdir.mkdir()
    previous = {"paper_id": "paper", "completed": True, "accepted": accepted, "final_qv": final_qv}
    (workdir / "harness_summary.json").write_text(json.dumps(previous))
    monkeypatch.setattr(pipeline, "PaperRun", lambda *a, **k: pytest.fail("completed summary reran"))
    assert await run_corpus(AppConfig(models={}, run={"final_qv": enabled}), [PaperInput("paper", "", "")],
                            tmp_path, concurrency=1) == [previous]


async def test_shared_driver_passes_repair_keyword_and_retains_accepted_result_on_error(tmp_path, fake_clients):
    workdir = tmp_path / "paper"
    workdir.mkdir()
    previous = {"paper_id": "paper", "completed": True, "accepted": True, "final_qv": None, "rounds": [{}]}
    (workdir / "harness_summary.json").write_text(json.dumps(previous))

    async def repair(*args, repair_final_qv=False, repair=False, prev=None):
        assert repair_final_qv and repair and prev == previous
        raise RuntimeError("verifier unavailable")

    result, = await run_papers(AppConfig(models={}), [PaperInput("paper", "", "")], tmp_path, concurrency=1,
                               resume=True, retry_errors=True, summary_filename="harness_summary.json",
                               is_done=pipeline._agentic_done, run_one=repair, roles=pipeline.AGENT_ROLES)
    assert result["accepted"] and result["rounds"] == previous["rounds"]
    assert result["final_qv"] is None and not result["final_accepted"]
    assert result["errors"] == ["final_qv: RuntimeError: verifier unavailable"]
    assert not (tmp_path / "_archive").exists()


def _health_config():
    return AppConfig(models={role: {"base_url": f"http://{host}/v1", "model": role, "api_key": "test-key"}
                             for role, host in (("main_agent", "agent"), ("challenger", "agent"),
                                                ("quality_verifier", "agent"), ("weak_solver", "weak"),
                                                ("strong_solver", "strong"), ("judge", "judge"))})


@pytest.mark.parametrize("failure", ["status", "connection"])
async def test_health_gate_polls_every_used_endpoint_until_all_recover(monkeypatch, failure):
    calls, sleeps, logs = [], [], []
    clock = [0.0]
    async_client = httpx.AsyncClient

    def probe(request):
        assert request.method == "GET" and request.url.path == "/v1/models"
        assert request.headers["authorization"] == "Bearer test-key"
        host = request.url.host
        calls.append(host)
        # Require all endpoints to be healthy in the same poll, including one that regresses.
        if (clock[0] == 0 and host == "strong") or (clock[0] == 15 and host == "agent"):
            if failure == "connection":
                raise httpx.ConnectError("restarting", request=request)
            return httpx.Response(503)
        return httpx.Response(200, json={"data": []})

    async def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(pipeline.httpx, "AsyncClient", lambda: async_client(transport=httpx.MockTransport(probe)))
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pipeline.asyncio, "sleep", sleep)
    await pipeline.wait_for_endpoints(_health_config(), pipeline.AGENT_ROLES, logs.append, health_wait_s=100)
    assert calls == ["agent", "weak", "strong", "judge"] * 3
    assert sleeps == [15, 15]
    assert "waiting for endpoints" in logs[0] and "strong" in logs[0]
    assert "ready after 30.0s" in logs[-1]


async def test_health_gate_deadline_logs_and_proceeds(monkeypatch):
    clock, sleeps, logs = [0.0], [], []
    async_client = httpx.AsyncClient

    async def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(pipeline.httpx, "AsyncClient", lambda: async_client(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))))
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pipeline.asyncio, "sleep", sleep)
    await pipeline.wait_for_endpoints(_health_config(), pipeline.AGENT_ROLES, logs.append, health_wait_s=20)
    assert sleeps == [15, 5]
    assert "timed out after 20s; proceeding" in logs[-1]


async def test_health_gate_runs_inside_semaphore_before_paper_starts(tmp_path, monkeypatch, fake_clients):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def wait(*args, **kwargs):
        calls.append("health")
        if len(calls) == 1:
            entered.set()
            await release.wait()

    async def run_one(paper, *args):
        calls.append(paper.paper_id)
        return {"paper_id": paper.paper_id}

    monkeypatch.setattr(pipeline, "wait_for_endpoints", wait)
    papers = [PaperInput(pid, "", "") for pid in ("first", "second")]
    task = asyncio.create_task(run_papers(AppConfig(models={}), papers, tmp_path, concurrency=1,
                                         resume=True, retry_errors=True, summary_filename="custom.json",
                                         is_done=lambda _: True, run_one=run_one, roles=pipeline.AGENT_ROLES))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert calls == ["health"]
        assert not (tmp_path / "first").exists()
        other = PaperLock(tmp_path / "second")
        assert other.acquire()
        other.release()
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
    assert calls == ["health", "first", "health", "second"]


def test_cli_passes_cohort_and_health_options(tmp_path, monkeypatch):
    cfg = AppConfig(models={})
    monkeypatch.setattr(pipeline, "load_config", lambda _: cfg)
    monkeypatch.setattr(pipeline, "load_papers", lambda *a, **k: [PaperInput("paper", "", "")])

    async def run(*args, **kwargs):
        assert kwargs["allow_config_mismatch"] is True
        assert kwargs["corpus_path"] == Path("corpus.jsonl")
        assert kwargs["health_wait_s"] == 42
        return []

    monkeypatch.setattr(pipeline, "run_corpus", run)
    assert pipeline.main(["--config", "config.yaml", "--corpus", "corpus.jsonl", "--workdir-root", str(tmp_path),
                          "--allow-config-mismatch", "--health-wait-s", "42"]) == 0


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
