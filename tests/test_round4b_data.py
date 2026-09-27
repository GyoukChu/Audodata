"""Corpus resume validation and shared limiter tests, using mocked HTTP and time."""
import json

import httpx
import pytest

from autodata.data import s2_client
from autodata.data.build_corpus import FilterConfig, iter_corpus

from .test_data import FakeClock, _build, bc, fake_release, make_client


def test_changed_filters_refused_even_when_existing_output_is_complete(fake_release):
    out = fake_release["tmp"] / "corpus.jsonl"
    _build(fake_release, out, 2, [0, 1])
    original = out.read_bytes()
    metadata = out.with_name(out.name + ".meta.json")
    before = metadata.read_bytes()
    assert json.loads(before)["filters"]["min_year"] == 2022
    with pytest.raises(ValueError, match="Filter mismatch.*min_year.*--allow-filter-mismatch"):
        _build(fake_release, out, 2, [0, 1], filt=FilterConfig(min_year=2024))
    assert out.read_bytes() == original and metadata.read_bytes() == before
    stats, _ = _build(fake_release, out, 2, [0, 1])
    assert stats["complete"] and stats["shards"] == []


@pytest.mark.parametrize("field,value,reason", [
    ("year", 2021, "year_too_old"), ("s2_fields", ["Biology"], "not_in_field"),
    ("body_text", "short", "body_too_short"), ("body_text", "x" * 200001, "body_too_long"),
    ("abstract", "", "no_abstract"), ("release_id", "older-release", "release_id"), ("shard", 5, "shard"),
])
def test_every_existing_record_is_validated_without_sidecar(fake_release, field, value, reason, capsys):
    out = fake_release["tmp"] / "corpus.jsonl"
    _build(fake_release, out, 2, [0, 1])
    out.with_name(out.name + ".meta.json").unlink()
    records = list(iter_corpus(out))
    records[-1][field] = value
    out.write_text("".join(json.dumps(record) + "\n" for record in records))
    before = out.read_bytes()
    with pytest.raises(ValueError, match=reason):
        _build(fake_release, out, 2, [0, 1])
    stats, _ = _build(fake_release, out, 2, [0, 1], allow_filter_mismatch=True)
    assert stats["complete"] and out.read_bytes() == before
    assert "--allow-filter-mismatch keeps existing records" in capsys.readouterr().err


def test_filter_metadata_detects_relaxation_and_override_preserves_history(fake_release):
    out = fake_release["tmp"] / "corpus.jsonl"
    _build(fake_release, out, 1, [0, 1])
    with pytest.raises(ValueError, match="build parameter filters"):
        _build(fake_release, out, 1, [0, 1], filt=FilterConfig(min_year=2020))
    _build(fake_release, out, 1, [0, 1], filt=FilterConfig(min_year=2020), allow_filter_mismatch=True)
    metadata = json.loads(out.with_name(out.name + ".meta.json").read_text())
    assert metadata["filters"]["min_year"] == 2020
    assert metadata["history"][0]["filters"]["min_year"] == 2022


def test_builder_cli_filter_mismatch_exits_nonzero_and_override_keeps_output(fake_release, monkeypatch):
    out = fake_release["tmp"] / "corpus.jsonl"
    _build(fake_release, out, 1, [0, 1])
    monkeypatch.setattr(bc, "S2Client", lambda: make_client(fake_release["handler"])[0])
    args = ["--out", str(out), "--n-papers", "1", "--shard-indices", "0,1", "--min-year", "2024",
            "--shard-cache", str(fake_release["tmp"] / "cache")]
    before = out.read_bytes()
    assert bc.main(args) == 1
    assert bc.main([*args, "--allow-filter-mismatch"]) == 0
    assert out.read_bytes() == before


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), float("-inf")])
def test_limiter_rejects_invalid_interval(value, monkeypatch):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        s2_client.RateLimiter(value)
    monkeypatch.setenv("S2_MIN_INTERVAL", str(value))
    with pytest.raises(ValueError, match="S2_MIN_INTERVAL.*finite and nonnegative"):
        s2_client.min_interval_from_env()


@pytest.mark.parametrize("interval", [0, 0.5, 1, 3, 4.5])
def test_direct_limiter_cannot_bypass_production_floor(interval):
    clock = FakeClock()
    limiter = s2_client.RateLimiter(interval, clock=clock.now, sleep=clock.sleep)
    for _ in range(2):
        with limiter.slot():
            pass
    assert limiter.history[1][0] - limiter.history[0][1] == max(interval, 3)


@pytest.mark.parametrize("retry_after", [601, 750, 900, 1200])
def test_retry_after_gate_is_honoured_by_fresh_limiter(tmp_path, retry_after):
    clock = FakeClock()
    path = tmp_path / "gate.lock"
    def limiter():
        return s2_client.RateLimiter(lock_path=path, clock=clock.now, sleep=clock.sleep, wall_clock=clock.now)
    first = limiter()
    client, _, _ = make_client(lambda req: httpx.Response(200))
    with first.slot() as slot:
        slot.cooldown = client.backoff_delay(1, httpx.Response(429, headers={"Retry-After": str(retry_after)}))
    second = limiter()
    with second.slot():
        pass
    assert second.history[0][0] - first.history[0][1] == min(retry_after, 900)
    assert max(clock.sleeps) <= 30
    assert second.total_wait_s == min(retry_after, 900)


def test_shared_gate_is_reread_after_each_bounded_sleep(tmp_path):
    clock = FakeClock()
    path = tmp_path / "gate.lock"
    path.write_text(str(clock.now() + 900))
    def sleep(seconds):
        clock.sleep(seconds)
        path.write_text(str(clock.now()))
    limiter = s2_client.RateLimiter(lock_path=path, clock=clock.now, sleep=sleep, wall_clock=clock.now)
    with limiter.slot():
        pass
    assert clock.sleeps == [30]
