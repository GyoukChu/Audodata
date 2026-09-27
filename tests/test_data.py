"""Tests for autodata.data (S2 client, limiter, corpus builder). No network: httpx.MockTransport + a fake clock."""
from __future__ import annotations

import gzip
import json
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import orjson
import pytest

import autodata.data.build_corpus as bc
from autodata.data import CorpusRecord, iter_corpus, paper_text
from autodata.data.build_corpus import (FilterConfig, MetaCache, body_outcome, build_corpus, build_record,
                                        corpus_stats, field_categories, iter_shard_records, load_existing_ids,
                                        metadata_outcome, normalize_authors, parse_shard_indices, resolve_shards)
from autodata.data.s2_client import (BACKOFF_SCHEDULE, PAPER_BATCH_FIELDS, RateLimiter, S2Client, S2Error,
                                     S2HTTPError, min_interval_from_env, url_basename)

SPEC_KEYS = ["paper_id", "corpus_id", "title", "abstract", "year", "publication_date", "venue", "s2_fields",
             "external_ids", "authors", "body_text", "n_body_chars", "shard", "release_id"]
API_KEY = "test-key-DO-NOT-LOG-123"


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def make_client(handler: Callable[[httpx.Request], httpx.Response], *, min_interval: float = 3.0,
                **kw: Any) -> tuple[S2Client, FakeClock, RateLimiter]:
    clock = FakeClock()
    lim = RateLimiter(min_interval, clock=clock.now, sleep=clock.sleep)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = S2Client(api_key=API_KEY, limiter=lim, http=http, rng=random.Random(0), **kw)
    return client, clock, lim


def gaps(lim: RateLimiter) -> list[float]:
    """Time from the end of each request to the start of the next one."""
    return [b[0] - a[1] for a, b in zip(lim.history, lim.history[1:])]


ABSTRACT = "We study a problem and propose a method; experiments show that it works well."


def meta(cid: int, *, year: int | None = 2023, fields: tuple[str, ...] = ("Computer Science",),
         abstract: str | None = ABSTRACT, title: str = "A Paper") -> dict[str, Any]:
    return {"paperId": f"sha{cid}", "corpusId": cid, "title": title, "abstract": abstract, "year": year,
            "publicationDate": f"{year}-05-01" if year else None, "venue": "ICML",
            "s2FieldsOfStudy": [{"category": f, "source": "s2-fos-model"} for f in fields],
            "externalIds": {"CorpusId": cid, "ArXiv": f"2301.{cid:05d}", "DOI": None}, "citationCount": 3}


def shard_rec(cid: Any, body_len: int, *, title: str = "Shard Title") -> dict[str, Any]:
    body = ("word " * (body_len // 5 + 1))[:body_len]
    return {"corpusid": cid, "title": title, "authors": ["Ada Lovelace", "Alan Turing"],
            "openaccessinfo": {"externalids": {"DOI": "10.1/x", "ArXiv": None}, "license": "CCBY",
                               "url": "u", "status": "GREEN"},
            "body": {"text": body, "annotations": {"paragraph": "[]", "section_header": None, "bib_ref": None}},
            "bibliography": {"text": "refs", "annotations": {}}}


# ---------------------------------------------------------------------------------------------- record + text
def test_paper_text_exact_format() -> None:
    rec = {"title": "T", "abstract": "A b.", "body_text": "Body\ntext"}
    assert paper_text(rec) == "Title: T\n\nAbstract: A b.\n\nBody\ntext"


def test_build_record_schema_and_values() -> None:
    rec = build_record(shard_rec(42, 9000), meta(42), shard=3, release_id="2026-09-22")
    assert list(rec) == SPEC_KEYS
    assert rec["paper_id"] == "s2_42" and rec["corpus_id"] == 42
    assert rec["title"] == "A Paper" and rec["abstract"] == ABSTRACT and rec["year"] == 2023
    assert rec["publication_date"] == "2023-05-01" and rec["venue"] == "ICML"
    assert rec["s2_fields"] == ["Computer Science"]
    assert rec["external_ids"] == {"CorpusId": 42, "ArXiv": "2301.00042"}   # None values dropped
    assert rec["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert rec["n_body_chars"] == len(rec["body_text"]) == 9000
    assert rec["shard"] == 3 and rec["release_id"] == "2026-09-22"
    CorpusRecord.model_validate(rec)
    assert paper_text(rec) == f"Title: A Paper\n\nAbstract: {ABSTRACT}\n\n{rec['body_text']}"
    assert paper_text(CorpusRecord.model_validate(rec)) == paper_text(rec)


def test_build_record_fallbacks() -> None:
    m = meta(7, title="  ")
    m["externalIds"] = None
    m["venue"] = None
    rec = build_record(shard_rec(7, 9000, title="Title\n from  shard"), m, shard=0, release_id="r")
    assert rec["title"] == "Title from shard"
    assert rec["external_ids"] == {"DOI": "10.1/x"}
    assert rec["venue"] == ""


def test_normalize_authors_accepts_strings_and_dicts() -> None:
    raw = ["  Ada   Lovelace ", {"name": "Alan Turing"}, {"first": "Grace", "middle": ["B."], "last": "Hopper"}, 5, ""]
    assert normalize_authors(raw) == ["Ada Lovelace", "Alan Turing", "Grace B. Hopper"]
    assert normalize_authors(None) == []


def test_iter_corpus_and_load_existing_ids(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_bytes(b'{"corpus_id": 1}\n\n{"corpus_id": 2}\n{"corpus_id": 3, "tit')   # torn last line
    assert load_existing_ids(p) == [1, 2]
    assert p.read_bytes() == b'{"corpus_id": 1}\n\n{"corpus_id": 2}\n'
    assert [r["corpus_id"] for r in iter_corpus(p)] == [1, 2]
    p.write_bytes(b'{"corpus_id": 1}')           # complete record, missing newline -> newline added
    assert load_existing_ids(p) == [1] and p.read_bytes().endswith(b"\n")
    p.write_bytes(b'{"corpus_id": 1}\nnot json\n{"corpus_id": 2}\n')
    with pytest.raises(ValueError):
        load_existing_ids(p)
    with pytest.raises(ValueError):
        list(iter_corpus(p))
    assert load_existing_ids(tmp_path / "missing.jsonl") == []


# ---------------------------------------------------------------------------------------------- filters
def test_body_length_bounds_are_inclusive() -> None:
    f = FilterConfig(min_body_chars=8000, max_body_chars=200_000)
    assert body_outcome(7999, f) == "body_too_short"
    assert body_outcome(8000, f) is None
    assert body_outcome(200_000, f) is None
    assert body_outcome(200_001, f) == "body_too_long"
    assert body_outcome(0, f) == "body_too_short"


def test_metadata_filter() -> None:
    f = FilterConfig()
    assert metadata_outcome(meta(1), f) is None
    assert metadata_outcome(meta(1, year=2022), f) is None
    assert metadata_outcome(None, f) == "not_found"
    assert metadata_outcome({}, f) == "not_found"
    assert metadata_outcome(meta(1, year=None), f) == "year_missing"
    assert metadata_outcome(meta(1, year=2021), f) == "year_too_old"
    assert metadata_outcome(meta(1, fields=("Biology", "Medicine")), f) == "not_in_field"
    assert metadata_outcome(meta(1, fields=()), f) == "not_in_field"
    assert metadata_outcome(meta(1, fields=("Mathematics", "Computer Science")), f) is None
    assert metadata_outcome(meta(1, abstract=None), f) == "no_abstract"
    assert metadata_outcome(meta(1, abstract="  \n "), f) == "no_abstract"
    for placeholder in (",", ".", "Graphical abstract"):          # real API placeholders
        assert metadata_outcome(meta(1, abstract=placeholder), f) == "no_abstract"
    assert metadata_outcome(meta(1, abstract="x" * 50), f) is None
    assert metadata_outcome(meta(1, abstract=","), FilterConfig(min_abstract_chars=1)) is None   # literal mode
    assert metadata_outcome(meta(1, abstract=" "), FilterConfig(min_abstract_chars=0)) == "no_abstract"
    m = meta(1)
    m["s2FieldsOfStudy"] = None
    assert metadata_outcome(m, f) == "not_in_field"
    assert metadata_outcome(meta(1, year=2023), FilterConfig(min_year=2024)) == "year_too_old"


def test_field_categories_unique_in_order() -> None:
    m = {"s2FieldsOfStudy": [{"category": "Computer Science", "source": "external"},
                             {"category": "Mathematics", "source": "s2-fos-model"},
                             {"category": "Computer Science", "source": "s2-fos-model"}, {"source": "x"}, "junk"]}
    assert field_categories(m) == ["Computer Science", "Mathematics"]
    assert field_categories(None) == []


def test_shard_selection() -> None:
    assert parse_shard_indices("0,1") == [0, 1]
    assert parse_shard_indices("3, 0-2 ,1") == [3, 0, 1, 2]
    with pytest.raises(ValueError):
        parse_shard_indices("")
    with pytest.raises(ValueError):
        parse_shard_indices("3-1")
    assert resolve_shards(10, None) == [0]
    assert resolve_shards(10, [2, 5]) == [2, 5]
    a = resolve_shards(329, None, shard_seed=7, n_shards=4)
    assert a == resolve_shards(329, None, shard_seed=7, n_shards=4) and len(set(a)) == 4
    with pytest.raises(ValueError):
        resolve_shards(10, [10])
    with pytest.raises(ValueError):
        resolve_shards(10, [1], shard_seed=1, n_shards=2)


def test_meta_cache_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "meta.jsonl"
    c = MetaCache(p)
    c.put_many({1: {"year": 2023}, 2: None})
    with open(p, "ab") as f:
        f.write(b'{"corpus_id": 3, "me')     # torn line is ignored
    c2 = MetaCache(p)
    assert 1 in c2 and 2 in c2 and 3 not in c2 and len(c2) == 2
    assert c2.get(1) == {"year": 2023} and c2.get(2) is None


# ---------------------------------------------------------------------------------------------- rate limiter
def test_limiter_spacing_with_fake_clock() -> None:
    clock = FakeClock()
    lim = RateLimiter(3.0, clock=clock.now, sleep=clock.sleep)
    durations = [0.5, 0.0, 2.0, 10.0, 0.1]
    for d in durations:
        with lim.slot():
            clock.t += d          # the request itself takes d seconds
    starts = [s for s, _ in lim.history]
    assert starts[0] == 1000.0 and clock.sleeps[0] == 3.0          # first request does not wait
    assert all(g == pytest.approx(3.0) for g in gaps(lim))        # end -> next start is exactly the interval
    assert all(b - a >= 3.0 for a, b in zip(starts, starts[1:]))  # hence start -> start >= interval too
    assert lim.n_slots == 5


def test_limiter_does_not_wait_when_idle_long_enough() -> None:
    clock = FakeClock()
    lim = RateLimiter(3.0, clock=clock.now, sleep=clock.sleep)
    with lim.slot():
        pass
    clock.t += 60
    with lim.slot():
        pass
    assert clock.sleeps == []


def test_limiter_cooldown_is_global() -> None:
    clock = FakeClock()
    lim = RateLimiter(3.0, clock=clock.now, sleep=clock.sleep)
    with lim.slot() as s:
        s.cooldown = 10.0
    with lim.slot():
        pass
    with lim.slot():
        pass
    assert gaps(lim) == [pytest.approx(10.0), pytest.approx(3.0)]


def test_limiter_threads_never_overlap() -> None:
    lim = RateLimiter(0.03)
    def worker() -> None:
        for _ in range(3):
            with lim.slot():
                time.sleep(0.005)
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    hist = sorted(lim.history)
    assert len(hist) == 12
    assert all(b[0] - a[1] >= 0.03 for a, b in zip(hist, hist[1:]))


def test_limiter_file_gate_spaces_separate_instances(tmp_path: Path) -> None:
    lock = tmp_path / "s2.lock"
    a, b = RateLimiter(0.3, lock_path=lock), RateLimiter(0.3, lock_path=lock)   # like two processes
    with a.slot():
        pass
    with b.slot():
        pass
    assert b.history[0][0] - a.history[0][1] >= 0.3 - 0.01


def test_min_interval_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("S2_MIN_INTERVAL", raising=False)
    assert min_interval_from_env() == 3.0
    monkeypatch.setenv("S2_MIN_INTERVAL", "4.5")
    assert min_interval_from_env() == 4.5
    monkeypatch.setenv("S2_MIN_INTERVAL", "0.2")
    assert min_interval_from_env() == 1.0      # floor
    monkeypatch.setenv("S2_MIN_INTERVAL", "fast")
    assert min_interval_from_env() == 3.0


# ---------------------------------------------------------------------------------------------- S2 client
def test_paper_batch_request_shape_and_alignment() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        ids = json.loads(req.content)["ids"]
        return httpx.Response(200, json=[meta(int(i.split(":")[1])) if i != "CorpusId:2" else None for i in ids])

    client, clock, lim = make_client(handler)
    out = client.paper_batch([1, 2, 3])
    assert [m["corpusId"] if m else None for m in out] == [1, None, 3]
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/graph/v1/paper/batch"
    assert req.url.params["fields"] == PAPER_BATCH_FIELDS
    assert json.loads(req.content) == {"ids": ["CorpusId:1", "CorpusId:2", "CorpusId:3"]}
    assert req.headers["x-api-key"] == API_KEY
    client.paper_batch([4])
    assert gaps(lim) == [pytest.approx(3.0)]
    with pytest.raises(ValueError):
        client.paper_batch(list(range(501)))
    assert client.paper_batch([]) == []


def test_paper_batch_length_mismatch_raises() -> None:
    client, _, _ = make_client(lambda req: httpx.Response(200, json=[None]))
    with pytest.raises(S2Error):
        client.paper_batch([1, 2])


def test_429_exponential_backoff_then_success(capsys: pytest.CaptureFixture[str]) -> None:
    statuses = iter([429, 429, 503, 200])

    def handler(req: httpx.Request) -> httpx.Response:
        st = next(statuses)
        return httpx.Response(st, json=[meta(1)] if st == 200 else {"message": "Too Many Requests"})

    client, clock, lim = make_client(handler)
    assert client.paper_batch([1])[0]["corpusId"] == 1
    g = gaps(lim)
    assert len(g) == 3
    for gap, base in zip(g, BACKOFF_SCHEDULE):
        assert base <= gap <= base * 1.2 + 1e-9
    assert client.stats["status:429"] == 2 and client.stats["status:503"] == 1 and client.stats["retries"] == 3
    err = capsys.readouterr().err
    assert "POST /graph/v1/paper/batch [1 ids] -> 429" in err and "backing off" in err
    assert API_KEY not in err


def test_backoff_gives_up_after_eight_tries() -> None:
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429)

    client, clock, lim = make_client(handler)
    with pytest.raises(S2Error):
        client.paper_batch([1])
    assert len(calls) == 8 and client.stats["retries"] == 7 and client.stats["fatal_errors"] == 1
    expected = [5, 10, 20, 40, 80, 160, 160]
    for gap, base in zip(gaps(lim), expected):
        assert base <= gap <= base * 1.2 + 1e-9


def test_retry_after_header_is_honoured() -> None:
    statuses = iter([429, 200])

    def handler(req: httpx.Request) -> httpx.Response:
        st = next(statuses)
        return httpx.Response(st, headers={"Retry-After": "30"}, json=[None])

    client, _, lim = make_client(handler)
    client.paper_batch([1])
    assert gaps(lim)[0] >= 30.0


def test_transport_errors_are_retried() -> None:
    n = {"calls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        n["calls"] += 1
        if n["calls"] == 1:
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, json=[None])

    client, _, lim = make_client(handler)
    assert client.paper_batch([1]) == [None]
    assert n["calls"] == 2 and gaps(lim)[0] >= 5.0


def test_client_errors_are_not_retried() -> None:
    n = {"calls": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        n["calls"] += 1
        return httpx.Response(400, json={"error": "bad ids"})

    client, _, _ = make_client(handler)
    with pytest.raises(S2HTTPError) as ei:
        client.paper_batch([1])
    assert ei.value.status == 400 and n["calls"] == 1


def test_list_dataset_sorts_files_and_caches() -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        assert req.headers["x-api-key"] == API_KEY
        if req.url.path == "/datasets/v1/release/latest":
            return httpx.Response(200, json={"release_id": "2026-09-22", "datasets": []})
        return httpx.Response(200, json={"name": "s2orc_v2", "README": "readme",
                                         "files": ["https://s3.x/a/f_002.gz?sig=1", "https://s3.x/a/f_000.gz?sig=1",
                                                   "https://s3.x/a/f_001.gz?sig=1"]})

    client, _, _ = make_client(handler)
    listing = client.list_dataset("s2orc_v2")
    assert listing.file_names == ["f_000.gz", "f_001.gz", "f_002.gz"]
    assert listing.release_id == "2026-09-22"          # resolved via /release/latest when absent
    assert client.list_dataset("s2orc_v2") is listing
    assert client.list_dataset("s2orc_v2", "2026-09-22") is listing
    assert calls == ["/datasets/v1/release/latest/dataset/s2orc_v2", "/datasets/v1/release/latest"]
    assert url_basename("https://s3.x/a/b/c.gz?X-Amz-Signature=zzz") == "c.gz"


class _FailingStream(httpx.SyncByteStream):
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):  # type: ignore[override]
        yield self.data
        raise httpx.ReadError("connection reset")


def test_download_relists_on_403_and_resumes_partial(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    content = bytes(range(256)) * 400          # 102,400 bytes
    state = {"listings": 0, "s3": []}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.semanticscholar.org":
            state["listings"] += 1
            sig = f"SECRET{state['listings']}"
            return httpx.Response(200, json={"release_id": "r1", "README": "",
                                             "files": [f"https://s3.fake/r1/s2orc_v2/shard_000.gz?sig={sig}"]})
        assert "x-api-key" not in req.headers    # the key is never sent to S3
        sig = req.url.params["sig"]
        rng = req.headers.get("range")
        state["s3"].append((sig, rng))
        if sig == "SECRET1":
            return httpx.Response(403, text="<Error>Request has expired</Error>")
        if rng is None:                          # first full GET dies half way through
            return httpx.Response(200, headers={"content-length": str(len(content))},
                                  stream=_FailingStream(content[:40_000]))
        start = int(rng.removeprefix("bytes=").rstrip("-"))
        return httpx.Response(206, headers={"content-range": f"bytes {start}-{len(content) - 1}/{len(content)}",
                                            "content-length": str(len(content) - start)},
                              stream=httpx.ByteStream(content[start:]))

    client, _, lim = make_client(handler)
    dest = tmp_path / "r1" / "shard_000.gz"
    assert client.download_dataset_file("s2orc_v2", "latest", 0, dest) == dest
    assert dest.read_bytes() == content and not dest.with_name("shard_000.gz.part").exists()
    assert state["listings"] == 2
    assert state["s3"] == [("SECRET1", None), ("SECRET2", None), ("SECRET2", "bytes=40000-")]
    err = capsys.readouterr().err
    assert "SECRET" not in err and "sig=" not in err and API_KEY not in err
    n_requests = lim.n_slots
    assert client.download_dataset_file("s2orc_v2", "latest", 0, dest) == dest   # cached: no request
    assert lim.n_slots == n_requests
    assert all(g >= 3.0 for g in gaps(lim))


# ---------------------------------------------------------------------------------------------- end-to-end build
def _write_shard(path: Path, lines: list[Any]) -> bytes:
    with gzip.open(path, "wb") as f:
        for ln in lines:
            f.write(ln if isinstance(ln, bytes) else orjson.dumps(ln))
            f.write(b"\n")
    return path.read_bytes()


@pytest.fixture()
def fake_release(tmp_path: Path) -> dict[str, Any]:
    shard0 = [shard_rec(1, 9000), shard_rec(2, 100), shard_rec(3, 9000), shard_rec(4, 9000), shard_rec(5, 9000),
              shard_rec(6, 9000), shard_rec(1, 9000), b"{not json", shard_rec(7, 300_000), shard_rec(8, 9000),
              shard_rec(9, 9000), shard_rec(None, 9000)]
    shard1 = [shard_rec(10, 9000), shard_rec(11, 20_000), shard_rec(12, 9000)]
    metas = {1: meta(1), 3: meta(3, year=2019), 4: meta(4, fields=("Biology",)), 5: meta(5, abstract=""),
             8: meta(8, year=2024), 9: meta(9, year=None), 10: meta(10), 11: meta(11, year=2022), 12: meta(12),
             2: meta(2), 7: meta(7)}   # 6 is unknown to the API
    src = tmp_path / "src"
    src.mkdir()
    blobs = {"s2orc_v2_000.gz": _write_shard(src / "a.gz", shard0), "s2orc_v2_001.gz": _write_shard(src / "b.gz", shard1)}
    counts = {"batch_ids": [], "listings": 0, "s3": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "s3.fake":
            counts["s3"] += 1
            blob = blobs[url_basename(str(req.url))]
            return httpx.Response(200, headers={"content-length": str(len(blob))}, stream=httpx.ByteStream(blob))
        if req.url.path.startswith("/datasets/"):
            counts["listings"] += 1
            return httpx.Response(200, json={"release_id": "2026-09-22", "README": "terms",
                                             "files": [f"https://s3.fake/p/{n}?sig=x" for n in sorted(blobs)[::-1]]})
        ids = [int(i.split(":")[1]) for i in json.loads(req.content)["ids"]]
        counts["batch_ids"].append(ids)
        return httpx.Response(200, json=[metas.get(i) for i in ids])

    return {"handler": handler, "counts": counts, "tmp": tmp_path}


def _build(fr: dict[str, Any], out: Path, n: int, shards: list[int], **kw: Any) -> tuple[dict[str, Any], S2Client]:
    client, _, _ = make_client(fr["handler"])
    kw.setdefault("meta_cache_path", fr["tmp"] / "cache" / "meta.jsonl")
    stats = build_corpus(out_path=out, n_papers=n, client=client, shard_indices=shards,
                         shard_cache=fr["tmp"] / "cache", batch_size=3, **kw)
    return stats, client


def test_build_filters_dedupes_and_stops_at_n(fake_release: dict[str, Any]) -> None:
    out = fake_release["tmp"] / "corpus" / "pilot.jsonl"
    stats, client = _build(fake_release, out, 4, [0, 1])
    recs = list(iter_corpus(out))
    assert [r["corpus_id"] for r in recs] == [1, 8, 10, 11]
    assert all(list(r) == SPEC_KEYS for r in recs)
    assert {r["shard"] for r in recs} == {0, 1} and {r["release_id"] for r in recs} == {"2026-09-22"}
    s0, s1 = stats["shards"]
    c0 = s0["counts"]
    assert s0["file"] == "s2orc_v2_000.gz" and s0["exhausted"]
    assert c0["scanned"] == 12 and c0["kept"] == 2
    assert (c0["bad_record"], c0["duplicate"], c0["body_too_short"], c0["body_too_long"]) == (2, 1, 1, 1)
    assert (c0["year_too_old"], c0["not_in_field"], c0["no_abstract"], c0["not_found"], c0["year_missing"]) == \
        (1, 1, 1, 1, 1)
    assert c0["looked_up"] == 7 and c0["repeated_corpusid_lines"] == 1
    assert sum(c0[k] for k in bc.OUTCOMES) == c0["scanned"]
    c1 = s1["counts"]
    assert c1["kept"] == 2 and c1["scanned"] == 2 and not s1["exhausted"]   # stopped at N inside the shard
    assert stats["complete"] and stats["n_papers_in_output"] == 4
    assert stats["corpus"]["year_histogram"] == {2022: 1, 2023: 2, 2024: 1}
    assert json.loads(out.with_name("pilot.stats.json").read_text())["n_papers_in_output"] == 4
    listing = json.loads((fake_release["tmp"] / "cache" / "2026-09-22" / "listing.json").read_text())
    assert listing["file_names"] == ["s2orc_v2_000.gz", "s2orc_v2_001.gz"] and "sig=" not in json.dumps(listing)
    # only body-in-range, non-duplicate ids were looked up, in stream order, <= batch_size per call
    looked = [i for b in fake_release["counts"]["batch_ids"] for i in b]
    assert looked == [1, 3, 4, 5, 6, 8, 9, 10, 11, 12] and max(map(len, fake_release["counts"]["batch_ids"])) <= 3
    assert gaps(client.limiter) and all(g >= 3.0 for g in gaps(client.limiter))


def test_smoke_is_prefix_of_pilot_and_uses_meta_cache(fake_release: dict[str, Any]) -> None:
    tmp = fake_release["tmp"]
    smoke_stats, _ = _build(fake_release, tmp / "smoke.jsonl", 1, [0])
    n_calls_smoke = len(fake_release["counts"]["batch_ids"])
    _build(fake_release, tmp / "pilot.jsonl", 3, [0, 1])
    smoke = list(iter_corpus(tmp / "smoke.jsonl"))
    pilot = list(iter_corpus(tmp / "pilot.jsonl"))
    assert smoke == pilot[: len(smoke)]
    later = [i for b in fake_release["counts"]["batch_ids"][n_calls_smoke:] for i in b]
    assert 1 not in later and 3 not in later       # ids fetched by the smoke run came from the cache
    assert smoke_stats["shards"][0]["counts"]["kept"] == 1
    assert fake_release["counts"]["s3"] == 2       # shard 0 downloaded once, reused by the pilot


def test_build_resumes_without_duplicates(fake_release: dict[str, Any]) -> None:
    out = fake_release["tmp"] / "resume.jsonl"
    _build(fake_release, out, 2, [0, 1])
    with open(out, "ab") as f:
        f.write(b'{"paper_id": "s2_10", "corp')            # simulated crash mid-write
    stats, _ = _build(fake_release, out, 4, [0, 1])
    ids = [r["corpus_id"] for r in iter_corpus(out)]
    assert ids == [1, 8, 10, 11]
    c0 = stats["shards"][0]["counts"]
    assert c0["already_in_output"] == 3 and c0["kept"] == 0          # ids 1, 8 + the repeated line of id 1
    stats3, _ = _build(fake_release, out, 4, [0, 1])       # already complete: nothing to do
    assert stats3["complete"] and stats3["shards"] == [] and len(list(iter_corpus(out))) == 4
    kept_stats = json.loads(out.with_name("resume.stats.json").read_text())   # the real run's stats survive
    assert kept_stats["shards"] and kept_stats["n_papers_before_this_run"] == 2


def test_build_without_meta_cache_and_incomplete_exit_code(fake_release: dict[str, Any],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = fake_release["tmp"]
    client, _, _ = make_client(fake_release["handler"])
    monkeypatch.setattr(bc, "S2Client", lambda: client)
    rc = bc.main(["--out", str(tmp / "all.jsonl"), "--n-papers", "50", "--shard-indices", "0,1",
                  "--shard-cache", str(tmp / "cache"), "--batch-size", "3", "--no-meta-cache"])
    assert rc == 2                                          # shards exhausted before N
    assert [r["corpus_id"] for r in iter_corpus(tmp / "all.jsonl")] == [1, 8, 10, 11, 12]
    assert not (tmp / "cache" / "paper_batch_meta.jsonl").exists()
    rc = bc.main(["--out", str(tmp / "x.jsonl"), "--n-papers", "5", "--shard-indices", "7",
                  "--shard-cache", str(tmp / "cache")])
    assert rc == 1                                          # shard index out of range


def test_iter_shard_records_detects_truncation(tmp_path: Path) -> None:
    p = tmp_path / "s.gz"
    data = _write_shard(p, [shard_rec(i, 100) for i in range(200)])
    assert sum(1 for _ in iter_shard_records(p)) == 200
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(RuntimeError, match="corrupt or truncated"):
        list(iter_shard_records(p))


def test_corpus_stats_heuristics(tmp_path: Path) -> None:
    en = ("We propose a method for the analysis of graphs and we show that it is efficient in practice. " * 120)
    de = ("Wir schlagen eine Methode zur Analyse von Graphen vor und zeigen, dass sie effizient ist. " * 120)
    pt = ("uma iniciativa acadêmica focada na Infovia de Louveira, que foi apresentada ao Conselho Municipal em "
          "agosto de 2010. Ela descreve os principais componentes da infovia, incluindo a arquitetura. " * 60)
    zh = ("我们提出了一种用于图分析的方法，并证明它在实践中是有效的。" * 400)
    recs = []
    for cid, body, title in [(1, en, "Graph Methods"), (2, de, "Graph Methoden"), (3, zh, "Graph methods!"),
                             (5, pt, "Louveira Digital City"),
                             (4, "An abstract that is repeated at the start of the body text of the paper, verbatim. "
                              + en, "Other")]:
        m = meta(cid, title=title,
                 abstract="An abstract that is repeated at the start of the body text of the paper, verbatim.")
        recs.append(build_record(shard_rec(cid, 10) | {"body": {"text": body}}, m, shard=0, release_id="r"))
    p = tmp_path / "c.jsonl"
    p.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in recs))
    st = corpus_stats(p)
    assert st["n_records"] == 5 and st["year_histogram"] == {2023: 5}
    assert {x["paper_id"] for x in st["suspected_non_english"]} == {"s2_2", "s2_3", "s2_5"}
    assert st["abstract_repeated_at_body_start"] == 1
    assert st["duplicate_titles"] == ["graph methods"]
    assert st["non_english_body_start"] == [] and st["body_starts_lowercase"] == 1   # the "uma iniciativa" body
    assert st["n_body_chars"]["max"] == max(r["n_body_chars"] for r in recs)


def test_language_heuristic_edge_cases() -> None:
    from autodata.data.build_corpus import looks_non_english
    cited = ("Semiconductor quantum dots enable precise control (Botzem et al., 2018; Baart et al., 2016; "
             "Moon et al., 2020; van der Maaten et al., 2008; Volk et al., 2019). ") * 20
    assert not looks_non_english(cited, strict=False)          # "et al." / "van der" are not French/Dutch
    pt_then_en = ("uma iniciativa acadêmica focada na Infovia de Louveira, que foi apresentada ao Conselho "
                  "Municipal em agosto de 2010. " * 20) + ("The network design of the project is described in "
                                                           "this section and it is based on optical fiber. " * 200)
    assert looks_non_english(pt_then_en[:2000], strict=False)  # translated abstract at the body start
    assert not looks_non_english(pt_then_en)                   # ... but the paper itself is English
    assert looks_non_english("Результати дослідження вказують на більш ефективну операційну структуру. " * 50)
