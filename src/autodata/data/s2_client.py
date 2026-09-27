"""Semantic Scholar (S2) client: dataset listing, ``/paper/batch`` metadata, dataset shard downloads.

HARD RULE (docs/IMPLEMENTATION_SPEC.md §2.5; the API key is a shared lab key): every request made by this module --
API calls *and* S3 shard downloads -- goes through ONE process-wide :class:`RateLimiter`:

* at least ``S2_MIN_INTERVAL`` seconds (default 3.0, never below 3.0) between the END of one request and the START
  of the next (stricter than start-to-start spacing; 429s were observed at 1.3 s spacing, 3 s with backoff worked);
* never two requests in flight: the limiter slot is held for the whole request, including a streamed download;
* exponential backoff on 429 / 5xx / transport errors: 5, 10, 20, 40, 80, 160 s (+ up to 20 % jitter, or
  ``Retry-After`` if longer), at most 8 tries. The backoff is a *global* cooldown -- nobody sends until it expires;
* best effort, processes on the same host are serialized as well through an ``flock``-ed state file
  (``S2_LIMITER_LOCK``, default ``/tmp/autodata_s2_limiter_<uid>.lock``; set it to an empty string to disable).

Each request is logged to stderr as one line (method, endpoint, status, elapsed). The API key and the presigned
S3 query strings (temporary credentials) are never logged or written anywhere.
"""
from __future__ import annotations

import contextlib
import fcntl
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import urlsplit

import httpx

API_BASE = "https://api.semanticscholar.org"
PAPER_BATCH_PATH = "/graph/v1/paper/batch"
PAPER_BATCH_FIELDS = "corpusId,title,abstract,year,publicationDate,s2FieldsOfStudy,externalIds,venue,citationCount"
MAX_BATCH_IDS = 500                      # /paper/batch hard limit
DEFAULT_MIN_INTERVAL = 3.0               # seconds between any two requests (S2_MIN_INTERVAL)
MIN_INTERVAL_FLOOR = 3.0                 # required spacing for the shared key across all endpoints
BACKOFF_SCHEDULE: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
MAX_TRIES = 8
JITTER_FRACTION = 0.2
MAX_BACKOFF = 900.0
FILE_GATE_POLL_S = 30.0                 # bounded waits recheck the shared deadline


def log(msg: str) -> None:
    print(f"[s2 {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class S2Error(RuntimeError):
    """Semantic Scholar request failed (after retries, or with a non-retryable status)."""


class S2HTTPError(S2Error):
    def __init__(self, status: int, what: str, detail: str = "") -> None:
        super().__init__(f"{what}: HTTP {status}" + (f" ({detail})" if detail else ""))
        self.status = status


# --------------------------------------------------------------------------------------------------------------
# rate limiter
# --------------------------------------------------------------------------------------------------------------
class LimiterSlot:
    """Handle yielded by :meth:`RateLimiter.slot`; set ``cooldown`` (seconds) to hold back every later request."""

    __slots__ = ("start", "cooldown")

    def __init__(self, start: float) -> None:
        self.start = start
        self.cooldown = 0.0


class RateLimiter:
    """Spacing of >= ``min_interval`` s from the end of a request to the start of the next, no overlap.

    ``clock``/``sleep`` are injectable so tests can drive the limiter with a fake clock. ``lock_path`` enables the
    inter-process gate (a file holding the wall-clock time before which no process may start a request).
    """

    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, lock_path: str | os.PathLike[str] | None = None,
                 wall_clock: Callable[[], float] = time.time) -> None:
        min_interval = float(min_interval)
        if not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("min_interval must be finite and nonnegative")
        self.min_interval = max(MIN_INTERVAL_FLOOR, min_interval)
        self._clock, self._sleep, self._wall = clock, sleep, wall_clock
        self._lock = threading.Lock()
        self._next_allowed: float | None = None   # in `clock` time
        self._lock_path = Path(lock_path) if lock_path else None
        self._file_gate_ok = True
        self.n_slots = 0
        self.total_wait_s = 0.0
        self.history: list[tuple[float, float]] = []   # (start, end) of the most recent slots, for tests/diagnostics

    @contextlib.contextmanager
    def slot(self) -> Iterator[LimiterSlot]:
        with self._lock, self._file_gate() as fd:
            self._wait_turn(fd)
            handle = LimiterSlot(self._clock())
            try:
                yield handle
            finally:
                end = self._clock()
                gap = max(self.min_interval, float(handle.cooldown or 0.0))
                self._next_allowed = end + gap
                self.n_slots += 1
                self.history.append((handle.start, end))
                if len(self.history) > 2000:
                    del self.history[:1000]
                if fd is not None:
                    self._write_file_deadline(fd, self._wall() + gap)

    def _wait_turn(self, fd: int | None) -> None:
        while True:
            wait = 0.0
            if self._next_allowed is not None:
                wait = self._next_allowed - self._clock()
            if fd is not None:
                deadline = self._read_file_deadline(fd)
                if deadline is not None:
                    file_wait = deadline - self._wall()
                    if math.isfinite(file_wait) and file_wait <= max(MAX_BACKOFF, self.min_interval) + 0.001:
                        wait = max(wait, file_wait)
            if wait <= 0:
                return
            wait = min(wait, FILE_GATE_POLL_S)
            self.total_wait_s += wait
            self._sleep(wait)

    @contextlib.contextmanager
    def _file_gate(self) -> Iterator[int | None]:
        fd: int | None = None
        if self._lock_path is not None and self._file_gate_ok:
            try:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError as e:
                if fd is not None:
                    os.close(fd)
                    fd = None
                self._file_gate_ok = False
                log(f"warning: inter-process limiter lock unavailable ({type(e).__name__}); in-process limiter only")
        try:
            yield fd
        finally:
            if fd is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    @staticmethod
    def _read_file_deadline(fd: int) -> float | None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 64).decode("ascii", "replace").strip()
            return float(raw) if raw else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_file_deadline(fd: int, deadline: float) -> None:
        with contextlib.suppress(OSError):
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, f"{deadline:.3f}\n".encode("ascii"))


def min_interval_from_env() -> float:
    raw = os.environ.get("S2_MIN_INTERVAL", "").strip()
    value = DEFAULT_MIN_INTERVAL
    if raw:
        try:
            value = float(raw)
        except ValueError:
            log(f"warning: S2_MIN_INTERVAL={raw!r} is not a number; using {DEFAULT_MIN_INTERVAL}")
    if not math.isfinite(value) or value < 0:
        raise ValueError("S2_MIN_INTERVAL must be finite and nonnegative")
    if value < MIN_INTERVAL_FLOOR:
        log(f"warning: S2_MIN_INTERVAL={value} is below the {MIN_INTERVAL_FLOOR} s floor; using {MIN_INTERVAL_FLOOR}")
        value = MIN_INTERVAL_FLOOR
    return value


def default_lock_path() -> Path | None:
    raw = os.environ.get("S2_LIMITER_LOCK")
    if raw is not None:
        return Path(raw) if raw.strip() else None
    return Path("/tmp") / f"autodata_s2_limiter_{os.getuid()}.lock"


_GLOBAL_LIMITER: RateLimiter | None = None
_GLOBAL_LIMITER_LOCK = threading.Lock()


def get_global_limiter() -> RateLimiter:
    """The one process-wide limiter (the only global state allowed by the coding standards)."""
    global _GLOBAL_LIMITER
    with _GLOBAL_LIMITER_LOCK:
        if _GLOBAL_LIMITER is None:
            _GLOBAL_LIMITER = RateLimiter(min_interval_from_env(), lock_path=default_lock_path())
        return _GLOBAL_LIMITER


# --------------------------------------------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------------------------------------------
def url_basename(url: str) -> str:
    """File name of a (presigned) URL, without the query string (which holds temporary credentials)."""
    return urlsplit(url).path.rsplit("/", 1)[-1]


@dataclass
class DatasetListing:
    dataset: str
    release_id: str
    readme: str
    files: list[str]      # presigned S3 URLs sorted by file name; shard index = position. NEVER log these.
    listed_at: float
    api_order: list[str] = field(default_factory=list)   # file names in the order the API returned them

    @property
    def file_names(self) -> list[str]:
        return [url_basename(u) for u in self.files]


_CONTENT_RANGE_RE = re.compile(r"bytes\s+(?:(\d+)-(\d+)|\*)/(\d+|\*)")


def _content_range_total(value: str | None) -> int | None:
    m = _CONTENT_RANGE_RE.match(value or "")
    if not m or m.group(3) == "*":
        return None
    return int(m.group(3))


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


class S2Client:
    """Synchronous Semantic Scholar client. All requests go through ``limiter`` (default: the global one)."""

    def __init__(self, api_key: str | None = None, *, limiter: RateLimiter | None = None,
                 http: httpx.Client | None = None, max_tries: int = MAX_TRIES,
                 backoff: Sequence[float] = BACKOFF_SCHEDULE, rng: random.Random | None = None,
                 timeout_s: float = 120.0, api_base: str = API_BASE) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("S2_API_KEY", "")
        if not self._api_key:
            log("warning: S2_API_KEY is not set; the datasets API needs it (source env.sh)")
        self.limiter = limiter if limiter is not None else get_global_limiter()
        self._http = http if http is not None else httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0))
        self._own_http = http is None
        self.max_tries = max(1, int(max_tries))
        self.backoff = tuple(float(b) for b in backoff) or BACKOFF_SCHEDULE
        self._rng = rng if rng is not None else random.Random()
        self.api_base = api_base.rstrip("/")
        self._listings: dict[tuple[str, str], DatasetListing] = {}
        self.stats: Counter[str] = Counter()

    # ---------------------------------------------------------------- plumbing
    def close(self) -> None:
        if self._own_http:
            self._http.close()

    def __enter__(self) -> "S2Client":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def backoff_delay(self, attempt: int, response: httpx.Response | None = None) -> float:
        """Delay after failed try number ``attempt`` (1-based): schedule + jitter, or Retry-After if longer."""
        base = self.backoff[min(attempt - 1, len(self.backoff) - 1)]
        delay = base + self._rng.uniform(0.0, JITTER_FRACTION * base)
        if response is not None:
            retry_after = response.headers.get("retry-after", "").strip()
            if retry_after.replace(".", "", 1).isdigit():
                delay = max(delay, min(float(retry_after), MAX_BACKOFF))
        return delay

    @staticmethod
    def _retryable(status: int | None, err: BaseException | None) -> bool:
        return err is not None or status == 429 or (status is not None and status >= 500)

    def _record(self, status: int | None, err: BaseException | None) -> None:
        self.stats["requests"] += 1
        if err is not None:
            self.stats[f"error:{type(err).__name__}"] += 1
        else:
            self.stats[f"status:{status}"] += 1

    def _request(self, method: str, url: str, *, what: str, api: bool = True, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", None) or {})
        if api and self._api_key:
            headers["x-api-key"] = self._api_key
        for attempt in range(1, self.max_tries + 1):
            resp: httpx.Response | None = None
            err: BaseException | None = None
            delay: float | None = None
            with self.limiter.slot() as slot:
                t0 = time.monotonic()
                try:
                    resp = self._http.request(method, url, headers=headers, **kwargs)
                except httpx.TransportError as e:
                    err = e
                elapsed = time.monotonic() - t0
                status = resp.status_code if resp is not None else None
                retryable = self._retryable(status, err)
                if retryable and attempt < self.max_tries:
                    delay = self.backoff_delay(attempt, resp)
                    slot.cooldown = delay
                self._record(status, err)
                outcome = str(status) if err is None else f"{type(err).__name__}"
                line = f"{what} -> {outcome} in {elapsed:.2f}s (try {attempt}/{self.max_tries})"
                if delay is not None:
                    line += f"; backing off {delay:.1f}s"
                log(line)
            if not retryable:
                assert resp is not None
                if resp.status_code >= 400:
                    self.stats["fatal_errors"] += 1
                    raise S2HTTPError(resp.status_code, what, resp.text[:300].replace("\n", " "))
                return resp
            if attempt < self.max_tries:
                self.stats["retries"] += 1
        self.stats["fatal_errors"] += 1
        raise S2Error(f"{what}: giving up after {self.max_tries} tries")

    # ---------------------------------------------------------------- datasets API
    def list_dataset(self, dataset: str = "s2orc_v2", release: str = "latest", *,
                     force: bool = False) -> DatasetListing:
        """``GET /datasets/v1/release/<release>/dataset/<dataset>`` (cached per process unless ``force``)."""
        key = (dataset, release)
        if not force and key in self._listings:
            return self._listings[key]
        path = f"/datasets/v1/release/{release}/dataset/{dataset}"
        data = self._request("GET", self.api_base + path, what=f"GET {path}").json()
        if not isinstance(data, dict) or not isinstance(data.get("files"), list):
            raise S2Error(f"GET {path}: unexpected response (no 'files' list)")
        release_id = str(data.get("release_id") or "")
        if not release_id:
            release_id = release if release != "latest" else self.latest_release_id()
        listing = DatasetListing(dataset=dataset, release_id=release_id, readme=str(data.get("README") or ""),
                                 files=sorted((str(u) for u in data["files"]), key=url_basename),
                                 listed_at=time.time(), api_order=[url_basename(str(u)) for u in data["files"]])
        for k in (key, (dataset, release_id)):
            prev = self._listings.get(k)
            if prev is not None and prev.release_id != release_id:
                raise S2Error(f"{dataset}: release changed while running ({prev.release_id} -> {release_id}); "
                              f"pin it with --release {prev.release_id}")
            self._listings[k] = listing
        return listing

    def latest_release_id(self) -> str:
        path = "/datasets/v1/release/latest"
        data = self._request("GET", self.api_base + path, what=f"GET {path}").json()
        return str(data.get("release_id") or "latest")

    # ---------------------------------------------------------------- graph API
    def paper_batch(self, corpus_ids: Sequence[int], *,
                    fields: str = PAPER_BATCH_FIELDS) -> list[dict[str, Any] | None]:
        """``POST /graph/v1/paper/batch`` for up to 500 corpus ids; result aligned with the input (None = unknown)."""
        ids = [int(c) for c in corpus_ids]
        if not ids:
            return []
        if len(ids) > MAX_BATCH_IDS:
            raise ValueError(f"paper_batch takes at most {MAX_BATCH_IDS} ids, got {len(ids)}")
        resp = self._request("POST", self.api_base + PAPER_BATCH_PATH, what=f"POST {PAPER_BATCH_PATH} [{len(ids)} ids]",
                             params={"fields": fields}, json={"ids": [f"CorpusId:{i}" for i in ids]})
        try:
            data = resp.json()
        except ValueError as e:
            raise S2Error(f"{PAPER_BATCH_PATH}: invalid JSON response ({e})") from e
        if not isinstance(data, list) or len(data) != len(ids):
            got = len(data) if isinstance(data, list) else type(data).__name__
            raise S2Error(f"{PAPER_BATCH_PATH}: expected a list of {len(ids)} entries, got {got}")
        return [d if isinstance(d, dict) else None for d in data]

    # ---------------------------------------------------------------- S3 shard download
    def download_dataset_file(self, dataset: str, release: str, index: int, dest: str | os.PathLike[str], *,
                              max_relists: int = 3) -> Path:
        """Download file ``index`` of the dataset listing to ``dest`` (reused if it exists).

        Streams into ``<dest>.part`` and renames atomically when the size matches; a partial ``.part`` is resumed with
        an HTTP Range request. Presigned URLs expire: a 403 triggers a re-listing (at most ``max_relists`` times).
        """
        dest = Path(dest)
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        relists = 0
        attempt = 0
        while True:
            listing = self.list_dataset(dataset, release)
            if not 0 <= index < len(listing.files):
                raise IndexError(f"{dataset} release {listing.release_id} has {len(listing.files)} files; no index {index}")
            url = listing.files[index]
            name = url_basename(url)
            attempt += 1
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            status: int | None = None
            err: BaseException | None = None
            total: int | None = None
            written = 0
            delay: float | None = None
            with self.limiter.slot() as slot:
                t0 = time.monotonic()
                try:
                    with self._http.stream("GET", url, headers=headers) as r:
                        status = r.status_code
                        if status in (200, 206):
                            if status == 200:
                                offset = 0           # full body (Range ignored or not sent): start over
                                cl = r.headers.get("content-length")
                                total = int(cl) if cl and cl.isdigit() else None
                            else:
                                total = _content_range_total(r.headers.get("content-range"))
                            with open(part, "ab" if offset else "wb") as f:
                                for chunk in r.iter_raw():   # as received: a dropped connection keeps its bytes
                                    f.write(chunk)
                                    written += len(chunk)
                        elif status == 416:
                            total = _content_range_total(r.headers.get("content-range"))
                        else:
                            r.read()
                except httpx.TransportError as e:
                    err = e
                elapsed = time.monotonic() - t0
                size_now = part.stat().st_size if part.exists() else 0
                ok_status = status in (200, 206) or (status == 416 and total is not None)
                complete = err is None and ok_status and (size_now == total if total is not None else status == 200)
                retryable = not complete and status != 403 and (
                    err is not None or status in (200, 206, 416) or self._retryable(status, None))
                if retryable and attempt < self.max_tries:
                    delay = self.backoff_delay(attempt)
                    slot.cooldown = delay
                self._record(status, err)
                outcome = str(status) if err is None else type(err).__name__
                rate = written / elapsed / 1e6 if elapsed > 0 else 0.0
                line = (f"GET s3:{dataset}/{name} (shard {index}, from byte {offset}) -> {outcome} in {elapsed:.1f}s, "
                        f"{_fmt_bytes(written)} at {rate:.1f} MB/s"
                        + (f", total {_fmt_bytes(total)}" if total is not None else "")
                        + f" (try {attempt}/{self.max_tries})")
                if delay is not None:
                    line += f"; backing off {delay:.1f}s"
                log(line)
            if complete:
                part.replace(dest)
                return dest
            if status == 416 and err is None and part.exists():
                log(f"s3:{dataset}/{name}: partial file does not match the remote size; restarting from byte 0")
                part.unlink()
            if status == 403 and err is None:
                relists += 1
                if relists > max_relists:
                    self.stats["fatal_errors"] += 1
                    raise S2HTTPError(403, f"GET s3:{dataset}/{name}", "still forbidden after re-listing")
                log(f"presigned URL rejected (403): re-listing {dataset} ({relists}/{max_relists})")
                self.list_dataset(dataset, release, force=True)
                attempt -= 1   # an expired URL is not a failed try
                continue
            if not retryable or attempt >= self.max_tries:
                self.stats["fatal_errors"] += 1
                raise S2Error(f"GET s3:{dataset}/{name}: download failed (status={status}, error={err!r}, "
                              f"{size_now} bytes on disk)")
            self.stats["retries"] += 1
