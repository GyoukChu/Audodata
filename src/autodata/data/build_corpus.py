"""Build a CS 2022+ paper corpus from Semantic Scholar ``s2orc_v2`` shards (docs/IMPLEMENTATION_SPEC.md §2.5).

Pipeline (deterministic for a given release + shard order):
  1. list the dataset (1 API call; presigned shard URLs, sorted by file name -> shard index = position);
  2. download each chosen shard once into ``--shard-cache/<release_id>/`` (reused later; Range-resumable);
  3. stream the shard's JSONL records in file order; local checks first (corpus-id dedupe, body length);
  4. look up the survivors with ``/paper/batch`` (<= 500 ids per call, through the global S2 limiter; results are cached
     on disk in ``<shard-cache>/paper_batch_meta.jsonl`` so re-runs and resumes do not repeat calls);
  5. keep ``year >= min_year`` AND ``field in s2FieldsOfStudy categories`` AND non-empty abstract; append records to
     the output JSONL in stream order until N papers are in the file (append-only; ids already present are skipped,
     so an interrupted run resumes where it stopped).
Progress and per-shard counts go to stderr; a ``<out>.stats.json`` summary is written at the end.

CLI: ``autodata-build-corpus --out data/corpus/<name>.jsonl --n-papers N --shard-indices 0,1 ...`` (see ``--help``).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import orjson
from pydantic import BaseModel, ConfigDict

from autodata.data.s2_client import MAX_BATCH_IDS, DatasetListing, S2Client, S2Error

DEFAULT_DATASET = "s2orc_v2"
DEFAULT_SHARD_CACHE = Path("data/s2orc_v2_shards")
META_CACHE_NAME = "paper_batch_meta.jsonl"

# tally outcomes, in pipeline order (every scanned record gets exactly one)
LOCAL_OUTCOMES = ("bad_record", "already_in_output", "duplicate", "body_too_short", "body_too_long")
META_OUTCOMES = ("not_found", "year_missing", "year_too_old", "not_in_field", "no_abstract")
OUTCOMES = LOCAL_OUTCOMES + META_OUTCOMES + ("kept",)


def log(msg: str) -> None:
    print(f"[build {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------------------------
# corpus record + readers
# --------------------------------------------------------------------------------------------------------------
class CorpusRecord(BaseModel):
    """One line of ``data/corpus/<name>.jsonl`` (field set and order fixed by spec §2.5)."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str                    # "s2_<corpusid>"
    corpus_id: int
    title: str
    abstract: str
    year: int
    publication_date: str | None     # "YYYY-MM-DD" (API publicationDate) or None
    venue: str                       # API venue ("" when unknown)
    s2_fields: list[str]             # unique s2FieldsOfStudy categories, API order
    external_ids: dict[str, Any]     # API externalIds (fallback: shard openaccessinfo.externalids)
    authors: list[str]               # shard author names
    body_text: str                   # shard body.text, verbatim (bibliography excluded)
    n_body_chars: int                # len(body_text)
    shard: int                       # shard index in the release listing (sorted by file name)
    release_id: str


def paper_text(record: Mapping[str, Any] | CorpusRecord) -> str:
    """The ``./paper.txt`` text for one corpus record: title + abstract (from the API) + body text."""
    if isinstance(record, CorpusRecord):
        title, abstract, body = record.title, record.abstract, record.body_text
    else:
        title, abstract, body = record.get("title"), record.get("abstract"), record.get("body_text")
    return f"Title: {title or ''}\n\nAbstract: {abstract or ''}\n\n{body or ''}"


def iter_corpus(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the records of a corpus JSONL file in order (blank lines skipped; a malformed line raises)."""
    with open(path, "rb") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: malformed JSON line ({e})") from e


def load_existing_ids(path: Path) -> list[int]:
    """Corpus ids already in ``path`` (for resuming). A torn last line from an interrupted write is truncated away."""
    if not path.exists():
        return []
    ids: list[int] = []
    good_end = 0
    torn = False
    with open(path, "rb") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                try:
                    ids.append(int(orjson.loads(stripped)["corpus_id"]))
                except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
                    if line.endswith(b"\n"):
                        raise ValueError(f"{path}: malformed record at byte {good_end}; refusing to append to it")
                    torn = True
                    break
            good_end += len(line)
    if torn:
        log(f"{path}: dropping a torn last line ({path.stat().st_size - good_end} bytes) before resuming")
        with open(path, "r+b") as f:
            f.truncate(good_end)
    elif good_end:
        with open(path, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
    return ids


# --------------------------------------------------------------------------------------------------------------
# filtering + record building
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterConfig:
    min_year: int = 2022
    field: str = "Computer Science"
    min_body_chars: int = 8_000
    max_body_chars: int = 200_000
    # "non-empty abstract": the API returns placeholders such as ",", "." or "Graphical abstract" for some papers,
    # so an abstract counts only with >= this many characters after stripping (1 = literally non-empty).
    min_abstract_chars: int = 50


def has_abstract(meta: Mapping[str, Any] | None, filt: FilterConfig) -> bool:
    abstract = (meta or {}).get("abstract")
    return isinstance(abstract, str) and len(abstract.strip()) >= max(1, filt.min_abstract_chars)


def body_outcome(n_chars: int, filt: FilterConfig) -> str | None:
    """None if ``min_body_chars <= n_chars <= max_body_chars``, else the rejection reason."""
    if n_chars < filt.min_body_chars:
        return "body_too_short"
    if n_chars > filt.max_body_chars:
        return "body_too_long"
    return None


def field_categories(meta: Mapping[str, Any] | None) -> list[str]:
    out: list[str] = []
    for f in (meta or {}).get("s2FieldsOfStudy") or []:
        cat = f.get("category") if isinstance(f, Mapping) else None
        if isinstance(cat, str) and cat and cat not in out:
            out.append(cat)
    return out


def metadata_outcome(meta: Mapping[str, Any] | None, filt: FilterConfig) -> str | None:
    """None if the /paper/batch metadata passes (year, field, abstract), else the first failing check."""
    if not meta:
        return "not_found"
    year = meta.get("year")
    if not isinstance(year, int) or isinstance(year, bool):
        return "year_missing"
    if year < filt.min_year:
        return "year_too_old"
    if filt.field not in field_categories(meta):
        return "not_in_field"
    if not has_abstract(meta, filt):
        return "no_abstract"
    return None


def shard_body_text(rec: Mapping[str, Any]) -> str:
    body = rec.get("body")
    text = body.get("text") if isinstance(body, Mapping) else None
    return text if isinstance(text, str) else ""


def normalize_authors(raw: Any) -> list[str]:
    out: list[str] = []
    for a in raw if isinstance(raw, list) else []:
        if isinstance(a, str):
            name = a
        elif isinstance(a, Mapping):
            name = a.get("name") or " ".join(
                str(p) for p in (a.get("first"), *(a.get("middle") or []), a.get("last")) if p)
        else:
            continue
        name = " ".join(str(name).split())
        if name:
            out.append(name)
    return out


def _one_line(s: Any) -> str:
    return " ".join(s.split()) if isinstance(s, str) else ""


def _shard_external_ids(rec: Mapping[str, Any]) -> dict[str, Any]:
    oa = rec.get("openaccessinfo")
    ext = oa.get("externalids") if isinstance(oa, Mapping) else None
    return {k: v for k, v in ext.items() if v not in (None, "")} if isinstance(ext, Mapping) else {}


def build_record(rec: Mapping[str, Any], meta: Mapping[str, Any], *, shard: int, release_id: str) -> dict[str, Any]:
    """Merge a shard record (body, authors) with its /paper/batch metadata into a :class:`CorpusRecord` dict."""
    cid = int(rec["corpusid"])
    body = shard_body_text(rec)
    api_ext = meta.get("externalIds")
    ext = {k: v for k, v in api_ext.items() if v not in (None, "")} if isinstance(api_ext, Mapping) else {}
    if not ext:
        ext = _shard_external_ids(rec)
    record = CorpusRecord(
        paper_id=f"s2_{cid}",
        corpus_id=cid,
        title=_one_line(meta.get("title")) or _one_line(rec.get("title")),
        abstract=str(meta.get("abstract") or "").strip(),
        year=int(meta["year"]),
        publication_date=meta.get("publicationDate") or None,
        venue=_one_line(meta.get("venue")),
        s2_fields=field_categories(meta),
        external_ids=ext,
        authors=normalize_authors(rec.get("authors")),
        body_text=body,
        n_body_chars=len(body),
        shard=int(shard),
        release_id=str(release_id),
    )
    return record.model_dump()


def _slim(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only what build_record needs (drops bibliography/annotations while a record waits for its metadata)."""
    oa = rec.get("openaccessinfo")
    return {"corpusid": rec.get("corpusid"), "title": rec.get("title"), "authors": rec.get("authors"),
            "openaccessinfo": {"externalids": oa.get("externalids")} if isinstance(oa, Mapping) else None,
            "body": {"text": shard_body_text(rec)}}


# --------------------------------------------------------------------------------------------------------------
# shards
# --------------------------------------------------------------------------------------------------------------
def iter_shard_records(path: Path) -> Iterator[dict[str, Any] | None]:
    """Stream a gzipped JSONL shard in file order; yields None for an unparseable line."""
    try:
        with gzip.open(path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = orjson.loads(line)
                except orjson.JSONDecodeError:
                    yield None
                    continue
                yield obj if isinstance(obj, dict) else None
    except (EOFError, zlib.error, gzip.BadGzipFile) as e:
        raise RuntimeError(f"{path}: corrupt or truncated shard ({type(e).__name__}: {e}); delete it and re-run "
                           f"(the build resumes from the records already written)") from e


def parse_shard_indices(spec: str) -> list[int]:
    """'0,1' / '0-3' / '0,2-4' -> ordered unique shard indices."""
    out: list[int] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"bad shard range {part!r}")
            vals = range(lo, hi + 1)
        else:
            vals = range(int(part), int(part) + 1)
        for v in vals:
            if v < 0:
                raise ValueError(f"negative shard index in {spec!r}")
            if v not in out:
                out.append(v)
    if not out:
        raise ValueError(f"no shard indices in {spec!r}")
    return out


def resolve_shards(n_files: int, indices: Sequence[int] | None = None, shard_seed: int | None = None,
                   n_shards: int | None = None) -> list[int]:
    """Explicit indices, or ``n_shards`` distinct indices drawn with ``random.Random(shard_seed)``."""
    if indices is not None and (shard_seed is not None or n_shards is not None):
        raise ValueError("use either --shard-indices or --shard-seed/--n-shards, not both")
    if shard_seed is not None or n_shards is not None:
        k = 1 if n_shards is None else int(n_shards)
        if not 1 <= k <= n_files:
            raise ValueError(f"--n-shards must be in [1, {n_files}]")
        chosen = random.Random(0 if shard_seed is None else shard_seed).sample(range(n_files), k)
    else:
        chosen = list(indices) if indices is not None else [0]
    bad = [i for i in chosen if not 0 <= i < n_files]
    if bad:
        raise ValueError(f"shard indices {bad} out of range: the release has {n_files} files")
    return chosen


class MetaCache:
    """Append-only on-disk cache of /paper/batch results keyed by corpus id (None = the API did not know the id)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._d: dict[int, dict[str, Any] | None] = {}
        if path is not None and path.exists():
            with open(path, "rb") as f:
                for line in f:
                    try:
                        obj = orjson.loads(line)
                        self._d[int(obj["corpus_id"])] = obj.get("meta")
                    except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue   # torn line from an interrupted run
        self.loaded = len(self._d)

    def __contains__(self, cid: int) -> bool:
        return cid in self._d

    def __len__(self) -> int:
        return len(self._d)

    def get(self, cid: int) -> dict[str, Any] | None:
        return self._d.get(cid)

    def put_many(self, items: Mapping[int, dict[str, Any] | None]) -> None:
        self._d.update(items)
        if self.path is None or not items:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as f:
            f.write(b"".join(orjson.dumps({"corpus_id": cid, "meta": m}) + b"\n" for cid, m in items.items()))


# --------------------------------------------------------------------------------------------------------------
# stats helpers
# --------------------------------------------------------------------------------------------------------------
def percentiles(values: Sequence[float], qs: Sequence[float] = (0, 10, 25, 50, 75, 90, 100)) -> dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)
    out: dict[str, float] = {}
    for q in qs:
        pos = (len(xs) - 1) * q / 100
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        val = xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
        out["min" if q == 0 else "max" if q == 100 else f"p{q:g}"] = round(val, 1)
    out["mean"] = round(sum(xs) / len(xs), 1)
    return out


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


# function words that are (nearly) exclusive to English vs. to Portuguese/Spanish/French/German/Italian/Dutch
_EN_FUNCTION_WORDS = frozenset(
    "the of and to is for that with on are by this be from which we our these was were has have been it its an or "
    "can".split())
# (no "et"/"van"/"der"/"von": they occur in English citations; no "y"/"do"/"em"/"com": variables, "EM", ".com")
_OTHER_FUNCTION_WORDS = frozenset(
    "de la le les des el los las en que del da das dos para um uma não une est dans pour sur und ist mit zu nicht "
    "het een il di che della".split())
_WORD_RE = re.compile(r"\w+")


def function_word_ratios(text: str, max_chars: int = 20_000) -> tuple[float, float]:
    """(English, other-language) function-word shares of the word tokens (English prose: ~0.3 vs ~0.01)."""
    words = [w for w in _WORD_RE.findall(text[:max_chars].lower()) if not w.isdigit()]
    if not words:
        return 0.0, 0.0
    return (sum(w in _EN_FUNCTION_WORDS for w in words) / len(words),
            sum(w in _OTHER_FUNCTION_WORDS for w in words) / len(words))


def english_stopword_ratio(text: str, max_chars: int = 20_000) -> float:
    return function_word_ratios(text, max_chars)[0]


def looks_non_english(text: str, *, strict: bool = True) -> bool:
    """Other-language function words dominate; ``strict`` also flags few English ones / many non-ASCII letters."""
    en, other = function_word_ratios(text)
    if other > en:
        return True
    return strict and (en < 0.12 or ascii_letter_share(text) < 0.9)


def ascii_letter_share(text: str, max_chars: int = 20_000) -> float:
    letters = [c for c in text[:max_chars] if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 0.0


def alpha_share(text: str) -> float:
    return sum(c.isalpha() for c in text) / len(text) if text else 0.0


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def corpus_stats(path: str | Path, *, filt: FilterConfig | None = None) -> dict[str, Any]:
    """Corpus-level statistics + data-quality heuristics over an output JSONL file (streams it; no network)."""
    filt = filt or FilterConfig()
    ids: Counter = Counter()
    titles: Counter = Counter()
    years: Counter = Counter()
    shards: Counter = Counter()
    fields: Counter = Counter()
    ext_types: Counter = Counter()
    venues: Counter = Counter()
    release_ids: set[str] = set()
    body_chars: list[int] = []
    text_chars: list[int] = []
    abstract_chars: list[int] = []
    n_authors: list[int] = []
    non_english: list[dict[str, Any]] = []
    non_english_start: list[dict[str, Any]] = []
    low_alpha: list[dict[str, Any]] = []
    abstract_in_body: list[str] = []
    lowercase_start: list[str] = []
    only_target = empty_venue = missing_date = no_authors = 0
    for r in iter_corpus(path):
        body = r["body_text"]
        ids[r["corpus_id"]] += 1
        titles[_norm_title(r["title"])] += 1
        years[r["year"]] += 1
        shards[r["shard"]] += 1
        fields.update(r["s2_fields"])
        ext_types.update(r["external_ids"].keys())
        venues[r["venue"] or "<empty>"] += 1
        release_ids.add(r["release_id"])
        body_chars.append(r["n_body_chars"])
        text_chars.append(len(paper_text(r)))
        abstract_chars.append(len(r["abstract"]))
        n_authors.append(len(r["authors"]))
        only_target += r["s2_fields"] == [filt.field]
        empty_venue += not r["venue"]
        missing_date += not r["publication_date"]
        no_authors += not r["authors"]
        if looks_non_english(body):
            en, other = function_word_ratios(body)
            non_english.append({"paper_id": r["paper_id"], "title": r["title"][:100], "en_function_words": round(en, 3),
                                "other_function_words": round(other, 3),
                                "ascii_letter_share": round(ascii_letter_share(body), 3), "body_start": body[:120]})
        elif looks_non_english(body[:2000], strict=False):   # e.g. translated abstracts before an English body
            non_english_start.append({"paper_id": r["paper_id"], "title": r["title"][:100],
                                      "body_start": body[:120]})
        al = alpha_share(body)
        if al < 0.6:
            low_alpha.append({"paper_id": r["paper_id"], "title": r["title"][:100], "alpha_share": round(al, 3)})
        first = body.lstrip()[:1]
        if first.isalpha() and first.islower():   # the parse lost the start of the text (begins mid-sentence)
            lowercase_start.append(r["paper_id"])
        head = _squash(r["abstract"])[:150]
        if len(head) >= 50 and head in _squash(body[:6000]):
            abstract_in_body.append(r["paper_id"])
    return {
        "n_records": sum(ids.values()),
        "n_unique_corpus_ids": len(ids),
        "duplicate_corpus_ids": sorted(i for i, c in ids.items() if c > 1),
        "duplicate_titles": sorted(t for t, c in titles.items() if c > 1 and t),
        "year_histogram": dict(sorted(years.items())),
        "shard_histogram": dict(sorted(shards.items())),
        "release_ids": sorted(release_ids),
        "n_body_chars": percentiles(body_chars),
        "paper_text_chars": percentiles(text_chars),
        "abstract_chars": percentiles(abstract_chars),
        "n_authors": percentiles(n_authors),
        "s2_fields": dict(fields.most_common()),
        "only_field_is_target": only_target,
        "external_id_types": dict(ext_types.most_common()),
        "top_venues": dict(venues.most_common(15)),
        "empty_venue": empty_venue,
        "missing_publication_date": missing_date,
        "no_authors": no_authors,
        "suspected_non_english": non_english,
        "non_english_body_start": non_english_start,
        "low_alpha_share_bodies": low_alpha,
        "body_starts_lowercase": len(lowercase_start),
        "body_starts_lowercase_examples": lowercase_start[:10],
        "abstract_repeated_at_body_start": len(abstract_in_body),
        "abstract_repeated_examples": abstract_in_body[:10],
    }


# --------------------------------------------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------------------------------------------
@dataclass
class _Entry:
    cid: int | None
    rec: dict[str, Any] | None      # slimmed shard record, only while waiting for metadata
    outcome: str | None             # None = waiting for /paper/batch
    body_len: int
    repeated: bool = False          # corpus id already seen on an earlier line of the stream


@dataclass
class _ShardStats:
    index: int
    file: str
    counters: Counter = field(default_factory=Counter)
    body_lens: list[int] = field(default_factory=list)
    years_looked_up: Counter = field(default_factory=Counter)
    exhausted: bool = False
    elapsed_s: float = 0.0

    def tally(self, e: _Entry) -> None:
        self.counters["scanned"] += 1
        self.counters[e.outcome or "?"] += 1
        if e.repeated:
            self.counters["repeated_corpusid_lines"] += 1
        if e.outcome != "bad_record":
            self.body_lens.append(e.body_len)

    def to_dict(self, filt: FilterConfig) -> dict[str, Any]:
        c = self.counters
        scanned, looked = c["scanned"], c["looked_up"]
        candidates = scanned - c["bad_record"] - c["already_in_output"] - c["duplicate"]
        lens = self.body_lens
        return {
            "index": self.index,
            "file": self.file,
            "exhausted": self.exhausted,
            "elapsed_s": round(self.elapsed_s, 1),
            "counts": {k: c[k] for k in ("scanned",) + OUTCOMES + (
                "looked_up", "meta_found", "meta_year_ok", "meta_in_field", "meta_year_ok_and_in_field",
                "meta_has_abstract", "repeated_corpusid_lines", "corpus_id_mismatch", "api_batch_calls",
                "meta_from_api", "meta_from_cache")},
            "rates": {
                "kept_per_scanned": _rate(c["kept"], scanned),
                "body_in_range_per_candidate": _rate(looked, candidates),
                "found_per_looked_up": _rate(c["meta_found"], looked),
                "year_ok_per_looked_up": _rate(c["meta_year_ok"], looked),
                "in_field_per_looked_up": _rate(c["meta_in_field"], looked),
                "year_ok_and_in_field_per_looked_up": _rate(c["meta_year_ok_and_in_field"], looked),
                "has_abstract_per_looked_up": _rate(c["meta_has_abstract"], looked),
                "kept_per_looked_up": _rate(c["kept"], looked),
            },
            "body_chars_all_scanned": {
                **percentiles(lens),
                "n": len(lens),
                "empty": sum(n == 0 for n in lens),
                "below_min": sum(n < filt.min_body_chars for n in lens),
                "above_max": sum(n > filt.max_body_chars for n in lens),
            },
            "year_histogram_looked_up": {str(k): v for k, v in sorted(self.years_looked_up.items(), key=str)},
        }


class _Build:
    def __init__(self, *, client: S2Client, out_path: Path, n_papers: int, filt: FilterConfig, shard_cache: Path,
                 dataset: str, release: str, batch_size: int, meta_cache_path: Path | None) -> None:
        if n_papers <= 0:
            raise ValueError("--n-papers must be positive")
        if not 1 <= batch_size <= MAX_BATCH_IDS:
            raise ValueError(f"batch size must be in [1, {MAX_BATCH_IDS}]")
        self.client, self.out_path, self.n_papers, self.filt = client, out_path, n_papers, filt
        self.shard_cache, self.dataset, self.release, self.batch_size = shard_cache, dataset, release, batch_size
        self.cache = MetaCache(meta_cache_path)
        self.shards: list[_ShardStats] = []
        self.existing_ids: set[int] = set()
        self.claimed: set[int] = set()     # ids in the output or waiting in the current window
        self.seen: set[int] = set()        # every id read so far (diagnostics only)
        self.n_total = 0
        self.release_id = ""
        self.listing: DatasetListing | None = None

    def save_listing(self, listing: DatasetListing) -> None:
        """Keep the release README + file names (never the presigned URLs) next to the cached shards."""
        path = self.shard_cache / listing.release_id / "listing.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dataset": listing.dataset, "release_id": listing.release_id,
                                    "listed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(listing.listed_at)),
                                    "n_files": len(listing.files), "file_names": listing.file_names,
                                    "api_order_is_sorted": listing.api_order == listing.file_names,
                                    "README": listing.readme}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    def ensure_shard(self, shard: int) -> Path:
        assert self.listing is not None
        name = self.listing.file_names[shard]
        dest = self.shard_cache / self.release_id / name
        if dest.exists():
            log(f"shard {shard}: using cached {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
            return dest
        log(f"shard {shard}: downloading {name} -> {dest}")
        return self.client.download_dataset_file(self.dataset, self.release_id, shard, dest)

    def classify(self, rec: dict[str, Any] | None) -> _Entry:
        if rec is None:
            return _Entry(None, None, "bad_record", 0)
        raw = rec.get("corpusid")
        try:
            if isinstance(raw, bool):
                raise TypeError
            cid = int(raw)
        except (TypeError, ValueError):
            return _Entry(None, None, "bad_record", 0)
        body_len = len(shard_body_text(rec))
        repeated = cid in self.seen
        self.seen.add(cid)
        if cid in self.existing_ids:
            return _Entry(cid, None, "already_in_output", body_len, repeated)
        if cid in self.claimed:
            return _Entry(cid, None, "duplicate", body_len, repeated)
        reason = body_outcome(body_len, self.filt)
        if reason is not None:
            return _Entry(cid, None, reason, body_len, repeated)
        self.claimed.add(cid)
        return _Entry(cid, _slim(rec), None, body_len, repeated)

    def metadata(self, ids: list[int], st: _ShardStats) -> dict[int, dict[str, Any] | None]:
        missing = [i for i in ids if i not in self.cache]
        st.counters["meta_from_cache"] += len(ids) - len(missing)
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start:start + self.batch_size]
            result = self.client.paper_batch(chunk)
            st.counters["api_batch_calls"] += 1
            st.counters["meta_from_api"] += len(chunk)
            got: dict[int, dict[str, Any] | None] = {}
            for cid, meta in zip(chunk, result):
                if meta is not None and meta.get("corpusId") not in (None, cid):
                    st.counters["corpus_id_mismatch"] += 1
                got[cid] = meta
            self.cache.put_many(got)
        return {i: self.cache.get(i) for i in ids}

    def count_marginals(self, meta: Mapping[str, Any] | None, st: _ShardStats) -> None:
        st.counters["looked_up"] += 1
        if not meta:
            return
        st.counters["meta_found"] += 1
        year = meta.get("year")
        st.years_looked_up[year if isinstance(year, int) else "none"] += 1
        year_ok = isinstance(year, int) and year >= self.filt.min_year
        in_field = self.filt.field in field_categories(meta)
        st.counters["meta_year_ok"] += year_ok
        st.counters["meta_in_field"] += in_field
        st.counters["meta_year_ok_and_in_field"] += year_ok and in_field
        st.counters["meta_has_abstract"] += has_abstract(meta, self.filt)

    def flush(self, window: list[_Entry], st: _ShardStats, out_f: Any) -> bool:
        """Resolve the window's metadata and tally it in stream order; True once N papers are in the output."""
        metas = self.metadata([e.cid for e in window if e.outcome is None and e.cid is not None], st)
        done = False
        for e in window:
            meta = None
            if e.outcome is None:
                meta = metas.get(e.cid)  # type: ignore[arg-type]
                self.count_marginals(meta, st)
                e.outcome = metadata_outcome(meta, self.filt) or "kept"
            st.tally(e)
            if e.outcome == "kept":
                assert e.rec is not None and meta is not None
                record = build_record(e.rec, meta, shard=st.index, release_id=self.release_id)
                out_f.write(orjson.dumps(record) + b"\n")
                out_f.flush()
                self.n_total += 1
                if self.n_total >= self.n_papers:
                    done = True
                    break
            e.rec = None
        c = st.counters
        log(f"shard {st.index}: scanned {c['scanned']:,} | body in range {c['looked_up']:,} "
            f"(api {c['meta_from_api']:,} in {c['api_batch_calls']} calls, cache {c['meta_from_cache']:,}) | "
            f"CS&{self.filt.min_year}+ {c['meta_year_ok_and_in_field']:,} | kept {c['kept']:,} | "
            f"corpus {self.n_total}/{self.n_papers}")
        return done

    def process_shard(self, shard: int, out_f: Any) -> bool:
        assert self.listing is not None
        st = _ShardStats(index=shard, file=self.listing.file_names[shard])
        self.shards.append(st)
        path = self.ensure_shard(shard)
        t0 = time.monotonic()
        window: list[_Entry] = []
        n_pending = n_read = 0
        done = False
        try:
            for rec in iter_shard_records(path):
                n_read += 1
                entry = self.classify(rec)
                window.append(entry)
                if entry.outcome is None:
                    n_pending += 1
                    if n_pending >= self.batch_size:
                        done = self.flush(window, st, out_f)
                        window, n_pending = [], 0
                        if done:
                            break
            else:
                if window:
                    done = self.flush(window, st, out_f)
                st.exhausted = st.counters["scanned"] == n_read   # every record of the shard was tallied
        finally:
            st.elapsed_s = time.monotonic() - t0
        log(f"shard {shard}: {'reached the target' if done else 'exhausted'}; "
            f"kept {st.counters['kept']:,} of {st.counters['scanned']:,} scanned records")
        return done

    def run(self, *, shard_indices: Sequence[int] | None, shard_seed: int | None,
            n_shards: int | None) -> dict[str, Any]:
        started = time.time()
        listing = self.client.list_dataset(self.dataset, self.release)
        self.listing, self.release_id = listing, listing.release_id
        chosen = resolve_shards(len(listing.files), shard_indices, shard_seed, n_shards)
        self.save_listing(listing)
        log(f"{self.dataset} release {self.release_id}: {len(listing.files)} files; shards {chosen}; "
            f"target {self.n_papers} papers -> {self.out_path}")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_existing_ids(self.out_path)
        self.existing_ids = set(existing)
        self.claimed = set(existing)
        self.n_total = len(self.existing_ids)
        if self.n_total:
            log(f"resuming: {self.n_total} papers already in {self.out_path}")
        if self.cache.path is not None:
            log(f"metadata cache {self.cache.path}: {self.cache.loaded:,} corpus ids")
        done = self.n_total >= self.n_papers
        with open(self.out_path, "ab") as out_f:
            for shard in chosen:
                if done:
                    break
                done = self.process_shard(shard, out_f)
        if not done:
            log(f"WARNING: only {self.n_total}/{self.n_papers} papers after shards {chosen}; add more shards")
        stats = self.summary(chosen, started)
        stats["n_papers_before_this_run"] = len(existing)
        stats_path = self.out_path.with_name(self.out_path.name.removesuffix(".jsonl") + ".stats.json")
        if self.shards or not stats_path.exists():
            stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            log(f"wrote {stats_path}")
        else:
            log(f"output already complete; kept the existing {stats_path}")
        return stats

    def summary(self, chosen: Sequence[int], started: float) -> dict[str, Any]:
        assert self.listing is not None
        per_shard = [s.to_dict(self.filt) for s in self.shards]
        tot: Counter = Counter()
        for s in self.shards:
            tot.update(s.counters)
        lim = self.client.limiter
        return {
            "out": str(self.out_path),
            "dataset": self.dataset,
            "release_id": self.release_id,
            "n_release_files": len(self.listing.files),
            "shard_indices": list(chosen),
            "shard_files": {str(i): self.listing.file_names[i] for i in chosen},
            "n_papers_target": self.n_papers,
            "n_papers_in_output": self.n_total,
            "complete": self.n_total >= self.n_papers,
            "filters": asdict(self.filt),
            "batch_size": self.batch_size,
            "meta_cache": str(self.cache.path) if self.cache.path else None,
            "shards": per_shard,
            "totals": {k: v for k, v in sorted(tot.items())},
            "api_requests": dict(sorted(self.client.stats.items())),
            "limiter": {"min_interval_s": lim.min_interval, "requests": lim.n_slots,
                        "total_wait_s": round(lim.total_wait_s, 1)},
            "corpus": corpus_stats(self.out_path, filt=self.filt),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
            "elapsed_s": round(time.time() - started, 1),
            "argv": sys.argv[1:],
        }


def build_corpus(*, out_path: str | Path, n_papers: int, client: S2Client, filt: FilterConfig | None = None,
                 shard_indices: Sequence[int] | None = None, shard_seed: int | None = None,
                 n_shards: int | None = None, shard_cache: str | Path = DEFAULT_SHARD_CACHE,
                 dataset: str = DEFAULT_DATASET, release: str = "latest", batch_size: int = MAX_BATCH_IDS,
                 meta_cache_path: str | Path | None = None) -> dict[str, Any]:
    """Build (or resume) a corpus JSONL; returns the stats dict (also written to ``<out>.stats.json``)."""
    b = _Build(client=client, out_path=Path(out_path), n_papers=n_papers, filt=filt or FilterConfig(),
               shard_cache=Path(shard_cache), dataset=dataset, release=release, batch_size=batch_size,
               meta_cache_path=Path(meta_cache_path) if meta_cache_path else None)
    return b.run(shard_indices=shard_indices, shard_seed=shard_seed, n_shards=n_shards)


# --------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autodata-build-corpus", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output JSONL (appended to; resumable)")
    p.add_argument("--n-papers", type=int, default=None,
                   help="stop once the output holds this many papers (required unless --stats-only)")
    p.add_argument("--shard-indices", default=None,
                   help="shards to stream, in order, e.g. '0,1' or '0-3' (None: shard 0, or --shard-seed)")
    p.add_argument("--shard-seed", type=int, default=None, help="pick --n-shards random shards with this seed")
    p.add_argument("--n-shards", type=int, default=None, help="number of random shards (with --shard-seed)")
    p.add_argument("--min-year", type=int, default=2022, help="keep papers with API year >= this")
    p.add_argument("--field", default="Computer Science", help="required s2FieldsOfStudy category")
    p.add_argument("--min-body-chars", type=int, default=8_000, help="minimum len(body.text)")
    p.add_argument("--max-body-chars", type=int, default=200_000, help="maximum len(body.text)")
    p.add_argument("--min-abstract-chars", type=int, default=50,
                   help="an abstract counts as non-empty with >= this many chars after stripping (the API has "
                        "placeholders like ',' or 'Graphical abstract'); 1 = literally non-empty")
    p.add_argument("--shard-cache", type=Path, default=DEFAULT_SHARD_CACHE,
                   help="shard download cache (files go to <cache>/<release_id>/)")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Semantic Scholar dataset name")
    p.add_argument("--release", default="latest", help="dataset release id, e.g. 2026-09-22")
    p.add_argument("--batch-size", type=int, default=MAX_BATCH_IDS, help="corpus ids per /paper/batch call")
    p.add_argument("--meta-cache", type=Path, default=None,
                   help=f"/paper/batch result cache (None: <shard-cache>/{META_CACHE_NAME})")
    p.add_argument("--no-meta-cache", action="store_true", help="do not read or write the metadata cache")
    p.add_argument("--stats-only", action="store_true",
                   help="print corpus statistics of an existing --out file and exit (no network)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    filt = FilterConfig(min_year=args.min_year, field=args.field, min_body_chars=args.min_body_chars,
                        max_body_chars=args.max_body_chars, min_abstract_chars=args.min_abstract_chars)
    if args.stats_only:
        print(json.dumps(corpus_stats(args.out, filt=filt), indent=2, ensure_ascii=False))
        return 0
    if args.n_papers is None:
        print("error: --n-papers is required", file=sys.stderr)
        return 1
    try:
        indices = parse_shard_indices(args.shard_indices) if args.shard_indices is not None else None
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    meta_cache = None if args.no_meta_cache else (args.meta_cache or args.shard_cache / META_CACHE_NAME)
    try:
        with S2Client() as client:
            stats = build_corpus(out_path=args.out, n_papers=args.n_papers, client=client, filt=filt,
                                 shard_indices=indices, shard_seed=args.shard_seed, n_shards=args.n_shards,
                                 shard_cache=args.shard_cache, dataset=args.dataset, release=args.release,
                                 batch_size=args.batch_size, meta_cache_path=meta_cache)
    except (S2Error, ValueError, IndexError, RuntimeError, OSError) as e:
        log(f"error: {type(e).__name__}: {e}")
        return 1
    log(f"done: {stats['n_papers_in_output']}/{stats['n_papers_target']} papers in {stats['out']} "
        f"(release {stats['release_id']}, {stats['elapsed_s']} s, {stats['limiter']['requests']} requests)")
    return 0 if stats["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
