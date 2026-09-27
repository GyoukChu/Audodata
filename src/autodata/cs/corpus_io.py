"""Corpus loading shared by the pipeline, the CoT baseline and the stats (using the canonical data module)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from autodata.cs.run_paper import PaperInput
from autodata.data import iter_corpus, paper_text


def record_to_paper_text(rec: dict[str, Any]) -> str:
    """Compatibility wrapper for the canonical corpus rendering."""
    return paper_text(rec)


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_paper_id(raw: Any) -> str:
    """Turn any id into a single safe directory component (no separators, no . / .. / hidden names)."""
    s = _SAFE_ID_RE.sub("_", str(raw)).strip("._")
    if not s:
        s = "paper"
    return s[:120]


def iter_corpus_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Compatibility wrapper for the canonical JSONL reader."""
    yield from iter_corpus(path)


def load_papers(path: str | Path, *, limit: int | None = None, offset: int = 0,
                paper_ids: set[str] | None = None, min_chars: int = 0) -> list[PaperInput]:
    if limit is not None and limit <= 0:
        return []
    papers: list[PaperInput] = []
    skipped_short = 0
    seen: set[str] = set()
    for i, rec in enumerate(iter_corpus_records(path)):
        pid = safe_paper_id(rec.get("paper_id") or f"s2_{rec.get('corpus_id', i)}")
        if paper_ids and pid not in paper_ids and str(rec.get("paper_id")) not in paper_ids:
            continue
        if pid in seen:
            continue
        seen.add(pid)
        if i < offset:
            continue
        text = record_to_paper_text(rec)
        if len(text) < min_chars:
            skipped_short += 1
            continue
        meta = {k: rec.get(k) for k in ("corpus_id", "year", "publication_date", "venue", "s2_fields", "external_ids",
                                         "n_body_chars", "shard", "release_id") if k in rec}
        papers.append(PaperInput(paper_id=str(pid), title=rec.get("title") or "", text=text, meta=meta))
        if limit is not None and len(papers) >= limit:
            break
    if skipped_short:
        print(f"[corpus] skipped {skipped_short} papers shorter than {min_chars} chars", flush=True)
    return papers
