import json

import pytest

from autodata.cs import corpus_io
from autodata.cs.corpus_io import iter_corpus_records, load_papers, record_to_paper_text, safe_paper_id
from autodata.data import iter_corpus, paper_text


@pytest.mark.parametrize("raw,expected", [
    ("../../paper/id", "paper_id"), ("...", "paper"), ("", "paper"),
    (".hidden", "hidden"), ("a b\\c", "a_b_c"), (123, "123"), ("a" * 121, "a" * 120),
])
def test_safe_paper_id(raw, expected):
    result = safe_paper_id(raw)
    assert result == expected
    assert "/" not in result and "\\" not in result and not result.startswith((".", "_"))


def corpus(tmp_path, records):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n" + "\n\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_canonical_data_helpers_and_no_fallback(tmp_path, monkeypatch):
    record = {"paper_id": "p", "title": " title ", "abstract": None, "body_text": "body", "text": "unused"}
    path = corpus(tmp_path, [record])
    assert list(iter_corpus_records(path)) == list(iter_corpus(path)) == [record]
    assert record_to_paper_text(record) == paper_text(record)
    assert record_to_paper_text({"text": "fallback must not be used"}) == paper_text({})
    monkeypatch.setattr(corpus_io, "paper_text", lambda rec: (_ for _ in ()).throw(ValueError("canonical error")))
    with pytest.raises(ValueError, match="canonical error"):
        record_to_paper_text(record)


def test_dedupe_including_sanitized_id_collisions_and_metadata(tmp_path):
    records = [{"paper_id": "a/b", "title": "first", "body_text": "body", "year": 2025},
               {"paper_id": "a/b", "title": "duplicate"}, {"paper_id": "a_b", "title": "collision"},
               {"corpus_id": 42, "body_text": "other"}]
    papers = load_papers(corpus(tmp_path, records))
    assert [paper.paper_id for paper in papers] == ["a_b", "s2_42"]
    assert papers[0].title == "first" and papers[0].meta == {"year": 2025}
    assert papers[0].text == paper_text(records[0])


@pytest.mark.parametrize("options,expected", [
    ({"offset": 2, "limit": 2}, ["p2", "p3"]),
    ({"paper_ids": {"p1", "p3"}, "limit": 1}, ["p1"]),
    ({"offset": 2, "paper_ids": {"p1", "p3"}}, ["p3"]),
    ({"offset": 20}, []), ({"limit": 0}, []),
])
def test_selection_offset_limit_and_paper_ids(tmp_path, options, expected):
    path = corpus(tmp_path, [{"paper_id": f"p{i}", "body_text": "body"} for i in range(5)])
    assert [paper.paper_id for paper in load_papers(path, **options)] == expected


@pytest.mark.parametrize("selection", [{"raw/id"}, {"raw_id"}])
def test_paper_ids_accept_original_or_safe_id(tmp_path, selection):
    path = corpus(tmp_path, [{"paper_id": "raw/id", "body_text": "body"}])
    assert [paper.paper_id for paper in load_papers(path, paper_ids=selection)] == ["raw_id"]


def test_min_chars_filter_then_limit(tmp_path, capsys):
    records = [{"paper_id": "short", "body_text": "x"}, {"paper_id": "long", "body_text": "x" * 100}]
    path = corpus(tmp_path, records)
    papers = load_papers(path, min_chars=len(paper_text(records[1])), limit=1)
    assert [paper.paper_id for paper in papers] == ["long"]
    assert "skipped 1 papers" in capsys.readouterr().out


def test_canonical_reader_error_is_not_swallowed(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("not JSON")
    with pytest.raises(ValueError, match="malformed JSON line"):
        load_papers(path)
