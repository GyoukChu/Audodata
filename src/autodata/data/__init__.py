"""S2ORC corpus: Semantic Scholar client (``s2_client``) and corpus builder (``build_corpus``).

Public helpers for the rest of the pipeline::

    from autodata.data import iter_corpus, paper_text
    for rec in iter_corpus("data/corpus/cs2022_pilot.jsonl"):
        text = paper_text(rec)   # "Title: ...\n\nAbstract: ...\n\n<body_text>"  -> ./paper.txt
"""
from autodata.data.build_corpus import CorpusRecord, iter_corpus, paper_text

__all__ = ["CorpusRecord", "iter_corpus", "paper_text"]
