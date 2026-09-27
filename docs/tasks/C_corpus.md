Build the S2ORC corpus module and the actual smoke/pilot corpora. Project: <repo>
(run `cd` there, `source env.sh` (exports S2_API_KEY, HF vars), `source .venv/bin/activate`). Read docs/IMPLEMENTATION_SPEC.md
section 2.5 (your contract) and the data notes in docs/knowledge-base/scope-and-data.md.
You own ONLY: src/autodata/data/s2_client.py, src/autodata/data/build_corpus.py, src/autodata/data/__init__.py (export paper_text,
iter_corpus), tests/test_data.py, data/ (outputs). Do not modify other files; do not install packages.

Hard rules for the Semantic Scholar API (the key is a shared lab key):
- ONE global limiter: >= S2_MIN_INTERVAL seconds (default 3.0) between any two requests, no concurrency, exponential backoff on 429
  (5,10,20,40,80,160 s + jitter, max 8 tries), log each request line to stderr (endpoint, status, elapsed). Verified facts: at 1.3 s
  spacing the API already returned 429s; 3 s spacing with backoff worked.
- Endpoints: GET https://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc_v2 (header x-api-key) -> {release_id, README,
  files: [presigned S3 URLs, ~329 .gz JSONL shards]}; presigned URLs expire, re-list when a download gets 403.
  POST https://api.semanticscholar.org/graph/v1/paper/batch?fields=corpusId,title,abstract,year,publicationDate,s2FieldsOfStudy,externalIds,venue,citationCount
  with JSON body {"ids": ["CorpusId:123", ...]} (max 500 ids). Shard record schema (verified on the real data): corpusid (int), title,
  authors [str], openaccessinfo{externalids{DOI,ArXiv,ACL,MAG,PubMedCentral,...}, license, url, status}, body{text, annotations{paragraph,
  section_header, bib_ref}}, bibliography{text, annotations}. Shard download needs no API key (S3). Body text median ~25k chars.
- Filter: year >= 2022 AND "Computer Science" in [f["category"] for f in s2FieldsOfStudy] AND abstract non-empty AND
  min_body_chars <= len(body.text) <= max_body_chars. Dedupe by corpus id. Keep the shard order deterministic (--shard-indices).
- Corpus JSONL record + paper_text() exactly as spec 2.5. paper_text = f"Title: {title}\n\nAbstract: {abstract}\n\n{body_text}".
- Streaming: decompress the .gz while downloading (or download the shard file to data/s2orc_v2_shards/ once and reuse); process records
  in batches of 500 corpus ids -> one /paper/batch call per batch; stop when n_papers collected; resumable.

Then RUN IT (this is required, real network, respect the limiter):
1. data/corpus/cs2022_smoke.jsonl with --n-papers 24 --shard-indices 0
2. data/corpus/cs2022_pilot.jsonl with --n-papers 320 --shard-indices 0,1 (add more shards only if needed)
   (the smoke set must be a subset of / disjoint from pilot? -> make the pilot the first 320 and the smoke the first 24 of the SAME
   stream so smoke ⊂ pilot; document this).
3. Write data/corpus/README.md: release_id, shards used, counts scanned/kept per shard, filter rates, year histogram, body-length stats,
   and the exact commands. Also tests/test_data.py (unit tests of filtering, record building, limiter timing with a fake clock; no network).
Report: what you built, the exact commands, counts, any API errors seen, and anything surprising about the data (e.g. papers with
mostly-empty bodies, non-English, duplicates).
