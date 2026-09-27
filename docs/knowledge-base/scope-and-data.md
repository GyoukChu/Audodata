# Scope and data

> Source: Deep Interview 2026-09-27 (round 1 answers)

## Key Points
- Scope now: **CS research-paper pipeline (Sec 3.1 / App. C.1) + CoT Self-Instruct baseline.**
  RL training, legal and scientific pipelines, and meta-optimization are later phases.
- Test plan: smoke test on 10-20 papers, then a ~200-paper pilot to reproduce Table 1 statistics
  (weak/strong avg, gap, rounds, question length, rubric items) for both CoT and Agentic.
  Scaling to the paper's 10k papers is a separate later decision.
- Corpus: **S2ORC via the Semantic Scholar API** (key stored in .env). Rate limit:
  **1 request per second cumulative across all endpoints** -> a single global rate limiter for
  every S2 call (metadata, bulk search, dataset listing). Target selection = CS papers, 2022+,
  as in the paper.

## Round-3 ruling (2026-09-27)
- **Corpus = `s2orc_v2` shards, streamed** (download only as many shards as needed), corpusIds looked up
  through `/graph/v1/paper/batch` (500 ids/request, >= 3 s spacing, backoff on 429) and filtered to
  s2FieldsOfStudy contains Computer Science AND year >= 2022. `paper.txt` = title + abstract (from the API)
  + body.text. Bibliography text excluded.

## Open Questions
- Exactly how to obtain S2ORC full text for CS 2022+ (Datasets API shards vs Graph API
  selection + PDF text). Paper text format for ./paper.txt.
