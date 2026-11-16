# Roadmap — RAG Trilogy

> Replaces the old `notes/TODO.md`. Archived notes and task plans live in `docs/archive/`
> (`improvements.md`, `plans/`).

## Done

- **Phase 1 — Vector RAG on HotpotQA**: ingest, retrieval, full RAG pipeline, 100-question
  answer generation and the complete benchmark (EM, Relaxed EM, Extracted EM, Token F1,
  Semantic Similarity, LLM-as-Judge, RAGAS).
- **Phase 2 — Graph RAG on HotpotQA**: pure Neo4j (native vector index + graph), implicit +
  LLM extractors, retrieval, answer generation and the same benchmark.
- **MetaQA v1 frozen**: evidence-gated eval set (212 questions) and corpus (994 docs),
  checksummed in `MetaQA/FROZEN.sha256`; 240 sampled → 212 kept.
- **Dataset-agnostic harness**: `shared/config.py::dataset_config(phase)` switches eval file,
  corpus, Qdrant collection, Neo4j database, chunk sizes and results dir
  (`hotpot` | `metaqa` | `metaqa_mini`); `run_meta.json` and `per_question.jsonl` record
  what actually ran.
- **Phase 1 retrieval check**: `scripts/metaqa_retrieval_check.py` scores evidence / answer /
  triple recall at k = 8/20/50 with no generation.
- **Phase 2 50-doc pilot**: mini-corpus ingest/retrieval smoke run with metadata scrubbing
  and isolated stores.
- **Phase 2 mini30 certificate**: 100 docs / 443 chunks / 0 extraction errors, 30/30 answers,
  full metric stack; the recorded `--no-neighbours` ablation shows the 1-hop neighbours add
  +18 gold recall and +11 evidence recall points on mini30 after ranking them by shared entities.
- **Full MetaQA baselines**: Phase 1 (212/212) and Phase 3a oracle graph (212/212, evidence
  recall .966, gold recall .876) are complete; Phase 2 full is pending a time/budget call.
- **Metric corrections (2026-11-12)**: judge parser fixed (previous judge columns were refusal
  rates), Extracted EM fixed for yes/no, RAGAS Answer Correctness removed, judge separated from
  the generator, Luna models used with no temperature parameter, neighbours ranked by shared
  entities, full-text/CONTAINS provenance recorded.
- **Mini30 track (30 q / 100 docs)**: additive plumbing corpus
  (`MetaQA/metaqa_eval_mini30.jsonl` + `corpus_manifest_mini30.txt`, checksummed) with its own
  collection, graph DB and results dir; the 12/50 mini stays frozen and runnable.
- **Operational gates**: run fingerprints (resume refuses a changed config), strict error
  accounting, explicit `--reset`, bounded Phase 2 concurrency, isolated PDF collection, offline
  `make test` suite, and a trackable `DATASET=local` for arbitrary TXT/PDF corpora.
- **Closed-book control**: `shared/closed_book.py` + `make answers-cb/bench-cb` (plumbing in
  place; a full 212-question control run is still pending).

## Next

- [x] **Phase 1 and Phase 2 clean baselines** — same top_k/chunk budget, no temperature,
      complete `run_meta.json`; full Phase 2 now exists.
- [x] **Full MetaQA Phase 2** — 4,522 chunks / 127,797 relationships, 212/212 answers,
      0 errors; gold .757 / evidence .841 / all_found .656. ~1.5–2 h ingest.
- [x] **Phase 3a — oracle graph from `kb.txt`** — typed schema, no extraction noise; upper bound
      for graph retrieval quality and the sanity check for the eval set (212/212, evidence .966).
- [x] **Phase 3b — schema-constrained LLM extraction** — full 212 done: 4,522 chunks /
      26,617 relationships, 0 parse/extraction errors; .779 gold / .893 evidence / .684 all_found.
      vs Phase 2: evidence +.052 (CI excludes 0), answer metrics tied; 3a remains the ceiling.
- [x] **Paired statistics on every comparison** — `scripts/compare_runs.py` (exact McNemar +
      bootstrap CI); full 212 used for the P1/P2/3b/3a claims.
- [ ] **Phase 1b — hybrid (dense + BM25) baseline** — second Phase 1 retrieval baseline.
- [ ] **HotpotQA controlled v2 (optional)** — retired from headline comparisons; if it returns,
      use 256/20 chunks, one prompt, same top_k/context budget, fixed judge, 500+ sampled questions.
