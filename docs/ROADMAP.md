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

## Next

- [ ] **Mini dataset (30 questions, 100 docs)** — scale `scripts/metaqa_make_mini.py` from the
      12-question/50-doc pilot (`--per-hop 10 --docs 100`) and re-freeze it
      (`MetaQA/FROZEN_mini.sha256`).
- [ ] **Phase 1 and Phase 2 clean baselines** — same top_k and chunk budget in both phases,
      similarity threshold off on MetaQA, temperature 0, no per-question tuning
      (protocol: `docs/EXPERIMENT_LOG.md`).
- [ ] **Closed-book control** — the same questions and models without retrieval; the floor the
      phases must beat.
- [ ] **Phase 3a — oracle graph from `kb.txt`** — build the MetaQA graph directly from the KB
      triples (no extraction noise); upper bound for graph retrieval quality.
- [ ] **Phase 3b — schema-constrained LLM extraction** — restrict LLM extraction to the KB
      relation schema and compare against 3a and Phase 2.
- [ ] **Phase 1b — hybrid (dense + BM25) baseline** — sparse + dense fusion as a second
      Phase 1 retrieval baseline before the graph comparison.
