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

- [ ] **Phase 1 and Phase 2 clean baselines** — same top_k and chunk budget in both phases,
      no temperature parameter, no per-question tuning, complete `run_meta.json`
      (protocol: `docs/EXPERIMENT_LOG.md`).
- [ ] **Full MetaQA Phase 2** — 994-doc ingest with gpt-6-luna extraction (measured: 13.1 min / 443 chunks;
      ~2.5 h and ~$3 for 4,950 chunks). Answers/bench scripts are ready (`make answers-p2 DATASET=metaqa`).
- [x] **Phase 3a — oracle graph from `kb.txt`** — typed schema, no extraction noise; upper bound
      for graph retrieval quality and the sanity check for the eval set (212/212, evidence .966).
- [ ] **Phase 3b — schema-constrained LLM extraction** — restrict extraction to the KB relation
      schema and typed entities; compare against 3a and Phase 2 (`phase3_ontology_rag/schema.py`
      is the ready schema contract).
- [ ] **Phase 1b — hybrid (dense + BM25) baseline** — second Phase 1 retrieval baseline.
- [ ] **HotpotQA controlled v2 (optional)** — retired from headline comparisons; if it returns,
      use 256/20 chunks, one prompt, same top_k/context budget, fixed judge, 500+ sampled questions.
