# Task 01 — Phase 2 (Graph RAG) Audit — UPDATED 2026-09-02

> This doc was stale (2025 values). Updated to reflect current implementation.
> Archive when Phase 2 is stable.

## Goal
Verify that `phase2_graph_rag/` ingest, retriever, and rag modules are correct, efficient, and consistent with pure Graph RAG (Neo4j only, native vector index).

## Scope
- `phase2_graph_rag/ingest.py`
- `phase2_graph_rag/retriever.py`
- `phase2_graph_rag/rag.py`

## Current Implementation (verified 2026-09-02)

### Ingest (`phase2_graph_rag/ingest.py:1`)
- [x] Batch size default 50 (`iter_document_batches:79`), `__main__` overrides to 2000 (`ingest.py:158`) — .txt files are tiny, 2000 ≈ whole corpus in one go. Finance corpus uses separate `ingest_finance.py` (LightRAG).
- [x] `EXTRACTION_LLM` now `settings.EXTRACTION_MODEL` default `gpt-5-mini` (`shared/config.py:26` / `ingest.py:24`) — not hardcoded, respects `EXTRACTION_MODEL` env var
- [x] Chunker `chunk_size=256, overlap=20` (`settings.CHUNK_SIZE_GRAPH`) — tuned for HotpotQA ~100-200 token paragraphs; 1024 would mash docs. Phase 1 uses 200/20 (hot) vs 512/64 (pdf) via shared config.
- [x] Two extractors: `ImplicitPathExtractor()` + `SimpleLLMPathExtractor(num_workers=8, max_paths_per_chunk=12)` (`ingest.py:68-76`)
- [x] `nest_asyncio.apply()` at top (`ingest.py:1-2`) for `asyncio.run`
- [x] First batch `PropertyGraphIndex.from_documents`, subsequent via `ainsert` (`ingest.py:119-135`)
- [x] Pure Graph RAG: `Neo4jPropertyGraphStore` only, vector index `chunk_embeddings` on `Chunk.embedding` cosine (`ingest.py:48-56`), plus full-text `chunk_text_fulltext` (`ingest.py:62`) — no Qdrant
- [x] Verify post-ingest `chunk_count` + `rel_count` (`ingest.py:140-152`)

### Retriever (`phase2_graph_rag/retriever.py:1`)
- [x] `top_k=5` default (`retriever.py:55`) vs Phase 1 `top_k=8` — comparable, both now documented in `shared/config.py` via `DEFAULT_TOP_K`
- [x] Hybrid: vector search (`chunk_embeddings`:`102-112`) + keyword anchor via entity extraction LLM + 1-hop `MENTIONS` traversal (`NEIGHBOR_QUERY:38-49`) — not `PropertyGraphIndex.from_existing`
- [x] Entity extraction cached per query string (`_ENTITY_CACHE:11` / `extract_entities_from_query:15`) — avoids LLM call per identical query
- [x] Keyword anchor prefers full-text index `chunk_text_fulltext`, falls back to `CONTAINS` scan (`retriever.py:118-140`)
- [x] No similarity threshold (Phase 1 has `node_threshold=0.3` default, `0.4` in `__main__`) — Phase 2 returns top-k regardless; consider threshold for parity
- [x] Preflight checks for Neo4j (`retriever.py:77-95`) + empty-input guards
- [x] Deduplication + `[Relevance: score]` annotation + neighbor texts (`retriever.py:148-184`)

### RAG (`phase2_graph_rag/rag.py:1`)
- [x] `build_graph_prompt()` hides triplets, tells LLM to use graph silently (`rag.py:5-29`) — finance variant still TODO (`plans/02-prompt-variants.md`)
- [x] Uses `shared.llm.generate_completion` with retry/backoff (`shared/llm.py:5-31`) and Maritaca fallback
- [x] Early-exit if no chunks (`rag.py:39-44`) — returns "I don't know"
- [x] Return `{"answer": ..., "contexts": ...}` matches RAGAS expectations

## Verification (manual — no test framework)
- [ ] Run `uv run python -m phase2_graph_rag.ingest` on small batch (`batch_size=2`) and check Neo4j `MATCH (c:Chunk) RETURN count(c)`
- [ ] Run `uv run python -m phase2_graph_rag.retriever` with `"Who is the younger brother..."` and inspect chunks
- [ ] Run `uv run python -m phase2_graph_rag.rag` and verify answer
- Or via Make: `make ingest-graph`, `make retrieve-p2`, `make rag-p2`, `make docker-up`

## Notes
- Isolation: Phase 1 `rag_phase_1_baseline` (Qdrant, `shared/config.py:28`) vs Phase 2 Neo4j database `hotpot-graph` (`NEO4J_DATABASE` default `shared/config.py:19`). Old doc's `rag_phase_2_graph`/`QDRANT_COLLECTION_NAME_PHASE2` no longer exists.
- Phase 2 ingest targets `data_hotpot/` (not `data/`); `data/` only has `HP-1.pdf` for manual QA. `ingest_finance.py` targets `data_finance/` via LightRAG.
- Python 3.11 required (`pyproject.toml:4` `<3.13`) — 3.13 breaks `ragas`+`qdrant-client` numpy pins; system has 3.13.5 but `.venv` is 3.11
- See `notes/TODO.md:61-72` + `notes/improvements.md` for remaining hygiene (threshold parity, prompt variants)
