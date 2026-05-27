# Task 01 — Phase 2 (Graph RAG) Audit

## Goal
Verify that `phase2_graph_rag/` ingest, retriever, and rag modules are correct, efficient, and consistent with the intended hybrid vector + property graph approach.

## Scope
- `phase2_graph_rag/ingest.py`
- `phase2_graph_rag/retriever.py`
- `phase2_graph_rag/rag.py`

## Implementation Steps

### Ingest (`ingest.py`)
- [ ] Verify batch size default (20) and confirm it's appropriate for the finance corpus
- [ ] Confirm `EXTRACTION_LLM` is `gpt-4o-mini` (cheap, fast) regardless of `GENERATION_MODEL`
- [ ] Check chunker settings: `chunk_size=1024, chunk_overlap=128` — validate these are optimal
- [ ] Verify two-extractor strategy: `ImplicitPathExtractor` (free) + `SimpleLLMPathExtractor` (8 workers)
- [ ] Confirm `nest_asyncio.apply()` is at module top
- [ ] Check that first batch creates the index, subsequent batches upsert via `ainsert`
- [ ] Verify both stores connect: Neo4j (`Neo4jPropertyGraphStore`) + Qdrant (`QdrantVectorStore` with `QDRANT_COLLECTION_NAME_PHASE2`)

### Retriever (`retriever.py`)
- [ ] Verify `top_k=3` default — is this enough for finance queries?
- [ ] Confirm `PropertyGraphIndex.from_existing()` loads without re-ingestion
- [ ] Check that retriever uses `index.as_retriever(similarity_top_k=top_k)` (auto VectorContextRetriever)
- [ ] Verify `get_content()` returns text + triplets correctly
- [ ] No similarity threshold applied — consider if one is needed

### RAG (`rag.py`)
- [ ] Verify `build_hybrid_prompt()` includes graph relationship instructions
- [ ] Confirm it uses `shared.llm.generate_completion` (litellm with Maritaca fallback)
- [ ] Check early-exit behavior if no chunks retrieved (currently no guard — consider adding)
- [ ] Verify return format matches RAGAS expectations: `{"answer": ..., "contexts": ...}`

## Verification
- Run `python -m phase2_graph_rag.ingest` on a small batch (edit `batch_size=2`) and check Neo4j + Qdrant
- Run `python -m phase2_graph_rag.retriever` with a test query and inspect output chunks
- Run `python -m phase2_graph_rag.rag` and verify answer quality

## Notes
- Phase 2 uses collection `rag_phase_2_graph` (isolated from Phase 1's `rag_phase_1_baseline`)
- Phase 2 ingest targets `data/` directory, not `data_finance/`
- No test framework — manual verification only
