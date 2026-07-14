# Improvement Notes — RAG Trilogy

> Points for future improvement, organized by priority and area.

## Code Quality

### Deduplication
- `phase1/ingest_pdf.py` and `phase1/ingest_hot.py` are ~90% identical — only file extension, chunk size, and target directory differ. Extract to `shared/ingest.py` parameterized by config.
- Benchmark and hotpot_answers already deduplicated into `shared/benchmark.py` and `shared/hotpot_runner.py`.

### Dead Code
- `shared/config.py`: `get_qdrant_client()` and `get_openai_client()` are defined but never called. Either use them to centralize client creation or remove them.
- `shared/config.py`: `EMBEDDING_DIMENSION` is now used in `ingest.py` but could be used more widely.
- `shared/llm.py`: `max_tokens` is commented out — consider enabling it to control response length and cost.

### Error Handling
- **No preflight checks**: No "is Qdrant running?" / "is Neo4j running?" checks. Connection failures produce opaque stack traces.
- **No collection/database existence checks**: Retrievers assume the store already has data.
- **No LLM failure recovery in RAG pipelines**: If `generate_completion` fails, partial retrieval results are lost. Wrap in try/except and return partial results.
- **No retry/backoff**: `RateLimitError` in `shared/llm.py` raises immediately. Use `tenacity` or litellm built-in retries.
- **No empty-input guards**: Empty queries, empty directories, empty result sets can cause crashes.

## Architecture

### Config
- `Settings` class evaluates `os.getenv()` at class definition time (module import). Consider using `pydantic-settings` for lazy loading and validation.
- `NEO4J_DATABASE` defaults to `"harry-potter"` — a stale default from an earlier stage. Should default to something neutral or fail-fast.
- `EXTRACTION_LLM` in `phase2/ingest.py` is hardcoded to `gpt-5-mini` and bypasses `settings.GENERATION_MODEL`. Move to config.
- `ingest_finance.py` bypasses `shared/config.py` entirely and loads its own `.env`. Should use the shared config.

### Chunk Size Inconsistency
| File | chunk_size | chunk_overlap |
|------|-----------|---------------|
| `phase1/ingest_pdf.py` | 512 | 64 |
| `phase1/ingest_hot.py` | 200 | 20 |
| `phase2/ingest.py` | 256 | 20 |
| `phase2/ingest_finance.py` | 4000 | 200 |

All hardcoded. Differences in RAG quality may be due to chunk size rather than retrieval strategy. Consider standardizing or making configurable.

### Retrieval
- Phase 1 `top_k=8` vs Phase 2 `top_k=5` — inconsistent defaults.
- Phase 1 has `node_threshold=0.3` (default) but `__main__` uses `0.4` — the RAG pipeline uses the default. Three different thresholds.
- Phase 2 has no similarity threshold filtering at all.
- Phase 2 `extract_entities_from_query` makes an LLM call per query — add caching.
- Phase 2 keyword anchor search uses `CONTAINS` (full scan) — should use a Neo4j full-text index.

## Evaluation

### Metrics
- LLM-as-Judge is binary (CORRECT/INCORRECT). Could add confidence scores (0-1 scale).
- No retrieval quality metrics — HotpotQA has `supporting_facts_titles` that could be used to measure if the retriever found the right documents.
- BERTScore was replaced with OpenAI embedding cosine similarity due to TensorFlow issues on macOS. Could revisit with `sentence-transformers` as a local alternative.

### Benchmark
- `save_results` was called inside the loop (writing 100x). Fixed in shared runner — now saves once at end.
- RAGAS API is version-sensitive. No version pinning for `ragas` in `requirements.txt` (now pinned to `0.1.21`).

## Infrastructure

### Docker
- `docker-compose.yml` has Qdrant and Neo4j. Neo4j Desktop is used instead for multi-database support.
- No Dockerfile for the Python app — everything runs locally with conda.
- No `Makefile` — all commands are `python -m ...` which is verbose.

### Dependency Management
- Currently using `requirements.txt` with pinned versions.
- Consider migrating to `uv` with `pyproject.toml` and dependency groups:
  - `[project.dependencies]` — base (llama_index, qdrant, neo4j, litellm, openai)
  - `[project.optional-deps.eval]` — ragas, datasets, langchain-openai
  - `[project.optional-deps.lightrag]` — lightrag-hku
  - `[project.optional-deps.dev]` — jupyter, ipykernel

## Phase 3 — Ontology RAG

### Design Questions
- Which ontology framework? `owlready2` (Python-native, easy) vs `rdflib` (RDF/SPARQL, more flexible)?
- Should the ontology be pre-defined (domain.owl) or auto-generated from the corpus?
- Should we use OWL reasoning (Pellet, HermiT) for inference, or just use the ontology as a schema guide?
- How to handle entity linking — match extracted entities to ontology classes via LLM or embeddings?

### Implementation Priorities
1. Start with a simple typed extraction (Person, Organization, Location)
2. Use ontology classes as Neo4j labels for typed nodes
3. Add ontology-guided retrieval (filter by class, traverse typed relationships)
4. Add OWL reasoning for implicit relationship inference
5. Benchmark against Phase 1 and Phase 2
