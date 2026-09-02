# TODO — RAG Trilogy Master Roadmap

> Complete roadmap to make this repo "done". Check off as you go.

## Phase 1 — Vector RAG (Functional)
- [x] Ingest HotpotQA TXT into Qdrant
- [x] Vector search retrieval
- [x] Full RAG pipeline (retrieve → prompt → generate)
- [x] HotpotQA answer generation (100 questions)
- [x] Benchmark evaluation (EM, Relaxed EM, Extracted EM, F1, Semantic, LLM-Judge, RAGAS)

## Phase 2 — Graph RAG (Functional)
- [x] Ingest HotpotQA into Neo4j (pure Graph RAG, no Qdrant)
- [x] Vector index on Chunk nodes in Neo4j
- [x] Graph traversal retrieval (vector → chunks → relationships)
- [x] Full RAG pipeline (retrieve → prompt → generate)
- [x] HotpotQA answer generation (100 questions)
- [x] Benchmark evaluation (same metrics as Phase 1)
- [x] Rename `preapre_hotpot.py` → `prepare_hotpot.py` (requires updating all references) — done via `prepare_hotpot.py` + deprecated shim `preapre_hotpot.py`
- [ ] Create LightRAG retriever for `ingest_finance.py` (or remove if not needed)

## Phase 3 — Ontology RAG (Not Implemented)
- [ ] Implement `ingest.py` — ontology loading, typed entity extraction, Neo4j storage
- [ ] Implement `retriever.py` — ontology-guided retrieval, class hierarchy traversal
- [ ] Implement `rag.py` — ontology-aware prompt with entity types
- [ ] Implement `benchmark.py` — reuse shared benchmark + ontology quality metrics
- [ ] Expand `ontology/domain.owl` with domain-specific classes and properties
- [ ] Add `owlready2` and `rdflib` to requirements.txt

## Repo Organization (Done)
- [x] Fix `requirements.txt` with pinned dependencies
- [x] Create `.env.example` with documented env vars
- [x] Write `README.md` with quickstart and architecture
- [x] Update `AGENTS.md` to match actual code
- [x] Write Phase 3 stub comments with design notes
- [x] Extract shared benchmark engine (`shared/benchmark.py`)
- [x] Extract shared hotpot runner (`shared/hotpot_runner.py`)
- [x] Fix `temperature=1` → `0` in `shared/llm.py`
- [x] Fix prompt indentation in `phase1/rag.py`
- [x] Add None-score guard in `phase1/rag.py`
- [x] Fix `preapre_hotpot.py` append bug (duplicates on re-run)
- [x] Add `__main__` guard to `preapre_hotpot.py`
- [x] Remove dead imports (`json` in ingest.py, `time` in hotpot_answers, `textwrap` in rag.py, `QueryParam` in ingest_finance.py)
- [x] Use `settings.EMBEDDING_DIMENSION` in `create_vector_index`
- [x] Document `ingest_finance.py` as experimental
- [x] Add `nest-asyncio` to requirements.txt
- [x] Fix llama-index integration package version ranges
- [x] Move `import time` to top-level in `phase2/ingest.py`

## UV Migration (Done)
- [x] Create `pyproject.toml` via `uv init --bare`
- [x] Add all dependencies via `uv add -r requirements.txt`
- [x] Run `uv sync` — all 132 packages installed successfully
- [x] Regenerate `requirements.txt` as fully-pinned fallback via `uv pip compile`
- [x] Update `.gitignore` for `.venv`
- [x] Update `README.md` with UV quickstart
- [x] Update `AGENTS.md` with UV commands
- [x] Verify Phase 1 retriever with `uv run` — works
- [x] Verify Phase 2 retriever with `uv run` — works

## Future Improvements
- [ ] Add prompt variants (generalist + finance) for Phase 1 and Phase 2
- [ ] Add connection preflight checks (Qdrant/Neo4j availability)
- [ ] Add retry/backoff for LLM API calls in `shared/llm.py`
- [ ] Cache entity extraction in `phase2/retriever.py` (LLM call per query)
- [ ] Create Neo4j full-text index for keyword anchor search
- [ ] Standardize `top_k` and threshold parameters across phases
- [ ] Extract shared ingest helper (dedup `ingest_pdf.py` and `ingest_hot.py`)
- [ ] Add `Makefile` with common commands
- [ ] Create cross-phase comparison notebook (`notebooks/benchmark_comparison.ipynb`)
- [ ] Add LLM-as-Judge with confidence scores (not just CORRECT/INCORRECT)
- [ ] Add retrieval quality metrics (did retriever find supporting facts?)
