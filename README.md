# RAG Trilogy

A 3-phase research project comparing **Retrieval-Augmented Generation (RAG)** approaches:

| Phase | Approach | Vector Store | Graph Store | Status |
|-------|----------|-------------|-------------|--------|
| **Phase 1** | Pure Vector RAG | Qdrant | — | Functional |
| **Phase 2** | Pure Graph RAG | — | Neo4j (native vector index) | Functional |
| **Phase 3** | Ontology RAG | — | Neo4j + OWL ontology | Not implemented |

## Architecture

```
                    ┌─────────────┐
                    │  PDFs / TXT │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
     Phase 1           Phase 2         Phase 3
   (Vector RAG)    (Graph RAG)    (Ontology RAG)
           │               │               │
      ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
      │ Qdrant  │    │  Neo4j  │    │  Neo4j  │
      │ (vector)│    │(vector+ │    │+ OWL    │
      │         │    │ graph)  │    │ontology)│
      └─────────┘    └─────────┘    └─────────┘
           │               │               │
           ▼               ▼               ▼
        retrieve       retrieve        retrieve
           │               │               │
           └───────┬───────┘               │
                   ▼                       │
            LLM Generation                 │
           (litellm/OpenAI)                │
                   │                       ▼
                   ▼               LLM Generation
              Answer              (litellm/OpenAI)
                                       │
                                       ▼
                                   Answer
```

## Quickstart

### 1. Prerequisites

- Python 3.11 or 3.12
- [UV](https://docs.astral.sh/uv/) package manager (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (for Qdrant)
- Neo4j Desktop or Neo4j Docker (for Phase 2)
- OpenAI API key
- (Optional) Maritaca Key (For PT/BR use cases) 

### 2. Install dependencies

**With UV (recommended):**
```bash
uv sync
```

**With pip (fallback):**
```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and Neo4j credentials
```

### 4. Start Qdrant (for Phase 1)

```bash
docker compose up -d qdrant
```

### 5. Start Neo4j (for Phase 2)

`Use Neo4j Desktop (recommended for multiple databases) or:`

```bash
docker compose up -d neo4j
```

Create a database named `hotpot-graph` in Neo4j.

## Usage

> **Tip:** With UV, prefix any command with `uv run` to execute it in the project's virtual environment automatically. Example: `uv run python -m phase1_vector_rag.rag`

### Prepare HotpotQA dataset

Downloads 100 HotpotQA questions + context documents:

```bash
python -m preapre_hotpot
```

Creates:
- `data_hotpot/*.txt` — context documents
- `hotpot_eval.jsonl` — 100 questions with golden answers

### Phase 1 — Vector RAG

```bash
# 1. Ingest documents into Qdrant
python -m phase1_vector_rag.ingest_hot

# 2. Test retrieval
python -m phase1_vector_rag.retriever

# 3. Test full RAG pipeline
python -m phase1_vector_rag.rag

# 4. Generate answers for all 100 HotpotQA questions
python -m phase1_vector_rag.hotpot_answers

# 5. Evaluate with benchmarks
python -m phase1_vector_rag.benchmark
```

### Phase 2 — Graph RAG

```bash
# 1. Ingest documents into Neo4j (creates vector index + graph relationships)
python -m phase2_graph_rag.ingest

# 2. Test retrieval
python -m phase2_graph_rag.retriever

# 3. Test full RAG pipeline
python -m phase2_graph_rag.rag

# 4. Generate answers for all 100 HotpotQA questions
python -m phase2_graph_rag.hotpot_answers

# 5. Evaluate with benchmarks
python -m phase2_graph_rag.benchmark
```

### Phase 3 — Ontology RAG (not yet implemented)

See `phase3_ontology_rag/*.py` for implementation stubs with design notes.

## Evaluation Metrics

The benchmark scripts compute:

| Metric | Description | API required |
|--------|-------------|-------------|
| Exact Match (EM) | Official HotpotQA strict match | No |
| Relaxed EM | Ground truth is substring of answer | No |
| Extracted EM | EM after stripping conversational filler | No |
| Token F1 | Official HotpotQA token-level F1 | No |
| Semantic Similarity | Cosine similarity via OpenAI embeddings | Yes |
| LLM-as-Judge | LLM grades answer correctness (CORRECT/INCORRECT) | Yes |
| RAGAS Answer Correctness | Semantic + factual overlap | Yes |
| RAGAS Faithfulness | Is the answer grounded in retrieved contexts? | Yes |
| RAGAS Context Recall | Did the contexts cover the ground truth? | Yes |

Results are saved to:
- `hotpot_results.json` — simple question/answer pairs
- `hotpot_results_detailed.json` — with contexts, scores, ground truth
- `hotpot_eval_results.json` — all metrics per question
- `hotpot_eval_report.txt` — human-readable report

## Project Structure

```
rag-trilogy/
├── shared/                    # Common config, LLM wrapper
│   ├── config.py              # Settings + env loading
│   ├── llm.py                 # LiteLLM generation wrapper
│   ├── benchmark.py           # Shared benchmark engine
│   └── hotpot_runner.py       # Shared answer generation runner
├── phase1_vector_rag/         # Pure vector RAG (Qdrant)
│   ├── ingest_pdf.py          # Ingest PDFs into Qdrant
│   ├── ingest_hot.py          # Ingest HotpotQA TXT into Qdrant
│   ├── retriever.py           # Vector search retrieval
│   ├── rag.py                 # Full RAG pipeline
│   ├── hotpot_answers.py      # Generate answers for benchmark
│   └── benchmark.py           # Evaluation script
├── phase2_graph_rag/          # Pure Graph RAG (Neo4j)
│   ├── ingest.py              # Ingest into Neo4j (vector + graph)
│   ├── ingest_finance.py      # EXPERIMENTAL: LightRAG finance path
│   ├── retriever.py           # Vector + graph traversal retrieval
│   ├── rag.py                 # Full RAG pipeline
│   ├── hotpot_answers.py      # Generate answers for benchmark
│   └── benchmark.py           # Evaluation script
├── phase3_ontology_rag/       # Ontology RAG (not implemented)
│   ├── ontology/domain.owl    # OWL ontology stub
│   ├── ingest.py
│   ├── retriever.py
│   ├── rag.py
│   └── benchmark.py
├── data/                      # Harry Potter PDF (testing)
├── data_hotpot/               # HotpotQA context documents (gitignored)
├── data_finance/              # Finance PDFs (gitignored)
├── notebooks/                 # Exploratory notebooks
├── notes/                     # Local notes and TODO
├── plans/                     # Detailed task plans
├── preapre_hotpot.py          # Download + prepare HotpotQA data
├── hotpot_eval.jsonl          # 100 questions with golden answers
├── docker-compose.yml         # Qdrant + Neo4j services
├── requirements.txt           # Python dependencies
└── .env.example               # Template environment file
```

## Key Design Decisions

- **Phase 2 is pure Graph RAG**: Neo4j stores both the property graph AND vector embeddings (via native vector indexes). No Qdrant dependency.
- **Two extractors in Phase 2**: `ImplicitPathExtractor` (free NLP) + `SimpleLLMPathExtractor` (cheap LLM, 8 async workers)
- **Collection/database isolation**: Phase 1 uses `rag_phase_1_baseline` in Qdrant; Phase 2 uses a separate Neo4j database
- **HotpotQA for benchmarking**: 100 questions with golden answers and supporting facts — small enough for fast iteration, diverse enough for meaningful evaluation

## Dependency Management

This project uses [UV](https://docs.astral.sh/uv/) for dependency management with `pyproject.toml` as the source of truth. A fully-pinned `requirements.txt` is kept as a fallback for pip users.

**Adding a new dependency:**
```bash
uv add <package-name>
```

**Regenerating requirements.txt after changes:**
```bash
uv pip compile pyproject.toml -o requirements.txt --python-version 3.11
```
