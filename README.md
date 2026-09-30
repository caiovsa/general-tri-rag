# RAG Trilogy

A 3-phase research project comparing **Retrieval-Augmented Generation (RAG)** approaches:

| Phase | Approach | Vector Store | Graph Store | Status |
|-------|----------|-------------|-------------|--------|
| **Phase 1** | Pure Vector RAG | Qdrant | — | Functional |
| **Phase 2** | Pure Graph RAG | — | Neo4j (native vector index) | Functional |
| **Phase 3** | Ontology RAG | — | Neo4j + OWL ontology | Not implemented |

Two benchmark tracks feed these phases:

| Track | Source | Eval set | Corpus |
|-------|--------|----------|--------|
| **HotpotQA** | 100 distractor questions (`prepare_hotpot.py`) | `hotpot_eval.jsonl` | `data_hotpot/` |
| **MetaQA** | Multi-hop film questions (`MetaQA/`) | `MetaQA/metaqa_eval_v1.jsonl` (frozen) | `MetaQA/corpus_manifest.txt` (frozen) |

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
- (MetaQA track only) `WIKI_CONTACT` — contact address for the Wikimedia API User-Agent

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
python -m prepare_hotpot
# legacy alias still works: python -m preapre_hotpot
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

## MetaQA Track — evidence-gated multi-hop films

A second track, built end-to-end by scripts and **frozen before any phase runs**.
Every question carries KB-derived evidence, and only questions whose evidence is
verifiable in the fetched Wikipedia docs survive.

Source: **MetaQA** (`MetaQA/kb.txt` + `1_hop/`, `2_hop/`, `3_hop/` QA files) —
16,427 movie nodes, 134,741 triples, 9 relations, 39,093 test questions.
Full dataset facts: `MetaQA/INSPECT.md`.

### Pipeline

| # | Step | Command | Output |
|---|------|---------|--------|
| 1 | Inspect the dataset | `uv run python -m scripts.metaqa_inspect` | `MetaQA/INSPECT.md` |
| 2 | Sample + self-check | `uv run python -m scripts.metaqa_sample` | `MetaQA/metaqa_candidates.jsonl` |
| 3 | Fetch Wikipedia docs | `uv run python -m scripts.metaqa_fetch_wiki` | `MetaQA/docs/*.txt`, `MetaQA/fetch_report.json`, `MetaQA/cache/` |
| 4 | Audit evidence | `uv run python -m scripts.metaqa_audit` | `MetaQA/metaqa_eval.jsonl`, `MetaQA/audit_dropped.jsonl` |
| 5 | Verify the freeze | `sha256sum -c MetaQA/FROZEN.sha256` | `MetaQA/metaqa_eval_v1.jsonl`, `MetaQA/corpus_manifest.txt` |

Step 3 needs a contact address for the Wikimedia User-Agent — set `WIKI_CONTACT`
in `.env` (see `.env.example`) or export it:

```bash
WIKI_CONTACT=you@example.com uv run python -m scripts.metaqa_fetch_wiki
```

Useful flags: `--dry-run` (corpus + cost estimate, no network), `--limit N`,
`--total-docs`, `--max-chars`, `--per-hop`, `--seed`, `--max-evidence`,
`--max-answers`, `--min-evidence-{1,2,3}hop`.

### Frozen v1

| Artifact | sha256 | Contents |
|----------|--------|----------|
| `metaqa_eval_v1.jsonl` | `824acf90381417721406146ff6263dc8ecb5bcde9f95d3dff70ed72300849cc6` | 212 questions (1-hop 75, 2-hop 67, 3-hop 70) |
| `corpus_manifest.txt` | `4e10fa73d3ae839aae6b0a0ac76c4fe6e8b4c97beb6dbe414c30bf7ca5fd1f29` | 994 docs = 620 must-have + 190 hard + 190 random |

**Phase runs must read `metaqa_eval_v1.jsonl` and ingest exactly the files in
`corpus_manifest.txt`.** Never glob `MetaQA/docs/` — that folder can hold stale
pages (leftovers are parked in `MetaQA/docs_stale/`), which would silently change
the corpus and make phase results incomparable.

### How the corpus and eval set are built

- **Sample (step 2)** — a qtype name encodes the chain (`movie_to_actor_to_movie`).
  Evidence = every movie node on a gold path, derived from `kb.txt`. Before any
  filtering, a self-check re-derives the answer set from the KB and compares it
  with the gold answers (exact set match, both topic-in and topic-out variants);
  qtypes below 95% agreement are dropped, loudly. Filters: unsafe movies (title
  reused by several films, missing `release_year`, title colliding with an
  actor/director/writer name), ≤10 evidence movies, ≤8 answers, and a per-hop
  evidence floor (1 / 2 / 2).
- **Fetch (step 3)** — MediaWiki `action=parse`, one title per request, ≤1 req/s,
  retries with backoff, and **every raw response cached** in `MetaQA/cache/`, so
  re-runs cost no requests. Title resolution: `<title> (<year> film)` →
  `<title> (film)` → `<title>` → search. A page is accepted only if it mentions a
  KB person linked to that movie **and** the KB `release_year` appears in the
  first 1500 chars.
- **Audit (step 4)** — keeps a question only if **every** evidence movie has a
  verified doc **and every** evidence triple passes its per-relation check against
  that movie's own doc: names anywhere in the doc; `in_language` in the infobox
  `Language:` line only; `release_year` in the first 1500 chars; `has_genre` in
  the first 1500 chars (exact, or a small synonym map such as
  `Sci-Fi → science fiction`). Dropped questions and their first failing reason
  go to `audit_dropped.jsonl`.

### v1 results

| Metric | Value |
|--------|-------|
| Questions | 240 sampled → **212 kept (88.3%)** |
| Per hop | 1-hop 75/80 (93.8%), 2-hop 67/80 (83.8%), 3-hop 70/80 (87.5%) |
| Corpus | 994 docs fetched (2030 API requests, 6 titles rejected by verification) |
| Triples passed | `release_year` 99/99, `starred_actors` 356/358, `written_by` 303/306, `directed_by` 242/247, `has_genre` 85/96, `in_language` 54/63 |
| Main failure modes | genre not stated in the first 1500 chars; language absent from the infobox line; a credited name missing from the page |

Re-running steps 2–4 is deterministic (fixed seeds); re-running step 4 on the
frozen corpus reproduces `metaqa_eval.jsonl` byte-identically to
`metaqa_eval_v1.jsonl`.

### Running the phases on MetaQA

The harness is dataset-agnostic. `DATASET=hotpot` (the default) keeps every
historical path, collection, chunk size and database untouched; `DATASET=metaqa`
switches the eval file, corpus, stores and results directory:

| | `DATASET=hotpot` (default) | `DATASET=metaqa` |
|---|---|---|
| Eval file | `hotpot_eval.jsonl` (100) | `MetaQA/metaqa_eval_v1.jsonl` (212) |
| Corpus | `data_hotpot/*.txt` (glob) | `MetaQA/corpus_manifest.txt` (994 — the manifest is read, never a glob) |
| Qdrant | `rag_phase_1_baseline` | `<QDRANT_COLLECTION_NAME>_metaqa` |
| Neo4j | `NEO4J_URI` / `NEO4J_DATABASE` | `NEO4J_METAQA_URI` / `NEO4J_METAQA_DATABASE` (refuses to start on the HotpotQA database) |
| Chunk size | 200/20 (P1), 256/20 (P2) | **256/32 for every phase** |
| Results | `phaseN_*/hotpot_*.json` | `results/metaqa/phaseN/` |
| Chunk text | as-is | prefixed with `[Movie: <title>]` before embedding/extraction |

```bash
make metaqa-check                # resolved config: docs, collection, DB, eval rows
make metaqa-ingest-p1            # Phase 1 -> Qdrant
make metaqa-ingest-p2            # Phase 2 -> isolated Neo4j database
make metaqa-answers-p1 LIMIT=3   # smoke run (also: --no-resume, --top-k)
make metaqa-bench-p1
```

`DATASET=metaqa_mini` is the same track on a **hop-balanced slice** — 12 questions
(4 per hop) over 50 docs (35 evidence + 15 distractor), built by
`uv run python -m scripts.metaqa_make_mini` (seed 42) into
`MetaQA/{metaqa_eval_mini.jsonl,corpus_manifest_mini.txt}`. It uses its own
collection, its own graph database (`metaqa-mini-graph`) and `results/metaqa_mini/`,
so a mini baseline can never touch the full run's stores. A 50-doc slice cannot
cover hop 3 unless the questions are chosen deliberately — picking the first 50
docs covers only 6 of 212 questions, all hop 1.

- `TOP_K` (env) or `--top-k` override retrieval depth; the dataset and top_k are
  recorded in `run_meta.json` and in the eval-results metadata.
- Per-question metrics land in `results/metaqa/phaseN/per_question.jsonl`
  (`id, phase, hop, evidence_n, gold_recall, all_found, evidence_recall, judge`)
  so two runs can be compared paired.
- `metaqa_eval.jsonl` uses the `hotpot_eval.jsonl` shape — `question`, `answer`,
  `supporting_facts_titles` (= evidence movie titles) plus `answers`, `hop`,
  `qtype`, `id`, `evidence_triples` — so the shared runner and benchmark engine can
  consume it as-is. Two things to wire when the MetaQA ingest is written:

1. ingest exactly the paths in `MetaQA/corpus_manifest.txt` (never glob `docs/`);
2. point the runner at `MetaQA/metaqa_eval_v1.jsonl`
   (`shared/hotpot_runner.py` hardcodes `EVAL_FILE = hotpot_eval.jsonl`).

### MetaQA graph isolation

MetaQA ingest calls `SHOW DATABASES` first and **refuses to run if it would touch
the HotpotQA database**. Default target: the same server, dedicated database
(`NEO4J_METAQA_DATABASE=metaqa-graph`). If your server does not support multiple
databases (Neo4j Community), use the isolated service instead:

```bash
docker compose up -d neo4j-metaqa      # ports 7475 (browser) / 7688 (bolt)
# .env: NEO4J_METAQA_URI=neo4j://127.0.0.1:7688  NEO4J_METAQA_DATABASE=neo4j
```

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
├── scripts/                   # MetaQA track (inspect → sample → fetch → audit)
│   ├── metaqa_inspect.py      # Dataset facts -> MetaQA/INSPECT.md
│   ├── metaqa_sample.py       # Candidate sampling + KB self-check
│   ├── metaqa_fetch_wiki.py   # Wikipedia fetch (cached, rate limited)
│   └── metaqa_audit.py        # Evidence audit -> frozen eval set
├── MetaQA/                    # MetaQA track data (gitignored)
│   ├── kb.txt, 1_hop/, 2_hop/, 3_hop/   # source dataset
│   ├── INSPECT.md             # dataset inspection report
│   ├── metaqa_candidates.jsonl          # sampled questions + evidence
│   ├── docs/                  # fetched pages (only manifest files belong here)
│   ├── docs_stale/            # parked leftovers from older runs
│   ├── cache/                 # raw API responses (re-runs are free)
│   ├── corpus_manifest.txt    # FROZEN: the 994 docs that make up the corpus
│   ├── metaqa_eval_v1.jsonl   # FROZEN: the 212-question eval set
│   ├── FROZEN.sha256          # checksums for the two files above
│   ├── fetch_report.json      # resolved / unverified / unresolved per title
│   └── audit_dropped.jsonl    # dropped questions + first failing reason
├── data/                      # Harry Potter PDF (testing)
├── data_hotpot/               # HotpotQA context documents (gitignored)
├── data_finance/              # Finance PDFs (gitignored)
├── notebooks/                 # Exploratory notebooks
├── notes/                     # Local notes and TODO
├── plans/                     # Detailed task plans
├── prepare_hotpot.py          # Download + prepare HotpotQA data (preapre_hotpot.py is deprecated shim)
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
- **MetaQA is evidence-gated**: every question ships the exact movies and triples that justify its gold answers, and the audit drops any question whose evidence is not verifiable in the fetched docs. No phase ever has to guess what the corpus is
- **Frozen artifacts over regeneration**: `corpus_manifest.txt` + `metaqa_eval_v1.jsonl` + `FROZEN.sha256` are the contract between phases; steps 2–4 are deterministic and reproducible, but a phase run must never rebuild them
- **One harness, two datasets**: `shared/config.py::dataset_config(phase)` is the single switch. HotpotQA keeps its historical paths/collections/chunk sizes; MetaQA reads the frozen manifest, tags every chunk with `[Movie: <title>]`, and writes to isolated stores and `results/metaqa/`

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
