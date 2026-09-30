# RAG Trilogy

A 3-phase research project comparing **Retrieval-Augmented Generation (RAG)** approaches on two
multi-hop QA benchmarks.

| Phase | Approach | Vector store | Graph store | Status |
|-------|----------|--------------|-------------|--------|
| **Phase 1** | Pure Vector RAG | Qdrant | — | Functional |
| **Phase 2** | Pure Graph RAG | — | Neo4j (native vector index) | Functional |
| **Phase 3** | Ontology RAG | — | Neo4j + OWL ontology | Not implemented (stubs) |

| Track | Eval set | Corpus |
|-------|----------|--------|
| **HotpotQA** (default) | `hotpot_eval.jsonl` (100) | `data_hotpot/*.txt` (990) |
| **MetaQA** | `MetaQA/metaqa_eval_v1.jsonl` (212, frozen) | `MetaQA/corpus_manifest.txt` (994, frozen) |
| **MetaQA mini** | `MetaQA/metaqa_eval_mini.jsonl` (12, frozen) | `MetaQA/corpus_manifest_mini.txt` (50, frozen) |

## Prerequisites

- Python 3.11 or 3.12
- [UV](https://docs.astral.sh/uv/) (recommended) or pip
- Docker (Qdrant and the `neo4j-metaqa` service)
- Neo4j Desktop or Neo4j Docker for the Phase 2 HotpotQA graph
- `OPENAI_API_KEY` in `.env` (see `.env.example`)
- MetaQA track only: `WIKI_CONTACT` for the Wikimedia User-Agent

## Quickstart

```bash
uv sync
cp .env.example .env        # add OPENAI_API_KEY and Neo4j credentials
docker compose up -d qdrant neo4j
# create the database named by NEO4J_DATABASE (default: hotpot-graph)
python -m prepare_hotpot    # -> data_hotpot/ + hotpot_eval.jsonl
make ingest-p1 && make answers-p1 && make bench-p1
make ingest-p2 && make answers-p2 && make bench-p2
```

Every pipeline target is dataset-driven — `DATASET=hotpot` (default), `metaqa` or `metaqa_mini`:

```bash
make ingest-p1 DATASET=metaqa_mini
make answers-p1 DATASET=metaqa LIMIT=3   # smoke run
make bench-p2 DATASET=metaqa
```

## Makefile commands

| Target | What it does |
|--------|--------------|
| `prepare` | Download 100 HotpotQA questions → `data_hotpot/` + `hotpot_eval.jsonl` |
| `ingest-p1` / `ingest-p2` | Ingest the active dataset into Qdrant / Neo4j |
| `answers-p1` / `answers-p2` | Generate answers for the active dataset (`LIMIT=N` for a smoke run) |
| `bench-p1` / `bench-p2` / `bench-all` | Evaluate one phase / both phases |
| `answers-cb` / `bench-cb` | MetaQA-only closed-book control: same prompt, no retrieval (`results/<dataset>/closed_book/`) |
| `check-config` | Show the resolved config for the active dataset |
| `freeze-verify` | `sha256sum -c` the frozen MetaQA v1 + mini artifacts |
| `clean-results` | Delete `results/<DATASET>/` after a y/N prompt (MetaQA tracks only) |
| `clean-hotpot-DANGEROUS` | DANGEROUS: remove generated HotpotQA artifacts (requires typing `yes`) |
| `ingest-hot`, `ingest-pdf`, `ingest-graph` | Historical direct ingest helpers |
| `retrieve-p1`, `rag-p1`, `retrieve-p2`, `rag-p2` | Retrieval / RAG smoke tests |
| `metaqa-up` | Start the isolated MetaQA Neo4j service (ports 7475/7688) |
| `docker-up`, `docker-down` | Start / stop Qdrant + Neo4j |
| `install` / `sync` | `uv sync` |

Deprecated aliases (`hotpot-p1`, `hotpot-p2`, `metaqa-check`, `metaqa-ingest-p1`,
`metaqa-answers-p1`, `metaqa-bench-p1`, …) still run, and print the equivalent new target.

## Documentation

- [`docs/metaqa-track.md`](docs/metaqa-track.md) — MetaQA pipeline, frozen v1, corpus construction, phase runs
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — freezes, checksums and comparison protocol rules
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — current state and next experiments
- [`AGENTS.md`](AGENTS.md) — module-level entrypoints and gotchas

## Evaluation metrics

| Metric | API required |
|--------|--------------|
| Exact Match, Relaxed EM, Extracted EM, Token F1 | No |
| Semantic Similarity (embedding cosine) | Yes |
| LLM-as-Judge | Yes |
| RAGAS Answer Correctness / Faithfulness / Context Recall | Yes |

## Project structure

```
shared/                       # config (the DATASET switch), LLM wrapper, benchmark engine, runner
phase1_vector_rag/            # Pure vector RAG: ingest, retriever, rag, answers, benchmark
phase2_graph_rag/             # Pure Graph RAG: ingest, retriever, rag, answers, benchmark
phase3_ontology_rag/          # Ontology RAG stubs + design notes
scripts/                      # MetaQA track
├── metaqa_inspect.py         # dataset facts -> MetaQA/INSPECT.md
├── metaqa_sample.py          # candidate sampling + KB self-check
├── metaqa_fetch_wiki.py      # cached Wikipedia fetch (<=1 req/s)
├── metaqa_audit.py           # evidence audit -> frozen eval set
├── metaqa_make_mini.py       # hop-balanced mini slice
└── metaqa_retrieval_check.py # retrieval-only diagnostics (no generation)
MetaQA/                       # frozen MetaQA corpus + eval (gitignored)
results/                      # MetaQA run outputs (gitignored)
data_hotpot/, data_finance/   # source corpora (gitignored)
```

Design invariants: Phase 2 is pure Graph RAG (Neo4j only, no Qdrant); phase runs read the frozen
artifacts and never rebuild them; `shared/config.py::dataset_config(phase)` is the single dataset
switch; MetaQA writes only to its isolated stores and refuses to touch the HotpotQA database.
