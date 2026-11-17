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
| **MetaQA mini30** | `MetaQA/metaqa_eval_mini30.jsonl` (30, frozen) | `MetaQA/corpus_manifest_mini30.txt` (100, frozen) |

The mini tracks are plumbing checks, not results: their corpora are tiny and evidence is
shared between questions, so never compare Phase 1 vs Phase 2 (or mini vs full) scores on them.

## Prerequisites

- Python 3.11 or 3.12
- [UV](https://docs.astral.sh/uv/) (recommended) or pip
- Docker (Qdrant and the `neo4j-metaqa` service)
- Neo4j Desktop or Neo4j Docker for the Phase 2 HotpotQA graph
- `OPENAI_API_KEY` in `.env` (see `.env.example`)
- Pinned models: generation/extraction `gpt-6-luna`, judge `gpt-5.6-luna` (no temperature parameter — Luna only accepts the default)
- MetaQA track only: `WIKI_CONTACT` for the Wikimedia User-Agent

### Model routing

The answer/extraction model is selected with `GENERATION_MODEL`. An OpenRouter model
must use the `openrouter/` prefix so the wrapper selects the OpenRouter key and base URL:

```bash
# .env
GENERATION_MODEL=openrouter/qwen/qwen3.5-9b
GENERATION_TEMPERATURE=0
GENERATION_REASONING_EFFORT=none
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_PROVIDER_ONLY=primeintellect/bf16
OPENROUTER_ALLOW_FALLBACKS=false
```

The existing Luna path is unchanged: `GENERATION_MODEL=gpt-6-luna` continues to use the
OpenAI/Maritaca route. Judge (`JUDGE_MODEL`), extraction (`EXTRACTION_MODEL`) and RAGAS
(`RAGAS_MODEL`) are separate knobs; changing `GENERATION_MODEL` does not change the judge.

## Quickstart

```bash
uv sync
cp .env.example .env        # add OPENAI_API_KEY and Neo4j credentials
docker compose up -d qdrant
python -m prepare_hotpot    # -> data_hotpot/ + hotpot_eval.jsonl
make ingest-p1 && make answers-p1 && make bench-p1
make ingest-p2 && make answers-p2 && make bench-p2
```

Neo4j: the bundled `neo4j` compose service is Community (one user database). For the default
multi-database layout (`hotpot-graph`, `metaqa-graph`, `metaqa-mini-graph`, `metaqa-mini30-graph`,
plus the Phase 3 ontology databases) use Neo4j Desktop or an Enterprise server. On a
single-database server, start the isolated `neo4j-metaqa` service and point
`NEO4J_METAQA_URI`/`NEO4J_METAQA_DATABASE` at it (see `docs/metaqa-track.md`).

Every pipeline target is dataset-driven — `DATASET=hotpot` (default), `metaqa`, `metaqa_mini`,
`metaqa_mini30` or `local`:

```bash
make mini30                              # build the 30q/100-doc mini30 track (local, no network)
make ingest-p1 DATASET=metaqa_mini30 RESET=1
make answers-p1 DATASET=metaqa_mini30    # smoke run with --limit is also supported
make bench-p1 DATASET=metaqa_mini30
make bench-p2 DATASET=metaqa
```

Phase 3 (ontology) is MetaQA-only and uses a typed graph built directly from `kb.txt`:

```bash
make ingest-p3 DATASET=metaqa_mini30 RESET=1   # 3a: typed KB graph + the same 256/32 chunks
make answers-p3 DATASET=metaqa_mini30          # entity match -> typed BFS -> chunk ranking
make bench-p3 DATASET=metaqa_mini30            # same metrics as Phase 1/2

make ingest-p3b DATASET=metaqa_mini30 RESET=1  # 3b: schema-constrained gpt-6-luna extraction
make answers-p3b DATASET=metaqa_mini30
make bench-p3b DATASET=metaqa_mini30
```

Phase 3a is the oracle graph (no extraction); Phase 3b is schema-constrained LLM extraction.
They use separate `*-ontology-graph` / `*-ontology-b-graph` databases and never touch the Phase 2 DB.
Full MetaQA results: vector .554 gold / .655 evidence, graph .757 / .841, 3b .779 / .893, 3a ceiling .876 / .966
(see `docs/RESULTS.md`).
Paired comparisons: `python -m scripts.compare_runs <run_a> <run_b> --metric all_found`.

## Makefile commands

| Target | What it does |
|--------|--------------|
| `prepare` | Download 100 HotpotQA questions → `data_hotpot/` + `hotpot_eval.jsonl` |
| `ingest-p1` / `ingest-p2` | Ingest the active dataset into Qdrant / Neo4j (`RESET=1` to recreate) |
| `answers-p1` / `answers-p2` | Generate answers for the active dataset (`LIMIT=N` for a smoke run) |
| `bench-p1` / `bench-p2` / `bench-all` | Evaluate one phase / both phases |
| `answers-cb` / `bench-cb` | MetaQA-only closed-book control: same prompt, no retrieval |
| `mini30` | Build the 30q/100-doc mini30 track (local, no network) |
| `test` | Offline test suite (no API, no services required) |
| `check-config` | Show the resolved config for the active dataset |
| `freeze-verify` | `sha256sum -c` the frozen v1 + mini + mini30 artifacts |
| `clean-results` | Delete `results/<DATASET>/` after a y/N prompt (MetaQA tracks + local) |
| `clean-hotpot-DANGEROUS` | DANGEROUS: remove generated HotpotQA artifacts (requires typing `yes`) |
| `ingest-hot`, `ingest-pdf`, `ingest-graph` | Historical direct ingest helpers (`ingest-pdf` uses an isolated collection) |
| `retrieve-p1`, `rag-p1`, `retrieve-p2`, `rag-p2` | Retrieval / RAG smoke tests |
| `metaqa-up` | Start the isolated MetaQA Neo4j service (ports 7475/7688) |
| `docker-up`, `docker-down` | Start / stop Qdrant (+ the compose Neo4j service) |
| `install` / `sync` | `uv sync` |

Deprecated aliases (`hotpot-p1`, `hotpot-p2`, `metaqa-check`, `metaqa-ingest-p1`,
`metaqa-answers-p1`, `metaqa-bench-p1`, …) still run, and print the equivalent new target.

## Documentation

- [`docs/metaqa-track.md`](docs/metaqa-track.md) — MetaQA pipeline, frozen v1, corpus construction, phase runs
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — freezes, checksums and comparison protocol rules
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — current state and next experiments
- [`docs/RESULTS.md`](docs/RESULTS.md) — protocol and the mini30 / full baselines actually measured
- [`docs/MINI30.md`](docs/MINI30.md) — how the 30q/100-doc plumbing corpus was selected
- [`AGENTS.md`](AGENTS.md) — module-level entrypoints and gotchas

## Evaluation metrics

| Metric | API required |
|--------|--------------|
| Exact Match, Relaxed EM, Extracted EM, Token F1 | No |
| Semantic Similarity (embedding cosine) | Yes |
| LLM-as-Judge (fixed whole-word parser; judge != generator) | Yes |
| RAGAS Faithfulness / Context Recall (Answer Correctness removed — unreliable) | Yes |

## Project structure

```
shared/                       # config (the DATASET switch), LLM wrapper, benchmark engine, runner
phase1_vector_rag/            # Pure vector RAG: ingest, retriever, rag, answers, benchmark
phase2_graph_rag/             # Pure Graph RAG: ingest, retriever, rag, answers, benchmark
phase3_ontology_rag/          # Phase 3a: typed KB graph (oracle) + retrieval
phase3b_ontology_rag/         # Phase 3b: schema-constrained LLM extraction
scripts/                      # MetaQA track
├── metaqa_inspect.py         # dataset facts -> MetaQA/INSPECT.md
├── metaqa_sample.py          # candidate sampling + KB self-check
├── metaqa_fetch_wiki.py      # cached Wikipedia fetch (<=1 req/s)
├── metaqa_audit.py           # evidence audit -> frozen eval set
├── metaqa_make_mini.py       # 12q/50-doc wiring slice
├── metaqa_make_mini30.py     # 30q/100-doc plumbing corpus
└── metaqa_retrieval_check.py # retrieval-only diagnostics (no generation)
tests/                        # offline unittest suite (config, metrics, resume, mini30)
MetaQA/                       # frozen eval/manifests + local docs (docs/kb are not committed)
results/                      # per-dataset run outputs (gitignored)
data_hotpot/, data_finance/   # source corpora (gitignored)
```

Design invariants: Phase 2 is pure Graph RAG (Neo4j only, no Qdrant); Phase 3 always uses a
database distinct from Phase 2; phase runs read the frozen artifacts and never rebuild them;
`shared/config.py::dataset_config(phase)` is the single dataset switch; every run writes a
fingerprint into `run_meta.json` and refuses to resume into a different configuration.
