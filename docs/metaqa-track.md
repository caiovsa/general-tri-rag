# MetaQA Track — evidence-gated multi-hop films

A second track, built end-to-end by scripts and **frozen before any phase runs**.
Every question carries KB-derived evidence, and only questions whose evidence is
verifiable in the fetched Wikipedia docs survive.

Source: **MetaQA** (`MetaQA/kb.txt` + `1_hop/`, `2_hop/`, `3_hop/` QA files) —
16,427 movie nodes, 134,741 triples, 9 relations, 39,093 test questions.
Full dataset facts: `MetaQA/INSPECT.md`.

## Pipeline

| # | Step | Command | Output |
|---|------|---------|--------|
| 1 | Inspect the dataset | `uv run python -m scripts.metaqa_inspect` | `MetaQA/INSPECT.md` |
| 2 | Sample + self-check | `uv run python -m scripts.metaqa_sample` | `MetaQA/metaqa_candidates.jsonl` |
| 3 | Fetch Wikipedia docs | `uv run python -m scripts.metaqa_fetch_wiki` | `MetaQA/docs/*.txt`, `MetaQA/fetch_report.json`, `MetaQA/cache/` |
| 4 | Audit evidence | `uv run python -m scripts.metaqa_audit` | `MetaQA/metaqa_eval.jsonl`, `MetaQA/audit_dropped.jsonl` |
| 5 | Verify the freeze | `make freeze-verify` | `MetaQA/metaqa_eval_v1.jsonl`, `MetaQA/corpus_manifest.txt` |

Step 3 needs a contact address for the Wikimedia User-Agent — set `WIKI_CONTACT`
in `.env` (see `.env.example`) or export it:

```bash
WIKI_CONTACT=you@example.com uv run python -m scripts.metaqa_fetch_wiki
```

Useful flags: `--dry-run` (corpus + cost estimate, no network), `--limit N`,
`--total-docs`, `--max-chars`, `--per-hop`, `--seed`, `--max-evidence`,
`--max-answers`, `--min-evidence-{1,2,3}hop`.

## Frozen v1

| Artifact | sha256 | Contents |
|----------|--------|----------|
| `metaqa_eval_v1.jsonl` | `824acf90381417721406146ff6263dc8ecb5bcde9f95d3dff70ed72300849cc6` | 212 questions (1-hop 75, 2-hop 67, 3-hop 70) |
| `corpus_manifest.txt` | `4e10fa73d3ae839aae6b0a0ac76c4fe6e8b4c97beb6dbe414c30bf7ca5fd1f29` | 994 docs (= 620 must-have + 190 hard + 190 random requested; 6 titles failed verification) |

**Phase runs must read `metaqa_eval_v1.jsonl` and ingest exactly the files in
`corpus_manifest.txt`.** Never glob `MetaQA/docs/` — that folder can hold stale
pages (leftovers are parked in `MetaQA/docs_stale/`), which would silently change
the corpus and make phase results incomparable.

## How the corpus and eval set are built

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

## v1 results

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

## Running the phases on MetaQA

The harness is dataset-agnostic. `DATASET=hotpot` (the default) keeps every
historical path, collection, chunk size and database untouched; `DATASET=metaqa`
switches the eval file, corpus, stores and results directory:

| | `DATASET=hotpot` (default) | `DATASET=metaqa` |
|---|---|---|
| Eval file | `hotpot_eval.jsonl` (100) | `MetaQA/metaqa_eval_v1.jsonl` (212) |
| Corpus | `data_hotpot/*.txt` (glob) | `MetaQA/corpus_manifest.txt` (994 — the manifest is read, never a glob) |
| Qdrant | `rag_phase_1_baseline` | `<QDRANT_COLLECTION_NAME>_metaqa` |
| Neo4j | `NEO4J_URI` / `NEO4J_DATABASE` (default `hotpot-graph`) | `NEO4J_METAQA_URI` / `NEO4J_METAQA_DATABASE` (refuses to start on the HotpotQA database) |
| Chunk size | 200/20 (P1), 256/20 (P2) | **256/32 for every phase** |
| Results | `phaseN_*/hotpot_*.json` | `results/metaqa/phaseN/` |
| Chunk text | as-is | prefixed with `[Movie: <title>]` before embedding/extraction |

```bash
make check-config DATASET=metaqa    # resolved config: docs, collection, DB, eval rows
make ingest-p1 DATASET=metaqa       # Phase 1 -> Qdrant
make ingest-p2 DATASET=metaqa       # Phase 2 -> isolated Neo4j database
make answers-p1 DATASET=metaqa LIMIT=3   # smoke run (also: --no-resume, --top-k)
make bench-p1 DATASET=metaqa
```

The old `make metaqa-check` / `metaqa-ingest-p1` / … aliases still work but print a
deprecation notice.

`DATASET=metaqa_mini` is the same track on a **hop-balanced slice** — 12 questions
(4 per hop) over 50 docs (35 evidence + 15 distractor), built by
`uv run python -m scripts.metaqa_make_mini` (defaults `--per-hop 4 --docs 50`,
seed 42) into `MetaQA/{metaqa_eval_mini.jsonl,corpus_manifest_mini.txt}`. It uses
its own collection, its own graph database (`metaqa-mini-graph`) and
`results/metaqa_mini/`, so a mini baseline can never touch the full run's stores.
This slice is a pilot; the roadmap scales it to 30 questions / 100 docs. A 50-doc
slice cannot cover hop 3 unless the questions are chosen deliberately — picking
the first 50 docs covers only 6 of 212 questions, all hop 1.

- `TOP_K` (env) or `--top-k` override retrieval depth; the dataset and top_k are
  recorded in `run_meta.json` and in the eval-results metadata.
- Per-question metrics land in `results/metaqa/phaseN/per_question.jsonl`
  (`id, phase, hop, evidence_n, gold_recall, all_found, evidence_recall, judge`)
  so two runs can be compared paired.
- `metaqa_eval.jsonl` uses the `hotpot_eval.jsonl` shape — `question`, `answer`,
  `supporting_facts_titles` (= evidence movie titles) plus `answers`, `hop`,
  `qtype`, `id`, `evidence_triples` — so the shared runner and benchmark engine can
  consume it as-is. Both wiring points are live: ingest reads exactly the paths in
  `MetaQA/corpus_manifest.txt` (never a glob of `docs/`), and
  `shared/config.py::dataset_config()` points the runner at
  `MetaQA/metaqa_eval_v1.jsonl`. `shared/hotpot_runner.py`'s
  `EVAL_FILE = hotpot_eval.jsonl` is only a legacy default — `run_phase()` always
  uses the resolved dataset config.

## MetaQA graph isolation

MetaQA ingest calls `SHOW DATABASES` first and **refuses to run if it would touch
the HotpotQA database** (the database named by `NEO4J_DATABASE`, default
`hotpot-graph`). Default target: the same server, dedicated database
(`NEO4J_METAQA_DATABASE=metaqa-graph`). If your server does not support multiple
databases (Neo4j Community), use the isolated service instead:

```bash
docker compose up -d neo4j-metaqa      # ports 7475 (browser) / 7688 (bolt)
# .env: NEO4J_METAQA_URI=neo4j://127.0.0.1:7688  NEO4J_METAQA_DATABASE=neo4j
```
