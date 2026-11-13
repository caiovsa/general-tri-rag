# Results — corrected baselines (2026-11-12)

> Authoritative per-question data lives in `results/<track>/phaseN/` (gitignored). This page
> records the protocol each number belongs to. **Mini tracks are plumbing checks**: tiny corpora,
> shared evidence, not for phase-vs-phase claims. Phase 3a is a **ceiling** (the graph is built
> from `kb.txt`, the same KB the questions came from), not a competitor.

## Protocol

| | |
|---|---|
| Chunking | 256 / 32 tokens (all MetaQA phases) |
| Retrieval depth | `top_k = 20` |
| Generation | `gpt-6-luna` — **no temperature parameter** (Luna accepts only the provider default, 1) |
| Judge | `gpt-5.6-luna` — no temperature parameter; whole-word verdict parser; raw verdict saved |
| Extraction (Phase 2/3b) | `gpt-6-luna` — no temperature; `max_completion_tokens` only |
| RAGAS | `gpt-4o-mini` — **Faithfulness + Context Recall only**; Answer Correctness removed |
| Embeddings | `text-embedding-3-small` |
| Run identity | every run writes `run_meta.json` (eval/corpus hashes, chunking, top_k, models, `"temperature": "default"`, prompt/ontology hashes); resume with a different fingerprint is refused |
| Neighbours | ranked by shared entities (`ORDER BY shared_entities DESC, chunk.id`), not insertion order |
| Full-text | `anchor_search` (`fulltext` / `contains` / `mixed`) is recorded; fallback is logged loudly |

The judge parser was broken (`INCORRECT` contains `CORRECT`). Every previous judge number was a
**refusal rate**, not accuracy. All numbers below are re-run with the fixed parser and the separate
judge model. Extracted EM was also fixed for yes/no answers.

## MetaQA mini30 (30 q / 100 docs) — plumbing certificate

| Phase | gold recall | all found | evidence recall | EM | judge | RAGAS faith | RAGAS ctx |
|---|---|---|---|---|---|---|---|
| Phase 1 — vector | 0.681 | 0.600 | 0.813 | 0.367 | 0.633 | 0.879 | 0.737 |
| Phase 2 — graph (LLM triples) | 0.822 | 0.733 | 0.959 | 0.333 | 0.733 | 0.719 | 0.919 |
| Phase 3a — ontology ceiling | **0.889** | **0.833** | **1.000** | 0.367 | **0.867** | 0.772 | **0.914** |
| Closed book | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — |

Phase 2 ingest with gpt-6-luna extraction: 443 chunks, 12,485 graph relationships, 0 extraction errors, 13.1 min. Phase 3a: full typed
KB (16,427 movies, 21,731 people, 134k facts) with the 443-chunk index, entity match 30/30,
evidence recall 1.0.

**Neighbour ablation (mini30):** baseline Phase 2 gold/evidence **0.822 / 0.959** vs
`--no-neighbours` **0.739 / 0.809** (judge 0.733 vs 0.700). The 1-hop neighbours add real signal
on this corpus — but this is still 30 questions; do not turn it into a general claim.

## MetaQA mini12 (12 q / 50 docs) — wiring check

| Run | gold recall | evidence recall | judge |
|---|---|---|---|
| Phase 1 | 0.500 | 0.786 | 0.333 |
| Phase 2 baseline | 0.542 | 0.906 | 0.500 |
| Phase 2 `--no-neighbours` | 0.542 | 0.762 | 0.500 |

## MetaQA full (212 q / 994 docs)

| Phase | status | gold recall | evidence recall | judge | RAGAS faith / ctx |
|---|---|---|---|---|---|
| Phase 1 — vector | **212/212, 0 errors** | 0.554 | 0.655 | 0.486 | 0.797 / 0.629 |
| Phase 2 — graph | **not run** (full ingest parked; gpt-6-luna extraction measured at 13.1 min / 443 chunks → ~2.5 h and ~$3 for 4,950 chunks) | — | — | — | — |
| Phase 3a — ontology ceiling | **212/212, 0 errors** | **0.876** | **0.966** | **0.821** | 0.751 / 0.868 |

Phase 1 ingest is deterministic: `hotpot-qa_metaqa` rebuilt to exactly **4,522 chunks**. Phase 3a
uses its own `metaqa-ontology-graph` and records the ontology sha256 in `run_meta.json`.

**No Phase 1 vs Phase 2 comparison exists on the full MetaQA set** (Phase 2 not run). The
mini30/12 numbers are not a substitute.

## Generic local track smoke

A 1-document text fixture with 2 questions was ingested through `DATASET=local` for both phases:
2/2 answered, 0 errors, EM 1.0, judge 1.0, RAGAS faith/ctx 1.0 for both. PDF text extraction was
verified separately (`data/HP-1.pdf` → 274 pages via `pypdf`).

## HotpotQA — retired (historical only)

The committed HotpotQA results are **not comparable** and are no longer a headline:
- Phase 1 used 200/20 chunks and top_k 8; Phase 2 used 256/20 and top_k 5.
- Context volume was ~3.6k vs ~7.8k characters (Phase 2: 14.9 blocks, 65% neighbours).
- The prompts differ; the Phase 2 prompt claimed triplets that were not in the context.
- The old judge numbers (0.80 / 0.83) were refusal rates.
- The historical corpus may contain one distractor lost to a filename collision in
  `prepare_hotpot.py` (now guarded).

If HotpotQA returns, it needs a controlled v2: same 256/20 chunking, same top_k/context budget,
same prompt, fixed judge, and 500+ sampled questions. Not started.

## Operational facts

- **0 `Error:` rows and 0 empty contexts** in every run above.
- Every phase run stores model/temperature/prompt/ingest provenance; `ingest_meta.json` records
  chunks/relations/errors/policy skips.
- Phase 2 content-policy rejections are per-chunk skips (chunk keeps text/embedding), never fatal.
- Phase 3a is a ceiling and must be presented as such.
