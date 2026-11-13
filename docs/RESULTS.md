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
| Extraction (Phase 2) | `gpt-6-luna` — no temperature; `max_completion_tokens` only |
| Extraction (Phase 3b) | `gpt-6-luna` — schema-constrained JSON (actors/directors/writers/genres/languages/year) |
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
| Phase 3b — schema-constrained extraction | 0.839 | 0.767 | 0.983 | 0.267 | 0.767 | 0.681 | 0.888 |
| Phase 3a — ontology ceiling | **0.889** | **0.833** | **1.000** | 0.367 | **0.867** | 0.772 | **0.914** |
| Closed book | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — |

Phase 2 ingest with gpt-6-luna extraction: 443 chunks, 12,485 graph relationships, 0 extraction errors, 13.1 min. Phase 3b ingest: 443 chunks, 1,617 actors / 126 directors / 226 writers / 155 genres / 119 languages / 100 years, **0 parse errors**, **0 extraction errors**, ~2 min. Phase 3a: full typed
KB (16,427 movies, 21,731 people, 134k facts) with the 443-chunk index, entity match 30/30,
evidence recall 1.0.

**Phase 3b vs Phase 2 (mini30, paired, n=30):** all_found +0.033, 95% CI [-0.200, +0.100], McNemar p=1.00; gold recall +0.017, CI [-0.133, +0.083]; evidence recall +0.024, CI [-0.083, +0.033]. **No detectable difference**; do not claim a direction.

**Phase 3b vs 3a ceiling (mini30, n=30):** 3a is ahead by +0.050 gold (CI [0.000, +0.133]), +0.067 all_found (CI [0.000, +0.167]), +0.017 evidence (CI [0.000, +0.050]), +0.100 judge (p=0.25). CIs touch zero; **no detectable difference** — 3b is close to the oracle on this small slice.

**Full Phase 3a vs Phase 1 (n=212, paired):** all_found +0.335, 95% CI [+0.274, +0.396], McNemar p<0.0001; gold +0.322, CI [+0.269, +0.377]; evidence +0.311, CI [+0.266, +0.356]. The ceiling is clearly above the vector baseline at full scale.

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
| Phase 3b — schema-constrained extraction | **not run** (mini30 certified; full 3b shares the ~2–3 h extraction cost) | — | — | — | — |
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
- Phase 3b is a separate `*-ontology-b-graph` database; it never overwrites 3a.
- Paired comparisons use `python -m scripts.compare_runs` (exact McNemar + bootstrap CI). Mini30 n=30 is exploratory; full n=212 is the claim-level sample.
