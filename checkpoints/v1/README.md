# RAG Trilogy — v1 freeze

**Frozen:** 2026-11-17T02:27:14+00:00  
**Tag:** `metaqa-results-v1-default-medium`  
**Source commit before checkpoint commit:** `1135eea0b58d7c302493e4f26d21cedc7c376c1d`  
**Scope:** MetaQA full 212 questions, plus mini30/mini12/local wiring tracks.

This is the **default-medium reasoning** v1 baseline. It is **not** a no-reasoning baseline.

## Important: reasoning status

`gpt-6-luna` is a reasoning model and defaults to **medium reasoning effort** unless `reasoning_effort` is set. The v1 code did **not** set `reasoning_effort`, and `run_meta.json` does not record it. Therefore:

- v1 answers use the provider-default reasoning policy (**medium**), not an explicit no-reasoning mode.
- A simple question can use zero reasoning tokens adaptively; harder 2-hop/3-hop questions may use reasoning tokens.
- Do **not** compare v1 with a future `reasoning_effort=none, temperature=0` run as if they were the same protocol.
- A live probe on a multi-hop prompt showed 168 reasoning tokens at default, 0 at `none`, and 344 at `low`.

The future no-reasoning/temperature-0 protocol must be a **new checkpoint/tag** (for example `metaqa-results-v2-no-reasoning`).

## Models

| Role | Model | Reasoning | Temperature |
|---|---|---|---|
| Answer | `gpt-6-luna` | provider default (**medium**) | omitted |
| Extraction (P2/P3b) | `gpt-6-luna` | provider default (**medium**) | omitted |
| Judge | `gpt-5.6-luna` | default | omitted |
| RAGAS | `gpt-4o-mini` | n/a | `0` client / RAGAS 1e-8 |
| Embeddings | `text-embedding-3-small` | n/a | n/a |

Phase 1 and Phase 3a do not use an LLM extraction model. Phase 3a builds the typed graph directly from `MetaQA/kb.txt` (oracle ceiling). Phase 3b uses schema-constrained `gpt-6-luna` extraction.

## Protocol

- Chunking: **256 / 32** tokens, all MetaQA phases
- Retrieval depth: **top_k = 20**
- Shared MetaQA prompt: `5de2288e17855119b17760d9b5d33689e05c0e8c1f8329288513ec2d0a4b98fa`
- Full eval: `824acf90381417721406146ff6263dc8ecb5bcde9f95d3dff70ed72300849cc6`
- Full corpus manifest: `4e10fa73d3ae839aae6b0a0ac76c4fe6e8b4c97beb6dbe414c30bf7ca5fd1f29`
- Neo4j/Qdrant stores: isolated per phase/track; see each run's `run_meta.json` and `ingest_meta.json`

## Headline results — full MetaQA (212 q / 994 docs)

| Run | n | gold_recall | all_found | evidence_recall | exact_match | llm_judge | ragas_faithfulness | ragas_context_recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase1 | 212 | 0.5538 | 0.4764 | 0.6550 | 0.2500 | 0.4858 | 0.7967 | 0.6294 |
| phase2 | 212 | 0.7568 | 0.6557 | 0.8406 | 0.3160 | 0.6604 | 0.7853 | 0.7511 |
| phase3b | 212 | 0.7787 | 0.6840 | 0.8929 | 0.2783 | 0.6934 | 0.8093 | 0.7823 |
| phase3 | 212 | 0.8758 | 0.8113 | 0.9657 | 0.3113 | 0.8208 | 0.7505 | 0.8676 |

## Mini30 — plumbing certificate (30 q / 100 docs)

| Run | n | gold_recall | all_found | evidence_recall | exact_match | llm_judge | ragas_faithfulness | ragas_context_recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase1 | 30 | 0.6806 | 0.6000 | 0.8133 | 0.3667 | 0.6333 | 0.8792 | 0.7367 |
| phase2 | 30 | 0.8222 | 0.7333 | 0.9589 | 0.3333 | 0.7333 | 0.7192 | 0.9194 |
| phase3b | 30 | 0.8389 | 0.7667 | 0.9833 | 0.2667 | 0.7667 | 0.6813 | 0.8883 |
| phase3 | 30 | 0.8889 | 0.8333 | 1.0000 | 0.3667 | 0.8667 | 0.7720 | 0.9139 |
| closed_book | 30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — |

Mini12 and local runs are included under `metaqa_mini/` and `local/`. Mini tracks are wiring/plumbing only; do not make phase-comparison claims from them.

## Paired conclusions (full n=212)

- **P2 − P1:** all_found +0.179 [95% CI +0.127, +0.236], McNemar p<0.0001.
- **P3a − P2:** all_found +0.156 [+0.104, +0.212], p<0.0001.
- **P3a − P1:** all_found +0.335 [+0.274, +0.396], p<0.0001.
- **P3b − P2:** all_found +0.028 [−0.019, +0.080], p=0.36; gold +0.022 [−0.018, +0.063]; **evidence +0.052 [+0.022, +0.083]**; judge +0.033 [−0.019, +0.090], p=0.31.
- **P3a − P3b:** all_found +0.127 [+0.085, +0.175], p<0.0001.

Interpretation: graph retrieval beats dense vector on full MetaQA; Phase 3b improves evidence recall over Phase 2 but not answer-level metrics; Phase 3a is the oracle ceiling.

## Checkpoint layout

```
checkpoints/v1/
├── README.md                 # this file
├── MANIFEST.json             # run list, configs, hashes, metrics
├── CHECKSUMS.sha256          # compact tracked files
├── ARCHIVE.sha256            # local detailed archive hash
├── local/                    # 2-question smoke runs
├── metaqa/                   # full: phase1, phase2, phase3, phase3b
├── metaqa_mini/              # mini12: phase1, phase2, phase2_no_neighbours
├── metaqa_mini30/            # mini30: all four phases + closed_book + no_neighbours
└── archive/                  # local-only results_detailed_v1.tar.zst
```

Only the compact artifacts are tracked: metrics, per-question rows, run/ingest metadata, eval reports, and answer-level results. The large `results_detailed.json` files (answers + retrieved contexts) are archived locally in `archive/results_detailed_v1.tar.zst` and are **not committed**, because they contain MetaQA document context text.

### Verify

```bash
cd /Users/caio/Caio_Things/Programacao/UFS/rag-trilogy
shasum -a 256 -c checkpoints/v1/CHECKSUMS.sha256
shasum -a 256 -c checkpoints/v1/ARCHIVE.sha256
zstd -t checkpoints/v1/archive/results_detailed_v1.tar.zst
```

## Caveats

- **Phase 2 ingest metadata** records `errors: "unknown"` because the LlamaIndex `apoc.meta.subGraph` schema refresh was terminated after the graph was already complete. The answer run has 0 errors; future ingests disable the refresh.
- Phase 2's graph has **4,522 embedded text chunks plus 985 document-level stubs** with no text/embedding; the text chunks and vector index are the retrieval set.
- **Full closed-book control is not run.** Only mini30 closed-book exists.
- **Phase 3a is an oracle ceiling**, built from `kb.txt`; it is not a fair competitor to P2/3b.
- **Mini tracks are plumbing only**; never compare phases on them.
- **HotpotQA is retired** from headline comparisons; historical results are not part of this freeze.
- v1 RAGAS uses the now-old `gpt-4o-mini`. Future RAGAS changes belong to a new eval checkpoint.
- The checkpoint files are snapshots; do not overwrite them. New judge/RAGAS/model experiments should create children, not edit v1.

## Provenance / source

- Source results: `results/` (gitignored operational directory)
- `MANIFEST.json` records every included file's SHA-256, every run's metrics, fingerprints and ingest metadata.
- This freeze was created from the repository state before the checkpoint commit; the annotated tag points at the freeze commit.
