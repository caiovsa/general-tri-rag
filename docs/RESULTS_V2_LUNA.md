# RAG Trilogy — v2 luna checkpoint

**Frozen:** 2026-11-18T01:39:37+00:00  
**Results tag:** `luna_v2`  
**Source commit:** `52b21ff61513e83676f2b078b03919041516471d`  
**Answer model:** `gpt-6-luna` via `default`, reasoning `none`, temperature `0`  
**OpenRouter provider pin:** `none`  
**Judge:** `gpt-6.1-sol` reasoning `low`  
**RAGAS:** `gpt-6-luna` reasoning `none`, version `0.4.4`

This is a v2 run: answer reasoning is explicitly disabled and temperature is 0.

| Run | n | gold | all found | evidence | judge | RAGAS faith | RAGAS ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| closed_book_luna_v2 | 212 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — |
| phase1_luna_v2 | 212 | 0.5390 | 0.4717 | 0.6550 | 0.4764 | 0.6303 | 0.5156 |
| phase1_p1b_luna_v2 | 212 | 0.6697 | 0.6226 | 0.7842 | 0.6274 | 0.6762 | 0.6259 |
| phase2_luna_v2 | 212 | 0.7133 | 0.6179 | 0.8412 | 0.6226 | 0.7319 | 0.7441 |
| phase3_luna_v2 | 212 | 0.8405 | 0.7736 | 0.9657 | 0.7783 | 0.8036 | 0.8833 |
| phase3b_luna_v2 | 212 | 0.7415 | 0.6415 | 0.8929 | 0.6462 | 0.7667 | 0.7530 |

## Important notes

- Phase 2/P3b graphs are the v1 graphs; v2 changes the answer model/protocol, not extraction.
- RAGAS 0.4.4 with `gpt-6-luna@none` produced valid scores on most questions, but a few per-phase samples were null; means ignore those nulls. RAGAS remains secondary.
- v1 generated each question once; no three-run noise check was performed. v1 variance is unmeasured.
- The detailed contexts archive is local and not committed: it contains MetaQA document text.

## Files

Compact metrics, per-question rows, run metadata and checksums are tracked. The archive
is `checkpoints/v2/luna/archive/results_detailed_luna_v2.tar.zst`.
