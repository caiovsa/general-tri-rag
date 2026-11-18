# RAG Trilogy — v2 qwen checkpoint

**Frozen:** 2026-11-18T01:39:39+00:00  
**Results tag:** `qwen_v2`  
**Source commit:** `52b21ff61513e83676f2b078b03919041516471d`  
**Answer model:** `openrouter/qwen/qwen3.5-9b` via `openrouter`, reasoning `none`, temperature `0`  
**OpenRouter provider pin:** `primeintellect/bf16`  
**Judge:** `gpt-6.1-sol` reasoning `low`  
**RAGAS:** `gpt-6-luna` reasoning `none`, version `0.4.4`

This is a v2 run: answer reasoning is explicitly disabled and temperature is 0.

| Run | n | gold | all found | evidence | judge | RAGAS faith | RAGAS ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| closed_book_qwen_v2 | 212 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — |
| phase1_p1b_qwen_v2 | 212 | 0.6483 | 0.5802 | 0.7842 | 0.5849 | 0.6145 | 0.6414 |
| phase1_qwen_v2 | 212 | 0.5414 | 0.4670 | 0.6555 | 0.4717 | 0.5879 | 0.5142 |
| phase2_qwen_v2 | 212 | 0.6958 | 0.6038 | 0.8406 | 0.6085 | 0.6877 | 0.7374 |
| phase3_qwen_v2 | 212 | 0.8053 | 0.7264 | 0.9657 | 0.7406 | 0.7308 | 0.8812 |
| phase3b_qwen_v2 | 212 | 0.7320 | 0.6462 | 0.8935 | 0.6604 | 0.6986 | 0.7723 |

## Important notes

- Phase 2/P3b graphs are the v1 graphs; v2 changes the answer model/protocol, not extraction.
- RAGAS 0.4.4 with `gpt-6-luna@none` produced valid scores on most questions, but a few per-phase samples were null; means ignore those nulls. RAGAS remains secondary.
- v1 generated each question once; no three-run noise check was performed. v1 variance is unmeasured.
- The detailed contexts archive is local and not committed: it contains MetaQA document text.

## Files

Compact metrics, per-question rows, run metadata and checksums are tracked. The archive
is `checkpoints/v2/qwen/archive/results_detailed_qwen_v2.tar.zst`.
