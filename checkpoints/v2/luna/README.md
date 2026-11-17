# RAG Trilogy — v2 luna checkpoint

**Frozen:** 2026-11-17T20:04:59+00:00  
**Results tag:** `luna_v2`  
**Source commit:** `5f8510170842662ba6b8cd10e6d6dc8df96ccc1a`  
**Answer model:** `gpt-6-luna` via `default`, reasoning `none`, temperature `0`  
**OpenRouter provider pin:** `none`  
**Judge:** `gpt-6.1-sol` reasoning `low`  
**RAGAS:** `gpt-6-luna` reasoning `none`

This is a v2 run: answer reasoning is explicitly disabled and temperature is 0.

| Run | n | gold | all found | evidence | judge | RAGAS faith | RAGAS ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| phase1_luna_v2 | 212 | 0.5390 | 0.4717 | 0.6550 | 0.4764 | 0.1000 | 0.5291 |
| phase2_luna_v2 | 212 | 0.7133 | 0.6179 | 0.8412 | 0.6226 | NaN | 0.7192 |
| phase3_luna_v2 | 212 | 0.8405 | 0.7736 | 0.9657 | 0.7783 | NaN | 0.8421 |
| phase3b_luna_v2 | 212 | 0.7415 | 0.6415 | 0.8929 | 0.6462 | 1.0000 | 0.7441 |

## Important notes

- Phase 2/P3b graphs are the v1 graphs; v2 changes the answer model/protocol, not extraction.
- RAGAS with `gpt-6-luna@none` produced `No statements` warnings and NaN faithfulness on different phases for each model; RAGAS is secondary and should not be treated as a clean/comparable score.
- v1 generated each question once; no three-run noise check was performed. v1 variance is unmeasured.
- The detailed contexts archive is local and not committed: it contains MetaQA document text.

## Files

Compact metrics, per-question rows, run metadata and checksums are tracked. The archive
is `checkpoints/v2/luna/archive/results_detailed_luna_v2.tar.zst`.
