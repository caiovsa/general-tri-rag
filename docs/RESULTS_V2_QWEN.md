# RAG Trilogy — v2 qwen checkpoint

**Frozen:** 2026-11-17T20:05:03+00:00  
**Results tag:** `qwen_v2`  
**Source commit:** `5f8510170842662ba6b8cd10e6d6dc8df96ccc1a`  
**Answer model:** `openrouter/qwen/qwen3.5-9b` via `openrouter`, reasoning `none`, temperature `0`  
**OpenRouter provider pin:** `primeintellect/bf16`  
**Judge:** `gpt-6.1-sol` reasoning `low`  
**RAGAS:** `gpt-6-luna` reasoning `none`

This is a v2 run: answer reasoning is explicitly disabled and temperature is 0.

| Run | n | gold | all found | evidence | judge | RAGAS faith | RAGAS ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| phase1_qwen_v2 | 212 | 0.5414 | 0.4670 | 0.6555 | 0.4717 | NaN | 0.5055 |
| phase2_qwen_v2 | 212 | 0.6958 | 0.6038 | 0.8406 | 0.6085 | 1.0000 | 0.7378 |
| phase3_qwen_v2 | 212 | 0.8053 | 0.7264 | 0.9657 | 0.7406 | 1.0000 | 0.8448 |
| phase3b_qwen_v2 | 212 | 0.7320 | 0.6462 | 0.8935 | 0.6604 | NaN | 0.7440 |

## Important notes

- Phase 2/P3b graphs are the v1 graphs; v2 changes the answer model/protocol, not extraction.
- RAGAS with `gpt-6-luna@none` produced `No statements` warnings and NaN faithfulness on different phases for each model; RAGAS is secondary and should not be treated as a clean/comparable score.
- v1 generated each question once; no three-run noise check was performed. v1 variance is unmeasured.
- The detailed contexts archive is local and not committed: it contains MetaQA document text.

## Files

Compact metrics, per-question rows, run metadata and checksums are tracked. The archive
is `checkpoints/v2/qwen/archive/results_detailed_qwen_v2.tar.zst`.
