# RAG Trilogy — v2 full MetaQA (Luna + Qwen)

**Status:** frozen after the Qwen run; tag `metaqa-results-v2-luna-qwen`.  
**Scope:** full MetaQA, four phases (P1/P2/P3a/P3b), two answer models.

## Answer models

| Model | Route | Reasoning | Temperature | Provider pin |
|---|---|---|---|---|
| `gpt-6-luna` | OpenAI | `none` | `0` | n/a |
| `openrouter/qwen/qwen3.5-9b` | OpenRouter | `none` | `0` | **`primeintellect/bf16`**, no fallbacks |

Both runs use:

- Judge: `gpt-6.1-sol`, reasoning `low`
- RAGAS: `gpt-6-luna`, reasoning `none`, temperature `0`
- Extraction: `gpt-6-luna`; Phase 2/3b graphs are the v1 graphs (not re-extracted)
- Prompt/chunking/top_k: shared prompt, 256/32, top_k 20

## Results

| Phase | Luna all found | Luna judge | Qwen all found | Qwen judge |
|---|---:|---:|---:|---:|
| P1 | .4717 | .4764 | .4670 | .4717 |
| P2 | .6179 | .6226 | .6038 | .6085 |
| P3a | .7736 | .7783 | .7264 | .7406 |
| P3b | .6415 | .6462 | .6462 | .6604 |

Detailed tables are in [luna/README.md](luna/README.md) and [qwen/README.md](qwen/README.md).
The Markdown reports are `docs/RESULTS_V2_LUNA.md` and `docs/RESULTS_V2_QWEN.md`.

## RAGAS warning

RAGAS with `gpt-6-luna@none` produced `No statements` warnings and NaN faithfulness
on different phases for each model. Those RAGAS numbers are **diagnostic only** and
are not a clean comparison. Do not use them as a headline result.

## Layout

```text
checkpoints/v2/
├── luna/
│   ├── README.md, MANIFEST.json, CHECKSUMS.sha256
│   ├── ARCHIVE.sha256
│   ├── metaqa/phase{1,2,3,3b}_luna_v2/
│   └── archive/results_detailed_luna_v2.tar.zst   # local only
└── qwen/
    ├── README.md, MANIFEST.json, CHECKSUMS.sha256
    ├── ARCHIVE.sha256
    ├── metaqa/phase{1,2,3,3b}_qwen_v2/
    └── archive/results_detailed_qwen_v2.tar.zst   # local only
```

Compact metrics, per-question rows, run metadata and checksums are tracked. The
large detailed-context archives are local because they contain MetaQA document text.

## Not yet run

- Full closed-book under v2
- Three-run v2 noise check / flip-rate report
- Mini-track v2 runs
- Any Qwen extraction re-ingest (Qwen is an answer model here; graphs are v1)

## v1 limitation

v1 generated each of the 212 questions once; no three-run noise check was performed,
so v1 variance is unmeasured. v2 is the explicit `reasoning=none, temperature=0`
protocol. The frozen v1 tag/checksums are unchanged.
