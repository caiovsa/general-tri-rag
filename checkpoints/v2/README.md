# RAG Trilogy — v2 full MetaQA (Luna + Qwen)

**Status:** provisional — v2 is not complete. The current tag
`metaqa-results-v2-luna-qwen` was force-moved after being pushed; do not move it
again. Final tag will be a new name (e.g. `metaqa-results-v2.1`) only after Phase 1b,
closed-book, noise, and judge-sensitivity checks.  
**Scope:** full MetaQA, four phases (P1/P2/P3a/P3b), two answer models.

## Answer models

| Model | Route | Reasoning | Temperature | Provider pin |
|---|---|---|---|---|
| `gpt-6-luna` | OpenAI | `none` | `0` | n/a |
| `openrouter/qwen/qwen3.5-9b` | OpenRouter | `none` | `0` | **`primeintellect/bf16`**, no fallbacks |

Both runs use:

- Judge: `gpt-6.1-sol`, reasoning `low`
- RAGAS: `gpt-6-luna`, reasoning `none`, temperature `0`, **RAGAS 0.4.4**
- Extraction: `gpt-6-luna`; Phase 2/3b graphs are the v1 graphs (not re-extracted)
- Prompt/chunking/top_k: shared prompt, 256/32, top_k 20

## Results

| Phase | Luna all found | Luna judge | Luna RAGAS 0.4.4 faith/ctx | Qwen all found | Qwen judge | Qwen RAGAS 0.4.4 faith/ctx |
|---|---:|---:|---:|---:|---:|---:|
| P1 | .4717 | .4764 | .6303 / .5156 | .4670 | .4717 | .5879 / .5142 |
| P2 | .6179 | .6226 | .7319 / .7441 | .6038 | .6085 | .6877 / .7374 |
| P3a | .7736 | .7783 | .8036 / .8833 | .7264 | .7406 | .7308 / .8812 |
| P3b | .6415 | .6462 | .7667 / .7530 | .6462 | .6604 | .6986 / .7723 |

RAGAS coverage: 206–212/212 valid per phase; the means ignore the few null samples.

Detailed tables are in [luna/README.md](luna/README.md) and [qwen/README.md](qwen/README.md).
The Markdown reports are `docs/RESULTS_V2_LUNA.md` and `docs/RESULTS_V2_QWEN.md`.

## RAGAS note

RAGAS 0.4.4 (`gpt-6-luna`, reasoning `none`, `max_tokens=4096`) produced valid scores
on 206–212 of 212 questions per phase. A few null samples are excluded from the means.
RAGAS 0.4.4 uses a newer metric implementation than v1's 0.1.21, so its scores are not
directly comparable to v1 RAGAS. RAGAS remains a secondary diagnostic.

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

- **Phase 1b full answers** (both generators); only retrieval-only screening exists
- Full closed-book under v2
- Three-run v2 noise check / flip-rate report
- Judge sensitivity (old judge on a subset)
- Mini-track v2 runs
- Any Qwen extraction re-ingest (Qwen is an answer model here; graphs are v1)

Qwen answer runs record provider, reasoning=none, temperature=0, provider pin and
allow_fallbacks. **Answer `max_tokens` was not set or recorded**; only the RAGAS
`max_tokens=4096` is recorded. The 0-reasoning-token check was a probe, not persisted
per benchmark answer.

## v1 limitation

v1 generated each of the 212 questions once; no three-run noise check was performed,
so v1 variance is unmeasured. v2 is the explicit `reasoning=none, temperature=0`
protocol. The frozen v1 tag/checksums are unchanged.
