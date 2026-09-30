# Experiment Log

> Dated record of what has been frozen, and the protocol rules every comparison run follows.
> `make check-config DATASET=<hotpot|metaqa|metaqa_mini>` prints a run's resolved config;
> `make freeze-verify` re-checks the frozen artifacts. Track details: `docs/metaqa-track.md`.

## Freezes

### 2026-09-29 — MetaQA v1 (full track)

| | |
|---|---|
| Questions | `MetaQA/metaqa_eval_v1.jsonl` — 212 kept (1-hop 75, 2-hop 67, 3-hop 70) out of 240 sampled |
| Corpus | `MetaQA/corpus_manifest.txt` — 994 docs (620 must-have + 190 hard + 190 random requested, 6 titles rejected by verification) |
| Git | commit `f9129b7` (2026-09-29, "feat(metaqa): add evidence-gated MetaQA track"); **no git tag** — `MetaQA/` is gitignored, so the artifacts are not in git history and the freeze is pinned by the checksum file plus this commit |
| sha256 | eval `824acf90381417721406146ff6263dc8ecb5bcde9f95d3dff70ed72300849cc6`; corpus `4e10fa73d3ae839aae6b0a0ac76c4fe6e8b4c97beb6dbe414c30bf7ca5fd1f29` |
| Evidence | 240 candidates → 212 kept (28 dropped, `MetaQA/audit_dropped.jsonl`); fetch used 2030 API requests, 6 titles rejected |
| Verify | `make freeze-verify` (`sha256sum -c MetaQA/FROZEN.sha256`) |

### 2026-09-29 — MetaQA mini (pilot slice)

| | |
|---|---|
| Questions | `MetaQA/metaqa_eval_mini.jsonl` — 12 questions (4 per hop), a subsequence of the v1 eval |
| Corpus | `MetaQA/corpus_manifest_mini.txt` — 50 docs (35 evidence + 15 distractor) |
| Git | commit `2d15e22` (2026-09-29, "feat(metaqa): add hop-balanced mini track and retrieval diagnostics"); **no git tag** — same reason as v1 |
| sha256 | eval `48a62a9aedef6a282acff1c20b04ac8fbc80b3054ae6e0a36b08d155277dd270`; corpus `f83ea02b90b098842331615ca14dc1450c9979d73178424fecb1b195953702a4` (`MetaQA/FROZEN_mini.sha256`) |
| Built by | `uv run python -m scripts.metaqa_make_mini` (seed 42, defaults `--per-hop 4 --docs 50`); it only reads the frozen v1 artifacts |
| Verify | `make freeze-verify` (`sha256sum -c MetaQA/FROZEN_mini.sha256`) |

## Protocol rules

Every cross-phase comparison must satisfy all of these. A run that violates one is not
comparable to a run that does not; note the deviation in the log instead of silently
mixing results.

1. **Same top_k and chunk budget for all phases.** Retrieval depth (`TOP_K` / `--top-k`) and
   the chunk size/overlap are part of the treatment, not free knobs: Phase 1 and Phase 2 must
   retrieve the same amount of text for a question. Both are recorded in `run_meta.json`.
2. **Similarity threshold off on MetaQA.** Phase 1's `SimilarityPostprocessor` filtering must
   be disabled (threshold 0 / no filter) so both phases return a full top-k list; a
   thresholded run is not comparable to an unthresholded one.
3. **Same `[Movie: <title>]` tagging and chunking in all phases.** `shared/ingest.py::MovieTagger`
   plus the MetaQA chunk size/overlap (256/32) apply identically to Phase 1, Phase 2 and
   Phase 3; no phase may retag, rechunk, or re-glob the corpus. The corpus is always
   `MetaQA/corpus_manifest.txt` (or the mini manifest), never a fresh glob of `docs/`.
4. **Pinned answer and judge models at temperature 0.** One pinned `GENERATION_MODEL` answers
   and judges every run; no model or sampling change mid-experiment.
   *Current deviation:* `shared/llm.py::generate_completion` still defaults to
   `temperature=1`, and the judge call passes no temperature — fix before recording clean
   baselines.
5. **No per-question tuning.** Prompts, retrieval parameters, thresholds and models are fixed
   before the run and applied to every question; a question that fails is a result, not an
   edit.

Frozen artifacts are inputs: a phase run never rebuilds `metaqa_eval_v1.jsonl`,
`corpus_manifest.txt`, or the mini equivalents (steps 2–4 rebuild only, and a rebuilt eval
must be re-frozen and re-checksummed before use).
