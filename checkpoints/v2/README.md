# RAG Trilogy — v2 (in preparation)

**Status:** not frozen. This directory will receive the v2 protocol results after the
post-freeze retrieval-only checks.

## Protocol

- Answer: `gpt-6-luna`, `reasoning_effort=none`, `temperature=0`
- Extraction: `gpt-6-luna` (keep current extraction protocol unless an ablation is recorded)
- Judge: `gpt-6.1-sol`, `reasoning_effort=low` (primary); old `gpt-5.6-luna` on a subset as a sensitivity check
- RAGAS: `gpt-6-luna`, `reasoning_effort=none`, `temperature=0`
- Embeddings: `text-embedding-3-small`

The judge and generator are both in the GPT-6 family; the paper must note this.

## v1 limitation written down before v2 starts

**v1 (tag `metaqa-results-v1-default-medium`) generated each of the 212 full-MetaQA
questions exactly once. No three-run noise/replay check was performed, so v1 answer
variance is unmeasured.** v1 used provider-default medium reasoning; it is not a
no-reasoning baseline. Do not modify the frozen v1 tag.

v2 will run the noise check on frozen contexts and report the per-question flip rate
under the primary `reasoning=none, temperature=0` protocol. If the variance is near
zero, that is itself the result.

## Planned order

0. Freeze v1 — done
1. Slices + 3a failure decomposition
2. 3b→3a induced-subgraph diagnostic
3. Phase 1b hybrid, retrieval-only
4. Easy cross-ablation (3b graph + generic retriever; Phase 2 graph + typed retriever is not identifiable)
5. [STOP — run v2 closed-book with the user]
6. v2 noise check + flip rate
7. Final cross-ablation answers only if justified
8. Qwen replay
9. Freeze v2 (Luna + Qwen together)

A Luna-only v2 checkpoint is committed after step 7 so a Qwen problem cannot block
the freeze; the v2 tag is applied only after Qwen is included.
