# Phase 1b — dense vs BM25 vs hybrid (retrieval-only)

No generation. Dense ranking is the frozen P1 top-20 context; BM25 is built
over the same 4,522 MetaQA chunks; hybrid is reciprocal-rank fusion (RRF, k=60).

| Retriever | n | gold-in-context recall | all gold in context | evidence recall | all-gold count |
|---|---:|---:|---:|---:|---:|
| dense | 212 | 0.6741 | 0.5896 | 0.6550 | 125 |
| bm25 | 212 | 0.7932 | 0.7500 | 0.7678 | 159 |
| hybrid | 212 | 0.7871 | 0.7406 | 0.7842 | 157 |

Dense is the same retrieval as Phase 1. Its **evidence recall .6550** matches the
frozen P1 number; the gold/all-gold values here are context coverage, not the
generated-answer metrics (P1 gold .5538 / all_found .4764).

This is a retrieval-only screen. Run answers only if the hybrid closes a
meaningful part of the dense→graph gap.

## Reading

BM25 is much stronger than dense retrieval on gold-in-context coverage
(.750 vs .590 all-gold) and also improves evidence recall (.768 vs .655). RRF
hybrids keep the evidence gain (.784) but give back some all-gold coverage (.741).
This is a strong signal that Phase 1b is not just a formality: a lexical baseline
deserves a full answer run before attributing the Phase 2 gain to the graph.

## Full-answer P1b vs P2 (v2 protocol)

P1b uses the same v2 answer protocol as the other phases.

| Generator | P1b all_found | P2 all_found | P2−P1b | McNemar p |
|---|---:|---:|---:|---:|
| Luna | .6226 | .6179 | −.0047 [−.0519, +.0425] | 1.00 |
| Qwen | .5802 | .6038 | +.0236 [−.0236, +.0660] | .42 |

P2 has significantly higher gold/evidence recall in both generators:
- Luna: gold +.0436, evidence +.0570
- Qwen: gold +.0475, evidence +.0564

**Interpretation:** once a lexical hybrid is introduced, the graph's answer-level advantage over vector retrieval disappears on this set. P2 still retrieves better evidence, but the generated-answer difference is not detectable. This is exactly why Phase 1b was the missing control.
