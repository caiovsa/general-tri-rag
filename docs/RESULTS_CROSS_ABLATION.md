# Easy cross-ablation — 3b graph, generic retriever (retrieval-only)

No generation. The only graph modification is a temporary `MENTIONS` compatibility
view from chunks to the movie/person nodes already in the 3b graph; it is removed
after the run.

| Retriever on 3b graph | n | gold-in-context recall | all gold in context | evidence recall |
|---|---:|---:|---:|---:|
| typed 3a/3b retriever | 212 | 0.8817 | 0.8113 | 0.8929 |
| Phase 2 generic retriever | 212 | 0.7737 | 0.7028 | 0.7805 |

The Phase 2 graph + typed retriever cell is **not identifiable**: the Phase 2 graph
has generic free-text relationships and no typed Movie/Person edges.

The generic retriever is a compatibility check, not a clean 2x2 identification.

## Reading

On the same 3b graph, the typed retriever is clearly better than the generic
Phase 2 retriever on all three retrieval measures. That is evidence that the
retrieval algorithm matters independently of the graph source. It is not a clean
2x2: the Phase 2 graph cannot be typed faithfully, so the graph-vs-retriever
interaction remains partially confounded.
