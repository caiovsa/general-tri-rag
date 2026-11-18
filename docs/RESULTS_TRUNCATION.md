# v2 truncation audit

Context truncation = retrieved context below the 20-chunk budget or an explicit graph `truncated` flag.
Answer endings are heuristic only; the answer API did not persist finish_reason/max tokens.

| Run | n | ctx mean | ctx min | ctx < 20 | graph truncated | suspicious answer endings |
|---|---:|---:|---:|---:|---:|---:|
| phase1_luna_v2 | 212 | 20.00 | 20 | 0 | 0 | 0 |
| phase1_p1b_luna_v2 | 212 | 20.00 | 20 | 0 | 0 | 0 |
| phase2_luna_v2 | 212 | 19.89 | 13 | 7 | 0 | 0 |
| phase3_luna_v2 | 212 | 14.27 | 1 | 113 | 0 | 0 |
| phase3b_luna_v2 | 212 | 17.49 | 1 | 48 | 0 | 0 |
| closed_book_luna_v2 | 212 | 0.00 | 0 | 212 | 0 | 0 |
| phase1_qwen_v2 | 212 | 20.00 | 20 | 0 | 0 | 0 |
| phase1_p1b_qwen_v2 | 212 | 20.00 | 20 | 0 | 0 | 0 |
| phase2_qwen_v2 | 212 | 19.89 | 13 | 7 | 0 | 0 |
| phase3_qwen_v2 | 212 | 14.27 | 1 | 113 | 0 | 0 |
| phase3b_qwen_v2 | 212 | 17.49 | 1 | 48 | 0 | 0 |
| closed_book_qwen_v2 | 212 | 0.00 | 0 | 212 | 0 | 0 |

Interpretation: context-budget truncation is visible from the saved retrieval counts. Answer
truncation cannot be proven retroactively because finish_reason and max-token usage were not stored
for the answer calls; rerun with those fields if this audit needs to be definitive.

P1/P1b always return 20 contexts by construction. Closed-book has 0 by design.
