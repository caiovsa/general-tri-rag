# MetaQA v1 — slices and Phase 3a failure audit

No API calls. Derived from `results/metaqa/phase*/per_question.jsonl` and Phase 3a's
`results_detailed.json`. The JSON artifact is `results/diagnostics/metaqa_slices_v1.json`.

## all_found by hop and evidence size

| Group | n | P1 | P2 | P3b | P3a |
|---|---:|---:|---:|---:|---:|
| hop 1 | 75 | 0.840 (63/75) | 0.947 (71/75) | 0.920 (69/75) | 0.933 (70/75) |
| hop 2 | 67 | 0.358 (24/67) | 0.597 (40/67) | 0.657 (44/67) | 0.746 (50/67) |
| hop 3 | 70 | 0.200 (14/70) | 0.400 (28/70) | 0.457 (32/70) | 0.743 (52/70) |
| hop 2+3 | 137 | 0.277 (38/137) | 0.496 (68/137) | 0.555 (76/137) | 0.745 (102/137) |
| evidence 1 | 66 | 0.894 (59/66) | 0.985 (65/66) | 0.939 (62/66) | 0.939 (62/66) |
| evidence 2 | 64 | 0.375 (24/64) | 0.703 (45/64) | 0.672 (43/64) | 0.859 (55/64) |
| evidence 3-5 | 63 | 0.238 (15/63) | 0.429 (27/63) | 0.540 (34/63) | 0.730 (46/63) |
| evidence 6+ | 19 | 0.158 (3/19) | 0.105 (2/19) | 0.316 (6/19) | 0.474 (9/19) |

Wilson 95% intervals are in the JSON artifact. Buckets are `evidence_n`: 1, 2, 3–5, 6+.

## Phase 3a failure decomposition

- all_found failures: **40 / 212**
- all gold answers present in context but all_found=0: **23** (generation/list)
- at least one gold answer missing from context: **17**
- of those, missing answer supported by evidence docs: **17** (retrieval/context selection)
- missing answer not supported by evidence docs: **0** (eval-set issue)

### The 17 missing-answer cases

| id | hop | missing answer(s) | evidence docs support? | verdict |
|---|---:|---|---|---|
| `metaqa-2hop-06001` | 2 | Eric Schweig, Jodie Foster, Val Kilmer, James Garner, Sam Shepard, Mel Gibson | all | retrieval/context selection |
| `metaqa-2hop-11669` | 2 | 1962 | all | retrieval/context selection |
| `metaqa-2hop-04497` | 2 | Louis Leterrier | all | retrieval/context selection |
| `metaqa-2hop-12714` | 2 | Shrek 2, Shrek the Halls, Shrek | all | retrieval/context selection |
| `metaqa-2hop-05583` | 2 | Moby Dick | all | retrieval/context selection |
| `metaqa-2hop-12463` | 2 | Rough Night in Jericho | all | retrieval/context selection |
| `metaqa-2hop-06704` | 2 | Charles Lederer, Billy Wilder, John Melson | all | retrieval/context selection |
| `metaqa-3hop-12938` | 3 | Luc Dardenne, Eric Rochant, Alexandre Aja | all | retrieval/context selection |
| `metaqa-3hop-10570` | 3 | 1969 | all | retrieval/context selection |
| `metaqa-3hop-11167` | 3 | 2009 | all | retrieval/context selection |
| `metaqa-3hop-07850` | 3 | Sam Claflin, Corey Sevier, Sean Bean | all | retrieval/context selection |
| `metaqa-3hop-08742` | 3 | Comedy, Crime | all | retrieval/context selection |
| `metaqa-3hop-05068` | 3 | Madame de La Fayette | all | retrieval/context selection |
| `metaqa-3hop-07066` | 3 | Patricia Highsmith, Michael Ondaatje | all | retrieval/context selection |
| `metaqa-3hop-05775` | 3 | Madeleine Stowe | all | retrieval/context selection |
| `metaqa-3hop-05592` | 3 | Arthur Penn, Franklin J. Schaffner | all | retrieval/context selection |
| `metaqa-3hop-03485` | 3 | Horror | all | retrieval/context selection |

After diacritic folding and title mapping, **all missing answers are present in the
corpus**. The apparent `Eric Rochant` case was an accent-normalization artifact:
the `Möbius` document writes `Éric Rochant`. The issue is that `Möbius` was not
selected into 3a's context, not that the eval answer is unsupported.

So the 17 cases are **context-selection/ranking failures**, not eval-set failures.
The audit uses corpus support rather than only evidence-doc support, precisely to
avoid misclassifying accented names or filenames such as `M_bius.txt`.

## Reading

The phase differences are concentrated in **hop 2/3** and in **larger evidence
sets**; single-evidence questions are close to ceiling. Phase 3a's gap is mixed:
23/40 failures have all gold answers in context (generation/list), while 17/40
are context-selection failures. None of the 17 is an eval-set failure after
diacritic folding and title mapping; the `Eric Rochant` scare was `Éric Rochant`
in `M_bius.txt`.
