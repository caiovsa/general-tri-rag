# Phase 3b vs Phase 3a — graph gap diagnostic

No API calls. Phase 3a is restricted to the **same 994 movie titles** present in
Phase 3b, so this compares extraction coverage rather than the full KB scope.

## Graph scope

- 3b movies: **994**
- 3a movies: **16427**
- 3a movies not in 3b (scope difference): **15433**
- 3b persons: **14108**
- 3a persons reachable from the 994 movies: **2822**

## Relation comparison on the 994-movie intersection

| Relation | 3b facts | 3a induced facts | TP | 3b-only | 3a-only | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| starred_actors | 15353 | 2161 | 2126 | 13227 | 35 | 0.138 | 0.984 | 0.243 |
| directed_by | 1129 | 985 | 965 | 164 | 20 | 0.855 | 0.980 | 0.913 |
| written_by | 1995 | 1282 | 1058 | 937 | 224 | 0.530 | 0.825 | 0.646 |
| has_genre | 1519 | 929 | 20 | 1499 | 909 | 0.013 | 0.022 | 0.016 |
| in_language | 1106 | 252 | 220 | 886 | 32 | 0.199 | 0.873 | 0.324 |
| release_year | 993 | 994 | 963 | 30 | 31 | 0.970 | 0.969 | 0.969 |
| **micro** | 22095 | 6603 | 5352 | 16743 | 1251 | 0.242 | 0.811 | — |

### Gold evidence-path coverage

- Questions: **212**
- Evidence triples checked: **990**
- 3b triple coverage: **0.896**
- 3a triple coverage: **1.000**
- 3b complete paths: **158 / 212**
- 3a complete paths: **212 / 212**
- Unsupported relations in eval triples: `[]`

The raw 26.6k vs 138k relationship totals are mostly a **scope** difference:
3a includes the whole KB, 3b only facts extracted from the 994-document corpus.
The induced-subgraph comparison above isolates extraction coverage/precision.

## Reading

The gap is not simply "3b is missing 80% of the relationships." On the shared
994-movie universe, 3b has **high recall** for actors (.984), directors (.980), and
release years (.969), but it also has many facts that are not in the KB. Those
extras may be genuine cast/crew names from the documents (the KB is selective),
so precision against the KB should not be read as truth precision. The genre
overlap is especially low (.013), consistent with free-text genre phrases not
matching KB genre tokens.

The most decision-relevant number is **complete gold paths: 3b 158/212 vs 3a
212/212**, with 89.6% vs 100% evidence-triple coverage. The remaining gap therefore
comes from both missing path facts and extraction noise/over-extraction, not from
the 26.6k-vs-138k raw scope difference.
