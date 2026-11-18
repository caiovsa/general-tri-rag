#!/usr/bin/env python3
"""Final v2 truncation audit (no API).

Reports context-budget truncation and answer-ending signals for v2 full, P1b and
closed-book runs. RAGAS/judge are not touched.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ["phase1_luna_v2", "phase1_p1b_luna_v2", "phase2_luna_v2", "phase3_luna_v2", "phase3b_luna_v2", "closed_book_luna_v2",
        "phase1_qwen_v2", "phase1_p1b_qwen_v2", "phase2_qwen_v2", "phase3_qwen_v2", "phase3b_qwen_v2", "closed_book_qwen_v2"]

def load(run):
    d=ROOT/"results/metaqa"/run
    detailed=json.load((d/"results_detailed.json").open())
    pq={r["id"]:r for r in map(json.loads,(d/"per_question.jsonl").open())}
    return detailed,pq

rows=[]
for run in RUNS:
    d=ROOT/"results/metaqa"/run
    if not (d/"results_detailed.json").exists(): continue
    detailed,pq=load(run)
    counts=[]; truncated=0; short=0; suspicious=[]
    for item in detailed:
        ctx=item.get("contexts") or []; counts.append(len(ctx))
        ret=item.get("retrieval") or {}
        if ret.get("truncated") is True: truncated += 1
        if len(ctx) < 20: short += 1
        ans=(item.get("answer") or "").strip()
        if not ans or re.search(r"[,;:\-]$|(?:\band|\bor|\bthe)$", ans, re.I) or ans.endswith("..."):
            suspicious.append(item.get("id"))
    rows.append({"run":run,"n":len(detailed),"ctx_mean":sum(counts)/len(counts),
                 "ctx_min":min(counts) if counts else 0,"short_lt20":short,
                 "retrieval_truncated":truncated,"suspicious_endings":len(suspicious)})

def fmt(v): return f"{v:.2f}" if isinstance(v,float) else str(v)
lines=["# v2 truncation audit","",
       "Context truncation = retrieved context below the 20-chunk budget or an explicit graph `truncated` flag.",
       "Answer endings are heuristic only; the answer API did not persist finish_reason/max tokens.","",
       "| Run | n | ctx mean | ctx min | ctx < 20 | graph truncated | suspicious answer endings |",
       "|---|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['run']} | {r['n']} | {fmt(r['ctx_mean'])} | {r['ctx_min']} | {r['short_lt20']} | {r['retrieval_truncated']} | {r['suspicious_endings']} |")
lines += ["", "Interpretation: context-budget truncation is visible from the saved retrieval counts. Answer", "truncation cannot be proven retroactively because finish_reason and max-token usage were not stored", "for the answer calls; rerun with those fields if this audit needs to be definitive.","",
          "P1/P1b always return 20 contexts by construction. Closed-book has 0 by design.", ""]
(ROOT/"docs/RESULTS_TRUNCATION.md").write_text("\n".join(lines),encoding="utf-8")
for r in rows: print(r)
print("wrote docs/RESULTS_TRUNCATION.md")
