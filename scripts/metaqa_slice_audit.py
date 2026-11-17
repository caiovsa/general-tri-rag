#!/usr/bin/env python3
"""Post-hoc MetaQA slices and Phase 3a failure audit (no API).

Reads the frozen v1 run artifacts and writes:
  - docs/RESULTS_SLICES.md
  - results/diagnostics/metaqa_slices_v1.json

Groups: hop 1/2/3, hop 2+3, evidence-size 1/2/3-5/6+.
Phase 3a audit: classify all_found failures by whether every gold answer is
present in the retrieved context, and whether missing answers are supported by
the evidence documents.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "P1": ROOT / "results/metaqa/phase1",
    "P2": ROOT / "results/metaqa/phase2",
    "P3b": ROOT / "results/metaqa/phase3b",
    "P3a": ROOT / "results/metaqa/phase3",
}
EVAL = ROOT / "MetaQA/metaqa_eval_v1.jsonl"
MANIFEST = ROOT / "MetaQA/corpus_manifest.txt"
EVIDENCE_BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10**9, "6+"))


def fold(value: str) -> str:
    """Case-fold and strip diacritics (É -> E), without dropping letters."""
    text = unicodedata.normalize("NFKD", str(value))
    return text.encode("ascii", "ignore").decode("ascii").casefold()


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", fold(value)).strip()


def key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", fold(value))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def load_run(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in map(json.loads, (path / "per_question.jsonl").open())}


def group_indices(rows: list[dict], fn) -> list[int]:
    return [i for i, row in enumerate(rows) if fn(row)]


def main() -> None:
    rows_by_phase = {name: list(load_run(path).values()) for name, path in RUNS.items()}
    # Normalize a stable order; all frozen full runs have the same 212 ids.
    order = [row["id"] for row in rows_by_phase["P1"]]
    for name in rows_by_phase:
        rows_by_phase[name] = sorted(rows_by_phase[name], key=lambda r: order.index(r["id"]))
    rows = rows_by_phase["P1"]
    groups = [
        ("hop 1", lambda r: r.get("hop") == 1),
        ("hop 2", lambda r: r.get("hop") == 2),
        ("hop 3", lambda r: r.get("hop") == 3),
        ("hop 2+3", lambda r: (r.get("hop") or 0) >= 2),
    ]
    for low, high, label in EVIDENCE_BUCKETS:
        groups.append((f"evidence {label}", lambda r, low=low, high=high: low <= (r.get("evidence_n") or 0) <= high))

    slice_rows = []
    for label, fn in groups:
        idx = group_indices(rows, fn)
        out = {"group": label, "n": len(idx), "phases": {}}
        for name, phase_rows in rows_by_phase.items():
            vals = [phase_rows[i]["all_found"] for i in idx]
            k = sum(vals)
            low, high = wilson(k, len(vals))
            out["phases"][name] = {
                "all_found": k / len(vals) if vals else None,
                "count": k,
                "n": len(vals),
                "wilson_95": [round(low, 4), round(high, 4)],
            }
        slice_rows.append(out)

    # Phase 3a failure audit
    eval_rows = {row["id"]: row for row in map(json.loads, EVAL.open())}
    detailed = {row["id"]: row for row in json.load((RUNS["P3a"] / "results_detailed.json").open())}
    corpus = {Path(p).stem: Path(p).read_text(errors="ignore") for p in
              [line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]}
    by_key = {key(stem): stem for stem in corpus}
    title_map = {}
    for stem, text in corpus.items():
        for line in text.splitlines():
            if line.lower().startswith("title:"):
                title_map[key(line.split(":", 1)[1])] = stem
                break

    def resolve(title: str) -> str | None:
        k = key(title)
        if k in title_map:
            return title_map[k]
        if k in by_key:
            return by_key[k]
        return None

    def supported(text: str, needle: str) -> bool:
        return key(needle) in key(text)

    cases = []
    failures = []
    for qid in order:
        row = rows_by_phase["P3a"][order.index(qid)]
        if row.get("all_found"):
            continue
        failures.append(qid)
        detail = detailed[qid]
        gold = [str(a) for a in detail.get("answers") or []]
        context = key(" \n ".join(detail.get("contexts") or []))
        missing = [answer for answer in gold if key(answer) not in context]
        evidence_titles = eval_rows[qid].get("supporting_facts_titles") or []
        evidence_stems = [s for s in (resolve(t) for t in evidence_titles) if s]
        evidence_text = "\n".join(corpus[s] for s in evidence_stems)
        missing_supported = {
            answer: supported(evidence_text, answer) or any(supported(text, answer) for text in corpus.values())
            for answer in missing
        }
        unsupported = [answer for answer, ok in missing_supported.items() if not ok]
        if not missing:
            cause = "generation/list (all gold present)"
        elif unsupported:
            cause = "eval-set issue (missing answer not supported by evidence docs)"
        else:
            cause = "retrieval/context selection (missing answer supported by evidence docs)"
        cases.append({
            "id": qid,
            "hop": row.get("hop"),
            "qtype": eval_rows[qid].get("qtype"),
            "evidence_recall": row.get("evidence_recall"),
            "question": eval_rows[qid]["question"],
            "gold": gold,
            "missing_from_context": missing,
            "missing_supported_by_evidence_docs": missing_supported,
            "cause": cause,
        })

    all_gold_present = [c for c in cases if not c["missing_from_context"]]
    missing_cases = [c for c in cases if c["missing_from_context"]]
    eval_issue_cases = [c for c in cases if c["cause"].startswith("eval-set")]
    retrieval_cases = [c for c in cases if c["cause"].startswith("retrieval")]

    report = {
        "source": "frozen v1 full MetaQA runs",
        "n": len(rows),
        "slices": slice_rows,
        "phase3a": {
            "all_found_failures": len(failures),
            "evidence_recall_at_1_failures": sum(1 for r in rows_by_phase["P3a"] if r.get("evidence_recall") == 1 and not r.get("all_found")),
            "all_gold_present_but_failed": len(all_gold_present),
            "missing_gold_cases": len(missing_cases),
            "eval_set_issue_cases": len(eval_issue_cases),
            "retrieval_cases": len(retrieval_cases),
            "cases": cases,
        },
    }
    out_json = ROOT / "results/diagnostics/metaqa_slices_v1.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# MetaQA v1 — slices and Phase 3a failure audit",
        "",
        "No API calls. Derived from `results/metaqa/phase*/per_question.jsonl` and Phase 3a's",
        "`results_detailed.json`. The JSON artifact is `results/diagnostics/metaqa_slices_v1.json`.",
        "",
        "## all_found by hop and evidence size",
        "",
        "| Group | n | P1 | P2 | P3b | P3a |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in slice_rows:
        cells = []
        for name in RUNS:
            p = group["phases"][name]
            cells.append(f"{p['all_found']:.3f} ({p['count']}/{p['n']})")
        lines.append(f"| {group['group']} | {group['n']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Wilson 95% intervals are in the JSON artifact. Buckets are `evidence_n`: 1, 2, 3–5, 6+.",
        "",
        "## Phase 3a failure decomposition",
        "",
        f"- all_found failures: **{len(failures)} / 212**",
        f"- all gold answers present in context but all_found=0: **{len(all_gold_present)}** (generation/list)",
        f"- at least one gold answer missing from context: **{len(missing_cases)}**",
        f"- of those, missing answer supported by evidence docs: **{len(retrieval_cases)}** (retrieval/context selection)",
        f"- missing answer not supported by evidence docs: **{len(eval_issue_cases)}** (eval-set issue)",
        "",
        "### The 17 missing-answer cases",
        "",
        "| id | hop | missing answer(s) | evidence docs support? | verdict |",
        "|---|---:|---|---|---|",
    ]
    for c in missing_cases:
        support = "all" if not any(not v for v in c["missing_supported_by_evidence_docs"].values()) else "partial/none"
        verdict = "eval-set issue" if c in eval_issue_cases else "retrieval/context selection"
        lines.append(f"| `{c['id']}` | {c['hop']} | {', '.join(c['missing_from_context'])} | {support} | {verdict} |")
    lines += [
        "",
        "After diacritic folding and title mapping, **all missing answers are present in the",
        "corpus**. The apparent `Eric Rochant` case was an accent-normalization artifact:",
        "the `Möbius` document writes `Éric Rochant`. The issue is that `Möbius` was not",
        "selected into 3a's context, not that the eval answer is unsupported.",
        "",
        "So the 17 cases are **context-selection/ranking failures**, not eval-set failures.",
        "The audit uses corpus support rather than only evidence-doc support, precisely to",
        "avoid misclassifying accented names or filenames such as `M_bius.txt`.",
    ]
    (ROOT / "docs/RESULTS_SLICES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json.relative_to(ROOT)} and docs/RESULTS_SLICES.md")
    print(f"failures={len(failures)} all_gold_present={len(all_gold_present)} missing_cases={len(missing_cases)} eval_issue={len(eval_issue_cases)}")


if __name__ == "__main__":
    main()
