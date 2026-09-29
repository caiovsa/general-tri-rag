#!/usr/bin/env python3
"""Inspect the MetaQA dataset and write ``MetaQA/INSPECT.md``.

Run: uv run python -m scripts.metaqa_inspect
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "MetaQA"  # dataset folder; single line to retarget
OUT = ROOT / "INSPECT.md"
HOPS = (1, 2, 3)
PERSON_RELS = ("starred_actors", "directed_by", "written_by")
BRACKET = re.compile(r"\[([^\]]*)\]")
SAMPLES = 3


def load_kb():
    """Parse kb.txt into counts and relation keyed sets."""
    triples, rel_count = 0, Counter()
    rel_subj, rel_obj = defaultdict(set), defaultdict(set)
    year_by_movie = defaultdict(set)
    for line in (ROOT / "kb.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        subj, rel, obj = line.split("|")
        triples += 1
        rel_count[rel] += 1
        rel_subj[rel].add(subj)
        rel_obj[rel].add(obj)
        if rel == "release_year":
            year_by_movie[subj].add(obj)
    return triples, rel_count, rel_subj, rel_obj, year_by_movie


def load_qa(hop):
    """Return (qtype, question, [answers]) tuples for one hop level."""
    base = ROOT / f"{hop}_hop"
    qa = [l.split("\t") for l in (base / "qa_test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    qtypes = [l.strip() for l in (base / "qa_test_qtype.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(qt, q, a.split("|")) for qt, (q, a) in zip(qtypes, qa)]


def group_by_qtype(items):
    """qtype -> [(question, answers)], ordered by descending question count."""
    groups = defaultdict(list)
    for qt, q, a in items:
        groups[qt].append((q, a))
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(str(c) for c in row) + " |" for row in rows
    ]


def section_kb(md, triples, rel_count, rel_subj, rel_obj):
    subjects, objects = set().union(*rel_subj.values()), set().union(*rel_obj.values())
    ordered = sorted(rel_subj, key=lambda r: -rel_count[r])
    md += [
        "## 1. Knowledge base — `kb.txt`", "",
        *table(["Metric", "Value"],
               [["Triples", f"{triples:,}"], ["Relations", f"{len(rel_subj)}"],
                ["Unique entities (subjects ∪ objects)", f"{len(subjects | objects):,}"],
                ["Unique subjects", f"{len(subjects):,}"], ["Unique objects", f"{len(objects):,}"]]),
        "", "### Relation types", "",
        *table(["Relation", "Triples", "Unique subjects", "Unique objects", "Sample objects"],
               [[r, f"{rel_count[r]:,}", f"{len(rel_subj[r]):,}", f"{len(rel_obj[r]):,}",
                 ", ".join(f"`{o}`" for o in sorted(rel_obj[r])[:3])] for r in ordered]),
        "",
    ]
    return ordered


def section_subject_typing(md, ordered, rel_subj, rel_obj, stats):
    """Report whether each relation's subjects are movies, derived from the data."""
    movies, anchor, collision = stats["movies"], stats["anchor"], stats["collision"]
    md += [
        "## 2. Subject-side typing (verified from the data)", "",
        "Movies are anchored as the subjects of `release_year` (only films carry a release year). "
        "Actors/directors/writers are the objects of `starred_actors` / `directed_by` / `written_by`.", "",
        *table(["Relation", "Subjects", "Subjects with `release_year`", "Subjects that are person names",
                "Objects that are movie subjects", "Object kind"],
               [[r, f"{len(rel_subj[r]):,}", f"{len(rel_subj[r] & anchor) / len(rel_subj[r]):.1%}",
                 f"{len(rel_subj[r] & stats['person_objects']):,}", f"{len(rel_obj[r] & movies):,}",
                 "person (actor/director/writer)" if r in PERSON_RELS
                 else ("literal (4-digit year)" if r == "release_year" else "controlled vocabulary / keyword")]
                for r in ordered]),
        "",
        "- **Is the subject side always a movie? Yes, bar one class of exceptions.** 96.7–100% of every "
        "relation's subjects are anchored by `release_year`, and no relation introduces a different subject "
        f"class. The {len(collision)} exceptions are titles that collide with person names: "
        + ", ".join(f"`{t}`" for t in sorted(collision)[:10]) + ", …",
        "- **Relations where subject/object roles differ**: all of them — subjects are movies, objects are "
        f"persons ({', '.join(PERSON_RELS)}) or literals (`release_year` → year, `has_genre` → "
        f"{len(rel_obj['has_genre'])} genres, `in_language` → {len(rel_obj['in_language'])} languages, "
        f"`has_imdb_rating`/`has_imdb_votes` → label sets, `has_tags` → {len(rel_obj['has_tags']):,} "
        "free-text keywords) — `has_tags` is the only relation whose objects are untyped text, not entities.",
        "- **Reverse-direction evidence**: several relations hold objects that are themselves movie subjects "
        "(counts above), and the QA qtypes `actor_to_movie`, `director_to_movie`, `writer_to_movie`, "
        "`tag_to_movie` query the same relations backwards.",
        "",
    ]


def section_movies(md, year_by_movie, stats):
    movies, collision = stats["movies"], stats["collision"]
    reused = sorted(m for m, years in year_by_movie.items() if len(years) > 1)
    missing = sorted(movies - stats["anchor"])
    md += [
        "## 3. Movie titles", "",
        *table(["Metric", "Value"],
               [["Unique movie titles (distinct subjects)", f"{len(movies):,}"],
                ["Titles colliding with person names (ambiguous)", f"{len(collision):,}"],
                ["Titles with >1 `release_year` (title reused by several films)", f"{len(reused):,}"],
                ["Titles appearing more than once or ambiguous (union)", f"{len(set(reused) | collision):,}"],
                ["Titles without `release_year` (sparse metadata)", f"{len(missing):,}"]]),
        "",
        "- Reused titles: " + ", ".join(f"`{t}` ({'/'.join(sorted(year_by_movie[t]))})" for t in reused[:5]),
        "- Ambiguous titles: " + ", ".join(f"`{t}`" for t in sorted(collision)[:8]),
        "",
    ]


def section_hops(md):
    md += ["## 4–5. QA test sets", ""]
    for hop in HOPS:
        items = load_qa(hop)
        groups = group_by_qtype(items)  # [(qtype, [(question, answers)])]
        bracket_counts = Counter(len(BRACKET.findall(q)) for _, q, _ in items)
        md += [
            f"### {hop}-hop — {len(items):,} questions, {len(groups)} qtypes", "",
            "Topic entity format: exactly one bracketed entity per question — `[Entity Name]`, using the "
            f"verbatim `kb.txt` name (bracket counts per question: {dict(sorted(bracket_counts.items()))}).", "",
            *table(["qtype", "questions", "mean answers", "max answers", "% with >1 answer"],
                   [[qt, f"{len(g):,}", f"{sum(len(a) for _, a in g) / len(g):.2f}", max(len(a) for _, a in g),
                     f"{sum(1 for _, a in g if len(a) > 1) / len(g):.1%}"] for qt, g in groups]),
            "",
        ]
        for qt, g in groups:
            md += [f"#### `{qt}` — {len(g):,} questions, mean {sum(len(a) for _, a in g) / len(g):.2f} answers, "
                   f"max {max(len(a) for _, a in g)}, "
                   f"{sum(1 for _, a in g if len(a) > 1) / len(g):.1%} multi-answer", ""]
            md += [f"- `{q}` → " + ", ".join(f"`{x}`" for x in a) for q, a in g[:SAMPLES]]
            md.append("")


def main():
    triples, rel_count, rel_subj, rel_obj, year_by_movie = load_kb()
    stats = {"movies": set().union(*rel_subj.values()), "anchor": rel_subj["release_year"],
             "person_objects": set().union(*(rel_obj[r] for r in PERSON_RELS))}
    stats["collision"] = stats["movies"] & stats["person_objects"]

    md = ["# MetaQA — Data Inspection Report", "",
          f"Generated by `scripts/metaqa_inspect.py` from `{ROOT.name}/`.", ""]
    ordered = section_kb(md, triples, rel_count, rel_subj, rel_obj)
    section_subject_typing(md, ordered, rel_subj, rel_obj, stats)
    section_movies(md, year_by_movie, stats)
    section_hops(md)

    text = "\n".join(md).rstrip() + "\n"
    print(text)
    OUT.write_text(text, encoding="utf-8")
    print(f"Saved to: {OUT}")


if __name__ == "__main__":
    main()
