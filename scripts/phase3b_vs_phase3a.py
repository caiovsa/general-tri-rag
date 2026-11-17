#!/usr/bin/env python3
"""Compare Phase 3b's extracted graph with Phase 3a's oracle KB graph (no API).

3a contains the whole MetaQA KB, while 3b contains only facts extracted from the
994-document corpus. To avoid comparing different universes, all 3a comparisons
are restricted to the 994 movies present in 3b.

Outputs:
  results/diagnostics/phase3b_vs_phase3a.json
  docs/RESULTS_3B_GAP.md
"""
from __future__ import annotations

import json
from pathlib import Path

from neo4j import GraphDatabase

from shared.config import settings

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "MetaQA/metaqa_eval_v1.jsonl"
RELATIONS = {
    "starred_actors": ("STARRED_ACTORS", "Person", "name"),
    "directed_by": ("DIRECTED_BY", "Person", "name"),
    "written_by": ("WRITTEN_BY", "Person", "name"),
    "has_genre": ("HAS_GENRE", "Genre", "name"),
    "in_language": ("IN_LANGUAGE", "Language", "name"),
    "release_year": ("RELEASED_IN", "Year", "value"),
}


def norm(value) -> str:
    return str(value).strip().casefold()


def facts(driver, database: str, titles: list[str]) -> dict[str, set[tuple[str, str]]]:
    out: dict[str, set[tuple[str, str]]] = {}
    with driver.session(database=database) as session:
        for relation, (edge, label, prop) in RELATIONS.items():
            rows = session.run(
                f"MATCH (m:Movie)-[:{edge}]->(o:{label}) "
                "WHERE m.title IN $titles "
                f"RETURN m.title AS movie, o.{prop} AS obj",
                titles=titles,
            )
            out[relation] = {(str(r["movie"]), str(r["obj"])) for r in rows if r["movie"] is not None and r["obj"] is not None}
    return out


def metrics(b: set, a: set) -> dict:
    tp = len(b & a)
    extra = len(b - a)
    missing = len(a - b)
    precision = tp / len(b) if b else None
    recall = tp / len(a) if a else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {
        "3b_facts": len(b), "3a_induced_facts": len(a), "tp": tp,
        "3b_only": extra, "3a_only": missing,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main() -> None:
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    try:
        with driver.session(database="metaqa-ontology-graph") as s:
            a_titles = [r["title"] for r in s.run("MATCH (m:Movie) RETURN m.title AS title ORDER BY title")]
        with driver.session(database="metaqa-ontology-b-graph") as s:
            b_titles = [r["title"] for r in s.run("MATCH (m:Movie) RETURN m.title AS title ORDER BY title")]
        a_only_titles = sorted(set(a_titles) - set(b_titles))
        b_facts = facts(driver, "metaqa-ontology-b-graph", b_titles)
        a_facts = facts(driver, "metaqa-ontology-graph", b_titles)
        relation_rows = []
        totals_b = totals_a = totals_tp = 0
        for relation in RELATIONS:
            m = metrics(b_facts[relation], a_facts[relation])
            m["relation"] = relation
            relation_rows.append(m)
            totals_b += m["3b_facts"]; totals_a += m["3a_induced_facts"]; totals_tp += m["tp"]
        overall = {
            "3b_facts": totals_b, "3a_induced_facts": totals_a, "tp": totals_tp,
            "3b_only": totals_b - totals_tp, "3a_only": totals_a - totals_tp,
            "precision": totals_tp / totals_b if totals_b else None,
            "recall": totals_tp / totals_a if totals_a else None,
        }

        # Gold evidence-path coverage on the frozen eval set.
        eval_rows = [json.loads(line) for line in RUNS.read_text().splitlines() if line.strip()]
        path_rows = []
        b_complete = a_complete = 0
        b_total_triples = a_total_triples = b_hit = a_hit = 0
        unsupported_relations = set()
        for row in eval_rows:
            triples = row.get("evidence_triples") or []
            b_cov, a_cov = [], []
            for movie, relation, obj in triples:
                if relation not in RELATIONS:
                    unsupported_relations.add(relation)
                    continue
                fact = (str(movie), str(obj))
                b_ok = fact in b_facts[relation]
                a_ok = fact in a_facts[relation]
                b_cov.append(b_ok); a_cov.append(a_ok)
                b_hit += b_ok; a_hit += a_ok
            b_total_triples += len(b_cov); a_total_triples += len(a_cov)
            b_complete += bool(b_cov) and all(b_cov)
            a_complete += bool(a_cov) and all(a_cov)
            path_rows.append({
                "id": row["id"], "hop": row.get("hop"),
                "triples": len(b_cov),
                "3b_fraction": sum(b_cov) / len(b_cov) if b_cov else None,
                "3a_fraction": sum(a_cov) / len(a_cov) if a_cov else None,
            })
        path_summary = {
            "questions": len(eval_rows),
            "triples_checked": b_total_triples,
            "3b_triple_coverage": b_hit / b_total_triples if b_total_triples else None,
            "3a_triple_coverage": a_hit / a_total_triples if a_total_triples else None,
            "3b_complete_paths": b_complete,
            "3a_complete_paths": a_complete,
            "unsupported_relations": sorted(unsupported_relations),
        }

        # Person coverage: 3a restricted to facts on 3b movies vs 3b persons.
        with driver.session(database="metaqa-ontology-graph") as s:
            a_persons = {r["name"] for r in s.run(
                "MATCH (m:Movie)-[:STARRED_ACTORS|DIRECTED_BY|WRITTEN_BY]->(p:Person) "
                "WHERE m.title IN $titles RETURN DISTINCT p.name AS name", titles=b_titles)}
        with driver.session(database="metaqa-ontology-b-graph") as s:
            b_persons = {r["name"] for r in s.run("MATCH (p:Person) RETURN p.name AS name")}
        coverage = {
            "3b_movies": len(b_titles), "3a_movies": len(a_titles),
            "3a_movies_not_in_3b": len(a_only_titles),
            "3b_persons": len(b_persons), "3a_persons_on_3b_movies": len(a_persons),
            "3b_persons_also_in_3a": len({p for p in b_persons if p in a_persons}),
        }

        result = {
            "scope_note": "All 3a comparisons are restricted to the 994 movie titles present in 3b.",
            "graph_scope": coverage,
            "relations": relation_rows,
            "micro": overall,
            "gold_paths": path_summary,
            "path_samples": path_rows,
        }
        out = ROOT / "results/diagnostics/phase3b_vs_phase3a.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

        lines = [
            "# Phase 3b vs Phase 3a — graph gap diagnostic",
            "",
            "No API calls. Phase 3a is restricted to the **same 994 movie titles** present in",
            "Phase 3b, so this compares extraction coverage rather than the full KB scope.",
            "",
            "## Graph scope",
            "",
            f"- 3b movies: **{coverage['3b_movies']}**",
            f"- 3a movies: **{coverage['3a_movies']}**",
            f"- 3a movies not in 3b (scope difference): **{coverage['3a_movies_not_in_3b']}**",
            f"- 3b persons: **{coverage['3b_persons']}**",
            f"- 3a persons reachable from the 994 movies: **{coverage['3a_persons_on_3b_movies']}**",
            "",
            "## Relation comparison on the 994-movie intersection",
            "",
            "| Relation | 3b facts | 3a induced facts | TP | 3b-only | 3a-only | precision | recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in relation_rows:
            def pct(v):
                return "—" if v is None else f"{v:.3f}"
            lines.append(f"| {r['relation']} | {r['3b_facts']} | {r['3a_induced_facts']} | {r['tp']} | {r['3b_only']} | {r['3a_only']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} |")
        def pct(v): return "—" if v is None else f"{v:.3f}"
        lines += [
            f"| **micro** | {overall['3b_facts']} | {overall['3a_induced_facts']} | {overall['tp']} | {overall['3b_only']} | {overall['3a_only']} | {pct(overall['precision'])} | {pct(overall['recall'])} | — |",
            "",
            "### Gold evidence-path coverage",
            "",
            f"- Questions: **{path_summary['questions']}**",
            f"- Evidence triples checked: **{path_summary['triples_checked']}**",
            f"- 3b triple coverage: **{pct(path_summary['3b_triple_coverage'])}**",
            f"- 3a triple coverage: **{pct(path_summary['3a_triple_coverage'])}**",
            f"- 3b complete paths: **{path_summary['3b_complete_paths']} / {path_summary['questions']}**",
            f"- 3a complete paths: **{path_summary['3a_complete_paths']} / {path_summary['questions']}**",
            f"- Unsupported relations in eval triples: `{path_summary['unsupported_relations']}`",
            "",
            "The raw 26.6k vs 138k relationship totals are mostly a **scope** difference:",
            "3a includes the whole KB, 3b only facts extracted from the 994-document corpus.",
            "The induced-subgraph comparison above isolates extraction coverage/precision.",
        ]
        (ROOT / "docs/RESULTS_3B_GAP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote results/diagnostics/phase3b_vs_phase3a.json and docs/RESULTS_3B_GAP.md")
        print("micro:", overall)
        print("gold paths:", path_summary)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
