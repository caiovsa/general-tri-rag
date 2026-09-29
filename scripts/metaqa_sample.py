#!/usr/bin/env python3
"""Build a clean MetaQA candidate set with KB-derived evidence.

Run: uv run python -m scripts.metaqa_sample

Pipeline (order matters — evidence is computed for the whole test pool, then
filtered, and only then sampled):
  1. Load kb.txt, mark "unsafe" movie titles (title reused by several films,
     missing release_year, or colliding with an actor/director/writer name).
  2. Drop the qtypes `movie_to_tags` / `tag_to_movie`.
  3. Derive, per question, the movies that must be read to answer it by
     following the chain encoded in the qtype name.
  4. Self-check: re-derive the answer set from the KB and compare with the gold
     answers (topic movie included / excluded). Qtypes below MIN_AGREEMENT are
     dropped and reported.
  5. Drop questions with unsafe evidence, too much evidence, or too many answers.
  6. Sample --per-hop questions per hop, spread evenly over surviving qtypes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PERSON_ROLES = ("actor", "director", "writer")
ROLE_REL = {
    "actor": "starred_actors",
    "director": "directed_by",
    "writer": "written_by",
    "year": "release_year",
    "genre": "has_genre",
    "language": "in_language",
    "tag": "has_tags",
}
PERSON_RELS = {ROLE_REL[role] for role in PERSON_ROLES}
ROLE_ALIASES = {"tags": "tag"}
EXCLUDED_QTYPES = {"movie_to_tags", "tag_to_movie"}
MIN_AGREEMENT = 0.95
BRACKET = re.compile(r"\[([^\]]*)\]")
EVIDENCE_BUCKETS = ("1", "2", "3-5", "6-10", ">10")
REQUIRED_KEYS = {"id", "hop", "qtype", "question", "question_raw", "topic_entity",
                 "answers", "answer", "evidence_movies", "evidence_triples"}


def resolve_root() -> Path:
    """First dataset folder containing kb.txt (data_metaqa/raw is canonical)."""
    for cand in (Path("data_metaqa/raw"), Path("MetaQA"), Path("data_metaqa")):
        if (cand / "kb.txt").exists():
            return cand
    raise SystemExit("kb.txt not found in data_metaqa/raw, MetaQA or data_metaqa")


def load_kb(root: Path):
    """Return (movies, forward[rel][movie], backward[rel][person], triples_by_movie)."""
    forward, backward = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    triples = defaultdict(set)
    for line in (root / "kb.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        movie, rel, obj = line.split("|")
        forward[rel][movie].add(obj)
        triples[movie].add((rel, obj))
        backward[rel][obj].add(movie)
    movies = {m for rel in forward for m in forward[rel]}
    return movies, forward, backward, triples


def unsafe_movies(movies, forward, backward) -> dict[str, set[str]]:
    """Movies that cannot be mapped to a single Wikipedia page."""
    person_names = set().union(*(backward[rel] for rel in PERSON_RELS))
    years = forward["release_year"]
    return {
        "reused_title": {m for m in movies if len(years.get(m, ())) > 1},
        "missing_release_year": {m for m in movies if not years.get(m)},
        "person_name_collision": movies & person_names,
    }


def chain_of(qtype: str) -> list[str]:
    return [ROLE_ALIASES.get(part, part) for part in qtype.split("_to_")]


def derive(chain, topic, kb, include_topic_self: bool):
    """Walk the chain from the topic entity.

    Returns (answer_set, movie_nodes, triples) or None when the topic does not
    hold the role the chain starts with. `include_topic_self` controls whether
    the topic movie may reappear as an intermediate/answer movie.
    """
    movies, forward, backward, _ = kb
    if chain[0] == "movie":
        if topic not in movies:
            return None
    elif topic not in backward[ROLE_REL[chain[0]]]:
        return None

    nodes, movie_nodes, used = {topic}, set(), set()
    if chain[0] == "movie":
        movie_nodes.add(topic)

    for src, dst in zip(chain, chain[1:]):
        rel = ROLE_REL[dst] if src == "movie" else ROLE_REL[src]
        found, step_used = set(), set()
        for node in nodes:
            if src == "movie":
                for obj in forward[rel].get(node, set()):
                    found.add(obj)
                    step_used.add((node, rel, obj))
            else:
                for movie in backward[rel].get(node, set()):
                    found.add(movie)
                    step_used.add((movie, rel, node))
        # `include_topic_self` off drops the topic entity only where it is a genuine
        # self-reference: back at its own role position (chain[0]) or in the final set.
        # Dropping it everywhere would also delete an unrelated movie that merely
        # shares the person's name (e.g. the film "Antwone Fisher").
        if not include_topic_self and (dst == chain[0] or dst == chain[-1]):
            found.discard(topic)
            step_used = {t for t in step_used if t[0 if dst == "movie" else 2] != topic}
        used |= step_used
        nodes = found
        if dst == "movie":
            movie_nodes |= nodes
        if not nodes:
            break
    return nodes, movie_nodes, used


def load_test(hop: int, root: Path):
    """Return (index, qtype, question_raw, question, topic, [gold answers]) rows."""
    base = root / f"{hop}_hop"
    qa = [l.split("\t") for l in (base / "qa_test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    qtypes = [l.strip() for l in (base / "qa_test_qtype.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for idx, (qt, (raw, answers)) in enumerate(zip(qtypes, qa)):
        match = BRACKET.search(raw)
        rows.append((idx, qt, raw, BRACKET.sub(r"\1", raw), match.group(1) if match else "",
                     answers.split("|")))
    return rows


def self_check(rows, kb):
    """Per-qtype exact-set agreement of the KB re-derivation vs the gold answers.

    Reported twice: over the whole pool and over the questions whose derived
    evidence is safe (the pool step 5 keeps), because unsafe topics — reused
    titles with merged nodes — cannot be reproduced by the gold answers.
    """
    unsafe = set().union(*unsafe_movies(kb[0], kb[1], kb[2]).values())
    stats = defaultdict(Counter)
    for _, qt, _, _, topic, gold in rows:
        chain = chain_of(qt)
        stats[qt]["n"] += 1
        for variant in (True, False):
            derived = derive(chain, topic, kb, include_topic_self=variant)
            stats[qt][f"match_{variant}"] += int(derived is not None and derived[0] == set(gold))
            if derived is not None and not (derived[1] & unsafe):
                stats[qt][f"n_safe_{variant}"] += 1
                stats[qt][f"safe_{variant}"] += int(derived[0] == set(gold))
    verdicts = {}
    for qt, counts in stats.items():
        pool = {v: counts[f"match_{v}"] / counts["n"] for v in (True, False)}
        safe = {v: counts[f"safe_{v}"] / counts[f"n_safe_{v}"] if counts[f"n_safe_{v}"] else 0.0
                for v in (True, False)}
        best = max(safe, key=lambda v: (safe[v], v))
        verdicts[qt] = {"n": counts["n"], "pool": pool, "safe": safe,
                        "n_safe": counts[f"n_safe_{best}"], "variant": best,
                        "best": safe[best], "unsafe_mismatches": counts["n"] - counts["match_True"]}
    return verdicts


def sample_evenly(pool, quota, rng):
    """Even spread over qtypes, redistributing unfilled quota to the rest."""
    for items in pool.values():
        rng.shuffle(items)
    picked, cursor = [], {qt: 0 for qt in pool}
    while len(picked) < quota:
        eligible = [qt for qt in pool if cursor[qt] < len(pool[qt])]
        if not eligible:
            break
        fewest = min(cursor[qt] for qt in eligible)
        qt = rng.choice([q for q in eligible if cursor[q] == fewest])
        picked.append(pool[qt][cursor[qt]])
        cursor[qt] += 1
    return picked


def evidence_bucket(size: int) -> str:
    return "1" if size <= 1 else "2" if size == 2 else "3-5" if size <= 5 else "6-10" if size <= 10 else ">10"


def evidence_histogram(rows, title: str):
    """len(evidence_movies) per hop, buckets 1 / 2 / 3-5 / 6-10."""
    print(f"\nEvidence size histogram — {title}")
    for hop in (1, 2, 3):
        sizes = [len(r["evidence_movies"]) for r in rows if r["hop"] == hop]
        counts = Counter(evidence_bucket(s) for s in sizes)
        body = "  ".join(f"{b}={counts[b]:,} ({counts[b] / len(sizes):.0%})"
                          for b in EVIDENCE_BUCKETS if counts.get(b)) or "none"
        print(f"  {hop}-hop (n={len(sizes):,}): {body}")


def validate_file(path: Path) -> int:
    """Every line must parse as JSON and carry the required keys; fail loudly otherwise."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"INVALID JSON on line {number} of {path}: {e}") from e
        missing = REQUIRED_KEYS - row.keys()
        if missing:
            raise SystemExit(f"Missing keys on line {number} of {path}: {sorted(missing)}")
    return len(lines)


def verify_rows(rows, kb, unsafe, max_evidence, max_answers):
    """Post-write invariants: safe evidence, triples in the KB, answers justified by evidence."""
    _, _, _, triples = kb
    for row in rows:
        assert row["evidence_movies"] and not set(row["evidence_movies"]) & unsafe, row["id"]
        assert len(row["evidence_movies"]) <= max_evidence, row["id"]
        assert len(row["answers"]) <= max_answers, row["id"]
        assert all(tuple(t[1:]) in triples[t[0]] for t in row["evidence_triples"]), row["id"]
        chain = chain_of(row["qtype"])
        rel = ROLE_REL[chain[-1] if chain[-2] == "movie" else chain[-2]]
        justified = {t[0 if chain[-1] == "movie" else 2] for t in row["evidence_triples"] if t[1] == rel}
        assert set(row["answers"]) <= justified, (row["id"], set(row["answers"]) - justified)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-hop", type=int, default=40)
    ap.add_argument("--max-evidence", type=int, default=10)
    ap.add_argument("--max-answers", type=int, default=8)
    ap.add_argument("--min-evidence-1hop", type=int, default=1)
    ap.add_argument("--min-evidence-2hop", type=int, default=2)
    ap.add_argument("--min-evidence-3hop", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    min_evidence = {1: args.min_evidence_1hop, 2: args.min_evidence_2hop, 3: args.min_evidence_3hop}

    root = resolve_root()
    out_path = (root.parent if root.name == "raw" else root) / "metaqa_candidates.jsonl"
    rng = random.Random(args.seed)

    kb = load_kb(root)
    unsafe_sets = unsafe_movies(kb[0], kb[1], kb[2])
    unsafe = set().union(*unsafe_sets.values())
    print(f"root: {root}")
    print(f"KB: {len(kb[0]):,} movie nodes | unsafe: {len(unsafe):,} "
          f"({', '.join(f'{k}={len(v):,}' for k, v in unsafe_sets.items())})")
    print(f"excluded qtypes: {', '.join(sorted(EXCLUDED_QTYPES))}\n")

    # --- 3. derive evidence for the WHOLE pool, 4. self-check per qtype ---
    per_hop_rows = {hop: load_test(hop, root) for hop in (1, 2, 3)}
    verdicts = self_check((r for rows in per_hop_rows.values() for r in rows), kb)

    print("Self-check: KB re-derivation vs gold answers (exact set match)")
    print("  pool   = all questions of the qtype; safe = questions whose evidence is safe")
    print("  (the pool step 5 keeps). Drop rule applies to `safe`: a qtype whose only")
    print("  failures are unsafe-topic questions is not a broken derivation.\n")
    print(f"{'hop':>3}  {'qtype':<38} {'n':>6} {'pool in':>8} {'pool out':>9} "
          f"{'n safe':>7} {'safe in':>8} {'safe out':>9}  verdict")
    dropped_qtypes = set()
    for hop in (1, 2, 3):
        for qt in dict.fromkeys(r[1] for r in per_hop_rows[hop]):
            v = verdicts[qt]
            if qt in EXCLUDED_QTYPES:
                verdict = "excluded (tags)"
            elif v["best"] < MIN_AGREEMENT:
                verdict = "DROPPED (<95%)"
                dropped_qtypes.add(qt)
            else:
                verdict = "keep (topic " + ("in)" if v["variant"] else "out)")
            print(f"{hop:>3}  {qt:<38} {v['n']:>6,} {v['pool'][True]:>7.1%} {v['pool'][False]:>8.1%} "
                  f"{v['n_safe']:>7,} {v['safe'][True]:>7.1%} {v['safe'][False]:>8.1%}  {verdict}")
    if dropped_qtypes:
        print("\nDropped qtypes (safe-pool agreement < 95%):")
        for qt in sorted(dropped_qtypes):
            print(f"  {qt}: best {verdicts[qt]['best']:.1%} "
                  f"(topic in {verdicts[qt]['safe'][True]:.1%}, topic out {verdicts[qt]['safe'][False]:.1%})")
    else:
        print("\nNo qtype was dropped: on safe evidence every non-tag qtype re-derives exactly.")
        print("Pool mismatches are unsafe topics (reused titles) — dropped in step 5, not silent.")

    # --- 5. filter the survivors ---
    survivors = {hop: defaultdict(list) for hop in (1, 2, 3)}
    drops = Counter()
    for hop in (1, 2, 3):
        for idx, qt, raw, question, topic, gold in per_hop_rows[hop]:
            if qt in EXCLUDED_QTYPES:
                drops["excluded_tag_qtype"] += 1
                continue
            if qt in dropped_qtypes:
                drops["dropped_qtype"] += 1
                continue
            derived = derive(chain_of(qt), topic, kb, verdicts[qt]["variant"])
            if derived is None:
                drops["underivable_topic"] += 1
                continue
            derived_answers, movie_nodes, used = derived
            if derived_answers != set(gold):
                drops["derivation_mismatch"] += 1
                continue
            if not movie_nodes:
                drops["no_evidence"] += 1
                continue
            if movie_nodes & unsafe:
                drops["unsafe_evidence"] += 1
                continue
            if len(movie_nodes) < min_evidence[hop]:
                drops["too_little_evidence"] += 1
                continue
            if len(movie_nodes) > args.max_evidence:
                drops["too_many_evidence"] += 1
                continue
            if len(gold) > args.max_answers:
                drops["too_many_answers"] += 1
                continue
            survivors[hop][qt].append({
                "id": f"metaqa-{hop}hop-{idx:05d}",
                "hop": hop,
                "qtype": qt,
                "question": question,
                "question_raw": raw,
                "topic_entity": topic,
                "answers": gold,
                "answer": ", ".join(gold),
                "evidence_movies": ([topic] + sorted(movie_nodes - {topic})) if topic in movie_nodes
                                    else sorted(movie_nodes),
                "evidence_triples": sorted([movie, rel, obj] for movie, rel, obj in used),
                "_index": idx,
            })

    print("\nSurvivors per hop/qtype BEFORE sampling (pool -> kept)")
    for hop in (1, 2, 3):
        kept = sum(len(v) for v in survivors[hop].values())
        print(f"  {hop}-hop: {len(per_hop_rows[hop]):,} -> {kept:,} kept")
        for qt in sorted(survivors[hop]):
            pool = verdicts[qt]["n"]
            print(f"    {qt:<40} {pool:>6,} -> {len(survivors[hop][qt]):>5,}")
    print("\nDrop reasons: " + (", ".join(f"{k}={v:,}" for k, v in drops.most_common()) or "none"))
    evidence_histogram([r for hop in survivors for qt in survivors[hop] for r in survivors[hop][qt]],
                       "after filtering (survivors)")

    # --- 6. sample ---
    print(f"\nChosen (target {args.per_hop}/hop, seed {args.seed})")
    chosen = []
    for hop in (1, 2, 3):
        picked = sample_evenly(survivors[hop], args.per_hop, rng)
        if len(picked) < args.per_hop:
            print(f"  WARNING: {hop}-hop could only fill {len(picked)}/{args.per_hop}")
        counts = Counter(r["qtype"] for r in picked)
        print(f"  {hop}-hop: {len(picked)} questions over {len(counts)} qtypes")
        for qt in sorted(counts):
            print(f"    {qt:<40} {counts[qt]:>3}")
        chosen += picked

    evidence_histogram(chosen, "after sampling (chosen)")

    chosen.sort(key=lambda r: (r["hop"], r["qtype"], r["_index"]))
    verify_rows(chosen, kb, unsafe, args.max_evidence, args.max_answers)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in chosen:
            f.write(json.dumps({k: v for k, v in row.items() if k != "_index"}, ensure_ascii=False) + "\n")

    evidence = {m for row in chosen for m in row["evidence_movies"]}
    written = validate_file(out_path)
    assert written == len(chosen), f"wrote {written} lines for {len(chosen)} rows"
    print(f"\nUnique evidence movies in sample: {len(evidence):,}")
    print(f"Verified {len(chosen)} rows: evidence safe, triples present in kb.txt, "
          "every gold answer justified by an evidence triple")
    print(f"Validated {written} lines in {out_path}: all parse and carry the required keys")
    print(f"Rows written: {len(chosen)} -> {out_path}")


if __name__ == "__main__":
    main()
