#!/usr/bin/env python3
"""Build the MetaQA mini30 track: 30 questions over a 100-doc corpus.

Run: uv run python -m scripts.metaqa_make_mini30

This is ADDITIVE: it writes
  MetaQA/metaqa_eval_mini30.jsonl
  MetaQA/corpus_manifest_mini30.txt
  MetaQA/FROZEN_mini30.sha256
  docs/MINI30.md

and never touches the frozen v1 artifacts or the 12/50 mini track.

Selection (seed 42):
  hop 1:  5 movie-anchored (qtype starts with movie_) + 5 person-anchored
  hop 2:  5 questions with exactly 2 evidence movies + 5 with 3-5
  hop 3:  same as hop 2
  Greedy preference for questions that share evidence movies; no qtype more
  than twice. If the must-have union exceeds --max-must-have, a compact pass
  minimises new evidence instead of maximising sharing.

Corpus: must-have = union of selected evidence movies; up to --docs docs from
the full manifest, with 60% hard distractors (share an actor/director/writer
with a must-have movie) and 40% random. No network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from scripts.metaqa_make_mini import doc_title
from scripts.metaqa_sample import PERSON_RELS, load_kb, resolve_root

REPO = Path(__file__).resolve().parents[1]
FULL_EVAL = "metaqa_eval_v1.jsonl"
FULL_MANIFEST = "corpus_manifest.txt"
OUT_EVAL = "metaqa_eval_mini30.jsonl"
OUT_MANIFEST = "corpus_manifest_mini30.txt"
OUT_CHECKSUM = "FROZEN_mini30.sha256"
OUT_DOC = REPO / "docs" / "MINI30.md"
QTYPE_CAP = 2
ATTEMPTS = 20


class SelectionError(RuntimeError):
    pass


def evidence_bucket(size: int) -> str:
    return "1" if size <= 1 else "2" if size == 2 else "3-5" if size <= 5 else "6+"


def anchor_of(row: dict) -> str:
    return "movie" if row["qtype"].startswith("movie_") else "person"


def bucket_specs(rows: list[dict]) -> list[tuple[str, int, list[dict]]]:
    """The six required buckets, in selection order."""
    def pool(hop, anchor=None, sizes=None):
        selected = [r for r in rows if r["hop"] == hop]
        if anchor is not None:
            selected = [r for r in selected if anchor_of(r) == anchor]
        if sizes is not None:
            selected = [r for r in selected if len(r["supporting_facts_titles"]) in sizes]
        return selected

    return [
        ("hop1 movie", 5, pool(1, "movie")),
        ("hop1 person", 5, pool(1, "person")),
        ("hop2 evidence 2", 5, pool(2, sizes={2})),
        ("hop3 evidence 2", 5, pool(3, sizes={2})),
        ("hop2 evidence 3-5", 5, pool(2, sizes={3, 4, 5})),
        ("hop3 evidence 3-5", 5, pool(3, sizes={3, 4, 5})),
    ]


def _greedy(rows: list[dict], rng: random.Random, mode: str) -> list[dict]:
    """One greedy pass; raises SelectionError when a bucket cannot be filled."""
    picked: list[dict] = []
    chosen_ids: set[str] = set()
    qtype_counts: Counter = Counter()
    evidence: set[str] = set()

    for name, quota, pool in bucket_specs(rows):
        candidates = list(pool)
        rng.shuffle(candidates)
        for _ in range(quota):
            eligible = [r for r in candidates
                        if r["id"] not in chosen_ids and qtype_counts[r["qtype"]] < QTYPE_CAP]
            if not eligible:
                raise SelectionError(f"cannot fill {name!r} with qtype cap {QTYPE_CAP}")

            def score(row):
                ev = set(row["supporting_facts_titles"])
                shared, new, size = len(ev & evidence), len(ev - evidence), len(ev)
                return (shared, -new, -size) if mode == "share" else (-new, shared, -size)

            best = max(eligible, key=score)
            picked.append(best)
            chosen_ids.add(best["id"])
            qtype_counts[best["qtype"]] += 1
            evidence |= set(best["supporting_facts_titles"])
    return picked


def select_questions(rows: list[dict], seed: int, max_must_have: int) -> tuple[list[dict], dict]:
    """Try share-first passes, then compact passes, until must-have fits."""
    for mode in ("share", "compact"):
        for attempt in range(ATTEMPTS):
            try:
                chosen = _greedy(rows, random.Random(seed + attempt), mode)
            except SelectionError:
                continue
            evidence = {t for row in chosen for t in row["supporting_facts_titles"]}
            if len(evidence) <= max_must_have:
                return chosen, {"mode": mode, "attempt": attempt, "must_have": len(evidence)}
            print(f"  {mode} attempt {attempt}: must-have={len(evidence)} > {max_must_have}")
    raise SystemExit(
        f"Could not select 30 questions with must-have <= {max_must_have}. "
        "Lower the evidence buckets or raise --max-must-have."
    )


def build_corpus(chosen: list[dict], full_manifest: list[str], kb, rng: random.Random,
                 total_docs: int, hard_frac: float) -> tuple[list[str], dict]:
    """must-have + 60/40 hard/random distractors, all from the frozen manifest."""
    title_to_path = {doc_title(path): path for path in full_manifest}
    if len(title_to_path) != len(full_manifest):
        raise SystemExit("Duplicate titles in the full manifest — refusing to continue.")
    must = {title for row in chosen for title in row["supporting_facts_titles"]}
    missing = sorted(must - set(title_to_path))
    if missing:
        raise SystemExit(f"{len(missing)} evidence movies have no doc in the manifest, e.g. {missing[:3]}")

    _, forward, backward, _ = kb
    hard_pool: set[str] = set()
    for movie in must:
        for rel in PERSON_RELS:
            for person in forward[rel].get(movie, ()):
                hard_pool |= backward[rel].get(person, set())
    hard_pool &= set(title_to_path) - must
    random_pool = set(title_to_path) - must - hard_pool

    need = total_docs - len(must)
    if need < 0:
        raise SystemExit(f"must-have {len(must)} exceeds --docs {total_docs}.")
    hard_target = round(need * hard_frac)

    hard_sorted = sorted(hard_pool)
    random_sorted = sorted(random_pool)
    rng.shuffle(hard_sorted)
    rng.shuffle(random_sorted)

    hard_pick = hard_sorted[:min(hard_target, len(hard_sorted))]
    random_pick = random_sorted[:min(need - hard_target, len(random_sorted))]
    shortfall = need - len(hard_pick) - len(random_pick)
    if shortfall:
        used = set(hard_pick) | set(random_pick)
        spare = [m for m in hard_sorted + random_sorted if m not in used]
        extra = spare[:shortfall]
        if len(hard_pick) < hard_target:
            hard_pick += extra
        else:
            random_pick += extra
    if len(hard_pick) + len(random_pick) != need:
        raise SystemExit(f"Manifest is too small for {total_docs} docs.")

    docs = ([title_to_path[t] for t in sorted(must)]
            + [title_to_path[t] for t in sorted(hard_pick)]
            + [title_to_path[t] for t in sorted(random_pick)])
    stats = {"must_have": len(must), "hard": len(hard_pick), "random": len(random_pick),
             "hard_pool": len(hard_pool), "random_pool": len(random_pool),
             "distractor_hard_share": len(hard_pick) / max(len(hard_pick) + len(random_pick), 1)}
    return docs, stats


def verify(chosen: list[dict], docs: list[str], v1: list[dict], full_manifest: list[str],
           must_have: int, max_must_have: int) -> None:
    """Hard invariants of the mini30 track; fail loudly."""
    assert len(chosen) == 30, len(chosen)
    counts = Counter((r["hop"], anchor_of(r)) for r in chosen)
    assert counts[(1, "movie")] == 5 and counts[(1, "person")] == 5, counts
    for hop in (2, 3):
        sizes = Counter(evidence_bucket(len(r["supporting_facts_titles"])) for r in chosen if r["hop"] == hop)
        assert sizes["2"] == 5 and sizes["3-5"] == 5, (hop, sizes)
    qtypes = Counter(r["qtype"] for r in chosen)
    assert max(qtypes.values()) <= QTYPE_CAP, qtypes
    assert must_have <= max_must_have, (must_have, max_must_have)

    full_set = set(full_manifest)
    assert len(docs) == 100 and len(set(docs)) == 100, len(docs)
    assert set(docs) <= full_set, sorted(set(docs) - full_set)[:3]
    assert all(Path(path).exists() for path in docs)

    manifest_titles = {doc_title(path) for path in docs}
    for row in chosen:
        assert set(row["supporting_facts_titles"]) <= manifest_titles, row["id"]

    v1_ids = {row["id"] for row in v1}
    assert {row["id"] for row in chosen} <= v1_ids
    assert len({row["id"] for row in chosen}) == 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(path: Path, chosen: list[dict], docs: list[str], stats: dict, selection: dict,
                  eval_sha: str, manifest_sha: str) -> None:
    hop_counts = Counter(r["hop"] for r in chosen)
    anchor_counts = Counter(anchor_of(r) for r in chosen)
    bucket_counts = Counter((r["hop"], evidence_bucket(len(r["supporting_facts_titles"])))
                            for r in chosen)
    qtype_counts = Counter(r["qtype"] for r in chosen)
    lines = [
        "# MetaQA mini30 — 30 questions / 100 docs",
        "",
        "Generated by `uv run python -m scripts.metaqa_make_mini30` (seed 42).",
        "This track is for **plumbing / operational checks**, not for headline results:",
        "the 100-doc corpus is tiny and evidence is deliberately shared between questions.",
        "Never compare Phase 1 vs Phase 2 scores on it, and never compare mini30 to full MetaQA.",
        "",
        "## Selection",
        "",
        "| bucket | questions |",
        "|---|---|",
    ]
    for name, quota, _ in bucket_specs(chosen):
        lines.append(f"| {name} | {quota} |")
    lines += [
        "",
        f"- **hop**: {dict(sorted(hop_counts.items()))}",
        f"- **anchor**: {dict(sorted(anchor_counts.items()))}",
        "- **evidence size**: " + ", ".join(
            f"hop {hop}/{bucket}={n}" for (hop, bucket), n in sorted(bucket_counts.items())),
        f"- **selection mode**: {selection['mode']} (attempt {selection['attempt']}) | "
        f"must-have: {stats['must_have']} | qtype cap: {QTYPE_CAP}",
        f"- **qtypes**: {len(qtype_counts)} distinct, max {max(qtype_counts.values())} per qtype",
        "",
        "## Corpus",
        "",
        f"- **total docs**: {len(docs)}",
        f"- **must-have (evidence)**: {stats['must_have']}",
        f"- **hard distractors**: {stats['hard']} (pool {stats['hard_pool']})",
        f"- **random distractors**: {stats['random']} (pool {stats['random_pool']})",
        f"- **hard share among distractors**: {stats['distractor_hard_share']:.0%}",
        "",
        "## Freeze",
        "",
        "| artifact | sha256 |",
        "|---|---|",
        f"| `MetaQA/{OUT_EVAL}` | `{eval_sha}` |",
        f"| `MetaQA/{OUT_MANIFEST}` | `{manifest_sha}` |",
        "",
        "Verify with `make freeze-verify`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", type=int, default=100, help="total docs in the mini30 corpus")
    ap.add_argument("--hard-frac", type=float, default=0.6, help="hard share among distractors")
    ap.add_argument("--max-must-have", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = resolve_root()
    base = root.parent if root.name == "raw" else root
    v1 = [json.loads(line) for line in (base / FULL_EVAL).read_text(encoding="utf-8").splitlines()
          if line.strip()]
    full_manifest = [line.strip() for line in (base / FULL_MANIFEST).read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    kb = load_kb(root)

    print(f"root: {root} | v1 questions: {len(v1)} | full manifest: {len(full_manifest)}")
    chosen, selection = select_questions(v1, args.seed, args.max_must_have)
    docs, stats = build_corpus(chosen, full_manifest, kb, random.Random(args.seed), args.docs, args.hard_frac)
    verify(chosen, docs, v1, full_manifest, stats["must_have"], args.max_must_have)

    # Preserve v1 order so the mini eval is a stable subsequence.
    order = {row["id"]: index for index, row in enumerate(v1)}
    chosen.sort(key=lambda row: order[row["id"]])

    eval_path = base / OUT_EVAL
    manifest_path = base / OUT_MANIFEST
    with open(eval_path, "w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path.write_text("\n".join(docs) + "\n", encoding="utf-8")

    eval_sha, manifest_sha = sha256(eval_path), sha256(manifest_path)
    checksum_path = base / OUT_CHECKSUM
    checksum_path.write_text(
        f"{eval_sha}  MetaQA/{OUT_EVAL}\n{manifest_sha}  MetaQA/{OUT_MANIFEST}\n",
        encoding="utf-8",
    )
    write_summary(OUT_DOC, chosen, docs, stats, selection, eval_sha, manifest_sha)

    print(f"\nmini30: {len(chosen)} questions over {len(docs)} docs")
    print(f"  must-have {stats['must_have']} | hard {stats['hard']} | random {stats['random']} "
          f"| mode {selection['mode']} attempt {selection['attempt']}")
    print(f"  hop counts: {dict(sorted(Counter(r['hop'] for r in chosen).items()))}")
    print(f"  wrote {eval_path}")
    print(f"  wrote {manifest_path}")
    print(f"  wrote {checksum_path}")
    print(f"  wrote {OUT_DOC}")


if __name__ == "__main__":
    main()
