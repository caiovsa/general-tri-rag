#!/usr/bin/env python3
"""Paired comparison of two runs from their per_question.jsonl files.

Usage:
  python -m scripts.compare_runs results/metaqa_mini30/phase2 results/metaqa_mini30/phase3b --metric all_found
  python -m scripts.compare_runs results/metaqa/phase1 results/metaqa/phase3 --metric gold_recall

Reports the mean difference, a paired bootstrap 95% CI, the discordant pairs,
and the exact McNemar p-value for binary metrics (all_found / judge / EM).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

BINARY = {"all_found", "judge", "EM"}
CONTINUOUS = {"gold_recall", "evidence_recall", "F1"}
METRICS = BINARY | CONTINUOUS


def load_run(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "per_question.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("id") or row.get("question"))
        rows[key] = row
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: list[float], iterations: int = 5000, seed: int = 42):
    if not diffs:
        return None, None
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path, help="run directory or per_question.jsonl (A)")
    ap.add_argument("b", type=Path, help="run directory or per_question.jsonl (B)")
    ap.add_argument("--metric", choices=sorted(METRICS), default="all_found")
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows_a, rows_b = load_run(args.a), load_run(args.b)
    common = sorted(set(rows_a) & set(rows_b))
    missing = (len(set(rows_a) - set(rows_b)), len(set(rows_b) - set(rows_a)))
    pairs = []
    for key in common:
        va, vb = rows_a[key].get(args.metric), rows_b[key].get(args.metric)
        if va is None or vb is None:
            continue
        pairs.append((float(va), float(vb)))

    n = len(pairs)
    if not n:
        raise SystemExit(f"No paired values for {args.metric!r}")
    mean_a = sum(a for a, _ in pairs) / n
    mean_b = sum(b for _, b in pairs) / n
    delta = mean_a - mean_b
    diffs = [a - b for a, b in pairs]
    low, high = bootstrap_ci(diffs, args.iterations, args.seed)

    print(f"metric       : {args.metric}")
    print(f"paired rows  : {n} (missing A/B: {missing[0]}/{missing[1]})")
    print(f"A mean       : {mean_a:.4f}")
    print(f"B mean       : {mean_b:.4f}")
    print(f"A - B        : {delta:+.4f}   95% CI [{low:+.4f}, {high:+.4f}]")

    if args.metric in BINARY:
        b_only = sum(1 for a, b in pairs if a == 1 and b == 0)
        a_only = sum(1 for a, b in pairs if a == 0 and b == 1)
        p = mcnemar_exact(b_only, a_only)
        print(f"discordant   : A only={b_only}, B only={a_only}")
        print(f"McNemar p    : {p:.4f}")
        if p >= 0.05:
            print("interpretation: no detectable difference at 5% (CI includes 0 is expected here).")
        else:
            print("interpretation: difference is unlikely under the null (p < 0.05).")
    else:
        print("interpretation: continuous paired difference; use the CI, not a p-value.")
    if low is not None and low <= 0 <= high:
        print("note         : 95% CI includes zero — do not claim a direction.")


if __name__ == "__main__":
    main()
