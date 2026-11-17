#!/usr/bin/env python3
"""Freeze a v2 model checkpoint from tagged results (no API)."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="RESULTS_TAG, e.g. luna_v2")
    ap.add_argument("--label", required=True, help="checkpoint label, e.g. luna")
    ap.add_argument("--answer-model", required=True)
    ap.add_argument("--answer-provider", default="default")
    ap.add_argument("--provider-pin", default="")
    ap.add_argument("--answer-reasoning", default="none")
    ap.add_argument("--judge-model", default="gpt-6.1-sol")
    ap.add_argument("--judge-reasoning", default="low")
    ap.add_argument("--ragas-model", default="gpt-6-luna")
    ap.add_argument("--ragas-reasoning", default="none")
    args = ap.parse_args()

    out = ROOT / "checkpoints/v2" / args.label
    archive_dir = out / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    source_dirs = sorted(p for p in (ROOT / "results/metaqa").glob(f"phase*_{args.tag}")
                         if (p / "eval_results.json").exists())
    if not source_dirs:
        raise SystemExit("No results found for " + args.tag)
    runs, detailed = [], []
    for src in source_dirs:
        dst = out / "metaqa" / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in ["eval_results.json", "eval_report.txt", "per_question.jsonl",
                     "results.json", "run_meta.json", "ingest_meta.json", "entity_cache.json"]:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        ev = json.load((dst / "eval_results.json").open())
        meta = json.load((dst / "run_meta.json").open())
        files = {f.name: {"sha256": sha(f), "bytes": f.stat().st_size}
                 for f in sorted(dst.iterdir()) if f.is_file()}
        runs.append({"run": src.name, "n": ev["summary"]["num_examples"],
                     "metrics": ev["summary"], "fingerprint": meta.get("fingerprint"),
                     "answered": meta.get("answered"), "errors": meta.get("errors"),
                     "empty_contexts": meta.get("empty_contexts"), "files": files})
        detailed.append(src / "results_detailed.json")

    archive = archive_dir / f"results_detailed_{args.label}_v2.tar.zst"
    if archive.exists():
        archive.unlink()
    detailed_rel = [str(p.relative_to(ROOT)) for p in detailed]
    tar_proc = subprocess.Popen(["tar", "-cf", "-", "-C", str(ROOT), *detailed_rel],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    zstd_proc = subprocess.Popen(["zstd", "-19", "-q", "-o", str(archive)],
                                 stdin=tar_proc.stdout, stderr=subprocess.PIPE)
    tar_proc.stdout.close()
    tar_proc.wait()
    zstd_proc.wait()
    if tar_proc.returncode or zstd_proc.returncode:
        raise SystemExit("archive failed")

    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema_version": 2, "checkpoint": "v2", "model_label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": source_commit, "results_tag": args.tag,
        "protocol": {"dataset": "metaqa", "chunk_size": 256, "chunk_overlap": 32, "top_k": 20},
        "models": {
            "answer": {"model": args.answer_model, "provider": args.answer_provider,
                       "provider_pin": args.provider_pin or None,
                       "reasoning_effort": args.answer_reasoning, "temperature": 0},
            "judge": {"model": args.judge_model, "reasoning_effort": args.judge_reasoning},
            "ragas": {"model": args.ragas_model, "reasoning_effort": args.ragas_reasoning, "temperature": 0},
            "extraction": {"model": "gpt-6-luna (reused v1 graphs)"},
        },
        "runs": runs,
        "archive": {"path": str(archive.relative_to(ROOT)), "sha256": sha(archive),
                    "bytes": archive.stat().st_size, "tracked": False},
        "caveats": [
            "v1 generated each question once; v1 noise is unmeasured.",
            "RAGAS with gpt-6-luna@none emitted 'No statements' warnings and NaN faithfulness on some phases; treat RAGAS as diagnostic only.",
        ],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "ARCHIVE.sha256").write_text(f"{sha(archive)}  {archive.relative_to(ROOT)}\n", encoding="utf-8")

    def val(run, key):
        v = run["metrics"].get(key)
        return "NaN" if isinstance(v, float) and v != v else ("—" if v is None else f"{v:.4f}")
    lines = [
        f"# RAG Trilogy — v2 {args.label} checkpoint",
        "", f"**Frozen:** {manifest['created_at']}  ",
        f"**Results tag:** `{args.tag}`  ",
        f"**Source commit:** `{source_commit}`  ",
        f"**Answer model:** `{args.answer_model}` via `{args.answer_provider}`, reasoning `{args.answer_reasoning}`, temperature `0`  ",
        f"**OpenRouter provider pin:** `{args.provider_pin or 'none'}`  ",
        f"**Judge:** `{args.judge_model}` reasoning `{args.judge_reasoning}`  ",
        f"**RAGAS:** `{args.ragas_model}` reasoning `{args.ragas_reasoning}`",
        "", "This is a v2 run: answer reasoning is explicitly disabled and temperature is 0.",
        "", "| Run | n | gold | all found | evidence | judge | RAGAS faith | RAGAS ctx |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        lines.append(f"| {r['run']} | {r['n']} | {val(r,'gold_recall')} | {val(r,'all_found')} | "
                     f"{val(r,'evidence_recall')} | {val(r,'llm_judge')} | {val(r,'ragas_faithfulness')} | {val(r,'ragas_context_recall')} |")
    lines += [
        "", "## Important notes",
        "",
        "- Phase 2/P3b graphs are the v1 graphs; v2 changes the answer model/protocol, not extraction.",
        "- RAGAS with `gpt-6-luna@none` produced `No statements` warnings and NaN faithfulness on different phases for each model; RAGAS is secondary and should not be treated as a clean/comparable score.",
        "- v1 generated each question once; no three-run noise check was performed. v1 variance is unmeasured.",
        "- The detailed contexts archive is local and not committed: it contains MetaQA document text.",
        "", "## Files",
        "",
        "Compact metrics, per-question rows, run metadata and checksums are tracked. The archive",
        f"is `{archive.relative_to(ROOT)}`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / f"docs/RESULTS_V2_{args.label.upper()}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    checks = []
    for f in sorted(out.rglob("*")):
        if f.is_file() and "archive" not in f.parts and f.name != "CHECKSUMS.sha256":
            checks.append(f"{sha(f)}  {f.relative_to(ROOT)}")
    (out / "CHECKSUMS.sha256").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print("wrote", out, "runs", len(runs), "archive", archive.stat().st_size)


if __name__ == "__main__":
    main()
