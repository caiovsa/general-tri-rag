"""
Shared benchmark engine for HotpotQA evaluation.
Used by phase1_vector_rag.benchmark and phase2_graph_rag.benchmark.
"""

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.linalg import norm

from shared.config import settings
from shared.ingest import MOVIE_TAG

FILLER_PATTERNS = [
    r"^(yes|no),?\s*",
    r"^(yes\.|no\.)\s*",
    r"^according to the (provided )?(context|passage|text)[,\.]?\s*",
    r"^based on the (provided )?(context|passage|text)[,\.]?\s*",
    r"^the (provided )?(context|passage|text) (states|says|indicates|shows|mentions)[,\.]?\s*",
    r"^it (is|was|are|were) (held|located|born|formed|founded|released|published) (in|on|at)?\s*",
    r"^the answer is\s*",
    r"^(he|she|it|they) (is|was|are|were)\s*",
    r"^this (is|was|refers to|means|describes)\s*",
]

JUDGE_PROMPT = """You are a strict evaluator for question-answering systems.
Given a question, a ground truth answer, and a model's predicted answer,
determine if the predicted answer is CORRECT.

Rules:
- The predicted answer MUST contain the same factual information as the ground truth.
- If the predicted answer is "I don't know", "not in context", "unknown", or similar — it is INCORRECT.
- If the predicted answer is vague, incomplete, or missing key information from the ground truth — it is INCORRECT.
- Extra context or elaboration is acceptable ONLY if the core ground truth answer is clearly present.
- For yes/no questions, "yes" and "Yes" are equivalent. But "I don't know" is INCORRECT even if the answer happens to be yes/no.
- For numerical answers, accept equivalent formats (e.g., "1999" vs "in 1999"), but the number must be present.
- If the predicted answer contradicts the ground truth, it is INCORRECT.

Respond with ONLY "CORRECT" or "INCORRECT". Nothing else.

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {answer}
Judgment:"""

# Set-answer variant (MetaQA): the gold list may be incomplete, so extra correct
# items in the prediction are fine — but every gold item must be present.
SET_JUDGE_PROMPT = """You are a strict evaluator for question-answering systems.
You are given a question, a list of gold answers, and a model's predicted answer.

Rules:
- The gold list may be incomplete: the prediction may legitimately contain extra correct items.
- Judge CORRECT only if EVERY gold item is present in the prediction (an equivalent
  spelling or paraphrase counts as present).
- A missing gold item makes the answer INCORRECT, even if everything else is right.
- If the predicted answer is "I don't know", "not in context", "unknown", or similar — it is INCORRECT.

Respond with ONLY "CORRECT" or "INCORRECT". Nothing else.

Question: {question}
Gold answers: {ground_truth}
Predicted Answer: {answer}
Judgment:"""

EVIDENCE_BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10 ** 9, "6+"))


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def relaxed_em(prediction: str, ground_truth: str) -> int:
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    return int(norm_gt in norm_pred or norm_pred in norm_gt)


def extract_answer(prediction: str) -> str:
    text = prediction.strip()
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    if not text:
        return prediction
    return text


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def contains_word(haystack: str, needle: str) -> bool:
    """Word-boundary containment on already-normalized strings."""
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def loose_normalize(text: str) -> str:
    """Lowercase/punctuation-strip WITHOUT dropping articles.

    Fallback for gold items that normalize_answer() empties out (e.g. the item
    "A"), which would otherwise never match anything.
    """
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def gold_recall(prediction: str, answers) -> float | None:
    """Fraction of gold `answers` present in the prediction (word-boundary, normalized)."""
    if not answers:
        return None
    normalized = normalize_answer(prediction)
    loose = loose_normalize(prediction)
    found = 0
    for item in answers:
        needle = normalize_answer(str(item))
        if needle:
            found += contains_word(normalized, needle)
        else:  # degenerate item (article-only) — compare without article stripping
            found += contains_word(loose, loose_normalize(str(item)))
    return found / len(answers)


def evidence_recall(contexts, titles) -> float | None:
    """Fraction of evidence movies whose [Movie: <title>] tag is in some retrieved context."""
    if not titles:
        return None
    blob = "\n".join(contexts or []).lower()
    found = sum(1 for title in titles if MOVIE_TAG.format(title=title).lower() in blob)
    return found / len(titles)


def evidence_bucket(size: int) -> str:
    for low, high, label in EVIDENCE_BUCKETS:
        if low <= size <= high:
            return label
    return EVIDENCE_BUCKETS[-1][2]


def compute_set_metrics(data, tagged: bool = False):
    """MetaQA-style metrics: gold_recall / all_found / evidence_recall.

    Returns (gold_recall_scores, all_found_scores, evidence_recall_scores), each
    a list with None where the field does not apply (e.g. HotpotQA has no `answers`
    and its docs are not movie-tagged, so evidence_recall is not meaningful).
    """
    recall_scores, all_found_scores, evidence_scores = [], [], []
    for item in data:
        recall = gold_recall(item.get("answer", ""), item.get("answers"))
        recall_scores.append(recall)
        all_found_scores.append(None if recall is None else int(recall == 1))
        evidence_scores.append(
            evidence_recall(item.get("contexts", []),
                            item.get("supporting_facts", []) or item.get("supporting_facts_titles", []))
            if tagged else None
        )

    if recall_scores and recall_scores[0] is not None:
        mean_recall = sum(recall_scores) / len(recall_scores)
        mean_all_found = sum(all_found_scores) / len(all_found_scores)
        print(f"  Gold Recall (set)  : {mean_recall:.4f}")
        print(f"  All Found          : {mean_all_found:.4f}  ({sum(all_found_scores)}/{len(all_found_scores)})")
    else:
        print("  Gold Recall (set)  : n/a (no `answers` list — single-answer dataset)")
    if evidence_scores and evidence_scores[0] is not None:
        print(f"  Evidence Recall    : {sum(evidence_scores) / len(evidence_scores):.4f}")
    else:
        print("  Evidence Recall    : n/a (contexts are not movie-tagged)")
    return recall_scores, all_found_scores, evidence_scores


def grouped_table(data, gold_recall_scores, all_found_scores, evidence_scores, judge_scores):
    """Per hop, per evidence-size bucket and pooled hop 2+3, each with n."""
    if not any(score is not None for score in gold_recall_scores):
        return []

    groups: list[tuple[str, list[int]]] = []
    hops = sorted({item.get("hop") for item in data if item.get("hop") is not None})
    for hop in hops:
        groups.append((f"hop {hop}", [i for i, item in enumerate(data) if item.get("hop") == hop]))
    if any(hop >= 2 for hop in hops):
        groups.append(("hop 2+3", [i for i, item in enumerate(data) if (item.get("hop") or 0) >= 2]))
    for low, high, label in EVIDENCE_BUCKETS:
        idx = [i for i, item in enumerate(data)
               if item.get("supporting_facts") and low <= len(item["supporting_facts"]) <= high]
        if idx:
            groups.append((f"evidence {label}", idx))

    def mean(values):
        values = [v for v in values if v is not None]
        return None if not values else sum(values) / len(values)

    lines = [f"{'group':<14} {'n':>4} {'gold_recall':>12} {'all_found':>10} {'evid_recall':>12} {'judge':>7}"]
    for name, idx in groups:
        gold = mean([gold_recall_scores[i] for i in idx])
        found = mean([all_found_scores[i] for i in idx])
        evidence = mean([evidence_scores[i] for i in idx])
        judge = mean([judge_scores[i] for i in idx])
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        lines.append(f"{name:<14} {len(idx):>4} {fmt(gold):>12} {fmt(found):>10} {fmt(evidence):>12} {fmt(judge):>7}")
    return lines


def compute_em_f1(data):
    em_scores, relaxed_em_scores, extracted_em_scores, f1_scores = [], [], [], []
    per_question = []

    for item in data:
        q = item["question"]
        gt = item["ground_truth"]
        ans = item["answer"]
        extracted = extract_answer(ans)

        em = exact_match(ans, gt)
        rem = relaxed_em(ans, gt)
        eem = exact_match(extracted, gt)
        f1 = token_f1(ans, gt)

        em_scores.append(em)
        relaxed_em_scores.append(rem)
        extracted_em_scores.append(eem)
        f1_scores.append(f1)

        per_question.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "extracted_answer": extracted,
            "EM": em,
            "RelaxedEM": rem,
            "ExtractedEM": eem,
            "F1": f1,
        })

    if not data:
        return 0.0, 0.0, 0.0, 0.0, []

    n = len(data)
    avg_em = sum(em_scores) / n
    avg_rem = sum(relaxed_em_scores) / n
    avg_eem = sum(extracted_em_scores) / n
    avg_f1 = sum(f1_scores) / n

    print(f"  Exact Match    : {avg_em:.4f}  ({sum(em_scores)}/{n} correct)")
    print(f"  Relaxed EM     : {avg_rem:.4f}  ({sum(relaxed_em_scores)}/{n} correct)")
    print(f"  Extracted EM   : {avg_eem:.4f}  ({sum(extracted_em_scores)}/{n} correct)")
    print(f"  Token F1       : {avg_f1:.4f}")

    return avg_em, avg_rem, avg_eem, avg_f1, per_question


def compute_semantic_similarity(data):
    if not settings.OPENAI_API_KEY:
        print("  OPENAI_API_KEY not set — skipping semantic similarity.")
        return None, [None] * len(data), False

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        predictions = [item["answer"] for item in data]
        references = [item["ground_truth"] for item in data]

        print("  Computing semantic similarity via OpenAI embeddings ...")

        def get_embeddings(texts, batch_size=50):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                resp = client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=batch,
                )
                embeddings.extend([e.embedding for e in resp.data])
            return np.array(embeddings)

        pred_emb = get_embeddings(predictions)
        ref_emb = get_embeddings(references)

        scores = []
        for p, r in zip(pred_emb, ref_emb):
            cos_sim = np.dot(p, r) / (norm(p) * norm(r))
            scores.append(float(cos_sim))

        avg_score = sum(scores) / len(scores)
        print(f"  Semantic Similarity : {avg_score:.4f}")
        return avg_score, scores, True

    except Exception as e:
        print(f"  Semantic similarity skipped: {e}")
        return None, [None] * len(data), False


def compute_llm_judge(data):
    if not settings.OPENAI_API_KEY:
        print("  OPENAI_API_KEY not set — skipping LLM-as-Judge.")
        return None, [None] * len(data), False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        print("  Running LLM-as-Judge evaluation ...")
        scores = []

        for i, item in enumerate(data):
            ans = item["answer"].strip()

            dont_know_patterns = [
                "i don't know", "i do not know", "i cannot answer",
                "i am unable to", "not provided", "not available",
                "based on the provided context", "context does not contain",
                "no relevant information", "cannot be determined",
            ]
            ans_lower = ans.lower()
            if any(p in ans_lower for p in dont_know_patterns):
                scores.append(0)
                if (i + 1) % 10 == 0:
                    print(f"    Judged {i + 1}/{len(data)} ...")
                continue

            prompt = (SET_JUDGE_PROMPT.format(question=item["question"],
                                              ground_truth="; ".join(item["answers"]), answer=ans)
                      if item.get("answers") else
                      JUDGE_PROMPT.format(question=item["question"], ground_truth=item["ground_truth"], answer=ans))

            resp = client.chat.completions.create(
                model=settings.GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            judgment = resp.choices[0].message.content.strip().upper()
            score = 1 if "CORRECT" in judgment else 0
            scores.append(score)

            if (i + 1) % 10 == 0:
                print(f"    Judged {i + 1}/{len(data)} ...")

        avg_score = sum(scores) / len(scores)
        print(f"  LLM-as-Judge Accuracy : {avg_score:.4f}  ({sum(scores)}/{len(scores)} correct)")
        return avg_score, scores, True

    except Exception as e:
        print(f"  LLM-as-Judge failed: {e}")
        return None, [None] * len(data), False


def compute_ragas(data):
    if not settings.OPENAI_API_KEY:
        print("  OPENAI_API_KEY not set — skipping RAGAS.")
        return None, False

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness, faithfulness, context_recall
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.GENERATION_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
        llm_wrapper = LangchainLLMWrapper(llm)

        for metric in [answer_correctness, faithfulness, context_recall]:
            metric.llm = llm_wrapper

        ragas_data = {
            "question": [item["question"] for item in data],
            "answer": [item["answer"] for item in data],
            "contexts": [item.get("contexts", []) for item in data],
            "ground_truth": [item["ground_truth"] for item in data],
        }
        dataset = Dataset.from_dict(ragas_data)

        print("  Running RAGAS evaluation ...")
        ragas_results = evaluate(
            dataset,
            metrics=[answer_correctness, faithfulness, context_recall],
        )

        print(f"  Answer Correctness : {ragas_results['answer_correctness']:.4f}")
        print(f"  Faithfulness       : {ragas_results['faithfulness']:.4f}")
        print(f"  Context Recall     : {ragas_results['context_recall']:.4f}")
        return ragas_results, True

    except Exception as e:
        print(f"  RAGAS evaluation failed: {e}")
        return None, False


def generate_report(phase_name, summary, per_question, ragas_ok, ragas_df, semantic_scores, judge_scores,
                    grouped_lines=None):
    lines = []
    lines.append("=" * 80)
    lines.append(f"  {phase_name} — Evaluation Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total questions     : {summary['num_examples']}")
    lines.append(f"  Exact Match         : {summary['exact_match']}")
    lines.append(f"  Relaxed EM          : {summary['relaxed_em']}")
    lines.append(f"  Extracted EM        : {summary['extracted_em']}")
    lines.append(f"  Token F1            : {summary['token_f1']}")
    if summary.get("gold_recall") is not None:
        lines.append(f"  Gold Recall (set)   : {summary['gold_recall']}")
        lines.append(f"  All Found           : {summary['all_found']}")
    if summary.get("evidence_recall") is not None:
        lines.append(f"  Evidence Recall     : {summary['evidence_recall']}")
    if summary.get("semantic_similarity") is not None:
        lines.append(f"  Semantic Similarity : {summary['semantic_similarity']}")
    if summary.get("llm_judge") is not None:
        lines.append(f"  LLM-as-Judge        : {summary['llm_judge']}")
    if ragas_ok:
        lines.append(f"  RAGAS Answer Corr   : {summary['ragas_answer_correctness']}")
        lines.append(f"  RAGAS Faithfulness  : {summary['ragas_faithfulness']}")
        lines.append(f"  RAGAS Context Recall: {summary['ragas_context_recall']}")
    if grouped_lines:
        lines.append("")
        lines.append("BY GROUP (hop / evidence size)")
        lines.append("-" * 40)
        lines.extend(f"  {line}" for line in grouped_lines)
    lines.append("")
    lines.append("PER-QUESTION BREAKDOWN")
    lines.append("-" * 40)

    for i, item in enumerate(per_question):
        q = item["question"]
        gt = item["ground_truth"]
        ans = item["answer"]
        lines.append(f"\nQ{i+1}: {q}")
        lines.append(f"  Ground truth   : {gt}")
        lines.append(f"  LLM answer     : {ans}")
        lines.append(f"  EM={item['EM']}  RelaxedEM={item['RelaxedEM']}  ExtractedEM={item['ExtractedEM']}  F1={item['F1']:.4f}")
        extra = []
        if semantic_scores[i] is not None:
            extra.append(f"Semantic={semantic_scores[i]:.4f}")
        if judge_scores[i] is not None:
            extra.append(f"Judge={'CORRECT' if judge_scores[i] else 'INCORRECT'}")
        if ragas_ok and ragas_df is not None:
            row = ragas_df.iloc[i]
            extra.append(f"AnswerCorr={row.get('answer_correctness', 0):.4f}")
            extra.append(f"Faith={row.get('faithfulness', 0):.4f}")
            extra.append(f"CtxRecall={row.get('context_recall', 0):.4f}")
        if extra:
            lines.append(f"  {'  '.join(extra)}")

    return "\n".join(lines)


def run_benchmark(phase_name, data_file, output_file, report_file, runner_command,
                  dataset=None, phase=None, per_question_file=None):
    """Run the full benchmark pipeline for a phase."""
    print("=" * 60)
    print(f"  {phase_name} — Evaluation")
    if dataset is not None:
        print(f"  dataset={dataset.name} | top_k={dataset.top_k} | eval={dataset.eval_file}")
    print("=" * 60)

    if not Path(data_file).exists():
        print(f"File not found: {data_file}")
        print(f"Run '{runner_command}' first.")
        sys.exit(1)

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("No data to evaluate.")
        return

    print(f"Loaded {len(data)} examples from {data_file}")

    print("\n--- EM + Token F1 (official HotpotQA metrics) ---")
    avg_em, avg_rem, avg_eem, avg_f1, per_question = compute_em_f1(data)

    print("\n--- Semantic Similarity (OpenAI embeddings) ---")
    avg_semantic, semantic_scores, semantic_ok = compute_semantic_similarity(data)

    print("\n--- LLM-as-Judge ---")
    avg_judge, judge_scores, judge_ok = compute_llm_judge(data)

    print("\n--- RAGAS (LLM-based metrics) ---")
    ragas_results, ragas_ok = compute_ragas(data)

    print("\n--- Set answers (gold recall / evidence recall) ---")
    gold_recall_scores, all_found_scores, evidence_scores = compute_set_metrics(
        data, tagged=bool(getattr(dataset, "tag_movies", False)))
    grouped_lines = grouped_table(data, gold_recall_scores, all_found_scores, evidence_scores, judge_scores)
    if grouped_lines:
        print("\n--- By group (hop / evidence size) ---")
        for line in grouped_lines:
            print(f"  {line}")

    print("\n--- Per-question breakdown ---")
    ragas_df = ragas_results.to_pandas() if ragas_ok else None

    for i, item in enumerate(per_question):
        q = item["question"]
        gt = item["ground_truth"]
        ans = item["answer"]
        em = item["EM"]
        rem = item["RelaxedEM"]
        eem = item["ExtractedEM"]
        f1 = item["F1"]

        em_str = "1" if em else "0"
        rem_str = "1" if rem else "0"
        eem_str = "1" if eem else "0"
        f1_str = f"{f1:.4f}"
        sem_str = f"{semantic_scores[i]:.4f}" if semantic_scores[i] is not None else "n/a"
        judge_str = "CORRECT" if judge_scores[i] == 1 else ("INCORRECT" if judge_scores[i] == 0 else "n/a")

        print(f"\n  Q{i+1}: {q[:80]}{'...' if len(q) > 80 else ''}")
        print(f"       Ground truth : {gt}")
        print(f"       LLM answer   : {ans[:120]}{'...' if len(ans) > 120 else ''}")
        print(f"       EM={em_str}  RelaxedEM={rem_str}  ExtractedEM={eem_str}  F1={f1_str}  Semantic={sem_str}  Judge={judge_str}", end="")

        if ragas_df is not None:
            row = ragas_df.iloc[i]
            ac = row.get("answer_correctness", float("nan"))
            fa = row.get("faithfulness", float("nan"))
            cr = row.get("context_recall", float("nan"))
            print(f"  AnswerCorr={ac:.4f}  Faith={fa:.4f}  CtxRecall={cr:.4f}", end="")
        print()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    summary = {
        "num_examples": len(data),
        "exact_match": round(avg_em, 4),
        "relaxed_em": round(avg_rem, 4),
        "extracted_em": round(avg_eem, 4),
        "token_f1": round(avg_f1, 4),
        "semantic_similarity": round(avg_semantic, 4) if semantic_ok else None,
        "llm_judge": round(avg_judge, 4) if judge_ok else None,
    }

    def _mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    summary["gold_recall"] = _mean(gold_recall_scores)
    summary["all_found"] = _mean(all_found_scores)
    summary["evidence_recall"] = _mean(evidence_scores)

    if ragas_ok:
        summary["ragas_answer_correctness"] = round(float(ragas_results["answer_correctness"]), 4)
        summary["ragas_faithfulness"] = round(float(ragas_results["faithfulness"]), 4)
        summary["ragas_context_recall"] = round(float(ragas_results["context_recall"]), 4)

    print(json.dumps(summary, indent=2))

    metadata = {
        "dataset": getattr(dataset, "name", None),
        "phase": phase,
        "top_k": getattr(dataset, "top_k", None),
        "eval_file": str(getattr(dataset, "eval_file", "")),
        "data_file": str(data_file),
        "runner": runner_command,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "summary": summary,
            "per_question": [
                {
                    "question": d["question"],
                    "ground_truth": d["ground_truth"],
                    "answer": d["answer"],
                    "extracted_answer": per_question[i]["extracted_answer"],
                    "EM": per_question[i]["EM"],
                    "RelaxedEM": per_question[i]["RelaxedEM"],
                    "ExtractedEM": per_question[i]["ExtractedEM"],
                    "F1": round(per_question[i]["F1"], 4),
                    "Semantic": round(semantic_scores[i], 4) if semantic_scores[i] is not None else None,
                    "LLM_Judge": "CORRECT" if judge_scores[i] == 1 else ("INCORRECT" if judge_scores[i] == 0 else None),
                    **(
                        {
                            "ragas_answer_correctness": round(float(ragas_df.iloc[i]["answer_correctness"]), 4),
                            "ragas_faithfulness": round(float(ragas_df.iloc[i]["faithfulness"]), 4),
                            "ragas_context_recall": round(float(ragas_df.iloc[i]["context_recall"]), 4),
                        } if ragas_ok else {}
                    ),
                }
                for i, d in enumerate(data)
            ],
        }, f, indent=2, ensure_ascii=False)

    report = generate_report(phase_name, summary, per_question, ragas_ok, ragas_df, semantic_scores, judge_scores,
                             grouped_lines=grouped_lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    if per_question_file:
        # One row per question, so two runs can be compared PAIRED later.
        Path(per_question_file).parent.mkdir(parents=True, exist_ok=True)
        with open(per_question_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(data):
                f.write(json.dumps({
                    "id": item.get("id"),
                    "phase": phase,
                    "dataset": metadata["dataset"],
                    "top_k": metadata["top_k"],
                    "hop": item.get("hop"),
                    "evidence_n": len(item.get("supporting_facts", []) or []),
                    "gold_recall": gold_recall_scores[i],
                    "all_found": all_found_scores[i],
                    "evidence_recall": evidence_scores[i],
                    "judge": None if judge_scores[i] is None else int(judge_scores[i]),
                    "EM": per_question[i]["EM"],
                    "F1": round(per_question[i]["F1"], 4),
                }, ensure_ascii=False) + "\n")
        print(f"Per-question metrics saved to: {per_question_file}")

    print(f"\nJSON results saved to: {output_file}")
    print(f"Text report saved to:  {report_file}")
