"""
HotpotQA Evaluation Script for Phase 2 (Graph RAG)
====================================================
Metrics computed:
  - Exact Match (EM)         — official HotpotQA metric
  - Relaxed EM               — GT is a substring of the answer
  - Token F1                 — official HotpotQA metric
  - Extracted EM             — EM after stripping conversational filler
  - Semantic Similarity      — cosine similarity via OpenAI embeddings
  - LLM-as-Judge             — LLM grades if answer is correct (0-1)
  - RAGAS Answer Correctness — semantic + factual overlap
  - RAGAS Faithfulness       — is the answer grounded in retrieved contexts?
  - RAGAS Context Recall     — did the contexts cover the ground truth?

Usage:
  python -m phase2_graph_rag.benchmark
  (reads OPENAI_API_KEY from .env via shared/config.py)
"""

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from shared.config import settings

DATA_FILE = Path("phase2_graph_rag/hotpot_results_detailed.json")
OUTPUT_FILE = Path("phase2_graph_rag/hotpot_eval_results.json")
REPORT_FILE = Path("phase2_graph_rag/hotpot_eval_report.txt")

FILLER_PATTERNS = [
    r"^(yes|no),?\s*",
    r"^(yes\.|no\.)\s*",
    r"^according to the (provided )?(context|passage|text)[,\.]?\s*",
    r"^based on the (provided )?(context|passage|text)[,\.]?\s*",
    r"^the (provided )?(context|passage|text) (states|says|indicates|shows|mentions)[,\.]?\s*",
    r"^it (is|was|are|were) (held|located|born|formed|founded|released|published|released|released) (in|on|at)?\s*",
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

    avg_em = sum(em_scores) / len(em_scores)
    avg_rem = sum(relaxed_em_scores) / len(relaxed_em_scores)
    avg_eem = sum(extracted_em_scores) / len(extracted_em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print(f"  Exact Match    : {avg_em:.4f}  ({sum(em_scores)}/{len(em_scores)} correct)")
    print(f"  Relaxed EM     : {avg_rem:.4f}  ({sum(relaxed_em_scores)}/{len(relaxed_em_scores)} correct)")
    print(f"  Extracted EM   : {avg_eem:.4f}  ({sum(extracted_em_scores)}/{len(extracted_em_scores)} correct)")
    print(f"  Token F1       : {avg_f1:.4f}")

    return avg_em, avg_rem, avg_eem, avg_f1, per_question


def compute_semantic_similarity(data):
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
                    model="text-embedding-3-small",
                    input=batch,
                )
                embeddings.extend([e.embedding for e in resp.data])
            return np.array(embeddings)

        pred_emb = get_embeddings(predictions)
        ref_emb = get_embeddings(references)

        from numpy.linalg import norm
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

            prompt = JUDGE_PROMPT.format(
                question=item["question"],
                ground_truth=item["ground_truth"],
                answer=ans,
            )

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


def generate_report(summary, per_question, ragas_ok, ragas_df, semantic_scores, judge_scores):
    lines = []
    lines.append("=" * 80)
    lines.append("  Phase 2 (Graph RAG) — HotpotQA Evaluation Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total questions     : {summary['num_examples']}")
    lines.append(f"  Exact Match         : {summary['exact_match']}")
    lines.append(f"  Relaxed EM          : {summary['relaxed_em']}")
    lines.append(f"  Extracted EM        : {summary['extracted_em']}")
    lines.append(f"  Token F1            : {summary['token_f1']}")
    if summary.get("semantic_similarity") is not None:
        lines.append(f"  Semantic Similarity : {summary['semantic_similarity']}")
    if summary.get("llm_judge") is not None:
        lines.append(f"  LLM-as-Judge        : {summary['llm_judge']}")
    if ragas_ok:
        lines.append(f"  RAGAS Answer Corr   : {summary['ragas_answer_correctness']}")
        lines.append(f"  RAGAS Faithfulness  : {summary['ragas_faithfulness']}")
        lines.append(f"  RAGAS Context Recall: {summary['ragas_context_recall']}")
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


def main():
    print("=" * 60)
    print("  Phase 2 (Graph RAG) — HotpotQA Evaluation")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        print("Run 'python -m phase2_graph_rag.hotpot_answers' first.")
        sys.exit(1)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from {DATA_FILE}")

    print("\n--- EM + Token F1 (official HotpotQA metrics) ---")
    avg_em, avg_rem, avg_eem, avg_f1, per_question = compute_em_f1(data)

    print("\n--- Semantic Similarity (OpenAI embeddings) ---")
    avg_semantic, semantic_scores, semantic_ok = compute_semantic_similarity(data)

    print("\n--- LLM-as-Judge ---")
    avg_judge, judge_scores, judge_ok = compute_llm_judge(data)

    print("\n--- RAGAS (LLM-based metrics) ---")
    ragas_results, ragas_ok = compute_ragas(data)

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

    if ragas_ok:
        summary["ragas_answer_correctness"] = round(float(ragas_results["answer_correctness"]), 4)
        summary["ragas_faithfulness"] = round(float(ragas_results["faithfulness"]), 4)
        summary["ragas_context_recall"] = round(float(ragas_results["context_recall"]), 4)

    print(json.dumps(summary, indent=2))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
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

    report = generate_report(summary, per_question, ragas_ok, ragas_df, semantic_scores, judge_scores)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nJSON results saved to: {OUTPUT_FILE}")
    print(f"Text report saved to:  {REPORT_FILE}")


if __name__ == "__main__":
    main()

# python -m phase2_graph_rag.benchmark
