"""
Prepare HotpotQA dataset for benchmarking.

Downloads 100 HotpotQA questions from Hugging Face, extracts unique
context documents as .txt files, and saves golden answers to a JSONL file.

Usage:
  python -m prepare_hotpot
  # Legacy alias (still works, deprecated): python -m preapre_hotpot
"""

import json
from pathlib import Path
from datasets import load_dataset


def main():
    print("Downloading HotpotQA from Hugging Face...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation[:100]")

    DATA_DIR = Path("data_hotpot")
    DATA_DIR.mkdir(exist_ok=True)

    saved_docs = {}

    eval_path = Path("hotpot_eval.jsonl")
    if eval_path.exists():
        eval_path.unlink()

    print("Extracting documents and questions...")
    for row in dataset:
        titles = row['context']['title']
        sentences_list = row['context']['sentences']

        for title, sentences in zip(titles, sentences_list):
            if title not in saved_docs:
                doc_text = " ".join(sentences)
                safe_filename = "".join([c for c in title if c.isalpha() or c.isdigit() or c == ' ']).rstrip()
                filepath = DATA_DIR / f"{safe_filename}.txt"

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(doc_text)

                saved_docs[title] = str(filepath)

        golden_data = {
            "question": row['question'],
            "answer": row['answer'],
            "supporting_facts_titles": row['supporting_facts']['title']
        }

        with open(eval_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(golden_data) + "\n")

    print(f"Done! Created {len(saved_docs)} unique .txt files in '{DATA_DIR}/'")
    print(f"Created evaluation file 'hotpot_eval.jsonl' with 100 questions and golden answers.")


if __name__ == "__main__":
    main()

# python -m prepare_hotpot
