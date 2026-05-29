import json
import os
from pathlib import Path
from datasets import load_dataset

# 1. Load a small subset of HotpotQA (the 'distractor' version includes hard negative docs)
print("Downloading HotpotQA from Hugging Face...")
dataset = load_dataset("hotpot_qa", "distractor", split="validation[:100]") # Get 100 questions

DATA_DIR = Path("data_hotpot")
DATA_DIR.mkdir(exist_ok=True)

# We need to keep track of which documents we've already saved to avoid duplicates
saved_docs = {}

print("Extracting documents and questions...")
for row in dataset:
    # HotpotQA stores context as a list of titles and a list of sentence lists
    titles = row['context']['title']
    sentences_list = row['context']['sentences']
    
    for title, sentences in zip(titles, sentences_list):
        if title not in saved_docs:
            # Combine the sentences into a single paragraph
            doc_text = " ".join(sentences)
            
            # Save as a text file
            safe_filename = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            filepath = DATA_DIR / f"{safe_filename}.txt"
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(doc_text)
            
            saved_docs[title] = str(filepath)

    # Save the Golden Answers for evaluation
    golden_data = {
        "question": row['question'],
        "answer": row['answer'],
        "supporting_facts_titles": row['supporting_facts']['title'] # Which docs the retriever SHOULD find
    }
    
    # Append to our evaluation file
    eval_path = Path("hotpot_eval.jsonl")
    with open(eval_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(golden_data) + "\n")

print(f"✅ Done! Created {len(saved_docs)} unique .txt files in '{DATA_DIR}/'")
print(f"✅ Created evaluation file 'hotpot_eval.jsonl' with 100 questions and golden answers.")


# python -m prepare_hotpot