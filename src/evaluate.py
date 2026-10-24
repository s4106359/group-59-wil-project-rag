"""Evaluate retrieval and answer quality for the RAG pipeline.

Run from the project root:
    python src/evaluate.py
"""

import csv
import json
import math
import re
from pathlib import Path

import generate
from generate import answer
from retrieval import CHUNKS, bm25_search, dense_search, hybrid_search

QUESTIONS_PATH = "data/evaluation/questions.json"
RETRIEVAL_CSV = "results/retrieval_results.csv"
ANSWER_CSV = "results/answer_results.csv"
SUMMARY_CSV = "results/summary.csv"

K = 5
METHODS = {"bm25_search": bm25_search, "dense_search": dense_search, "hybrid_search": hybrid_search}

generate.DEBUG = False  # no prompt printing during evaluation


def save_csv(path, rows):
    """Write a list of dicts to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gold_pages_of(q):
    return [int(p) for p in (q.get("pages") or [])]


# ---------- Retrieval metrics ----------

def ndcg_at_k(rels, n_relevant_total, k=K):
    """Binary-relevance NDCG. Ideal ranking = all relevant chunks first."""
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1))
    ideal = min(n_relevant_total, k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_retrieval(questions):
    rows = []
    for q in questions:
        gold = gold_pages_of(q)
        if not gold:
            continue  # answerable questions only
        # Number of chunks that are relevant in the whole corpus (for NDCG)
        n_relevant_total = sum(1 for c in CHUNKS if c["page"] in gold)

        for name, search in METHODS.items():
            results = search(q["question"], k=K)
            rels = [1 if r["page"] in gold else 0 for r in results]
            first = next((i for i, r in enumerate(rels, start=1) if r), None)
            rows.append({
                "id": q["id"],
                "type": q.get("type", ""),
                "method": name,
                "hit@5": int(any(rels)),
                "rr": 1 / first if first else 0.0,
                "ndcg@5": ndcg_at_k(rels, n_relevant_total),
                "retrieved_pages": json.dumps([r["page"] for r in results]),
            })
    return rows


# ---------- Answer metrics ----------

def numbers_in(text):
    """Extract numbers after removing $, commas and the word 'million'."""
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"million", "", text, flags=re.IGNORECASE)
    return re.findall(r"\d+(?:\.\d+)?", text)


def is_correct(q, model_answer, answerable):
    if not answerable:
        return "could not find" in model_answer.lower()
    gold_nums = numbers_in(q["answer"])
    if not gold_nums:  # no numbers in gold answer: fall back to text match
        return q["answer"].strip().lower() in model_answer.lower()
    answer_nums = set(numbers_in(model_answer))
    return all(n in answer_nums for n in gold_nums)


def cited_pages(model_answer):
    """Pages cited like (p.15) or (p.15, p.16)."""
    return [int(p) for p in re.findall(r"p{1,2}\.\s*(\d+)", model_answer)]


def evaluate_answers(questions):
    rows = []
    total = len(questions)
    for n, q in enumerate(questions, start=1):
        print(f"Answering question {n}/{total}...", flush=True)
        gold = gold_pages_of(q)
        answerable = bool(gold)
        result = answer(q["question"], method="hybrid", k=8)
        text = result["answer"]

        cited_ok = ""  # only meaningful for answerable questions
        if answerable:
            cited_ok = any(p in gold for p in cited_pages(text))

        rows.append({
            "id": q["id"],
            "type": q.get("type", ""),
            "answerable": answerable,
            "question": q["question"],
            "gold_answer": q["answer"],
            "gold_pages": json.dumps(gold),
            "answer": text,
            "sources": json.dumps(result["sources"]),
            "correct": is_correct(q, text, answerable),
            "cited_page_ok": cited_ok,
            "faithful_manual": "",  # fill in by hand
        })
        save_csv(ANSWER_CSV, rows)  # save after every question
    return rows


# ---------- Summary ----------

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    # 1) Retrieval
    retrieval_rows = evaluate_retrieval(questions)
    save_csv(RETRIEVAL_CSV, retrieval_rows)

    # 2) Answers
    answer_rows = evaluate_answers(questions)
    save_csv(ANSWER_CSV, answer_rows)

    # 3) Summary
    summary = []
    print("\nRETRIEVAL (answerable questions, k=5)")
    print(f"{'method':<15}{'Hit@5':>8}{'MRR':>8}{'NDCG@5':>9}")
    for name in METHODS:
        rows = [r for r in retrieval_rows if r["method"] == name]
        hit, mrr, ndcg = mean(r["hit@5"] for r in rows), mean(r["rr"] for r in rows), mean(r["ndcg@5"] for r in rows)
        print(f"{name:<15}{hit:>8.3f}{mrr:>8.3f}{ndcg:>9.3f}")
        summary += [
            {"metric": f"{name} Hit@5", "value": round(hit, 4)},
            {"metric": f"{name} MRR", "value": round(mrr, 4)},
            {"metric": f"{name} NDCG@5", "value": round(ndcg, 4)},
        ]

    answerable = [r for r in answer_rows if r["answerable"]]
    unanswerable = [r for r in answer_rows if not r["answerable"]]
    answer_metrics = [
        ("Accuracy (all)", mean(r["correct"] for r in answer_rows)),
        ("Accuracy (answerable)", mean(r["correct"] for r in answerable)),
        ("Refusal rate (unanswerable)", mean(r["correct"] for r in unanswerable)),
        ("Cites correct page (answerable)", mean(r["cited_page_ok"] for r in answerable)),
    ]
    print("\nANSWERS (hybrid, k=8)")
    for name, value in answer_metrics:
        print(f"{name:<34}{value:>7.1%}")
        summary.append({"metric": name, "value": round(value, 4)})

    save_csv(SUMMARY_CSV, summary)
    print(f"\nSaved: {RETRIEVAL_CSV}, {ANSWER_CSV}, {SUMMARY_CSV}")


if __name__ == "__main__":
    main()