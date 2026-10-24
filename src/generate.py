"""Generate answers from retrieved chunks using a local Ollama model.

Run from the project root:
    python src/generate.py
"""

import ollama

from retrieval import bm25_search, dense_search, hybrid_search

MODEL = "llama3.2:3b"
NOT_FOUND = "I could not find this in the report."
DEBUG = False # print the exact prompt sent to the model


def build_prompt(question, chunks):
    """Context first, then instructions, then the question."""
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)
    return (
        f"Context:\n{context}\n\n"
        "Use the context to answer. The context may contain tables with "
        "flattened text, read the numbers carefully. Cite the page numbers "
        "you used like (p.16). "
        f"Only reply '{NOT_FOUND}' if the context has nothing relevant "
        "to the question.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def answer(question, method="hybrid", k=8):
    # Retrieve top-k chunks with the chosen method
    search = {"dense": dense_search, "bm25": bm25_search, "hybrid": hybrid_search}[method]
    chunks = search(question, k=k)

    prompt = build_prompt(question, chunks)
    if DEBUG:
        print("=" * 60, "\nPROMPT SENT TO MODEL:\n", prompt, "\n", "=" * 60)

    # Ask the model, temperature 0 for consistent output
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    return {
        "question": question,
        "answer": response["message"]["content"].strip(),
        "sources": [{"chunk_id": c["chunk_id"], "page": c["page"]} for c in chunks],
    }


if __name__ == "__main__":
    questions = [
        "What was the net profit and dividend for FY2025?",
        "What is the capital of France?",
    ]
    for q in questions:
        result = answer(q)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")