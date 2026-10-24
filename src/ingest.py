"""Ingest the Woolworths annual report PDF into overlapping text chunks.

Run from the project root:
    python src/ingest.py
"""

import json
import re
from pathlib import Path

import fitz  # pymupdf

# Paths relative to the project root
PDF_PATH = "data/raw/woolworths_annual_report_2025.pdf"
OUTPUT_PATH = "data/processed/chunks.json"

# Chunking settings (in words)
CHUNK_SIZE = 200
OVERLAP = 40


def clean_text(text: str) -> str:
    """Collapse extra whitespace and remove blank lines."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop blank lines
    return " ".join(lines)


def chunk_words(words: list[str]) -> list[str]:
    """Split a list of words into overlapping chunks of text."""
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + CHUNK_SIZE]))
        if start + CHUNK_SIZE >= len(words):
            break  # last chunk reached the end of the page
        start += step
    return chunks


def main():
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)

    chunks = []
    chunk_id = 0

    # Process page by page so chunks never cross a page boundary
    for page_index, page in enumerate(doc):
        text = clean_text(page.get_text())
        if not text:
            continue  # skip empty pages
        for chunk_text in chunk_words(text.split()):
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_index + 1,  # 1-based page number
                "text": chunk_text,
            })
            chunk_id += 1

    doc.close()

    # Save chunks to JSON
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Print summary stats
    avg_len = sum(len(c["text"].split()) for c in chunks) / len(chunks) if chunks else 0
    print(f"Total pages: {total_pages}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk length: {avg_len:.1f} words")


if __name__ == "__main__":
    main()