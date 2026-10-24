"""Retrieval over report chunks: BM25 (keyword) and dense (embeddings).

Run from the project root:
    python src/retrieval.py
"""

import json
from pathlib import Path
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.json"
EMBEDDINGS_PATH = "data/processed/embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load chunks once
with open(CHUNKS_PATH, encoding="utf-8") as f:
    CHUNKS = json.load(f)

# BM25 index: lowercase + split tokenization
BM25 = BM25Okapi([c["text"].lower().split() for c in CHUNKS])

# Dense model and embeddings (loaded lazily)
_model = None
_embeddings = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_embeddings():
    """Embed all chunks once, then reuse the cached .npy file."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    path = Path(EMBEDDINGS_PATH)
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(CHUNKS):  # cache matches current chunks
            _embeddings = emb
            return _embeddings

    texts = [c["text"] for c in CHUNKS]
    _embeddings = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, _embeddings)
    return _embeddings


def _top_k(scores, k):
    """Build result dicts for the k highest scores, best first."""
    top = np.argsort(scores)[::-1][:k]
    return [
        {
            "chunk_id": CHUNKS[i]["chunk_id"],
            "page": CHUNKS[i]["page"],
            "text": CHUNKS[i]["text"],
            "score": float(scores[i]),
        }
        for i in top
    ]


def bm25_search(query, k=5):
    scores = BM25.get_scores(query.lower().split())
    return _top_k(scores, k)


def dense_search(query, k=5):
    q = _get_model().encode([query], normalize_embeddings=True)[0]
    scores = _get_embeddings() @ q  # cosine similarity (vectors are normalized)
    return _top_k(scores, k)

def hybrid_search(query, k=5):
    """Merge BM25 and dense results with Reciprocal Rank Fusion."""
    fused = {}  # chunk_id -> result dict with fused score
    for results in (bm25_search(query, k=20), dense_search(query, k=20)):
        for rank, r in enumerate(results, start=1):
            if r["chunk_id"] not in fused:
                fused[r["chunk_id"]] = {**r, "score": 0.0}
            fused[r["chunk_id"]]["score"] += 1 / (60 + rank)

    # Best fused score first
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:k]


if __name__ == "__main__":
    query = "What was the net profit and dividend for FY2025?"
    for name, search in [("BM25", bm25_search), ("Dense", dense_search)]:
        print(f"\n{name} top 3:")
        for r in search(query, k=3):
            print(f"  chunk_id={r['chunk_id']}  page={r['page']}  score={r['score']:.4f}")