"""TF-IDF retrieval over corpus chunks."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.settings_store import DATA_DIR, _connect, get_setting, init_db

INDEX_PATH = DATA_DIR / "rag_index.pkl"

_vectorizer: TfidfVectorizer | None = None
_matrix = None
_chunk_texts: list[str] = []


def _load_chunks() -> list[str]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT text FROM chunks ORDER BY document_id, chunk_index"
        ).fetchall()
    return [row["text"] for row in rows]


def rebuild_index() -> None:
    global _vectorizer, _matrix, _chunk_texts

    _chunk_texts = _load_chunks()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not _chunk_texts:
        _vectorizer = None
        _matrix = None
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        return

    _vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=1,
    )
    _matrix = _vectorizer.fit_transform(_chunk_texts)
    with INDEX_PATH.open("wb") as fh:
        pickle.dump(
            {
                "vectorizer": _vectorizer,
                "matrix": _matrix,
                "chunk_texts": _chunk_texts,
            },
            fh,
        )


def _ensure_index() -> None:
    global _vectorizer, _matrix, _chunk_texts

    if _matrix is not None and _chunk_texts:
        return
    if INDEX_PATH.exists():
        with INDEX_PATH.open("rb") as fh:
            data = pickle.load(fh)
        _vectorizer = data["vectorizer"]
        _matrix = data["matrix"]
        _chunk_texts = data["chunk_texts"]
        return
    rebuild_index()


def retrieve(query: str, top_k: int | None = None) -> list[str]:
    _ensure_index()
    if not query.strip() or _matrix is None or not _chunk_texts or _vectorizer is None:
        return []

    k = top_k or int(get_setting("rag_top_k", 6))
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _matrix).flatten()
    if not np.any(scores):
        return []

    top_indices = np.argsort(scores)[::-1][:k]
    seen: set[str] = set()
    results: list[str] = []
    for idx in top_indices:
        if scores[idx] <= 0 and len(_chunk_texts) > 1:
            continue
        text = _chunk_texts[int(idx)]
        key = text[:120]
        if key in seen:
            continue
        seen.add(key)
        results.append(text)
    return results