"""Corpus ingestion: file upload, text extraction, chunking."""

from __future__ import annotations

import re
from pathlib import Path

from services.settings_store import _connect, init_db

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".pdf"}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_file(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _extract_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def add_text(name: str, text: str, source_type: str = "paste") -> dict:
    init_db()
    text = text.strip()
    if not text:
        raise ValueError("No text content to add.")

    chunks = _chunk_text(text)
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents (name, source_type, char_count, word_count)
            VALUES (?, ?, ?, ?)
            """,
            (name, source_type, len(text), _word_count(text)),
        )
        doc_id = cursor.lastrowid
        for idx, chunk in enumerate(chunks):
            conn.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, text)
                VALUES (?, ?, ?)
                """,
                (doc_id, idx, chunk),
            )
        conn.commit()

    from services.rag import rebuild_index

    rebuild_index()
    return get_stats()


def add_file(file_path: str | Path) -> dict:
    path = Path(file_path)
    text = extract_text_from_file(path)
    return add_text(path.name, text, source_type="file")


def get_stats() -> dict:
    init_db()
    with _connect() as conn:
        doc_rows = conn.execute(
            """
            SELECT id, name, source_type, char_count, word_count, created_at
            FROM documents ORDER BY id DESC
            """
        ).fetchall()
        totals = conn.execute(
            """
            SELECT
                COUNT(DISTINCT document_id) AS documents,
                COUNT(*) AS chunks,
                COALESCE(SUM(LENGTH(text)), 0) AS chars,
                COALESCE(SUM(
                    (LENGTH(text) - LENGTH(REPLACE(text, ' ', ''))) + 1
                ), 0) AS approx_words
            FROM chunks
            """
        ).fetchone()

    documents = [dict(row) for row in doc_rows]
    return {
        "documents": len(documents),
        "chunks": totals["chunks"] if totals else 0,
        "chars": totals["chars"] if totals else 0,
        "words": sum(d["word_count"] for d in documents),
        "document_list": documents,
    }


def clear_corpus() -> dict:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.commit()
    from services.rag import rebuild_index

    rebuild_index()
    return get_stats()


def reindex() -> dict:
    from services.rag import rebuild_index

    rebuild_index()
    return get_stats()