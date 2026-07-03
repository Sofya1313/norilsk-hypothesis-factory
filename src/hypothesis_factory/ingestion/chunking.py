from __future__ import annotations

from hypothesis_factory.models import SourceDocument, TextChunk


def chunk_text(text: str, source_id: str, chunk_size: int = 700, overlap: int = 100) -> list[TextChunk]:
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks: list[TextChunk] = []
    start = 0
    index = 1
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(TextChunk(id=f"{source_id}-CH{index:03d}", source_id=source_id, text=clean[start:end], position=index))
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
        index += 1
    return chunks


def chunk_documents(documents: list[SourceDocument]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for doc in documents:
        chunks.extend(chunk_text(doc.text, doc.id))
    return chunks

