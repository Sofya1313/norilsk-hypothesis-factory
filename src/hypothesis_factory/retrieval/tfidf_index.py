from __future__ import annotations

import math
import re
from collections import Counter

from hypothesis_factory.models import TextChunk


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-я0-9%+-]+", text.lower())


class TfidfIndex:
    def __init__(self, chunks: list[TextChunk]):
        self.chunks = chunks
        self._sklearn = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._sklearn = TfidfVectorizer()
            self._matrix = self._sklearn.fit_transform([c.text for c in chunks])
        except Exception:
            self._doc_tokens = [Counter(_tokens(c.text)) for c in chunks]
            doc_freq = Counter(token for doc in self._doc_tokens for token in doc)
            n = max(len(chunks), 1)
            self._idf = {token: math.log((1 + n) / (1 + df)) + 1 for token, df in doc_freq.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[TextChunk, float]]:
        if not self.chunks:
            return []
        if self._sklearn is not None:
            query_vec = self._sklearn.transform([query])
            scores = (self._matrix @ query_vec.T).toarray().ravel()
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
            return [(self.chunks[i], float(score)) for i, score in ranked if score > 0]

        query_counts = Counter(_tokens(query))
        scored = []
        for chunk, counts in zip(self.chunks, self._doc_tokens):
            score = sum(query_counts[t] * counts.get(t, 0) * self._idf.get(t, 0.0) for t in query_counts)
            scored.append((chunk, float(score)))
        return [(chunk, score) for chunk, score in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k] if score > 0]

