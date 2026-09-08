"""BM25 lexical retrieval, implemented from the ranking function up.

Dense embeddings miss exact identifiers - error codes, flag names, "RDS-4471".
BM25 nails those, which is exactly why production RAG systems run both and fuse
the results rather than betting on vectors alone.

    score(q, d) = Σ  IDF(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))
                 t∈q
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: dict[str, Counter[str]] = {}
        self.doc_length: dict[str, int] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.idf: dict[str, float] = {}
        self.average_length = 0.0

    def add(self, doc_id: str, tokens: list[str]) -> None:
        counts = Counter(tokens)
        self.doc_tokens[doc_id] = counts
        self.doc_length[doc_id] = len(tokens)
        for token in counts:
            self.postings[token].add(doc_id)

    def build(self) -> None:
        """Precompute IDF and the average document length."""
        total_docs = len(self.doc_tokens)
        self.average_length = (
            sum(self.doc_length.values()) / total_docs if total_docs else 0.0
        )
        self.idf = {}
        for token, docs in self.postings.items():
            df = len(docs)
            # Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative.
            self.idf[token] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        counts = self.doc_tokens.get(doc_id)
        if not counts:
            return 0.0
        length = self.doc_length[doc_id]
        norm = self.k1 * (1 - self.b + self.b * length / (self.average_length or 1))

        total = 0.0
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if frequency == 0:
                continue
            total += self.idf.get(token, 0.0) * (frequency * (self.k1 + 1)) / (frequency + norm)
        return total

    def search(self, query_tokens: list[str], k: int = 10) -> list[tuple[str, float]]:
        # Only documents sharing at least one query term can score above zero.
        candidates: set[str] = set()
        for token in query_tokens:
            candidates |= self.postings.get(token, set())

        scored = [(doc_id, self.score(query_tokens, doc_id)) for doc_id in candidates]
        scored = [(doc_id, score) for doc_id, score in scored if score > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    @property
    def size(self) -> int:
        return len(self.doc_tokens)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)
