"""Retrieval and answer metrics.

A RAG system without an evaluation harness is a demo. These are the standard
information-retrieval metrics plus the two answer-quality checks that matter in
production: are the cited passages real, and does the system refuse when it
should?
"""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top k."""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def hit_rate(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant document is in the top k - the "did it work at all" metric."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant hit. Rewards putting the answer first."""
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def citation_validity(citations: list[dict], sources: list[dict]) -> float:
    """Fraction of citations that point at a passage actually retrieved.

    A hallucinated citation is worse than no citation, because it looks
    verifiable and is not.
    """
    if not citations:
        return 1.0
    valid_ids = {source["chunk_id"] for source in sources}
    return sum(1 for citation in citations if citation["chunk_id"] in valid_ids) / len(citations)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Fraction of required keywords present in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    lowered = answer.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered) / len(keywords)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
