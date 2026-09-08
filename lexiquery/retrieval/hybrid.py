"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion.

Neither retriever alone is enough:

* BM25 finds exact tokens - error codes, flag names, "RDS-4471" - but misses
  paraphrases ("how do I undo a release?" vs "rollback procedure").
* Embeddings capture paraphrase but blur rare identifiers into their neighbours.

RRF fuses the two *rankings* rather than their scores, which is the standard
trick precisely because BM25 scores and cosine similarities live on
incomparable scales and normalising them is fragile:

    RRF(d) = Σ 1 / (k + rank_i(d))          k = 60 by convention
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float = 0.0
    bm25_score: float = 0.0
    vector_score: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "section": self.chunk.section,
            "text": self.chunk.text,
            "score": round(self.score, 6),
            "bm25_score": round(self.bm25_score, 4),
            "vector_score": round(self.vector_score, 4),
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
        }


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], *, k: int = 60, weights: dict[str, float] | None = None
) -> dict[str, float]:
    """Fuse named rankings of ids into one score per id."""
    weights = weights or {}
    fused: dict[str, float] = {}
    for name, ranking in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return fused
