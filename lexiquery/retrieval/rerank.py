"""Second-stage reranking.

First-stage retrieval optimises recall over thousands of chunks; reranking
optimises precision over the ~30 that survive. A hosted cross-encoder is the
usual production choice, but it is a network call per candidate, so this project
ships a transparent feature-based reranker that runs offline and explains its
own score:

    coverage   - fraction of the query's content terms present in the chunk
    proximity  - how tightly those terms cluster (a chunk containing
                 "rollback" and "deploy" three words apart beats one where they
                 are 200 words apart)
    phrase     - exact bigram/quoted-phrase hits
    heading    - query terms appearing in the section heading
    density    - a mild penalty for very long chunks, which dilute relevance

`CrossEncoderReranker` shows where a hosted model plugs in.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

from ..chunking import Chunk
from ..text import normalise, tokenize

WEIGHTS = {"coverage": 0.45, "proximity": 0.2, "phrase": 0.2, "heading": 0.1, "density": 0.05}


@dataclass
class RerankSignals:
    coverage: float
    proximity: float
    phrase: float
    heading: float
    density: float

    def score(self) -> float:
        return (
            WEIGHTS["coverage"] * self.coverage
            + WEIGHTS["proximity"] * self.proximity
            + WEIGHTS["phrase"] * self.phrase
            + WEIGHTS["heading"] * self.heading
            + WEIGHTS["density"] * self.density
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "coverage": round(self.coverage, 4),
            "proximity": round(self.proximity, 4),
            "phrase": round(self.phrase, 4),
            "heading": round(self.heading, 4),
            "density": round(self.density, 4),
        }


class FeatureReranker:
    """Deterministic, explainable, and free - no model call per candidate."""

    name = "feature-reranker"

    def signals(self, query: str, chunk: Chunk) -> RerankSignals:
        query_terms = tokenize(query)
        if not query_terms:
            return RerankSignals(0.0, 0.0, 0.0, 0.0, 0.0)

        chunk_terms = tokenize(chunk.text)
        chunk_set = set(chunk_terms)
        unique_query = list(dict.fromkeys(query_terms))

        matched = [term for term in unique_query if term in chunk_set]
        coverage = len(matched) / len(unique_query)

        positions = [i for i, term in enumerate(chunk_terms) if term in set(matched)]
        if len(matched) > 1 and positions:
            span = self._min_span(chunk_terms, set(matched))
            # 1.0 when every matched term sits inside a tight window.
            proximity = 1.0 / (1.0 + math.log1p(max(0, span - len(matched))))
        else:
            proximity = 1.0 if matched else 0.0

        normalised_query = normalise(query)
        normalised_chunk = normalise(chunk.text)
        bigrams = [
            " ".join(pair)
            for pair in zip(normalised_query.split(), normalised_query.split()[1:])
        ]
        phrase_hits = sum(1 for bigram in bigrams if bigram in normalised_chunk)
        phrase = min(1.0, phrase_hits / max(1, len(bigrams))) if bigrams else 0.0

        heading_terms = set(tokenize(chunk.section)) if chunk.section else set()
        heading = (
            len([t for t in unique_query if t in heading_terms]) / len(unique_query)
            if heading_terms
            else 0.0
        )

        density = 1.0 / (1.0 + max(0, chunk.token_estimate - 250) / 250)

        return RerankSignals(coverage, proximity, phrase, heading, density)

    def score(self, query: str, chunk: Chunk) -> float:
        return self.signals(query, chunk).score()

    @staticmethod
    def _min_span(tokens: list[str], targets: set[str]) -> int:
        """Shortest window of tokens containing every target term."""
        if not targets:
            return 0
        last_seen: dict[str, int] = {}
        best = len(tokens)
        for index, token in enumerate(tokens):
            if token not in targets:
                continue
            last_seen[token] = index
            if len(last_seen) == len(targets):
                best = min(best, index - min(last_seen.values()) + 1)
        return best


class CrossEncoderReranker:  # pragma: no cover - requires network + credentials
    """Hosted cross-encoder reranking (Cohere Rerank or a compatible endpoint).

    Set ``LEXIQUERY_RERANK_API_KEY`` (see .env.example) and select it with
    ``LEXIQUERY_RERANKER=cross-encoder``.
    """

    name = "cross-encoder"

    def __init__(self, model: str = "rerank-english-v3.0") -> None:
        self.model = model
        self.api_key = os.environ.get("LEXIQUERY_RERANK_API_KEY", "")
        self.endpoint = os.environ.get("LEXIQUERY_RERANK_URL", "https://api.cohere.com/v1/rerank")

    def rank(self, query: str, chunks: list[Chunk]) -> list[float]:
        import httpx

        if not self.api_key:
            raise RuntimeError("LEXIQUERY_RERANK_API_KEY is not set - use the feature reranker")
        response = httpx.post(
            self.endpoint,
            headers={"Authorization": "Bearer " + self.api_key},
            json={"model": self.model, "query": query, "documents": [c.text for c in chunks]},
            timeout=30.0,
        )
        response.raise_for_status()
        scores = [0.0] * len(chunks)
        for item in response.json()["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores


def build_reranker(kind: str = "feature"):
    if kind == "cross-encoder":
        return CrossEncoderReranker()
    return FeatureReranker()
