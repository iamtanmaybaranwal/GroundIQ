"""The RAG pipeline: ingest → index → hybrid retrieve → rerank → ground → answer."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk, chunk_document
from .config import Settings, settings as default_settings
from .generation.answerers import Answer, REFUSAL, build_answerer, build_context
from .index.bm25 import BM25Index
from .index.embeddings import build_embedder
from .index.vector import VectorIndex
from .retrieval.hybrid import ScoredChunk, reciprocal_rank_fusion
from .retrieval.rerank import FeatureReranker, build_reranker
from .text import tokenize


@dataclass
class Document:
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] | None = None


class LexiQuery:
    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.bm25 = BM25Index()
        self.vectors = VectorIndex()
        self.embedder = build_embedder(
            self.config.embedding_provider, dimension=self.config.embedding_dimension
        )
        self.reranker = build_reranker(self.config.reranker)
        self.answerer = build_answerer(self.config.answerer)
        self._feature_reranker = FeatureReranker()  # always available for guardrail signals
        self.built = False

    # ------------------------------------------------------------------ ingest
    def add_document(self, document: Document) -> list[Chunk]:
        self.documents[document.id] = document
        chunks = chunk_document(
            document.id,
            document.text,
            target_tokens=self.config.chunk_target_tokens,
            overlap_sentences=self.config.chunk_overlap_sentences,
            metadata={"title": document.title, **(document.metadata or {})},
        )
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        self.built = False
        return chunks

    def add_directory(self, path: str | Path, pattern: str = "*.md") -> int:
        directory = Path(path)
        count = 0
        for file in sorted(directory.glob(pattern)):
            text = file.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip() if text else file.stem
            self.add_document(Document(id=file.stem, text=text, title=title))
            count += 1
        return count

    def build(self) -> "LexiQuery":
        """Fit the embedder and build both indexes. Cheap enough to redo on ingest."""
        if not self.chunks:
            raise ValueError("no documents ingested")

        self.bm25 = BM25Index()
        for chunk in self.chunks.values():
            self.bm25.add(chunk.id, tokenize(chunk.text))
        self.bm25.build()

        texts = [chunk.text for chunk in self.chunks.values()]
        ids = [chunk.id for chunk in self.chunks.values()]
        self.embedder.fit(texts)
        self.vectors = VectorIndex()
        self.vectors.add(ids, self.embedder.embed(texts))

        self.built = True
        return self

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[ScoredChunk]:
        if not self.built:
            self.build()

        top_k = top_k or self.config.top_k
        candidates = self.config.candidates_per_retriever

        lexical = self.bm25.search(tokenize(query), k=candidates)
        dense = self.vectors.search(self.embedder.embed([query])[0], k=candidates)

        fused = reciprocal_rank_fusion(
            {"bm25": [cid for cid, _ in lexical], "vector": [cid for cid, _ in dense]},
            k=self.config.rrf_k,
            weights={"bm25": self.config.bm25_weight, "vector": self.config.vector_weight},
        )
        if not fused:
            return []

        bm25_scores = dict(lexical)
        vector_scores = dict(dense)
        bm25_ranks = {cid: i + 1 for i, (cid, _) in enumerate(lexical)}
        vector_ranks = {cid: i + 1 for i, (cid, _) in enumerate(dense)}

        scored = [
            ScoredChunk(
                chunk=self.chunks[chunk_id],
                score=fused_score,
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                vector_score=vector_scores.get(chunk_id, 0.0),
                bm25_rank=bm25_ranks.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
            )
            for chunk_id, fused_score in fused.items()
        ]
        scored.sort(key=lambda item: -item.score)

        # Rerank only the shortlist - the expensive stage stays bounded.
        shortlist = scored[: max(top_k * 4, 12)]
        best_fused = shortlist[0].score or 1.0
        for candidate in shortlist:
            signals = self._feature_reranker.signals(query, candidate.chunk)
            candidate.signals = signals.to_dict()
            candidate.rerank_score = (
                signals.score()
                if isinstance(self.reranker, FeatureReranker)
                else self.reranker.score(query, candidate.chunk)  # type: ignore[union-attr]
            )
            weight = self.config.rerank_weight
            candidate.score = (1 - weight) * (candidate.score / best_fused) + weight * candidate.rerank_score

        shortlist.sort(key=lambda item: -item.score)
        return shortlist[:top_k]

    # ------------------------------------------------------------------ answering
    def ask(self, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        passages = self.retrieve(question, top_k=top_k)

        if not passages or not self._is_grounded(question, passages):
            answer = Answer(
                REFUSAL,
                [],
                grounded=False,
                refused=True,
                provider=getattr(self.answerer, "name", "extractive"),
                confidence=self._confidence(question, passages),
            )
            return self._envelope(question, answer, passages, started)

        _, citations = build_context(passages, token_budget=self.config.context_token_budget)
        answer = self.answerer.answer(
            question, passages[: len(citations)], citations, term_weights=self.bm25.idf
        )
        answer.confidence = self._confidence(question, passages)
        return self._envelope(question, answer, passages, started)

    def idf_coverage(self, query: str, chunk: Chunk) -> float:
        """Share of the query's *information* that the passage actually contains.

        Plain term overlap treats "the" and "RDS-4471" alike. Weighting each
        query term by its IDF - and giving out-of-vocabulary terms the maximum
        weight, since a term the corpus has never seen is both informative and
        certainly absent - makes an off-topic question score ~0 and gives the
        refusal guardrail something trustworthy to threshold on.
        """
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms:
            return 0.0

        max_idf = max(self.bm25.idf.values(), default=1.0)
        weights = {term: self.bm25.idf.get(term, max_idf) for term in query_terms}
        present = set(tokenize(chunk.text))
        covered = sum(weight for term, weight in weights.items() if term in present)
        total = sum(weights.values()) or 1.0
        return covered / total

    def _is_grounded(self, question: str, passages: list[ScoredChunk]) -> bool:
        """Refuse when the best passage is neither lexically nor semantically close."""
        best = passages[0]
        coverage = self.idf_coverage(question, best.chunk)
        if coverage >= self.config.min_coverage:
            return True
        # A strong semantic match can rescue a paraphrase, but only if at least
        # some of the query's information is genuinely present.
        return best.vector_score >= self.config.min_vector_score and coverage >= 0.15

    def _confidence(self, question: str, passages: list[ScoredChunk]) -> float:
        if not passages:
            return 0.0
        best = passages[0]
        return round(
            0.6 * self.idf_coverage(question, best.chunk) + 0.4 * max(0.0, best.vector_score), 4
        )

    def _envelope(
        self, question: str, answer: Answer, passages: list[ScoredChunk], started: float
    ) -> dict[str, Any]:
        return {
            "question": question,
            **answer.to_dict(),
            "sources": [scored.to_dict() for scored in passages],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path | None = None) -> Path:
        """Persist chunks and vectors so a service can start without re-indexing."""
        import joblib

        directory = Path(path or self.config.index_path)
        directory.mkdir(parents=True, exist_ok=True)

        (directory / "chunks.json").write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks.values()], indent=2),
            encoding="utf-8",
        )
        (directory / "documents.json").write_text(
            json.dumps(
                [
                    {"id": d.id, "title": d.title, "text": d.text, "metadata": d.metadata}
                    for d in self.documents.values()
                ]
            ),
            encoding="utf-8",
        )
        if self.vectors.matrix is not None:
            np.save(directory / "vectors.npy", self.vectors.matrix)
            (directory / "vector_ids.json").write_text(json.dumps(self.vectors.ids), encoding="utf-8")
        joblib.dump(self.embedder, directory / "embedder.joblib")
        return directory

    @classmethod
    def load(cls, path: str | Path, config: Settings | None = None) -> "LexiQuery":
        import joblib

        directory = Path(path)
        engine = cls(config)

        for raw in json.loads((directory / "documents.json").read_text(encoding="utf-8")):
            engine.documents[raw["id"]] = Document(**raw)
        for raw in json.loads((directory / "chunks.json").read_text(encoding="utf-8")):
            chunk = Chunk.from_dict(raw)
            engine.chunks[chunk.id] = chunk

        engine.embedder = joblib.load(directory / "embedder.joblib")
        engine.bm25 = BM25Index()
        for chunk in engine.chunks.values():
            engine.bm25.add(chunk.id, tokenize(chunk.text))
        engine.bm25.build()

        engine.vectors = VectorIndex()
        ids = json.loads((directory / "vector_ids.json").read_text(encoding="utf-8"))
        engine.vectors.add(ids, np.load(directory / "vectors.npy"))
        engine.built = True
        return engine

    # ------------------------------------------------------------------ introspection
    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "built": self.built,
            "bm25_vocabulary": self.bm25.vocabulary_size,
            "vector_dimension": getattr(self.embedder, "dimension", None),
            "embedding_provider": getattr(self.embedder, "name", "unknown"),
            "reranker": getattr(self.reranker, "name", "unknown"),
            "answerer": getattr(self.answerer, "name", "unknown"),
        }
