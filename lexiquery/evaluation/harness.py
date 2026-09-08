"""Evaluation harness.

Runs the labelled question set through the pipeline and reports retrieval
quality, answer quality and refusal behaviour - and can compare retrieval
strategies (BM25 only, vectors only, hybrid, hybrid+rerank) so that every
design choice in this project is backed by a number rather than a vibe.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..pipeline import LexiQuery
from ..text import tokenize
from .metrics import (
    citation_validity,
    hit_rate,
    keyword_recall,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass
class EvalCase:
    id: str
    question: str
    expected_docs: list[str]
    must_include: list[str] = field(default_factory=list)


def load_questions(path: str | Path, key: str = "questions") -> tuple[list[EvalCase], list[str]]:
    """Load a labelled set. `key` selects "questions" or "paraphrase_questions"."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [
        EvalCase(
            id=item["id"],
            question=item["question"],
            expected_docs=item["expected_docs"],
            must_include=item.get("must_include", []),
        )
        for item in raw[key]
    ]
    return cases, raw.get("out_of_scope", [])


def evaluate(
    engine: LexiQuery,
    cases: list[EvalCase],
    out_of_scope: list[str] | None = None,
    *,
    k: int = 5,
) -> dict[str, Any]:
    per_question: list[dict[str, Any]] = []

    for case in cases:
        result = engine.ask(case.question, top_k=k)
        retrieved_docs: list[str] = []
        for source in result["sources"]:
            if source["doc_id"] not in retrieved_docs:  # dedupe: rank documents, not chunks
                retrieved_docs.append(source["doc_id"])

        relevant = set(case.expected_docs)
        per_question.append(
            {
                "id": case.id,
                "question": case.question,
                "retrieved": retrieved_docs,
                "expected": case.expected_docs,
                "hit@1": hit_rate(retrieved_docs, relevant, 1),
                "hit@3": hit_rate(retrieved_docs, relevant, 3),
                "recall@5": recall_at_k(retrieved_docs, relevant, k),
                "precision@1": precision_at_k(retrieved_docs, relevant, 1),
                "mrr": reciprocal_rank(retrieved_docs, relevant),
                "ndcg@5": ndcg_at_k(retrieved_docs, relevant, k),
                "citation_validity": citation_validity(result["citations"], result["sources"]),
                "keyword_recall": keyword_recall(result["answer"], case.must_include),
                "refused": result["refused"],
                "answer": result["answer"],
            }
        )

    refusal_report = []
    for question in out_of_scope or []:
        result = engine.ask(question, top_k=k)
        refusal_report.append({"question": question, "refused": result["refused"]})

    summary = {
        "questions": len(cases),
        "hit@1": round(mean([q["hit@1"] for q in per_question]), 4),
        "hit@3": round(mean([q["hit@3"] for q in per_question]), 4),
        "recall@5": round(mean([q["recall@5"] for q in per_question]), 4),
        "precision@1": round(mean([q["precision@1"] for q in per_question]), 4),
        "mrr": round(mean([q["mrr"] for q in per_question]), 4),
        "ndcg@5": round(mean([q["ndcg@5"] for q in per_question]), 4),
        "citation_validity": round(mean([q["citation_validity"] for q in per_question]), 4),
        "keyword_recall": round(mean([q["keyword_recall"] for q in per_question]), 4),
        "answered": sum(1 for q in per_question if not q["refused"]),
        "out_of_scope_refusal_rate": round(
            mean([1.0 if item["refused"] else 0.0 for item in refusal_report]), 4
        ),
    }
    return {"summary": summary, "per_question": per_question, "out_of_scope": refusal_report}


def compare_strategies(engine: LexiQuery, cases: list[EvalCase], *, k: int = 5) -> dict[str, dict]:
    """Ablation: how much does each stage of the pipeline actually contribute?"""
    results: dict[str, dict] = {}

    def doc_ranking(pairs: list[tuple[str, float]]) -> list[str]:
        ordered: list[str] = []
        for chunk_id, _ in pairs:
            doc_id = engine.chunks[chunk_id].doc_id
            if doc_id not in ordered:
                ordered.append(doc_id)
        return ordered

    strategies = {
        "bm25_only": lambda q: doc_ranking(engine.bm25.search(tokenize(q), k=25)),
        "vector_only": lambda q: doc_ranking(
            engine.vectors.search(engine.embedder.embed([q])[0], k=25)
        ),
        "hybrid_rerank": lambda q: [
            doc
            for doc in dict.fromkeys(
                scored.chunk.doc_id for scored in engine.retrieve(q, top_k=k)
            )
        ],
    }

    for name, retrieve in strategies.items():
        rows = []
        for case in cases:
            retrieved = retrieve(case.question)
            relevant = set(case.expected_docs)
            rows.append(
                {
                    "hit@1": hit_rate(retrieved, relevant, 1),
                    "hit@3": hit_rate(retrieved, relevant, 3),
                    "recall@5": recall_at_k(retrieved, relevant, k),
                    "mrr": reciprocal_rank(retrieved, relevant),
                    "ndcg@5": ndcg_at_k(retrieved, relevant, k),
                }
            )
        results[name] = {
            metric: round(mean([row[metric] for row in rows]), 4)
            for metric in ("hit@1", "hit@3", "recall@5", "mrr", "ndcg@5")
        }

    return results
