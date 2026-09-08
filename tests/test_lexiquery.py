"""Tests: text pipeline, indexes, fusion, guardrails, quality thresholds, API."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexiquery.api import create_app  # noqa: E402
from lexiquery.chunking import chunk_document, split_sections  # noqa: E402
from lexiquery.config import Settings  # noqa: E402
from lexiquery.evaluation.harness import compare_strategies, evaluate, load_questions  # noqa: E402
from lexiquery.evaluation.metrics import (  # noqa: E402
    citation_validity,
    hit_rate,
    keyword_recall,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from lexiquery.index.bm25 import BM25Index  # noqa: E402
from lexiquery.index.vector import VectorIndex  # noqa: E402
from lexiquery.pipeline import Document, LexiQuery  # noqa: E402
from lexiquery.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402
from lexiquery.retrieval.rerank import FeatureReranker  # noqa: E402
from lexiquery.text import stem, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
QUESTIONS = ROOT / "eval" / "questions.json"


@pytest.fixture(scope="module")
def engine() -> LexiQuery:
    instance = LexiQuery(Settings(api_key="test-key"))
    instance.add_directory(CORPUS)
    instance.build()
    return instance


# --------------------------------------------------------------------------- text


def test_tokenizer_normalises_stems_and_drops_stopwords():
    tokens = tokenize("How do I ROLL BACK the deployments quickly?")
    assert "the" not in tokens and "do" not in tokens
    assert "deploy" in tokens  # deployments -> deploy
    assert all(token.islower() for token in tokens)


def test_stemmer_keeps_short_words_intact():
    assert stem("running") == "run"
    assert stem("policies") == "polic"
    assert stem("api") == "api"


# --------------------------------------------------------------------------- chunking


def test_sections_are_split_by_heading_hierarchy():
    markdown = "# Title\nIntro text.\n\n## First\nAlpha body.\n\n## Second\nBeta body."
    sections = split_sections(markdown)
    paths = [path for path, _ in sections]
    assert paths == ["Title", "Title > First", "Title > Second"]


def test_chunks_carry_their_heading_and_respect_the_budget():
    text = "# Runbook\n\n## Failover\n" + " ".join(f"Sentence number {i} about failover." for i in range(80))
    chunks = chunk_document("doc", text, target_tokens=120)

    assert len(chunks) > 1
    assert all(chunk.doc_id == "doc" for chunk in chunks)
    assert all("Failover" in chunk.text for chunk in chunks)
    assert all(chunk.token_estimate < 300 for chunk in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunks_overlap_so_boundaries_stay_retrievable():
    text = "## S\n" + " ".join(f"Fact {i} is important and distinct." for i in range(40))
    chunks = chunk_document("doc", text, target_tokens=60, overlap_sentences=1)
    first_tail = chunks[0].text.split(".")[-2]
    assert first_tail.strip() in chunks[1].text


# --------------------------------------------------------------------------- BM25


def test_bm25_ranks_the_document_containing_the_rare_term_first():
    index = BM25Index()
    index.add("a", tokenize("the deployment freeze runs over the winter holidays"))
    index.add("b", tokenize("rate limits are enforced with a token bucket algorithm"))
    index.add("c", tokenize("token bucket refill rate and burst capacity per tier"))
    index.build()

    results = index.search(tokenize("token bucket"), k=3)
    assert results[0][0] in {"b", "c"}
    assert all(score > 0 for _, score in results)
    assert index.search(tokenize("kubernetes"), k=3) == []


def test_bm25_idf_downweights_common_terms():
    index = BM25Index()
    for doc_id in "abcdefghij":
        index.add(doc_id, ["common", "term"])
    index.add("rare", ["common", "unique"])
    index.build()
    assert index.idf["unique"] > index.idf["common"]


def test_bm25_saturates_term_frequency():
    """Doubling a term's frequency must not double the score (that is the k1 term)."""
    index = BM25Index()
    index.add("few", ["alpha"] * 2 + ["filler"] * 50)
    index.add("many", ["alpha"] * 20 + ["filler"] * 50)
    index.build()
    few = index.score(["alpha"], "few")
    many = index.score(["alpha"], "many")
    assert many > few
    assert many < few * 10


# --------------------------------------------------------------------------- vectors & fusion


def test_vector_index_returns_nearest_neighbours_by_cosine():
    index = VectorIndex()
    index.add(["a", "b", "c"], np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32))
    results = index.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert [doc_id for doc_id, _ in results] == ["a", "b"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_rrf_rewards_documents_found_by_both_retrievers():
    # "b" is retrieved by both; "a" and "x" by only one each.
    fused = reciprocal_rank_fusion({"bm25": ["a", "b"], "vector": ["b", "x"]}, k=60)
    assert fused["b"] > fused["a"] > fused["x"]


def test_rrf_prefers_the_consistently_higher_ranked_document():
    # "b" is 2nd then 1st; "a" is 1st then 3rd - consistency wins.
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"], "vector": ["b", "c", "a"]}, k=60)
    assert fused["b"] > fused["a"] > fused["c"]


def test_rrf_is_scale_free():
    """Fusion uses ranks, so a retriever with huge scores cannot dominate."""
    fused = reciprocal_rank_fusion({"bm25": ["x"], "vector": ["y"]}, k=60)
    assert fused["x"] == fused["y"]


# --------------------------------------------------------------------------- reranker


def test_reranker_prefers_the_passage_where_terms_are_close_together():
    reranker = FeatureReranker()
    from lexiquery.chunking import Chunk

    tight = Chunk("t", "d", "Rollback procedure: run shipctl rollback to restore the release.", "Rollback", 0, 20)
    loose = Chunk("l", "d", "Rollback is mentioned here. " + "Filler sentence. " * 40 + "Release notes exist.", "Other", 1, 200)

    assert reranker.score("how do I rollback a release", tight) > reranker.score(
        "how do I rollback a release", loose
    )


def test_reranker_signals_are_explainable():
    from lexiquery.chunking import Chunk

    signals = FeatureReranker().signals(
        "canary rollout schedule",
        Chunk("c", "d", "The canary rollout schedule is 5% then 25%.", "Progressive delivery", 0, 15),
    )
    assert signals.coverage == pytest.approx(1.0)
    assert 0.0 <= signals.proximity <= 1.0
    assert signals.phrase > 0


# --------------------------------------------------------------------------- metrics


def test_retrieval_metrics():
    retrieved = ["b", "a", "c"]
    relevant = {"a"}
    assert hit_rate(retrieved, relevant, 1) == 0.0
    assert hit_rate(retrieved, relevant, 2) == 1.0
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], {"a"}, 2) < 1.0


def test_citation_and_keyword_metrics():
    sources = [{"chunk_id": "doc#0"}, {"chunk_id": "doc#1"}]
    assert citation_validity([{"chunk_id": "doc#0"}], sources) == 1.0
    assert citation_validity([{"chunk_id": "ghost#9"}], sources) == 0.0
    assert keyword_recall("Canary starts at 5% of traffic", ["5%"]) == 1.0
    assert keyword_recall("no numbers here", ["5%", "25%"]) == 0.0


# --------------------------------------------------------------------------- pipeline


def test_corpus_is_indexed(engine: LexiQuery):
    stats = engine.stats()
    assert stats["documents"] >= 10
    assert stats["chunks"] > stats["documents"]
    assert stats["built"] is True


def test_answers_are_grounded_and_cited(engine: LexiQuery):
    result = engine.ask("What is the canary rollout schedule for a production release?")
    assert result["refused"] is False
    assert "5%" in result["answer"]
    assert result["citations"], "an answer must cite its sources"
    assert citation_validity(result["citations"], result["sources"]) == 1.0
    assert result["sources"][0]["doc_id"] == "deployment-policy"


def test_exact_identifier_lookup_uses_the_lexical_path(engine: LexiQuery):
    """Rare identifiers are where dense-only retrieval fails."""
    result = engine.ask("What does RDS-4471 mean?")
    assert result["sources"][0]["doc_id"] == "database-runbook"
    assert "connection pool" in result["answer"].lower()


def test_paraphrased_question_still_retrieves(engine: LexiQuery):
    """No keyword overlap with the document heading - this is the vector path."""
    result = engine.ask("how long do we keep audit records around for compliance?")
    assert "data-retention" in [source["doc_id"] for source in result["sources"]]


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital city of France?",
        "How do I bake sourdough bread at home?",
        "Who won the 1998 football world cup?",
    ],
)
def test_out_of_scope_questions_are_refused(engine: LexiQuery, question: str):
    result = engine.ask(question)
    assert result["refused"] is True
    assert result["citations"] == []
    assert "could not find" in result["answer"].lower()


def test_retrieval_returns_scored_signals(engine: LexiQuery):
    results = engine.retrieve("rate limit headers", top_k=3)
    assert len(results) == 3
    top = results[0]
    assert top.bm25_rank is not None or top.vector_rank is not None
    assert top.rerank_score is not None
    assert set(top.signals) == {"coverage", "proximity", "phrase", "heading", "density"}


def test_index_persists_and_reloads(engine: LexiQuery, tmp_path: Path):
    path = engine.save(tmp_path / "index")
    reloaded = LexiQuery.load(path)
    assert reloaded.stats()["chunks"] == engine.stats()["chunks"]

    original = engine.ask("How quickly must a Sev1 be acknowledged?")
    restored = reloaded.ask("How quickly must a Sev1 be acknowledged?")
    assert restored["sources"][0]["doc_id"] == original["sources"][0]["doc_id"]


def test_new_documents_can_be_added_at_runtime(engine: LexiQuery):
    scratch = LexiQuery(Settings())
    scratch.add_document(Document(id="policy", text="# Policy\n## Laptops\nLaptops are replaced every three years."))
    scratch.build()
    result = scratch.ask("How often are laptops replaced?")
    assert "three years" in result["answer"]


# --------------------------------------------------------------------------- quality gates


def test_retrieval_quality_meets_the_bar(engine: LexiQuery):
    """These thresholds are the project's contract - a regression fails CI."""
    cases, out_of_scope = load_questions(QUESTIONS)
    report = evaluate(engine, cases, out_of_scope)
    summary = report["summary"]

    assert summary["hit@1"] >= 0.80, summary
    assert summary["hit@3"] >= 0.90, summary
    assert summary["mrr"] >= 0.85, summary
    assert summary["citation_validity"] == 1.0, summary
    assert summary["out_of_scope_refusal_rate"] == 1.0, summary


def test_hybrid_matches_the_best_retriever_on_keyword_questions(engine: LexiQuery):
    """On keyword-style questions BM25 is already near-perfect; hybrid must not regress it."""
    cases, _ = load_questions(QUESTIONS)
    table = compare_strategies(engine, cases)
    assert table["hybrid_rerank"]["hit@3"] >= 0.95
    assert table["hybrid_rerank"]["mrr"] >= table["vector_only"]["mrr"]
    assert table["hybrid_rerank"]["mrr"] >= table["bm25_only"]["mrr"] - 0.05


def test_hybrid_beats_lexical_retrieval_on_paraphrased_questions(engine: LexiQuery):
    """Paraphrases share little vocabulary with the source - this is why we fuse."""
    cases, _ = load_questions(QUESTIONS, key="paraphrase_questions")
    table = compare_strategies(engine, cases)
    assert table["hybrid_rerank"]["mrr"] > table["bm25_only"]["mrr"]
    assert table["hybrid_rerank"]["hit@3"] >= table["bm25_only"]["hit@3"]


# --------------------------------------------------------------------------- API


@pytest.fixture(scope="module")
def client(engine: LexiQuery):
    app = create_app(engine, Settings(api_key="test-key"))
    with TestClient(app) as test_client:
        test_client.headers.update({"X-API-Key": "test-key"})
        yield test_client


def test_health_is_public(client):
    client.headers.pop("X-API-Key")
    response = client.get("/health")
    client.headers.update({"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_requires_a_key(client):
    client.headers.pop("X-API-Key")
    assert client.post("/v1/ask", json={"question": "anything at all"}).status_code == 401
    client.headers.update({"X-API-Key": "test-key"})


def test_ask_endpoint(client):
    response = client.post("/v1/ask", json={"question": "How long are audit logs retained?"})
    assert response.status_code == 200
    body = response.json()
    assert "seven years" in body["answer"]
    assert body["sources"][0]["doc_id"] == "data-retention"
    assert body["latency_ms"] > 0


def test_search_endpoint_exposes_ranking_signals(client):
    response = client.post("/v1/search", json={"query": "postgres failover", "top_k": 3})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    assert "bm25_score" in results[0] and "vector_score" in results[0]


def test_document_listing(client):
    response = client.get("/v1/documents")
    assert response.status_code == 200
    assert len(response.json()["documents"]) >= 10


def test_validation_rejects_a_too_short_question(client):
    assert client.post("/v1/ask", json={"question": "a"}).status_code == 422
