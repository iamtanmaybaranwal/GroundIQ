"""Configuration (12-factor, env driven)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEXIQUERY_", env_file=".env", extra="ignore")

    # ---------------------------------------------------------------- corpus
    corpus_path: str = "corpus"
    index_path: str = "artifacts/index"

    # ---------------------------------------------------------------- chunking
    chunk_target_tokens: int = 220
    chunk_overlap_sentences: int = 1

    # ---------------------------------------------------------------- retrieval
    embedding_provider: str = "local"  # local | openai
    embedding_dimension: int = 192
    reranker: str = "feature"  # feature | cross-encoder
    answerer: str = "extractive"  # extractive | anthropic | openai

    candidates_per_retriever: int = 25
    rrf_k: int = 60
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    # Blend of fused retrieval rank and reranker score in the final ordering.
    rerank_weight: float = 0.5

    top_k: int = 5
    context_token_budget: int = 1_800

    # ---------------------------------------------------------------- guardrails
    # Below these, the system says "I don't know" instead of guessing.
    min_coverage: float = 0.34
    min_vector_score: float = 0.35

    api_key: str = "change-me-lexiquery-api-key"
    log_level: str = "INFO"


settings = Settings()
