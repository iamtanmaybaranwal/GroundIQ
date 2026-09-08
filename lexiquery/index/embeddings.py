"""Embedding providers.

The default provider is **local**: TF-IDF over word and character n-grams
reduced by truncated SVD (latent semantic analysis). It captures real semantic
similarity, needs no network, no GPU and no API key, and it makes the whole
project runnable and reproducible offline.

Hosted providers (OpenAI, Cohere, or any OpenAI-compatible endpoint) implement
the same three-method interface, so switching is one line of configuration -
see `.env.example` for where the API key goes.
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def fit(self, corpus: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LocalLSAEmbedder:
    """TF-IDF + truncated SVD, L2-normalised so dot product == cosine."""

    name = "local-lsa"

    def __init__(self, dimension: int = 192, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.dimension = dimension
        self.ngram_range = ngram_range
        self._pipeline = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            raise ValueError("cannot fit an embedder on an empty corpus")
        # SVD cannot produce more components than the rank of the matrix.
        components = max(2, min(self.dimension, len(corpus) - 1 if len(corpus) > 2 else 2))
        self._pipeline = make_pipeline(
            TfidfVectorizer(
                sublinear_tf=True,
                ngram_range=self.ngram_range,
                min_df=1,
                strip_accents="unicode",
                lowercase=True,
            ),
            TruncatedSVD(n_components=components, random_state=42),
            Normalizer(copy=False),
        )
        self._pipeline.fit(corpus)
        self.dimension = components
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("embedder must be fitted before use")
        return np.asarray(self._pipeline.transform(texts), dtype=np.float32)


class OpenAIEmbedder:  # pragma: no cover - requires network + credentials
    """OpenAI-compatible embeddings (also works with Azure OpenAI or vLLM).

    Set ``LEXIQUERY_OPENAI_API_KEY`` (see .env.example) and select this provider
    with ``LEXIQUERY_EMBEDDING_PROVIDER=openai``.
    """

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 1536) -> None:
        self.model = model
        self.dimension = dimension
        self.api_key = os.environ.get("LEXIQUERY_OPENAI_API_KEY", "")
        self.base_url = os.environ.get("LEXIQUERY_OPENAI_BASE_URL", "https://api.openai.com/v1")

    def fit(self, corpus: list[str]) -> None:
        return None  # hosted models are pre-trained: nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        if not self.api_key:
            raise RuntimeError(
                "LEXIQUERY_OPENAI_API_KEY is not set - fill it in .env or use the local provider"
            )
        response = httpx.post(
            self.base_url + "/embeddings",
            headers={"Authorization": "Bearer " + self.api_key},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        vectors = [item["embedding"] for item in response.json()["data"]]
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)


def build_embedder(provider: str = "local", **kwargs) -> EmbeddingProvider:
    if provider == "openai":
        return OpenAIEmbedder(**kwargs)
    return LocalLSAEmbedder(**kwargs)
