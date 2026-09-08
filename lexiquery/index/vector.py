"""Dense vector index: exact cosine search over an L2-normalised matrix.

At corpus sizes up to a few hundred thousand chunks a single matrix multiply is
both the simplest and the fastest option (it is one BLAS call), and unlike an
approximate index it has no recall loss to tune. The interface is deliberately
the same shape as an ANN library's, so swapping in FAISS/HNSW later touches only
this file.
"""
from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.matrix: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must be the same length")
        vectors = self._normalise(np.asarray(vectors, dtype=np.float32))
        self.ids.extend(ids)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-9, None)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if self.matrix is None or not self.ids:
            return []
        query = self._normalise(np.atleast_2d(np.asarray(query, dtype=np.float32)))[0]

        scores = self.matrix @ query  # cosine, because everything is unit length
        k = min(k, len(self.ids))
        # argpartition finds the top-k in O(n) before sorting only those k.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    @property
    def size(self) -> int:
        return len(self.ids)
