"""Common interface for first-stage (candidate retrieval) models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

from hybrec.data import Catalog, IndexMaps


class Recommender(ABC):
    """A model that scores every item for a user.

    `fit` receives the binary train interaction matrix plus the catalog, so
    content-aware models can use item metadata and pure CF models can ignore it.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: sp.csr_matrix, catalog: Catalog, index: IndexMaps) -> Recommender: ...

    @abstractmethod
    def score(self, user: int) -> np.ndarray:
        """Dense score vector over all items for one user index."""

    def recommend(self, user: int, k: int, exclude: sp.csr_matrix | None = None) -> np.ndarray:
        """Top-k item indices, excluding items the user already interacted with."""
        scores = self.score(user).astype(np.float64, copy=True)
        if exclude is not None:
            seen = exclude.indices[exclude.indptr[user] : exclude.indptr[user + 1]]
            scores[seen] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return top[np.isfinite(scores[top])]
