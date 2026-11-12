"""Popularity baseline: the bar every personalized model has to clear."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from hybrec.data import Catalog, IndexMaps
from hybrec.models.base import Recommender


class Popularity(Recommender):
    name = "popularity"

    def fit(self, X: sp.csr_matrix, catalog: Catalog, index: IndexMaps) -> Popularity:
        self.counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, user: int) -> np.ndarray:
        return self.counts
