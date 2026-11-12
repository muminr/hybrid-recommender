"""Item-based collaborative filtering with a pruned cosine-similarity graph.

"People who liked X also liked Y." Keeping only the top-k neighbours per item
keeps the similarity matrix sparse (memory scales with n_items * k, not
n_items^2) and makes every recommendation explainable by its neighbour items.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from hybrec.data import Catalog, IndexMaps
from hybrec.models.base import Recommender


def _top_k_per_row(S: sp.csr_matrix, k: int) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        start, end = S.indptr[i], S.indptr[i + 1]
        if start == end:
            continue
        idx, data = S.indices[start:end], S.data[start:end]
        if len(data) > k:
            keep = np.argpartition(-data, k - 1)[:k]
            idx, data = idx[keep], data[keep]
        rows.append(np.full(len(idx), i))
        cols.append(idx)
        vals.append(data)
    if not rows:
        return sp.csr_matrix(S.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=S.shape,
        dtype=np.float32,
    )


class ItemKNN(Recommender):
    name = "itemknn"

    def __init__(self, k: int = 50, shrinkage: float = 10.0):
        self.k = k
        self.shrinkage = shrinkage  # damps similarities supported by few co-occurrences

    def fit(self, X: sp.csr_matrix, catalog: Catalog, index: IndexMaps) -> ItemKNN:
        Xc = X.tocsc().astype(np.float32)
        co = (Xc.T @ Xc).tocsr()  # item x item co-occurrence counts
        co.setdiag(0)
        co.eliminate_zeros()
        norms = np.sqrt(np.asarray(Xc.multiply(Xc).sum(axis=0)).ravel())
        co = co.tocoo()
        denom = norms[co.row] * norms[co.col] + self.shrinkage
        sim = sp.csr_matrix((co.data / denom, (co.row, co.col)), shape=co.shape)
        self.sim = _top_k_per_row(sim, self.k)
        self.sim_T = self.sim.T.tocsr()
        self.X = X.tocsr()
        return self

    def score(self, user: int) -> np.ndarray:
        row = self.X[user]
        return np.asarray((row @ self.sim).todense()).ravel()

    def because_of(self, user: int, item: int, n: int = 2) -> list[int]:
        """Items in the user's history that contribute most to this item's score."""
        history = self.X.indices[self.X.indptr[user] : self.X.indptr[user + 1]]
        if len(history) == 0:
            return []
        contrib = np.asarray(self.sim[history, item].todense()).ravel()
        order = np.argsort(-contrib)[:n]
        return [int(history[i]) for i in order if contrib[i] > 0]

    def similar_items(self, item: int, n: int = 10) -> list[tuple[int, float]]:
        start, end = self.sim.indptr[item], self.sim.indptr[item + 1]
        idx, data = self.sim.indices[start:end], self.sim.data[start:end]
        order = np.argsort(-data)[:n]
        return [(int(idx[i]), float(data[i])) for i in order]
