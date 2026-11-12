"""Implicit-feedback matrix factorisation via Alternating Least Squares.

Follows Hu, Koren & Volinsky (2008), "Collaborative Filtering for Implicit
Feedback Datasets": every (user, item) cell has a preference p in {0, 1} and a
confidence c = 1 + alpha * r. Unobserved cells are weak negatives, not missing.

Written directly in NumPy rather than pulled from a library so the maths is
visible: each half-step is a ridge regression solved in closed form, using the
YtY trick so the cost per user scales with the items they touched, not the
whole catalogue.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from hybrec.data import Catalog, IndexMaps
from hybrec.models.base import Recommender


def _als_step(R: sp.csr_matrix, fixed: np.ndarray, reg: float, alpha: float) -> np.ndarray:
    """Solve for one side's factors given the other side held fixed."""
    n, f = R.shape[0], fixed.shape[1]
    YtY = fixed.T @ fixed
    reg_eye = reg * np.eye(f)
    out = np.zeros((n, f), dtype=np.float64)
    for u in range(n):
        start, end = R.indptr[u], R.indptr[u + 1]
        if start == end:
            continue
        idx = R.indices[start:end]
        conf = 1.0 + alpha * R.data[start:end]
        Yu = fixed[idx]
        # A = YtY + Yu^T (C_u - I) Yu + reg*I ;  b = Yu^T C_u p_u  (p_u = 1 on observed)
        A = YtY + (Yu.T * (conf - 1.0)) @ Yu + reg_eye
        b = Yu.T @ conf
        out[u] = np.linalg.solve(A, b)
    return out


class ImplicitALS(Recommender):
    name = "als"

    def __init__(
        self,
        factors: int = 32,
        iterations: int = 10,
        reg: float = 10.0,
        alpha: float = 5.0,
        seed: int = 42,
    ):
        self.factors = factors
        self.iterations = iterations
        self.reg = reg
        self.alpha = alpha
        self.seed = seed

    def fit(self, X: sp.csr_matrix, catalog: Catalog, index: IndexMaps) -> ImplicitALS:
        rng = np.random.default_rng(self.seed)
        R = X.tocsr().astype(np.float64)
        RT = R.T.tocsr()
        self.user_factors = rng.normal(0, 0.01, (R.shape[0], self.factors))
        self.item_factors = rng.normal(0, 0.01, (R.shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = _als_step(R, self.item_factors, self.reg, self.alpha)
            self.item_factors = _als_step(RT, self.user_factors, self.reg, self.alpha)
        return self

    def score(self, user: int) -> np.ndarray:
        return self.item_factors @ self.user_factors[user]
