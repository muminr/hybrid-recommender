"""Content-based retrieval from item metadata (genres + user tags).

Collaborative models know nothing about an item until people interact with it.
Content similarity covers that cold-start gap: a brand-new title with the right
genres and tags can still be retrieved for the right users.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from hybrec.data import Catalog, IndexMaps
from hybrec.models.base import Recommender


def item_documents(catalog: Catalog, index: IndexMaps) -> list[str]:
    """One text document per item: genre tokens plus normalised tag tokens."""
    genres = catalog.movies.set_index("movieId")["genres"].to_dict()
    tags = (
        catalog.tags.assign(tag=catalog.tags["tag"].astype(str).str.lower())
        .groupby("movieId")["tag"]
        .apply(lambda s: " ".join(re.sub(r"[^a-z0-9]+", "_", t).strip("_") for t in s))
        .to_dict()
        if len(catalog.tags)
        else {}
    )
    docs = []
    for m in index.item_ids:
        g = str(genres.get(int(m), ""))
        g = "" if g == "(no genres listed)" else " ".join(f"genre_{x}" for x in g.split("|") if x)
        docs.append(f"{g} {tags.get(int(m), '')}".strip())
    return docs


class ContentTFIDF(Recommender):
    name = "content"

    def fit(self, X: sp.csr_matrix, catalog: Catalog, index: IndexMaps) -> ContentTFIDF:
        vec = TfidfVectorizer(token_pattern=r"[^\s]+", lowercase=False, sublinear_tf=True)
        self.item_vectors = normalize(vec.fit_transform(item_documents(catalog, index)))
        self.vocabulary = vec.get_feature_names_out()
        self.X = X.tocsr()
        return self

    def user_profile(self, user: int) -> sp.csr_matrix:
        row = self.X[user]
        if row.nnz == 0:
            return sp.csr_matrix((1, self.item_vectors.shape[1]))
        return normalize(row @ self.item_vectors)

    def score(self, user: int) -> np.ndarray:
        return np.asarray((self.item_vectors @ self.user_profile(user).T).todense()).ravel()

    def score_genres(self, genres: list[str]) -> np.ndarray:
        """Cold-start scoring for a user we know nothing about except stated genres."""
        tokens = {f"genre_{g}" for g in genres}
        mask = np.isin(self.vocabulary, list(tokens))
        if not mask.any():
            return np.zeros(self.item_vectors.shape[0])
        return np.asarray(self.item_vectors[:, mask].sum(axis=1)).ravel()


def genre_matrix(catalog: Catalog, index: IndexMaps) -> tuple[sp.csr_matrix, list[str]]:
    """Binary item x genre matrix, used for ranker features."""
    g = catalog.movies.set_index("movieId")["genres"].reindex(index.item_ids).fillna("")
    dummies = g.str.get_dummies(sep="|")
    dummies = dummies.drop(columns=["(no genres listed)"], errors="ignore")
    return sp.csr_matrix(dummies.to_numpy(dtype=np.float32)), list(dummies.columns)


def release_years(catalog: Catalog, index: IndexMaps) -> np.ndarray:
    titles = catalog.movies.set_index("movieId")["title"].reindex(index.item_ids).fillna("")
    years = titles.str.extract(r"\((\d{4})\)\s*$")[0]
    return pd.to_numeric(years, errors="coerce").to_numpy(dtype=np.float64)
