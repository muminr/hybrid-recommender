"""Data loading, interaction matrices, temporal splitting, and a synthetic generator.

All data is in MovieLens CSV format (ratings.csv, movies.csv, tags.csv), so the
synthetic generator and the real dataset flow through exactly the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

POSITIVE_THRESHOLD = 4.0  # ratings >= 4.0 count as a positive (implicit) signal


@dataclass
class Catalog:
    """Raw MovieLens-format tables."""

    ratings: pd.DataFrame  # userId, movieId, rating, timestamp
    movies: pd.DataFrame  # movieId, title, genres
    tags: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    )


def load_movielens(path: str | Path) -> Catalog:
    path = Path(path)
    ratings = pd.read_csv(path / "ratings.csv")
    movies = pd.read_csv(path / "movies.csv")
    tags_path = path / "tags.csv"
    tags = (
        pd.read_csv(tags_path)
        if tags_path.exists()
        else pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    )
    return Catalog(ratings=ratings, movies=movies, tags=tags)


@dataclass
class IndexMaps:
    """Bidirectional maps between external ids and contiguous matrix indices."""

    user_ids: np.ndarray  # index -> userId
    item_ids: np.ndarray  # index -> movieId

    def __post_init__(self) -> None:
        self.user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self.item_index = {int(m): i for i, m in enumerate(self.item_ids)}

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def build_index(catalog: Catalog) -> IndexMaps:
    """Items come from the full movie catalog (not just rated ones) so that
    content-based retrieval can surface items nobody has interacted with yet."""
    users = np.sort(catalog.ratings["userId"].unique())
    items = np.sort(catalog.movies["movieId"].unique())
    return IndexMaps(user_ids=users, item_ids=items)


def interaction_matrix(
    ratings: pd.DataFrame, index: IndexMaps, threshold: float = POSITIVE_THRESHOLD
) -> sp.csr_matrix:
    """Binary user x item matrix of positive interactions."""
    pos = ratings[ratings["rating"] >= threshold]
    pos = pos[pos["userId"].isin(index.user_index) & pos["movieId"].isin(index.item_index)]
    rows = pos["userId"].map(index.user_index).to_numpy()
    cols = pos["movieId"].map(index.item_index).to_numpy()
    data = np.ones(len(pos), dtype=np.float32)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(index.n_users, index.n_items))
    mat.data[:] = 1.0  # collapse duplicates
    return mat


def temporal_split(
    ratings: pd.DataFrame, test_fraction: float = 0.2, min_interactions: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user temporal holdout: each user's most recent `test_fraction` of
    ratings goes to test. This mimics production (predict the future from the
    past) and avoids the leakage a random split introduces."""
    ratings = ratings.sort_values(["userId", "timestamp"], kind="mergesort")
    counts = ratings.groupby("userId")["movieId"].transform("size")
    eligible = ratings[counts >= min_interactions].copy()
    rank = eligible.groupby("userId").cumcount()
    size = eligible.groupby("userId")["movieId"].transform("size")
    n_test = np.maximum(1, np.floor(size * test_fraction)).astype(int)
    is_test = rank >= (size - n_test)
    train = pd.concat([ratings[counts < min_interactions], eligible[~is_test]])
    test = eligible[is_test]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def relevant_items(
    ratings: pd.DataFrame, index: IndexMaps, threshold: float = POSITIVE_THRESHOLD
) -> dict[int, set[int]]:
    """user index -> set of positively-rated item indices."""
    pos = ratings[ratings["rating"] >= threshold]
    pos = pos[pos["userId"].isin(index.user_index) & pos["movieId"].isin(index.item_index)]
    out: dict[int, set[int]] = {}
    for u, m in zip(pos["userId"], pos["movieId"], strict=True):
        out.setdefault(index.user_index[int(u)], set()).add(index.item_index[int(m)])
    return out


GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

TAG_VOCAB = {
    "Action": ["explosions", "fight scenes"],
    "Adventure": ["quest", "epic"],
    "Animation": ["pixar", "animated"],
    "Children": ["family", "kids"],
    "Comedy": ["funny", "witty"],
    "Crime": ["heist", "mafia"],
    "Documentary": ["true story", "educational"],
    "Drama": ["emotional", "character study"],
    "Fantasy": ["magic", "dragons"],
    "Horror": ["scary", "gore"],
    "Musical": ["songs", "broadway"],
    "Mystery": ["twist ending", "whodunit"],
    "Romance": ["love story", "romantic"],
    "Sci-Fi": ["space", "dystopia"],
    "Thriller": ["suspense", "tense"],
    "War": ["wwii", "military"],
    "Western": ["cowboys", "frontier"],
}


def synthetic_catalog(
    n_users: int = 300,
    n_items: int = 800,
    n_factors: int = 8,
    ratings_per_user: tuple[int, int] = (20, 120),
    seed: int = 7,
) -> Catalog:
    """Generate a MovieLens-shaped dataset with real latent structure.

    Items get 1-3 genres; user taste is a mix over genres plus latent factors,
    and item popularity is long-tailed. This is enough signal that collaborative,
    content-based, and hybrid models separate measurably in tests and CI.
    """
    rng = np.random.default_rng(seed)
    g = len(GENRES)

    item_genres = np.zeros((n_items, g))
    for i in range(n_items):
        k = rng.integers(1, 4)
        item_genres[i, rng.choice(g, size=k, replace=False)] = 1.0
    item_factors = rng.normal(0, 1, (n_items, n_factors))
    popularity = rng.pareto(1.5, n_items) + 1.0
    popularity /= popularity.sum()

    user_genre_taste = rng.dirichlet(np.full(g, 0.3), n_users)
    user_factors = rng.normal(0, 1, (n_users, n_factors))

    years = rng.integers(1960, 2024, n_items)
    movies = pd.DataFrame(
        {
            "movieId": np.arange(1, n_items + 1),
            "title": [f"Synthetic Movie {i + 1} ({years[i]})" for i in range(n_items)],
            "genres": ["|".join(np.array(GENRES)[item_genres[i] > 0]) for i in range(n_items)],
        }
    )

    rows = []
    t0 = 1_500_000_000
    for u in range(n_users):
        n = int(rng.integers(*ratings_per_user))
        affinity = 2.0 * item_genres @ user_genre_taste[u] + 0.35 * item_factors @ user_factors[
            u
        ] / np.sqrt(n_factors)
        p = np.exp(affinity) * popularity
        p /= p.sum()
        chosen = rng.choice(n_items, size=min(n, n_items), replace=False, p=p)
        z = (affinity[chosen] - affinity.mean()) / (affinity.std() + 1e-9)
        stars = np.clip(np.round((3.2 + 0.9 * z + rng.normal(0, 0.6, len(chosen))) * 2) / 2, 0.5, 5)
        ts = t0 + np.sort(rng.integers(0, 50_000_000, len(chosen)))
        for i, r, t in zip(chosen, stars, ts, strict=True):
            rows.append((u + 1, i + 1, float(r), int(t)))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])

    tag_rows = []
    for i in rng.choice(n_items, size=n_items // 2, replace=False):
        for gi in np.flatnonzero(item_genres[i]):
            tag_rows.append((1, i + 1, str(rng.choice(TAG_VOCAB[GENRES[gi]])), t0))
    tags = pd.DataFrame(tag_rows, columns=["userId", "movieId", "tag", "timestamp"])
    return Catalog(ratings=ratings, movies=movies, tags=tags)


def write_catalog(catalog: Catalog, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    catalog.ratings.to_csv(path / "ratings.csv", index=False)
    catalog.movies.to_csv(path / "movies.csv", index=False)
    catalog.tags.to_csv(path / "tags.csv", index=False)
