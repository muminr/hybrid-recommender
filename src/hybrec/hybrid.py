"""Two-stage hybrid recommender.

Stage 1 (retrieval): several cheap, complementary models each propose their
top-N unseen items. Their union is the candidate set.

Stage 2 (ranking): a LightGBM LambdaRank model re-orders the candidates using
every retriever's score and rank plus item, user, and user-item features.
LambdaRank optimises NDCG directly, so it learns *how much to trust each
retriever for which kind of user and item* instead of us hand-tuning blend
weights.

Leakage control: the ranker is trained on labels the retrievers never saw. The
training window is split again in time; retrievers fit on the earlier part,
the ranker learns to predict the later part. For serving, retrievers are then
refit on the full training window.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from hybrec.data import (
    Catalog,
    IndexMaps,
    interaction_matrix,
    relevant_items,
    temporal_split,
)
from hybrec.models import ContentTFIDF, ImplicitALS, ItemKNN, Popularity, Recommender
from hybrec.models.content import genre_matrix, release_years

SOURCE_LABELS = {
    "itemknn": "Viewers with similar taste liked it",
    "als": "Matches your overall taste profile",
    "content": "Similar genres and themes to what you rate highly",
    "popularity": "Popular right now",
}


def default_sources() -> list[Recommender]:
    return [ItemKNN(), ImplicitALS(), ContentTFIDF(), Popularity()]


@dataclass
class _Stage1:
    """Fitted retrievers plus the item/user statistics computed on the same data."""

    sources: list[Recommender]
    X_pos: sp.csr_matrix  # positives, what the models learned from
    X_seen: sp.csr_matrix  # everything rated, excluded from recommendations
    item_log_pop: np.ndarray
    item_pop_pct: np.ndarray  # popularity percentile, stable across data volumes
    item_mean_rating: np.ndarray
    user_log_activity: np.ndarray
    user_genre_profile: np.ndarray


def _fit_stage1(
    ratings: pd.DataFrame,
    catalog: Catalog,
    index: IndexMaps,
    genres: sp.csr_matrix,
    make_sources,
) -> _Stage1:
    X_pos = interaction_matrix(ratings, index)
    X_seen = interaction_matrix(ratings, index, threshold=-np.inf)
    sources = [m.fit(X_pos, catalog, index) for m in make_sources()]

    r = ratings[ratings["movieId"].isin(index.item_index)]
    item_idx = r["movieId"].map(index.item_index).to_numpy()
    n_ratings = np.bincount(item_idx, minlength=index.n_items).astype(np.float64)
    sum_ratings = np.bincount(item_idx, weights=r["rating"], minlength=index.n_items)
    global_mean = float(r["rating"].mean()) if len(r) else 3.5
    # Bayesian-smoothed mean so a single 5-star rating doesn't dominate
    item_mean = (sum_ratings + 10 * global_mean) / (n_ratings + 10)

    profile = np.asarray((X_pos @ genres).todense())
    norms = np.linalg.norm(profile, axis=1, keepdims=True)
    profile = profile / np.where(norms > 0, norms, 1.0)

    pop = np.asarray(X_pos.sum(axis=0)).ravel()
    return _Stage1(
        sources=sources,
        X_pos=X_pos,
        X_seen=X_seen,
        item_log_pop=np.log1p(pop),
        item_pop_pct=pd.Series(pop).rank(pct=True).to_numpy(),
        item_mean_rating=item_mean,
        user_log_activity=np.log1p(np.asarray(X_pos.sum(axis=1)).ravel()),
        user_genre_profile=profile,
    )


class HybridRecommender:
    def __init__(
        self,
        n_candidates: int = 100,
        ranker_params: dict | None = None,
        make_sources=default_sources,
        seed: int = 42,
    ):
        self.n_candidates = n_candidates
        self.make_sources = make_sources
        self.ranker_params = {
            "objective": "lambdarank",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 50,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "verbose": -1,
            **(ranker_params or {}),
        }

    # ------------------------------------------------------------------ fitting
    def fit(self, train: pd.DataFrame, catalog: Catalog, index: IndexMaps) -> HybridRecommender:
        self.catalog = catalog
        self.index = index
        self.genres, self.genre_names = genre_matrix(catalog, index)
        years = release_years(catalog, index)
        self.item_year = np.where(np.isnan(years), np.nanmedian(years), years)
        self.titles = catalog.movies.set_index("movieId")["title"].reindex(index.item_ids)

        # 1) Ranker training data: retrievers on the earlier window, labels from the later one.
        base, label_window = temporal_split(train, test_fraction=0.2)
        stage = _fit_stage1(base, catalog, index, self.genres, self.make_sources)
        labels = relevant_items(label_window, index)
        users = np.array(sorted(labels))
        feats = self._features(stage, users)
        feats["label"] = [
            int(i in labels[u]) for u, i in zip(feats["user"], feats["item"], strict=True)
        ]
        has_pos = feats.groupby("user")["label"].transform("max") > 0
        feats = feats[has_pos].sort_values("user", kind="mergesort")
        self.feature_names = [c for c in feats.columns if c not in ("user", "item", "label")]
        self.ranker = lgb.LGBMRanker(**self.ranker_params)
        self.ranker.fit(
            feats[self.feature_names],
            feats["label"],
            group=feats.groupby("user", sort=False).size().to_numpy(),
        )
        self.train_label_rate_ = float(feats["label"].mean())

        # 2) Serving: refit retrievers on the full training window.
        self.stage = _fit_stage1(train, catalog, index, self.genres, self.make_sources)
        return self

    # ----------------------------------------------------------------- features
    def _candidates(self, stage: _Stage1, user: int) -> tuple[np.ndarray, dict]:
        per_source = {}
        for m in stage.sources:
            scores = m.score(user).astype(np.float64)
            top = m.recommend(user, self.n_candidates, exclude=stage.X_seen)
            per_source[m.name] = (scores, top)
        cand = np.unique(np.concatenate([t for _, t in per_source.values()]))
        return cand, per_source

    def _features(self, stage: _Stage1, users) -> pd.DataFrame:
        frames = []
        n = self.n_candidates
        for u in users:
            cand, per_source = self._candidates(stage, u)
            if len(cand) == 0:
                continue
            f = {"user": np.full(len(cand), u), "item": cand}
            n_hits = np.zeros(len(cand))
            for name, (scores, top) in per_source.items():
                rank = np.full(stage.X_pos.shape[1], n + 1, dtype=np.float64)
                rank[top] = np.arange(1, len(top) + 1)
                s = scores[cand]
                # Raw scores drift with data volume (item-kNN sums grow with history,
                # popularity counts grow with traffic), so the ranker would learn
                # thresholds that break at serving time. Within-user z-scores and
                # ranks are scale-free and transfer across training windows.
                f[f"{name}_z"] = (s - s.mean()) / (s.std() + 1e-9)
                f[f"{name}_rank"] = rank[cand]
                n_hits += rank[cand] <= n
            f["n_sources"] = n_hits
            f["item_pop_pct"] = stage.item_pop_pct[cand]
            f["item_mean_rating"] = stage.item_mean_rating[cand]
            f["item_year"] = self.item_year[cand]
            f["user_log_activity"] = np.full(len(cand), stage.user_log_activity[u])
            f["genre_affinity"] = np.asarray(
                self.genres[cand] @ stage.user_genre_profile[u]
            ).ravel()
            frames.append(pd.DataFrame(f))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------ serving
    def recommend(self, user: int, k: int = 10, mode: str = "ranker") -> list[int]:
        """mode='ranker' (stage 2) or 'blend' (equal-weight normalised score blend baseline)."""
        feats = self._features(self.stage, [user])
        if feats.empty:
            return []
        if mode == "ranker":
            score = self.ranker.predict(feats[self.feature_names])
        elif mode == "blend":
            score = sum(feats[f"{m.name}_z"].to_numpy() for m in self.stage.sources)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        order = np.argsort(-score, kind="stable")[:k]
        return [int(i) for i in feats["item"].to_numpy()[order]]

    def recommend_cold_start(self, k: int = 10, genres: list[str] | None = None) -> list[int]:
        """For users with no history: popularity, optionally steered by stated genres."""
        pop = self.stage.item_log_pop
        if genres:
            content = next(m for m in self.stage.sources if isinstance(m, ContentTFIDF))
            g = content.score_genres(genres)
            score = (g > 0) * (1.0 + g) * (1.0 + pop) + pop * 1e-3
        else:
            score = pop
        return [int(i) for i in np.argsort(-score, kind="stable")[:k]]

    def explain(self, user: int, item: int) -> list[str]:
        reasons = []
        for m in self.stage.sources:
            if m.name == "popularity":
                continue
            top = m.recommend(user, self.n_candidates, exclude=self.stage.X_seen)
            if item in set(top.tolist()):
                reasons.append(SOURCE_LABELS[m.name])
        knn = next(m for m in self.stage.sources if isinstance(m, ItemKNN))
        because = [str(self.titles.iloc[i]) for i in knn.because_of(user, item)]
        if because:
            reasons.append("Because you liked " + " and ".join(because))
        return reasons or [SOURCE_LABELS["popularity"]]

    def feature_importance(self) -> pd.Series:
        imp = self.ranker.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
