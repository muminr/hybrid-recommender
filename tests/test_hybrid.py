import numpy as np

from hybrec.data import relevant_items
from hybrec.metrics import ndcg_at_k


def test_recommendations_are_unseen_and_unique(hybrid):
    for u in range(10):
        recs = hybrid.recommend(u, 10)
        seen = set(hybrid.stage.X_seen[u].indices)
        assert len(recs) == len(set(recs)) == 10
        assert not seen & set(recs)


def test_ranker_beats_popularity_floor(hybrid, split, index):
    truth = relevant_items(split[1], index)
    pop = next(m for m in hybrid.stage.sources if m.name == "popularity")
    users = sorted(truth)
    ranked = np.mean([ndcg_at_k(hybrid.recommend(u, 10), truth[u], 10) for u in users])
    popular = np.mean(
        [ndcg_at_k(pop.recommend(u, 10, hybrid.stage.X_seen).tolist(), truth[u], 10) for u in users]
    )
    assert ranked > popular


def test_features_are_scale_free(hybrid):
    """Guards against train/serve skew: retriever features must be per-user z-scores."""
    feats = hybrid._features(hybrid.stage, [0, 1, 2])
    for col in [c for c in feats.columns if c.endswith("_z")]:
        means = feats.groupby("user")[col].mean().abs()
        assert (means < 1e-6).all(), col
    assert not any(c.endswith("_score") for c in hybrid.feature_names)


def test_cold_start_respects_genres(hybrid):
    recs = hybrid.recommend_cold_start(10, genres=["Horror"])
    genres = hybrid.catalog.movies.set_index("movieId")["genres"]
    assert all("Horror" in genres[int(hybrid.index.item_ids[i])] for i in recs)


def test_explanations_are_present(hybrid):
    u = 0
    for item in hybrid.recommend(u, 5):
        assert hybrid.explain(u, item)
