import numpy as np
import pandas as pd
import pytest

from hybrec.data import Catalog, build_index, interaction_matrix, relevant_items
from hybrec.metrics import ndcg_at_k
from hybrec.models import ContentTFIDF, ImplicitALS, ItemKNN, Popularity

MODELS = [Popularity, ItemKNN, ImplicitALS, ContentTFIDF]


@pytest.fixture(scope="module")
def fitted(catalog, index, X_train):
    return {cls.name: cls().fit(X_train, catalog, index) for cls in MODELS}


@pytest.mark.parametrize("name", [c.name for c in MODELS])
def test_scores_cover_catalog_and_exclude_seen(fitted, X_train, index, name):
    model = fitted[name]
    assert model.score(0).shape == (index.n_items,)
    recs = model.recommend(0, 20, exclude=X_train)
    seen = set(X_train[0].indices)
    assert len(recs) == len(set(recs)) and not seen & set(recs.tolist())


@pytest.mark.parametrize("name", ["itemknn", "als", "content"])
def test_personalised_models_beat_random(fitted, split, index, X_train, name):
    truth = relevant_items(split[1], index)
    rng = np.random.default_rng(0)
    model, users = fitted[name], sorted(truth)
    ours = np.mean(
        [ndcg_at_k(model.recommend(u, 10, X_train).tolist(), truth[u], 10) for u in users]
    )
    rand = np.mean([ndcg_at_k(rng.choice(index.n_items, 10).tolist(), truth[u], 10) for u in users])
    assert ours > 2 * rand


def test_itemknn_keeps_at_most_k_neighbours(catalog, index, X_train):
    knn = ItemKNN(k=5).fit(X_train, catalog, index)
    assert np.diff(knn.sim.indptr).max() <= 5
    assert (knn.sim.data >= 0).all()


def test_content_retrieves_never_rated_items():
    """Cold-start items: no interactions at all, but metadata makes them retrievable."""
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["A (2000)", "B (2001)", "Brand New (2026)"],
            "genres": ["Horror", "Comedy", "Horror"],
        }
    )
    ratings = pd.DataFrame(
        {"userId": [1, 1], "movieId": [1, 2], "rating": [5.0, 1.0], "timestamp": [1, 2]}
    )
    cat = Catalog(ratings=ratings, movies=movies)
    idx = build_index(cat)
    X = interaction_matrix(ratings, idx)
    model = ContentTFIDF().fit(X, cat, idx)
    assert model.recommend(0, 1, exclude=interaction_matrix(ratings, idx, -np.inf))[0] == 2
    assert model.score_genres(["Comedy"]).argmax() == 1
