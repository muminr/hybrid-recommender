import numpy as np

from hybrec.data import interaction_matrix, synthetic_catalog, temporal_split


def test_synthetic_is_deterministic():
    a = synthetic_catalog(n_users=20, n_items=50, seed=3)
    b = synthetic_catalog(n_users=20, n_items=50, seed=3)
    assert a.ratings.equals(b.ratings)
    assert set(a.ratings.columns) == {"userId", "movieId", "rating", "timestamp"}


def test_temporal_split_never_puts_past_in_test(split):
    train, test = split
    last_train = train.groupby("userId")["timestamp"].max()
    first_test = test.groupby("userId")["timestamp"].min()
    common = first_test.index.intersection(last_train.index)
    assert (first_test[common] >= last_train[common]).all()


def test_temporal_split_keeps_every_rating(catalog, split):
    train, test = split
    assert len(train) + len(test) == len(catalog.ratings)


def test_small_users_stay_in_train():
    ratings = synthetic_catalog(n_users=10, n_items=50, ratings_per_user=(2, 4)).ratings
    train, test = temporal_split(ratings, min_interactions=5)
    assert test.empty and len(train) == len(ratings)


def test_interaction_matrix_is_binary_positives(split, index):
    train, _ = split
    X = interaction_matrix(train, index)
    assert X.shape == (index.n_users, index.n_items)
    assert set(np.unique(X.data)) == {1.0}
    assert X.nnz == len(train[train["rating"] >= 4.0].drop_duplicates(["userId", "movieId"]))
