import pytest

from hybrec.data import build_index, interaction_matrix, synthetic_catalog, temporal_split
from hybrec.hybrid import HybridRecommender


@pytest.fixture(scope="session")
def catalog():
    return synthetic_catalog(n_users=200, n_items=500, seed=11)


@pytest.fixture(scope="session")
def index(catalog):
    return build_index(catalog)


@pytest.fixture(scope="session")
def split(catalog):
    return temporal_split(catalog.ratings)


@pytest.fixture(scope="session")
def X_train(split, index):
    return interaction_matrix(split[0], index)


@pytest.fixture(scope="session")
def hybrid(catalog, index, split):
    return HybridRecommender(ranker_params={"n_estimators": 50}).fit(split[0], catalog, index)
