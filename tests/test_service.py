import pytest
from fastapi.testclient import TestClient

from hybrec.service import create_app


@pytest.fixture(scope="module")
def client(hybrid):
    return TestClient(create_app(hybrid))


def test_health(client, index):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["items"] == index.n_items


def test_known_user_gets_explained_ranked_list(client, index):
    user_id = int(index.user_ids[0])
    body = client.get(f"/recommendations/{user_id}?k=5").json()
    assert body["strategy"] == "hybrid_ranker"
    assert len(body["items"]) == 5
    assert all(item["reasons"] for item in body["items"])


def test_unknown_user_falls_back_to_cold_start(client):
    body = client.get("/recommendations/999999?k=3").json()
    assert body["strategy"] == "cold_start_popularity" and len(body["items"]) == 3


def test_new_user_with_genres(client):
    body = client.get("/recommendations?genres=Comedy&genres=Romance&k=5").json()
    assert body["strategy"] == "cold_start_genres"
    assert all(("Comedy" in i["genres"]) or ("Romance" in i["genres"]) for i in body["items"])


def test_similar_items(client, index):
    movie_id = int(index.item_ids[0])
    assert client.get(f"/items/{movie_id}/similar?k=3").status_code == 200
    assert client.get("/items/-1/similar").status_code == 404


def test_k_is_validated(client, index):
    assert client.get(f"/recommendations/{int(index.user_ids[0])}?k=0").status_code == 422
