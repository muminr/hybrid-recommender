"""FastAPI service over a trained artifact.

MODEL_PATH=artifacts/model.joblib uvicorn hybrec.service:app --reload
"""

from __future__ import annotations

import os
import time
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from hybrec.hybrid import HybridRecommender
from hybrec.models import ItemKNN


class Item(BaseModel):
    movie_id: int
    title: str
    genres: str
    reasons: list[str] = []


class Recommendations(BaseModel):
    user_id: int | None
    strategy: str
    items: list[Item]
    latency_ms: float


def create_app(model: HybridRecommender | None = None) -> FastAPI:
    app = FastAPI(
        title="hybrid-recommender",
        description="Two-stage recommender: multi-source retrieval + LightGBM LambdaRank.",
        version="0.1.0",
    )
    state: dict = {"model": model}

    def get_model() -> HybridRecommender:
        if state["model"] is None:
            path = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
            if not os.path.exists(path):
                raise HTTPException(503, f"No model artifact at {path}. Run hybrec.train first.")
            state["model"] = joblib.load(path)
        return state["model"]

    def to_item(m: HybridRecommender, item_idx: int, reasons: list[str] | None = None) -> Item:
        movie_id = int(m.index.item_ids[item_idx])
        row = m.catalog.movies.loc[m.catalog.movies["movieId"] == movie_id].iloc[0]
        return Item(
            movie_id=movie_id, title=row["title"], genres=row["genres"], reasons=reasons or []
        )

    @app.get("/health")
    def health() -> dict:
        m = get_model()
        return {"status": "ok", "users": m.index.n_users, "items": m.index.n_items}

    @app.get("/recommendations/{user_id}", response_model=Recommendations)
    def recommend(
        user_id: int,
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        explain: bool = True,
    ) -> Recommendations:
        m = get_model()
        t0 = time.perf_counter()
        u = m.index.user_index.get(user_id)
        if u is None or m.stage.X_pos[u].nnz == 0:
            items = [to_item(m, i, ["Popular right now"]) for i in m.recommend_cold_start(k)]
            strategy = "cold_start_popularity"
        else:
            recs = m.recommend(u, k)
            items = [to_item(m, i, m.explain(u, i) if explain else None) for i in recs]
            strategy = "hybrid_ranker"
        return Recommendations(
            user_id=user_id,
            strategy=strategy,
            items=items,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    @app.get("/recommendations", response_model=Recommendations)
    def recommend_new_user(
        genres: Annotated[list[str] | None, Query()] = None,
        k: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> Recommendations:
        """Cold start for a brand-new user, optionally steered by genres they pick at sign-up."""
        m = get_model()
        t0 = time.perf_counter()
        recs = m.recommend_cold_start(k, genres=genres)
        reason = f"Popular in {', '.join(genres)}" if genres else "Popular right now"
        return Recommendations(
            user_id=None,
            strategy="cold_start_genres" if genres else "cold_start_popularity",
            items=[to_item(m, i, [reason]) for i in recs],
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    @app.get("/items/{movie_id}/similar", response_model=list[Item])
    def similar(movie_id: int, k: Annotated[int, Query(ge=1, le=50)] = 10) -> list[Item]:
        m = get_model()
        i = m.index.item_index.get(movie_id)
        if i is None:
            raise HTTPException(404, f"Unknown movie_id {movie_id}")
        knn = next(s for s in m.stage.sources if isinstance(s, ItemKNN))
        return [
            to_item(m, j, [f"Co-liked (similarity {s:.2f})"]) for j, s in knn.similar_items(i, k)
        ]

    return app


app = create_app()
