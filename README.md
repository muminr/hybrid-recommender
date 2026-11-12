# hybrid-recommender

[![CI](https://github.com/muminr/hybrid-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/muminr/hybrid-recommender/actions/workflows/ci.yml)

A two-stage recommender system built from first principles: several complementary
retrievers propose candidates, and a LightGBM LambdaRank model learns how to order them.
It includes offline evaluation with a leakage-safe temporal split, cold-start handling,
per-item explanations, and a FastAPI service.

I've led personalization and ML platform teams at enterprise scale. This repo is a
hands-on, from-scratch implementation of the core patterns behind those systems: no
recommender libraries, and every model is small enough to read in one sitting.

## Architecture

```mermaid
flowchart LR
    U[User history] --> R1[Item-kNN<br/>co-liked items]
    U --> R2[Implicit ALS<br/>latent factors]
    U --> R3[Content TF-IDF<br/>genres + tags]
    U --> R4[Popularity]
    R1 & R2 & R3 & R4 --> C[Candidate union<br/>top-100 per source]
    C --> F[Features<br/>per-user z-scores, ranks,<br/>agreement, item and genre signals]
    F --> L[LightGBM LambdaRank]
    L --> K[Top-K + explanations]
    N[New user] -.->|no history| CS[Cold start:<br/>popularity, steered by<br/>chosen genres]
```

| Stage | Component | Why it's there |
|---|---|---|
| Retrieval | **Item-kNN** (`models/itemknn.py`) | "People who liked X also liked Y." Pruned top-k cosine graph with shrinkage; every result is explainable by its neighbour items. |
| Retrieval | **Implicit ALS** (`models/als.py`) | Matrix factorisation for implicit feedback (Hu, Koren & Volinsky 2008), written in NumPy so the closed-form updates are visible. Captures taste beyond direct co-occurrence. |
| Retrieval | **Content TF-IDF** (`models/content.py`) | Genres and user tags. The only retriever that can surface items nobody has interacted with yet. |
| Retrieval | **Popularity** | Baseline, plus cold-start fallback. |
| Ranking | **LightGBM LambdaRank** (`hybrid.py`) | Optimises NDCG directly and learns how much to trust each retriever for each kind of user and item, instead of hand-tuned blend weights. |
| Serving | **FastAPI** (`service.py`) | Ranked lists with reasons, a cold-start path for new users, and similar-item lookups. |

## Results

Synthetic data at MovieLens scale (610 users, 2,000 items, ~100k ratings), with a
temporal per-user holdout. Reproduce with `make eval-synthetic`, or see
[`reports/results_synthetic.md`](reports/results_synthetic.md).

| Model | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
|---|---|---|---|---|
| Popularity | 0.053 | 0.054 | 0.366 | 0.023 |
| Content TF-IDF | 0.030 | 0.033 | 0.246 | 0.599 |
| Item-kNN | 0.087 | 0.095 | 0.523 | 0.055 |
| Implicit ALS | 0.096 | 0.104 | 0.551 | 0.279 |
| Equal-weight blend | 0.101 | 0.109 | 0.573 | 0.086 |
| **Two-stage hybrid (LightGBM)** | **0.113** | **0.121** | **0.597** | **0.139** |

The learned ranker beats the best single retriever by **+17% NDCG@10** and the naive
blend by **+11%**, and recommends from 1.6× more of the catalogue than the blend.

Real MovieLens results: run `make data && make eval` (see below). The report is written
to `reports/results_ml-latest-small.md`.

## Quickstart

```bash
pip install -e ".[dev]"
make test              # 30 tests, ~5s, no download needed
make eval-synthetic    # full offline evaluation on synthetic data

make data              # MovieLens ml-latest-small (research licence, never committed)
make eval              # offline evaluation on real data
make train             # fit on all ratings -> artifacts/model.joblib
make serve             # http://localhost:8000/docs
```

If `make data` fails on a TLS error, download `ml-latest-small.zip` from
[grouplens.org](https://grouplens.org/datasets/movielens/latest/) and run
`python scripts/download_movielens.py --zip path/to/ml-latest-small.zip`.

## API

```bash
curl "localhost:8000/recommendations/42?k=3"
```

```json
{
  "user_id": 42,
  "strategy": "hybrid_ranker",
  "items": [
    {
      "movie_id": 592,
      "title": "...",
      "genres": "Adventure|War|Western",
      "reasons": [
        "Viewers with similar taste liked it",
        "Matches your overall taste profile",
        "Because you liked <title A> and <title B>"
      ]
    }
  ],
  "latency_ms": 36.3
}
```

| Endpoint | Purpose |
|---|---|
| `GET /recommendations/{user_id}?k=` | Ranked, explained recommendations. Unknown users fall back to cold start. |
| `GET /recommendations?genres=Comedy&genres=Romance` | Brand-new user steered by genres picked at sign-up. |
| `GET /items/{movie_id}/similar` | Item-to-item similarity ("more like this"). |
| `GET /health` | Readiness and catalogue size. |

## Design decisions

**Temporal split, not random.** Each user's most recent 20% of ratings is the test
set. A random split lets the model see the future, which inflates every metric and hides
the bias toward popular items.

**The ranker never trains on labels its retrievers saw.** The training window is split
again in time: retrievers fit on the earlier part, and the ranker learns to predict the
later part. Retrievers are then refit on the full window for serving.

**Scale-free features, a bug I hit and fixed.** The first version fed raw retriever
scores to the ranker, and the hybrid scored *below* the simple blend. The cause was
train/serve skew. Item-kNN scores grow with history length and popularity counts grow
with traffic, so the thresholds the ranker learned on the smaller training window were
wrong at serving time. Replacing raw scores with per-user z-scores, ranks and popularity
percentiles fixed it. A test (`test_features_are_scale_free`) keeps it fixed.

**Coverage is reported next to accuracy.** Popularity scores respectably on accuracy
while recommending 2% of the catalogue. Optimising for accuracy alone pushes a system
toward the same hits for everyone.

**Explanations come from the models themselves.** Each recommendation lists which
retrievers proposed it, and the item-kNN graph names the specific titles in the user's
history that drove it.

## Limitations and next steps

- ALS hyperparameters were picked from a small grid on synthetic data. For real data,
  tune them on a validation window before trusting the comparison.
- Retrieval is brute-force scoring, which is fine at 10k items. At catalogue scale,
  swap in an approximate-nearest-neighbour index (FAISS or ScaNN) over the ALS
  embeddings.
- Features are computed at request time. In production they'd come from a feature store
  with point-in-time correctness.
- Offline metrics only. The natural next step is shadow scoring against a live baseline,
  then an A/B test on engagement.

## Project layout

```
src/hybrec/
  data.py          loading, temporal split, synthetic MovieLens-format generator
  models/          popularity, itemknn, als, content (each a few dozen lines)
  hybrid.py        candidate union, features, LambdaRank training and serving
  metrics.py       Recall / NDCG / HitRate / coverage
  evaluate.py      offline evaluation CLI
  train.py         fit and save a servable artifact
  service.py       FastAPI app
tests/             30 tests: data, metrics, models, hybrid, API
```

## License

MIT. MovieLens data is © GroupLens and is not included or redistributed.
