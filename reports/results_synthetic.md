# Offline evaluation: synthetic

Temporal per-user holdout (last 20% of each user's ratings). Evaluated on 606 users, catalogue of 2000 items. Full fit (both stages) took 7.1s.

| Model | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
|---|---|---|---|---|
| itemknn | 0.0870 | 0.0948 | 0.5231 | 0.0545 |
| als | 0.0961 | 0.1036 | 0.5512 | 0.2785 |
| content | 0.0302 | 0.0327 | 0.2459 | 0.5990 |
| popularity | 0.0532 | 0.0536 | 0.3663 | 0.0230 |
| blend | 0.1012 | 0.1091 | 0.5726 | 0.0860 |
| hybrid (LightGBM) | 0.1134 | 0.1208 | 0.5974 | 0.1390 |

## Ranker feature importance (gain)

| Feature | Gain share |
|---|---|
| itemknn_z | 15.4% |
| n_sources | 13.8% |
| als_z | 10.8% |
| genre_affinity | 10.6% |
| content_z | 7.6% |
| user_log_activity | 7.5% |
| itemknn_rank | 5.9% |
| item_mean_rating | 5.8% |
| popularity_z | 5.5% |
| item_pop_pct | 4.9% |
| item_year | 3.9% |
| als_rank | 3.7% |
