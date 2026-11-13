# Offline evaluation: ml-latest-small

Temporal per-user holdout (last 20% of each user's ratings). Evaluated on 591 users, catalogue of 9742 items. Full fit (both stages) took 8.3s.

| Model | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
|---|---|---|---|---|
| itemknn | 0.1165 | 0.1044 | 0.4416 | 0.0297 |
| als | 0.1317 | 0.1135 | 0.4958 | 0.0637 |
| content | 0.0063 | 0.0048 | 0.0338 | 0.1845 |
| popularity | 0.0770 | 0.0782 | 0.3164 | 0.0093 |
| blend | 0.1270 | 0.1153 | 0.4856 | 0.0277 |
| hybrid (LightGBM) | 0.1270 | 0.1122 | 0.4805 | 0.0636 |

## Ranker feature importance (gain)

| Feature | Gain share |
|---|---|
| als_z | 28.6% |
| itemknn_z | 11.1% |
| als_rank | 9.9% |
| genre_affinity | 8.0% |
| content_z | 7.8% |
| user_log_activity | 7.4% |
| item_mean_rating | 6.3% |
| popularity_z | 6.2% |
| item_year | 4.5% |
| itemknn_rank | 3.9% |
| item_pop_pct | 3.8% |
| popularity_rank | 1.9% |
