"""Offline evaluation: every retriever alone, the naive blend, and the full two-stage model.

python -m hybrec.evaluate --data data/ml-latest-small
python -m hybrec.evaluate --synthetic
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hybrec.data import (
    build_index,
    load_movielens,
    relevant_items,
    synthetic_catalog,
    temporal_split,
)
from hybrec.hybrid import HybridRecommender
from hybrec.metrics import catalog_coverage, hit_rate_at_k, ndcg_at_k, recall_at_k


def evaluate(
    catalog, k: int = 10, max_users: int | None = None, seed: int = 0
) -> tuple[pd.DataFrame, HybridRecommender]:
    index = build_index(catalog)
    train, test = temporal_split(catalog.ratings)
    t0 = time.perf_counter()
    model = HybridRecommender().fit(train, catalog, index)
    fit_seconds = time.perf_counter() - t0

    truth = relevant_items(test, index)
    users = np.array(sorted(truth))
    if max_users and len(users) > max_users:
        users = np.random.default_rng(seed).choice(users, max_users, replace=False)

    systems = {m.name: m for m in model.stage.sources}
    results = {name: [] for name in [*systems, "blend", "hybrid (LightGBM)"]}
    for u in users:
        for name, m in systems.items():
            results[name].append(m.recommend(u, k, exclude=model.stage.X_seen).tolist())
        results["blend"].append(model.recommend(u, k, mode="blend"))
        results["hybrid (LightGBM)"].append(model.recommend(u, k, mode="ranker"))

    rows = []
    for name, recs in results.items():
        rel = [truth[u] for u in users]
        rows.append(
            {
                "model": name,
                f"Recall@{k}": np.mean(
                    [recall_at_k(r, t, k) for r, t in zip(recs, rel, strict=True)]
                ),
                f"NDCG@{k}": np.mean([ndcg_at_k(r, t, k) for r, t in zip(recs, rel, strict=True)]),
                f"HitRate@{k}": np.mean(
                    [hit_rate_at_k(r, t, k) for r, t in zip(recs, rel, strict=True)]
                ),
                "Coverage": catalog_coverage(recs, index.n_items),
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    table.attrs.update(
        users=len(users), fit_seconds=fit_seconds, n_items=index.n_items, n_train=len(train)
    )
    return table, model


def to_markdown(table: pd.DataFrame, model: HybridRecommender, dataset: str) -> str:
    lines = [
        f"# Offline evaluation: {dataset}",
        "",
        f"Temporal per-user holdout (last 20% of each user's ratings). "
        f"Evaluated on {table.attrs['users']} users, catalogue of {table.attrs['n_items']} items. "
        f"Full fit (both stages) took {table.attrs['fit_seconds']:.1f}s.",
        "",
        "| Model | " + " | ".join(table.columns) + " |",
        "|---|" + "---|" * len(table.columns),
    ]
    for name, row in table.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    lines += [
        "",
        "## Ranker feature importance (gain)",
        "",
        "| Feature | Gain share |",
        "|---|---|",
    ]
    imp = model.feature_importance()
    for f, v in (imp / imp.sum()).head(12).items():
        lines.append(f"| {f} | {v:.1%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="Folder with MovieLens ratings/movies/tags CSVs")
    src.add_argument("--synthetic", action="store_true", help="Use the built-in synthetic dataset")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--max-users", type=int, default=None, help="Subsample users for a faster run")
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()

    catalog = (
        synthetic_catalog(n_users=610, n_items=2000, ratings_per_user=(20, 300))
        if args.synthetic
        else load_movielens(args.data)
    )
    dataset = "synthetic" if args.synthetic else args.data.name
    table, model = evaluate(catalog, k=args.k, max_users=args.max_users)
    report = to_markdown(table, model, dataset)
    print(report)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"results_{dataset}.md").write_text(report)


if __name__ == "__main__":
    main()
