"""Train on all available ratings and save a servable artifact.

python -m hybrec.train --data data/ml-latest-small --out artifacts/model.joblib
python -m hybrec.train --synthetic --out artifacts/model.joblib
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib

from hybrec.data import build_index, load_movielens, synthetic_catalog
from hybrec.hybrid import HybridRecommender


def train(catalog) -> HybridRecommender:
    index = build_index(catalog)
    return HybridRecommender().fit(catalog.ratings, catalog, index)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path)
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("artifacts/model.joblib"))
    args = p.parse_args()

    catalog = synthetic_catalog() if args.synthetic else load_movielens(args.data)
    t0 = time.perf_counter()
    model = train(catalog)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out, compress=3)
    print(f"Trained in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
