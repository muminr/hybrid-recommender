"""Top-K ranking metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Hits over min(k, |relevant|), so a perfect top-k list scores 1.0 even
    when the user has more than k relevant items."""
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / min(k, len(relevant))


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    gains = [1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return float(sum(gains) / ideal)


def hit_rate_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    return float(bool(set(recommended[:k]) & relevant))


def catalog_coverage(all_recommendations: Iterable[Sequence[int]], n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's list. Guards
    against a model that scores well by recommending the same hits to everyone."""
    seen: set[int] = set()
    for recs in all_recommendations:
        seen.update(recs)
    return len(seen) / n_items if n_items else 0.0
