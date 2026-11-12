import pytest

from hybrec.metrics import catalog_coverage, hit_rate_at_k, ndcg_at_k, recall_at_k


def test_perfect_ranking_scores_one():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert recall_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0


def test_recall_caps_denominator_at_k():
    assert recall_at_k([1, 2], {1, 2, 3, 4, 5}, 2) == 1.0


def test_ndcg_rewards_earlier_hits():
    assert ndcg_at_k([9, 1], {1}, 2) < ndcg_at_k([1, 9], {1}, 2)


def test_misses_and_empty():
    assert ndcg_at_k([7, 8], {1}, 2) == 0.0
    assert recall_at_k([1], set(), 1) == 0.0
    assert hit_rate_at_k([7, 1], {1}, 2) == 1.0


def test_coverage():
    assert catalog_coverage([[0, 1], [1, 2]], 10) == pytest.approx(0.3)
