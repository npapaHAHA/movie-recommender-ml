import unittest
from math import log2

import pandas as pd

from evaluate import ranking_metrics, split_ratings
from recommender import fit


def example_ratings():
    return pd.DataFrame(
        [(1, 1, 5), (1, 4, 2), (2, 1, 5), (2, 2, 4), (3, 3, 5), (3, 4, 4), (4, 1, 2)],
        columns=["user_id", "movie_id", "rating"],
    )


class RecommenderTest(unittest.TestCase):
    def setUp(self):
        self.model = fit(example_ratings())

    def test_personalizes_and_filters_every_previously_rated_movie(self):
        self.assertEqual(self.model.recommend(1, 10).tolist(), [2, 3])

    def test_popularity_fallback_without_positive_ratings(self):
        expected = self.model.recommend(4, 3, personalized=False)
        self.assertEqual(self.model.recommend(4, 3).tolist(), expected.tolist())
        self.assertEqual(expected.tolist(), [2, 3, 4])

    def test_temporal_split_keeps_cutoff_ties_in_test(self):
        ratings = pd.DataFrame({"timestamp": [1, 2, 3, 4, 5, 5]})
        cutoff, train, test = split_ratings(ratings)
        self.assertEqual(cutoff, 5)
        self.assertEqual(train["timestamp"].tolist(), [1, 2, 3, 4])
        self.assertEqual(test["timestamp"].tolist(), [5, 5])

    def test_recall_and_ndcg_for_one_hit_at_rank_two(self):
        recall, ndcg = ranking_metrics([9, 2], {2, 3}, k=2)
        self.assertEqual(recall, 0.5)
        self.assertAlmostEqual(ndcg, 1 / (log2(3) + 1))
