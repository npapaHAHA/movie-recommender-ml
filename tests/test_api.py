import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
from fastapi.testclient import TestClient

from api import app
from recommender import fit
from test_recommender import example_ratings


class ApiTest(unittest.TestCase):
    def setUp(self):
        data_dir = Path(self.enterContext(TemporaryDirectory()))
        artifact = data_dir / "recommender.joblib"
        joblib.dump(
            {"recommender": fit(example_ratings()), "titles": {i: f"Movie {i}" for i in range(1, 5)}},
            artifact,
        )
        self.enterContext(patch.dict(os.environ, {"MOVIE_MODEL_PATH": str(artifact)}))
        self.client = self.enterContext(TestClient(app))

    def test_health_after_loading_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_returns_ranked_movies_with_titles_and_limits_k(self):
        response = self.client.get("/recommendations", params={"user_id": 1, "k": 1})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {"user_id": 1, "recommendations": [{"movie_id": 2, "title": "Movie 2"}]},
        )

    def test_returns_not_found_for_unknown_user(self):
        response = self.client.get("/recommendations", params={"user_id": 999, "k": 3})
        self.assertEqual(response.status_code, 404, response.text)

    def test_rejects_invalid_parameters(self):
        for params in ({"k": 3}, {"user_id": 0}, {"user_id": 1, "k": 0}, {"user_id": 1, "k": 51}):
            with self.subTest(params=params):
                response = self.client.get("/recommendations", params=params)
                self.assertEqual(response.status_code, 422, response.text)
