from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


@dataclass
class MovieRecommender:
    movie_ids: np.ndarray
    user_index: dict[int, int]
    watched: dict[int, set[int]]
    profiles: csr_matrix
    similarities: csr_matrix
    popularity: np.ndarray

    def recommend(self, user_id: int, k: int, *, personalized: bool = True) -> np.ndarray:
        profile = self.profiles[self.user_index[user_id]]
        scores = (profile @ self.similarities).toarray().ravel() if personalized else self.popularity
        available = ~np.isin(self.movie_ids, list(self.watched[user_id]))
        ranked = np.lexsort((self.movie_ids, -self.popularity, -scores))
        return self.movie_ids[ranked[available[ranked]][:k]]


def fit(train) -> MovieRecommender:
    movie_ids = np.sort(train["movie_id"].unique())
    movie_index = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    user_ids = np.sort(train["user_id"].unique())
    user_index = {user_id: i for i, user_id in enumerate(user_ids)}
    positive = train[train["rating"] >= 4]
    profiles = csr_matrix(
        (np.ones(len(positive), dtype=np.float32),
         (positive["user_id"].map(user_index), positive["movie_id"].map(movie_index))),
        shape=(len(user_ids), len(movie_ids)),
    )
    item_profiles = normalize(profiles.T.tocsr())
    return MovieRecommender(
        movie_ids=movie_ids,
        user_index=user_index,
        watched=train.groupby("user_id")["movie_id"].agg(set).to_dict(),
        profiles=profiles,
        similarities=item_profiles @ item_profiles.T,
        popularity=np.asarray(profiles.sum(axis=0)).ravel(),
    )
