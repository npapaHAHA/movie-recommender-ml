from zipfile import ZipFile

import pandas as pd


def load_ratings(data_dir):
    with ZipFile(data_dir / "ml-1m.zip") as archive:
        with archive.open("ml-1m/ratings.dat") as source:
            return pd.read_csv(
                source, sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"],
            )


def load_titles(data_dir):
    with ZipFile(data_dir / "ml-1m.zip") as archive:
        with archive.open("ml-1m/movies.dat") as source:
            movies = pd.read_csv(
                source, sep="::", engine="python", encoding="latin-1",
                names=["movie_id", "title", "genres"],
            )
    return dict(zip(movies["movie_id"], movies["title"], strict=True))
