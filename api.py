import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel


class Movie(BaseModel):
    movie_id: int
    title: str


class Recommendations(BaseModel):
    user_id: int
    recommendations: list[Movie]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("MOVIE_MODEL_PATH", "artifacts/recommender.joblib"))
    artifact = joblib.load(model_path)
    app.state.recommender = artifact["recommender"]
    app.state.titles = artifact["titles"]
    yield


app = FastAPI(title="Movie Recommender", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/recommendations")
def recommendations(
    request: Request,
    user_id: Annotated[int, Query(ge=1)],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> Recommendations:
    model = request.app.state.recommender
    if user_id not in model.user_index:
        raise HTTPException(status_code=404, detail="No training history for this user")

    titles = request.app.state.titles
    ranked = model.recommend(user_id, k).tolist()
    return Recommendations(
        user_id=user_id,
        recommendations=[Movie(movie_id=movie_id, title=titles[movie_id]) for movie_id in ranked],
    )
