FROM python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

FROM base AS train
COPY download_data.py evaluate.py movielens.py recommender.py ./
RUN python download_data.py \
    && python evaluate.py --model-output artifacts/recommender.joblib

FROM base AS serve
COPY api.py recommender.py ./
COPY --from=train /app/artifacts/recommender.joblib artifacts/recommender.joblib

USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
