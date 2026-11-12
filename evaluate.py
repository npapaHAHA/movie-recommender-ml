import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from movielens import load_ratings, load_titles
from recommender import fit


def split_ratings(ratings):
    cutoff = int(ratings["timestamp"].quantile(0.8))
    return cutoff, ratings[ratings["timestamp"] < cutoff], ratings[ratings["timestamp"] >= cutoff]


def ranking_metrics(recommended, relevant, k):
    hits = np.isin(recommended, list(relevant))
    discounts = 1 / np.log2(np.arange(2, len(recommended) + 2))
    ideal_discounts = 1 / np.log2(np.arange(2, min(k, len(relevant)) + 2))
    return float(hits.sum() / len(relevant)), float((hits * discounts).sum() / ideal_discounts.sum())


def evaluate(train, test, cutoff, model, k):
    catalog = set(model.movie_ids)
    relevant_by_user = test[test["rating"] >= 4].groupby("user_id")["movie_id"].agg(set)
    scores = {"popularity": [], "item_cosine": []}
    for user, future_movies in relevant_by_user.items():
        if user not in model.user_index:
            continue
        relevant = (future_movies - model.watched[user]) & catalog
        if not relevant:
            continue
        for name in scores:
            top = model.recommend(user, k, personalized=name == "item_cosine")
            scores[name].append(ranking_metrics(top, relevant, k))

    if not scores["popularity"]:
        raise ValueError("No users with relevant held-out movies from the training catalog")
    test_metrics = {}
    for name, results in scores.items():
        recall, ndcg = np.mean(results, axis=0)
        test_metrics[name] = {f"recall_at_{k}": float(recall), f"ndcg_at_{k}": float(ndcg)}
    return {
        "cutoff_utc": str(pd.to_datetime(cutoff, unit="s", utc=True)),
        "train_ratings": len(train),
        "test_ratings": len(test),
        "evaluated_users": len(scores["popularity"]),
        "catalog_size": len(catalog),
        "k": k,
        "test": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")

    cutoff, train, test = split_ratings(load_ratings(args.data_dir))
    model = fit(train)
    report = evaluate(train, test, cutoff, model, args.k)
    if args.user_id is not None:
        if args.user_id not in model.user_index:
            parser.error(f"user {args.user_id} has no ratings before the cutoff")
        report["example"] = {
            "user_id": args.user_id,
            "recommended_movie_ids": model.recommend(args.user_id, args.k).tolist(),
        }
    if args.model_output is not None:
        titles = load_titles(args.data_dir)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"recommender": model, "titles": titles}, args.model_output, compress=3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
