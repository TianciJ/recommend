import argparse
import math
import re
from pathlib import Path

import torch

from .two_tower import build_model_from_checkpoint
from utils import get_device


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_RATINGS_PATH = BASE_DIR / "train_data" / "ratings.dat"
TEST_RATINGS_PATH = BASE_DIR / "test_data" / "ratings.dat"
MODEL_DIR = BASE_DIR / "models" / "recall"
FINAL_MODEL_PATH = MODEL_DIR / "two_tower.pt"


def load_checkpoint(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)


def find_model_paths(model_dir=MODEL_DIR):
    paths = list(model_dir.glob("two_tower_epoch_*.pt"))
    if FINAL_MODEL_PATH.exists():
        paths.append(FINAL_MODEL_PATH)
    return sorted(paths, key=model_sort_key)


def model_sort_key(p):
    m = re.search(r"two_tower_epoch_(\d+)\.pt$", p.name)
    return (0, int(m.group(1))) if m else (1, p.name)


def _load_user_movie_set(split, ratings_path, min_rating=None):
    mysql_dataset = load_mysql_dataset_if_configured(split=split)
    ratings = mysql_dataset["ratings"] if mysql_dataset else load_ratings_from_dat(ratings_path)
    result = {}
    for r in ratings:
        if min_rating is None or int(r["rating"]) >= min_rating:
            result.setdefault(int(r["user_id"]), set()).add(int(r["movie_id"]))
    return result


def load_user_seen_movies(ratings_path=TRAIN_RATINGS_PATH):
    return _load_user_movie_set("train", ratings_path)


def load_test_liked_movies(ratings_path=TEST_RATINGS_PATH):
    return _load_user_movie_set("test", ratings_path, min_rating=4)


def load_ratings_from_dat(ratings_path):
    with ratings_path.open("r", encoding="utf-8") as f:
        return [
            {"user_id": int(a), "movie_id": int(b), "rating": int(c), "timestamp": int(d)}
            for a, b, c, d in (line.strip().split("::") for line in f)
        ]


def load_mysql_dataset_if_configured(split="train"):
    from database.dataset_repository import load_mysql_dataset
    return load_mysql_dataset(split=split)


def build_movie_tensors(feature_info, device):
    fi = feature_info
    movie_count = len(fi["index_to_movie_id"])
    movie_tensor = torch.arange(movie_count, dtype=torch.long, device=device)
    genre_tensor = torch.tensor(
        [fi["movie_features"][fi["index_to_movie_id"][i]]["genre_vector"] for i in range(movie_count)],
        dtype=torch.float, device=device,
    )
    return movie_tensor, genre_tensor


def recommend_for_eval(model, feature_info, user_id, user_seen_movies, movie_tensor, genre_tensor, top_k, device):
    fi = feature_info
    if user_id not in fi["user_id_to_index"]:
        return []

    uf = fi["user_features"][user_id]
    movie_count = len(fi["index_to_movie_id"])
    repeat = lambda val, dtype: torch.tensor([val] * movie_count, dtype=dtype, device=device)
    max_count = fi.get("max_rating_count", 1) or 1
    user_behavior = torch.tensor(
        [[uf["avg_rating"] / 5.0, math.log1p(uf["rating_count"]) / math.log1p(max_count)]] * movie_count,
        dtype=torch.float, device=device,
    )

    with torch.no_grad():
        scores = model(
            repeat(fi["user_id_to_index"][user_id], torch.long),
            repeat(uf["gender_index"], torch.long),
            repeat(uf["age_index"], torch.long),
            repeat(uf["occupation_index"], torch.long),
            movie_tensor,
            genre_tensor,
            user_behavior,
        )
        for seen_id in user_seen_movies.get(user_id, set()):
            if seen_id in fi["movie_id_to_index"]:
                scores[fi["movie_id_to_index"][seen_id]] = -float("inf")

        _, top_idxs = torch.topk(scores, min(top_k, movie_count))

    return [fi["index_to_movie_id"][int(i)] for i in top_idxs.cpu()]


def evaluate_model(model_path, k_list, user_seen_movies, test_liked_movies, max_users):
    device = get_device()
    checkpoint = load_checkpoint(model_path, device)
    feature_info = checkpoint["feature_info"]
    model = build_model_from_checkpoint(checkpoint, device)
    movie_tensor, genre_tensor = build_movie_tensors(feature_info, device)

    metrics = {k: {"precision_sum": 0, "recall_sum": 0, "hit_sum": 0, "user_count": 0} for k in k_list}
    evaluated_users = 0

    for user_id, liked_movies in test_liked_movies.items():
        if user_id not in feature_info["user_id_to_index"]:
            continue
        if max_users is not None and evaluated_users >= max_users:
            break

        recs = recommend_for_eval(
            model=model, feature_info=feature_info, user_id=user_id,
            user_seen_movies=user_seen_movies, movie_tensor=movie_tensor,
            genre_tensor=genre_tensor, top_k=max(k_list), device=device,
        )
        if not recs:
            continue

        evaluated_users += 1
        for k in k_list:
            hits = len(set(recs[:k]) & liked_movies)
            metrics[k]["precision_sum"] += hits / k
            metrics[k]["recall_sum"] += hits / len(liked_movies)
            metrics[k]["hit_sum"] += 1 if hits > 0 else 0
            metrics[k]["user_count"] += 1

    return {
        k: (
            {"precision": 0, "recall": 0, "hit_rate": 0, "user_count": 0}
            if metrics[k]["user_count"] == 0
            else {
                "precision": metrics[k]["precision_sum"] / metrics[k]["user_count"],
                "recall": metrics[k]["recall_sum"] / metrics[k]["user_count"],
                "hit_rate": metrics[k]["hit_sum"] / metrics[k]["user_count"],
                "user_count": metrics[k]["user_count"],
            }
        )
        for k in k_list
    }


def print_results(model_name, results):
    print(f"\n模型: {model_name}")
    for k in sorted(results):
        r = results[k]
        print(f"K={k} Precision@{k}={r['precision']:.4f} Recall@{k}={r['recall']:.4f} HitRate@{k}={r['hit_rate']:.4f} Users={r['user_count']}")


def parse_k_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", default="10,20,100,300")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--select-metric", choices=["precision", "recall", "hit_rate"], default="recall")
    parser.add_argument("--select-k", type=int, default=300)
    args = parser.parse_args()

    k_list = parse_k_list(args.ks)
    if args.select_k not in k_list:
        k_list = sorted(k_list + [args.select_k])

    model_paths = find_model_paths()
    if not model_paths:
        print(f"没有在 {MODEL_DIR} 找到模型文件")
        return

    user_seen_movies = load_user_seen_movies()
    test_liked_movies = load_test_liked_movies()

    all_results = []
    for model_path in model_paths:
        results = evaluate_model(model_path, k_list, user_seen_movies, test_liked_movies, args.max_users)
        print_results(model_path.name, results)
        all_results.append({"model_path": model_path, "results": results, "score": results[args.select_k][args.select_metric]})

    best = max(all_results, key=lambda x: x["score"])
    print(f"\n最佳模型: {best['model_path'].name} ({args.select_metric}@{args.select_k}={best['score']:.4f})")


if __name__ == "__main__":
    main()
