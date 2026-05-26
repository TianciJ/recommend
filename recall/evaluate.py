import argparse
import re
from pathlib import Path

import torch

from .two_tower import build_model_from_checkpoint
from .two_tower import get_device


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_RATINGS_PATH = BASE_DIR / "train_data" / "ratings.dat"
TEST_RATINGS_PATH = BASE_DIR / "test_data" / "ratings.dat"
MODEL_DIR = BASE_DIR / "model_weights"
FINAL_MODEL_PATH = MODEL_DIR / "two_tower.pt"


def load_checkpoint(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)


def find_model_paths(model_dir=MODEL_DIR):
    model_paths = []

    for model_path in model_dir.glob("model_epoch_*.pt"):
        model_paths.append(model_path)

    if FINAL_MODEL_PATH.exists():
        model_paths.append(FINAL_MODEL_PATH)

    return sorted(model_paths, key=model_sort_key)


def model_sort_key(model_path):
    match = re.search(r"model_epoch_(\d+)\.pt$", model_path.name)

    if match:
        return (0, int(match.group(1)))

    return (1, model_path.name)


def load_user_seen_movies(ratings_path=TRAIN_RATINGS_PATH):
    mysql_dataset = load_mysql_dataset_if_configured(split="train")

    if mysql_dataset is not None:
        return build_user_seen_movies(mysql_dataset["ratings"])

    return build_user_seen_movies(load_ratings_from_dat(ratings_path))


def build_user_seen_movies(ratings):
    user_seen_movies = {}

    for rating in ratings:
        user_id = int(rating["user_id"])
        movie_id = int(rating["movie_id"])

        if user_id not in user_seen_movies:
            user_seen_movies[user_id] = set()

        user_seen_movies[user_id].add(movie_id)

    return user_seen_movies


def load_test_liked_movies(ratings_path=TEST_RATINGS_PATH):
    mysql_dataset = load_mysql_dataset_if_configured(split="test")

    if mysql_dataset is not None:
        return build_test_liked_movies(mysql_dataset["ratings"])

    return build_test_liked_movies(load_ratings_from_dat(ratings_path))


def build_test_liked_movies(ratings):
    test_liked_movies = {}

    for rating_row in ratings:
        user_id = int(rating_row["user_id"])
        movie_id = int(rating_row["movie_id"])
        rating = int(rating_row["rating"])

        if rating < 4:
            continue

        if user_id not in test_liked_movies:
            test_liked_movies[user_id] = set()

        test_liked_movies[user_id].add(movie_id)

    return test_liked_movies


def load_ratings_from_dat(ratings_path):
    ratings = []

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            ratings.append(
                {
                    "user_id": int(user_id),
                    "movie_id": int(movie_id),
                    "rating": int(rating),
                    "timestamp": int(timestamp),
                }
            )

    return ratings


def load_mysql_dataset_if_configured(split="train"):
    from database.dataset_repository import load_mysql_dataset

    return load_mysql_dataset(split=split)


def build_movie_tensors(feature_info, device):
    index_to_movie_id = feature_info["index_to_movie_id"]
    movie_features = feature_info["movie_features"]
    movie_count = len(index_to_movie_id)

    movie_tensor = torch.arange(movie_count, dtype=torch.long, device=device)

    genre_vectors = []
    for movie_index in range(movie_count):
        movie_id = index_to_movie_id[movie_index]
        genre_vectors.append(movie_features[movie_id]["genre_vector"])

    genre_tensor = torch.tensor(genre_vectors, dtype=torch.float, device=device)

    return movie_tensor, genre_tensor


def recommend_for_eval(
    model,
    feature_info,
    user_id,
    user_seen_movies,
    movie_tensor,
    genre_tensor,
    top_k,
    device,
):
    user_id_to_index = feature_info["user_id_to_index"]
    index_to_movie_id = feature_info["index_to_movie_id"]
    movie_id_to_index = feature_info["movie_id_to_index"]
    user_features = feature_info["user_features"]
    movie_count = len(index_to_movie_id)

    if user_id not in user_id_to_index:
        return []

    user_index = user_id_to_index[user_id]
    user_feature = user_features[user_id]

    with torch.no_grad():
        user_tensor = torch.tensor(
            [user_index] * movie_count, dtype=torch.long, device=device
        )
        gender_tensor = torch.tensor(
            [user_feature["gender_index"]] * movie_count,
            dtype=torch.long,
            device=device,
        )
        age_tensor = torch.tensor(
            [user_feature["age_index"]] * movie_count,
            dtype=torch.long,
            device=device,
        )
        occupation_tensor = torch.tensor(
            [user_feature["occupation_index"]] * movie_count,
            dtype=torch.long,
            device=device,
        )

        scores = model(
            user_tensor,
            gender_tensor,
            age_tensor,
            occupation_tensor,
            movie_tensor,
            genre_tensor,
        )

        # 过滤训练集中已经评分过的电影，避免推荐旧电影
        for seen_movie_id in user_seen_movies.get(user_id, set()):
            if seen_movie_id in movie_id_to_index:
                seen_movie_index = movie_id_to_index[seen_movie_id]
                scores[seen_movie_index] = -float("inf")

        top_k = min(top_k, movie_count)
        top_scores, top_movie_indexes = torch.topk(scores, top_k)
        top_movie_indexes = top_movie_indexes.cpu()

    recommendations = []
    for movie_index in top_movie_indexes:
        movie_id = index_to_movie_id[int(movie_index)]
        recommendations.append(movie_id)

    return recommendations


def evaluate_model(model_path, k_list, user_seen_movies, test_liked_movies, max_users):
    device = get_device()
    checkpoint = load_checkpoint(model_path, device)
    feature_info = checkpoint["feature_info"]
    model = build_model_from_checkpoint(checkpoint, device)
    movie_tensor, genre_tensor = build_movie_tensors(feature_info, device)

    metrics = {}
    for k in k_list:
        metrics[k] = {
            "precision_sum": 0,
            "recall_sum": 0,
            "hit_sum": 0,
            "user_count": 0,
        }

    evaluated_users = 0
    for user_id, liked_movies in test_liked_movies.items():
        if user_id not in feature_info["user_id_to_index"]:
            continue

        if max_users is not None and evaluated_users >= max_users:
            break

        max_k = max(k_list)
        recommendations = recommend_for_eval(
            model=model,
            feature_info=feature_info,
            user_id=user_id,
            user_seen_movies=user_seen_movies,
            movie_tensor=movie_tensor,
            genre_tensor=genre_tensor,
            top_k=max_k,
            device=device,
        )

        if not recommendations:
            continue

        evaluated_users += 1

        for k in k_list:
            top_k_recommendations = recommendations[:k]
            hit_count = len(set(top_k_recommendations) & liked_movies)

            metrics[k]["precision_sum"] += hit_count / k
            metrics[k]["recall_sum"] += hit_count / len(liked_movies)

            if hit_count > 0:
                metrics[k]["hit_sum"] += 1

            metrics[k]["user_count"] += 1

    results = {}
    for k in k_list:
        user_count = metrics[k]["user_count"]

        if user_count == 0:
            results[k] = {
                "precision": 0,
                "recall": 0,
                "hit_rate": 0,
                "user_count": 0,
            }
            continue

        results[k] = {
            "precision": metrics[k]["precision_sum"] / user_count,
            "recall": metrics[k]["recall_sum"] / user_count,
            "hit_rate": metrics[k]["hit_sum"] / user_count,
            "user_count": user_count,
        }

    return results


def print_results(model_name, results):
    print(f"\n模型: {model_name}")

    for k in sorted(results):
        result = results[k]
        print(
            f"K={k} "
            f"Precision@{k}={result['precision']:.4f} "
            f"Recall@{k}={result['recall']:.4f} "
            f"HitRate@{k}={result['hit_rate']:.4f} "
            f"Users={result['user_count']}"
        )


def parse_k_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def get_score(results, metric, k):
    return results[k][metric]


def main():
    parser = argparse.ArgumentParser(description="评估所有双塔召回模型")
    parser.add_argument("--ks", default="10,20,100,300", help="例如：10,20,100,300")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument(
        "--select-metric",
        choices=["precision", "recall", "hit_rate"],
        default="recall",
    )
    parser.add_argument("--select-k", type=int, default=300)
    args = parser.parse_args()

    k_list = parse_k_list(args.ks)
    if args.select_k not in k_list:
        k_list.append(args.select_k)
        k_list = sorted(k_list)

    model_paths = find_model_paths()
    if not model_paths:
        print(f"没有在 {MODEL_DIR} 找到模型文件")
        return

    user_seen_movies = load_user_seen_movies()
    test_liked_movies = load_test_liked_movies()

    all_results = []
    for model_path in model_paths:
        results = evaluate_model(
            model_path=model_path,
            k_list=k_list,
            user_seen_movies=user_seen_movies,
            test_liked_movies=test_liked_movies,
            max_users=args.max_users,
        )
        print_results(model_path.name, results)
        all_results.append(
            {
                "model_path": model_path,
                "results": results,
                "score": get_score(results, args.select_metric, args.select_k),
            }
        )

    best_result = max(all_results, key=lambda item: item["score"])
    best_model_name = best_result["model_path"].name
    best_score = best_result["score"]

    print(
        f"\n最佳模型: {best_model_name} "
        f"({args.select_metric}@{args.select_k}={best_score:.4f})"
    )


if __name__ == "__main__":
    main()
