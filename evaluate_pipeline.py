import argparse
import math
from pathlib import Path
from time import perf_counter

from recommender_pipeline import RecommenderPipeline


BASE_DIR = Path(__file__).resolve().parent
TEST_RATINGS_PATH = BASE_DIR / "test_data" / "ratings.dat"


def parse_k_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_test_liked_movies(ratings_path=TEST_RATINGS_PATH, min_rating=4):
    test_liked_movies = {}

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            user_id = int(user_id)
            movie_id = int(movie_id)
            rating = int(rating)

            if rating < min_rating:
                continue

            if user_id not in test_liked_movies:
                test_liked_movies[user_id] = set()

            test_liked_movies[user_id].add(movie_id)

    return test_liked_movies


def get_movie_id(item):
    if isinstance(item, dict):
        return item.get("movie_id", item.get("item_id"))

    return item


def extract_movie_ids(recommendations):
    movie_ids = []

    for item in recommendations:
        movie_id = get_movie_id(item)

        if movie_id is not None:
            movie_ids.append(movie_id)

    return movie_ids


def calculate_ranking_metrics(recommendations, liked_movies, k_list):
    movie_ids = extract_movie_ids(recommendations)
    metrics = {}

    for k in k_list:
        top_k_movie_ids = movie_ids[:k]
        hit_count = len(set(top_k_movie_ids) & liked_movies)

        if liked_movies:
            recall = hit_count / len(liked_movies)
        else:
            recall = 0

        metrics[k] = {
            "precision": hit_count / k if k > 0 else 0,
            "recall": recall,
            "hit_rate": 1 if hit_count > 0 else 0,
            "mrr": calculate_mrr(top_k_movie_ids, liked_movies),
            "ndcg": calculate_ndcg(top_k_movie_ids, liked_movies, k),
        }

    return metrics


def calculate_mrr(movie_ids, liked_movies):
    for rank, movie_id in enumerate(movie_ids, start=1):
        if movie_id in liked_movies:
            return 1 / rank

    return 0


def calculate_ndcg(movie_ids, liked_movies, k):
    dcg = 0

    for rank, movie_id in enumerate(movie_ids[:k], start=1):
        if movie_id in liked_movies:
            dcg += 1 / math.log2(rank + 1)

    ideal_hit_count = min(len(liked_movies), k)
    ideal_dcg = 0

    for rank in range(1, ideal_hit_count + 1):
        ideal_dcg += 1 / math.log2(rank + 1)

    if ideal_dcg == 0:
        return 0

    return dcg / ideal_dcg


def build_empty_metric_sums(k_list):
    metric_names = ["precision", "recall", "hit_rate", "mrr", "ndcg"]

    return {
        k: {
            **{metric_name: 0 for metric_name in metric_names},
            "user_count": 0,
        }
        for k in k_list
    }


def add_user_metrics(metric_sums, user_metrics):
    for k, metrics in user_metrics.items():
        for metric_name, value in metrics.items():
            metric_sums[k][metric_name] += value

        metric_sums[k]["user_count"] += 1


def average_metric_sums(metric_sums):
    averaged_metrics = {}

    for k, sums in metric_sums.items():
        user_count = sums["user_count"]

        if user_count == 0:
            averaged_metrics[k] = {
                "precision": 0,
                "recall": 0,
                "hit_rate": 0,
                "mrr": 0,
                "ndcg": 0,
                "user_count": 0,
            }
            continue

        averaged_metrics[k] = {
            "precision": sums["precision"] / user_count,
            "recall": sums["recall"] / user_count,
            "hit_rate": sums["hit_rate"] / user_count,
            "mrr": sums["mrr"] / user_count,
            "ndcg": sums["ndcg"] / user_count,
            "user_count": user_count,
        }

    return averaged_metrics


def add_timing_summary(timing_sums, timing):
    timing_sums["total_ms"] += timing["total_ms"]
    timing_sums["request_count"] += 1

    for stage_name, stage_timing in timing["stages"].items():
        if stage_name not in timing_sums["stages"]:
            timing_sums["stages"][stage_name] = {
                "elapsed_ms": 0,
                "item_count": 0,
            }

        timing_sums["stages"][stage_name]["elapsed_ms"] += stage_timing["elapsed_ms"]
        timing_sums["stages"][stage_name]["item_count"] += stage_timing["item_count"]


def average_timing_sums(timing_sums):
    request_count = timing_sums["request_count"]

    if request_count == 0:
        return None

    return {
        "avg_total_ms": timing_sums["total_ms"] / request_count,
        "request_count": request_count,
        "stages": {
            stage_name: {
                "avg_elapsed_ms": stage_sums["elapsed_ms"] / request_count,
                "avg_item_count": stage_sums["item_count"] / request_count,
            }
            for stage_name, stage_sums in timing_sums["stages"].items()
        },
    }


def elapsed_ms(start_time):
    return (perf_counter() - start_time) * 1000


def build_command_timing_summary(
    pipeline_init_ms,
    evaluation_wall_ms,
    output_print_ms,
    command_total_ms,
    evaluated_users,
):
    avg_command_ms_per_user = command_total_ms / evaluated_users if evaluated_users > 0 else 0

    return {
        "pipeline_init_ms": pipeline_init_ms,
        "evaluation_wall_ms": evaluation_wall_ms,
        "output_print_ms": output_print_ms,
        "command_total_ms": command_total_ms,
        "evaluated_users": evaluated_users,
        "avg_command_ms_per_user": avg_command_ms_per_user,
    }


def evaluate_pipeline(
    pipeline,
    k_list,
    max_users=None,
    recall_size=300,
    rough_rank_size=100,
    fine_rank_size=50,
    include_timing=False,
):
    test_liked_movies = load_test_liked_movies()
    max_k = max(k_list)
    metric_sums = build_empty_metric_sums(k_list)
    timing_sums = {
        "total_ms": 0,
        "request_count": 0,
        "stages": {},
    }
    evaluated_users = 0
    skipped_users = 0

    for user_id, liked_movies in test_liked_movies.items():
        if max_users is not None and evaluated_users >= max_users:
            break

        if include_timing:
            recommendations, timing = pipeline.recommend_with_timing(
                user_id=user_id,
                top_k=max_k,
                recall_size=recall_size,
                rough_rank_size=rough_rank_size,
                fine_rank_size=fine_rank_size,
            )
            add_timing_summary(timing_sums, timing)
        else:
            recommendations = pipeline.recommend(
                user_id=user_id,
                top_k=max_k,
                recall_size=recall_size,
                rough_rank_size=rough_rank_size,
                fine_rank_size=fine_rank_size,
            )

        if not recommendations:
            skipped_users += 1
            continue

        user_metrics = calculate_ranking_metrics(
            recommendations=recommendations,
            liked_movies=liked_movies,
            k_list=k_list,
        )
        add_user_metrics(metric_sums, user_metrics)
        evaluated_users += 1

    return {
        "metrics": average_metric_sums(metric_sums),
        "evaluated_users": evaluated_users,
        "skipped_users": skipped_users,
        "timing": average_timing_sums(timing_sums),
    }


def print_metric_results(results):
    print("\nEnd-to-end pipeline evaluation")
    print(
        f"Evaluated users={results['evaluated_users']} "
        f"Skipped users={results['skipped_users']}"
    )

    for k in sorted(results["metrics"]):
        metrics = results["metrics"][k]
        print(
            f"K={k} "
            f"Precision@{k}={metrics['precision']:.4f} "
            f"Recall@{k}={metrics['recall']:.4f} "
            f"HitRate@{k}={metrics['hit_rate']:.4f} "
            f"MRR@{k}={metrics['mrr']:.4f} "
            f"NDCG@{k}={metrics['ndcg']:.4f} "
            f"Users={metrics['user_count']}"
        )


def print_timing_results(timing):
    if timing is None:
        return

    print("\nLatency timing")
    print(
        f"AvgTotal={timing['avg_total_ms']:.2f}ms "
        f"Requests={timing['request_count']}"
    )

    for stage_name, stage_timing in timing["stages"].items():
        print(
            f"{stage_name}: "
            f"AvgElapsed={stage_timing['avg_elapsed_ms']:.2f}ms "
            f"AvgItems={stage_timing['avg_item_count']:.1f}"
        )


def print_command_timing_results(command_timing):
    print("\nCommand timing")
    print(
        f"PipelineInit={command_timing['pipeline_init_ms']:.2f}ms "
        f"EvaluationWall={command_timing['evaluation_wall_ms']:.2f}ms "
        f"OutputPrint={command_timing['output_print_ms']:.2f}ms"
    )
    print(
        f"CommandTotal={command_timing['command_total_ms']:.2f}ms "
        f"AvgCommandPerUser={command_timing['avg_command_ms_per_user']:.2f}ms "
        f"EvaluatedUsers={command_timing['evaluated_users']}"
    )


def main():
    command_start = perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate the full recommendation pipeline.")
    parser.add_argument("--ks", default="10,20", help="Comma-separated K values.")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--recall-size", type=int, default=300)
    parser.add_argument("--rough-rank-size", type=int, default=100)
    parser.add_argument("--fine-rank-size", type=int, default=50)
    parser.add_argument("--with-timing", action="store_true")
    args = parser.parse_args()

    k_list = parse_k_list(args.ks)
    pipeline_init_start = perf_counter()
    pipeline = RecommenderPipeline()
    pipeline_init_ms = elapsed_ms(pipeline_init_start)

    evaluation_start = perf_counter()
    results = evaluate_pipeline(
        pipeline=pipeline,
        k_list=k_list,
        max_users=args.max_users,
        recall_size=args.recall_size,
        rough_rank_size=args.rough_rank_size,
        fine_rank_size=args.fine_rank_size,
        include_timing=args.with_timing,
    )
    evaluation_wall_ms = elapsed_ms(evaluation_start)

    output_print_start = perf_counter()
    print_metric_results(results)
    print_timing_results(results["timing"])
    output_print_ms = elapsed_ms(output_print_start)

    command_timing = build_command_timing_summary(
        pipeline_init_ms=pipeline_init_ms,
        evaluation_wall_ms=evaluation_wall_ms,
        output_print_ms=output_print_ms,
        command_total_ms=elapsed_ms(command_start),
        evaluated_users=results["evaluated_users"],
    )
    print_command_timing_results(command_timing)


if __name__ == "__main__":
    main()
