from pathlib import Path
from time import perf_counter


BASE_DIR = Path(__file__).resolve().parent
TRAIN_RATINGS_PATH = BASE_DIR / "train_data" / "ratings.dat"
MOVIES_PATH = BASE_DIR / "data" / "movies.dat"


class RecommenderPipeline:
    def __init__(self, user_profile_repository=None, dataset_repository=None):
        self.recaller = build_recaller()
        self.rough_ranker = build_rough_ranker()
        self.fine_ranker = build_fine_ranker()
        self.user_profile_repository = user_profile_repository or build_user_profile_repository()
        self.dataset_repository = dataset_repository or build_dataset_repository()
        self.cold_start_recommender = build_cold_start_recommender(
            self.user_profile_repository,
            self.dataset_repository,
        )
        self.reranker = Reranker(dataset_repository=self.dataset_repository)

    def recall(self, user_id, recall_size=300):
        return two_tower_recall(self.recaller, user_id, recall_size)

    def rough_rank(self, user_id, candidates, rough_rank_size=100):
        return self.rough_ranker.rank(user_id=user_id, recalled_items=candidates, top_k=rough_rank_size)

    def fine_rank(self, user_id, candidates, fine_rank_size=50):
        return self.fine_ranker.rank(user_id=user_id, candidates=candidates, top_k=fine_rank_size)

    def rerank(self, user_id, ranked_items, top_k=20):
        return self.reranker.rerank(user_id=user_id, ranked_items=ranked_items, top_k=top_k)

    def recommend(self, user_id, top_k=20, recall_size=300, rough_rank_size=100, fine_rank_size=50, age=None, occupation=None):
        recalled_items = self.recall(user_id, recall_size)
        if not recalled_items:
            return self.cold_start(user_id=user_id, age=age, occupation=occupation, top_k=top_k)

        rough_ranked_items = self.rough_rank(user_id=user_id, candidates=recalled_items, rough_rank_size=rough_rank_size)
        if not rough_ranked_items:
            return self.cold_start(user_id=user_id, age=age, occupation=occupation, top_k=top_k)

        fine_ranked_items = self.fine_rank(user_id=user_id, candidates=rough_ranked_items, fine_rank_size=fine_rank_size)
        if not fine_ranked_items:
            return self.cold_start(user_id=user_id, age=age, occupation=occupation, top_k=top_k)

        return self.rerank(user_id=user_id, ranked_items=fine_ranked_items, top_k=top_k)

    def cold_start(self, user_id, age=None, occupation=None, top_k=20):
        age, occupation = self.resolve_cold_start_profile(user_id=user_id, age=age, occupation=occupation)
        return self.cold_start_recommender.recommend(user_id=user_id, age=age, occupation=occupation, top_k=top_k)

    def resolve_cold_start_profile(self, user_id, age=None, occupation=None):
        if age is not None and occupation is not None:
            return age, occupation

        if self.user_profile_repository is None:
            return age, occupation

        try:
            profile = self.user_profile_repository.get_user_profile(user_id)
        except Exception as error:
            print(f"MySQL user profile lookup failed; using cold-start fallback: {error}")
            return age, occupation

        if profile is None:
            return age, occupation

        return profile.get("age") if age is None else age, profile.get("occupation") if occupation is None else occupation

    def recommend_with_timing(self, user_id, top_k=20, recall_size=300, rough_rank_size=100, fine_rank_size=50, age=None, occupation=None):
        timing = {"stages": {}}
        total_start = perf_counter()

        stage_start = perf_counter()
        recalled_items = self.recall(user_id, recall_size)
        record_stage_timing(timing, "recall", stage_start, recalled_items)
        if not recalled_items:
            return self._timed_cold_start(timing, user_id, age, occupation, top_k), {**timing, "total_ms": elapsed_ms(total_start)}

        stage_start = perf_counter()
        rough_ranked_items = self.rough_rank(user_id=user_id, candidates=recalled_items, rough_rank_size=rough_rank_size)
        record_stage_timing(timing, "rough_rank", stage_start, rough_ranked_items)
        if not rough_ranked_items:
            return self._timed_cold_start(timing, user_id, age, occupation, top_k), {**timing, "total_ms": elapsed_ms(total_start)}

        stage_start = perf_counter()
        fine_ranked_items = self.fine_rank(user_id=user_id, candidates=rough_ranked_items, fine_rank_size=fine_rank_size)
        record_stage_timing(timing, "fine_rank", stage_start, fine_ranked_items)
        if not fine_ranked_items:
            return self._timed_cold_start(timing, user_id, age, occupation, top_k), {**timing, "total_ms": elapsed_ms(total_start)}

        stage_start = perf_counter()
        final_items = self.rerank(user_id=user_id, ranked_items=fine_ranked_items, top_k=top_k)
        record_stage_timing(timing, "rerank", stage_start, final_items)

        timing["total_ms"] = elapsed_ms(total_start)
        return final_items, timing

    def _timed_cold_start(self, timing, user_id, age, occupation, top_k):
        stage_start = perf_counter()
        items = self.cold_start(user_id=user_id, age=age, occupation=occupation, top_k=top_k)
        record_stage_timing(timing, "cold_start", stage_start, items)
        return items


class Reranker:
    def __init__(self, dataset_repository=None):
        ratings = movies = None
        if dataset_repository is not None:
            try:
                ratings = dataset_repository.list_ratings(split="train")
                movies = dataset_repository.list_movies()
            except Exception as error:
                print(f"MySQL rerank data loading failed; using dat fallback: {error}")

        self.user_seen_movies = load_user_seen_movies(ratings=ratings)
        self.movie_genres = load_movie_genres(movies=movies)

    def rerank(self, user_id, ranked_items, top_k=20):
        return self.diversify_by_genre(self.filter_seen_movies(user_id, ranked_items), top_k)

    def filter_seen_movies(self, user_id, ranked_items):
        seen = self.user_seen_movies.get(user_id, set())
        return [item for item in ranked_items if item.get("movie_id", item.get("item_id")) not in seen]

    def diversify_by_genre(self, ranked_items, top_k):
        selected, remaining, last_genre = [], list(ranked_items), None

        while remaining and len(selected) < top_k:
            idx = self.find_next_different_genre(remaining, last_genre)
            if idx is None:
                idx = 0
            item = remaining.pop(idx)
            last_genre = self.get_primary_genre(item)
            selected.append({**item, "rerank_primary_genre": last_genre})

        return selected

    def find_next_different_genre(self, items, last_genre):
        if last_genre is None:
            return 0
        for idx, item in enumerate(items):
            if self.get_primary_genre(item) != last_genre:
                return idx
        return None

    def get_primary_genre(self, item):
        genres = self.movie_genres.get(item.get("movie_id", item.get("item_id")), [])
        return genres[0] if genres else "Unknown"


def load_user_seen_movies(ratings_path=TRAIN_RATINGS_PATH, ratings=None):
    user_seen_movies = {}

    if ratings is not None:
        for r in ratings:
            user_seen_movies.setdefault(int(r["user_id"]), set()).add(int(r["movie_id"]))
        return user_seen_movies

    with ratings_path.open("r", encoding="utf-8") as f:
        for line in f:
            user_id, movie_id, *_ = line.strip().split("::")
            user_seen_movies.setdefault(int(user_id), set()).add(int(movie_id))

    return user_seen_movies


def load_movie_genres(movies_path=MOVIES_PATH, movies=None):
    if movies is not None:
        return {int(m["movie_id"]): list(m["genres"]) for m in movies}

    movie_genres = {}
    with movies_path.open("r", encoding="latin-1") as f:
        for line in f:
            movie_id, _, genres = line.strip().split("::")
            movie_genres[int(movie_id)] = genres.split("|")
    return movie_genres


def elapsed_ms(start_time):
    return (perf_counter() - start_time) * 1000


def record_stage_timing(timing, stage_name, start_time, items):
    timing["stages"][stage_name] = {
        "elapsed_ms": elapsed_ms(start_time),
        "item_count": len(items),
    }


def format_score(score):
    return f"{score:.4f}" if score is not None else "-"


def format_recommendation_line(rank, item):
    movie_id = item.get("movie_id", item.get("item_id", ""))
    parts = [
        f"{rank}. movie_id={movie_id}",
        f"title={item.get('title', '')}",
        f"genre={item.get('rerank_primary_genre', '')}",
        f"recall_score={format_score(item.get('recall_score'))}",
    ]
    if "cold_start_score" in item:
        parts.append(f"cold_start_score={format_score(item.get('cold_start_score'))}")
    parts += [
        f"rough_rank_score={format_score(item.get('rough_rank_score'))}",
        f"fine_rank_score={format_score(item.get('fine_rank_score'))}",
    ]
    return " ".join(parts)


def build_recaller():
    from recall.two_tower import TwoTowerRecaller
    return TwoTowerRecaller()


def build_rough_ranker():
    from rough_rank.inference import RoughRanker
    return RoughRanker()


def build_fine_ranker():
    from fine_rank.inference import MMoEFineRanker
    return MMoEFineRanker()


def build_cold_start_recommender(user_profile_repository=None, dataset_repository=None):
    from cold_start import ColdStartRecommender

    if user_profile_repository is None:
        return ColdStartRecommender()

    try:
        ratings = movies = None
        if dataset_repository is not None:
            ratings = dataset_repository.list_ratings(split="train")
            movies = dataset_repository.list_movies()
        return ColdStartRecommender(
            user_profiles=user_profile_repository.list_user_profiles(),
            ratings=ratings,
            movies=movies,
        )
    except Exception as error:
        print(f"MySQL cold-start profile loading failed; using dat fallback: {error}")
        return ColdStartRecommender()


def build_user_profile_repository():
    from database.mysql_client import get_mysql_config_from_env
    if get_mysql_config_from_env() is None:
        return None
    from database import UserProfileRepository
    return UserProfileRepository()


def build_dataset_repository():
    from database.mysql_client import get_mysql_config_from_env
    if get_mysql_config_from_env() is None:
        return None
    from database import MysqlDatasetRepository
    return MysqlDatasetRepository()


def two_tower_recall(recaller, user_id, recall_size):
    return [
        {
            "item_id": m["movie_id"],
            "movie_id": m["movie_id"],
            "title": m.get("title", ""),
            "recall_score": m["score"],
            "recall_source": "two_tower",
        }
        for m in recaller.recommend(user_id=user_id, top_k=recall_size)
    ]


def recommend_for_user_id_or_register(user_id, user_profile_repository, pipeline, input_func=input, output_func=print, top_k=20):
    if user_profile_repository is None:
        output_func("MySQL user repository is not configured. Running recommendation fallback.")
        return pipeline.recommend(user_id=user_id, top_k=top_k)

    profile = user_profile_repository.get_user_profile(user_id)

    if profile is not None:
        output_func(f"user_id={user_id} found. Running full recommendation flow.")
        return pipeline.recommend(user_id=user_id, top_k=top_k)

    output_func(f"user_id={user_id} not found. Registering a new user profile.")
    username = input_func("username: ").strip()
    age = int(input_func("age: ").strip())
    occupation = int(input_func("occupation: ").strip())

    registered_user_id = user_profile_repository.create_user(username=username, age=age, occupation=occupation)
    output_func(f"Registered user_id={registered_user_id}. Running cold start.")

    return pipeline.cold_start(user_id=registered_user_id, age=age, occupation=occupation, top_k=top_k)


def main():
    user_profile_repository = build_user_profile_repository()
    pipeline = RecommenderPipeline(user_profile_repository=user_profile_repository)
    user_id = int(input("user_id: ").strip())
    recommendations = recommend_for_user_id_or_register(
        user_id=user_id,
        user_profile_repository=user_profile_repository,
        pipeline=pipeline,
        top_k=20,
    )
    for rank, item in enumerate(recommendations, start=1):
        print(format_recommendation_line(rank, item))


if __name__ == "__main__":
    main()
