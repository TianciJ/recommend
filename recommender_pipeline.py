from pathlib import Path
from time import perf_counter


BASE_DIR = Path(__file__).resolve().parent
TRAIN_RATINGS_PATH = BASE_DIR / "train_data" / "ratings.dat"
MOVIES_PATH = BASE_DIR / "data" / "movies.dat"


class RecommenderPipeline:
    def __init__(self, user_profile_repository=None, dataset_repository=None):
        # 模型只在初始化时加载一次，避免每次请求重复加载权重
        self.recaller = build_recaller()
        self.rough_ranker = build_rough_ranker()
        self.fine_ranker = build_fine_ranker()
        self.user_profile_repository = (
            user_profile_repository
            if user_profile_repository is not None
            else build_user_profile_repository()
        )
        self.dataset_repository = (
            dataset_repository
            if dataset_repository is not None
            else build_dataset_repository()
        )
        self.cold_start_recommender = build_cold_start_recommender(
            self.user_profile_repository,
            self.dataset_repository,
        )
        self.reranker = Reranker(dataset_repository=self.dataset_repository)

    def recall(self, user_id, recall_size=300):
        # 召回阶段：双塔召回 300 部候选电影
        candidates = two_tower_recall(self.recaller, user_id, recall_size)
        return candidates

    def rough_rank(self, user_id, candidates, rough_rank_size=100):
        # 粗排阶段：三塔粗排模型打分，保留前 100 部
        rough_ranked_items = self.rough_ranker.rank(
            user_id=user_id,
            recalled_items=candidates,
            top_k=rough_rank_size,
        )
        return rough_ranked_items

    def fine_rank(self, user_id, candidates, fine_rank_size=50):
        # 精排阶段：MMoE epoch 6 模型打分，保留前 50 部
        fine_ranked_items = self.fine_ranker.rank(
            user_id=user_id,
            candidates=candidates,
            top_k=fine_rank_size,
        )
        return fine_ranked_items

    def rerank(self, user_id, ranked_items, top_k=20):
        # 重排阶段：过滤已看电影，并按电影主题/类型打散
        final_items = self.reranker.rerank(
            user_id=user_id,
            ranked_items=ranked_items,
            top_k=top_k,
        )
        return final_items

    def recommend(
        self,
        user_id,
        top_k=20,
        recall_size=300,
        rough_rank_size=100,
        fine_rank_size=50,
        age=None,
        occupation=None,
    ):
        recalled_items = self.recall(user_id, recall_size)

        if not recalled_items:
            return self.cold_start(
                user_id=user_id,
                age=age,
                occupation=occupation,
                top_k=top_k,
            )

        rough_ranked_items = self.rough_rank(
            user_id=user_id,
            candidates=recalled_items,
            rough_rank_size=rough_rank_size,
        )

        if not rough_ranked_items:
            return self.cold_start(
                user_id=user_id,
                age=age,
                occupation=occupation,
                top_k=top_k,
            )

        fine_ranked_items = self.fine_rank(
            user_id=user_id,
            candidates=rough_ranked_items,
            fine_rank_size=fine_rank_size,
        )

        if not fine_ranked_items:
            return self.cold_start(
                user_id=user_id,
                age=age,
                occupation=occupation,
                top_k=top_k,
            )

        final_items = self.rerank(
            user_id=user_id,
            ranked_items=fine_ranked_items,
            top_k=top_k,
        )

        return final_items

    def cold_start(self, user_id, age=None, occupation=None, top_k=20):
        age, occupation = self.resolve_cold_start_profile(
            user_id=user_id,
            age=age,
            occupation=occupation,
        )
        return self.cold_start_recommender.recommend(
            user_id=user_id,
            age=age,
            occupation=occupation,
            top_k=top_k,
        )

    def resolve_cold_start_profile(self, user_id, age=None, occupation=None):
        resolved_age = age
        resolved_occupation = occupation

        if resolved_age is not None and resolved_occupation is not None:
            return resolved_age, resolved_occupation

        if self.user_profile_repository is None:
            return resolved_age, resolved_occupation

        try:
            profile = self.user_profile_repository.get_user_profile(user_id)
        except Exception as error:
            print(f"MySQL user profile lookup failed; using cold-start fallback: {error}")
            return resolved_age, resolved_occupation

        if profile is None:
            return resolved_age, resolved_occupation

        if resolved_age is None:
            resolved_age = profile.get("age")

        if resolved_occupation is None:
            resolved_occupation = profile.get("occupation")

        return resolved_age, resolved_occupation

    def recommend_with_timing(
        self,
        user_id,
        top_k=20,
        recall_size=300,
        rough_rank_size=100,
        fine_rank_size=50,
        age=None,
        occupation=None,
    ):
        timing = {"stages": {}}
        total_start = perf_counter()

        stage_start = perf_counter()
        recalled_items = self.recall(user_id, recall_size)
        record_stage_timing(timing, "recall", stage_start, recalled_items)

        if not recalled_items:
            final_items = self.timed_cold_start(
                timing=timing,
                user_id=user_id,
                age=age,
                occupation=occupation,
                top_k=top_k,
            )
            timing["total_ms"] = elapsed_ms(total_start)
            return final_items, timing

        stage_start = perf_counter()
        rough_ranked_items = self.rough_rank(
            user_id=user_id,
            candidates=recalled_items,
            rough_rank_size=rough_rank_size,
        )
        record_stage_timing(timing, "rough_rank", stage_start, rough_ranked_items)

        if not rough_ranked_items:
            final_items = self.timed_cold_start(
                timing=timing,
                user_id=user_id,
                age=age,
                occupation=occupation,
                top_k=top_k,
            )
            timing["total_ms"] = elapsed_ms(total_start)
            return final_items, timing

        stage_start = perf_counter()
        fine_ranked_items = self.fine_rank(
            user_id=user_id,
            candidates=rough_ranked_items,
            fine_rank_size=fine_rank_size,
        )
        record_stage_timing(timing, "fine_rank", stage_start, fine_ranked_items)

        if not fine_ranked_items:
            final_items = self.timed_cold_start(
                timing=timing,
                user_id=user_id,
                age=age,
                occupation=occupation,
                top_k=top_k,
            )
            timing["total_ms"] = elapsed_ms(total_start)
            return final_items, timing

        stage_start = perf_counter()
        final_items = self.rerank(
            user_id=user_id,
            ranked_items=fine_ranked_items,
            top_k=top_k,
        )
        record_stage_timing(timing, "rerank", stage_start, final_items)

        timing["total_ms"] = elapsed_ms(total_start)
        return final_items, timing

    def timed_cold_start(self, timing, user_id, age=None, occupation=None, top_k=20):
        stage_start = perf_counter()
        final_items = self.cold_start(
            user_id=user_id,
            age=age,
            occupation=occupation,
            top_k=top_k,
        )
        record_stage_timing(timing, "cold_start", stage_start, final_items)
        return final_items


class Reranker:
    def __init__(self, dataset_repository=None):
        ratings = None
        movies = None

        if dataset_repository is not None:
            try:
                ratings = dataset_repository.list_ratings(split="train")
                movies = dataset_repository.list_movies()
            except Exception as error:
                print(f"MySQL rerank data loading failed; using dat fallback: {error}")

        self.user_seen_movies = load_user_seen_movies(ratings=ratings)
        self.movie_genres = load_movie_genres(movies=movies)

    def rerank(self, user_id, ranked_items, top_k=20):
        unseen_items = self.filter_seen_movies(user_id, ranked_items)
        diversified_items = self.diversify_by_genre(unseen_items, top_k)
        return diversified_items

    def filter_seen_movies(self, user_id, ranked_items):
        seen_movies = self.user_seen_movies.get(user_id, set())
        filtered_items = []

        for item in ranked_items:
            movie_id = item.get("movie_id", item.get("item_id"))

            if movie_id not in seen_movies:
                filtered_items.append(item)

        return filtered_items

    def diversify_by_genre(self, ranked_items, top_k):
        selected_items = []
        remaining_items = list(ranked_items)
        last_genre = None

        while remaining_items and len(selected_items) < top_k:
            chosen_index = self.find_next_different_genre(remaining_items, last_genre)

            if chosen_index is None:
                chosen_index = 0

            chosen_item = remaining_items.pop(chosen_index)
            chosen_genre = self.get_primary_genre(chosen_item)

            selected_items.append(
                {
                    **chosen_item,
                    "rerank_primary_genre": chosen_genre,
                }
            )
            last_genre = chosen_genre

        return selected_items

    def find_next_different_genre(self, items, last_genre):
        if last_genre is None:
            return 0

        for index, item in enumerate(items):
            current_genre = self.get_primary_genre(item)

            if current_genre != last_genre:
                return index

        return None

    def get_primary_genre(self, item):
        movie_id = item.get("movie_id", item.get("item_id"))
        genres = self.movie_genres.get(movie_id, [])

        if not genres:
            return "Unknown"

        return genres[0]


def load_user_seen_movies(ratings_path=TRAIN_RATINGS_PATH, ratings=None):
    user_seen_movies = {}

    if ratings is not None:
        for rating in ratings:
            user_id = int(rating["user_id"])
            movie_id = int(rating["movie_id"])
            user_seen_movies.setdefault(user_id, set()).add(movie_id)
        return user_seen_movies

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            user_id = int(user_id)
            movie_id = int(movie_id)

            if user_id not in user_seen_movies:
                user_seen_movies[user_id] = set()

            user_seen_movies[user_id].add(movie_id)

    return user_seen_movies


def load_movie_genres(movies_path=MOVIES_PATH, movies=None):
    movie_genres = {}

    if movies is not None:
        for movie in movies:
            movie_genres[int(movie["movie_id"])] = list(movie["genres"])
        return movie_genres

    with movies_path.open("r", encoding="latin-1") as movies_file:
        for line in movies_file:
            movie_id, title, genres = line.strip().split("::")
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
    if score is None:
        return "-"

    return f"{score:.4f}"


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

    parts.extend(
        [
            f"rough_rank_score={format_score(item.get('rough_rank_score'))}",
            f"fine_rank_score={format_score(item.get('fine_rank_score'))}",
        ]
    )

    return " ".join(parts)


def build_recaller():
    from recall.two_tower import TwoTowerRecaller

    return TwoTowerRecaller()


def build_rough_ranker():
    # 延迟导入，避免没有 torch 的环境在导入 pipeline 时直接报错
    from rough_rank.rough_rank_inference import RoughRanker

    return RoughRanker()


def build_fine_ranker():
    # 延迟导入，避免没有 torch 的环境在导入 pipeline 时直接报错
    from fine_rank.mmoe_inference import MMoEFineRanker

    return MMoEFineRanker()


def build_cold_start_recommender(user_profile_repository=None, dataset_repository=None):
    from cold_start import ColdStartRecommender

    if user_profile_repository is None:
        return ColdStartRecommender()

    try:
        ratings = None
        movies = None

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
    recalled_movies = recaller.recommend(user_id=user_id, top_k=recall_size)
    candidates = []

    for movie in recalled_movies:
        candidates.append(
            {
                "item_id": movie["movie_id"],
                "movie_id": movie["movie_id"],
                "title": movie.get("title", ""),
                "recall_score": movie["score"],
                "recall_source": "two_tower",
            }
        )

    return candidates


def main():
    pipeline = RecommenderPipeline()
    recommendations = pipeline.recommend(user_id=15857, top_k=20, recall_size=300)

    for rank, item in enumerate(recommendations, start=1):
        print(format_recommendation_line(rank, item))


if __name__ == "__main__":
    main()
