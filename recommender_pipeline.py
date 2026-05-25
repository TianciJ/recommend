from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TRAIN_RATINGS_PATH = BASE_DIR / "train_data" / "ratings.dat"
MOVIES_PATH = BASE_DIR / "data" / "movies.dat"


class RecommenderPipeline:
    def __init__(self):
        # 模型只在初始化时加载一次，避免每次请求重复加载权重
        self.rough_ranker = build_rough_ranker()
        self.fine_ranker = build_fine_ranker()
        self.reranker = Reranker()

    def recall(self, user_id, recall_size=300):
        # 召回阶段：双塔召回 300 部候选电影
        candidates = two_tower_recall(user_id, recall_size)
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
    ):
        recalled_items = self.recall(user_id, recall_size)

        rough_ranked_items = self.rough_rank(
            user_id=user_id,
            candidates=recalled_items,
            rough_rank_size=rough_rank_size,
        )

        fine_ranked_items = self.fine_rank(
            user_id=user_id,
            candidates=rough_ranked_items,
            fine_rank_size=fine_rank_size,
        )

        final_items = self.rerank(
            user_id=user_id,
            ranked_items=fine_ranked_items,
            top_k=top_k,
        )

        return final_items


class Reranker:
    def __init__(self):
        self.user_seen_movies = load_user_seen_movies()
        self.movie_genres = load_movie_genres()

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


def load_user_seen_movies(ratings_path=TRAIN_RATINGS_PATH):
    user_seen_movies = {}

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            user_id = int(user_id)
            movie_id = int(movie_id)

            if user_id not in user_seen_movies:
                user_seen_movies[user_id] = set()

            user_seen_movies[user_id].add(movie_id)

    return user_seen_movies


def load_movie_genres(movies_path=MOVIES_PATH):
    movie_genres = {}

    with movies_path.open("r", encoding="latin-1") as movies_file:
        for line in movies_file:
            movie_id, title, genres = line.strip().split("::")
            movie_genres[int(movie_id)] = genres.split("|")

    return movie_genres


def build_rough_ranker():
    # 延迟导入，避免没有 torch 的环境在导入 pipeline 时直接报错
    from rough_rank.rough_rank_inference import RoughRanker

    return RoughRanker()


def build_fine_ranker():
    # 延迟导入，避免没有 torch 的环境在导入 pipeline 时直接报错
    from fine_rank.mmoe_inference import MMoEFineRanker

    return MMoEFineRanker()


def two_tower_recall(user_id, recall_size):
    # 延迟导入，避免没有 torch 的环境在导入 pipeline 时直接报错
    from recall.two_tower import recommend_for_user

    recalled_movies = recommend_for_user(user_id=user_id, top_k=recall_size)
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
    recommendations = pipeline.recommend(user_id=1, top_k=10, recall_size=300)

    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. movie_id={item['movie_id']} "
            f"title={item.get('title', '')} "
            f"genre={item.get('rerank_primary_genre', '')} "
            f"recall_score={item['recall_score']:.4f} "
            f"rough_rank_score={item['rough_rank_score']:.4f} "
            f"fine_rank_score={item['fine_rank_score']:.4f}"
        )


if __name__ == "__main__":
    main()
