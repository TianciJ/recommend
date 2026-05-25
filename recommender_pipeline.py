class RecommenderPipeline:
    def __init__(self):
        # 后面可以在这里加载粗排、精排、重排模型
        self.rough_rank_model = None
        self.fine_rank_model = None
        self.rerank_model = None

    def recall(self, user_id, recall_size=300):
        # 双塔模型
        candidates = two_tower_recall(user_id, recall_size)
        return candidates

    def rough_rank(self, user_id, candidates, rough_rank_size=100):
        # 粗排阶段：暂时用占位逻辑
        rough_ranked_items = fake_rough_rank(user_id, candidates, rough_rank_size)
        return rough_ranked_items

    def fine_rank(self, user_id, candidates, fine_rank_size=50):
        # 精排阶段：暂时用占位逻辑
        fine_ranked_items = fake_fine_rank(user_id, candidates, fine_rank_size)
        return fine_ranked_items

    def rerank(self, user_id, ranked_items, top_k=20):
        # 重排阶段：暂时用占位逻辑
        final_items = fake_rerank(user_id, ranked_items, top_k)
        return final_items

    def recommend(
        self,
        user_id,
        top_k=20,
        recall_size=300,
        rough_rank_size=100,
        fine_rank_size=50,
    ):
        # 1. 召回：从全量电影中取出候选集
        recalled_items = self.recall(user_id, recall_size)

        # 2. 粗排：从召回候选里保留一部分
        rough_ranked_items = self.rough_rank(
            user_id=user_id,
            candidates=recalled_items,
            rough_rank_size=rough_rank_size,
        )

        # 3. 精排：对粗排结果做更细的排序
        fine_ranked_items = self.fine_rank(
            user_id=user_id,
            candidates=rough_ranked_items,
            fine_rank_size=fine_rank_size,
        )

        # 4. 重排：做最终过滤、打散、多样性控制
        final_items = self.rerank(
            user_id=user_id,
            ranked_items=fine_ranked_items,
            top_k=top_k,
        )

        return final_items


def two_tower_recall(user_id, recall_size):
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


def fake_rough_rank(user_id, candidates, rough_rank_size):
    # 临时占位：目前直接沿用召回分数做粗排分
    ranked_items = []

    for item in candidates:
        rough_score = item["recall_score"]
        ranked_items.append(
            {
                **item,
                "rough_rank_score": rough_score,
            }
        )

    ranked_items.sort(key=lambda item: item["rough_rank_score"], reverse=True)
    return ranked_items[:rough_rank_size]


def fake_fine_rank(user_id, candidates, fine_rank_size):
    # 临时占位：目前直接沿用粗排分数做精排分
    ranked_items = []

    for item in candidates:
        fine_score = item["rough_rank_score"]
        ranked_items.append(
            {
                **item,
                "fine_rank_score": fine_score,
            }
        )

    ranked_items.sort(key=lambda item: item["fine_rank_score"], reverse=True)
    return ranked_items[:fine_rank_size]


def fake_rerank(user_id, ranked_items, top_k):
    # 临时占位：目前只做截断
    final_items = ranked_items[:top_k]
    return final_items


def main():
    pipeline = RecommenderPipeline()
    recommendations = pipeline.recommend(user_id=1, top_k=10, recall_size=300)

    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. movie_id={item['movie_id']} "
            f"title={item.get('title', '')} "
            f"recall_score={item['recall_score']:.4f} "
            f"rough_rank_score={item['rough_rank_score']:.4f} "
            f"fine_rank_score={item['fine_rank_score']:.4f}"
        )


if __name__ == "__main__":
    main()
