class RecommenderPipeline:
    def __init__(self):
        # 这里先加载粗排模型，避免每次请求都重复加载权重
        self.rough_ranker = build_rough_ranker()
        self.fine_rank_model = None
        self.rerank_model = None

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


def build_rough_ranker():
    # 延迟导入，避免没有 torch 的环境在导入 pipeline 时直接报错
    from rough_rank.rough_rank_inference import RoughRanker

    return RoughRanker()


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
