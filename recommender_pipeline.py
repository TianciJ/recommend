class RecommenderPipeline:
    def __init__(self):
        # 后面可以在这里加载模型、索引、配置、特征表等
        self.recall_model = None
        self.rough_rank_model = None
        self.fine_rank_model = None
        self.rerank_model = None

    def recall(self, user_id, recall_size=200):
        # 召回阶段：从全量物品中快速找出一批候选物品
        # 后面可以接入：热门召回、平均分召回、双塔召回、ItemCF 等
        candidates = fake_recall(user_id, recall_size)
        return candidates

    def rough_rank(self, user_id, candidates, rough_rank_size=100):
        # 粗排阶段：对召回候选做初步打分，保留更可能相关的一部分
        # 后面可以接入：简单规则模型、LR、GBDT、轻量 MLP 等
        rough_ranked_items = fake_rough_rank(user_id, candidates, rough_rank_size)
        return rough_ranked_items

    def fine_rank(self, user_id, candidates, fine_rank_size=50):
        # 精排阶段：使用更复杂的特征和模型做更准确的排序
        # 后面可以接入：DIN、DCN、DeepFM、MMoE 等模型
        fine_ranked_items = fake_fine_rank(user_id, candidates, fine_rank_size)
        return fine_ranked_items

    def rerank(self, user_id, ranked_items, top_k=20):
        # 重排阶段：在精排结果上做业务规则和多样性控制
        # 后面可以接入：去重、过滤已看、多样性、MMR、类目打散等
        final_items = fake_rerank(user_id, ranked_items, top_k)
        return final_items

    def recommend(
        self,
        user_id,
        top_k=20,
        recall_size=100,
        rough_rank_size=100,
        fine_rank_size=50,
    ):
        # 1. 召回
        recalled_items = self.recall(user_id, recall_size)

        # 2. 粗排
        rough_ranked_items = self.rough_rank(
            user_id=user_id,
            candidates=recalled_items,
            rough_rank_size=rough_rank_size,
        )

        # 3. 精排
        fine_ranked_items = self.fine_rank(
            user_id=user_id,
            candidates=rough_ranked_items,
            fine_rank_size=fine_rank_size,
        )

        # 4. 重排
        final_items = self.rerank(
            user_id=user_id,
            ranked_items=fine_ranked_items,
            top_k=top_k,
        )

        return final_items


def fake_recall(user_id, recall_size):
    # 临时占位：模拟召回出一批 item_id
    candidates = []

    for item_index in range(1, recall_size + 1):
        candidates.append(
            {
                "item_id": item_index,
                "recall_score": 1 / item_index,
            }
        )

    return candidates


def fake_rough_rank(user_id, candidates, rough_rank_size):
    # 临时占位：模拟粗排打分
    ranked_items = []

    for item in candidates:
        rough_score = item["recall_score"] * 0.8
        ranked_items.append(
            {
                **item,
                "rough_rank_score": rough_score,
            }
        )

    ranked_items.sort(key=lambda item: item["rough_rank_score"], reverse=True)
    return ranked_items[:rough_rank_size]


def fake_fine_rank(user_id, candidates, fine_rank_size):
    # 临时占位：模拟精排打分
    ranked_items = []

    for item in candidates:
        fine_score = item["rough_rank_score"] * 0.9
        ranked_items.append(
            {
                **item,
                "fine_rank_score": fine_score,
            }
        )

    ranked_items.sort(key=lambda item: item["fine_rank_score"], reverse=True)
    return ranked_items[:fine_rank_size]


def fake_rerank(user_id, ranked_items, top_k):
    # 临时占位：模拟重排，目前只做截断
    final_items = ranked_items[:top_k]

    return final_items


def main():
    pipeline = RecommenderPipeline()
    recommendations = pipeline.recommend(user_id=1, top_k=10, recall_size=100)

    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. item_id={item['item_id']} "
            f"recall_score={item['recall_score']:.4f} "
            f"rough_rank_score={item['rough_rank_score']:.4f} "
            f"fine_rank_score={item['fine_rank_score']:.4f}"
        )


if __name__ == "__main__":
    main()
