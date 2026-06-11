# 三塔粗排推理
# 对双塔召回的候选电影打粗排分数，保留 top_k
from pathlib import Path

import torch

from .model import ThreeTowerRoughRankModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "rough_rank"
MODEL_PATH = MODEL_DIR / "three_tower.pt"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model_path=MODEL_PATH, device=None):
    if device is None:
        device = get_device()
    return torch.load(model_path, map_location=device, weights_only=False)


def build_model_from_checkpoint(checkpoint, device):
    # 从 checkpoint 的 feature_info 中读取 Embedding 大小，重建模型结构后载入权重
    fi = checkpoint["feature_info"]
    model = ThreeTowerRoughRankModel(
        user_count=len(fi["user_id_to_index"]),
        movie_count=len(fi["movie_id_to_index"]),
        gender_count=fi["gender_count"],
        age_count=fi["age_count"],
        occupation_count=fi["occupation_count"],
        genre_count=fi["genre_count"],
        dense_feature_dim=fi["dense_feature_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def build_dense_features(user_id, movie_id, rating_stats, recall_score=0.0):
    # 构建 Dense 特征向量（5 维）：
    #   [user_avg_rating/5, user_count_norm, movie_avg_rating/5, movie_count_norm, recall_score_norm]
    u_count = rating_stats["user_rating_count"].get(user_id, 0)
    m_count = rating_stats["movie_rating_count"].get(movie_id, 0)
    u_avg = rating_stats["user_rating_sum"][user_id] / u_count if u_count > 0 else 3
    m_avg = rating_stats["movie_rating_sum"][movie_id] / m_count if m_count > 0 else 3
    return [
        u_avg / 5,
        u_count / rating_stats["max_user_count"],
        m_avg / 5,
        m_count / rating_stats["max_movie_count"],
        (float(recall_score) + 1.0) / 2.0,  # 余弦相似度 [-1,1] -> [0,1]
    ]


class RoughRanker:
    """加载三塔粗排模型，对召回候选批量打分，返回按分数排序的 top_k。"""

    def __init__(self, model_path=MODEL_PATH):
        self.device = get_device()
        self.checkpoint = load_checkpoint(model_path=model_path, device=self.device)
        self.feature_info = self.checkpoint["feature_info"]
        self.model = build_model_from_checkpoint(self.checkpoint, self.device)

    def rank(self, user_id, recalled_items, top_k=100):
        fi = self.feature_info

        # 用户不在训练集则无法打分，直接返回空
        if user_id not in fi["user_id_to_index"]:
            return []

        # 过滤掉训练时未出现的电影（Embedding 层无法处理未知 id）
        valid_items = [
            item for item in recalled_items
            if item.get("movie_id", item.get("item_id")) in fi["movie_id_to_index"]
        ]
        if not valid_items:
            return []

        user_index = fi["user_id_to_index"][user_id]
        user_feature = fi["user_features"][user_id]

        # 为每个候选电影构建输入特征
        user_indexes, gender_indexes, age_indexes, occupation_indexes = [], [], [], []
        movie_indexes, genre_vectors, dense_features = [], [], []

        for item in valid_items:
            mid = item.get("movie_id", item.get("item_id"))
            user_indexes.append(user_index)
            gender_indexes.append(user_feature["gender_index"])
            age_indexes.append(user_feature["age_index"])
            occupation_indexes.append(user_feature["occupation_index"])
            movie_indexes.append(fi["movie_id_to_index"][mid])
            genre_vectors.append(fi["movie_features"][mid]["genre_vector"])
            dense_features.append(
                build_dense_features(
                    user_id=user_id,
                    movie_id=mid,
                    rating_stats=fi["rating_stats"],
                    recall_score=item.get("recall_score", 0.0),
                )
            )

        # 批量前向，一次推理所有候选
        with torch.no_grad():
            scores = self.model(
                torch.tensor(user_indexes, dtype=torch.long, device=self.device),
                torch.tensor(gender_indexes, dtype=torch.long, device=self.device),
                torch.tensor(age_indexes, dtype=torch.long, device=self.device),
                torch.tensor(occupation_indexes, dtype=torch.long, device=self.device),
                torch.tensor(movie_indexes, dtype=torch.long, device=self.device),
                torch.tensor(genre_vectors, dtype=torch.float, device=self.device),
                torch.tensor(dense_features, dtype=torch.float, device=self.device),
            ).cpu().tolist()

        # 把分数附加到原 item 上，排序后取 top_k
        ranked_items = [{**item, "rough_rank_score": score} for item, score in zip(valid_items, scores)]
        ranked_items.sort(key=lambda item: item["rough_rank_score"], reverse=True)
        return ranked_items[:top_k]


def rough_rank(user_id, recalled_items, top_k=100, model_path=MODEL_PATH):
    # 便捷函数：每次都重新加载模型，适合单次调用；批量请直接用 RoughRanker 类
    ranker = RoughRanker(model_path=model_path)
    return ranker.rank(user_id=user_id, recalled_items=recalled_items, top_k=top_k)
