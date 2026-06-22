# MMoE 精排推理
# 对粗排候选电影用 MMoE 模型打分，默认取 like 任务分数排序，返回 top_k
from pathlib import Path
import torch
from .model import MMoERanker, get_device

BASE_DIR = Path(__file__).resolve().parent.parent
# 默认加载 epoch 6 的精排模型（验证集表现最优）
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "fine_rank" / "mmoe_epoch_6.pt"


def load_checkpoint(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)


def build_model_from_checkpoint(checkpoint, device):
    # 从 feature_info 中读取各 Embedding 大小，重建模型结构后载入权重
    fi = checkpoint["feature_info"]
    model = MMoERanker(
        user_count=len(fi["user_id_to_index"]),
        movie_count=len(fi["movie_id_to_index"]),
        gender_count=fi["gender_count"],
        age_count=fi["age_count"],
        occupation_count=fi["occupation_count"],
        genre_count=fi["genre_count"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


class MMoEFineRanker:
    """加载 MMoE 精排模型，对候选电影批量打分，按指定任务分数返回 top_k。"""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, score_name="like"):
        self.device = get_device()
        self.checkpoint = load_checkpoint(model_path, self.device)
        self.feature_info = self.checkpoint["feature_info"]
        self.model = build_model_from_checkpoint(self.checkpoint, self.device)
        self.score_name = score_name  # 取哪个任务分数作为最终精排分（like/high_rating/rating）

    def rank(self, user_id, candidates, top_k=50):
        fi = self.feature_info

        # 用户不在训练集则无法打分
        if user_id not in fi["user_id_to_index"]:
            return []

        # 过滤未知电影
        valid_items = [
            item for item in candidates
            if item.get("movie_id", item.get("item_id")) in fi["movie_id_to_index"]
        ]
        if not valid_items:
            return []

        user_index = fi["user_id_to_index"][user_id]
        user_feature = fi["user_features"][user_id]

        # 为每个候选电影组装输入特征
        user_indexes, gender_indexes, age_indexes, occupation_indexes = [], [], [], []
        movie_indexes, genre_indexes, recall_scores, coarse_scores = [], [], [], []

        for item in valid_items:
            mid = item.get("movie_id", item.get("item_id"))
            user_indexes.append(user_index)
            gender_indexes.append(user_feature["gender"])
            age_indexes.append(user_feature["age"])
            occupation_indexes.append(user_feature["occupation"])
            movie_indexes.append(fi["movie_id_to_index"][mid])
            genre_indexes.append(fi["movie_features"][mid]["genres"])
            recall_scores.append(item.get("recall_score", 0.0))
            coarse_scores.append(item.get("rough_rank_score", 0.0))

        # 批量推理
        t = lambda vals, dtype: torch.tensor(vals, dtype=dtype, device=self.device)
        with torch.no_grad():
            outputs = self.model(
                user_id=t(user_indexes, torch.long),
                gender=t(gender_indexes, torch.long),
                age=t(age_indexes, torch.long),
                occupation=t(occupation_indexes, torch.long),
                movie_id=t(movie_indexes, torch.long),
                genres=t(genre_indexes, torch.long),
                recall_score=t(recall_scores, torch.float),
                coarse_score=t(coarse_scores, torch.float),
            )

            # 根据 score_name 选取对应任务的输出作为最终精排分
            if self.score_name == "like":
                scores = torch.sigmoid(outputs["like_logit"])
            elif self.score_name == "high_rating":
                scores = torch.sigmoid(outputs["high_rating_logit"])
            elif self.score_name == "rating":
                scores = outputs["rating_pred"]
            else:
                raise ValueError("score_name 须为 like、high_rating 或 rating 之一")

            scores = scores.cpu().tolist()

        # 把精排分附加到原 item 上，排序后取 top_k
        ranked_items = [
            {**item, "fine_rank_score": score, "fine_rank_source": f"mmoe_epoch_6_{self.score_name}"}
            for item, score in zip(valid_items, scores)
        ]
        ranked_items.sort(key=lambda item: item["fine_rank_score"], reverse=True)
        return ranked_items[:top_k]
