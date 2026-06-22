# 双塔召回模型
# 用户塔和物品塔各自输出 64 维向量，用余弦相似度作为召回分数
# 训练时以 rating >= 4 为正样本，rating <= 2 为负样本，rating == 3 跳过
import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ReLU, Sequential
from torch.utils.data import DataLoader, Dataset

from .movie_utils import add_movie_titles, print_recommendations
from utils import get_device

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "train_data"
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"
USERS_PATH = TRAIN_DIR / "users.dat"
MOVIES_PATH = TRAIN_DIR / "movies.dat"
MODEL_DIR = BASE_DIR / "models" / "recall"
MODEL_PATH = MODEL_DIR / "two_tower.pt"

logger = logging.getLogger(__name__)


# ---------- 训练数据集 ----------

class RatingDataset(Dataset):
    # 把样本列表包装成 PyTorch Dataset，供 DataLoader 使用
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "user_index": torch.tensor(s["user_index"], dtype=torch.long),
            "gender_index": torch.tensor(s["gender_index"], dtype=torch.long),
            "age_index": torch.tensor(s["age_index"], dtype=torch.long),
            "occupation_index": torch.tensor(s["occupation_index"], dtype=torch.long),
            "movie_index": torch.tensor(s["movie_index"], dtype=torch.long),
            "genre_vector": torch.tensor(s["genre_vector"], dtype=torch.float),
            "user_behavior": torch.tensor(s["user_behavior"], dtype=torch.float),
            "label": torch.tensor(s["label"], dtype=torch.float),
        }


# ---------- 模型结构 ----------

class TwoTowerModel(nn.Module):
    # 用户塔: user_id(32) + gender(4) + age(8) + occupation(8) + behavior(2) = 54 维
    # 物品塔: movie_id(32) + genre(16) = 48 维
    # 两塔各经过 MLP 压缩到 64 维，最终用余弦相似度打分
    def __init__(self, user_count, movie_count, gender_count, age_count, occupation_count, genre_count):
        super().__init__()
        # 用户侧 embedding
        self.user_embedding = Embedding(user_count, 32)
        self.gender_embedding = Embedding(gender_count, 4)
        self.age_embedding = Embedding(age_count, 8)
        self.occupation_embedding = Embedding(occupation_count, 8)
        self.user_tower = Sequential(Linear(54, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 64))

        # 物品侧 embedding
        self.movie_embedding = Embedding(movie_count, 32)
        self.genre_layer = Linear(genre_count, 16)  # genre one-hot -> 16 维稠密表示
        self.movie_tower = Sequential(Linear(48, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 64))

    def forward(self, user_index, gender_index, age_index, occupation_index, movie_index, genre_vector, user_behavior):
        # 拼接用户侧所有特征，过用户塔
        user_input = torch.cat([
            self.user_embedding(user_index),
            self.gender_embedding(gender_index),
            self.age_embedding(age_index),
            self.occupation_embedding(occupation_index),
            user_behavior,  # [avg_rating/5, log_count_norm]
        ], dim=1)
        # 拼接物品侧特征，过物品塔
        movie_input = torch.cat([
            self.movie_embedding(movie_index),
            self.genre_layer(genre_vector),
        ], dim=1)
        # 余弦相似度作为最终分数，范围 [-1, 1]
        return F.cosine_similarity(self.user_tower(user_input), self.movie_tower(movie_input), dim=1)


# ---------- 特征加载 ----------

def load_user_features(users_path=USERS_PATH, users=None, ratings=None):
    # 构建用户特征字典，同时统计行为特征（平均评分、评分数量）
    if users is None:
        users = load_users_from_dat(users_path)

    gender_to_index, age_to_index, occupation_to_index = {}, {}, {}
    user_features = {}

    for user in users:
        uid = int(user["user_id"])
        gender, age, occ = str(user["gender"]), str(user["age"]), str(user["occupation"])

        # 遇到新值时动态分配索引（label encoding）
        if gender not in gender_to_index:
            gender_to_index[gender] = len(gender_to_index)
        if age not in age_to_index:
            age_to_index[age] = len(age_to_index)
        if occ not in occupation_to_index:
            occupation_to_index[occ] = len(occupation_to_index)

        user_features[uid] = {
            "gender_index": gender_to_index[gender],
            "age_index": age_to_index[age],
            "occupation_index": occupation_to_index[occ],
            "avg_rating": 3.0,   # 默认均值，有 ratings 时会覆盖
            "rating_count": 0,
        }

    # 用 ratings 数据更新每个用户的行为统计
    if ratings is not None:
        rating_sum, rating_count = {}, {}
        for row in ratings:
            uid = int(row["user_id"])
            rating_sum[uid] = rating_sum.get(uid, 0) + int(row["rating"])
            rating_count[uid] = rating_count.get(uid, 0) + 1

        max_count = max(rating_count.values()) if rating_count else 1
        for uid, count in rating_count.items():
            if uid in user_features:
                user_features[uid]["avg_rating"] = rating_sum[uid] / count
                user_features[uid]["rating_count"] = count
        # max_rating_count 存到每条记录，推理时归一化用
        for uid in user_features:
            user_features[uid]["max_rating_count"] = max_count

    return user_features, gender_to_index, age_to_index, occupation_to_index


def load_users_from_dat(users_path=USERS_PATH):
    users = []
    with users_path.open("r", encoding="utf-8") as f:
        for line in f:
            user_id, gender, age, occupation, zip_code = line.strip().split("::")
            users.append({"user_id": int(user_id), "gender": gender, "age": int(age), "occupation": int(occupation), "zip_code": zip_code})
    return users


def load_movie_features(movies_path=MOVIES_PATH, movies=None):
    # 构建电影 genre one-hot 特征：每部电影对应一个 genre_count 维的 0/1 向量
    if movies is None:
        movies = load_movies_from_dat(movies_path)

    genre_to_index = {}
    movie_genres = {}
    for movie in movies:
        mid = int(movie["movie_id"])
        genre_list = list(movie["genres"])
        for g in genre_list:
            if g not in genre_to_index:
                genre_to_index[g] = len(genre_to_index)
        movie_genres[mid] = genre_list

    genre_count = len(genre_to_index)
    movie_features = {}
    for mid, genre_list in movie_genres.items():
        vec = [0] * genre_count
        for g in genre_list:
            vec[genre_to_index[g]] = 1
        movie_features[mid] = {"genre_vector": vec}

    return movie_features, genre_to_index


def load_movies_from_dat(movies_path=MOVIES_PATH):
    movies = []
    with movies_path.open("r", encoding="latin-1") as f:
        for line in f:
            movie_id, title, genres = line.strip().split("::")
            movies.append({"movie_id": int(movie_id), "title": title, "genres": genres.split("|")})
    return movies


def load_ratings_from_dat(ratings_path=TRAIN_RATINGS_PATH):
    ratings = []
    with ratings_path.open("r", encoding="utf-8") as f:
        for line in f:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            ratings.append({"user_id": int(user_id), "movie_id": int(movie_id), "rating": int(rating), "timestamp": int(timestamp)})
    return ratings


def load_train_samples(ratings_path=TRAIN_RATINGS_PATH, ratings=None, users=None, movies=None):
    # 优先从 MySQL 读数据，无配置时回退到 .dat 文件
    if ratings is None and users is None and movies is None:
        mysql_dataset = load_mysql_dataset_if_configured(split="train")
        if mysql_dataset is not None:
            ratings, users, movies = mysql_dataset["ratings"], mysql_dataset["users"], mysql_dataset["movies"]

    if ratings is None:
        ratings = load_ratings_from_dat(ratings_path)

    user_features, gender_to_index, age_to_index, occupation_to_index = load_user_features(users=users, ratings=ratings)
    movie_features, genre_to_index = load_movie_features(movies=movies)
    max_rating_count = next(iter(user_features.values()), {}).get("max_rating_count", 1) or 1

    samples = []
    user_id_to_index, movie_id_to_index, index_to_movie_id = {}, {}, {}

    for row in ratings:
        uid, mid, rating = int(row["user_id"]), int(row["movie_id"]), int(row["rating"])
        if rating == 3:
            continue  # 跳过中立评分，避免引入噪声

        # 动态建立 id -> 连续整数索引的映射（Embedding 层需要连续索引）
        if uid not in user_id_to_index:
            user_id_to_index[uid] = len(user_id_to_index)
        if mid not in movie_id_to_index:
            movie_idx = len(movie_id_to_index)
            movie_id_to_index[mid] = movie_idx
            index_to_movie_id[movie_idx] = mid

        uf = user_features[uid]
        # 行为特征归一化：avg_rating 除以 5，rating_count 做 log 压缩后归一化
        avg_rating_norm = uf["avg_rating"] / 5.0
        count_norm = math.log1p(uf["rating_count"]) / math.log1p(max_rating_count)

        samples.append({
            "user_index": user_id_to_index[uid],
            "gender_index": uf["gender_index"],
            "age_index": uf["age_index"],
            "occupation_index": uf["occupation_index"],
            "movie_index": movie_id_to_index[mid],
            "genre_vector": movie_features[mid]["genre_vector"],
            "user_behavior": [avg_rating_norm, count_norm],
            "label": 1 if rating >= 4 else 0,
        })

    # feature_info 随模型权重一起保存，推理时复用相同的索引映射和特征统计
    feature_info = {
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": index_to_movie_id,
        "user_features": user_features,
        "movie_features": movie_features,
        "gender_count": len(gender_to_index),
        "age_count": len(age_to_index),
        "occupation_count": len(occupation_to_index),
        "genre_count": len(genre_to_index),
        "max_rating_count": max_rating_count,
    }
    return samples, feature_info


def load_mysql_dataset_if_configured(split="train"):
    from database.dataset_repository import load_mysql_dataset
    return load_mysql_dataset(split=split)


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def build_model_from_checkpoint(checkpoint, device):
    # 从 checkpoint 中读取 feature_info，用于确定各 Embedding 层的大小
    fi = checkpoint["feature_info"]
    model = TwoTowerModel(
        user_count=len(fi["user_id_to_index"]),
        movie_count=len(fi["movie_id_to_index"]),
        gender_count=fi["gender_count"],
        age_count=fi["age_count"],
        occupation_count=fi["occupation_count"],
        genre_count=fi["genre_count"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------- 推理工具 ----------

def _build_user_tensors(user_feature, movie_count, feature_info, device):
    """推理时为一个用户广播出 movie_count 份用户侧张量，对所有电影统一打分。"""
    max_count = feature_info.get("max_rating_count", 1) or 1
    avg_norm = user_feature["avg_rating"] / 5.0
    count_norm = math.log1p(user_feature["rating_count"]) / math.log1p(max_count)
    # repeat 将标量值复制 movie_count 份，生成对应的 1D tensor
    repeat = lambda val, dtype: torch.tensor([val] * movie_count, dtype=dtype, device=device)
    return (
        repeat(user_feature["_index"], torch.long),
        repeat(user_feature["gender_index"], torch.long),
        repeat(user_feature["age_index"], torch.long),
        repeat(user_feature["occupation_index"], torch.long),
        torch.tensor([[avg_norm, count_norm]] * movie_count, dtype=torch.float, device=device),
    )


class TwoTowerRecaller:
    """加载训练好的双塔模型，对单个用户批量计算所有电影的召回分数，返回 top_k。"""

    def __init__(self, model_path=MODEL_PATH):
        self.device = get_device()
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.feature_info = self.checkpoint["feature_info"]
        self.model = build_model_from_checkpoint(self.checkpoint, self.device)

        # 预计算所有电影的索引和 genre 向量并缓存为 tensor
        # 电影侧特征在模型不变时是静态的，无需每次请求重新构建
        # 推理时只需计算用户向量，再与缓存的电影矩阵做批量余弦相似度
        fi = self.feature_info
        movie_count = len(fi["index_to_movie_id"])
        self._movie_tensor = torch.arange(movie_count, dtype=torch.long, device=self.device)
        self._genre_tensor = torch.tensor(
            [fi["movie_features"][fi["index_to_movie_id"][i]]["genre_vector"] for i in range(movie_count)],
            dtype=torch.float, device=self.device,
        )

    def recommend(self, user_id, top_k=10, include_title=True):
        fi = self.feature_info
        if user_id not in fi["user_id_to_index"]:
            logger.warning("user_id=%s 不在训练集中，双塔召回跳过", user_id)
            return []

        # 把用户索引附加到 feature dict 上，供 _build_user_tensors 使用
        user_feature = {**fi["user_features"][user_id], "_index": fi["user_id_to_index"][user_id]}
        movie_count = len(fi["index_to_movie_id"])

        with torch.no_grad():
            u, g, a, o, beh = _build_user_tensors(user_feature, movie_count, fi, self.device)
            # 直接使用初始化时预计算的电影侧 tensor，避免每次请求重建
            scores = self.model(u, g, a, o, self._movie_tensor, self._genre_tensor, beh)
            top_scores, top_idxs = torch.topk(scores, top_k)

        recommendations = [
            {"movie_id": fi["index_to_movie_id"][int(idx)], "score": float(s)}
            for s, idx in zip(top_scores.cpu(), top_idxs.cpu())
        ]
        return add_movie_titles(recommendations) if include_title else recommendations


def recommend_for_user(user_id, top_k=10):
    # 命令行单次推理入口（每次都重新加载模型，批量使用请用 TwoTowerRecaller 类）
    return TwoTowerRecaller().recommend(user_id=user_id, top_k=top_k)


# ---------- 训练 ----------

def train_model(epochs=3, batch_size=1024, learning_rate=0.001):
    samples, feature_info = load_train_samples()
    device = get_device()
    logger.info("当前训练设备: %s", device)

    dataloader = DataLoader(RatingDataset(samples), batch_size=batch_size, shuffle=True)
    model = TwoTowerModel(
        user_count=len(feature_info["user_id_to_index"]),
        movie_count=len(feature_info["movie_id_to_index"]),
        gender_count=feature_info["gender_count"],
        age_count=feature_info["age_count"],
        occupation_count=feature_info["occupation_count"],
        genre_count=feature_info["genre_count"],
    ).to(device)

    # BCE 损失：把余弦相似度当作二分类 logit（喜欢/不喜欢）
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    MODEL_DIR.mkdir(exist_ok=True)

    for epoch in range(epochs):
        total_loss = sum(
            _train_step(model, move_batch_to_device(batch, device), loss_fn, optimizer)
            for batch in dataloader
        )
        avg_loss = total_loss / len(dataloader)
        logger.info("epoch=%d  loss=%.4f", epoch + 1, avg_loss)

        # 每轮保存一个 checkpoint，方便后续评估选最优 epoch
        torch.save({"model_state_dict": model.state_dict(), "feature_info": feature_info, "epoch": epoch + 1, "loss": avg_loss},
                   MODEL_DIR / f"two_tower_epoch_{epoch + 1}.pt")

    # 最终模型覆盖写入 two_tower.pt
    torch.save({"model_state_dict": model.state_dict(), "feature_info": feature_info, "epoch": epochs}, MODEL_PATH)
    logger.info("模型已保存到: %s", MODEL_PATH)


def _train_step(model, batch, loss_fn, optimizer):
    # 单个 batch 的前向 + 反向 + 参数更新，返回 loss 标量
    score = model(batch["user_index"], batch["gender_index"], batch["age_index"],
                  batch["occupation_index"], batch["movie_index"], batch["genre_vector"], batch["user_behavior"])
    loss = loss_fn(score, batch["label"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "recommend"], default="train")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "train":
        train_model(epochs=args.epochs)
    else:
        print_recommendations(recommend_for_user(user_id=args.user_id, top_k=args.top_k))


if __name__ == "__main__":
    main()
