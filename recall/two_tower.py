import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ReLU, Sequential
from torch.utils.data import DataLoader, Dataset

from .movie_utils import add_movie_titles, print_recommendations

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "train_data"
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"
USERS_PATH = TRAIN_DIR / "users.dat"
MOVIES_PATH = TRAIN_DIR / "movies.dat"
MODEL_DIR = BASE_DIR / "models" / "recall"
MODEL_PATH = MODEL_DIR / "two_tower.pt"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RatingDataset(Dataset):
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


class TwoTowerModel(nn.Module):
    # 用户塔: user_id(32) + gender(4) + age(8) + occupation(8) + behavior(2) = 54
    # 物品塔: movie_id(32) + genre(16) = 48
    def __init__(self, user_count, movie_count, gender_count, age_count, occupation_count, genre_count):
        super().__init__()
        self.user_embedding = Embedding(user_count, 32)
        self.gender_embedding = Embedding(gender_count, 4)
        self.age_embedding = Embedding(age_count, 8)
        self.occupation_embedding = Embedding(occupation_count, 8)
        self.user_tower = Sequential(Linear(54, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 64))

        self.movie_embedding = Embedding(movie_count, 32)
        self.genre_layer = Linear(genre_count, 16)
        self.movie_tower = Sequential(Linear(48, 128), ReLU(), Linear(128, 64), ReLU(), Linear(64, 64))

    def forward(self, user_index, gender_index, age_index, occupation_index, movie_index, genre_vector, user_behavior):
        user_input = torch.cat([
            self.user_embedding(user_index),
            self.gender_embedding(gender_index),
            self.age_embedding(age_index),
            self.occupation_embedding(occupation_index),
            user_behavior,
        ], dim=1)
        movie_input = torch.cat([
            self.movie_embedding(movie_index),
            self.genre_layer(genre_vector),
        ], dim=1)
        return F.cosine_similarity(self.user_tower(user_input), self.movie_tower(movie_input), dim=1)


def load_user_features(users_path=USERS_PATH, users=None, ratings=None):
    if users is None:
        users = load_users_from_dat(users_path)

    gender_to_index, age_to_index, occupation_to_index = {}, {}, {}
    user_features = {}

    for user in users:
        uid = int(user["user_id"])
        gender, age, occ = str(user["gender"]), str(user["age"]), str(user["occupation"])

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
            "avg_rating": 3.0,
            "rating_count": 0,
        }

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
            continue

        if uid not in user_id_to_index:
            user_id_to_index[uid] = len(user_id_to_index)
        if mid not in movie_id_to_index:
            movie_idx = len(movie_id_to_index)
            movie_id_to_index[mid] = movie_idx
            index_to_movie_id[movie_idx] = mid

        uf = user_features[uid]
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


def _build_user_tensors(user_feature, movie_count, feature_info, device):
    """推理时为一个用户广播出 movie_count 份用户侧张量。"""
    max_count = feature_info.get("max_rating_count", 1) or 1
    avg_norm = user_feature["avg_rating"] / 5.0
    count_norm = math.log1p(user_feature["rating_count"]) / math.log1p(max_count)
    repeat = lambda val, dtype: torch.tensor([val] * movie_count, dtype=dtype, device=device)
    return (
        repeat(user_feature["_index"], torch.long),
        repeat(user_feature["gender_index"], torch.long),
        repeat(user_feature["age_index"], torch.long),
        repeat(user_feature["occupation_index"], torch.long),
        torch.tensor([[avg_norm, count_norm]] * movie_count, dtype=torch.float, device=device),
    )


class TwoTowerRecaller:
    def __init__(self, model_path=MODEL_PATH):
        self.device = get_device()
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.feature_info = self.checkpoint["feature_info"]
        self.model = build_model_from_checkpoint(self.checkpoint, self.device)

    def recommend(self, user_id, top_k=10, include_title=True):
        fi = self.feature_info
        if user_id not in fi["user_id_to_index"]:
            print("这个用户没有出现在训练集中，暂时无法用双塔召回")
            return []

        user_feature = {**fi["user_features"][user_id], "_index": fi["user_id_to_index"][user_id]}
        movie_count = len(fi["index_to_movie_id"])

        with torch.no_grad():
            u, g, a, o, beh = _build_user_tensors(user_feature, movie_count, fi, self.device)
            movie_tensor = torch.arange(movie_count, dtype=torch.long, device=self.device)
            genre_tensor = torch.tensor(
                [fi["movie_features"][fi["index_to_movie_id"][i]]["genre_vector"] for i in range(movie_count)],
                dtype=torch.float, device=self.device,
            )
            scores = self.model(u, g, a, o, movie_tensor, genre_tensor, beh)
            top_scores, top_idxs = torch.topk(scores, top_k)

        recommendations = [
            {"movie_id": fi["index_to_movie_id"][int(idx)], "score": float(s)}
            for s, idx in zip(top_scores.cpu(), top_idxs.cpu())
        ]
        return add_movie_titles(recommendations) if include_title else recommendations


def recommend_for_user(user_id, top_k=10):
    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    fi = checkpoint["feature_info"]

    if user_id not in fi["user_id_to_index"]:
        print("这个用户没有出现在训练集中，暂时无法用双塔召回")
        return []

    model = build_model_from_checkpoint(checkpoint, device)
    user_feature = {**fi["user_features"][user_id], "_index": fi["user_id_to_index"][user_id]}
    movie_count = len(fi["index_to_movie_id"])

    with torch.no_grad():
        u, g, a, o, beh = _build_user_tensors(user_feature, movie_count, fi, device)
        movie_tensor = torch.arange(movie_count, dtype=torch.long, device=device)
        genre_tensor = torch.tensor(
            [fi["movie_features"][fi["index_to_movie_id"][i]]["genre_vector"] for i in range(movie_count)],
            dtype=torch.float, device=device,
        )
        scores = model(u, g, a, o, movie_tensor, genre_tensor, beh)
        top_scores, top_idxs = torch.topk(scores, top_k)

    recommendations = [
        {"movie_id": fi["index_to_movie_id"][int(idx)], "score": float(s)}
        for s, idx in zip(top_scores.cpu(), top_idxs.cpu())
    ]
    return add_movie_titles(recommendations)


def train_model(epochs=3, batch_size=1024, learning_rate=0.001):
    samples, feature_info = load_train_samples()
    device = get_device()
    print(f"当前训练设备: {device}")

    dataloader = DataLoader(RatingDataset(samples), batch_size=batch_size, shuffle=True)
    model = TwoTowerModel(
        user_count=len(feature_info["user_id_to_index"]),
        movie_count=len(feature_info["movie_id_to_index"]),
        gender_count=feature_info["gender_count"],
        age_count=feature_info["age_count"],
        occupation_count=feature_info["occupation_count"],
        genre_count=feature_info["genre_count"],
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    MODEL_DIR.mkdir(exist_ok=True)

    for epoch in range(epochs):
        total_loss = sum(
            _train_step(model, move_batch_to_device(batch, device), loss_fn, optimizer)
            for batch in dataloader
        )
        avg_loss = total_loss / len(dataloader)
        print(f"epoch={epoch + 1}, loss={avg_loss:.4f}")

        torch.save({"model_state_dict": model.state_dict(), "feature_info": feature_info, "epoch": epoch + 1, "loss": avg_loss},
                   MODEL_DIR / f"two_tower_epoch_{epoch + 1}.pt")

    torch.save({"model_state_dict": model.state_dict(), "feature_info": feature_info, "epoch": epochs}, MODEL_PATH)
    print(f"模型已保存到: {MODEL_PATH}")


def _train_step(model, batch, loss_fn, optimizer):
    score = model(batch["user_index"], batch["gender_index"], batch["age_index"],
                  batch["occupation_index"], batch["movie_index"], batch["genre_vector"], batch["user_behavior"])
    loss = loss_fn(score, batch["label"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
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
