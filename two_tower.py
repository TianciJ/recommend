import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from movie_utils import add_movie_titles
from movie_utils import print_recommendations


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train_data"
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"
USERS_PATH = TRAIN_DIR / "users.dat"
MOVIES_PATH = TRAIN_DIR / "movies.dat"
MODEL_DIR = BASE_DIR / "model_weights"
MODEL_PATH = MODEL_DIR / "two_tower.pt"


class RatingDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        return {
            "user_index": torch.tensor(sample["user_index"], dtype=torch.long),
            "gender_index": torch.tensor(sample["gender_index"], dtype=torch.long),
            "age_index": torch.tensor(sample["age_index"], dtype=torch.long),
            "occupation_index": torch.tensor(
                sample["occupation_index"], dtype=torch.long
            ),
            "movie_index": torch.tensor(sample["movie_index"], dtype=torch.long),
            "genre_vector": torch.tensor(sample["genre_vector"], dtype=torch.float),
            "label": torch.tensor(sample["label"], dtype=torch.float),
        }


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_count,
        movie_count,
        gender_count,
        age_count,
        occupation_count,
        genre_count,

    ):
        super().__init__()

        # 用户侧特征：user_id、gender、age、occupation
        self.user_embedding = nn.Embedding(user_count, 32)
        self.gender_embedding = nn.Embedding(gender_count, 4)
        self.age_embedding = nn.Embedding(age_count, 8)
        self.occupation_embedding = nn.Embedding(occupation_count, 8)

        # 32 + 4 + 8 + 8 = 52
        # MLP: 52 -> 128 -> 64 -> 64
        user_input_dim = 52
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # 电影侧特征：movie_id、genres
        self.movie_embedding = nn.Embedding(movie_count, 32)
        self.genre_layer = nn.Linear(genre_count, 16)

        # 32 + 16 = 48
        # MLP: 48 -> 128 -> 64 -> 64
        movie_input_dim = 48
        self.movie_tower = nn.Sequential(
            nn.Linear(movie_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(
        self,
        user_index,
        gender_index,
        age_index,
        occupation_index,
        movie_index,
        genre_vector,
    ):
        # 取出用户 ID、性别、年龄、职业对应的向量
        user_id_vector = self.user_embedding(user_index)
        gender_vector = self.gender_embedding(gender_index)
        age_vector = self.age_embedding(age_index)
        occupation_vector = self.occupation_embedding(occupation_index)

        # 拼接用户画像特征，然后送入用户塔
        user_input = torch.cat(
            [user_id_vector, gender_vector, age_vector, occupation_vector], dim=1
        )
        user_vector = self.user_tower(user_input)

        # 取出电影 ID 向量，并把 genres 多热向量转换成电影类型向量
        movie_id_vector = self.movie_embedding(movie_index)
        genre_feature_vector = self.genre_layer(genre_vector)

        # 拼接电影画像特征，然后送入电影塔
        movie_input = torch.cat([movie_id_vector, genre_feature_vector], dim=1)
        movie_vector = self.movie_tower(movie_input)

        # 用户向量和电影向量做余弦相似度，得到匹配分数
        score = F.cosine_similarity(user_vector, movie_vector, dim=1)
        return score


def load_user_features(users_path=USERS_PATH):
    user_features = {}
    gender_to_index = {}
    age_to_index = {}
    occupation_to_index = {}

    with users_path.open("r", encoding="utf-8") as users_file:
        for line in users_file:
            # users.dat 格式：UserID::Gender::Age::Occupation::Zip-code
            user_id, gender, age, occupation, zip_code = line.strip().split("::")
            user_id = int(user_id)

            if gender not in gender_to_index:
                gender_to_index[gender] = len(gender_to_index)

            if age not in age_to_index:
                age_to_index[age] = len(age_to_index)

            if occupation not in occupation_to_index:
                occupation_to_index[occupation] = len(occupation_to_index)

            user_features[user_id] = {
                "gender_index": gender_to_index[gender],
                "age_index": age_to_index[age],
                "occupation_index": occupation_to_index[occupation],
            }

    return user_features, gender_to_index, age_to_index, occupation_to_index


def load_movie_features(movies_path=MOVIES_PATH):
    movie_genres = {}
    genre_to_index = {}

    with movies_path.open("r", encoding="latin-1") as movies_file:
        for line in movies_file:
            # movies.dat 格式：MovieID::Title::Genres
            movie_id, title, genres = line.strip().split("::")
            movie_id = int(movie_id)
            genre_list = genres.split("|")

            for genre in genre_list:
                if genre not in genre_to_index:
                    genre_to_index[genre] = len(genre_to_index)

            movie_genres[movie_id] = genre_list

    movie_features = {}
    genre_count = len(genre_to_index)

    for movie_id in movie_genres:
        # 一个电影可能有多个类型，所以这里用多热向量
        # 例如 Action|Comedy 会在 Action 和 Comedy 两个位置都填 1
        genre_vector = [0] * genre_count

        for genre in movie_genres[movie_id]:
            genre_index = genre_to_index[genre]
            genre_vector[genre_index] = 1

        movie_features[movie_id] = {
            "genre_vector": genre_vector,
        }

    return movie_features, genre_to_index


def load_train_samples(ratings_path=TRAIN_RATINGS_PATH):
    samples = []
    user_id_to_index = {}
    movie_id_to_index = {}
    index_to_movie_id = {}

    user_features, gender_to_index, age_to_index, occupation_to_index = (
        load_user_features()
    )
    movie_features, genre_to_index = load_movie_features()

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            user_id = int(user_id)
            movie_id = int(movie_id)
            rating = int(rating)

            # 3 分先跳过；4/5 分是喜欢，1/2 分是不喜欢
            if rating == 3:
                continue

            if rating >= 4:
                label = 1
            else:
                label = 0

            if user_id not in user_id_to_index:
                user_id_to_index[user_id] = len(user_id_to_index)

            if movie_id not in movie_id_to_index:
                movie_index = len(movie_id_to_index)
                movie_id_to_index[movie_id] = movie_index
                index_to_movie_id[movie_index] = movie_id

            user_feature = user_features[user_id]
            movie_feature = movie_features[movie_id]

            samples.append(
                {
                    "user_index": user_id_to_index[user_id],
                    "gender_index": user_feature["gender_index"],
                    "age_index": user_feature["age_index"],
                    "occupation_index": user_feature["occupation_index"],
                    "movie_index": movie_id_to_index[movie_id],
                    "genre_vector": movie_feature["genre_vector"],
                    "label": label,
                }
            )

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
    }

    return samples, feature_info


def train_model(epochs=3, batch_size=1024, embedding_dim=64, learning_rate=0.001):
    samples, feature_info = load_train_samples()

    dataset = RatingDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TwoTowerModel(
        user_count=len(feature_info["user_id_to_index"]),
        movie_count=len(feature_info["movie_id_to_index"]),
        gender_count=feature_info["gender_count"],
        age_count=feature_info["age_count"],
        occupation_count=feature_info["occupation_count"],
        genre_count=feature_info["genre_count"],
        embedding_dim=embedding_dim,
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            score = model(
                batch["user_index"],
                batch["gender_index"],
                batch["age_index"],
                batch["occupation_index"],
                batch["movie_index"],
                batch["genre_vector"],
            )
            loss = loss_fn(score, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"epoch={epoch + 1}, loss={average_loss:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    feature_info["embedding_dim"] = embedding_dim

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_info": feature_info,
        },
        MODEL_PATH,
    )

    print(f"模型已保存到: {MODEL_PATH}")


def build_model_from_checkpoint(checkpoint):
    feature_info = checkpoint["feature_info"]

    model = TwoTowerModel(
        user_count=len(feature_info["user_id_to_index"]),
        movie_count=len(feature_info["movie_id_to_index"]),
        gender_count=feature_info["gender_count"],
        age_count=feature_info["age_count"],
        occupation_count=feature_info["occupation_count"],
        genre_count=feature_info["genre_count"],
        embedding_dim=feature_info["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def recommend_for_user(user_id, top_k=10):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    feature_info = checkpoint["feature_info"]

    user_id_to_index = feature_info["user_id_to_index"]
    index_to_movie_id = feature_info["index_to_movie_id"]
    user_features = feature_info["user_features"]
    movie_features = feature_info["movie_features"]

    if user_id not in user_id_to_index:
        print("这个用户没有出现在训练集中，暂时无法用双塔召回")
        return []

    model = build_model_from_checkpoint(checkpoint)

    user_index = user_id_to_index[user_id]
    user_feature = user_features[user_id]
    movie_count = len(index_to_movie_id)

    recommendations = []

    with torch.no_grad():
        # 当前用户固定不变，依次和所有电影计算匹配分数
        user_tensor = torch.tensor([user_index] * movie_count, dtype=torch.long)
        gender_tensor = torch.tensor(
            [user_feature["gender_index"]] * movie_count, dtype=torch.long
        )
        age_tensor = torch.tensor(
            [user_feature["age_index"]] * movie_count, dtype=torch.long
        )
        occupation_tensor = torch.tensor(
            [user_feature["occupation_index"]] * movie_count, dtype=torch.long
        )
        movie_tensor = torch.arange(movie_count, dtype=torch.long)

        genre_vectors = []
        for movie_index in range(movie_count):
            movie_id = index_to_movie_id[movie_index]
            genre_vectors.append(movie_features[movie_id]["genre_vector"])
        genre_tensor = torch.tensor(genre_vectors, dtype=torch.float)

        scores = model(
            user_tensor,
            gender_tensor,
            age_tensor,
            occupation_tensor,
            movie_tensor,
            genre_tensor,
        )
        top_scores, top_movie_indexes = torch.topk(scores, top_k)

    for score, movie_index in zip(top_scores, top_movie_indexes):
        movie_id = index_to_movie_id[int(movie_index)]
        recommendations.append(
            {
                "movie_id": movie_id,
                "score": float(score),
            }
        )

    return add_movie_titles(recommendations)


def main():
    parser = argparse.ArgumentParser(description="带用户画像和电影画像的双塔召回模型")
    parser.add_argument("--mode", choices=["train", "recommend"], default="train")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "train":
        train_model(epochs=args.epochs)
    else:
        results = recommend_for_user(user_id=args.user_id, top_k=args.top_k)
        print_recommendations(results)


if __name__ == "__main__":
    main()
