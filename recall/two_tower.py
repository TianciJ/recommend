import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, ReLU, Sequential, Embedding,Relu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .movie_utils import add_movie_titles
from .movie_utils import print_recommendations

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "train_data"
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"
USERS_PATH = TRAIN_DIR / "users.dat"
MOVIES_PATH = TRAIN_DIR / "movies.dat"
MODEL_DIR = BASE_DIR / "model_weights"
MODEL_PATH = MODEL_DIR / "two_tower.pt"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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

        # 用户塔输入：user_id_emb 32维 + gender_emb 4维 + age_emb 8维 + occupation_emb 8维
        self.user_embedding = Embedding(user_count, 32)
        self.gender_embedding = Embedding(gender_count, 4)
        self.age_embedding = Embedding(age_count, 8)
        self.occupation_embedding = Embedding(occupation_count, 8)

        # 用户塔 MLP: 52 -> 128 -> 64 -> 64
        self.user_tower = Sequential(
            Linear(52, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 64),
        )

        # 物品塔输入：movie_id_emb 32维 + genres_emb 16维
        self.movie_embedding = Embedding(movie_count, 32)
        self.genre_layer = Linear(genre_count, 16)

        # 物品塔 MLP: 48 -> 128 -> 64 -> 64
        self.movie_tower = Sequential(
            Linear(48, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 64),
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
        user_id_vector = self.user_embedding(user_index)
        gender_vector = self.gender_embedding(gender_index)
        age_vector = self.age_embedding(age_index)
        occupation_vector = self.occupation_embedding(occupation_index)

        user_input = torch.cat(
            [user_id_vector, gender_vector, age_vector, occupation_vector], dim=1
        )
        user_vector = self.user_tower(user_input)

        movie_id_vector = self.movie_embedding(movie_index)
        genre_feature_vector = self.genre_layer(genre_vector)

        movie_input = torch.cat([movie_id_vector, genre_feature_vector], dim=1)
        movie_vector = self.movie_tower(movie_input)

        score = F.cosine_similarity(user_vector, movie_vector, dim=1)
        return score


def load_user_features(users_path=USERS_PATH, users=None):
    user_features = {}
    gender_to_index = {}
    age_to_index = {}
    occupation_to_index = {}

    if users is None:
        users = load_users_from_dat(users_path)

    for user in users:
        user_id = int(user["user_id"])
        gender = str(user["gender"])
        age = str(user["age"])
        occupation = str(user["occupation"])

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


def load_users_from_dat(users_path=USERS_PATH):
    users = []

    with users_path.open("r", encoding="utf-8") as users_file:
        for line in users_file:
            user_id, gender, age, occupation, zip_code = line.strip().split("::")
            users.append(
                {
                    "user_id": int(user_id),
                    "gender": gender,
                    "age": int(age),
                    "occupation": int(occupation),
                    "zip_code": zip_code,
                }
            )

    return users


def load_movie_features(movies_path=MOVIES_PATH, movies=None):
    movie_genres = {}
    genre_to_index = {}

    if movies is None:
        movies = load_movies_from_dat(movies_path)

    for movie in movies:
        movie_id = int(movie["movie_id"])
        genre_list = list(movie["genres"])

        for genre in genre_list:
            if genre not in genre_to_index:
                genre_to_index[genre] = len(genre_to_index)

        movie_genres[movie_id] = genre_list

    movie_features = {}
    genre_count = len(genre_to_index)

    for movie_id in movie_genres:
        genre_vector = [0] * genre_count

        for genre in movie_genres[movie_id]:
            genre_index = genre_to_index[genre]
            genre_vector[genre_index] = 1

        movie_features[movie_id] = {
            "genre_vector": genre_vector,
        }

    return movie_features, genre_to_index


def load_movies_from_dat(movies_path=MOVIES_PATH):
    movies = []

    with movies_path.open("r", encoding="latin-1") as movies_file:
        for line in movies_file:
            movie_id, title, genres = line.strip().split("::")
            movies.append(
                {
                    "movie_id": int(movie_id),
                    "title": title,
                    "genres": genres.split("|"),
                }
            )

    return movies


def load_ratings_from_dat(ratings_path=TRAIN_RATINGS_PATH):
    ratings = []

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            ratings.append(
                {
                    "user_id": int(user_id),
                    "movie_id": int(movie_id),
                    "rating": int(rating),
                    "timestamp": int(timestamp),
                }
            )

    return ratings


def load_train_samples(ratings_path=TRAIN_RATINGS_PATH, ratings=None, users=None, movies=None):
    if ratings is None and users is None and movies is None:
        mysql_dataset = load_mysql_dataset_if_configured(split="train")

        if mysql_dataset is not None:
            ratings = mysql_dataset["ratings"]
            users = mysql_dataset["users"]
            movies = mysql_dataset["movies"]

    if ratings is None:
        ratings = load_ratings_from_dat(ratings_path)

    samples = []
    user_id_to_index = {}
    movie_id_to_index = {}
    index_to_movie_id = {}

    user_features, gender_to_index, age_to_index, occupation_to_index = (
        load_user_features(users=users)
    )
    movie_features, genre_to_index = load_movie_features(movies=movies)

    for rating_row in ratings:
        user_id = int(rating_row["user_id"])
        movie_id = int(rating_row["movie_id"])
        rating = int(rating_row["rating"])

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


def load_mysql_dataset_if_configured(split="train"):
    from database.dataset_repository import load_mysql_dataset

    return load_mysql_dataset(split=split)


def move_batch_to_device(batch, device):
    return {
        name: tensor.to(device)
        for name, tensor in batch.items()
    }


def train_model(epochs=3, batch_size=1024, learning_rate=0.001):
    samples, feature_info = load_train_samples()
    device = get_device()
    print(f"当前训练设备: {device}")

    dataset = RatingDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        total_loss = 0

        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

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

        epoch_model_path = MODEL_DIR / f"model_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_info": feature_info,
                "epoch": epoch + 1,
                "loss": average_loss,
            },
            epoch_model_path,
        )
        print(f"第 {epoch + 1} 轮模型已保存到: {epoch_model_path}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_info": feature_info,
            "epoch": epochs,
        },
        MODEL_PATH,
    )

    print(f"模型已保存到: {MODEL_PATH}")


def build_model_from_checkpoint(checkpoint, device):
    feature_info = checkpoint["feature_info"]

    model = TwoTowerModel(
        user_count=len(feature_info["user_id_to_index"]),
        movie_count=len(feature_info["movie_id_to_index"]),
        gender_count=feature_info["gender_count"],
        age_count=feature_info["age_count"],
        occupation_count=feature_info["occupation_count"],
        genre_count=feature_info["genre_count"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


class TwoTowerRecaller:
    def __init__(self, model_path=MODEL_PATH):
        self.device = get_device()
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.feature_info = self.checkpoint["feature_info"]
        self.model = build_model_from_checkpoint(self.checkpoint, self.device)

    def recommend(self, user_id, top_k=10, include_title=True):
        user_id_to_index = self.feature_info["user_id_to_index"]
        index_to_movie_id = self.feature_info["index_to_movie_id"]
        user_features = self.feature_info["user_features"]
        movie_features = self.feature_info["movie_features"]

        if user_id not in user_id_to_index:
            print("这个用户没有出现在训练集中，暂时无法用双塔召回")
            return []

        user_index = user_id_to_index[user_id]
        user_feature = user_features[user_id]
        movie_count = len(index_to_movie_id)

        with torch.no_grad():
            user_tensor = torch.tensor(
                [user_index] * movie_count, dtype=torch.long, device=self.device
            )
            gender_tensor = torch.tensor(
                [user_feature["gender_index"]] * movie_count,
                dtype=torch.long,
                device=self.device,
            )
            age_tensor = torch.tensor(
                [user_feature["age_index"]] * movie_count,
                dtype=torch.long,
                device=self.device,
            )
            occupation_tensor = torch.tensor(
                [user_feature["occupation_index"]] * movie_count,
                dtype=torch.long,
                device=self.device,
            )
            movie_tensor = torch.arange(movie_count, dtype=torch.long, device=self.device)

            genre_vectors = []
            for movie_index in range(movie_count):
                movie_id = index_to_movie_id[movie_index]
                genre_vectors.append(movie_features[movie_id]["genre_vector"])
            genre_tensor = torch.tensor(
                genre_vectors, dtype=torch.float, device=self.device
            )

            scores = self.model(
                user_tensor,
                gender_tensor,
                age_tensor,
                occupation_tensor,
                movie_tensor,
                genre_tensor,
            )
            top_scores, top_movie_indexes = torch.topk(scores, top_k)
            top_scores = top_scores.cpu()
            top_movie_indexes = top_movie_indexes.cpu()

        recommendations = []
        for score, movie_index in zip(top_scores, top_movie_indexes):
            movie_id = index_to_movie_id[int(movie_index)]
            recommendations.append(
                {
                    "movie_id": movie_id,
                    "score": float(score),
                }
            )

        if include_title:
            recommendations = add_movie_titles(recommendations)

        return recommendations


def recommend_for_user(user_id, top_k=10):
    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    feature_info = checkpoint["feature_info"]

    user_id_to_index = feature_info["user_id_to_index"]
    index_to_movie_id = feature_info["index_to_movie_id"]
    user_features = feature_info["user_features"]
    movie_features = feature_info["movie_features"]

    if user_id not in user_id_to_index:
        print("这个用户没有出现在训练集中，暂时无法用双塔召回")
        return []

    model = build_model_from_checkpoint(checkpoint, device)

    user_index = user_id_to_index[user_id]
    user_feature = user_features[user_id]
    movie_count = len(index_to_movie_id)

    with torch.no_grad():
        user_tensor = torch.tensor(
            [user_index] * movie_count, dtype=torch.long, device=device
        )
        gender_tensor = torch.tensor(
            [user_feature["gender_index"]] * movie_count,
            dtype=torch.long,
            device=device,
        )
        age_tensor = torch.tensor(
            [user_feature["age_index"]] * movie_count,
            dtype=torch.long,
            device=device,
        )
        occupation_tensor = torch.tensor(
            [user_feature["occupation_index"]] * movie_count,
            dtype=torch.long,
            device=device,
        )
        movie_tensor = torch.arange(movie_count, dtype=torch.long, device=device)

        genre_vectors = []
        for movie_index in range(movie_count):
            movie_id = index_to_movie_id[movie_index]
            genre_vectors.append(movie_features[movie_id]["genre_vector"])
        genre_tensor = torch.tensor(genre_vectors, dtype=torch.float, device=device)

        scores = model(
            user_tensor,
            gender_tensor,
            age_tensor,
            occupation_tensor,
            movie_tensor,
            genre_tensor,
        )
        top_scores, top_movie_indexes = torch.topk(scores, top_k)
        top_scores = top_scores.cpu()
        top_movie_indexes = top_movie_indexes.cpu()

    recommendations = []
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
