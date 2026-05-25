import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

try:
    from .rough_rank_three_tower import ThreeTowerRoughRankModel
except ImportError:
    from rough_rank_three_tower import ThreeTowerRoughRankModel



BASE_DIR = Path(__file__).resolve().parent.parent
ROUGH_RANK_MODEL_DIR = BASE_DIR / "rough_rank_model"
TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"
TEST_RATINGS_PATH = TEST_DIR / "ratings.dat"
USERS_PATH = TRAIN_DIR / "users.dat"
MOVIES_PATH = TRAIN_DIR / "movies.dat"
MODEL_DIR = ROUGH_RANK_MODEL_DIR
MODEL_PATH = MODEL_DIR / "rough_rank_three_tower.pt"

DENSE_FEATURE_DIM = 4


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class RoughRankDataset(Dataset):
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
            "dense_features": torch.tensor(
                sample["dense_features"], dtype=torch.float
            ),
            "label": torch.tensor(sample["label"], dtype=torch.float),
        }


def load_user_features(users_path=USERS_PATH):
    user_features = {}
    gender_to_index = {}
    age_to_index = {}
    occupation_to_index = {}

    with users_path.open("r", encoding="utf-8") as users_file:
        for line in users_file:
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
            movie_id, title, genres = line.strip().split("::")
            movie_id = int(movie_id)
            genre_list = genres.split("|")

            for genre in genre_list:
                if genre not in genre_to_index:
                    genre_to_index[genre] = len(genre_to_index)

            movie_genres[movie_id] = genre_list

    movie_features = {}
    genre_count = len(genre_to_index)

    for movie_id, genre_list in movie_genres.items():
        genre_vector = [0] * genre_count

        for genre in genre_list:
            genre_index = genre_to_index[genre]
            genre_vector[genre_index] = 1

        movie_features[movie_id] = {
            "genre_vector": genre_vector,
        }

    return movie_features, genre_to_index


def build_rating_stats(ratings_path=TRAIN_RATINGS_PATH):
    user_rating_sum = {}
    user_rating_count = {}
    movie_rating_sum = {}
    movie_rating_count = {}

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            user_id = int(user_id)
            movie_id = int(movie_id)
            rating = int(rating)

            if user_id not in user_rating_sum:
                user_rating_sum[user_id] = 0
                user_rating_count[user_id] = 0

            if movie_id not in movie_rating_sum:
                movie_rating_sum[movie_id] = 0
                movie_rating_count[movie_id] = 0

            user_rating_sum[user_id] += rating
            user_rating_count[user_id] += 1
            movie_rating_sum[movie_id] += rating
            movie_rating_count[movie_id] += 1

    max_user_count = max(user_rating_count.values())
    max_movie_count = max(movie_rating_count.values())

    return {
        "user_rating_sum": user_rating_sum,
        "user_rating_count": user_rating_count,
        "movie_rating_sum": movie_rating_sum,
        "movie_rating_count": movie_rating_count,
        "max_user_count": max_user_count,
        "max_movie_count": max_movie_count,
    }


def build_dense_features(user_id, movie_id, rating_stats):
    user_rating_sum = rating_stats["user_rating_sum"]
    user_rating_count = rating_stats["user_rating_count"]
    movie_rating_sum = rating_stats["movie_rating_sum"]
    movie_rating_count = rating_stats["movie_rating_count"]

    user_count = user_rating_count.get(user_id, 0)
    movie_count = movie_rating_count.get(movie_id, 0)

    if user_count > 0:
        user_avg_rating = user_rating_sum[user_id] / user_count
    else:
        user_avg_rating = 3

    if movie_count > 0:
        movie_avg_rating = movie_rating_sum[movie_id] / movie_count
    else:
        movie_avg_rating = 3

    user_count_feature = user_count / rating_stats["max_user_count"]
    movie_count_feature = movie_count / rating_stats["max_movie_count"]

    return [
        user_avg_rating / 5,
        user_count_feature,
        movie_avg_rating / 5,
        movie_count_feature,
    ]


def build_feature_info():
    user_features, gender_to_index, age_to_index, occupation_to_index = (
        load_user_features()
    )
    movie_features, genre_to_index = load_movie_features()
    rating_stats = build_rating_stats()

    user_id_to_index = {}
    movie_id_to_index = {}
    index_to_movie_id = {}

    with TRAIN_RATINGS_PATH.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            user_id = int(user_id)
            movie_id = int(movie_id)
            rating = int(rating)

            if rating == 3:
                continue

            if user_id not in user_id_to_index:
                user_id_to_index[user_id] = len(user_id_to_index)

            if movie_id not in movie_id_to_index:
                movie_index = len(movie_id_to_index)
                movie_id_to_index[movie_id] = movie_index
                index_to_movie_id[movie_index] = movie_id

    return {
        "user_features": user_features,
        "movie_features": movie_features,
        "rating_stats": rating_stats,
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": index_to_movie_id,
        "gender_count": len(gender_to_index),
        "age_count": len(age_to_index),
        "occupation_count": len(occupation_to_index),
        "genre_count": len(genre_to_index),
        "dense_feature_dim": DENSE_FEATURE_DIM,
    }


def load_samples(ratings_path, feature_info, skip_unknown=True):
    samples = []
    user_features = feature_info["user_features"]
    movie_features = feature_info["movie_features"]
    rating_stats = feature_info["rating_stats"]
    user_id_to_index = feature_info["user_id_to_index"]
    movie_id_to_index = feature_info["movie_id_to_index"]

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            user_id = int(user_id)
            movie_id = int(movie_id)
            rating = int(rating)

            if rating == 3:
                continue

            if skip_unknown and (
                user_id not in user_id_to_index or movie_id not in movie_id_to_index
            ):
                continue

            if rating >= 4:
                label = 1
            else:
                label = 0

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
                    "dense_features": build_dense_features(
                        user_id=user_id,
                        movie_id=movie_id,
                        rating_stats=rating_stats,
                    ),
                    "label": label,
                }
            )

    return samples


def move_batch_to_device(batch, device):
    return {
        name: tensor.to(device)
        for name, tensor in batch.items()
    }


def build_model(feature_info, device):
    model = ThreeTowerRoughRankModel(
        user_count=len(feature_info["user_id_to_index"]),
        movie_count=len(feature_info["movie_id_to_index"]),
        gender_count=feature_info["gender_count"],
        age_count=feature_info["age_count"],
        occupation_count=feature_info["occupation_count"],
        genre_count=feature_info["genre_count"],
        dense_feature_dim=feature_info["dense_feature_dim"],
    )

    return model.to(device)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

            score = model(
                batch["user_index"],
                batch["gender_index"],
                batch["age_index"],
                batch["occupation_index"],
                batch["movie_index"],
                batch["genre_vector"],
                batch["dense_features"],
            )
            loss = loss_fn(score, batch["label"])
            total_loss += loss.item()

            probability = torch.sigmoid(score)
            prediction = (probability >= 0.5).float()
            correct_count += (prediction == batch["label"]).sum().item()
            sample_count += batch["label"].shape[0]

    average_loss = total_loss / len(dataloader)
    accuracy = correct_count / sample_count
    model.train()

    return average_loss, accuracy


def train_model(epochs=3, batch_size=1024, learning_rate=0.001):
    device = get_device()
    print(f"当前训练设备: {device}")

    feature_info = build_feature_info()
    train_samples = load_samples(TRAIN_RATINGS_PATH, feature_info)
    test_samples = load_samples(TEST_RATINGS_PATH, feature_info)

    train_dataset = RoughRankDataset(train_samples)
    test_dataset = RoughRankDataset(test_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(feature_info, device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    MODEL_DIR.mkdir(exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_dataloader:
            batch = move_batch_to_device(batch, device)

            score = model(
                batch["user_index"],
                batch["gender_index"],
                batch["age_index"],
                batch["occupation_index"],
                batch["movie_index"],
                batch["genre_vector"],
                batch["dense_features"],
            )
            loss = loss_fn(score, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_dataloader)
        test_loss, test_accuracy = evaluate(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_accuracy={test_accuracy:.4f}"
        )

        epoch_model_path = MODEL_DIR / f"rough_rank_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_info": feature_info,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            },
            epoch_model_path,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_info": feature_info,
            "epoch": epochs,
        },
        MODEL_PATH,
    )
    print(f"模型已保存到: {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="训练三塔粗排模型")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
