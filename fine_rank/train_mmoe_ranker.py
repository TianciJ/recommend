import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .mmoe_ranker import MMoERanker
from .mmoe_ranker import evaluate
from .mmoe_ranker import get_device
from .mmoe_ranker import train_one_epoch

from recall.two_tower import build_model_from_checkpoint as build_recall_model
from recall.two_tower import load_movies_from_dat
from recall.two_tower import load_mysql_dataset_if_configured
from recall.two_tower import load_ratings_from_dat
from recall.two_tower import load_users_from_dat
from rough_rank.rough_rank_inference import build_dense_features
from rough_rank.rough_rank_inference import build_model_from_checkpoint as build_rough_model


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"
TEST_RATINGS_PATH = TEST_DIR / "ratings.dat"
USERS_PATH = TRAIN_DIR / "users.dat"
MOVIES_PATH = TRAIN_DIR / "movies.dat"

RECALL_MODEL_PATH = BASE_DIR / "model_weights" / "two_tower.pt"
ROUGH_MODEL_PATH = BASE_DIR / "rough_rank_model" / "rough_rank_three_tower.pt"
FINE_RANK_MODEL_DIR = BASE_DIR / "fine_rank_model"
FINE_RANK_MODEL_PATH = FINE_RANK_MODEL_DIR / "mmoe_ranker.pt"


class MMoEDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        return {
            "user_id": torch.tensor(sample["user_id"], dtype=torch.long),
            "gender": torch.tensor(sample["gender"], dtype=torch.long),
            "age": torch.tensor(sample["age"], dtype=torch.long),
            "occupation": torch.tensor(sample["occupation"], dtype=torch.long),
            "movie_id": torch.tensor(sample["movie_id"], dtype=torch.long),
            "genres": torch.tensor(sample["genres"], dtype=torch.long),
            "recall_score": torch.tensor(sample["recall_score"], dtype=torch.float),
            "coarse_score": torch.tensor(sample["coarse_score"], dtype=torch.float),
            "like_label": torch.tensor(sample["like_label"], dtype=torch.float),
            "high_rating_label": torch.tensor(
                sample["high_rating_label"], dtype=torch.float
            ),
            "rating_label": torch.tensor(sample["rating_label"], dtype=torch.float),
        }


def load_checkpoint(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)


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
            "gender": gender_to_index[gender],
            "age": age_to_index[age],
            "occupation": occupation_to_index[occupation],
        }

    return user_features, gender_to_index, age_to_index, occupation_to_index


def load_movie_features(movies_path=MOVIES_PATH, movies=None):
    movie_features = {}
    genre_to_index = {}
    max_genre_length = 0

    if movies is not None:
        for movie in movies:
            movie_id = int(movie["movie_id"])
            genre_indexes = []

            for genre in movie["genres"]:
                if genre not in genre_to_index:
                    genre_to_index[genre] = len(genre_to_index) + 1

                genre_indexes.append(genre_to_index[genre])

            max_genre_length = max(max_genre_length, len(genre_indexes))
            movie_features[movie_id] = {
                "genres": genre_indexes,
            }

        for movie_id in movie_features:
            genres = movie_features[movie_id]["genres"]
            padding_count = max_genre_length - len(genres)
            movie_features[movie_id]["genres"] = genres + [0] * padding_count

        return movie_features, genre_to_index, max_genre_length

    with movies_path.open("r", encoding="latin-1") as movies_file:
        for line in movies_file:
            movie_id, title, genres = line.strip().split("::")
            movie_id = int(movie_id)
            genre_list = genres.split("|")

            genre_indexes = []
            for genre in genre_list:
                if genre not in genre_to_index:
                    # 0 留给 padding，所以真实 genre 从 1 开始
                    genre_to_index[genre] = len(genre_to_index) + 1

                genre_indexes.append(genre_to_index[genre])

            max_genre_length = max(max_genre_length, len(genre_indexes))
            movie_features[movie_id] = {
                "genres": genre_indexes,
            }

    for movie_id in movie_features:
        genres = movie_features[movie_id]["genres"]
        padding_count = max_genre_length - len(genres)
        movie_features[movie_id]["genres"] = genres + [0] * padding_count

    return movie_features, genre_to_index, max_genre_length


def build_feature_info(users=None, movies=None, ratings=None):
    if users is None and movies is None and ratings is None:
        mysql_dataset = load_mysql_dataset_if_configured(split="train")

        if mysql_dataset is not None:
            users = mysql_dataset["users"]
            movies = mysql_dataset["movies"]
            ratings = mysql_dataset["ratings"]

    if ratings is None:
        ratings = load_ratings_from_dat(TRAIN_RATINGS_PATH)

    user_features, gender_to_index, age_to_index, occupation_to_index = (
        load_user_features(users=users)
    )
    movie_features, genre_to_index, max_genre_length = load_movie_features(movies=movies)

    user_id_to_index = {}
    movie_id_to_index = {}
    index_to_movie_id = {}

    for rating_row in ratings:
        user_id = int(rating_row["user_id"])
        movie_id = int(rating_row["movie_id"])

        if user_id not in user_id_to_index:
            user_id_to_index[user_id] = len(user_id_to_index)

        if movie_id not in movie_id_to_index:
            movie_index = len(movie_id_to_index)
            movie_id_to_index[movie_id] = movie_index
            index_to_movie_id[movie_index] = movie_id

    return {
        "user_features": user_features,
        "movie_features": movie_features,
        "user_id_to_index": user_id_to_index,
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": index_to_movie_id,
        "gender_count": len(gender_to_index),
        "age_count": len(age_to_index),
        "occupation_count": len(occupation_to_index),
        "genre_count": len(genre_to_index) + 1,
        "max_genre_length": max_genre_length,
    }


def load_base_samples(ratings_path, feature_info, skip_unknown=True, ratings=None):
    if ratings is None:
        split = "test" if ratings_path == TEST_RATINGS_PATH else "train"
        mysql_dataset = load_mysql_dataset_if_configured(split=split)
        ratings = (
            mysql_dataset["ratings"]
            if mysql_dataset is not None
            else load_ratings_from_dat(ratings_path)
        )

    samples = []
    user_features = feature_info["user_features"]
    movie_features = feature_info["movie_features"]
    user_id_to_index = feature_info["user_id_to_index"]
    movie_id_to_index = feature_info["movie_id_to_index"]

    for rating_row in ratings:
        raw_user_id = int(rating_row["user_id"])
        raw_movie_id = int(rating_row["movie_id"])
        rating = int(rating_row["rating"])

        if skip_unknown and (
            raw_user_id not in user_id_to_index
            or raw_movie_id not in movie_id_to_index
        ):
            continue

        user_feature = user_features[raw_user_id]
        movie_feature = movie_features[raw_movie_id]

        samples.append(
            {
                "raw_user_id": raw_user_id,
                "raw_movie_id": raw_movie_id,
                "user_id": user_id_to_index[raw_user_id],
                "gender": user_feature["gender"],
                "age": user_feature["age"],
                "occupation": user_feature["occupation"],
                "movie_id": movie_id_to_index[raw_movie_id],
                "genres": movie_feature["genres"],
                "like_label": 1 if rating >= 4 else 0,
                "high_rating_label": 1 if rating == 5 else 0,
                "rating_label": rating / 5,
            }
        )

    return samples


class RecallScorer:
    def __init__(self, device):
        checkpoint = load_checkpoint(RECALL_MODEL_PATH, device)
        self.feature_info = checkpoint["feature_info"]
        self.model = build_recall_model(checkpoint, device)
        self.device = device

    def score_samples(self, samples, batch_size):
        scores = []
        user_id_to_index = self.feature_info["user_id_to_index"]
        movie_id_to_index = self.feature_info["movie_id_to_index"]
        user_features = self.feature_info["user_features"]
        movie_features = self.feature_info["movie_features"]

        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch_samples = samples[start : start + batch_size]

                valid_samples = [
                    sample
                    for sample in batch_samples
                    if sample["raw_user_id"] in user_id_to_index
                    and sample["raw_movie_id"] in movie_id_to_index
                ]

                batch_scores = [0.0] * len(batch_samples)
                if not valid_samples:
                    scores.extend(batch_scores)
                    continue

                user_indexes = []
                gender_indexes = []
                age_indexes = []
                occupation_indexes = []
                movie_indexes = []
                genre_vectors = []
                valid_positions = []

                for position, sample in enumerate(batch_samples):
                    raw_user_id = sample["raw_user_id"]
                    raw_movie_id = sample["raw_movie_id"]

                    if raw_user_id not in user_id_to_index:
                        continue

                    if raw_movie_id not in movie_id_to_index:
                        continue

                    user_feature = user_features[raw_user_id]
                    movie_feature = movie_features[raw_movie_id]

                    user_indexes.append(user_id_to_index[raw_user_id])
                    gender_indexes.append(user_feature["gender_index"])
                    age_indexes.append(user_feature["age_index"])
                    occupation_indexes.append(user_feature["occupation_index"])
                    movie_indexes.append(movie_id_to_index[raw_movie_id])
                    genre_vectors.append(movie_feature["genre_vector"])
                    valid_positions.append(position)

                output = self.model(
                    torch.tensor(user_indexes, dtype=torch.long, device=self.device),
                    torch.tensor(gender_indexes, dtype=torch.long, device=self.device),
                    torch.tensor(age_indexes, dtype=torch.long, device=self.device),
                    torch.tensor(
                        occupation_indexes, dtype=torch.long, device=self.device
                    ),
                    torch.tensor(movie_indexes, dtype=torch.long, device=self.device),
                    torch.tensor(genre_vectors, dtype=torch.float, device=self.device),
                )

                output = output.cpu().tolist()
                for position, score in zip(valid_positions, output):
                    batch_scores[position] = score

                scores.extend(batch_scores)

        return scores


class CoarseScorer:
    def __init__(self, device):
        checkpoint = load_checkpoint(ROUGH_MODEL_PATH, device)
        self.feature_info = checkpoint["feature_info"]
        self.model = build_rough_model(checkpoint, device)
        self.device = device

    def score_samples(self, samples, batch_size):
        scores = []
        user_id_to_index = self.feature_info["user_id_to_index"]
        movie_id_to_index = self.feature_info["movie_id_to_index"]
        user_features = self.feature_info["user_features"]
        movie_features = self.feature_info["movie_features"]
        rating_stats = self.feature_info["rating_stats"]

        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch_samples = samples[start : start + batch_size]
                batch_scores = [0.0] * len(batch_samples)

                user_indexes = []
                gender_indexes = []
                age_indexes = []
                occupation_indexes = []
                movie_indexes = []
                genre_vectors = []
                dense_features = []
                valid_positions = []

                for position, sample in enumerate(batch_samples):
                    raw_user_id = sample["raw_user_id"]
                    raw_movie_id = sample["raw_movie_id"]

                    if raw_user_id not in user_id_to_index:
                        continue

                    if raw_movie_id not in movie_id_to_index:
                        continue

                    user_feature = user_features[raw_user_id]
                    movie_feature = movie_features[raw_movie_id]

                    user_indexes.append(user_id_to_index[raw_user_id])
                    gender_indexes.append(user_feature["gender_index"])
                    age_indexes.append(user_feature["age_index"])
                    occupation_indexes.append(user_feature["occupation_index"])
                    movie_indexes.append(movie_id_to_index[raw_movie_id])
                    genre_vectors.append(movie_feature["genre_vector"])
                    dense_features.append(
                        build_dense_features(
                            user_id=raw_user_id,
                            movie_id=raw_movie_id,
                            rating_stats=rating_stats,
                        )
                    )
                    valid_positions.append(position)

                if not valid_positions:
                    scores.extend(batch_scores)
                    continue

                output = self.model(
                    torch.tensor(user_indexes, dtype=torch.long, device=self.device),
                    torch.tensor(gender_indexes, dtype=torch.long, device=self.device),
                    torch.tensor(age_indexes, dtype=torch.long, device=self.device),
                    torch.tensor(
                        occupation_indexes, dtype=torch.long, device=self.device
                    ),
                    torch.tensor(movie_indexes, dtype=torch.long, device=self.device),
                    torch.tensor(genre_vectors, dtype=torch.float, device=self.device),
                    torch.tensor(dense_features, dtype=torch.float, device=self.device),
                )

                output = output.cpu().tolist()
                for position, score in zip(valid_positions, output):
                    batch_scores[position] = score

                scores.extend(batch_scores)

        return scores


def attach_model_scores(samples, recall_scorer, coarse_scorer, batch_size):
    recall_scores = recall_scorer.score_samples(samples, batch_size)
    coarse_scores = coarse_scorer.score_samples(samples, batch_size)

    for sample, recall_score, coarse_score in zip(
        samples, recall_scores, coarse_scores
    ):
        sample["recall_score"] = recall_score
        sample["coarse_score"] = coarse_score


def build_model(feature_info, device):
    model = MMoERanker(
        user_count=len(feature_info["user_id_to_index"]),
        movie_count=len(feature_info["movie_id_to_index"]),
        gender_count=feature_info["gender_count"],
        age_count=feature_info["age_count"],
        occupation_count=feature_info["occupation_count"],
        genre_count=feature_info["genre_count"],
    )

    return model.to(device)


def save_checkpoint(model, feature_info, model_path, extra_info=None):
    FINE_RANK_MODEL_DIR.mkdir(exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_info": feature_info,
    }

    if extra_info is not None:
        checkpoint["extra_info"] = extra_info

    torch.save(checkpoint, model_path)


def train_model(epochs=3, batch_size=1024, score_batch_size=4096, learning_rate=0.001):
    device = get_device()
    print(f"当前训练设备: {device}")

    feature_info = build_feature_info()
    train_samples = load_base_samples(TRAIN_RATINGS_PATH, feature_info)
    test_samples = load_base_samples(TEST_RATINGS_PATH, feature_info)

    recall_scorer = RecallScorer(device)
    coarse_scorer = CoarseScorer(device)

    print("正在生成训练集 recall_score 和 coarse_score...")
    attach_model_scores(train_samples, recall_scorer, coarse_scorer, score_batch_size)

    print("正在生成测试集 recall_score 和 coarse_score...")
    attach_model_scores(test_samples, recall_scorer, coarse_scorer, score_batch_size)

    train_dataset = MMoEDataset(train_samples)
    test_dataset = MMoEDataset(test_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(feature_info, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
        )
        test_metrics = evaluate(
            model=model,
            dataloader=test_dataloader,
            device=device,
        )

        print(
            f"epoch={epoch + 1} "
            f"train_total_loss={train_metrics['total_loss']:.4f} "
            f"test_total_loss={test_metrics['total_loss']:.4f} "
            f"test_like_acc={test_metrics['like_accuracy']:.4f} "
            f"test_high_acc={test_metrics['high_rating_accuracy']:.4f}"
        )

        epoch_model_path = FINE_RANK_MODEL_DIR / f"mmoe_epoch_{epoch + 1}.pt"
        save_checkpoint(
            model=model,
            feature_info=feature_info,
            model_path=epoch_model_path,
            extra_info={
                "epoch": epoch + 1,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            },
        )

    save_checkpoint(
        model=model,
        feature_info=feature_info,
        model_path=FINE_RANK_MODEL_PATH,
        extra_info={"epoch": epochs},
    )
    print(f"模型已保存到: {FINE_RANK_MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="训练 MMoE 精排模型")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        score_batch_size=args.score_batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
