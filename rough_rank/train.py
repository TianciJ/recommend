import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import ThreeTowerRoughRankModel
from recall.two_tower import load_movies_from_dat, load_mysql_dataset_if_configured, load_ratings_from_dat, load_users_from_dat

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"
TEST_RATINGS_PATH = TEST_DIR / "ratings.dat"
USERS_PATH = TRAIN_DIR / "users.dat"
MOVIES_PATH = TRAIN_DIR / "movies.dat"
MODEL_DIR = BASE_DIR / "models" / "rough_rank"
MODEL_PATH = MODEL_DIR / "three_tower.pt"

DENSE_FEATURE_DIM = 5  # user_avg_rating, user_count, movie_avg_rating, movie_count, recall_score


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoughRankDataset(Dataset):
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
            "dense_features": torch.tensor(s["dense_features"], dtype=torch.float),
            "label": torch.tensor(s["label"], dtype=torch.float),
        }


def load_user_features(users_path=USERS_PATH, users=None):
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
        }

    return user_features, gender_to_index, age_to_index, occupation_to_index


def load_movie_features(movies_path=MOVIES_PATH, movies=None):
    if movies is None:
        movies = load_movies_from_dat(movies_path)

    genre_to_index, movie_genres = {}, {}
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


def build_rating_stats(ratings_path=TRAIN_RATINGS_PATH, ratings=None):
    if ratings is None:
        ratings = load_ratings_from_dat(ratings_path)

    user_sum, user_count, movie_sum, movie_count = {}, {}, {}, {}
    for row in ratings:
        uid, mid, r = int(row["user_id"]), int(row["movie_id"]), int(row["rating"])
        user_sum[uid] = user_sum.get(uid, 0) + r
        user_count[uid] = user_count.get(uid, 0) + 1
        movie_sum[mid] = movie_sum.get(mid, 0) + r
        movie_count[mid] = movie_count.get(mid, 0) + 1

    return {
        "user_rating_sum": user_sum,
        "user_rating_count": user_count,
        "movie_rating_sum": movie_sum,
        "movie_rating_count": movie_count,
        "max_user_count": max(user_count.values()),
        "max_movie_count": max(movie_count.values()),
    }


def build_dense_features(user_id, movie_id, rating_stats, recall_score=0.0):
    u_count = rating_stats["user_rating_count"].get(user_id, 0)
    m_count = rating_stats["movie_rating_count"].get(movie_id, 0)
    u_avg = rating_stats["user_rating_sum"][user_id] / u_count if u_count > 0 else 3
    m_avg = rating_stats["movie_rating_sum"][movie_id] / m_count if m_count > 0 else 3
    return [
        u_avg / 5,
        u_count / rating_stats["max_user_count"],
        m_avg / 5,
        m_count / rating_stats["max_movie_count"],
        (float(recall_score) + 1.0) / 2.0,  # [-1,1] → [0,1]
    ]


def build_feature_info(users=None, movies=None, ratings=None):
    if users is None and movies is None and ratings is None:
        mysql_dataset = load_mysql_dataset_if_configured(split="train")
        if mysql_dataset is not None:
            users, movies, ratings = mysql_dataset["users"], mysql_dataset["movies"], mysql_dataset["ratings"]

    if ratings is None:
        ratings = load_ratings_from_dat(TRAIN_RATINGS_PATH)

    user_features, gender_to_index, age_to_index, occupation_to_index = load_user_features(users=users)
    movie_features, genre_to_index = load_movie_features(movies=movies)
    rating_stats = build_rating_stats(ratings=ratings)

    user_id_to_index, movie_id_to_index, index_to_movie_id = {}, {}, {}
    for row in ratings:
        uid, mid, r = int(row["user_id"]), int(row["movie_id"]), int(row["rating"])
        if r == 3:
            continue
        if uid not in user_id_to_index:
            user_id_to_index[uid] = len(user_id_to_index)
        if mid not in movie_id_to_index:
            idx = len(movie_id_to_index)
            movie_id_to_index[mid] = idx
            index_to_movie_id[idx] = mid

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


def load_samples(ratings_path, feature_info, skip_unknown=True, ratings=None):
    if ratings is None:
        split = "test" if ratings_path == TEST_RATINGS_PATH else "train"
        mysql_dataset = load_mysql_dataset_if_configured(split=split)
        ratings = mysql_dataset["ratings"] if mysql_dataset else load_ratings_from_dat(ratings_path)

    uf = feature_info["user_features"]
    mf = feature_info["movie_features"]
    rs = feature_info["rating_stats"]
    uid_to_idx = feature_info["user_id_to_index"]
    mid_to_idx = feature_info["movie_id_to_index"]

    samples = []
    for row in ratings:
        uid, mid, r = int(row["user_id"]), int(row["movie_id"]), int(row["rating"])
        if r == 3:
            continue
        if skip_unknown and (uid not in uid_to_idx or mid not in mid_to_idx):
            continue
        samples.append({
            "user_index": uid_to_idx[uid],
            "gender_index": uf[uid]["gender_index"],
            "age_index": uf[uid]["age_index"],
            "occupation_index": uf[uid]["occupation_index"],
            "movie_index": mid_to_idx[mid],
            "genre_vector": mf[mid]["genre_vector"],
            "dense_features": build_dense_features(uid, mid, rs, recall_score=0.0),
            "label": 1 if r >= 4 else 0,
        })
    return samples


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def build_model(feature_info, device):
    fi = feature_info
    return ThreeTowerRoughRankModel(
        user_count=len(fi["user_id_to_index"]),
        movie_count=len(fi["movie_id_to_index"]),
        gender_count=fi["gender_count"],
        age_count=fi["age_count"],
        occupation_count=fi["occupation_count"],
        genre_count=fi["genre_count"],
        dense_feature_dim=fi["dense_feature_dim"],
    ).to(device)


def run_batch(model, batch):
    return model(
        batch["user_index"], batch["gender_index"], batch["age_index"],
        batch["occupation_index"], batch["movie_index"], batch["genre_vector"], batch["dense_features"],
    )


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            score = run_batch(model, batch)
            total_loss += loss_fn(score, batch["label"]).item()
            pred = (torch.sigmoid(score) >= 0.5).float()
            correct += (pred == batch["label"]).sum().item()
            total += batch["label"].shape[0]

    model.train()
    return total_loss / len(dataloader), correct / total


def train_model(epochs=3, batch_size=1024, learning_rate=0.001):
    device = get_device()
    print(f"当前训练设备: {device}")

    feature_info = build_feature_info()
    train_loader = DataLoader(RoughRankDataset(load_samples(TRAIN_RATINGS_PATH, feature_info)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(RoughRankDataset(load_samples(TEST_RATINGS_PATH, feature_info)), batch_size=batch_size, shuffle=False)

    model = build_model(feature_info, device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    MODEL_DIR.mkdir(exist_ok=True)

    for epoch in range(epochs):
        train_loss = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            score = run_batch(model, batch)
            loss = loss_fn(score, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} test_loss={test_loss:.4f} test_accuracy={test_acc:.4f}")

        torch.save({"model_state_dict": model.state_dict(), "feature_info": feature_info,
                    "epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss, "test_accuracy": test_acc},
                   MODEL_DIR / f"three_tower_epoch_{epoch + 1}.pt")

    torch.save({"model_state_dict": model.state_dict(), "feature_info": feature_info, "epoch": epochs}, MODEL_PATH)
    print(f"模型已保存到: {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
