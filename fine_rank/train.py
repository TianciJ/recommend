import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .model import MMoERanker, evaluate, get_device, train_one_epoch
from recall.two_tower import build_model_from_checkpoint as build_recall_model
from recall.two_tower import load_movies_from_dat, load_mysql_dataset_if_configured, load_ratings_from_dat, load_users_from_dat
from rough_rank.inference import build_dense_features, build_model_from_checkpoint as build_rough_model

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"
TRAIN_RATINGS_PATH = TRAIN_DIR / "ratings.dat"
TEST_RATINGS_PATH = TEST_DIR / "ratings.dat"
USERS_PATH = TRAIN_DIR / "users.dat"
MOVIES_PATH = TRAIN_DIR / "movies.dat"

RECALL_MODEL_PATH = BASE_DIR / "models" / "recall" / "two_tower.pt"
ROUGH_MODEL_PATH = BASE_DIR / "models" / "rough_rank" / "three_tower.pt"
FINE_RANK_MODEL_DIR = BASE_DIR / "models" / "fine_rank"
FINE_RANK_MODEL_PATH = FINE_RANK_MODEL_DIR / "mmoe.pt"


class MMoEDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "user_id": torch.tensor(s["user_id"], dtype=torch.long),
            "gender": torch.tensor(s["gender"], dtype=torch.long),
            "age": torch.tensor(s["age"], dtype=torch.long),
            "occupation": torch.tensor(s["occupation"], dtype=torch.long),
            "movie_id": torch.tensor(s["movie_id"], dtype=torch.long),
            "genres": torch.tensor(s["genres"], dtype=torch.long),
            "recall_score": torch.tensor(s["recall_score"], dtype=torch.float),
            "coarse_score": torch.tensor(s["coarse_score"], dtype=torch.float),
            "like_label": torch.tensor(s["like_label"], dtype=torch.float),
            "high_rating_label": torch.tensor(s["high_rating_label"], dtype=torch.float),
            "rating_label": torch.tensor(s["rating_label"], dtype=torch.float),
        }


def load_checkpoint(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)


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
            "gender": gender_to_index[gender],
            "age": age_to_index[age],
            "occupation": occupation_to_index[occ],
        }

    return user_features, gender_to_index, age_to_index, occupation_to_index


def load_movie_features(movies_path=MOVIES_PATH, movies=None):
    if movies is None:
        movies = load_movies_from_dat(movies_path)

    genre_to_index, movie_features = {}, {}
    max_genre_length = 0

    for movie in movies:
        mid = int(movie["movie_id"])
        idxs = []
        for g in movie["genres"]:
            if g not in genre_to_index:
                genre_to_index[g] = len(genre_to_index) + 1  # 0 留给 padding
            idxs.append(genre_to_index[g])
        max_genre_length = max(max_genre_length, len(idxs))
        movie_features[mid] = {"genres": idxs}

    for mid in movie_features:
        g = movie_features[mid]["genres"]
        movie_features[mid]["genres"] = g + [0] * (max_genre_length - len(g))

    return movie_features, genre_to_index, max_genre_length


def build_feature_info(users=None, movies=None, ratings=None):
    if users is None and movies is None and ratings is None:
        mysql_dataset = load_mysql_dataset_if_configured(split="train")
        if mysql_dataset is not None:
            users, movies, ratings = mysql_dataset["users"], mysql_dataset["movies"], mysql_dataset["ratings"]

    if ratings is None:
        ratings = load_ratings_from_dat(TRAIN_RATINGS_PATH)

    user_features, gender_to_index, age_to_index, occupation_to_index = load_user_features(users=users)
    movie_features, genre_to_index, max_genre_length = load_movie_features(movies=movies)

    user_id_to_index, movie_id_to_index, index_to_movie_id = {}, {}, {}
    for row in ratings:
        uid, mid = int(row["user_id"]), int(row["movie_id"])
        if uid not in user_id_to_index:
            user_id_to_index[uid] = len(user_id_to_index)
        if mid not in movie_id_to_index:
            idx = len(movie_id_to_index)
            movie_id_to_index[mid] = idx
            index_to_movie_id[idx] = mid

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
        ratings = mysql_dataset["ratings"] if mysql_dataset else load_ratings_from_dat(ratings_path)

    uf = feature_info["user_features"]
    mf = feature_info["movie_features"]
    uid_to_idx = feature_info["user_id_to_index"]
    mid_to_idx = feature_info["movie_id_to_index"]

    samples = []
    for row in ratings:
        uid, mid, r = int(row["user_id"]), int(row["movie_id"]), int(row["rating"])
        if skip_unknown and (uid not in uid_to_idx or mid not in mid_to_idx):
            continue
        samples.append({
            "raw_user_id": uid,
            "raw_movie_id": mid,
            "user_id": uid_to_idx[uid],
            "gender": uf[uid]["gender"],
            "age": uf[uid]["age"],
            "occupation": uf[uid]["occupation"],
            "movie_id": mid_to_idx[mid],
            "genres": mf[mid]["genres"],
            "like_label": 1 if r >= 4 else 0,
            "high_rating_label": 1 if r == 5 else 0,
            "rating_label": r / 5,
        })
    return samples


def _score_batch(model, batch_samples, fi, device, extra_features_fn):
    """通用批量打分，valid_positions 对应 batch_samples 中可打分的位置。"""
    uid_to_idx = fi["user_id_to_index"]
    mid_to_idx = fi["movie_id_to_index"]
    uf = fi["user_features"]
    mf = fi["movie_features"]

    rows, positions = [], []
    for pos, s in enumerate(batch_samples):
        uid, mid = s["raw_user_id"], s["raw_movie_id"]
        if uid in uid_to_idx and mid in mid_to_idx:
            rows.append((uid, mid, uf[uid], mf[mid]))
            positions.append(pos)

    batch_scores = [0.0] * len(batch_samples)
    if not rows:
        return batch_scores

    t = lambda vals, dtype: torch.tensor(vals, dtype=dtype, device=device)
    outputs = model(*extra_features_fn(rows, fi, t, device)).cpu().tolist()
    for pos, score in zip(positions, outputs):
        batch_scores[pos] = score
    return batch_scores


class RecallScorer:
    def __init__(self, device):
        cp = load_checkpoint(RECALL_MODEL_PATH, device)
        self.fi = cp["feature_info"]
        self.model = build_recall_model(cp, device)
        self.device = device

    def score_samples(self, samples, batch_size):
        fi = self.fi
        scores = []
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                chunk = samples[start:start + batch_size]
                scores.extend(_score_batch(self.model, chunk, fi, self.device, self._features))
        return scores

    def _features(self, rows, fi, t, device):
        uid_to_idx, mid_to_idx = fi["user_id_to_index"], fi["movie_id_to_index"]
        return (
            t([uid_to_idx[r[0]] for r in rows], torch.long),
            t([r[2]["gender_index"] for r in rows], torch.long),
            t([r[2]["age_index"] for r in rows], torch.long),
            t([r[2]["occupation_index"] for r in rows], torch.long),
            t([mid_to_idx[r[1]] for r in rows], torch.long),
            t([r[3]["genre_vector"] for r in rows], torch.float),
        )


class CoarseScorer:
    def __init__(self, device):
        cp = load_checkpoint(ROUGH_MODEL_PATH, device)
        self.fi = cp["feature_info"]
        self.model = build_rough_model(cp, device)
        self.device = device

    def score_samples(self, samples, batch_size):
        fi = self.fi
        scores = []
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                chunk = samples[start:start + batch_size]
                scores.extend(_score_batch(self.model, chunk, fi, self.device, self._features))
        return scores

    def _features(self, rows, fi, t, device):
        uid_to_idx, mid_to_idx = fi["user_id_to_index"], fi["movie_id_to_index"]
        rs = fi["rating_stats"]
        return (
            t([uid_to_idx[r[0]] for r in rows], torch.long),
            t([r[2]["gender_index"] for r in rows], torch.long),
            t([r[2]["age_index"] for r in rows], torch.long),
            t([r[2]["occupation_index"] for r in rows], torch.long),
            t([mid_to_idx[r[1]] for r in rows], torch.long),
            t([r[3]["genre_vector"] for r in rows], torch.float),
            t([build_dense_features(r[0], r[1], rs) for r in rows], torch.float),
        )


def attach_model_scores(samples, recall_scorer, coarse_scorer, batch_size):
    recall_scores = recall_scorer.score_samples(samples, batch_size)
    coarse_scores = coarse_scorer.score_samples(samples, batch_size)
    for s, rs, cs in zip(samples, recall_scores, coarse_scores):
        s["recall_score"] = rs
        s["coarse_score"] = cs


def build_model(feature_info, device):
    fi = feature_info
    return MMoERanker(
        user_count=len(fi["user_id_to_index"]),
        movie_count=len(fi["movie_id_to_index"]),
        gender_count=fi["gender_count"],
        age_count=fi["age_count"],
        occupation_count=fi["occupation_count"],
        genre_count=fi["genre_count"],
    ).to(device)


def save_checkpoint(model, feature_info, model_path, extra_info=None):
    FINE_RANK_MODEL_DIR.mkdir(exist_ok=True)
    cp = {"model_state_dict": model.state_dict(), "feature_info": feature_info}
    if extra_info is not None:
        cp["extra_info"] = extra_info
    torch.save(cp, model_path)


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

    train_loader = DataLoader(MMoEDataset(train_samples), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MMoEDataset(test_samples), batch_size=batch_size, shuffle=False)

    model = build_model(feature_info, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        tm = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, device=device)
        vm = evaluate(model=model, dataloader=test_loader, device=device)
        print(f"epoch={epoch + 1} train_loss={tm['total_loss']:.4f} test_loss={vm['total_loss']:.4f} "
              f"test_like_acc={vm['like_accuracy']:.4f} test_high_acc={vm['high_rating_accuracy']:.4f}")

        save_checkpoint(model, feature_info, FINE_RANK_MODEL_DIR / f"mmoe_epoch_{epoch + 1}.pt",
                        extra_info={"epoch": epoch + 1, "train_metrics": tm, "test_metrics": vm})

    save_checkpoint(model, feature_info, FINE_RANK_MODEL_PATH, extra_info={"epoch": epochs})
    print(f"模型已保存到: {FINE_RANK_MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, score_batch_size=args.score_batch_size, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
