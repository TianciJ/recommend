from pathlib import Path

import torch

from .model import ThreeTowerRoughRankModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "rough_rank"
MODEL_PATH = MODEL_DIR / "three_tower.pt"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(model_path=MODEL_PATH, device=None):
    if device is None:
        device = get_device()

    return torch.load(model_path, map_location=device, weights_only=False)


def build_model_from_checkpoint(checkpoint, device):
    feature_info = checkpoint["feature_info"]

    model = ThreeTowerRoughRankModel(
        user_count=len(feature_info["user_id_to_index"]),
        movie_count=len(feature_info["movie_id_to_index"]),
        gender_count=feature_info["gender_count"],
        age_count=feature_info["age_count"],
        occupation_count=feature_info["occupation_count"],
        genre_count=feature_info["genre_count"],
        dense_feature_dim=feature_info["dense_feature_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


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


class RoughRanker:
    def __init__(self, model_path=MODEL_PATH):
        self.device = get_device()
        self.checkpoint = load_checkpoint(model_path=model_path, device=self.device)
        self.feature_info = self.checkpoint["feature_info"]
        self.model = build_model_from_checkpoint(self.checkpoint, self.device)

    def rank(self, user_id, recalled_items, top_k=100):
        user_id_to_index = self.feature_info["user_id_to_index"]
        movie_id_to_index = self.feature_info["movie_id_to_index"]
        user_features = self.feature_info["user_features"]
        movie_features = self.feature_info["movie_features"]
        rating_stats = self.feature_info["rating_stats"]

        if user_id not in user_id_to_index:
            return []

        valid_items = []
        for item in recalled_items:
            movie_id = item.get("movie_id", item.get("item_id"))

            if movie_id in movie_id_to_index:
                valid_items.append(item)

        if not valid_items:
            return []

        user_index = user_id_to_index[user_id]
        user_feature = user_features[user_id]

        user_indexes = []
        gender_indexes = []
        age_indexes = []
        occupation_indexes = []
        movie_indexes = []
        genre_vectors = []
        dense_features = []

        for item in valid_items:
            movie_id = item.get("movie_id", item.get("item_id"))

            user_indexes.append(user_index)
            gender_indexes.append(user_feature["gender_index"])
            age_indexes.append(user_feature["age_index"])
            occupation_indexes.append(user_feature["occupation_index"])
            movie_indexes.append(movie_id_to_index[movie_id])
            genre_vectors.append(movie_features[movie_id]["genre_vector"])
            dense_features.append(
                build_dense_features(
                    user_id=user_id,
                    movie_id=movie_id,
                    rating_stats=rating_stats,
                )
            )

        with torch.no_grad():
            scores = self.model(
                torch.tensor(user_indexes, dtype=torch.long, device=self.device),
                torch.tensor(gender_indexes, dtype=torch.long, device=self.device),
                torch.tensor(age_indexes, dtype=torch.long, device=self.device),
                torch.tensor(occupation_indexes, dtype=torch.long, device=self.device),
                torch.tensor(movie_indexes, dtype=torch.long, device=self.device),
                torch.tensor(genre_vectors, dtype=torch.float, device=self.device),
                torch.tensor(dense_features, dtype=torch.float, device=self.device),
            )
            scores = scores.cpu().tolist()

        ranked_items = []
        for item, score in zip(valid_items, scores):
            ranked_items.append(
                {
                    **item,
                    "rough_rank_score": score,
                }
            )

        ranked_items.sort(key=lambda item: item["rough_rank_score"], reverse=True)
        return ranked_items[:top_k]


def rough_rank(user_id, recalled_items, top_k=100, model_path=MODEL_PATH):
    ranker = RoughRanker(model_path=model_path)
    return ranker.rank(user_id=user_id, recalled_items=recalled_items, top_k=top_k)
