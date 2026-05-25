from pathlib import Path

import torch

from .mmoe_ranker import MMoERanker
from .mmoe_ranker import get_device


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "fine_rank_model" / "mmoe_epoch_6.pt"


def load_checkpoint(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)


def build_model_from_checkpoint(checkpoint, device):
    feature_info = checkpoint["feature_info"]

    model = MMoERanker(
        user_count=len(feature_info["user_id_to_index"]),
        movie_count=len(feature_info["movie_id_to_index"]),
        gender_count=feature_info["gender_count"],
        age_count=feature_info["age_count"],
        occupation_count=feature_info["occupation_count"],
        genre_count=feature_info["genre_count"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


class MMoEFineRanker:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, score_name="like"):
        self.device = get_device()
        self.checkpoint = load_checkpoint(model_path, self.device)
        self.feature_info = self.checkpoint["feature_info"]
        self.model = build_model_from_checkpoint(self.checkpoint, self.device)
        self.score_name = score_name

    def rank(self, user_id, candidates, top_k=50):
        user_id_to_index = self.feature_info["user_id_to_index"]
        movie_id_to_index = self.feature_info["movie_id_to_index"]
        user_features = self.feature_info["user_features"]
        movie_features = self.feature_info["movie_features"]

        if user_id not in user_id_to_index:
            return []

        valid_items = []
        for item in candidates:
            movie_id = item.get("movie_id", item.get("item_id"))

            if movie_id in movie_id_to_index:
                valid_items.append(item)

        if not valid_items:
            return []

        user_feature = user_features[user_id]
        user_index = user_id_to_index[user_id]

        user_indexes = []
        gender_indexes = []
        age_indexes = []
        occupation_indexes = []
        movie_indexes = []
        genre_indexes = []
        recall_scores = []
        coarse_scores = []

        for item in valid_items:
            movie_id = item.get("movie_id", item.get("item_id"))

            user_indexes.append(user_index)
            gender_indexes.append(user_feature["gender"])
            age_indexes.append(user_feature["age"])
            occupation_indexes.append(user_feature["occupation"])
            movie_indexes.append(movie_id_to_index[movie_id])
            genre_indexes.append(movie_features[movie_id]["genres"])
            recall_scores.append(item.get("recall_score", 0.0))
            coarse_scores.append(item.get("rough_rank_score", 0.0))

        with torch.no_grad():
            outputs = self.model(
                user_id=torch.tensor(
                    user_indexes, dtype=torch.long, device=self.device
                ),
                gender=torch.tensor(
                    gender_indexes, dtype=torch.long, device=self.device
                ),
                age=torch.tensor(age_indexes, dtype=torch.long, device=self.device),
                occupation=torch.tensor(
                    occupation_indexes, dtype=torch.long, device=self.device
                ),
                movie_id=torch.tensor(
                    movie_indexes, dtype=torch.long, device=self.device
                ),
                genres=torch.tensor(
                    genre_indexes, dtype=torch.long, device=self.device
                ),
                recall_score=torch.tensor(
                    recall_scores, dtype=torch.float, device=self.device
                ),
                coarse_score=torch.tensor(
                    coarse_scores, dtype=torch.float, device=self.device
                ),
            )

            if self.score_name == "like":
                scores = torch.sigmoid(outputs["like_logit"])
            elif self.score_name == "high_rating":
                scores = torch.sigmoid(outputs["high_rating_logit"])
            elif self.score_name == "rating":
                scores = outputs["rating_pred"]
            else:
                raise ValueError("score_name must be like, high_rating, or rating")

            scores = scores.cpu().tolist()

        ranked_items = []
        for item, score in zip(valid_items, scores):
            ranked_items.append(
                {
                    **item,
                    "fine_rank_score": score,
                    "fine_rank_source": f"mmoe_epoch_6_{self.score_name}",
                }
            )

        ranked_items.sort(key=lambda item: item["fine_rank_score"], reverse=True)
        return ranked_items[:top_k]
