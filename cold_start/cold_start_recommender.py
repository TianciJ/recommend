import math
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_USERS_PATH = BASE_DIR / "train_data" / "users.dat"
TRAIN_RATINGS_PATH = BASE_DIR / "train_data" / "ratings.dat"
MOVIES_PATH = BASE_DIR / "data" / "movies.dat"


class ColdStartRecommender:
    def __init__(
        self,
        users_path=TRAIN_USERS_PATH,
        ratings_path=TRAIN_RATINGS_PATH,
        movies_path=MOVIES_PATH,
        user_profiles=None,
        ratings=None,
        movies=None,
    ):
        self.user_profiles = (
            normalize_user_profiles(user_profiles)
            if user_profiles is not None
            else load_user_profiles(users_path)
        )
        self.movie_info = (
            normalize_movie_info(movies)
            if movies is not None
            else load_movie_info(movies_path)
        )
        self.movie_stats = {}
        self.segment_movie_counts = {}
        self.age_movie_counts = {}
        self.occupation_movie_counts = {}
        self.global_movie_counts = Counter()

        self._build_rating_statistics(ratings_path=ratings_path, ratings=ratings)
        self.max_rating_count_log = self._get_max_rating_count_log()

    def recommend(self, user_id, age=None, occupation=None, top_k=20):
        candidate_scores = {}
        age_key = normalize_profile_value(age)
        occupation_key = normalize_profile_value(occupation)

        if age_key is not None and occupation_key is not None:
            segment_counts = self.segment_movie_counts.get((age_key, occupation_key))
            self._merge_candidates(
                candidate_scores=candidate_scores,
                source_counts=segment_counts,
                source_name="age_occupation",
                top_k=top_k,
            )

        if len(candidate_scores) < top_k and age_key is not None:
            age_counts = self.age_movie_counts.get(age_key)
            self._merge_candidates(
                candidate_scores=candidate_scores,
                source_counts=age_counts,
                source_name="age",
                top_k=top_k,
            )

        if len(candidate_scores) < top_k and occupation_key is not None:
            occupation_counts = self.occupation_movie_counts.get(occupation_key)
            self._merge_candidates(
                candidate_scores=candidate_scores,
                source_counts=occupation_counts,
                source_name="occupation",
                top_k=top_k,
            )

        if len(candidate_scores) < top_k:
            self._merge_candidates(
                candidate_scores=candidate_scores,
                source_counts=self.global_movie_counts,
                source_name="global",
                top_k=top_k,
            )

        ranked_items = sorted(
            candidate_scores.values(),
            key=lambda item: (item["cold_start_score"], item["movie_id"]),
            reverse=True,
        )
        return diversify_by_primary_genre(ranked_items, top_k)

    def _build_rating_statistics(self, ratings_path, ratings=None):
        for rating_row in iter_rating_rows(ratings_path=ratings_path, ratings=ratings):
            user_id = int(rating_row["user_id"])
            movie_id = int(rating_row["movie_id"])
            rating = int(rating_row["rating"])

            if movie_id not in self.movie_info:
                continue

            self._add_movie_rating(movie_id, rating)

            if rating < 4:
                continue

            self.global_movie_counts[movie_id] += 1
            user_profile = self.user_profiles.get(user_id)

            if user_profile is None:
                continue

            age = user_profile["age"]
            occupation = user_profile["occupation"]
            segment_key = (age, occupation)

            self.segment_movie_counts.setdefault(segment_key, Counter())[movie_id] += 1
            self.age_movie_counts.setdefault(age, Counter())[movie_id] += 1
            self.occupation_movie_counts.setdefault(occupation, Counter())[movie_id] += 1

        for movie_id, stats in self.movie_stats.items():
            stats["avg_rating"] = stats["rating_sum"] / stats["rating_count"]

    def _add_movie_rating(self, movie_id, rating):
        if movie_id not in self.movie_stats:
            self.movie_stats[movie_id] = {
                "rating_sum": 0,
                "rating_count": 0,
                "avg_rating": 0,
            }

        self.movie_stats[movie_id]["rating_sum"] += rating
        self.movie_stats[movie_id]["rating_count"] += 1

    def _get_max_rating_count_log(self):
        if not self.movie_stats:
            return 1

        return max(
            math.log1p(stats["rating_count"])
            for stats in self.movie_stats.values()
        ) or 1

    def _merge_candidates(self, candidate_scores, source_counts, source_name, top_k):
        if not source_counts:
            return

        max_source_count = max(source_counts.values()) or 1
        candidate_limit = max(top_k * 5, top_k)
        top_candidates = source_counts.most_common(candidate_limit)

        for movie_id, positive_count in top_candidates:
            if movie_id in candidate_scores:
                continue

            movie_stats = self.movie_stats.get(movie_id)
            movie_info = self.movie_info.get(movie_id)

            if movie_stats is None or movie_info is None:
                continue

            segment_positive_score = positive_count / max_source_count
            movie_avg_rating_score = movie_stats["avg_rating"] / 5
            movie_popularity_score = (
                math.log1p(movie_stats["rating_count"]) / self.max_rating_count_log
            )
            cold_start_score = (
                0.45 * segment_positive_score
                + 0.35 * movie_avg_rating_score
                + 0.20 * movie_popularity_score
            )
            primary_genre = get_primary_genre(movie_info["genres"])

            candidate_scores[movie_id] = {
                "item_id": movie_id,
                "movie_id": movie_id,
                "title": movie_info["title"],
                "cold_start_score": cold_start_score,
                "recall_score": cold_start_score,
                "recall_source": "cold_start",
                "cold_start_source": source_name,
                "rerank_primary_genre": primary_genre,
            }


def normalize_profile_value(value):
    return str(value) if value is not None else None


def load_user_profiles(users_path):
    user_profiles = {}

    with users_path.open("r", encoding="utf-8") as users_file:
        for line in users_file:
            user_id, gender, age, occupation, zip_code = line.strip().split("::")
            user_profiles[int(user_id)] = {
                "age": normalize_profile_value(age),
                "occupation": normalize_profile_value(occupation),
            }

    return user_profiles


def normalize_user_profiles(user_profiles):
    return {
        int(user_id): {
            "age": normalize_profile_value(profile.get("age")),
            "occupation": normalize_profile_value(profile.get("occupation")),
        }
        for user_id, profile in user_profiles.items()
    }


def iter_rating_rows(ratings_path, ratings=None):
    if ratings is not None:
        for rating in ratings:
            yield rating
        return

    with ratings_path.open("r", encoding="utf-8") as ratings_file:
        for line in ratings_file:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            yield {
                "user_id": int(user_id),
                "movie_id": int(movie_id),
                "rating": int(rating),
                "timestamp": int(timestamp),
            }


def load_movie_info(movies_path):
    movie_info = {}

    with movies_path.open("r", encoding="latin-1") as movies_file:
        for line in movies_file:
            movie_id, title, genres = line.strip().split("::")
            movie_info[int(movie_id)] = {
                "title": title,
                "genres": genres.split("|"),
            }

    return movie_info


def normalize_movie_info(movies):
    return {
        int(movie["movie_id"]): {
            "title": movie["title"],
            "genres": list(movie["genres"]),
        }
        for movie in movies
    }


def get_primary_genre(genres):
    return genres[0] if genres else "Unknown"


def diversify_by_primary_genre(ranked_items, top_k):
    selected_items = []
    remaining_items = list(ranked_items)
    last_genre = None

    while remaining_items and len(selected_items) < top_k:
        chosen_index = find_next_different_genre(remaining_items, last_genre)

        if chosen_index is None:
            chosen_index = 0

        chosen_item = remaining_items.pop(chosen_index)
        selected_items.append(chosen_item)
        last_genre = chosen_item.get("rerank_primary_genre", "Unknown")

    return selected_items


def find_next_different_genre(items, last_genre):
    if last_genre is None:
        return 0

    for index, item in enumerate(items):
        if item.get("rerank_primary_genre") != last_genre:
            return index

    return None
