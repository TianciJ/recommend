import unittest
from pathlib import Path

from cold_start.cold_start_recommender import ColdStartRecommender


class ColdStartRecommenderTest(unittest.TestCase):
    def build_recommender(self):
        base_path = Path(__file__).resolve().parent / "fixtures" / "cold_start"
        users_path = base_path / "users.dat"
        ratings_path = base_path / "ratings.dat"
        movies_path = base_path / "movies.dat"

        recommender = ColdStartRecommender(
            users_path=users_path,
            ratings_path=ratings_path,
            movies_path=movies_path,
        )
        return recommender

    def test_recommends_from_age_occupation_segment(self):
        recommender = self.build_recommender()

        recommendations = recommender.recommend(
            user_id=900001,
            age=25,
            occupation=4,
            top_k=3,
        )

        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0]["movie_id"], 10)
        self.assertEqual(recommendations[0]["recall_source"], "cold_start")
        self.assertEqual(recommendations[0]["cold_start_source"], "age_occupation")
        self.assertIn("cold_start_score", recommendations[0])
        self.assertEqual(recommendations[0]["recall_score"], recommendations[0]["cold_start_score"])
        self.assertEqual(recommendations[0]["rerank_primary_genre"], "Drama")

    def test_falls_back_to_occupation_when_age_segment_is_missing(self):
        recommender = self.build_recommender()

        recommendations = recommender.recommend(
            user_id=900002,
            age=99,
            occupation=7,
            top_k=2,
        )

        sources = {item["cold_start_source"] for item in recommendations}
        self.assertIn("occupation", sources)
        self.assertIn(recommendations[0]["movie_id"], {40, 50})

    def test_falls_back_to_global_when_profile_is_missing(self):
        recommender = self.build_recommender()

        recommendations = recommender.recommend(
            user_id=900003,
            top_k=2,
        )

        self.assertEqual(len(recommendations), 2)
        self.assertEqual(
            {item["cold_start_source"] for item in recommendations},
            {"global"},
        )

    def test_can_build_segments_from_mysql_rows_without_dat_files(self):
        base_path = Path(__file__).resolve().parent / "fixtures" / "cold_start"
        missing_users_path = base_path / "missing-users.dat"
        missing_ratings_path = base_path / "missing-ratings.dat"
        missing_movies_path = base_path / "missing-movies.dat"

        recommender = ColdStartRecommender(
            users_path=missing_users_path,
            ratings_path=missing_ratings_path,
            movies_path=missing_movies_path,
            user_profiles={
                1: {"age": "25", "occupation": "4"},
                2: {"age": "35", "occupation": "7"},
            },
            ratings=[
                {"user_id": 1, "movie_id": 10, "rating": 5, "timestamp": 100},
                {"user_id": 1, "movie_id": 10, "rating": 5, "timestamp": 101},
                {"user_id": 2, "movie_id": 20, "rating": 5, "timestamp": 200},
            ],
            movies=[
                {"movie_id": 10, "title": "Movie A", "genres": ["Drama"]},
                {"movie_id": 20, "title": "Movie B", "genres": ["Comedy"]},
            ],
        )

        recommendations = recommender.recommend(
            user_id=900001,
            age=25,
            occupation=4,
            top_k=3,
        )

        self.assertEqual(recommendations[0]["movie_id"], 10)
        self.assertEqual(recommendations[0]["cold_start_source"], "age_occupation")


if __name__ == "__main__":
    unittest.main()
