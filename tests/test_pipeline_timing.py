import unittest

from recommender_pipeline import RecommenderPipeline
from recommender_pipeline import format_recommendation_line
from recommender_pipeline import recommend_for_user_id_or_register
from recommender_pipeline import load_movie_genres
from recommender_pipeline import load_user_seen_movies


class FakeRecaller:
    def recommend(self, user_id, top_k):
        return [
            {"movie_id": index, "score": 1.0 / index, "title": f"movie-{index}"}
            for index in range(1, top_k + 1)
        ]


class FakeEmptyRecaller:
    def recommend(self, user_id, top_k):
        return []


class FakeRoughRanker:
    def rank(self, user_id, recalled_items, top_k):
        return [
            {
                **item,
                "rough_rank_score": item["recall_score"] + 1,
            }
            for item in recalled_items[:top_k]
        ]


class FakeFineRanker:
    def rank(self, user_id, candidates, top_k):
        return [
            {
                **item,
                "fine_rank_score": item["rough_rank_score"] + 1,
            }
            for item in candidates[:top_k]
        ]


class FakeReranker:
    def rerank(self, user_id, ranked_items, top_k):
        return [
            {
                **item,
                "rerank_primary_genre": "Drama",
            }
            for item in ranked_items[:top_k]
        ]


class FakeColdStartRecommender:
    def recommend(self, user_id, age=None, occupation=None, top_k=20):
        return [
            {
                "item_id": index,
                "movie_id": index,
                "title": f"cold-movie-{index}",
                "cold_start_score": 1.0 / index,
                "recall_score": 1.0 / index,
                "recall_source": "cold_start",
                "cold_start_source": "age_occupation",
                "rerank_primary_genre": "Drama",
                "age": age,
                "occupation": occupation,
            }
            for index in range(1, top_k + 1)
        ]


class RankerShouldNotRun:
    def rank(self, *args, **kwargs):
        raise AssertionError("ranker should not run when recall is empty")


class FakeUserProfileRepository:
    def __init__(self, profile):
        self.profile = profile
        self.requested_user_ids = []

    def get_user_profile(self, user_id):
        self.requested_user_ids.append(user_id)
        return self.profile


class RaisingUserProfileRepository:
    def get_user_profile(self, user_id):
        raise RuntimeError("database unavailable")


class FakeRegistrationRepository:
    def __init__(self, profile=None, new_user_id=900123):
        self.profile = profile
        self.new_user_id = new_user_id
        self.created_users = []
        self.requested_user_ids = []

    def get_user_profile(self, user_id):
        self.requested_user_ids.append(user_id)
        return self.profile

    def create_user(self, username, age, occupation):
        self.created_users.append(
            {
                "username": username,
                "age": age,
                "occupation": occupation,
            }
        )
        return self.new_user_id


class FakeInteractivePipeline:
    def __init__(self):
        self.recommend_calls = []
        self.cold_start_calls = []

    def recommend(self, user_id, top_k=20):
        self.recommend_calls.append({"user_id": user_id, "top_k": top_k})
        return [{"movie_id": 10, "recall_source": "two_tower"}]

    def cold_start(self, user_id, age=None, occupation=None, top_k=20):
        self.cold_start_calls.append(
            {
                "user_id": user_id,
                "age": age,
                "occupation": occupation,
                "top_k": top_k,
            }
        )
        return [{"movie_id": 20, "recall_source": "cold_start"}]


class PipelineTimingTest(unittest.TestCase):
    def build_pipeline(self):
        pipeline = RecommenderPipeline.__new__(RecommenderPipeline)
        pipeline.recaller = FakeRecaller()
        pipeline.rough_ranker = FakeRoughRanker()
        pipeline.fine_ranker = FakeFineRanker()
        pipeline.reranker = FakeReranker()
        pipeline.cold_start_recommender = FakeColdStartRecommender()
        pipeline.user_profile_repository = None
        return pipeline

    def build_cold_start_pipeline(self, user_profile_repository=None):
        pipeline = RecommenderPipeline.__new__(RecommenderPipeline)
        pipeline.recaller = FakeEmptyRecaller()
        pipeline.rough_ranker = RankerShouldNotRun()
        pipeline.fine_ranker = RankerShouldNotRun()
        pipeline.reranker = FakeReranker()
        pipeline.cold_start_recommender = FakeColdStartRecommender()
        pipeline.user_profile_repository = user_profile_repository
        return pipeline

    def test_recommend_with_timing_returns_recommendations_and_stage_metadata(self):
        pipeline = self.build_pipeline()

        recommendations, timing = pipeline.recommend_with_timing(
            user_id=1,
            top_k=2,
            recall_size=5,
            rough_rank_size=4,
            fine_rank_size=3,
        )

        self.assertEqual([item["movie_id"] for item in recommendations], [1, 2])
        self.assertIn("total_ms", timing)
        self.assertGreaterEqual(timing["total_ms"], 0)

        expected_counts = {
            "recall": 5,
            "rough_rank": 4,
            "fine_rank": 3,
            "rerank": 2,
        }
        self.assertEqual(set(timing["stages"]), set(expected_counts))

        for stage_name, expected_count in expected_counts.items():
            stage_timing = timing["stages"][stage_name]
            self.assertEqual(stage_timing["item_count"], expected_count)
            self.assertGreaterEqual(stage_timing["elapsed_ms"], 0)

    def test_recommend_keeps_original_return_shape(self):
        pipeline = self.build_pipeline()

        recommendations = pipeline.recommend(
            user_id=1,
            top_k=2,
            recall_size=5,
            rough_rank_size=4,
            fine_rank_size=3,
        )

        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 2)
        self.assertNotIsInstance(recommendations, tuple)

    def test_recommend_uses_cold_start_when_recall_is_empty(self):
        pipeline = self.build_cold_start_pipeline()

        recommendations = pipeline.recommend(
            user_id=900001,
            age=25,
            occupation=4,
            top_k=3,
        )

        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0]["recall_source"], "cold_start")
        self.assertEqual(recommendations[0]["age"], 25)
        self.assertEqual(recommendations[0]["occupation"], 4)

    def test_recommend_with_timing_records_cold_start_stage(self):
        pipeline = self.build_cold_start_pipeline()

        recommendations, timing = pipeline.recommend_with_timing(
            user_id=900001,
            age=25,
            occupation=4,
            top_k=3,
        )

        self.assertEqual(len(recommendations), 3)
        self.assertEqual(timing["stages"]["recall"]["item_count"], 0)
        self.assertEqual(timing["stages"]["cold_start"]["item_count"], 3)
        self.assertNotIn("rough_rank", timing["stages"])
        self.assertNotIn("fine_rank", timing["stages"])

    def test_recommend_uses_repository_profile_for_cold_start(self):
        repository = FakeUserProfileRepository(
            profile={"user_id": 900001, "age": 35, "occupation": 7}
        )
        pipeline = self.build_cold_start_pipeline(user_profile_repository=repository)

        recommendations = pipeline.recommend(user_id=900001, top_k=2)

        self.assertEqual(repository.requested_user_ids, [900001])
        self.assertEqual(recommendations[0]["age"], 35)
        self.assertEqual(recommendations[0]["occupation"], 7)

    def test_recommend_prefers_explicit_profile_over_repository_profile(self):
        repository = FakeUserProfileRepository(
            profile={"user_id": 900001, "age": 35, "occupation": 7}
        )
        pipeline = self.build_cold_start_pipeline(user_profile_repository=repository)

        recommendations = pipeline.recommend(
            user_id=900001,
            age=25,
            occupation=4,
            top_k=2,
        )

        self.assertEqual(recommendations[0]["age"], 25)
        self.assertEqual(recommendations[0]["occupation"], 4)

    def test_recommend_keeps_cold_start_when_repository_returns_none(self):
        repository = FakeUserProfileRepository(profile=None)
        pipeline = self.build_cold_start_pipeline(user_profile_repository=repository)

        recommendations = pipeline.recommend(user_id=900001, top_k=2)

        self.assertEqual(len(recommendations), 2)
        self.assertIsNone(recommendations[0]["age"])
        self.assertIsNone(recommendations[0]["occupation"])

    def test_recommend_keeps_cold_start_when_repository_raises(self):
        pipeline = self.build_cold_start_pipeline(
            user_profile_repository=RaisingUserProfileRepository()
        )

        recommendations = pipeline.recommend(user_id=900001, top_k=2)

        self.assertEqual(len(recommendations), 2)
        self.assertIsNone(recommendations[0]["age"])
        self.assertIsNone(recommendations[0]["occupation"])

    def test_format_recommendation_line_handles_cold_start_item(self):
        item = {
            "movie_id": 1,
            "title": "cold-movie-1",
            "rerank_primary_genre": "Drama",
            "cold_start_score": 1.0,
            "recall_score": 1.0,
            "recall_source": "cold_start",
        }

        line = format_recommendation_line(1, item)

        self.assertIn("cold_start_score=1.0000", line)
        self.assertIn("rough_rank_score=-", line)
        self.assertIn("fine_rank_score=-", line)

    def test_load_user_seen_movies_can_use_mysql_rating_rows(self):
        ratings = [
            {"user_id": 1, "movie_id": 10, "rating": 5, "timestamp": 100},
            {"user_id": 1, "movie_id": 20, "rating": 3, "timestamp": 200},
        ]

        self.assertEqual(load_user_seen_movies(ratings=ratings), {1: {10, 20}})

    def test_load_movie_genres_can_use_mysql_movie_rows(self):
        movies = [
            {"movie_id": 10, "title": "Movie A", "genres": ["Drama", "Comedy"]},
        ]

        self.assertEqual(load_movie_genres(movies=movies), {10: ["Drama", "Comedy"]})

    def test_interactive_existing_user_uses_full_recommend_flow(self):
        repository = FakeRegistrationRepository(
            profile={"user_id": 100, "age": 25, "occupation": 4}
        )
        pipeline = FakeInteractivePipeline()
        prompts = []

        recommendations = recommend_for_user_id_or_register(
            user_id=100,
            user_profile_repository=repository,
            pipeline=pipeline,
            input_func=lambda prompt: prompts.append(prompt) or "",
            output_func=lambda message: None,
            top_k=5,
        )

        self.assertEqual(recommendations, [{"movie_id": 10, "recall_source": "two_tower"}])
        self.assertEqual(pipeline.recommend_calls, [{"user_id": 100, "top_k": 5}])
        self.assertEqual(pipeline.cold_start_calls, [])
        self.assertEqual(repository.created_users, [])
        self.assertEqual(prompts, [])

    def test_interactive_missing_user_registers_then_uses_cold_start(self):
        repository = FakeRegistrationRepository(profile=None, new_user_id=900123)
        pipeline = FakeInteractivePipeline()
        answers = iter(["alice", "25", "4"])

        recommendations = recommend_for_user_id_or_register(
            user_id=999999,
            user_profile_repository=repository,
            pipeline=pipeline,
            input_func=lambda prompt: next(answers),
            output_func=lambda message: None,
            top_k=3,
        )

        self.assertEqual(recommendations, [{"movie_id": 20, "recall_source": "cold_start"}])
        self.assertEqual(
            repository.created_users,
            [{"username": "alice", "age": 25, "occupation": 4}],
        )
        self.assertEqual(pipeline.recommend_calls, [])
        self.assertEqual(
            pipeline.cold_start_calls,
            [{"user_id": 900123, "age": 25, "occupation": 4, "top_k": 3}],
        )

    def test_interactive_without_repository_uses_existing_recommend_flow(self):
        pipeline = FakeInteractivePipeline()
        outputs = []

        recommendations = recommend_for_user_id_or_register(
            user_id=15857,
            user_profile_repository=None,
            pipeline=pipeline,
            input_func=lambda prompt: self.fail("input should not be requested"),
            output_func=outputs.append,
            top_k=4,
        )

        self.assertEqual(recommendations, [{"movie_id": 10, "recall_source": "two_tower"}])
        self.assertEqual(pipeline.recommend_calls, [{"user_id": 15857, "top_k": 4}])
        self.assertEqual(pipeline.cold_start_calls, [])
        self.assertEqual(
            outputs,
            ["MySQL user repository is not configured. Running recommendation fallback."],
        )


if __name__ == "__main__":
    unittest.main()
