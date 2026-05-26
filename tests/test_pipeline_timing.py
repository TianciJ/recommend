import unittest

from recommender_pipeline import RecommenderPipeline


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


class PipelineTimingTest(unittest.TestCase):
    def build_pipeline(self):
        pipeline = RecommenderPipeline.__new__(RecommenderPipeline)
        pipeline.recaller = FakeRecaller()
        pipeline.rough_ranker = FakeRoughRanker()
        pipeline.fine_ranker = FakeFineRanker()
        pipeline.reranker = FakeReranker()
        pipeline.cold_start_recommender = FakeColdStartRecommender()
        return pipeline

    def build_cold_start_pipeline(self):
        pipeline = RecommenderPipeline.__new__(RecommenderPipeline)
        pipeline.recaller = FakeEmptyRecaller()
        pipeline.rough_ranker = RankerShouldNotRun()
        pipeline.fine_ranker = RankerShouldNotRun()
        pipeline.reranker = FakeReranker()
        pipeline.cold_start_recommender = FakeColdStartRecommender()
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


if __name__ == "__main__":
    unittest.main()
