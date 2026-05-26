import unittest

from recommender_pipeline import RecommenderPipeline


class FakeRecaller:
    def recommend(self, user_id, top_k):
        return [
            {"movie_id": index, "score": 1.0 / index, "title": f"movie-{index}"}
            for index in range(1, top_k + 1)
        ]


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


class PipelineTimingTest(unittest.TestCase):
    def build_pipeline(self):
        pipeline = RecommenderPipeline.__new__(RecommenderPipeline)
        pipeline.recaller = FakeRecaller()
        pipeline.rough_ranker = FakeRoughRanker()
        pipeline.fine_ranker = FakeFineRanker()
        pipeline.reranker = FakeReranker()
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


if __name__ == "__main__":
    unittest.main()
