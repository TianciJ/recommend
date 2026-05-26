import unittest

from evaluate_pipeline import build_command_timing_summary
from evaluate_pipeline import calculate_ranking_metrics


class EvaluatePipelineMetricsTest(unittest.TestCase):
    def test_calculate_ranking_metrics_for_multiple_cutoffs(self):
        recommendations = [10, 20, 30, 40]
        liked_movies = {20, 40, 50}

        metrics = calculate_ranking_metrics(
            recommendations=recommendations,
            liked_movies=liked_movies,
            k_list=[2, 4],
        )

        self.assertAlmostEqual(metrics[2]["precision"], 0.5)
        self.assertAlmostEqual(metrics[2]["recall"], 1 / 3)
        self.assertAlmostEqual(metrics[2]["hit_rate"], 1.0)
        self.assertAlmostEqual(metrics[2]["mrr"], 0.5)
        self.assertAlmostEqual(metrics[2]["ndcg"], 0.38685280723454163)

        self.assertAlmostEqual(metrics[4]["precision"], 0.5)
        self.assertAlmostEqual(metrics[4]["recall"], 2 / 3)
        self.assertAlmostEqual(metrics[4]["hit_rate"], 1.0)
        self.assertAlmostEqual(metrics[4]["mrr"], 0.5)
        self.assertAlmostEqual(metrics[4]["ndcg"], 0.49818925746641285)

    def test_calculate_ranking_metrics_handles_no_liked_movies(self):
        metrics = calculate_ranking_metrics(
            recommendations=[10, 20],
            liked_movies=set(),
            k_list=[2],
        )

        self.assertEqual(metrics[2]["precision"], 0)
        self.assertEqual(metrics[2]["recall"], 0)
        self.assertEqual(metrics[2]["hit_rate"], 0)
        self.assertEqual(metrics[2]["mrr"], 0)
        self.assertEqual(metrics[2]["ndcg"], 0)

    def test_build_command_timing_summary_includes_wall_clock_average(self):
        summary = build_command_timing_summary(
            pipeline_init_ms=1000,
            evaluation_wall_ms=23000,
            output_print_ms=10,
            command_total_ms=24050,
            evaluated_users=1000,
        )

        self.assertEqual(summary["pipeline_init_ms"], 1000)
        self.assertEqual(summary["evaluation_wall_ms"], 23000)
        self.assertEqual(summary["output_print_ms"], 10)
        self.assertEqual(summary["command_total_ms"], 24050)
        self.assertEqual(summary["evaluated_users"], 1000)
        self.assertAlmostEqual(summary["avg_command_ms_per_user"], 24.05)


if __name__ == "__main__":
    unittest.main()
