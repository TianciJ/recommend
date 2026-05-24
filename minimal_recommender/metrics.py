"""Evaluation metrics for minimal recommender.

Implement:
- Precision@K and Recall@K for recommendation lists
- support simple ground truth item sets for evaluation
"""

from typing import List, Set


def precision_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    recommended_k = recommended[:k]
    hit_count = sum(1 for item in recommended_k if item in ground_truth)
    return hit_count / k


def recall_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    recommended_k = recommended[:k]
    hit_count = sum(1 for item in recommended_k if item in ground_truth)
    return hit_count / len(ground_truth)
