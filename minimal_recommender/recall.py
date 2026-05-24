"""Recall module for minimal recommender.

Implement:
- popular recall channel based on global hotness ranking
- ItemCF recall channel based on item co-occurrence and similarity
- merge recall results into candidate score dictionary
- filter out already interacted items
"""

from collections import defaultdict
from math import sqrt
from typing import Dict, List, Set

import pandas as pd

from .config import ITEMCF_TOP_N, SIMILARITY_EPS
from .preprocess import build_user_history, build_item_info, build_popular_items


def popular_recall(popular_items: Dict[str, float], user_history: Set[str], top_k: int) -> List[str]:
    candidates = [item for item, _ in sorted(popular_items.items(), key=lambda x: x[1], reverse=True)]
    return [item for item in candidates if item not in user_history][:top_k]


def build_item_user_inverse(interactions: pd.DataFrame) -> Dict[str, Set[str]]:
    inverse = defaultdict(set)
    for _, row in interactions.iterrows():
        inverse[str(row["item_id"])].add(str(row["user_id"]))
    return dict(inverse)


def build_item_similarities(interactions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    item_users = build_item_user_inverse(interactions)
    item_counts = {item: len(users) for item, users in item_users.items()}
    co_counts = defaultdict(lambda: defaultdict(int))
    for item_i, users_i in item_users.items():
        for item_j, users_j in item_users.items():
            if item_i == item_j:
                continue
            score = len(users_i & users_j)
            if score > 0:
                co_counts[item_i][item_j] = score
    similarities = defaultdict(dict)
    for item_i, neighbors in co_counts.items():
        for item_j, co_count in neighbors.items():
            sim = co_count / sqrt(item_counts[item_i] * item_counts[item_j] + SIMILARITY_EPS)
            similarities[item_i][item_j] = sim
    return {item: dict(sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:ITEMCF_TOP_N]) for item, neighbors in similarities.items()}


def itemcf_recall(user_id: str, user_item_ratings: Dict[str, Dict[str, float]], similarities: Dict[str, Dict[str, float]], user_history: Set[str], top_k: int) -> List[str]:
    scores = defaultdict(float)
    history = user_item_ratings.get(user_id, {})
    for item_i, rating in history.items():
        for item_j, sim in similarities.get(item_i, {}).items():
            if item_j in user_history:
                continue
            scores[item_j] += rating * sim
    sorted_items = [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return sorted_items[:top_k]


def merge_recalls(itemcf_scores: Dict[str, float], popular_items: List[str]) -> Dict[str, float]:
    merged = dict(itemcf_scores)
    for rank, item in enumerate(popular_items):
        if item not in merged:
            merged[item] = 1.0 / (rank + 1)
    return merged
