"""Main service orchestration for minimal recommender.

Implement:
- load data and preprocess once
- orchestrate recall, ranking, rerank flow
- expose recommend(user_id, top_k) method
- combine recall results with priority for ItemCF
"""

from typing import Dict, List

import pandas as pd

from .data_loader import load_users, load_items, load_interactions
from .preprocess import build_user_history, build_user_item_rating, build_item_info, build_popular_items
from .recall import popular_recall, build_item_similarities, itemcf_recall, merge_recalls
from .ranking import rank_candidates
from .rerank import rerank_candidates


class MinimalRecommender:
    def __init__(self):
        self.users = load_users()
        self.items = load_items()
        self.interactions = load_interactions()
        self.user_history = build_user_history(self.interactions)
        self.user_item_rating = build_user_item_rating(self.interactions)
        self.item_info = build_item_info(self.items)
        self.popular_items = build_popular_items(self.interactions)
        self.similarities = build_item_similarities(self.interactions)

    def recommend(self, user_id: str, top_k: int = 10) -> List[str]:
        history = self.user_history.get(user_id, set())
        popular = popular_recall(self.popular_items, history, top_k)
        itemcf_list = itemcf_recall(user_id, self.user_item_rating, self.similarities, history, top_k)
        itemcf_scores = {item: float(rank) for rank, item in enumerate(itemcf_list[::-1], start=1)}
        merged = merge_recalls(itemcf_scores, popular)
        ranked = rank_candidates(merged, top_k)
        reranked = rerank_candidates(dict(ranked), history, self.item_info, top_k)
        return reranked
