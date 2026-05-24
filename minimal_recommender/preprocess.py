"""Data preprocessing for the minimal recommender.

Implement:
- clean missing values in user/item/interaction tables
- split category and keyword strings into lists
- build user_history mapping user_id -> set(item_id)
- build user_item_rating mapping user_id -> {item_id: rating}
- build item_info for item metadata lookup
- compute popular_items ranking using weighted behaviors
"""

from collections import defaultdict
from typing import Dict, Set, List, Any

import pandas as pd

from .config import BEHAVIOR_WEIGHTS


def split_to_list(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.split("|") if token.strip()]
    return []


def build_user_history(interactions: pd.DataFrame) -> Dict[str, Set[str]]:
    history = defaultdict(set)
    for _, row in interactions.iterrows():
        history[str(row["user_id"])].add(str(row["item_id"]))
    return dict(history)


def build_user_item_rating(interactions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    user_ratings = defaultdict(dict)
    for _, row in interactions.iterrows():
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        user_ratings[user_id][item_id] = float(row.get("rating", 0))
    return dict(user_ratings)


def build_item_info(items: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    info = {}
    for _, row in items.iterrows():
        item_id = str(row["item_id"])
        info[item_id] = {
            "categories": split_to_list(row.get("item_categories")),
            "keywords": split_to_list(row.get("item_keywords")),
            "price": float(row.get("price", 0.0)),
            "create_time": row.get("create_time"),
        }
    return info


def build_popular_items(interactions: pd.DataFrame) -> Dict[str, float]:
    scores = defaultdict(float)
    for _, row in interactions.iterrows():
        item_id = str(row["item_id"])
        score = 0.0
        for field, weight in BEHAVIOR_WEIGHTS.items():
            score += float(row.get(field, 0)) * weight
        scores[item_id] += score
    return dict(scores)
