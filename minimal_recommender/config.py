"""Minimal recommender config.

Implement:
- global data paths for users_new.csv, items_new.csv, interactions_new.csv
- recall configuration: number of recall candidates, recommendation size
- weights for popularity and interaction scoring
- similarity parameters and thresholds
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "../data"

USER_CSV = DATA_DIR / "users_new.csv"
ITEM_CSV = DATA_DIR / "items_new.csv"
INTERACTION_CSV = DATA_DIR / "interactions_new.csv"

RECALL_TOP_K = 100
RECOMMEND_TOP_K = 20
POPULAR_WEIGHT = 1.0
ITEMCF_TOP_N = 50

BEHAVIOR_WEIGHTS = {
    "click": 1,
    "cart": 3,
    "buy": 5,
    "forward": 2,
}

SIMILARITY_EPS = 1e-8
