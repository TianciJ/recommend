"""Data reading module for minimal recommender.

Implement:
- load users_new.csv, items_new.csv, interactions_new.csv
- return pandas DataFrames for users, items, interactions
- support basic validation and missing file errors
"""

import pandas as pd
from pathlib import Path

from .config import USER_CSV, ITEM_CSV, INTERACTION_CSV


def load_users() -> pd.DataFrame:
    """Load user profile data."""
    return pd.read_csv(USER_CSV)


def load_items() -> pd.DataFrame:
    """Load item profile data."""
    return pd.read_csv(ITEM_CSV)


def load_interactions() -> pd.DataFrame:
    """Load user-item interaction data."""
    return pd.read_csv(INTERACTION_CSV)
