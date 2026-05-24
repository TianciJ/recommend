"""Ranking module for minimal recommender.

Implement:
- score recall candidates with simple heuristics
- sort candidates by combined score
- support final top-K ordering
"""

from typing import Dict, List, Tuple


def rank_candidates(candidate_scores: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
    ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
