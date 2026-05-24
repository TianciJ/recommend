"""Re-ranking module for minimal recommender.

Implement:
- remove items already interacted by the user
- limit repeated categories in the final list
- preserve recommendation diversity
"""

from typing import Dict, List, Set


def rerank_candidates(candidate_scores: Dict[str, float], user_history: Set[str], item_info: Dict[str, Dict], top_k: int) -> List[str]:
    selected = []
    seen_categories = set()
    for item_id, _score in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True):
        if item_id in user_history:
            continue
        categories = item_info.get(item_id, {}).get("categories", [])
        if categories:
            if any(cat in seen_categories for cat in categories):
                continue
            seen_categories.update(categories[:1])
        selected.append(item_id)
        if len(selected) >= top_k:
            break
    return selected
