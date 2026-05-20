import numpy as np

def hit_rate(ranked_list: np.ndarray, target_item: int, k: int) -> int:
    """Hit Rate @ K."""
    return int(target_item in ranked_list[:k])

def ndcg(ranked_list: np.ndarray, target_item: int, k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    if target_item in ranked_list[:k]:
        rank = np.where(ranked_list[:k] == target_item)[0][0]
        return 1.0 / np.log2(rank + 2)
    return 0.0

def mrr(ranked_list: np.ndarray, target_item: int) -> float:
    """Mean Reciprocal Rank."""
    if target_item in ranked_list:
        rank = np.where(ranked_list == target_item)[0][0]
        return 1.0 / (rank + 1)
    return 0.0