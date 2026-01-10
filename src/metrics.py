"""
Metrics Module for Recommender System Evaluation.

Provides RMSE, loss computation, and prediction utilities.
"""

import numpy as np
from numba import njit, prange


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        predictions: Predicted ratings
        targets: Actual ratings
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def compute_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    lambda_: float = 0.02
) -> float:
    """
    Compute regularized squared error loss.
    
    Loss = sum((r - pred)^2) + lambda * (sum(b_u^2) + sum(b_i^2))
    
    Args:
        predictions: Predicted ratings
        targets: Actual ratings
        user_biases: User bias array
        item_biases: Item bias array
        lambda_: Regularization strength
        
    Returns:
        Total loss value
    """
    squared_error = np.sum((predictions - targets) ** 2)
    reg_term = lambda_ * (np.sum(user_biases ** 2) + np.sum(item_biases ** 2))
    return squared_error + reg_term


def predict_bias(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    global_mean: float
) -> np.ndarray:
    """
    Make predictions using bias model: pred = mu + b_u + b_i.
    
    Vectorized implementation.
    
    Args:
        user_idx: User indices
        item_idx: Item indices
        user_biases: User bias array
        item_biases: Item bias array
        global_mean: Global mean rating
        
    Returns:
        Array of predicted ratings
    """
    return global_mean + user_biases[user_idx] + item_biases[item_idx]


@njit
def predict_bias_numba(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    global_mean: float
) -> np.ndarray:
    """
    Numba-optimized bias predictions.
    
    Args:
        user_idx: User indices
        item_idx: Item indices
        user_biases: User bias array
        item_biases: Item bias array
        global_mean: Global mean rating
        
    Returns:
        Array of predicted ratings
    """
    n = len(user_idx)
    preds = np.empty(n, dtype=np.float32)
    for i in range(n):
        preds[i] = global_mean + user_biases[user_idx[i]] + item_biases[item_idx[i]]
    return preds


@njit
def compute_rmse_numba(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Numba-optimized RMSE computation.
    
    Args:
        predictions: Predicted ratings
        targets: Actual ratings
        
    Returns:
        RMSE value
    """
    n = len(predictions)
    total = 0.0
    for i in range(n):
        diff = predictions[i] - targets[i]
        total += diff * diff
    return np.sqrt(total / n)


# =============================================================================
# Ranking Metrics for Implicit Feedback
# =============================================================================

def precision_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    Compute Precision@K for a single user.
    
    Precision@K = (# relevant items in top K) / K
    
    Args:
        recommended: Array of recommended item indices (sorted by score)
        relevant: Array of relevant item indices (ground truth)
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K value (0 to 1)
    """
    if k <= 0:
        return 0.0
    
    top_k = set(recommended[:k])
    relevant_set = set(relevant)
    n_relevant_in_k = len(top_k & relevant_set)
    
    return n_relevant_in_k / k


def recall_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    Compute Recall@K for a single user.
    
    Recall@K = (# relevant items in top K) / (# total relevant items)
    
    Args:
        recommended: Array of recommended item indices (sorted by score)
        relevant: Array of relevant item indices (ground truth)
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K value (0 to 1)
    """
    if len(relevant) == 0:
        return 0.0
    
    top_k = set(recommended[:k])
    relevant_set = set(relevant)
    n_relevant_in_k = len(top_k & relevant_set)
    
    return n_relevant_in_k / len(relevant)


def ndcg_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K for a single user.
    
    NDCG@K = DCG@K / IDCG@K
    
    Args:
        recommended: Array of recommended item indices (sorted by score)
        relevant: Array of relevant item indices (ground truth)
        k: Number of top recommendations to consider
        
    Returns:
        NDCG@K value (0 to 1)
    """
    if len(relevant) == 0 or k <= 0:
        return 0.0
    
    relevant_set = set(relevant)
    top_k = recommended[:k]
    
    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because position is 1-indexed
    
    # Compute ideal DCG (all relevant items at top)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_ranking_metrics(
    user_recommendations: dict,
    test_items_per_user: dict,
    k_values: list = [5, 10, 20]
) -> dict:
    """
    Evaluate ranking metrics across all users.
    
    Args:
        user_recommendations: Dict mapping user_id -> array of recommended items
        test_items_per_user: Dict mapping user_id -> array of test items
        k_values: List of K values to evaluate
        
    Returns:
        Dict with average metrics: {'precision@5': ..., 'recall@5': ..., etc.}
    """
    results = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_id, recommendations in user_recommendations.items():
            if user_id not in test_items_per_user:
                continue
            
            relevant = test_items_per_user[user_id]
            if len(relevant) == 0:
                continue
            
            precisions.append(precision_at_k(recommendations, relevant, k))
            recalls.append(recall_at_k(recommendations, relevant, k))
            ndcgs.append(ndcg_at_k(recommendations, relevant, k))
        
        if len(precisions) > 0:
            results[f'precision@{k}'] = np.mean(precisions)
            results[f'recall@{k}'] = np.mean(recalls)
            results[f'ndcg@{k}'] = np.mean(ndcgs)
        else:
            results[f'precision@{k}'] = 0.0
            results[f'recall@{k}'] = 0.0
            results[f'ndcg@{k}'] = 0.0
    
    return results
