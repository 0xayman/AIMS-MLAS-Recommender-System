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
