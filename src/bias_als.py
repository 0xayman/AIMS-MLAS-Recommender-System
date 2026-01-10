"""
Bias-Only ALS Training Module.

Implements Alternating Least Squares for the bias-only model:
    r_mn ≈ μ + b_m + b_n

Uses Numba with parallel=True for efficient updates.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, List, Optional

from .data_structures import RatingMatrix


@njit(parallel=True)
def update_user_biases(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_ratings: np.ndarray,
    item_biases: np.ndarray,
    user_biases: np.ndarray,
    global_mean: float,
    lambda_: float,
    gamma: float
) -> None:
    """
    Update all user biases in parallel.
    
    Objective: min 0.5*λ*sum((r - pred)^2) + 0.5*γ*sum(b^2)
    Derivative w.r.t b_m: -λ*sum(r - pred) + γ*b_m = 0
    => λ*sum(r - μ - b_n - b_m) = γ*b_m
    => λ*sum(r - μ - b_n) - λ*|I(m)|*b_m = γ*b_m
    => b_m = sum(r - μ - b_n) / (|I(m)| + γ/λ)
    
    Args:
        user_indptr: CSR row pointers for users
        user_item_indices: Item indices for each rating (CSR format)
        user_ratings: Rating values (CSR format)
        item_biases: Current item biases
        user_biases: User biases array to update (in-place)
        global_mean: Global mean rating
        lambda_: Error term scaling factor
        gamma: Bias regularization factor
    """
    n_users = len(user_indptr) - 1
    
    # Avoid division by zero if lambda_ is 0 (though unlikely in valid settings)
    reg_term = gamma / lambda_ if lambda_ > 1e-9 else 0.0
    
    for u in prange(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        count = end - start
        
        if count == 0:
            user_biases[u] = 0.0
            continue
        
        total = 0.0
        for idx in range(start, end):
            item = user_item_indices[idx]
            rating = user_ratings[idx]
            total += rating - global_mean - item_biases[item]
        
        user_biases[u] = total / (count + reg_term)



@njit(parallel=True)
def update_item_biases(
    item_indptr: np.ndarray,
    item_user_indices: np.ndarray,
    item_ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    global_mean: float,
    lambda_: float,
    gamma: float
) -> None:
    """
    Update all item biases in parallel.
    
    Objective: min 0.5*λ*sum((r - pred)^2) + 0.5*γ*sum(b^2)
    Derivative w.r.t b_n: -λ*sum(r - pred) + γ*b_n = 0
    => b_n = sum(r - μ - b_m) / (|U(n)| + γ/λ)
    
    Args:
        item_indptr: CSR row pointers for items
        item_user_indices: User indices for each rating (CSR format)
        item_ratings: Rating values (CSR format)
        user_biases: Current user biases
        item_biases: Item biases array to update (in-place)
        global_mean: Global mean rating
        lambda_: Error term scaling factor
        gamma: Bias regularization factor
    """
    n_items = len(item_indptr) - 1
    
    # Avoid division by zero
    reg_term = gamma / lambda_ if lambda_ > 1e-9 else 0.0
    
    for i in prange(n_items):
        start = item_indptr[i]
        end = item_indptr[i + 1]
        count = end - start
        
        if count == 0:
            item_biases[i] = 0.0
            continue
        
        total = 0.0
        for idx in range(start, end):
            user = item_user_indices[idx]
            rating = item_ratings[idx]
            total += rating - global_mean - user_biases[user]
        
        item_biases[i] = total / (count + reg_term)



@njit(parallel=True)
def compute_predictions_csr(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    global_mean: float
) -> np.ndarray:
    """
    Compute predictions for all ratings in CSR order.
    
    Args:
        user_indptr: CSR row pointers
        user_item_indices: Item indices
        user_biases: User bias array
        item_biases: Item bias array
        global_mean: Global mean rating
        
    Returns:
        Predictions array
    """
    n_ratings = len(user_item_indices)
    n_users = len(user_indptr) - 1
    predictions = np.empty(n_ratings, dtype=np.float32)
    
    for u in prange(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        for idx in range(start, end):
            item = user_item_indices[idx]
            predictions[idx] = global_mean + user_biases[u] + item_biases[item]
    
    return predictions


@njit
def compute_loss_csr(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    global_mean: float,
    lambda_: float,
    gamma: float
) -> float:
    """
    Compute loss matching user's formulation.
    
    Loss = 0.5 * λ * sum((r - pred)^2) + 0.5 * γ * (sum(b_u^2) + sum(b_i^2))
    
    Args:
        user_indptr: CSR row pointers
        user_item_indices: Item indices
        user_ratings: Rating values
        user_biases: User bias array
        item_biases: Item bias array
        global_mean: Global mean rating
        lambda_: Error term scaling factor
        gamma: Bias regularization factor
        
    Returns:
        Total loss value
    """
    n_users = len(user_indptr) - 1
    n_items = len(item_biases)
    
    # Squared error term
    squared_error = 0.0
    for u in range(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        for idx in range(start, end):
            item = user_item_indices[idx]
            rating = user_ratings[idx]
            pred = global_mean + user_biases[u] + item_biases[item]
            diff = rating - pred
            squared_error += diff * diff
    
    # Regularization term
    reg_user = 0.0
    for u in range(n_users):
        reg_user += user_biases[u] * user_biases[u]
    
    reg_item = 0.0
    for i in range(n_items):
        reg_item += item_biases[i] * item_biases[i]
    
    return 0.5 * lambda_ * squared_error + 0.5 * gamma * (reg_user + reg_item)


def train_bias_als(
    rating_matrix: RatingMatrix,
    n_iters: int = 20,
    lambda_: float = 1.0,
    gamma: float = 0.5,
    test_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, List[float], List[float], Optional[List[float]]]:
    """
    Train bias-only ALS model using user's specific formulation.
    
    Objective: min 0.5*λ*sum((r - pred)^2) + 0.5*γ*sum(b^2)
    
    Args:
        rating_matrix: RatingMatrix with dual CSR indexing
        n_iters: Number of ALS iterations
        lambda_: Error term scaling factor
        gamma: Bias regularization factor
        test_data: Optional (user_idx, item_idx, ratings) for test RMSE
        verbose: Print progress
        
    Returns:
        Tuple of:
            - user_biases: Trained user biases
            - item_biases: Trained item biases  
            - global_mean: Global mean rating
            - loss_history: Loss per iteration
            - train_rmse_history: Train RMSE per iteration
            - test_rmse_history: Test RMSE per iteration (None if no test_data)
    """
    # Extract arrays from RatingMatrix
    user_indptr = rating_matrix.user_index.indptr
    user_item_indices = rating_matrix.user_index.indices
    user_ratings = rating_matrix.user_index.data
    
    item_indptr = rating_matrix.item_index.indptr
    item_user_indices = rating_matrix.item_index.indices
    item_ratings = rating_matrix.item_index.data
    
    global_mean = rating_matrix.global_mean
    n_users = rating_matrix.n_users
    n_items = rating_matrix.n_items
    
    # Initialize biases to zero
    user_biases = np.zeros(n_users, dtype=np.float32)
    item_biases = np.zeros(n_items, dtype=np.float32)
    
    # History tracking
    loss_history = []
    train_rmse_history = []
    test_rmse_history = [] if test_data is not None else None
    
    for iteration in range(n_iters):
        # Update user biases (parallel)
        update_user_biases(
            user_indptr, user_item_indices, user_ratings,
            item_biases, user_biases, global_mean, lambda_, gamma
        )
        
        # Update item biases (parallel)
        update_item_biases(
            item_indptr, item_user_indices, item_ratings,
            user_biases, item_biases, global_mean, lambda_, gamma
        )
        
        # Compute loss
        loss = compute_loss_csr(
            user_indptr, user_item_indices, user_ratings,
            user_biases, item_biases, global_mean, lambda_, gamma
        )
        loss_history.append(loss)
        
        # Compute train RMSE
        train_preds = compute_predictions_csr(
            user_indptr, user_item_indices, user_biases, item_biases, global_mean
        )
        train_rmse = np.sqrt(np.mean((user_ratings - train_preds) ** 2))
        train_rmse_history.append(train_rmse)
        
        # Compute test RMSE if test data provided
        if test_data is not None:
            test_users, test_items, test_ratings = test_data
            test_preds = global_mean + user_biases[test_users] + item_biases[test_items]
            test_rmse = np.sqrt(np.mean((test_ratings - test_preds) ** 2))
            test_rmse_history.append(test_rmse)
        
        if verbose:
            msg = f"Iteration {iteration+1:3d}: Loss={loss:.4f}, Train RMSE={train_rmse:.4f}"
            if test_data is not None:
                msg += f", Test RMSE={test_rmse:.4f}"
            print(msg)
    
    return (
        user_biases,
        item_biases,
        global_mean,
        loss_history,
        train_rmse_history,
        test_rmse_history
    )


def save_model(
    filepath: str,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    global_mean: float
) -> None:
    """
    Save trained bias model to compressed npz file.
    
    Args:
        filepath: Path to save the model
        user_biases: User bias array
        item_biases: Item bias array
        global_mean: Global mean rating
    """
    np.savez_compressed(
        filepath,
        user_biases=user_biases,
        item_biases=item_biases,
        global_mean=np.array([global_mean])
    )
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load trained bias model from npz file.
    
    Args:
        filepath: Path to the model file
        
    Returns:
        Tuple of (user_biases, item_biases, global_mean)
    """
    data = np.load(filepath)
    return (
        data['user_biases'],
        data['item_biases'],
        float(data['global_mean'][0])
    )
