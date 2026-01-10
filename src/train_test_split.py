"""
Train/Test Splitting Module for Recommender Systems.

Provides time-based splitting for temporal evaluation.
"""

import numpy as np
from typing import Tuple


def time_based_split(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    ratings: np.ndarray,
    timestamps: np.ndarray,
    test_ratio: float = 0.2
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split ratings by timestamp (last test_ratio of ratings go to test).
    
    Args:
        user_idx: User indices array
        item_idx: Item indices array
        ratings: Rating values array
        timestamps: Timestamp array
        test_ratio: Fraction of data for test set
        
    Returns:
        (train_tuple, test_tuple) where each tuple is (user_idx, item_idx, ratings)
    """
    # Sort by timestamp
    sort_idx = np.argsort(timestamps)
    
    # Split point
    n_ratings = len(ratings)
    split_point = int(n_ratings * (1 - test_ratio))
    
    train_idx = sort_idx[:split_point]
    test_idx = sort_idx[split_point:]
    
    train_data = (
        user_idx[train_idx].copy(),
        item_idx[train_idx].copy(),
        ratings[train_idx].copy()
    )
    
    test_data = (
        user_idx[test_idx].copy(),
        item_idx[test_idx].copy(),
        ratings[test_idx].copy()
    )
    
    return train_data, test_data


def random_split(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    ratings: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Random train/test split.
    
    Args:
        user_idx: User indices array
        item_idx: Item indices array
        ratings: Rating values array
        test_ratio: Fraction of data for test set
        seed: Random seed for reproducibility
        
    Returns:
        (train_tuple, test_tuple) where each tuple is (user_idx, item_idx, ratings)
    """
    rng = np.random.default_rng(seed)
    n_ratings = len(ratings)
    
    # Shuffle indices
    perm = rng.permutation(n_ratings)
    split_point = int(n_ratings * (1 - test_ratio))
    
    train_idx = perm[:split_point]
    test_idx = perm[split_point:]
    
    train_data = (
        user_idx[train_idx].copy(),
        item_idx[train_idx].copy(),
        ratings[train_idx].copy()
    )
    
    test_data = (
        user_idx[test_idx].copy(),
        item_idx[test_idx].copy(),
        ratings[test_idx].copy()
    )
    
    return train_data, test_data
