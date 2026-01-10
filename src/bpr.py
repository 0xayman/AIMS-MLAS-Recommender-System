"""
BPR (Bayesian Personalized Ranking) Model for Implicit Feedback.

Implements BPR with SGD optimization as described in:
"BPR: Bayesian Personalized Ranking from Implicit Feedback" (Rendle et al., 2009)

Key idea: Learn to rank positive items above negative items for each user.
"""

import numpy as np
from numba import njit, prange
from typing import List, Tuple, Optional, Set
import time


@njit
def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


@njit
def _sample_negative(n_items: int, positive_items: np.ndarray, rng_state: np.ndarray) -> int:
    """
    Sample a negative item that is not in positive_items.
    Uses a simple rejection sampling approach.
    
    Args:
        n_items: Total number of items
        positive_items: Array of positive item indices for this user
        rng_state: Random state array for Numba
        
    Returns:
        Index of a negative item
    """
    while True:
        j = np.random.randint(0, n_items)
        # Check if j is not in positive_items
        is_positive = False
        for p in positive_items:
            if j == p:
                is_positive = True
                break
        if not is_positive:
            return j


@njit(parallel=True)
def _bpr_update_batch(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    samples: np.ndarray,  # Shape: (n_samples, 3) - user, pos_item, neg_item
    learning_rate: float,
    lambda_reg: float
) -> float:
    """
    Perform batched BPR updates with parallelization.
    
    Args:
        user_factors: User embedding matrix (n_users, K)
        item_factors: Item embedding matrix (n_items, K)
        samples: Array of (user, positive_item, negative_item) triplets
        learning_rate: SGD learning rate
        lambda_reg: L2 regularization strength
        
    Returns:
        Total loss for this batch
    """
    n_samples = samples.shape[0]
    K = user_factors.shape[1]
    total_loss = 0.0
    
    for s in prange(n_samples):
        u = samples[s, 0]
        i = samples[s, 1]
        j = samples[s, 2]
        
        # Compute x_uij = prediction difference
        x_uij = 0.0
        for k in range(K):
            x_uij += user_factors[u, k] * (item_factors[i, k] - item_factors[j, k])
        
        # Compute sigmoid and loss
        sigmoid_x = _sigmoid(-x_uij)
        loss = -np.log(_sigmoid(x_uij) + 1e-10)
        total_loss += loss
        
        # Compute gradients and update
        for k in range(K):
            # Gradient for user
            grad_u = sigmoid_x * (item_factors[i, k] - item_factors[j, k]) - lambda_reg * user_factors[u, k]
            
            # Gradient for positive item
            grad_i = sigmoid_x * user_factors[u, k] - lambda_reg * item_factors[i, k]
            
            # Gradient for negative item
            grad_j = -sigmoid_x * user_factors[u, k] - lambda_reg * item_factors[j, k]
            
            # SGD updates
            user_factors[u, k] += learning_rate * grad_u
            item_factors[i, k] += learning_rate * grad_i
            item_factors[j, k] += learning_rate * grad_j
    
    return total_loss / n_samples


def _generate_training_samples(
    user_positive_items: List[np.ndarray],
    n_items: int,
    n_samples: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate BPR training samples (user, positive_item, negative_item).
    
    Args:
        user_positive_items: List of arrays, each containing positive items for a user
        n_items: Total number of items
        n_samples: Number of samples to generate
        rng: NumPy random generator
        
    Returns:
        Array of shape (n_samples, 3) with (user, pos_item, neg_item) triplets
    """
    samples = np.empty((n_samples, 3), dtype=np.int32)
    
    # Get users who have at least one positive item
    valid_users = [u for u, items in enumerate(user_positive_items) if len(items) > 0]
    
    for s in range(n_samples):
        # Sample a user
        u = rng.choice(valid_users)
        
        # Sample a positive item for this user
        pos_items = user_positive_items[u]
        i = rng.choice(pos_items)
        
        # Sample a negative item (not in positive items)
        while True:
            j = rng.integers(0, n_items)
            if j not in pos_items:
                break
        
        samples[s, 0] = u
        samples[s, 1] = i
        samples[s, 2] = j
    
    return samples


class BPRModel:
    """
    Bayesian Personalized Ranking (BPR) model for implicit feedback.
    
    This model learns user and item embeddings by optimizing the BPR loss,
    which encourages the model to rank positive items higher than negative items.
    
    Attributes:
        K: Latent dimension
        learning_rate: SGD learning rate
        lambda_reg: L2 regularization strength
        n_epochs: Number of training epochs
        n_samples_per_epoch: Number of samples per epoch
        verbose: Whether to print training progress
    """
    
    def __init__(
        self,
        K: int = 20,
        learning_rate: float = 0.05,
        lambda_reg: float = 0.01,
        n_epochs: int = 20,
        n_samples_per_epoch: Optional[int] = None,
        verbose: bool = True,
        seed: int = 42
    ):
        """
        Initialize BPR model.
        
        Args:
            K: Number of latent factors
            learning_rate: Learning rate for SGD
            lambda_reg: L2 regularization strength
            n_epochs: Number of training epochs
            n_samples_per_epoch: Samples per epoch (default: n_interactions * 10)
            verbose: Print training progress
            seed: Random seed for reproducibility
        """
        self.K = K
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs
        self.n_samples_per_epoch = n_samples_per_epoch
        self.verbose = verbose
        self.seed = seed
        
        # Model parameters (initialized during fit)
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.n_users: int = 0
        self.n_items: int = 0
        
        # Training history
        self.loss_history: List[float] = []
        
    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None
    ) -> 'BPRModel':
        """
        Train the BPR model on implicit feedback data.
        
        Args:
            user_ids: Array of user indices (positive interactions)
            item_ids: Array of item indices (positive interactions)
            n_users: Total number of users (inferred if not provided)
            n_items: Total number of items (inferred if not provided)
            
        Returns:
            self
        """
        rng = np.random.default_rng(self.seed)
        
        # Infer dimensions
        self.n_users = n_users if n_users is not None else int(user_ids.max()) + 1
        self.n_items = n_items if n_items is not None else int(item_ids.max()) + 1
        
        # Build user -> positive items mapping
        user_positive_items: List[Set[int]] = [set() for _ in range(self.n_users)]
        for u, i in zip(user_ids, item_ids):
            user_positive_items[u].add(i)
        
        # Convert to list of arrays for faster sampling
        user_positive_arrays = [np.array(list(items), dtype=np.int32) for items in user_positive_items]
        
        # Initialize factors
        self.user_factors = rng.normal(0, 0.01, (self.n_users, self.K)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.01, (self.n_items, self.K)).astype(np.float32)
        
        # Determine samples per epoch
        n_interactions = len(user_ids)
        samples_per_epoch = self.n_samples_per_epoch or n_interactions * 10
        
        self.loss_history = []
        
        if self.verbose:
            print(f"Training BPR model: K={self.K}, lr={self.learning_rate}, λ={self.lambda_reg}")
            print(f"Users: {self.n_users}, Items: {self.n_items}, Interactions: {n_interactions}")
            print(f"Samples per epoch: {samples_per_epoch:,}")
        
        for epoch in range(self.n_epochs):
            start_time = time.time()
            
            # Generate training samples
            samples = _generate_training_samples(
                user_positive_arrays,
                self.n_items,
                samples_per_epoch,
                rng
            )
            
            # Perform BPR update
            epoch_loss = _bpr_update_batch(
                self.user_factors,
                self.item_factors,
                samples,
                self.learning_rate,
                self.lambda_reg
            )
            
            self.loss_history.append(epoch_loss)
            elapsed = time.time() - start_time
            
            if self.verbose:
                print(f"Epoch {epoch+1:3d}/{self.n_epochs}: Loss={epoch_loss:.4f} ({elapsed:.2f}s)")
        
        return self
    
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        Predict scores for given user-item pairs.
        
        Args:
            user_ids: Array of user indices
            item_ids: Array of item indices
            
        Returns:
            Array of predicted scores
        """
        scores = np.sum(
            self.user_factors[user_ids] * self.item_factors[item_ids],
            axis=1
        )
        return scores
    
    def predict_user(self, user_id: int) -> np.ndarray:
        """
        Predict scores for all items for a given user.
        
        Args:
            user_id: User index
            
        Returns:
            Array of scores for all items
        """
        return self.user_factors[user_id] @ self.item_factors.T
    
    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_items: Optional[Set[int]] = None
    ) -> np.ndarray:
        """
        Get top-N item recommendations for a user.
        
        Args:
            user_id: User index
            n_items: Number of items to recommend
            exclude_items: Set of item indices to exclude (e.g., training items)
            
        Returns:
            Array of recommended item indices (sorted by score, descending)
        """
        scores = self.predict_user(user_id)
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        top_items = np.argsort(scores)[::-1][:n_items]
        return top_items
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        np.savez(
            filepath,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            K=self.K,
            learning_rate=self.learning_rate,
            lambda_reg=self.lambda_reg,
            n_users=self.n_users,
            n_items=self.n_items,
            loss_history=np.array(self.loss_history)
        )
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BPRModel':
        """Load model from disk."""
        data = np.load(filepath)
        model = cls(
            K=int(data['K']),
            learning_rate=float(data['learning_rate']),
            lambda_reg=float(data['lambda_reg'])
        )
        model.user_factors = data['user_factors']
        model.item_factors = data['item_factors']
        model.n_users = int(data['n_users'])
        model.n_items = int(data['n_items'])
        model.loss_history = list(data['loss_history'])
        return model
