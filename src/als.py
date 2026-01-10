"""
Latent Factor Model (Matrix Factorization) with ALS.

Implements a matrix factorization model with biases:
    r_mn ≈ μ + b_m + b_n + u_m^T v_n

Uses Numba-optimized Alternating Least Squares (ALS) for training.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, List, Optional, Callable
import time

from .data_structures import RatingMatrix
from scipy import sparse


@njit(parallel=True)
def update_user_factors(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    item_factors: np.ndarray,
    user_factors: np.ndarray,
    global_mean: float,
    lambda_: float,
    tau: float
) -> None:
    """
    Update user latent factors in parallel.
    
    Follows oldcode formulation:
    Solves (τ*I + λ*V^T V) u_m = λ * V^T (r - μ - b_m - b_n)
    
    Where:
        tau is the factor regularization
        lambda_ is the data term weight
    
    Args:
        user_indptr: CSR row pointers for users
        user_item_indices: Item indices for each rating
        user_ratings: Rating values
        user_biases: User biases
        item_biases: Item biases
        item_factors: Item latent factors (N x K)
        user_factors: User latent factors to update (M x K)
        global_mean: Global mean rating
        lambda_: Data term weight (precision)
        tau: Factor regularization strength
    """
    n_users = len(user_indptr) - 1
    K = item_factors.shape[1]
    
    # tau * I is the regularization term (matching oldcode)
    tau_I = np.eye(K, dtype=np.float32) * np.float32(tau)
    lambda_f32 = np.float32(lambda_)
    
    for u in prange(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        
        if end == start:
            # No ratings, shrinking to zero (prior mean)
            user_factors[u] = 0.0
            continue
            
        # Build A and b using oldcode formulation:
        # A = τ*I + λ * sum(v_n * v_n^T)
        # b = λ * sum(residual * v_n)
        
        A = np.copy(tau_I)
        b = np.zeros(K, dtype=np.float32)
        
        bias_u = user_biases[u]
        
        for idx in range(start, end):
            item = user_item_indices[idx]
            rating = user_ratings[idx]
            
            # Get item factor
            v_n = item_factors[item]
            
            # Compute residual for the dot product part: r - (μ + b_u + b_i)
            residual = rating - global_mean - bias_u - item_biases[item]
            
            # Update A: A += λ * v_n * v_n^T
            # Update b: b += λ * residual * v_n
            for k1 in range(K):
                val_k1 = v_n[k1]
                b[k1] += lambda_f32 * residual * val_k1
                for k2 in range(K):
                    A[k1, k2] += lambda_f32 * val_k1 * v_n[k2]
        
        # Solve Ax = b
        user_factors[u] = np.linalg.solve(A, b)


@njit(parallel=True)
def update_item_factors(
    item_indptr: np.ndarray,
    item_user_indices: np.ndarray,
    item_ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    global_mean: float,
    lambda_: float,
    tau: float
) -> None:
    """
    Update item latent factors in parallel.
    
    Follows oldcode formulation:
    Solves (τ*I + λ*U^T U) v_n = λ * U^T (r - μ - b_m - b_n)
    
    Where:
        tau is the factor regularization
        lambda_ is the data term weight
    
    Args:
        item_indptr: CSR row pointers for items
        item_user_indices: User indices for each rating
        item_ratings: Rating values
        user_biases: User biases
        item_biases: Item biases
        user_factors: User latent factors (M x K)
        item_factors: Item latent factors to update (N x K)
        global_mean: Global mean rating
        lambda_: Data term weight (precision)
        tau: Factor regularization strength
    """
    n_items = len(item_indptr) - 1
    K = user_factors.shape[1]
    
    # tau * I is the regularization term (matching oldcode)
    tau_I = np.eye(K, dtype=np.float32) * np.float32(tau)
    lambda_f32 = np.float32(lambda_)
    
    for i in prange(n_items):
        start = item_indptr[i]
        end = item_indptr[i + 1]
        
        if end == start:
            item_factors[i] = 0.0
            continue

        # Build A and b using oldcode formulation:
        # A = τ*I + λ * sum(u_m * u_m^T)
        # b = λ * sum(residual * u_m)
        
        A = np.copy(tau_I)
        b = np.zeros(K, dtype=np.float32)
        
        bias_i = item_biases[i]
        
        for idx in range(start, end):
            user = item_user_indices[idx]
            rating = item_ratings[idx]
            
            u_m = user_factors[user]
            residual = rating - global_mean - user_biases[user] - bias_i
            
            # Update A: A += λ * u_m * u_m^T
            # Update b: b += λ * residual * u_m
            for k1 in range(K):
                val_k1 = u_m[k1]
                b[k1] += lambda_f32 * residual * val_k1
                for k2 in range(K):
                    A[k1, k2] += lambda_f32 * val_k1 * u_m[k2]
        
        item_factors[i] = np.linalg.solve(A, b)


@njit
def compute_loss_with_factors(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    global_mean: float,
    lambda_: float,
    tau: float,
    gamma: float
) -> float:
    """
    Compute total loss including regularization.
    
    Follows the formulation from oldcode:
    L = (λ/2) * sum((r - pred)^2) 
      + (γ/2) * (sum(b_u^2) + sum(b_i^2))
      + (τ/2) * (sum(||U||^2) + sum(||V||^2))
    
    Where:
        lambda_ weights the data term (squared error)
        gamma regularizes biases
        tau regularizes latent factors
    """
    n_users = len(user_indptr) - 1
    
    squared_error = 0.0
    
    for u in range(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        
        for idx in range(start, end):
            item = user_item_indices[idx]
            rating = user_ratings[idx]
            
            pred = global_mean + user_biases[u] + item_biases[item]
            
            # specific dot product
            dot = 0.0
            for k in range(user_factors.shape[1]):
                dot += user_factors[u, k] * item_factors[item, k]
            
            pred += dot
            
            diff = rating - pred
            squared_error += diff * diff
            
    # Regularization - match oldcode formulation
    reg_biases = np.sum(user_biases**2) + np.sum(item_biases**2)
    reg_factors = np.sum(user_factors**2) + np.sum(item_factors**2)
    
    # Loss = λ/2 * SSE + γ/2 * bias_reg + τ/2 * factor_reg
    return 0.5 * lambda_ * squared_error + 0.5 * gamma * reg_biases + 0.5 * tau * reg_factors


@njit(parallel=True)
def predict_batch(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    global_mean: float
) -> np.ndarray:
    """Make predictions for a batch of user-item pairs."""
    n = len(user_indices)
    preds = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        u = user_indices[i]
        item = item_indices[i]
        
        dot = 0.0
        for k in range(user_factors.shape[1]):
            dot += user_factors[u, k] * item_factors[item, k]
            
        preds[i] = global_mean + user_biases[u] + item_biases[item] + dot
        
    return preds


# Import bias updates from bias_als to reuse them
# We need to make sure we can import them. 
# Relative import is fine since they are in the same package.
from .bias_als import update_user_biases, update_item_biases



@njit(parallel=True)
def update_user_factors_advanced(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_ratings: np.ndarray,
    user_weights: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    item_factors: np.ndarray,
    user_factors: np.ndarray,
    global_mean: float,
    lambda_array: np.ndarray,
    tau: float
) -> None:
    """
    Update user factors with sample weights and per-user regularization.
    """
    n_users = len(user_indptr) - 1
    K = item_factors.shape[1]
    tau_f32 = np.float32(tau)
    
    for u in prange(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        
        if end == start:
            user_factors[u] = 0.0
            continue
            
        # Per-user regularization: lambda_u * I
        lambda_val = lambda_array[u]
        
        A = np.eye(K, dtype=np.float32) * np.float32(lambda_val)
        b = np.zeros(K, dtype=np.float32)
        
        bias_u = user_biases[u]
        
        for idx in range(start, end):
            item = user_item_indices[idx]
            rating = user_ratings[idx]
            weight = user_weights[idx]
            
            # Effective precision for this observation
            eff_weight = tau_f32 * np.float32(weight)
            
            v_n = item_factors[item]
            residual = rating - global_mean - bias_u - item_biases[item]
            
            for k1 in range(K):
                val_k1 = v_n[k1]
                b[k1] += eff_weight * residual * val_k1
                for k2 in range(K):
                    A[k1, k2] += eff_weight * val_k1 * v_n[k2]
                    
        user_factors[u] = np.linalg.solve(A, b)


@njit(parallel=True)
def update_item_factors_advanced(
    item_indptr: np.ndarray,
    item_user_indices: np.ndarray,
    item_ratings: np.ndarray,
    item_weights: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    global_mean: float,
    lambda_array: np.ndarray,
    tau: float
) -> None:
    """
    Update item factors with sample weights and per-item regularization.
    """
    n_items = len(item_indptr) - 1
    K = user_factors.shape[1]
    tau_f32 = np.float32(tau)
    
    for i in prange(n_items):
        start = item_indptr[i]
        end = item_indptr[i + 1]
        
        if end == start:
            item_factors[i] = 0.0
            continue

        lambda_val = lambda_array[i]
        
        A = np.eye(K, dtype=np.float32) * np.float32(lambda_val)
        b = np.zeros(K, dtype=np.float32)
        
        bias_i = item_biases[i]
        
        for idx in range(start, end):
            user = item_user_indices[idx]
            rating = item_ratings[idx]
            weight = item_weights[idx]
            
            eff_weight = tau_f32 * np.float32(weight)
            
            u_m = user_factors[user]
            residual = rating - global_mean - user_biases[user] - bias_i
            
            for k1 in range(K):
                val_k1 = u_m[k1]
                b[k1] += eff_weight * residual * val_k1
                for k2 in range(K):
                    A[k1, k2] += eff_weight * val_k1 * u_m[k2]
        
        item_factors[i] = np.linalg.solve(A, b)


class MatrixFactorizationModel:
    def __init__(
        self,
        K: int = 10,
        lambda_: float = 0.1,
        tau: float = 1.0,  # Precision of ratings, default 1.0 implies standard MSE
        gamma: float = 0.1, # Separate bias reg if needed, otherwise use lambda_
        n_iters: int = 20,
        verbose: bool = True
    ):
        self.K = K
        self.lambda_ = lambda_
        self.tau = tau
        self.gamma = gamma if gamma is not None else lambda_
        self.n_iters = n_iters
        self.verbose = verbose
        
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = 0.0
        
        self.loss_history = []
        self.train_rmse_history = []
        self.test_rmse_history = []
        
    def fit(
        self, 
        rating_matrix: RatingMatrix, 
        test_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        weight_matrix: Optional[RatingMatrix] = None,
        alpha: float = 0.0,
        callback: Optional[Callable[[int, float], bool]] = None
    ):
        """
        Train the model using ALS.
        
        Args:
            rating_matrix: Training data
            test_data: Optional validation/test data for RMSE tracking
            weight_matrix: Optional sample weights
            alpha: Power-law exponent for count-based regularization
            callback: Optional function called after each iteration.
                     Signature: callback(iteration, val_rmse) -> should_stop
                     If callback returns True, training stops early (e.g., HPO pruning).
        """
        # Data preparation
        user_indptr = rating_matrix.user_index.indptr
        user_item_indices = rating_matrix.user_index.indices
        user_ratings = rating_matrix.user_index.data
        
        item_indptr = rating_matrix.item_index.indptr
        item_user_indices = rating_matrix.item_index.indices
        item_ratings = rating_matrix.item_index.data
        
        self.global_mean = rating_matrix.global_mean
        n_users = rating_matrix.n_users
        n_items = rating_matrix.n_items
        
        # Initialization
        # Initialize factors randomly (small values)
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.K)).astype(np.float32)
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.K)).astype(np.float32)
        
        self.user_biases = np.zeros(n_users, dtype=np.float32)
        self.item_biases = np.zeros(n_items, dtype=np.float32)
        
        for iteration in range(self.n_iters):
            start_time = time.time()
            
            # 1. Update User Biases (Latent)
            update_user_biases_latent(
                user_indptr, user_item_indices, user_ratings,
                self.item_biases, self.user_factors, self.item_factors,
                self.user_biases, self.global_mean, self.lambda_, self.gamma
            )
            
            # Update Item Biases (Latent)
            update_item_biases_latent(
                item_indptr, item_user_indices, item_ratings,
                self.user_biases, self.user_factors, self.item_factors,
                self.item_biases, self.global_mean, self.lambda_, self.gamma
            )
            
            # 2. Update User Factors
            # Check if advanced features are needed
            use_advanced = (weight_matrix is not None) or (alpha != 0.0)
            
            if not use_advanced:
                update_user_factors(
                    user_indptr, user_item_indices, user_ratings,
                    self.user_biases, self.item_biases, 
                    self.item_factors, self.user_factors,
                    self.global_mean, self.lambda_, self.tau
                )
            else:
                # Prepare inputs for advanced kernel
                if weight_matrix is not None:
                    u_weights = weight_matrix.user_index.data
                else:
                    u_weights = np.ones_like(user_ratings, dtype=np.float32)
                    
                if alpha != 0.0:
                    # Calculate per-user lambda
                    diffs = np.diff(user_indptr)
                    # Example: lambda_u = lambda * (count / avg_count)^alpha ?? 
                    # Simpler: lambda_u = lambda * (1 + log(1+count))?
                    # Using direct power law as requested: lambda * count^alpha
                    # Note: if alpha > 0, frequent users get MORE regularization.
                    # if alpha < 0, frequent users get LESS regularization.
                    user_lambdas = self.lambda_ * (diffs.astype(np.float32) ** alpha)
                else:
                    user_lambdas = np.full(n_users, self.lambda_, dtype=np.float32)

                update_user_factors_advanced(
                    user_indptr, user_item_indices, user_ratings, u_weights,
                    self.user_biases, self.item_biases, 
                    self.item_factors, self.user_factors,
                    self.global_mean, user_lambdas, self.tau
                )
            
            # 3. Update Item Factors
            if not use_advanced:
                update_item_factors(
                    item_indptr, item_user_indices, item_ratings,
                    self.user_biases, self.item_biases, 
                    self.user_factors, self.item_factors,
                    self.global_mean, self.lambda_, self.tau
                )
            else:
                if weight_matrix is not None:
                    i_weights = weight_matrix.item_index.data
                else:
                    i_weights = np.ones_like(item_ratings, dtype=np.float32)
                
                if alpha != 0.0:
                    diffs = np.diff(item_indptr)
                    item_lambdas = self.lambda_ * (diffs.astype(np.float32) ** alpha)
                else:
                    item_lambdas = np.full(n_items, self.lambda_, dtype=np.float32)
                    
                update_item_factors_advanced(
                    item_indptr, item_user_indices, item_ratings, i_weights,
                    self.user_biases, self.item_biases, 
                    self.user_factors, self.item_factors,
                    self.global_mean, item_lambdas, self.tau
                )
            
            # 4. Compute Metrics
            loss = compute_loss_with_factors(
                user_indptr, user_item_indices, user_ratings,
                self.user_biases, self.item_biases,
                self.user_factors, self.item_factors,
                self.global_mean, self.lambda_, self.tau, self.gamma
            )
            self.loss_history.append(loss)
            
            # Train RMSE
            # We can't compute full predictions for large matrix, need batch or iterate
            # Let's use a sample or just iterate if data is small (MovieLens Small is ~100k, fine)
            # But for optimization, let's implement a compute_rmse_latent kernel?
            # Or use compute_loss's squared error part.
            # compute_loss returns regularized loss.
            # Let's just create a helper for RMSE.
            train_rmse = self._compute_rmse(user_indptr, user_item_indices, user_ratings)
            self.train_rmse_history.append(train_rmse)
            
            # Test RMSE
            if test_data:
                test_users, test_items, test_ratings = test_data
                test_preds = predict_batch(
                    test_users, test_items, 
                    self.user_biases, self.item_biases,
                    self.user_factors, self.item_factors,
                    self.global_mean
                )
                test_rmse = np.sqrt(np.mean((test_ratings - test_preds)**2))
                self.test_rmse_history.append(test_rmse)
            
            if self.verbose:
                elapsed = time.time() - start_time
                msg = f"Iter {iteration+1}: Loss={loss:.2f}, Train RMSE={train_rmse:.4f}"
                if test_data:
                    msg += f", Test RMSE={test_rmse:.4f}"
                msg += f" ({elapsed:.2f}s)"
                print(msg)
            
            # HPO Callback for intermediate reporting and early stopping
            if callback is not None and test_data is not None:
                should_stop = callback(iteration + 1, test_rmse)
                if should_stop:
                    if self.verbose:
                        print(f"Early stopping at iteration {iteration + 1} (pruned)")
                    break

    def _compute_rmse(self, user_indptr, user_item_indices, user_ratings):
        # Kernel for RMSE
        return compute_rmse_latent(
            user_indptr, user_item_indices, user_ratings,
            self.user_biases, self.item_biases,
            self.user_factors, self.item_factors,
            self.global_mean
        )

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        """
        Predict ratings for given user-item pairs.
        """
        return predict_batch(
            user_indices, item_indices,
            self.user_biases, self.item_biases,
            self.user_factors, self.item_factors,
            self.global_mean
        )
        
    def save(self, filepath: str):
        """Save model to disk."""
        np.savez_compressed(
            filepath,
            user_biases=self.user_biases,
            item_biases=self.item_biases,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            global_mean=np.array([self.global_mean]),
            K=np.array([self.K]),
            hyperparams=np.array([self.lambda_, self.tau, self.gamma])
        )
        print(f"Model saved to {filepath}")
        
    @staticmethod
    def load(filepath: str) -> 'MatrixFactorizationModel':
        """Load model from disk."""
        data = np.load(filepath)
        lambda_, tau, gamma = data['hyperparams']
        K = int(data['K'][0])
        
        model = MatrixFactorizationModel(
            K=K, 
            lambda_=float(lambda_), 
            tau=float(tau), 
            gamma=float(gamma),
            verbose=False
        )
        model.user_biases = data['user_biases']
        model.item_biases = data['item_biases']
        model.user_factors = data['user_factors']
        model.item_factors = data['item_factors']
        model.global_mean = float(data['global_mean'][0])
        
        return model


# Additional Kernels for Biases with Factors

@njit(parallel=True)
def update_user_biases_latent(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_ratings: np.ndarray,
    item_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_biases: np.ndarray,
    global_mean: float,
    lambda_: float,
    gamma: float
):
    """
    Update user biases with proper lambda_ weighting (matching oldcode).
    
    b_u = (lambda_ * sum(r - mu - b_i - u.v)) / (lambda_ * count + gamma)
    
    Where:
        lambda_ is the data term weight
        gamma is the bias regularization
    """
    n_users = len(user_indptr) - 1
    lambda_f32 = np.float32(lambda_)
    gamma_f32 = np.float32(gamma)
    
    for u in prange(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        count = end - start
        
        if count == 0:
            user_biases[u] = 0.0
            continue
            
        total_residual = 0.0
        u_factor = user_factors[u]
        
        for idx in range(start, end):
            item = user_item_indices[idx]
            rating = user_ratings[idx]
            
            # Prediction from other parts: mu + b_i + u.v
            part_pred = global_mean + item_biases[item]
            
            dot = 0.0
            for k in range(len(u_factor)):
                dot += u_factor[k] * item_factors[item, k]
                
            total_residual += (rating - part_pred - dot)
        
        # Oldcode formula: b_u = (lambda_ * sum_residual) / (lambda_ * count + gamma)
        user_biases[u] = (lambda_f32 * total_residual) / (lambda_f32 * np.float32(count) + gamma_f32)

@njit(parallel=True)
def update_item_biases_latent(
    item_indptr: np.ndarray,
    item_user_indices: np.ndarray,
    item_ratings: np.ndarray,
    user_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    item_biases: np.ndarray,
    global_mean: float,
    lambda_: float,
    gamma: float
):
    """
    Update item biases with proper lambda_ weighting (matching oldcode).
    
    b_i = (lambda_ * sum(r - mu - b_u - u.v)) / (lambda_ * count + gamma)
    
    Where:
        lambda_ is the data term weight
        gamma is the bias regularization
    """
    n_items = len(item_indptr) - 1
    lambda_f32 = np.float32(lambda_)
    gamma_f32 = np.float32(gamma)
    
    for i in prange(n_items):
        start = item_indptr[i]
        end = item_indptr[i + 1]
        count = end - start
        
        if count == 0:
            item_biases[i] = 0.0
            continue
            
        total_residual = 0.0
        i_factor = item_factors[i]
        
        for idx in range(start, end):
            user = item_user_indices[idx]
            rating = item_ratings[idx]
            
            part_pred = global_mean + user_biases[user]
            
            dot = 0.0
            for k in range(len(i_factor)):
                dot += user_factors[user, k] * i_factor[k]
                
            total_residual += (rating - part_pred - dot)
        
        # Oldcode formula: b_i = (lambda_ * sum_residual) / (lambda_ * count + gamma)
        item_biases[i] = (lambda_f32 * total_residual) / (lambda_f32 * np.float32(count) + gamma_f32)

@njit
def compute_rmse_latent(
    user_indptr, user_item_indices, user_ratings,
    user_biases, item_biases,
    user_factors, item_factors,
    global_mean
):
    n_users = len(user_indptr) - 1
    sse = 0.0
    count = 0
    
    for u in range(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        count += (end - start)
        
        if end > start:
            u_vec = user_factors[u]
            u_bias = user_biases[u]
            
            for idx in range(start, end):
                item = user_item_indices[idx]
                rating = user_ratings[idx]
                
                i_vec = item_factors[item]
                i_bias = item_biases[item]
                
                dot = 0.0
                for k in range(len(u_vec)):
                    dot += u_vec[k] * i_vec[k]
                    
                pred = global_mean + u_bias + i_bias + dot
                diff = rating - pred
                sse += diff * diff
                
    if count == 0:
        return 0.0
    return np.sqrt(sse / count)


@njit
def compute_bias_only_rmse(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    global_mean: float
) -> float:
    """
    Compute RMSE using only biases (no latent factors).
    
    This computes: RMSE(r, μ + b_u + b_i)
    
    Used for diagnostics to measure how much of the signal is captured
    by biases alone vs. the latent factor interactions.
    """
    n_users = len(user_indptr) - 1
    sse = 0.0
    count = 0
    
    for u in range(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        count += (end - start)
        
        if end > start:
            u_bias = user_biases[u]
            
            for idx in range(start, end):
                item = user_item_indices[idx]
                rating = user_ratings[idx]
                
                i_bias = item_biases[item]
                
                # Prediction using only biases (no dot product)
                pred = global_mean + u_bias + i_bias
                diff = rating - pred
                sse += diff * diff
                
    if count == 0:
        return 0.0
    return np.sqrt(sse / count)



@njit(parallel=True)
def update_item_bias_bins(
    item_indptr: np.ndarray,
    item_user_indices: np.ndarray,
    item_ratings: np.ndarray,
    item_time_bins: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    item_bias_bins: np.ndarray,
    global_mean: float,
    gamma_bin: float
) -> None:
    """
    Update time-bin-specific item biases using ALS.
    
    For each item i and time bin t:
        b_i,t = sum(r - μ - b_u - b_i) / (count + γ_bin)
    
    Args:
        item_indptr: CSR row pointers for items
        item_user_indices: User indices for each rating
        item_ratings: Rating values
        item_time_bins: Time bin for each rating
        user_biases: User biases
        item_biases: Static item biases
        item_bias_bins: Item bias bins to update (n_items, n_bins)
        global_mean: Global mean rating
        gamma_bin: Regularization for bin biases
    """
    n_items = len(item_indptr) - 1
    n_bins = item_bias_bins.shape[1]
    gamma_f32 = np.float32(gamma_bin)
    
    for i in prange(n_items):
        start = item_indptr[i]
        end = item_indptr[i + 1]
        
        if end == start:
            continue
        
        # Accumulators per bin
        bin_sums = np.zeros(n_bins, dtype=np.float32)
        bin_counts = np.zeros(n_bins, dtype=np.float32)
        
        b_i = item_biases[i]
        
        for idx in range(start, end):
            user = item_user_indices[idx]
            rating = item_ratings[idx]
            t_bin = item_time_bins[idx]
            
            # Residual without bin bias
            residual = rating - global_mean - user_biases[user] - b_i
            
            bin_sums[t_bin] += residual
            bin_counts[t_bin] += 1.0
        
        # Update each bin's bias
        for t in range(n_bins):
            if bin_counts[t] > 0:
                item_bias_bins[i, t] = bin_sums[t] / (bin_counts[t] + gamma_f32)
            else:
                item_bias_bins[i, t] = 0.0


@njit(parallel=True)
def predict_batch_temporal(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    time_bins: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    item_bias_bins: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    global_mean: float
) -> np.ndarray:
    """
    Predict ratings with temporal item biases.
    
    r̂_ui(t) = μ + b_u + b_i + b_i,bin(t) + u^T v
    """
    n = len(user_indices)
    preds = np.empty(n, dtype=np.float32)
    K = user_factors.shape[1]
    
    for i in prange(n):
        u = user_indices[i]
        item = item_indices[i]
        t_bin = time_bins[i]
        
        pred = global_mean + user_biases[u] + item_biases[item]
        
        # Add temporal bias
        pred += item_bias_bins[item, t_bin]
        
        # Dot product
        for k in range(K):
            pred += user_factors[u, k] * item_factors[item, k]
        
        preds[i] = pred
    
    return preds


@njit
def compute_rmse_temporal(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    user_ratings: np.ndarray,
    user_time_bins: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    item_bias_bins: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    global_mean: float
) -> float:
    """Compute RMSE with temporal biases."""
    n_users = len(user_indptr) - 1
    K = user_factors.shape[1]
    sse = 0.0
    count = 0
    
    for u in range(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        count += (end - start)
        
        if end > start:
            u_bias = user_biases[u]
            
            for idx in range(start, end):
                item = user_item_indices[idx]
                rating = user_ratings[idx]
                t_bin = user_time_bins[idx]
                
                pred = global_mean + u_bias + item_biases[item] + item_bias_bins[item, t_bin]
                
                for k in range(K):
                    pred += user_factors[u, k] * item_factors[item, k]
                
                diff = rating - pred
                sse += diff * diff
    
    if count == 0:
        return 0.0
    return np.sqrt(sse / count)


@njit(parallel=True)
def compute_user_implicit_sum(
    user_indptr: np.ndarray,
    user_item_indices: np.ndarray,
    implicit_factors: np.ndarray
) -> np.ndarray:
    """
    Compute |N(u)|^(-0.5) * Σ y_j for each user.
    
    This is the implicit feedback contribution to user representation.
    
    Args:
        user_indptr: CSR row pointers for users
        user_item_indices: Item indices for each rating
        implicit_factors: Implicit item factors Y (n_items, K)
    
    Returns:
        User implicit sums (n_users, K)
    """
    n_users = len(user_indptr) - 1
    K = implicit_factors.shape[1]
    result = np.zeros((n_users, K), dtype=np.float32)
    
    for u in prange(n_users):
        start = user_indptr[u]
        end = user_indptr[u + 1]
        count = end - start
        
        if count > 0:
            norm = np.float32(1.0 / np.sqrt(count))
            
            for idx in range(start, end):
                item = user_item_indices[idx]
                for k in range(K):
                    result[u, k] += implicit_factors[item, k]
            
            for k in range(K):
                result[u, k] *= norm
    
    return result


@njit(parallel=True)
def update_implicit_factors(
    item_indptr: np.ndarray,
    item_user_indices: np.ndarray,
    item_ratings: np.ndarray,
    user_indptr: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    implicit_factors: np.ndarray,
    user_implicit_sums: np.ndarray,
    global_mean: float,
    lambda_: float,
    tau_implicit: float
) -> None:
    """
    Update implicit item factors Y using ALS.
    
    For each item j, solve for y_j that minimizes:
        sum_u [(r - pred)^2 * |N(u)|^(-1)] + τ * ||y_j||^2
    
    where pred = μ + b_u + b_i + (u + user_implicit_sum)^T v
    """
    n_items = len(item_indptr) - 1
    K = implicit_factors.shape[1]
    tau_f32 = np.float32(tau_implicit)
    lambda_f32 = np.float32(lambda_)
    
    for j in prange(n_items):
        start = item_indptr[j]
        end = item_indptr[j + 1]
        
        if end == start:
            implicit_factors[j] = 0.0
            continue
        
        # Build normal equations
        A = np.eye(K, dtype=np.float32) * tau_f32
        b = np.zeros(K, dtype=np.float32)
        
        for idx in range(start, end):
            user = item_user_indices[idx]
            rating = item_ratings[idx]
            
            # User's rating count for normalization
            u_start = user_indptr[user]
            u_end = user_indptr[user + 1]
            u_count = u_end - u_start
            
            if u_count == 0:
                continue
            
            norm = np.float32(1.0 / np.sqrt(u_count))
            weight = lambda_f32 / np.float32(u_count)  # Scale by |N(u)|^(-1)
            
            v_j = item_factors[j]
            
            # Effective user factor: u + sum_j y_j / sqrt(|N(u)|)
            eff_u = np.zeros(K, dtype=np.float32)
            for k in range(K):
                eff_u[k] = user_factors[user, k] + user_implicit_sums[user, k]
            
            # Residual (without this item's y_j contribution)
            pred_base = global_mean + user_biases[user] + item_biases[j]
            for k in range(K):
                pred_base += eff_u[k] * v_j[k]
            
            residual = rating - pred_base
            
            # Contribution to normal equations
            for k1 in range(K):
                b[k1] += weight * residual * v_j[k1] * norm
                for k2 in range(K):
                    A[k1, k2] += weight * (norm * v_j[k1]) * (norm * v_j[k2])
        
        implicit_factors[j] = np.linalg.solve(A, b)


@njit(parallel=True)
def predict_batch_svdpp(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_implicit_sums: np.ndarray,
    global_mean: float
) -> np.ndarray:
    """
    Predict ratings with SVD++ model.
    
    r̂_ui = μ + b_u + b_i + (u + |N(u)|^(-0.5) Σ y_j)^T v
    """
    n = len(user_indices)
    preds = np.empty(n, dtype=np.float32)
    K = user_factors.shape[1]
    
    for i in prange(n):
        u = user_indices[i]
        item = item_indices[i]
        
        pred = global_mean + user_biases[u] + item_biases[item]
        
        # Dot product with enhanced user representation
        for k in range(K):
            eff_u_k = user_factors[u, k] + user_implicit_sums[u, k]
            pred += eff_u_k * item_factors[item, k]
        
        preds[i] = pred
    
    return preds



@njit(parallel=True)
def update_item_factors_with_features(
    item_indptr: np.ndarray,
    item_user_indices: np.ndarray,
    item_ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    feature_matrix_indptr: np.ndarray,
    feature_matrix_indices: np.ndarray,
    feature_embeddings_T: np.ndarray, # Shape (K, F) for easier dot product ?? No, W is (F, K).
                                      # Wait, if we want W*f_n, W should be (K, F) or we transpose.
                                      # Let's say feature_factors is W (F x K).
    feature_factors: np.ndarray,      # W (F x K)
    global_mean: float,
    lambda_: float,
    tau: float,
    lambda_features: float
) -> None:
    """
    Update item factors with regularization towards feature embeddings.
    
    Objective per item i:
    min sum_u (r_ui - ... - u_u.v_i)^2 + tau ||v_i||^2 + lambda_features ||v_i - W f_i||^2
    
    Deriv w.r.t v_i:
    -2 lambda sum_u (res) u_u + 2 tau v_i + 2 lambda_features (v_i - W f_i) = 0
    
    (lambda sum u u^T + (tau + lambda_features) I) v_i = lambda sum (res_without_v) u + lambda_features W f_i
    
    Args:
        feature_matrix: CSR structure for features (N items x F features)
        feature_factors: W matrix (F features x K latent)
    """
    n_items = len(item_indptr) - 1
    K = user_factors.shape[1]
    
    # Regularization term offset
    # The A matrix diagonal gets added: (tau + lambda_features)
    reg_val = np.float32(tau + lambda_features)
    diag_reg = np.eye(K, dtype=np.float32) * reg_val
    
    lambda_f32 = np.float32(lambda_)
    lambda_feat_f32 = np.float32(lambda_features)
    
    for i in prange(n_items):
        start = item_indptr[i]
        end = item_indptr[i + 1]
        
        # Build A and b
        A = np.copy(diag_reg)
        b = np.zeros(K, dtype=np.float32)
        
        # 1. Add feature prior to b: lambda_features * W * f_i
        # f_i is sparse vector for item i
        f_start = feature_matrix_indptr[i]
        f_end = feature_matrix_indptr[i + 1]
        
        # Compute W * f_i
        # f_i is binary 1.0 at indices. So we just sum rows of W.
        w_f_i = np.zeros(K, dtype=np.float32)
        
        for idx in range(f_start, f_end):
            feat_idx = feature_matrix_indices[idx]
            # feat_val = feature_matrix_data[idx] # Assume binary 1.0 for now as per plan
            # W is (F, K). Row feat_idx.
            for k in range(K):
                w_f_i[k] += feature_factors[feat_idx, k]
        
        for k in range(K):
            b[k] += lambda_feat_f32 * w_f_i[k]
            
        # 2. Add interaction terms
        if end > start:
            bias_i = item_biases[i]
            
            for idx in range(start, end):
                user = item_user_indices[idx]
                rating = item_ratings[idx]
                
                u_m = user_factors[user]
                # Residual without v_i part: r - (mu + b_u + b_i)
                residual = rating - global_mean - user_biases[user] - bias_i
                
                # A += lambda * u * u^T
                # b += lambda * residual * u
                for k1 in range(K):
                    val_k1 = u_m[k1]
                    b[k1] += lambda_f32 * residual * val_k1
                    for k2 in range(K):
                        A[k1, k2] += lambda_f32 * val_k1 * u_m[k2]
        
        # Solve
        item_factors[i] = np.linalg.solve(A, b)


class FeatureAwareMatrixFactorizationModel(MatrixFactorizationModel):
    def __init__(
        self,
        K: int = 10,
        lambda_: float = 0.1,
        tau: float = 1.0,
        gamma: float = 0.1,
        lambda_features: float = 1.0, # Regularization strength for features
        n_iters: int = 20,
        verbose: bool = True
    ):
        super().__init__(K, lambda_, tau, gamma, n_iters, verbose)
        self.lambda_features = lambda_features
        self.feature_factors = None # W matrix (n_features x K)
        
    def fit(
        self,
        rating_matrix: RatingMatrix,
        feature_matrix: sparse.csr_matrix,
        test_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        callback: Optional[Callable[[int, float], bool]] = None
    ):
        """
        Train Feature-Aware ALS.
        
        Args:
            rating_matrix: Ratings
            feature_matrix: Sparse feature matrix (n_items x n_features)
        """
        from scipy import sparse
        
        # Data prep
        user_indptr = rating_matrix.user_index.indptr
        user_item_indices = rating_matrix.user_index.indices
        user_ratings = rating_matrix.user_index.data
        
        item_indptr = rating_matrix.item_index.indptr
        item_user_indices = rating_matrix.item_index.indices
        item_ratings = rating_matrix.item_index.data
        
        # Feature matrix prep (ensure CSR for fast row slicing in kernel)
        if not sparse.isspmatrix_csr(feature_matrix):
            feature_matrix = feature_matrix.tocsr()
            
        feat_indptr = feature_matrix.indptr
        feat_indices = feature_matrix.indices
        # feat_data = feature_matrix.data # Assuming binary for now
        
        self.global_mean = rating_matrix.global_mean
        n_users = rating_matrix.n_users
        n_items = rating_matrix.n_items
        n_features = feature_matrix.shape[1]
        
        # Initialize
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.K)).astype(np.float32)
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.K)).astype(np.float32)
        self.feature_factors = np.zeros((n_features, self.K), dtype=np.float32) 
        # Initialize W? Maybe regularize towards 0. zero is fine.
        
        self.user_biases = np.zeros(n_users, dtype=np.float32)
        self.item_biases = np.zeros(n_items, dtype=np.float32)
        
        # Precompute F^T F for feature update step
        # Since F is constant, F^T F is constant.
        # W update: (F^T F + lambda_W I) W = F^T V
        # Wait, the loss term is lambda_features ||v - W f||^2.
        # Sum over items i: lambda_features ||v_i - W f_i||^2
        # = lambda_features * sum_i (v_i - W f_i)^T (v_i - W f_i)
        # This is essentially Ridge Regression: Targets=V, Inputs=F.
        # Weights = W.
        # But we update W.
        # W_opt = (F^T F + (lambda_W / lambda_features) I)^-1 F^T V ??
        # No, wait.
        # Loss for W: lambda_features * sum_i ||v_i - W f_i||^2 + lambda_W ||W||^2 ?
        # course.md doesn't explicitly mention lambda_W (reg on W itself).
        # It says "Prior -> features -> ...". Implicitly W has a prior too?
        # "Learn embedding for each feature".
        # Let's assume a small regularization on W for stability, e.g. same as tau?
        # Or just use the lambda_features term.
        # If we only have ||v - W f||^2, and F is rank deficient, W is unconstrained.
        # We should add lambda_W ||W||^2. Let's use tau for consistency or a small value.
        # Let's use tau_w = 0.001 * lambda_features or just 0.1.
        tau_w = 0.1 # Small ridge penalty on W
        
        # Precompute P_inv = (F^T F + reg I)^-1 F^T
        # F is (N x D). D small (20). N large.
        # F^T F is (D x D).
        # P_inv is (D x N). N is large. We can't store P_inv dense.
        # But F is sparse.
        # F^T V is (D x K). We can compute this term efficiently: F.T @ V.
        # (F^T F) is small.
        # So update step:
        # 1. LHS = (F.T @ F) + eye * reg  -> Shape (D, D)
        # 2. RHS = F.T @ item_factors	-> Shape (D, K)
        # 3. W = solve(LHS, RHS)		 -> Shape (D, K)
        
        F = feature_matrix
        FTF = (F.T @ F).toarray().astype(np.float32)
        # Add regularization to diagonal
        # Note: if lambda_features is part of the loss scale, we divide by it?
        # Loss term: alpha ||v - Wf||^2 + beta ||W||^2
        # d/dW = -2 alpha F^T(V - FW) + 2 beta W = 0
        # -alpha F^T V + alpha F^T F W + beta W = 0
        # (alpha F^T F + beta I) W = alpha F^T V
        # (F^T F + beta/alpha I) W = F^T V
        # Let beta/alpha = ridge_ratio.
        # If we take beta small, say 1e-3. alpha = lambda_features.
        ridge_ratio = 1e-3 / self.lambda_features
        
        LHS = FTF + np.eye(n_features, dtype=np.float32) * ridge_ratio
        
        # Invert LHS once? (D x D) is tiny.
        LHS_inv = np.linalg.inv(LHS)
        
        for iteration in range(self.n_iters):
            start_time = time.time()
            
            # 1. Update Biases (Standard or Latent)
            # Use latent updates for consistency
            update_user_biases_latent(
                user_indptr, user_item_indices, user_ratings,
                self.item_biases, self.user_factors, self.item_factors,
                self.user_biases, self.global_mean, self.lambda_, self.gamma
            )
            
            update_item_biases_latent(
                item_indptr, item_user_indices, item_ratings,
                self.user_biases, self.user_factors, self.item_factors,
                self.item_biases, self.global_mean, self.lambda_, self.gamma
            )
            
            # 2. Update User Factors (Standard, no features here)
            update_user_factors(
                user_indptr, user_item_indices, user_ratings,
                self.user_biases, self.item_biases, 
                self.item_factors, self.user_factors,
                self.global_mean, self.lambda_, self.tau
            )
            
            # 3. Update Feature Factors (W)
            # W = (F^T F + reg)^-1 F^T V
            # V is current item_factors
            # F is feature_matrix
            # This is a global update, fast.
            # RHS = F.T @ V
            RHS = F.T @ self.item_factors
            self.feature_factors = LHS_inv @ RHS
            
            # 4. Update Item Factors (Regularized towards W f_i)
            update_item_factors_with_features(
                item_indptr, item_user_indices, item_ratings,
                self.user_biases, self.item_biases,
                self.user_factors, self.item_factors,
                feat_indptr, feat_indices, None,
                self.feature_factors,
                self.global_mean, self.lambda_, self.tau, self.lambda_features
            )
            
            # 5. Metrics
            # Re-use compute_loss? 
            # Ideally loss should include feature terms.
            # Base loss
            loss = compute_loss_with_factors(
                user_indptr, user_item_indices, user_ratings,
                self.user_biases, self.item_biases,
                self.user_factors, self.item_factors,
                self.global_mean, self.lambda_, self.tau, self.gamma
            )
            # Add feature loss: 0.5 * lambda_features * ||V - FW||^2
            # V - FW
            diff = self.item_factors - (F @ self.feature_factors)
            feat_loss = 0.5 * self.lambda_features * np.sum(diff**2)
            
            total_loss = loss + feat_loss
            self.loss_history.append(total_loss)
            
            train_rmse = self._compute_rmse(user_indptr, user_item_indices, user_ratings)
            self.train_rmse_history.append(train_rmse)
            
            if test_data:
                test_users, test_items, test_ratings = test_data
                test_preds = predict_batch(
                    test_users, test_items, 
                    self.user_biases, self.item_biases,
                    self.user_factors, self.item_factors,
                    self.global_mean
                )
                test_rmse = np.sqrt(np.mean((test_ratings - test_preds)**2))
                self.test_rmse_history.append(test_rmse)
                
            if self.verbose:
                elapsed = time.time() - start_time
                msg = f"Iter {iteration+1}: Loss={total_loss:.2f}, Train RMSE={train_rmse:.4f}"
                if test_data:
                    msg += f", Test RMSE={test_rmse:.4f}"
                msg += f" ({elapsed:.2f}s)"
                print(msg)
                
            if callback:
                if callback(iteration+1, test_rmse if test_data else 0.0):
                    break
