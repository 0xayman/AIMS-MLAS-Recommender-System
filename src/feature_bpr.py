"""
Feature-Aware BPR Model.

Extends BPR to incorporate item features (e.g., genres).
Item representation: h_i = v_i + W * f_i
"""

import numpy as np
from numba import njit, prange
from typing import List, Tuple, Optional, Set
import time
from scipy import sparse

from .bpr import BPRModel, _sigmoid, _sample_negative

@njit(parallel=True)
def _bpr_update_feature_batch(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    feature_embeddings: np.ndarray,
    samples: np.ndarray,
    feat_indptr: np.ndarray,
    feat_indices: np.ndarray,
    learning_rate: float,
    lambda_reg: float,
    lambda_features: float
) -> float:
    """
    Perform batched BPR updates with features.
    
    Item score: s_ui = u . (v_i + sum(w_f))
    """
    n_samples = samples.shape[0]
    K = user_factors.shape[1]
    total_loss = 0.0
    
    for s in prange(n_samples):
        u = samples[s, 0]
        i = samples[s, 1]
        j = samples[s, 2]
        
        # 1. Compute composite item vectors
        # Pos item i
        start_i = feat_indptr[i]
        end_i = feat_indptr[i+1]
        
        # h_i = v_i + sum(w_f)
        h_i = item_factors[i].copy()
        for idx in range(start_i, end_i):
            f_idx = feat_indices[idx]
            dim_f = feature_embeddings[f_idx]
            for k in range(K):
                h_i[k] += dim_f[k]
                
        # Neg item j
        start_j = feat_indptr[j]
        end_j = feat_indptr[j+1]
        
        h_j = item_factors[j].copy()
        for idx in range(start_j, end_j):
            f_idx = feat_indices[idx]
            dim_f = feature_embeddings[f_idx]
            for k in range(K):
                h_j[k] += dim_f[k]
        
        # 2. Compute score difference
        x_uij = 0.0
        for k in range(K):
            x_uij += user_factors[u, k] * (h_i[k] - h_j[k])
            
        # 3. Loss
        sigmoid_x = _sigmoid(-x_uij)
        loss = -np.log(_sigmoid(x_uij) + 1e-10)
        total_loss += loss
        
        # 4. Gradients
        # For user u: dL/du = sig(-x) * (h_i - h_j) + reg * u
        # For item v_i: dL/dv_i = sig(-x) * u + reg * v_i
        # For feature w_f (in i): dL/dw_f = sig(-x) * u + reg_f * w_f
        
        # Common term: multiplier = sig(-x)
        mult = sigmoid_x
        
        # Update User
        for k in range(K):
            grad_u = mult * (h_i[k] - h_j[k]) - lambda_reg * user_factors[u, k]
            user_factors[u, k] += learning_rate * grad_u
            
        # Update Item i (v_i)
        for k in range(K):
            grad_i = mult * user_factors[u, k] - lambda_reg * item_factors[i, k]
            item_factors[i, k] += learning_rate * grad_i
            
        # Update Item j (v_j)
        for k in range(K):
            grad_j = mult * (-user_factors[u, k]) - lambda_reg * item_factors[j, k]
            item_factors[j, k] += learning_rate * grad_j
            
        # Update Features for i
        for idx in range(start_i, end_i):
            f_idx = feat_indices[idx]
            for k in range(K):
                # Gradient is same as v_i but with feature regularization
                grad_f = mult * user_factors[u, k] - lambda_features * feature_embeddings[f_idx, k]
                feature_embeddings[f_idx, k] += learning_rate * grad_f
                
        # Update Features for j
        for idx in range(start_j, end_j):
            f_idx = feat_indices[idx]
            for k in range(K):
                # Gradient is same as v_j (so negative u)
                grad_f = mult * (-user_factors[u, k]) - lambda_features * feature_embeddings[f_idx, k]
                feature_embeddings[f_idx, k] += learning_rate * grad_f
                
    return total_loss / n_samples

class FeatureAwareBPRModel(BPRModel):
    def __init__(
        self,
        K: int = 20,
        learning_rate: float = 0.05,
        lambda_reg: float = 0.01,
        lambda_features: float = 0.01,
        n_epochs: int = 20,
        n_samples_per_epoch: Optional[int] = None,
        verbose: bool = True,
        seed: int = 42
    ):
        super().__init__(K, learning_rate, lambda_reg, n_epochs, n_samples_per_epoch, verbose, seed)
        self.lambda_features = lambda_features
        self.feature_embeddings = None
        self.feature_matrix = None # Store for prediction
        
    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        feature_matrix: sparse.csr_matrix,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None
    ) -> 'FeatureAwareBPRModel':
        
        # Ensure feature matrix is CSR
        if not sparse.isspmatrix_csr(feature_matrix):
            feature_matrix = feature_matrix.tocsr()
        self.feature_matrix = feature_matrix
        
        feat_indptr = feature_matrix.indptr
        feat_indices = feature_matrix.indices
        n_features = feature_matrix.shape[1]
        
        # Base Fit Setup
        rng = np.random.default_rng(self.seed)
        self.n_users = n_users if n_users is not None else int(user_ids.max()) + 1
        self.n_items = n_items if n_items is not None else int(item_ids.max()) + 1
        
        # Dimensions check
        if self.n_items > feature_matrix.shape[0]:
            raise ValueError(f"Feature matrix has fewer items ({feature_matrix.shape[0]}) than dataset ({self.n_items})")
            
        # Initialize Base Factors
        self.user_factors = rng.normal(0, 0.01, (self.n_users, self.K)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.01, (self.n_items, self.K)).astype(np.float32)
        
        # Initialize Feature Embeddings
        self.feature_embeddings = rng.normal(0, 0.01, (n_features, self.K)).astype(np.float32)
        
        # Prepare sampling
        from .bpr import _generate_training_samples
        user_positive_items: List[Set[int]] = [set() for _ in range(self.n_users)]
        for u, i in zip(user_ids, item_ids):
            user_positive_items[u].add(i)
        user_positive_arrays = [np.array(list(items), dtype=np.int32) for items in user_positive_items]
        
        n_interactions = len(user_ids)
        samples_per_epoch = self.n_samples_per_epoch or n_interactions * 10
        self.loss_history = []
        
        if self.verbose:
            print(f"Training Feature-Aware BPR: K={self.K}, lr={self.learning_rate}, λ={self.lambda_reg}, λ_feat={self.lambda_features}")
        
        for epoch in range(self.n_epochs):
            start_time = time.time()
            
            samples = _generate_training_samples(
                user_positive_arrays,
                self.n_items,
                samples_per_epoch,
                rng
            )
            
            loss = _bpr_update_feature_batch(
                self.user_factors,
                self.item_factors,
                self.feature_embeddings,
                samples,
                feat_indptr,
                feat_indices,
                self.learning_rate,
                self.lambda_reg,
                self.lambda_features
            )
            
            self.loss_history.append(loss)
            if self.verbose:
                print(f"Epoch {epoch+1}: Loss={loss:.4f} ({time.time()-start_time:.2f}s)")
                
        return self

    def get_item_representation(self, item_id: int) -> np.ndarray:
        """Get composite item vector h_i = v_i + sum(w_f)."""
        h_i = self.item_factors[item_id].copy()
        
        # Add feature contributions
        # Use CSR slicing
        # item_id is row index
        start = self.feature_matrix.indptr[item_id]
        end = self.feature_matrix.indptr[item_id+1]
        
        for idx in range(start, end):
            f_idx = self.feature_matrix.indices[idx]
            h_i += self.feature_embeddings[f_idx]
            
        return h_i

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        # TODO: Vectorize this properly or iterate
        # For simplicity, iterate or use batch
        n = len(user_ids)
        scores = np.zeros(n, dtype=np.float32)
        for idx in range(n):
            u = user_ids[idx]
            i = item_ids[idx]
            h_i = self.get_item_representation(i)
            scores[idx] = np.dot(self.user_factors[u], h_i)
        return scores
        
    def predict_user(self, user_id: int) -> np.ndarray:
        # Efficiently compute scores for all items
        # Score = u . V^T + u . (W F^T) = u . (V + F W)^T
        # We can precompute H = V + F @ W  (Shape N x K)
        # Then scores = u @ H.T
        
        # Compute H only if needed or on the fly? 
        # For a single user, it's cheap to just multiply.
        
        # Let's compute effective item matrix H = V + F @ W
        # F is sparse (N x D), W is (D x K). F@W is (N x K).
        feat_part = self.feature_matrix @ self.feature_embeddings
        H = self.item_factors + feat_part
        
        return self.user_factors[user_id] @ H.T
