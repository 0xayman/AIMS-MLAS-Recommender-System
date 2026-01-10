"""
Data Structures Module for Recommender Systems.

Provides CSR-like indexing for efficient access to ratings by user and item.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CSRIndex:
    """
    CSR-like index structure for sparse matrix access.
    
    Allows efficient iteration over non-zero entries by row.
    
    Attributes:
        indptr: Row pointers (length = n_rows + 1)
        indices: Column indices of non-zero entries
        data: Values of non-zero entries
        n_rows: Number of rows
        n_cols: Number of columns
        time_bins: Optional time bin indices for temporal modeling
    """
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    n_rows: int
    n_cols: int
    time_bins: Optional[np.ndarray] = None
    
    def get_row(self, row_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all entries for a given row.
        
        Args:
            row_idx: Row index
            
        Returns:
            Tuple of (column_indices, values) for this row
        """
        start = self.indptr[row_idx]
        end = self.indptr[row_idx + 1]
        return self.indices[start:end], self.data[start:end]
    
    def get_row_with_bins(self, row_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all entries for a row including time bins.
        
        Args:
            row_idx: Row index
            
        Returns:
            Tuple of (column_indices, values, time_bins) for this row
        """
        start = self.indptr[row_idx]
        end = self.indptr[row_idx + 1]
        bins = self.time_bins[start:end] if self.time_bins is not None else None
        return self.indices[start:end], self.data[start:end], bins
    
    def get_row_count(self, row_idx: int) -> int:
        """Get number of non-zero entries in a row."""
        return self.indptr[row_idx + 1] - self.indptr[row_idx]
    
    @property
    def nnz(self) -> int:
        """Total number of non-zero entries."""
        return len(self.data)


class RatingMatrix:
    """
    Sparse rating matrix with dual indexing (by user and by item).
    
    Stores ratings in CSR format indexed both ways for efficient
    access during ALS-style alternating updates.
    
    Attributes:
        n_users: Number of users
        n_items: Number of items
        user_index: CSR index for access by user (user → items, ratings)
        item_index: CSR index for access by item (item → users, ratings)
        global_mean: Mean of all ratings
        timestamps: Original timestamps (optional)
        time_bins: Discretized time bins per rating (optional)
        n_bins: Number of time bins
    """
    
    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
        timestamps: Optional[np.ndarray] = None,
        n_bins: int = 30
    ):
        """
        Build rating matrix from COO-format arrays.
        
        Args:
            user_ids: Array of user indices (0-indexed)
            item_ids: Array of item indices (0-indexed)
            ratings: Array of rating values
            n_users: Total number of users (inferred if None)
            n_items: Total number of items (inferred if None)
            timestamps: Optional array of timestamps for temporal modeling
            n_bins: Number of time bins for temporal biases
        """
        self.n_users = n_users if n_users is not None else int(user_ids.max()) + 1
        self.n_items = n_items if n_items is not None else int(item_ids.max()) + 1
        self.global_mean = float(np.mean(ratings))
        self.n_bins = n_bins
        
        # Compute time bins if timestamps provided
        if timestamps is not None:
            self.timestamps = timestamps
            self.time_bins = self._compute_time_bins(timestamps, n_bins)
        else:
            self.timestamps = None
            self.time_bins = None
        
        # Build user index (user → items)
        self.user_index = self._build_csr_index(
            user_ids, item_ids, ratings, self.n_users, self.n_items,
            self.time_bins
        )
        
        # Build item index (item → users)
        self.item_index = self._build_csr_index(
            item_ids, user_ids, ratings, self.n_items, self.n_users,
            self.time_bins
        )
    
    @staticmethod
    def _compute_time_bins(timestamps: np.ndarray, n_bins: int) -> np.ndarray:
        """
        Convert timestamps to discrete time bins.
        
        Uses quantile-based binning to ensure roughly equal ratings per bin.
        
        Args:
            timestamps: Array of timestamps
            n_bins: Number of bins to create
            
        Returns:
            Array of bin indices (0 to n_bins-1)
        """
        # Use percentile-based binning for balanced bins
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(timestamps, percentiles)
        
        # Digitize assigns each timestamp to a bin
        bins = np.digitize(timestamps, bin_edges[1:-1]).astype(np.int32)
        return bins
        
    @staticmethod
    def _build_csr_index(
        row_ids: np.ndarray,
        col_ids: np.ndarray,
        values: np.ndarray,
        n_rows: int,
        n_cols: int,
        time_bins: Optional[np.ndarray] = None
    ) -> CSRIndex:
        """
        Build a CSR index from COO arrays.
        
        Args:
            row_ids: Row indices for each entry
            col_ids: Column indices for each entry
            values: Values for each entry
            n_rows: Total number of rows
            n_cols: Total number of columns
            time_bins: Optional time bin indices for temporal modeling
            
        Returns:
            CSRIndex structure
        """
        # Sort by row index for CSR construction
        sort_idx = np.argsort(row_ids)
        sorted_rows = row_ids[sort_idx]
        sorted_cols = col_ids[sort_idx].astype(np.int32)
        sorted_vals = values[sort_idx].astype(np.float32)
        sorted_bins = time_bins[sort_idx].astype(np.int32) if time_bins is not None else None
        
        # Build indptr
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.add.at(indptr[1:], sorted_rows, 1)
        np.cumsum(indptr, out=indptr)
        
        return CSRIndex(
            indptr=indptr,
            indices=sorted_cols,
            data=sorted_vals,
            n_rows=n_rows,
            n_cols=n_cols,
            time_bins=sorted_bins
        )
    
    def get_user_ratings(self, user_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all ratings by a user.
        
        Args:
            user_idx: User index
            
        Returns:
            Tuple of (item_indices, ratings)
        """
        return self.user_index.get_row(user_idx)
    
    def get_item_ratings(self, item_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all ratings for an item.
        
        Args:
            item_idx: Item index
            
        Returns:
            Tuple of (user_indices, ratings)
        """
        return self.item_index.get_row(item_idx)
    
    def get_user_count(self, user_idx: int) -> int:
        """Get number of ratings by a user."""
        return self.user_index.get_row_count(user_idx)
    
    def get_item_count(self, item_idx: int) -> int:
        """Get number of ratings for an item."""
        return self.item_index.get_row_count(item_idx)

    def get_item_counts(self) -> np.ndarray:
        """Get number of ratings for all items."""
        return np.diff(self.item_index.indptr)
    
    @property
    def nnz(self) -> int:
        """Total number of ratings."""
        return self.user_index.nnz
    
    @property
    def sparsity(self) -> float:
        """Sparsity of the matrix."""
        total = self.n_users * self.n_items
        return 1.0 - (self.nnz / total)
    
    def get_all_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all arrays needed for ALS training.
        
        Returns:
            Tuple of:
                - user_indptr: User index pointers
                - user_indices: Item indices per user
                - item_indptr: Item index pointers
                - item_indices: User indices per item
                - ratings: Rating values (user-indexed order)
        """
        return (
            self.user_index.indptr,
            self.user_index.indices,
            self.item_index.indptr,
            self.item_index.indices,
            self.user_index.data
        )
    
    def summary(self) -> str:
        """Return summary string."""
        return (
            f"RatingMatrix(users={self.n_users:,}, items={self.n_items:,}, "
            f"ratings={self.nnz:,}, sparsity={self.sparsity:.4%})"
        )
    
    def __repr__(self) -> str:
        return self.summary()
