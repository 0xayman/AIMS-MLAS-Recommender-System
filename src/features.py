"""
Features Module for MovieLens datasets.

Provides functionality to extracting genre features and creating 
feature matrices for use in Feature-Aware Matrix Factorization.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Tuple, Dict, List

from .data_loader import MovieLensDataset

def load_genre_features(dataset: MovieLensDataset) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Load genre features from the dataset and return a sparse feature matrix.
    
    The feature matrix F will be of shape (n_items, n_features), where
    n_items matches the internal item indexing of the dataset.
    
    Args:
        dataset: The loaded MovieLensDataset.
        
    Returns:
        Tuple containing:
        - feature_matrix: CSR matrix of shape (n_items, n_features)
        - feature_names: List of genre names corresponding to columns
    """
    print("Building genre feature matrix...")
    
    # 1. Get mappings to internal item indices
    # We need to map movies['movieId'] to our internal 0..N-1 indices
    # internal indices are defined in dataset.ratings['item_idx']
    
    # Create a map from original movieId -> internal item_idx
    # We leverage the ratings dataframe which already has item_idx populated by _load_ratings
    if 'item_idx' not in dataset.ratings.columns:
        # Should imply ratings haven't been loaded or processed fully? 
        # Accessing dataset.ratings triggers load.
        # But we need to ensure unique mapping.
        # If dataset.ratings is accessed, it computes item_idx.
        pass
        
    # Extract unique mapping
    # Drop duplicates to get unique items
    item_map_df = dataset.ratings[['movieId', 'item_idx']].drop_duplicates('movieId')
    item_map = item_map_df.set_index('movieId')['item_idx']
    
    # 2. Process Movies Dataframe
    movies_df = dataset.movies.copy()
    
    # Keep only movies that exist in our ratings (and thus have an item_idx)
    # Movies without ratings won't be in the interaction matrix anyway.
    # (Although for cold-start, we might care, but our MF model relies on item_idx 0..N-1 
    # defined by the rating matrix structure. Standard practice: ignore items with 0 ratings 
    # or add them to the index. The Dataset class defines n_items based on ratings.)
    movies_df = movies_df[movies_df['movieId'].isin(item_map.index)]
    
    # Add internal index
    movies_df['item_idx'] = movies_df['movieId'].map(item_map)
    
    # 3. Parse Genres
    # Genres are pipe-separated: "Adventure|Children|Fantasy"
    # We want to create binary features.
    
    # Get all unique genres
    all_genres = set()
    for genres_str in movies_df['genres']:
        if pd.isna(genres_str) or genres_str == '(no genres listed)':
            continue
        valid_genres = [g for g in genres_str.split('|')]
        all_genres.update(valid_genres)
        
    sorted_genres = sorted(list(all_genres))
    genre_to_idx = {g: i for i, g in enumerate(sorted_genres)}
    n_features = len(sorted_genres)
    n_items = dataset.n_items
    
    print(f"Found {n_features} unique genres: {sorted_genres}")
    
    # 4. Build Sparse Matrix
    # We'll build a COO matrix first: (row, col, data)
    rows = []
    cols = []
    data = []
    
    for _, row in movies_df.iterrows():
        item_idx = row['item_idx']
        genres_str = row['genres']
        
        if pd.isna(genres_str) or genres_str == '(no genres listed)':
            continue
            
        for g in genres_str.split('|'):
            if g in genre_to_idx:
                rows.append(item_idx)
                cols.append(genre_to_idx[g])
                data.append(1.0)
                
    # Create CSR matrix
    # Use max dimensions to ensure shape matches n_items even if last item has no genre
    feature_matrix = sparse.csr_matrix(
        (data, (rows, cols)), 
        shape=(n_items, n_features),
        dtype=np.float32
    )
    
    # Normalize features? 
    # Usually for multi-hot, we might normalize so ||f_n|| = 1?
    # Or leave as binary. The prompt implies binary vectors.
    # Course notes: "Binary genre vectors"
    # "Loss: | v_n - W f_n |^2"
    # Leaving as binary.
    
    print(f"Feature matrix statistics:")
    print(f"  Shape: {feature_matrix.shape}")
    print(f"  Stored elements: {feature_matrix.nnz}")
    print(f"  Sparsity: {1.0 - feature_matrix.nnz / (n_items * n_features):.2%}")
    
    return feature_matrix, sorted_genres
