"""
Data Loader Module for MovieLens datasets.

Provides functionality to download and load MovieLens datasets.
Supports both ml-latest-small (100k) and ml-32m datasets.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# Dataset URLs
DATASET_URLS: Dict[str, str] = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}


def download_dataset(dataset_name: str = "ml-latest-small", data_dir: str = "data") -> Path:
    """
    Download and extract a MovieLens dataset.
    
    Args:
        dataset_name: Name of the dataset ("ml-latest-small" or "ml-32m")
        data_dir: Directory to store the dataset
        
    Returns:
        Path to the extracted dataset directory
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_URLS.keys())}")
    
    url = DATASET_URLS[dataset_name]
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_path / f"{dataset_name}.zip"
    extract_path = data_path / dataset_name
    
    # Check if already extracted
    if extract_path.exists():
        print(f"Dataset already exists at {extract_path}")
        return extract_path
    
    # Download
    print(f"Downloading {dataset_name} from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded to {zip_path}")
    
    # Extract
    print(f"Extracting to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
    # Remove zip file to save space
    zip_path.unlink()
    print(f"Extraction complete. Dataset at {extract_path}")
    
    return extract_path


class MovieLensDataset:
    """
    Object-oriented wrapper for MovieLens dataset.
    
    Provides easy access to ratings, movies, and computed statistics.
    
    Attributes:
        data_path: Path to the dataset directory
        ratings: DataFrame with user_id, movie_id, rating, timestamp
        movies: DataFrame with movie_id, title, genres
        n_users: Number of unique users
        n_items: Number of unique items
        n_ratings: Total number of ratings
    """
    
    def __init__(self, data_path: str | Path):
        """
        Initialize dataset from a directory path.
        
        Args:
            data_path: Path to the extracted dataset directory
        """
        self.data_path = Path(data_path)
        self._ratings: Optional[pd.DataFrame] = None
        self._movies: Optional[pd.DataFrame] = None
        self._user_id_map: Optional[Dict[int, int]] = None
        self._item_id_map: Optional[Dict[int, int]] = None
        
    @property
    def ratings(self) -> pd.DataFrame:
        """Load ratings lazily on first access."""
        if self._ratings is None:
            self._load_ratings()
        return self._ratings
    
    @property
    def movies(self) -> pd.DataFrame:
        """Load movies lazily on first access."""
        if self._movies is None:
            self._load_movies()
        return self._movies
    
    def _load_ratings(self) -> None:
        """Load ratings.csv and create continuous ID mappings."""
        ratings_path = self.data_path / "ratings.csv"
        print(f"Loading ratings from {ratings_path}...")
        
        self._ratings = pd.read_csv(
            ratings_path,
            dtype={
                'userId': np.int32,
                'movieId': np.int32,
                'rating': np.float32,
                'timestamp': np.int64
            }
        )
        
        # Create continuous user IDs (0-indexed)
        unique_users = self._ratings['userId'].unique()
        self._user_id_map = {uid: idx for idx, uid in enumerate(sorted(unique_users))}
        self._ratings['user_idx'] = self._ratings['userId'].map(self._user_id_map).astype(np.int32)
        
        # Create continuous item IDs (0-indexed)
        unique_items = self._ratings['movieId'].unique()
        self._item_id_map = {mid: idx for idx, mid in enumerate(sorted(unique_items))}
        self._ratings['item_idx'] = self._ratings['movieId'].map(self._item_id_map).astype(np.int32)
        
        print(f"Loaded {len(self._ratings):,} ratings")
        
    def _load_movies(self) -> None:
        """Load movies.csv."""
        movies_path = self.data_path / "movies.csv"
        print(f"Loading movies from {movies_path}...")
        
        self._movies = pd.read_csv(movies_path)
        print(f"Loaded {len(self._movies):,} movies")
        
    @property
    def n_users(self) -> int:
        """Number of unique users."""
        return len(self._user_id_map) if self._user_id_map else len(self.ratings['userId'].unique())
    
    @property
    def n_items(self) -> int:
        """Number of unique items."""
        return len(self._item_id_map) if self._item_id_map else len(self.ratings['movieId'].unique())
    
    @property
    def n_ratings(self) -> int:
        """Total number of ratings."""
        return len(self.ratings)
    
    @property
    def sparsity(self) -> float:
        """Sparsity of the rating matrix (percentage of missing entries)."""
        total_possible = self.n_users * self.n_items
        return 1.0 - (self.n_ratings / total_possible)
    
    @property
    def global_mean(self) -> float:
        """Global mean rating."""
        return float(self.ratings['rating'].mean())
    
    def get_user_counts(self) -> np.ndarray:
        """Get number of ratings per user."""
        return self.ratings.groupby('user_idx').size().values
    
    def get_item_counts(self) -> np.ndarray:
        """Get number of ratings per item."""
        return self.ratings.groupby('item_idx').size().values
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get rating data as NumPy arrays.
        
        Returns:
            Tuple of (user_indices, item_indices, ratings)
        """
        return (
            self.ratings['user_idx'].values,
            self.ratings['item_idx'].values,
            self.ratings['rating'].values
        )
    
    def get_arrays_with_timestamps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get rating data with timestamps as NumPy arrays.
        
        Returns:
            Tuple of (user_indices, item_indices, ratings, timestamps)
        """
        return (
            self.ratings['user_idx'].values,
            self.ratings['item_idx'].values,
            self.ratings['rating'].values,
            self.ratings['timestamp'].values
        )
    
    def get_movie_title(self, movie_id: int) -> str:
        """Get movie title by original movie ID."""
        row = self.movies[self.movies['movieId'] == movie_id]
        if len(row) == 0:
            return f"Unknown (ID: {movie_id})"
        return row['title'].values[0]
    
    def summary(self) -> str:
        """Return a summary string of the dataset."""
        return (
            f"MovieLens Dataset Summary\n"
            f"{'='*40}\n"
            f"Users:     {self.n_users:>12,}\n"
            f"Items:     {self.n_items:>12,}\n"
            f"Ratings:   {self.n_ratings:>12,}\n"
            f"Sparsity:  {self.sparsity:>12.4%}\n"
            f"Mean:      {self.global_mean:>12.3f}\n"
        )
    
    def __repr__(self) -> str:
        return f"MovieLensDataset(path='{self.data_path}', ratings={self.n_ratings:,})"
