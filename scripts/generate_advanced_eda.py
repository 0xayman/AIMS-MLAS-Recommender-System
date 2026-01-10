import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import MovieLensDataset

# Set style
plt.style.use('default')
sns.set_theme(style="whitegrid", context="paper")
colors = sns.color_palette("viridis")

FIGURES_DIR = Path("report/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_temporal_drift(dataset: MovieLensDataset, name: str):
    """Plot average rating and volume over time (Monthly)."""
    print(f"[{name}] Analyzing temporal drift...")
    ratings = dataset.ratings
    
    # Convert timestamp to datetime if not already
    if not np.issubdtype(ratings['timestamp'].dtype, np.datetime64):
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    # Resample by month
    monthly = ratings.set_index('datetime').resample('M').agg({
        'rating': ['mean', 'count']
    })
    monthly.columns = ['avg_rating', 'count']
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Ratings (Monthly)', color=color1)
    ax1.plot(monthly.index, monthly['count'], color=color1, alpha=0.6, label='Volume')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.fill_between(monthly.index, 0, monthly['count'], color=color1, alpha=0.1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Average Rating', color=color2)
    ax2.plot(monthly.index, monthly['avg_rating'], color=color2, linewidth=2, label='Avg Rating')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(2.5, 4.5)
    
    plt.title(f'Temporal Dynamics: Rating Volume vs. Average Sentiment ({name})')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"eda_temporal_drift_{name}.pdf")
    plt.close()

def plot_genre_trends(dataset: MovieLensDataset, name: str):
    """Plot evolution of genre popularity over years."""
    print(f"[{name}] Analyzing genre trends...")
    
    # Check if we have genre data
    movies = dataset.movies
    ratings = dataset.ratings
    
    if 'genres' not in movies.columns:
        return

    # Merge ratings with movie genres
    # This can be heavy for 32M, so let's optimize
    # Instead of full merge, just map movieId to genres
    
    # Explode genres first
    movies_exploded = movies.assign(genre=movies['genres'].str.split('|')).explode('genre')
    
    # Filter out '(no genres listed)'
    movies_exploded = movies_exploded[movies_exploded['genre'] != '(no genres listed)']
    
    # We want Year -> Genre Counts
    # Extract year from rating timestamp
    if 'datetime' not in ratings.columns:
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    ratings['year'] = ratings['datetime'].dt.year
    
    # Group ratings by (movieId, year) -> count
    # This reduces size before joining with genres
    movie_year_counts = ratings.groupby(['movieId', 'year']).size().reset_index(name='count')
    
    # Join with exploded genres
    genre_year = movie_year_counts.merge(movies_exploded[['movieId', 'genre']], on='movieId')
    
    # Group by (year, genre)
    genre_year_counts = genre_year.groupby(['year', 'genre'])['count'].sum().reset_index()
    
    # Calculate share per year
    yearly_totals = genre_year_counts.groupby('year')['count'].transform('sum')
    genre_year_counts['share'] = genre_year_counts['count'] / yearly_totals
    
    # Filter top genres
    top_genres = genre_year_counts.groupby('genre')['count'].sum().nlargest(7).index
    plot_data = genre_year_counts[genre_year_counts['genre'].isin(top_genres)]
    
    # Limit to reasonable years (e.g. > 1995 since timestamps start then)
    plot_data = plot_data[plot_data['year'] >= 1996]
    
    # Plot Stackplot or Lineplot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for genre in top_genres:
        subset = plot_data[plot_data['genre'] == genre]
        ax.plot(subset['year'], subset['share'], label=genre, linewidth=2)
        
    ax.set_ylabel('Share of Ratings')
    ax.set_xlabel('Year')
    ax.set_title(f'Evolution of Genre Popularity ({name})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"eda_genre_trends_{name}.pdf")
    plt.close()

def plot_cold_start_cdf(dataset: MovieLensDataset, name: str):
    """Plot CDF of item rating counts (Long Tail analysis)."""
    print(f"[{name}] Analyzing cold-start / heavy tail...")
    
    item_counts = dataset.get_item_counts()
    item_counts.sort()
    
    # CDF
    n = len(item_counts)
    y = np.arange(1, n+1) / n
    x = item_counts
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(x, y, where='post', linewidth=2, color='#2c3e50')
    
    ax.set_xscale('log')
    ax.set_xlabel('Number of Ratings (log scale)')
    ax.set_ylabel('Cumulative Proportion of Items')
    ax.set_title(f'Item Popularity CDF: The Long Tail ({name})')
    
    # Annotate key percentiles
    for pct in [0.2, 0.5, 0.8]:
        idx = int(pct * n)
        val = x[idx]
        ax.axvline(val, color='red', linestyle=':', alpha=0.5)
        ax.text(val, pct, f' {pct:.0%} < {val} ratings', color='red', fontsize=8, va='bottom')
        
    ax.grid(True, which="both", ls="-", alpha=0.2)
    fig.savefig(FIGURES_DIR / f"eda_cold_start_cdf_{name}.pdf")
    plt.close()

def run_analysis(data_name: str, data_path: str):
    try:
        print(f"\n--- Loading {data_name} ---")
        dataset = MovieLensDataset(data_path)
        # Touch ratings to load
        _ = dataset.ratings
        
        plot_temporal_drift(dataset, data_name)
        plot_genre_trends(dataset, data_name)
        plot_cold_start_cdf(dataset, data_name)
        
    except Exception as e:
        print(f"Failed to analyze {data_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Define paths
    base_data = Path("data")
    
    # 1. Small Dataset (100k)
    path_100k = base_data / "ml-latest-small"
    if path_100k.exists():
        run_analysis("100k", str(path_100k))
    
    # 2. Large Dataset (32M)
    path_32m = base_data / "ml-32m"
    if path_32m.exists():
        run_analysis("32m", str(path_32m))
    else:
        print(f"Dataset ml-32m not found at {path_32m}, skipping.")
