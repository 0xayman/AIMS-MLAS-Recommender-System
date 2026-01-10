import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data():
    """Load MovieLens data and recreate ID mapping"""
    # Assuming standard location
    ratings_path = Path("../data/ml-latest-small/ratings.csv")
    movies_path = Path("../data/ml-latest-small/movies.csv")
    
    if not ratings_path.exists():
        print(f"Data not found at {ratings_path}")
        return None, None
        
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    # Recreate mapping logic (sorted unique IDs -> 0..N-1)
    unique_user_ids = np.sort(ratings_df['userId'].unique())
    user_id_map = {uid: i for i, uid in enumerate(unique_user_ids)}
    
    unique_item_ids = np.sort(ratings_df['movieId'].unique())
    item_id_map = {mid: i for i, mid in enumerate(unique_item_ids)}
    
    # Map ratings
    ratings_df['user_idx'] = ratings_df['userId'].map(user_id_map)
    ratings_df['item_idx'] = ratings_df['movieId'].map(item_id_map)
    movies_df['item_idx'] = movies_df['movieId'].map(item_id_map)
    
    return ratings_df, movies_df

def load_model(k=20):
    """Load saved model arrays"""
    model_path = Path(f"../models/als_k{k}.npz")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
        
    # Load arrays
    data = np.load(model_path)
    return data

def plot_factor_distributions(data):
    """Plot distribution of user and item latent factors"""
    user_factors = data['user_factors']
    item_factors = data['item_factors']
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(user_factors.flatten(), bins=50, kde=True, color='blue', alpha=0.6)
    plt.title("Distribution of User Factors")
    plt.xlabel("Factor Value")
    
    plt.subplot(1, 2, 2)
    sns.histplot(item_factors.flatten(), bins=50, kde=True, color='green', alpha=0.6)
    plt.title("Distribution of Item Factors")
    plt.xlabel("Factor Value")
    
    plt.tight_layout()
    plt.savefig('../report/figures/practical_3_factor_dist.pdf')
    print("Saved practical_3_factor_dist.pdf")
    plt.close()

def plot_bias_vs_popularity(data, ratings_df, movies_df):
    """Plot Item Bias vs Item Popularity (Log Scale)"""
    item_biases = data['item_biases']
    
    # Calculate item popularity (count)
    item_counts = ratings_df.groupby('item_idx').size()
    
    # Align counts with biases (some items might have 0 ratings if valid in map but filtered?)
    # Usually dataset includes all items present in ratings.
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'item_idx': np.arange(len(item_biases)),
        'bias': item_biases
    })
    
    # Merge with counts
    plot_df = plot_df.merge(item_counts.rename('count'), left_on='item_idx', right_index=True)
    
    # Merge titles for tooltip/labeling potential
    # movies_valid = movies_df.dropna(subset=['item_idx'])
    # plot_df = plot_df.merge(movies_valid[['item_idx', 'title']], on='item_idx', how='left')
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(plot_df['count'], plot_df['bias'], alpha=0.3, s=10)
    plt.xscale('log')
    
    plt.title("Item Bias vs. Popularity (Log Scale)")
    plt.xlabel("Number of Ratings (Log Scale)")
    plt.ylabel("Item Bias")
    
    # Highlights
    # top_items = plot_df.nlargest(5, 'count')
    # for _, row in top_items.iterrows():
    #     plt.annotate(row['item_idx'], (row['count'], row['bias']))
        
    plt.tight_layout()
    plt.savefig('../report/figures/practical_3_bias_vs_popularity.pdf')
    print("Saved practical_3_bias_vs_popularity.pdf")
    plt.close()

def plot_loss_curves(results):
    # This requires recreating the 'results' dictionary which isn't saved in npz.
    # The user implies they might have 'practical_3_rmse.pdf' already.
    # We will verify if it exists.
    pass

def main():
    print("Generating extra plots for Practical 3...")
    
    # Ensure figures dir exists
    os.makedirs('../report/figures', exist_ok=True)
    
    ratings_df, movies_df = load_data()
    if ratings_df is None:
        return

    data = load_model(k=20)
    if data is None:
        return
        
    plot_factor_distributions(data)
    plot_bias_vs_popularity(data, ratings_df, movies_df)
    
if __name__ == "__main__":
    main()
