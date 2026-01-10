
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path to import als module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.als import MatrixFactorizationModel
from src.data_loader import MovieLensDataset

def load_data_and_model():
    print("Loading dataset...")
    # Use absolute path relative to current working directory
    # Assumes collecting running from project root
    data_path = os.path.abspath('data/ml-32m/')
    if not os.path.exists(data_path):
         # Try going up one level if we are in src
         data_path = os.path.abspath('../data/ml-32m/')
    
    print(f"Using data path: {data_path}")
    dataset = MovieLensDataset(data_path=data_path)

    print("Loading model...")
    # Load the best model from practical 4
    try:
        model_path = '../models/als_optuna_final.npz'
        model = MatrixFactorizationModel.load(model_path)
    except FileNotFoundError:
        try: 
            model_path = 'models/als_optuna_final.npz'
            model = MatrixFactorizationModel.load(model_path)
        except FileNotFoundError:
             model_path = r'c:/Users/ayman/Downloads/aims/Applied ML at Scale/project/models/als_optuna_final.npz'
             model = MatrixFactorizationModel.load(model_path)
            
    return dataset, model

def generate_lotr_plot(dataset, model):
    print("Generating LOTR recommendation plot...")
    
    # recreate the lotr user vector logic from notebook
    # Access internal map as it's not exposed publicly in the class definition we saw
    movie_id_map = dataset._item_id_map 
    if movie_id_map is None:
        # Force load if not loaded
        _ = dataset.ratings
        movie_id_map = dataset._item_id_map
        
    idx_to_movie_id = {v: k for k, v in movie_id_map.items()} # mapping internal index -> raw movieID
    
    # movies df property
    movies_df = dataset.movies
    
    def find_movies(query):
        return movies_df[movies_df['title'].str.contains(query, case=False)]

    lotr_movies = find_movies("Lord of the Rings")
    print(f"Found LOTR movies: {lotr_movies['title'].tolist()}")
    
    # Create fake user ratings
    valid_movie_indices = []
    for mid in lotr_movies['movieId']:
        if mid in movie_id_map:
            valid_movie_indices.append(movie_id_map[mid])
            
    if not valid_movie_indices:
        print("No LOTR movies found in training set. Checking dataset...")
        return

    # Simulate user vector calculation (simplified from ALS fold-in)
    # roughly: new_user = (V_rated.T @ V_rated + lambda * I)^-1 @ V_rated.T @ ratings
    # Or simplified: average of rated item vectors (heuristic) or using ALS solver
    # Let's use the fold_in method if available in MatrixFactorizationModel or implement it
    
    # We will implement a quick fold-in using the model's item factors
    # Equation: xu = (Y.T Y + lambda I)^-1 Y.T pu
    # Y = item_factors[indices], pu = ratings
    
    K = model.item_factors.shape[1]
    lambda_reg = model.lambda_
    
    Y = model.item_factors[valid_movie_indices]
    
    
    # Target for dot product is residual: rating - global_mean - item_bias
    # We assume user bias is 0 or absorbed into this optimization
    raw_ratings = np.array([5.0] * len(valid_movie_indices))
    residuals = raw_ratings - model.global_mean - model.item_biases[valid_movie_indices]
    
    # Exact ALS update for user vector:
    # (tau * I + lambda * Y.T @ Y) xu = lambda * Y.T @ residuals
    
    A = model.tau * np.eye(K) + model.lambda_ * (Y.T @ Y)
    b = model.lambda_ * (Y.T @ residuals)
    
    user_vector = np.linalg.solve(A, b)
    
    # Predict all scores: mu + bi + xu.yi
    all_scores = model.item_factors @ user_vector + model.item_biases + model.global_mean
    
    # Get Top Recommendations (excluding already rated)
    top_indices = np.argsort(all_scores)[::-1]
    
    recs = []
    count = 0
    seen_indices = set(valid_movie_indices)
    
    print("\n--- Top LOTR Recommendations ---")
    for idx in top_indices:
        if idx not in seen_indices:
            mid = idx_to_movie_id[idx] # internal -> raw movieID
            # Need to map raw movieID back to metadata (dataset.movies)
            title = movies_df[movies_df['movieId'] == mid]['title'].values
            if len(title) > 0:
                score = all_scores[idx]
                recs.append({'title': title[0], 'score': score})
                print(f"{count+1}. {title[0]} (Score: {score:.3f})")
                count += 1
            if count >= 10:
                break
    print("--------------------------------\n")
                
    # Create Plot
    rec_df = pd.DataFrame(recs)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='score', y='title', data=rec_df, palette='viridis')
    plt.title('Top 10 Recommendations for a "Lord of the Rings" Fan', fontsize=14)
    plt.xlabel('Predicted Rating', fontsize=12)
    plt.ylabel('')
    plt.xlim(3, 5.5) # Focus on high ratings area
    plt.axvline(x=5.0, color='r', linestyle='--', alpha=0.5, label='Max Rating')
    plt.tight_layout()
    
    params = {'dpi': 300, 'bbox_inches': 'tight', 'format': 'pdf'}
    save_path = '../report/figures/practical_4_lotr_recs.pdf'
    
    # handle execution from src/ or project root
    if not os.path.exists(os.path.dirname(save_path)):
         # try creating or adjusting path
         os.makedirs(os.path.dirname(save_path), exist_ok=True)
         
    try:
        plt.savefig(save_path, **params)
    except FileNotFoundError:
        # Fallback for path issues
        plt.savefig('report/figures/practical_4_lotr_recs.pdf', **params)
        
    print(f"Saved LOTR plot to extensive locations")
    plt.close()

def generate_similarity_matrix(dataset, model):
    print("Generating Similarity Matrix plot...")
    
    # Define franchises to look for
    franchises = {
        'Star Wars': ['Star Wars: Episode IV', 'Star Wars: Episode V', 'Star Wars: Episode VI'],
        'LOTR': ['Lord of the Rings: The Fellowship', 'Lord of the Rings: The Two Towers', 'Lord of the Rings: The Return'],
        'Matrix': ['Matrix, The', 'Matrix Reloaded', 'Matrix Revolutions'],
        'Toy Story': ['Toy Story (1995)', 'Toy Story 2']
    }
    
    # Find actual titles and indices
    selected_indices = []
    selected_labels = []
    
    movies_df = dataset.movies
    movie_id_map = dataset._item_id_map
    
    label_map = {
        'Star Wars: Episode IV': 'SW: New Hope',
        'Star Wars: Episode V': 'SW: Empire', 
        'Star Wars: Episode VI': 'SW: Jedi',
        'Lord of the Rings: The Fellowship': 'LOTR: Fellowship',
        'Lord of the Rings: The Two Towers': 'LOTR: Two Towers', 
        'Lord of the Rings: The Return': 'LOTR: Return',
        'Matrix, The': 'The Matrix',
        'Matrix Reloaded': 'Matrix Reloaded',
        'Matrix Revolutions': 'Matrix Revolutions',
        'Toy Story (1995)': 'Toy Story 1',
        'Toy Story 2': 'Toy Story 2'
    }

    for franchise, queries in franchises.items():
        for q in queries:
            # Fuzzy match title
            match = movies_df[movies_df['title'].str.contains(q, case=False, regex=False)]
            if len(match) > 0:
                mid = match.iloc[0]['movieId']
                if mid in movie_id_map:
                    idx = movie_id_map[mid]
                    title = match.iloc[0]['title']
                    
                    # Use manual label if available, else heuristic
                    found_label = None
                    for k, v in label_map.items():
                        if k in q: # Match query key
                            found_label = v
                            break
                    
                    if not found_label:
                        short_title = title.split('(')[0].strip()
                        if len(short_title) > 20: 
                            short_title = short_title[:17] + "..."
                        found_label = short_title
                        
                    selected_indices.append(idx)
                    selected_labels.append(found_label)

    if not selected_indices:
        print("No matching movies found for similarity matrix.")
        return

    # Compute Cosine Similarity
    vectors = model.item_factors[selected_indices]
    sim_matrix = cosine_similarity(vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, xticklabels=selected_labels, yticklabels=selected_labels, 
                cmap='coolwarm', annot=True, fmt=".2f", vmin=0, vmax=1)
    
    plt.title('Latent Space Cosine Similarity: Popular Franchises', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    params = {'dpi': 300, 'bbox_inches': 'tight', 'format': 'pdf'}
    try:
        plt.savefig('../report/figures/practical_4_similarity_matrix.pdf', **params)
    except FileNotFoundError:
        plt.savefig('report/figures/practical_4_similarity_matrix.pdf', **params)
        
    print("Saved Similarity Matrix plot.")
    plt.close()

def generate_global_embeddings_plot(dataset, model):
    print("Generating Global 2D Embeddings plot...")
    from sklearn.decomposition import PCA
    
    # PCA to 2D
    pca = PCA(n_components=2)
    # Use a large sample if needed, but 32k is handleable
    vectors = model.item_factors
    print(f"Reducing {vectors.shape[0]} vectors to 2D...")
    vectors_2d = pca.fit_transform(vectors)
    
    # Prepare DataFrame for plotting
    df_plot = pd.DataFrame(vectors_2d, columns=['x', 'y'])
    df_plot['alpha'] = 0.05
    df_plot['color'] = 'lightgrey'
    df_plot['size'] = 5
    df_plot['label'] = None
    
    # Identify clusters/franchises
    movies_df = dataset.movies
    movie_id_map = dataset._item_id_map
    
    highlights = {
        'Star Wars': {'query': 'Star Wars: Episode', 'color': 'red', 'marker': 'o'},
        'LOTR': {'query': 'Lord of the Rings', 'color': 'blue', 'marker': '^'},
        'Toy Story': {'query': 'Toy Story', 'color': 'green', 'marker': 's'},
        'Godfather': {'query': 'Godfather', 'color': 'purple', 'marker': 'D'}
    }
    
    plt.figure(figsize=(12, 10))
    
    # Plot background (all movies)
    # Sampling for background to avoid too heavy PDF
    if len(df_plot) > 10000:
        bg_sample = df_plot.sample(10000, random_state=42)
    else:
        bg_sample = df_plot
        
    plt.scatter(bg_sample['x'], bg_sample['y'], c='lightgrey', alpha=0.2, s=5, label='Other Movies')
    
    # Plot highlights
    for name, props in highlights.items():
        matches = movies_df[movies_df['title'].str.contains(props['query'], case=False, regex=False)]
        indices = []
        for mid in matches['movieId']:
            if mid in movie_id_map:
                indices.append(movie_id_map[mid])
        
        if indices:
            subset = df_plot.iloc[indices]
            plt.scatter(subset['x'], subset['y'], c=props['color'], label=name, 
                       s=50, marker=props['marker'], edgecolors='white', linewidth=0.5)
            
            # Annotate a few points
            for i in range(min(2, len(indices))):
                idx = indices[i]
                row = matches[matches['movieId'] == list(movie_id_map.keys())[list(movie_id_map.values()).index(idx)]]
                if len(row) > 0:
                    title = row.iloc[0]['title'].split('(')[0].strip()
                    plt.text(subset.iloc[i]['x'], subset.iloc[i]['y'], title, fontsize=8, alpha=0.8)

    plt.title('Global Item Embeddings (PCA) with Franchise Clustering', fontsize=16)
    plt.xlabel('PC 1', fontsize=12)
    plt.ylabel('PC 2', fontsize=12)
    plt.legend(loc='best')
    plt.tight_layout()
    
    params = {'dpi': 300, 'bbox_inches': 'tight', 'format': 'pdf'}
    try:
        plt.savefig('../report/figures/practical_4_2d_embeddings.pdf', **params)
    except FileNotFoundError:
        plt.savefig('report/figures/practical_4_2d_embeddings.pdf', **params)
        
    print("Saved Global Embeddings plot.")
    plt.close()

if __name__ == "__main__":
    dataset, model = load_data_and_model()
    generate_lotr_plot(dataset, model)
    generate_similarity_matrix(dataset, model)
    generate_global_embeddings_plot(dataset, model)
    print("Done.")
