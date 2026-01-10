import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def main():
    print("Generating best model metrics plot for Practical 3 (K=20)...")
    os.makedirs('../report/figures', exist_ok=True)

    iterations = np.arange(1, 16)
    
    # Data for K=20 extracted from practical_3.ipynb output
    loss = [2695.84, 2377.13, 2260.73, 2228.83, 2215.08, 2207.44, 2202.66, 2199.42, 2197.11, 2195.39, 2194.07, 2193.02, 2192.18, 2191.47, 2190.89]
    train_rmse = [0.7853, 0.6794, 0.6186, 0.5990, 0.5908, 0.5866, 0.5840, 0.5822, 0.5810, 0.5801, 0.5794, 0.5789, 0.5785, 0.5781, 0.5778]
    test_rmse = [0.8783, 0.8624, 0.8614, 0.8609, 0.8603, 0.8598, 0.8595, 0.8593, 0.8592, 0.8591, 0.8591, 0.8591, 0.8590, 0.8590, 0.8589]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss
    ax1.plot(iterations, loss, 'o-', color='#e74c3c', label='Training Loss', linewidth=2, markersize=5)
    ax1.set_title('Training Loss Convergence (K=20)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Regularized Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: RMSE
    ax2.plot(iterations, train_rmse, 'o-', color='#3498db', label='Train RMSE', linewidth=2, markersize=5)
    ax2.plot(iterations, test_rmse, 's-', color='#2ecc71', label='Test RMSE', linewidth=2, markersize=5)
    ax2.set_title('RMSE Performance (K=20)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '../report/figures/practical_3_best_model.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
