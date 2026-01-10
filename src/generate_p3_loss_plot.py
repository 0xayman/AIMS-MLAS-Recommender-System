import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def main():
    print("Generating Loss Curve for Practical 3...")
    
    # Ensure figures dir exists
    os.makedirs('../report/figures', exist_ok=True)
    
    # Data extracted from practical_3.ipynb output
    # Iterations 1 to 15
    iters = np.arange(1, 16)
    
    loss_k5 = [
        2722.92, 2533.38, 2441.22, 2412.72, 2399.89, 
        2392.46, 2387.43, 2383.69, 2380.75, 2378.38, 
        2376.41, 2374.78, 2373.39, 2372.23, 2371.24
    ]
    
    loss_k10 = [
        2713.68, 2470.94, 2356.43, 2321.55, 2305.75, 
        2296.79, 2291.05, 2287.07, 2284.16, 2281.97, 
        2280.26, 2278.89, 2277.80, 2276.89, 2276.13
    ]
    
    loss_k20 = [
        2695.84, 2377.13, 2260.73, 2228.83, 2215.08, 
        2207.44, 2202.66, 2199.42, 2197.11, 2195.39, 
        2194.07, 2193.02, 2192.18, 2191.47, 2190.89
    ]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(iters, loss_k5, marker='o', label='K=5', linestyle='--')
    plt.plot(iters, loss_k10, marker='s', label='K=10', linestyle='-.')
    plt.plot(iters, loss_k20, marker='D', label='K=20 (Optimal)', linewidth=2)
    
    plt.title("ALS Training Loss over Iterations (Monotonic Convergence)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Regularized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = '../report/figures/practical_3_loss.pdf'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
