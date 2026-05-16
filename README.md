# AIMS MLAS Recommender System

Scalable recommender system project built for **Applied Machine Learning at Scale (AIMS)** using the **MovieLens** datasets.  
The project evolves from a global-mean baseline to bias modeling, latent-factor matrix factorization (ALS), and a feature-aware extension for cold-start robustness.

## Highlights

- End-to-end recommender pipeline from data loading to evaluation
- Numba-accelerated ALS and ranking models for scalability
- Support for both **MovieLens Small** and **MovieLens 32M**
- Feature-aware modeling with genre metadata for sparse/cold items
- Reproducible experiments and plots across practical notebooks

## Repository Structure

```text
AIMS-MLAS-Recommender-System/
├── src/                 # Core library: data, models, metrics, splits, plots
├── notebooks/           # Practical experiment notebooks (P1–P5 + ablation)
├── scripts/             # Standalone analysis script(s)
├── models/              # Saved trained model checkpoints (.npz)
└── report/              # Final report (LaTeX) and generated figures
```

## Core Components (`src/`)

- `data_loader.py`: dataset download/loading and ID mapping (`MovieLensDataset`)
- `data_structures.py`: sparse rating matrix/index abstractions
- `bias_als.py`: bias-only ALS baseline (`μ + b_u + b_i`)
- `als.py`: latent-factor ALS model (`μ + b_u + b_i + uᵀv`)
- `bpr.py`: Bayesian Personalized Ranking (implicit feedback)
- `feature_bpr.py`: feature-aware BPR extension
- `features.py`: genre feature matrix construction
- `metrics.py`: RMSE and ranking metrics (Precision@K, Recall@K, NDCG@K)
- `train_test_split.py`: random and time-based train/test split utilities

## Installation

This repository does not currently include a pinned requirements file.  
Create a virtual environment and install the packages used across `src/` and notebooks:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy pandas scipy numba matplotlib seaborn scikit-learn optuna jupyter
```

## Data

The code supports:

- `ml-latest-small`
- `ml-32m`

You can download a dataset programmatically:

```python
from src.data_loader import download_dataset
download_dataset("ml-32m", data_dir="data")
```

Expected location after extraction:

```text
data/
├── ml-latest-small/
└── ml-32m/
```

## Usage

### 1) Run the practical notebooks

```bash
jupyter notebook notebooks/
```

Recommended order:

1. `practical_1.ipynb` – EDA
2. `practical_2.ipynb` – Bias-only ALS baseline
3. `practical_3.ipynb` – Latent factor ALS
4. `practical_4.ipynb` – Scaled training + tuning
5. `practical_5.ipynb` – Feature-aware extension
6. `ablation_study.ipynb` – Comparative analysis

### 2) Generate advanced EDA figures

```bash
python scripts/generate_advanced_eda.py
```

### 3) Generate practical 4 advanced plots

```bash
python src/generate_p4_advanced_plots.py
```

## Saved Models

Pretrained checkpoints are provided in `models/` (e.g. `als_optuna_final.npz`).  
You can load ALS checkpoints with:

```python
from src.als import MatrixFactorizationModel
model = MatrixFactorizationModel.load("models/als_optuna_final.npz")
```

## Reported Results (from `report/main.tex`)

- Global Mean baseline: **RMSE 1.054**
- Bias-only model: **RMSE 0.8623**
- Standard ALS (`K=10`): **RMSE 0.8560**
- Optimized ALS (`K=20`): **RMSE 0.7709** (best global RMSE)
- Feature-aware ALS (`K=20`): **RMSE 0.7779**
- Cold-start gain: up to **~3.9% RMSE improvement** on sparse items

## Reproducibility Notes

- Model training uses random initialization and data splitting; set seeds where applicable for repeatability.
- Some scripts assume execution from the repository root.
- Large-scale experiments on MovieLens 32M require significant RAM and compute time.

## Author

**Ayman Tarig**  
African Institute for Mathematical Sciences (AIMS), South Africa
