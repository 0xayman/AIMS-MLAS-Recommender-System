# Applied ML at Scale

> ## ⚡ Performance & Optimization Guidelines (READ FIRST)
>
> All implementations in this course are expected to be **production-grade and scalable**.  
> Correctness alone is not sufficient — **efficient computation is part of the assessment**.
>
> ### 1. Vectorization (Primary Requirement)
> - Prefer **NumPy vectorized operations** over Python loops.
> - Replace explicit `for` loops with:
>   - Matrix multiplications
>   - Broadcasting
>   - Boolean masking
> - Use BLAS-backed operations (`A @ B`, `np.einsum`) whenever possible.
>
> ### 2. Sparse Data Structures
> - Always represent user–item interactions using **sparse matrices**:
> - Avoid dense user–item matrices — they are infeasible at scale.
>
> ### 3. Numba Acceleration
> - Use **Numba (`@njit`, `@njit(parallel=True)`)** to accelerate unavoidable loops.
> - Apply Numba especially for:
>   - Alternating Least Squares (ALS) updates
>   - Iteration over users or items
>   - Sampling loops (e.g., BPR negative sampling)
> - Ensure Numba functions operate on:
>   - NumPy arrays
>   - Python lists of tuples (pre-built, not dynamic)
>
> ### 4. Parallelism by Design
> - Exploit **embarrassingly parallel structure**:
>   - User updates are independent
>   - Item updates are independent
> - Structure code so updates can run:
>   - In parallel with `prange`
>   - In distributed settings if needed
>
> ### 5. Pre-Indexing (Critical for Speed)
> - Precompute and store:
>   - `user → (item, rating)` lists
>   - `item → (user, rating)` lists
> - Never scan the full dataset inside training loops.
>
> ### 6. Memory Efficiency
> - Avoid copying large arrays inside loops.
> - Reuse allocated matrices where possible.
> - Use `float32` instead of `float64` when precision allows.
>
> ### 7. Algorithmic Choices
> - Prefer **Alternating Least Squares (ALS)** over SGD for large explicit datasets.
> - Prefer **ranking losses (BPR)** over regression for implicit feedback.
> - Down-weight global popularity (item bias) during ranking to improve personalization.
>
> ### 8. Training Efficiency
> - Use **time-aware train/test splits** (not random splits).
> - Track convergence using:
>   - RMSE (explicit feedback)
>   - Precision@K / Recall@K (implicit feedback)
> - Stop training when validation metrics saturate.
>
> ### 9. Scaling to 25M / 32M Ratings
> - Naive Python implementations are acceptable **only** for debugging on small data.
> - Final experiments must:
>   - Use vectorization and/or Numba
>   - Avoid nested Python loops over ratings
>   - Demonstrate feasible runtime at scale
>
> ### 10. Reporting Optimization Decisions
> - Clearly describe:
>   - Which parts were vectorized
>   - Which parts used Numba
>   - Why certain loops could not be vectorized
> - Optimization choices are part of the **final report evaluation**.
>
> > **Rule of thumb:**  
> > If your implementation cannot train on MovieLens-25M within a reasonable time,
> > it is **not considered complete**, even if mathematically correct.


**Ulrich Paquet**

---

## Credits / Contributors

* **Yvan Marcel Carré**, Vilmorin, AIMS 2024
* **Emmanuel Kwame Ayanful**, AIMS 2025
* **Isaac Houngue**, AIMS 2025

---

## The Early Days

### The Story of Greg Linden and Amazon Recommendations

Amazon’s early recommender systems played a key role in demonstrating the business value of personalization at scale. Greg Linden’s work at Amazon popularized the idea that *recommendations materially change user behavior and revenue*.

---

## Netflix – Predicting Ratings

Netflix built an early recommender system to predict user ratings for movies.

### The Netflix Prize

**Goal**
Achieve a **10% improvement in RMSE** over Netflix’s internal recommender system, *Cinematch*.

* Trivial mean-score prediction:

  * RMSE ≈ **1.0540**
* Cinematch:

  * RMSE ≈ **0.9514**
* Target to win:

  * RMSE ≤ **0.8572**

---

### Netflix Prize Timeline

* **October 2, 2006** – Competition started

  * After 6 days: Cinematch beaten
  * After 1 week: First team qualified for annual progress prize (1% improvement)
  * After 1 year: 40,000+ teams from 186 countries

* **November 13, 2007**

  * BellKor (AT&T): RMSE = **0.8712** (8.43% improvement)
  * $50,000 progress prize

* **2008 Progress Prize**

  * BellKor + BigChaos (Toscher & Jahrer)
  * RMSE = **0.8616**
  * Ensemble of **207 predictors**

* **June 26, 2009**

  * BellKor + BigChaos + Pragmatic Theory
  * RMSE = **10.05% improvement**
  * Entered 30-day “last call”

* **July 26, 2009** – Submissions closed

* **September 18, 2009**

  * Winner: **BellKor’s Pragmatic Chaos**
  * Same score as runner-up, but submitted **20 minutes earlier**

---

## Introduction – Reading List

* Yehuda Koren
  **Matrix Factorization Techniques for Recommender Systems**
* **Yahoo! Music Recommendations**
  Modeling Music Ratings with Temporal Dynamics and Item Taxonomy
* **The Xbox Recommender System**

---

## A New Component in Xbox 360

### Online Components

* Runtime recommender
* Model parameters

### Offline Components

* Offline storage
* Offline modeling

Telemetry flows from the Xbox device to offline systems, enabling large-scale learning.

---

## The Goal of Recommendation

Given a user, recommend items they are likely to enjoy.

> “You may also like…”

Recommendations depend on **past interactions**, not explicit instructions.

---

## The Kind of Data

* Bipartite user–item graphs
* Sparse interaction matrices
* Long-tail distributions

---

## Degree Distributions

Most users interact with **few items**.
A few users interact with **many items**.

---

## Power Laws

### Observations

* Item popularity follows a **power-law**
* User activity often follows a **truncated power-law**

Formally:

[
p(k) \propto k^{-\gamma}
]

[
\log p(k) \sim -\gamma \log k
]

* Least popular items dominate the catalog
* Most popular items dominate traffic

---

## Scale-Free Networks

* No intrinsic scale
* The average degree ⟨k⟩ does **not** characterize the system
* The maximum degree ( k_{\max} ) increases with system size

Moments diverge for ( n > \gamma - 1 ).

---

## Zipf’s Law

Word frequency is inversely proportional to rank:

[
\text{frequency} \propto \frac{1}{\text{rank}}
]

Observed in:

* Language
* Web traffic
* Product views
* Media consumption

---

## Power-Law Distributions in Online Data

* Many products
* Few blockbusters
* Heavy-tailed popularity curve

This has **major implications** for:

* Evaluation
* Overfitting
* Bias handling
* Recommendation ranking

---

## Recommendation Approaches

1. **Feature-based**
2. **Nearest-neighbor**
3. **Collaborative filtering**
4. **Blended systems**
5. **Modern RL-based approaches**

   * Maximize *expected future reward*

---

## Feature-Based Example

Binary genre vectors:

```
[ adventure, animated, boys, cars, kids ]
[ 0 1 1 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 ]
```

Similarity via **Jaccard similarity**.

---

## Xbox Movies and Games (2010–2011)

* Terabytes of interaction logs
* Offline training
* Online serving
* Early industrial-scale recommender systems

---

## The Netflix Prize (Context Recap)

* $1M prize
* One submission per day
* Winning solutions were **ensembles**
* No single “best” model

---

## Indexing Twice

Data indexed:

* By **user**
* By **item**

This enables efficient parallel updates in matrix factorization.

---

## Practical 1 — Understanding the Data

### Reading List

* Koren – Matrix Factorization
* Yahoo! Music recommendations
* Xbox recommender system

---

### Practical 1 Tasks

1. Download **MovieLens 32M**
2. Understand the README
3. Inspect:

   * `movies.csv`
   * `ratings.csv`
   * `tags.csv`
4. Build data structures indexed by:

   * User
   * Movie
5. Plot:

   * Rating distributions
6. Analyze:

   * Power laws
   * Scale-free properties

---

# Matrix Factorization Foundations

## Neural Network Interpretation

* Each **user** is represented as a one-hot vector
  (millions of dimensions)
* Each user is embedded into a **low-dimensional vector**
* Same for items

This is equivalent to learning **embedding layers**.

---

## A Simple Likelihood

We model observed ratings ( r_{mn} ) using latent vectors:

[
r_{mn} \approx u_m^\top v_n
]

Assume Gaussian noise:

[
p(r_{mn} \mid u_m, v_n) = \mathcal{N}(u_m^\top v_n, \tau^{-1})
]

---

### Simplification

To simplify exposition:

* **Biases are temporarily removed**
* Focus only on latent vectors

---

## Objective Function (Log-Likelihood)

Summing over all observed user–item pairs:

[
\mathcal{L} =
-\frac{\tau}{2}
\sum_{(m,n)\in\mathcal{D}}
(r_{mn} - u_m^\top v_n)^2
]

---

## Regularization

Add Gaussian priors:

[
p(u_m) = \mathcal{N}(0, \lambda^{-1} I)
]

[
p(v_n) = \mathcal{N}(0, \lambda^{-1} I)
]

Resulting **regularized objective**:

[
\mathcal{L}_{reg}
=================

-\frac{\tau}{2}
\sum_{(m,n)}
(r_{mn} - u_m^\top v_n)^2
-------------------------

\frac{\lambda}{2}
\sum_m |u_m|^2
--------------

\frac{\lambda}{2}
\sum_n |v_n|^2
]

---

## Gradient Descent

Gradient with respect to user vector ( u_m ):

[
\nabla_{u_m} =
\tau
\sum_{n \in \mathcal{I}(m)}
(r_{mn} - u_m^\top v_n) v_n
---------------------------

\lambda u_m
]

Similar expression for ( v_n ).

---

## Gradient Descent Dynamics

* Positive interactions pull vectors together
* Negative residuals push vectors apart
* Similar users/items cluster in embedding space

---

## A “Good” Solution

* Similar users/items → **large positive inner product**
* Dissimilar users/items → **negative inner product**
* Embeddings encode semantics implicitly

---

## Low-Rank Approximation View

Matrix factorization approximates the rating matrix:

[
R \approx U V^\top
]

Where:

* ( U \in \mathbb{R}^{M \times K} )
* ( V \in \mathbb{R}^{N \times K} )
* ( K \ll \min(M, N) )

---

## Alternating Least Squares (ALS)

Instead of SGD:

* Fix ( V ), optimize ( U )
* Fix ( U ), optimize ( V )

Each subproblem is **convex**.

---

## Gradient Descent vs ALS

| Method | Pros           | Cons                 |
| ------ | -------------- | -------------------- |
| SGD    | Flexible       | Hard to parallelize  |
| ALS    | Parallelizable | Requires full passes |

---

## ALS Update Structure

```
repeat:
  for each user m (parallel):
    update user bias
    update user vector u_m

  for each item n (parallel):
    update item bias
    update item vector v_n
```

---

## Bipartite Graph Interpretation

* Users and items form a **bipartite graph**
* ALS updates correspond to:

  * Fixing one partition
  * Solving least squares on the other

---

## Embarrassingly Parallel Computation

ALS updates:

* Independent across users
* Independent across items
* Ideal for distributed systems

---

## Class Exercise (Conceptual)

Derive the update for ( u_m ):

[
u_m =
\left(
\lambda I + \tau \sum_{n \in \mathcal{I}(m)} v_n v_n^\top
\right)^{-1}
\left(
\tau \sum_{n \in \mathcal{I}(m)} r_{mn} v_n
\right)
]

---

## Initializing Embeddings

* Random initialization
* Rotational invariance:

  * Multiple equivalent solutions exist

---

## Practical Properties

In production, we need only:

* One vector per user
* One vector per item
* Fast dot products at runtime

---

## Making Predictions

At runtime:

[
\hat{r}_{mn} = u_m^\top v_n
]

Rank items by descending score.

---

## Evaluation Protocol

### Dataset Splits

* **Train set**
* **Validation set**
* **Test set**

Recommender datasets are **time-dependent**.

---

### Time-Based Splits

* Random splits are unrealistic
* Use last ( k ) interactions per user for validation/test

---

## Long-Tail Considerations

* Heavy users dominate data
* Do not remove them
* Instead:

  * Hold out last interactions per user

---

## Practical 2 — Biases Only Model

### Task Overview

* Download **smaller MovieLens dataset**
* Index by user and item
* Split into train/test

---

### Model

[
r_{mn} \approx \mu + b_m + b_n
]

Where:

* ( b_m ): user bias
* ( b_n ): item bias

---

### ALS for Biases

Initialization:

```python
user_biases = np.zeros(M)
item_biases = np.zeros(N)
```

Iterative updates:

```
repeat:
  for each user m:
    update user_bias[m]

  for each item n:
    update item_bias[n]
```

---

### User Bias Update

[
b_m =
\frac{
\sum_{n \in \mathcal{I}(m)} (r_{mn} - b_n)
}{
\lambda |\mathcal{I}(m)| + \gamma
}
]

---

### Evaluation Metrics

* Negative log likelihood
* RMSE on:

  * Training set
  * Test set

---

## Practical 3 — Embeddings Model

### Extensions

Add:

* User vectors ( u_m )
* Item vectors ( v_n )
* Regularization

---

### ALS Update Pattern

Important:

* **Update bias first**
* **Update vectors second**

Common bug:

> Updating everything in one loop

---

# Large-Scale Training & Generalization

## Practical 4 — Going Big

### Dataset

* MovieLens **25M or 32M**
* Time-based split into:

  * Training
  * Test

> No cross-validation in this practical — focus on **overfitting detection**.

---

## Practical 4 Tasks

1. Create a **sensible time-based split**
2. Train using **training set only**
3. Measure RMSE on:

   * Training set
   * Test set
4. Plot both curves together
5. Observe:

   * Overfitting
   * Underfitting

---

## Latent Dimension Analysis

Questions to answer:

* Does **K = 20** overfit more than **K = 10**?
* For which users?
* Does regularization help?
* Effect of λ and τ?

---

## Full Dataset Training

Once hyperparameters are chosen:

* Train on **all ratings**
* Parallelize ALS if needed

---

## Sanity Check: Dummy User

Create a synthetic user:

* Rates *“Lord of the Rings”* as 5 stars

Expected outcome:

* Top recommendations are other LOTR movies

---

## Polarizing Movies

Definition:

* Movies that **rapidly separate users**
* Long trait vectors ( |v_n| )

Used to:

* Quickly infer user taste

---

## Recommendation Scoring

Baseline scoring:

[
\text{score}_n = u_m^\top v_n + b_n
]

Issue:

* Popular items dominate rankings

---

## Trick: Downweight Item Bias

Improved scoring:

[
\text{score}_n = u_m^\top v_n + 0.05 \cdot b_n
]

Effect:

* More personalized recommendations

---

## Rare Item Filtering

Problem:

* Items with very few ratings appear at top

Solution:

* Filter items with < 100 ratings

---

## Why Predictions Look Bad

Reasons:

* Overfitting
* Long-tail effects
* Rare-item noise
* Too-large latent dimension

---

# Bayesian Remedies

## Averaging Over the Posterior

Instead of point estimates:

[
p(r \mid \mathcal{D}) =
\int p(r \mid U, V) p(U, V \mid \mathcal{D}) dU dV
]

---

## Problem

* Integral is intractable
* Need approximations

---

## MCMC Approach

* Sample from posterior
* Average predictions
* Problems:

  * Memory scales with samples
  * Runtime scales with samples

---

## Mean Field Variational Inference

Approximate posterior:

[
p(U, V \mid \mathcal{D}) \approx q(U) q(V)
]

Store:

* Mean vector
* Variance vector

---

## Variational Objective

Maximize ELBO:

[
\mathcal{L}(q) =
\mathbb{E}_q[\log p(R, U, V)]
-----------------------------

\mathbb{E}_q[\log q(U, V)]
]

---

## Advantages

* Captures uncertainty
* Same storage order as ALS
* Same computational complexity

---

## Simple Example

One user, one item, one rating.

* Joint density visualization
* Shows posterior uncertainty

---

## Class Practical

Recreate joint density plots:

```python
u = np.arange(-5, 5, 0.25)
v = np.arange(-5, 5, 0.25)
U, V = np.meshgrid(u, v)
P = np.exp(-0.5 * tau * (U**2 + V**2) ...)
```

---

## Variational Updates

* Closed-form
* Similar to Gibbs sampling
* Component-wise updates

---

## Comparison

| Method        | Storage | Runtime | Uncertainty |
| ------------- | ------- | ------- | ----------- |
| MAP           | Low     | Fast    | ❌           |
| MCMC          | High    | Slow    | ✅           |
| Mean-field VB | Medium  | Fast    | ✅           |

---

## Production Transition

### Implicit Feedback

* No explicit ratings
* Only positive signals

Examples:

* Watching a movie
* Clicking an item
* Playing a game

---

## One-Class Collaborative Filtering

* Missing ≠ negative
* Treat unobserved as *unknown*
* Sample negatives stochastically

---

## Random Negative Graphs

Likelihood defined as expectation over:

* All possible negative graphs
* Same power-law distribution

Optimized via **SGD**

---

## Practical Insight

A basic probabilistic model is **almost production-ready**.

Storage:

* Mean & variance per user
* Mean & variance per item
* Biases

---

# Adding Features & Cold Start

## Motivation

Latent factor models fail when:

* Items have **no ratings**
* Users are **new**

Solution:

* Incorporate **side information**

---

## Adding Features

Examples:

* Genres
* Categories
* Metadata

Binary feature vectors:

```
[ adventure, animated, boys, cars, kids ]
[ 0 1 1 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 ]
```

---

## Better Priors via Features

Hierarchical Bayesian model:

```
prior → features → items → usage ← users ← prior
```

Instead of:

```
prior → items → usage ← users ← prior
```

---

## Feature-Conditioned Item Prior

[
p(v_n \mid F_n)
]

Where:

* ( F_n ) = feature vector of item ( n )

---

## Loss Function (Without Features)

[
\mathcal{L} =
\sum_{(m,n)}
(r_{mn} - u_m^\top v_n)^2
+
\lambda (|u_m|^2 + |v_n|^2)
]

---

## Loss Function (With Features)

Add term:

[
| v_n - W f_n |^2
]

Where:

* ( f_n ): feature vector
* ( W ): feature embedding matrix

---

## Benefits

* Shared statistical strength
* Cold-start mitigation
* Better generalization

---

## Posterior Means of Features

* Learn embedding for each feature
* Features cluster semantically
* Improve early predictions

---

## Probabilistic Graphical Model

```
u_m → r_{mn} ← v_n
        ↑
       f_n
```

---

## Practical 5 — Adding Features

### Task Overview

* Use MovieLens genres
* Extend ALS or BPR model
* Incorporate feature vectors

---

### Dataset Example

```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
```

---

## Practical 5 Tasks

1. Build genre feature matrix
2. Modify loss function
3. Update:

   * User vectors
   * Item vectors
   * Feature vectors

---

## ALS Structure with Features

```
repeat:
  for users:
    update biases
    update user vectors

  for items:
    update biases
    update item vectors

  for features:
    update feature vectors
```

---

## Cold Start Behavior

* Items with no ratings:

  * Still get meaningful embeddings
* Recommendations improve immediately

---

## Practical 5 — Optional Extra

Investigate:

* How quickly cold-start items improve
* Feature importance

---

## Transition to Ranking Problems

* Explicit ratings → regression
* Implicit feedback → ranking

---

# Implicit Feedback & Ranking

## From Ratings to Actions

### Feedback Types

**Explicit feedback**

* 5-star ratings (Netflix)
* 0–100 scores (Yahoo! Music)
* Thumbs up / down

**Implicit feedback**

* Watching a movie
* Clicking an article
* Purchasing an item
* Playing / skipping a song

---

## Why Explicit Feedback Is Hard

1. Explicit ratings often **do not exist**
2. Much more implicit feedback than explicit
3. Ratings depend on:

   * Context
   * Order
   * Time
4. Users are inconsistent

---

## Missing Not At Random (MNAR)

* Users rate what they consume
* Unconsumed ≠ disliked
* Even low ratings are *positive signals* (because consumption occurred)

Reference:

> Marlin et al., *Collaborative Filtering and the Missing at Random Assumption*

---

## Bias Dominates Signal

In many datasets:

* User bias + item bias explain most variance
* Personalization contributes less than expected

---

## R² in Collaborative Filtering

[
R^2 = 1 - \frac{\text{MSE}}{\text{Var(data)}}
]

* Mean predictor → ( R^2 = 0 )
* Negative ( R^2 ) → model worse than mean

---

### Netflix Prize Breakdown

* Total variance ≈ **1.276**
* Winning RMSE ≈ **0.85**
* ( R^2 ≈ 0.43 )

  * **0.33 from biases**
  * **0.10 from personalization**

---

## Issues with Implicit Feedback (1)

* No negative labels
* Mostly positive signals
* One-class problem
* Signals can be cumulative (e.g. play count)

---

## Issues with Implicit Feedback (2)

* Noisy signals
* External factors (sales, availability)
* Evaluation is harder

---

## Explicit vs Implicit Feedback

| Aspect    | Explicit   | Implicit              |
| --------- | ---------- | --------------------- |
| Signal    | Ratings    | Actions               |
| Volume    | Sparse     | Abundant              |
| Stability | Noisy      | More stable           |
| Models    | Regression | Ranking               |
| Metrics   | RMSE, MAE  | Precision@K, Recall@K |

---

## Prominent Implicit Feedback Papers

* **Hu, Koren, Volinsky** — CF for Implicit Feedback
* **Rendle et al.** — Bayesian Personalized Ranking (BPR)
* **Takács & Tikk** — RankALS
* **Shi et al.** — CLiMF

---

# Ranking Objectives

## Why Ranking?

We care about:

* **Top-K recommendations**
* Not precise rating prediction

---

## Practical 6 — Ranking

### Reading List

* BPR: Bayesian Personalized Ranking
* One-class Collaborative Filtering with Random Graphs
* Dive into Deep Learning — §16.5

---

## Practical 6 Tasks

1. Convert MovieLens into **implicit feedback**
2. Treat high ratings (e.g. ≥ 4) as positive
3. Implement **BPR with SGD**
4. Compare with ALS

---

## Bayesian Personalized Ranking (BPR)

For user ( u ), positive item ( i ), negative item ( j ):

[
\mathcal{L}_{BPR} =

* \log \sigma(\hat{r}*{ui} - \hat{r}*{uj})

- \lambda |\Theta|^2
  ]

Where:

* ( \hat{r}_{ui} = u^\top v_i )
* ( \sigma ) is the sigmoid

---

## BPR Optimization

* Sample triples ( (u, i, j) )
* Perform SGD
* Negative items sampled randomly

---

## Practical 6 — Evaluation

Use:

* Precision@K
* Recall@K

---

## Precision@K

[
\text{Precision@K} =
\frac{\text{relevant items in top K}}{K}
]

---

## Recall@K

[
\text{Recall@K} =
\frac{\text{relevant items in top K}}{\text{total relevant items}}
]

---

## Ranking Illustration

Ground truth positives:

* Item 1
* Item 2
* Item 3

Ranking induced by algorithm:

1. Positive
2. Positive
3. Negative
4. Positive

---

## Bird’s-Eye View

* Explicit → regression
* Implicit → ranking
* Different losses
* Different evaluation

---

# Evaluation of Recommender Systems

## Offline vs Online Evaluation

* **Offline**:
  Uses historical data
  Cheap, fast, but biased

* **Online**:
  Uses live users
  Expensive, slow, but causal

---

## Offline Evaluation (Implicit Feedback)

### Precision@K / Recall@K

Used when:

* Only positive feedback is observed
* We evaluate ranking quality

---

## Precision@K

Measures *how many recommended items are relevant*:

[
\text{Precision@K} =
\frac{\text{# relevant items in top K}}{K}
]

Used for:

* Implicit feedback
* Top-K recommendation quality

---

## Recall@K

Measures *how many relevant items are recovered*:

[
\text{Recall@K} =
\frac{\text{# relevant items in top K}}{\text{# relevant items}}
]

---

## Precision–Recall Tradeoff

* Increasing K → higher recall
* Increasing K → lower precision

---

## Mean Average Precision (MAP)

* Computes precision at each relevant position
* Averages over users

---

## NDCG@K

**Normalized Discounted Cumulative Gain**

Accounts for:

* Relevance scores
* Ranking position

Example relevance:

* Item 1: relevance 5
* Item 2: relevance 3
* Item 3: relevance 1

---

## Mean Percentile Rank (MPR)

* Often used in implicit feedback
* Lower is better

Used when:

* Only one positive item exists in test set

---

## Summary of Offline Metrics

| Metric      | Used for | Notes            |
| ----------- | -------- | ---------------- |
| RMSE        | Explicit | Rating accuracy  |
| Precision@K | Implicit | Top-K quality    |
| Recall@K    | Implicit | Coverage         |
| MAP         | Implicit | Ranking quality  |
| NDCG        | Implicit | Graded relevance |
| MPR         | Implicit | Sparse positives |

---

# Online Experiments

## Why Online Evaluation?

Offline metrics:

* Do not capture user behavior change
* Cannot establish causality

---

## Randomized Controlled Experiments

Also known as:

* **A/B tests**

Measure:

* KPIs (Key Performance Indicators)

Examples:

* Click-through rate
* Watch time
* Conversion rate

---

## Causality: Data vs HiPPO

**HiPPO**:

> Highest Paid Person’s Opinion

Data beats intuition.

---

## A/B Testing Setup

* Randomly split users:

  * Group A
  * Group B
* Serve different models
* Measure KPIs

---

## Logging Requirements

Must log:

* Experiment ID
* User ID
* Model version
* Feedback

---

## Example Log Record

```
user_id, item_id, feedback, model_version
```

---

## Common Pitfalls

* Wrong KPI
* Too short experiment
* External confounders
* Cannibalization effects
* High engineering cost

---

## Resources on A/B Testing

* Ron Kohavi — A/B Testing Lectures
* KDD 2015 Keynote
* *Seven Pitfalls to Avoid When Running Controlled Experiments*
* *Unexpected Results in Online Controlled Experiments*

---

## Class Work Assignments

### Group Presentations

Topics:

1. Unexpected Results in Online Controlled Experiments
2. Seven Pitfalls to Avoid
3. Practical Guide to Controlled Experiments
4. Controlled Experiments Survey

---

## Bonus Reading

Léon Bottou et al.
**Counterfactual Reasoning and Learning Systems**

---

# Practical 7 — Building in A/B Testing

## Goal

Wrap your recommender system in a **mock production setup** and demonstrate:

* Logging
* Controlled experimentation
* KPI comparison

---

## System Wrapper

You should expose a simple recommendation API, e.g.:

```python
def createreco(userid, versionid):
    """
    Returns a list of (movie_id, title)
    """
    ...
```

Where:

* `userid` identifies the user
* `versionid` identifies the model variant (A or B)

---

## Simulated User Feedback

1. Serve a recommendation list
2. Simulate user interaction
3. Record feedback

Example feedback signals:

* Click
* Watch
* Like / dislike

---

## Logging

Each interaction should be written to disk (e.g. CSV):

```
user_id, item_id, feedback
```

Extended logging for A/B testing:

```
user_id, item_id, feedback, model_version
```

---

## Dummy Users

* Create ~100 synthetic users
* Simulate interaction behavior
* Assign users randomly to:

  * Group A
  * Group B

---

## A/B Test Setup

1. Randomly split users
2. Serve recommendations using:

   * Model A (baseline)
   * Model B (modified)
3. Track interactions
4. Compare KPIs

---

## Example Model Change

Change **one thing only**, for example:

* Down-weight item bias during ranking

```python
score = u.dot(v) + 0.05 * item_bias
```

vs baseline:

```python
score = u.dot(v) + item_bias
```

---

## Evaluation

* Aggregate KPIs per group
* Example KPIs:

  * Click-through rate
  * Watch rate
  * Precision@K

---

## Statistical Significance

Perform a statistical test:

* t-test
* permutation test

Question:

> Is the difference between A and B statistically significant?

---

## Outcome

Demonstrate:

* Which model performs better
* Why
* How confident you are

---

## Optimization Loop

> Optimize the user experience and KPIs
> using data
> lots of data

---

Here is **Part 8 / 8 (FINAL)** of the **complete Markdown conversion**, finishing the document with **final report instructions, grading rubric, bonuses, and course wrap-up**.
This concludes the **full, faithful PDF → Markdown conversion** .

---

# Final Report

## Purpose

The final report evaluates your ability to:

* Understand recommender system theory
* Implement scalable algorithms
* Analyze results rigorously
* Communicate like a research engineer

This is a **practice run for real research writing**.

---

## Scope

The course contains **four core practicals**, spread over three weeks, plus extensions.

Your report must showcase:

* Your recommender system prototype
* The full modeling pipeline
* Empirical results
* Clear explanations

---

## Length & Format

* **Minimum length**: 8 pages
  (excluding technical appendices)
* You may include:

  * Long derivations
  * Extra figures
  * Additional experiments
    in appendices

---

## Style Guide

* Follow **AIMS–ICML Style Guide**
* This is *not* a real ICML submission, but formatting discipline matters
* LaTeX is strongly encouraged

---

## Code Submission

Your source code must be:

* Hosted on **GitHub**
* Shared with GitHub user: `upaq`
* Linked **in the abstract** of your report

Code quality expectations:

* Legible
* Well-structured
* PEP-8 compliant (Python)

---

## Required Report Sections

### 1. Introduction

* Problem motivation
* Prior work
* Context

---

### 2. Problem Statement

* What is being predicted?
* What data is used?
* What assumptions are made?

---

### 3. Dataset Analysis

Include plots for:

* Rating distributions
* Power-law behavior
* Long-tail effects

---

### 4. Models

Describe **all implemented models**, including:

* Bias-only ALS
* ALS with user/item embeddings
* Feature-aware ALS
* Implicit feedback / BPR (if implemented)

Include:

* Loss functions
* Regularization terms
* Optimization procedure

---

### 5. Results (Major Section)

Must include:

#### Train/Test Split

* How it was constructed
* Why it is realistic

---

#### Biases-Only ALS

* RMSE (train + test)
* Interpretation

---

#### ALS with Embeddings (U, V)

* RMSE (train + test)
* Effect of latent dimension K

---

#### Loss Curves

* Loss vs iteration
* Must be monotonic

---

#### 2D Embeddings

* Plot item vectors ( v_n )
* Show that similar movies cluster
* Include biases during training

---

### 6. Feature-Aware Model

Using MovieLens genres:

```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
```

Include:

* Modified loss function
* Optimization steps
* Feature embeddings (2D)
* Cold-start discussion

---

### 7. Hyperparameter Analysis

Describe and justify:

* λ (regularization)
* τ (precision)
* Latent dimension K

Suggested values:

* K = 0, 2, 4, 8, 16, (optionally 32)

Discuss:

* Diminishing returns
* Overfitting behavior

---

### 8. Large-Scale Results (Bonus-Heavy)

If using full 25M or 32M dataset:

* Show runtime optimizations
* Parallelization
* Memory-aware implementation

---

### 9. Recommendation Sanity Checks

Demonstrate:

* Dummy user experiment
  (e.g. 5-star “Lord of the Rings”)
* Recommendations make semantic sense

---

### 10. Polarization Analysis

Identify:

* Most polarizing movies
* Least polarizing movies

Discuss:

* Trait vector norms
* Avoid overfitting explanations

---

## Implicit Feedback (Optional / Bonus)

If implemented:

* Convert explicit → implicit
* Implement **BPR**
* Evaluate with:

  * Precision@K
  * Recall@K
* Discuss limitations

---

## A/B Testing (Optional / Bonus)

Show:

* Mocked recommendation API
* Logging
* Simulated users
* Two model variants
* KPI comparison
* Statistical significance test (e.g. t-test)

---

## Marking Breakdown (100%)

### Core

* 5% Dataset plots
* 4% Bias-only ALS RMSE
* 8% ALS with embeddings
* 9% Code quality
* 5% 2D embeddings
* 7% Genre features
* 4% Hyperparameter selection
* 10% Writing quality

---

### Bonus

* 9% Full 32M dataset + optimizations
* 5% Recommendation sanity check
* 2% Polarization analysis
* 7% Implicit feedback (BPR)
* 5% A/B testing system
* 3% Statistical testing
* 7% Variational Bayes (advanced)

---

## Optional Advanced Bonus

Implement **Variational Bayes**:

* Go beyond point estimates
* Derive ELBO
* Show uncertainty-aware predictions

---

## The End

> Optimize, optimize, optimize
> the user experience
> using data
> lots of data

