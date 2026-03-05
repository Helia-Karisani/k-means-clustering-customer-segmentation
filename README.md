# K-Means Clustering for Customer Segmentation

This project applies **K-Means clustering** to segment customers into groups with similar characteristics.  
It includes:
1) a quick synthetic demo using `make_blobs` (to visualize how K-Means works), and  
2) real customer segmentation on the **Cust_Segmentation.csv** dataset.

---

## What this repo does

- Loads customer data (CSV)
- Cleans it (drops non-numeric / missing values)
- Scales features so different units don’t dominate
- Fits a K-Means model with a chosen number of clusters `k`
- Assigns each customer a cluster label
- Visualizes clusters in 2D + interactive 3D (Plotly) + pairplots (Seaborn)
- Summarizes cluster characteristics using group means

---

## Tech stack

- Python
- NumPy, Pandas
- scikit-learn (KMeans, StandardScaler)
- Matplotlib, Seaborn
- Plotly (interactive 3D)

---

## Data

### Synthetic demo (for intuition)
Generated using:
- `n_samples = 5000`
- `centers = [[4,4], [-2,-1], [2,-3], [1,1]]`
- `cluster_std = 0.9`

Output shapes:
- `X` is the feature matrix with shape `(n_samples, n_features)`  
  Here it’s `(5000, 2)` because each point has 2 features: `(x, y)`.
- `y` is the true blob label for each point (used only for generation, not needed by K-Means).

### Real customer dataset
Loaded from:
- `Cust_Segmentation.csv` (IBM Skills Network / public lab dataset)

Preprocessing:
- Drop `Address` (non-numeric / not used)
- Drop missing rows (`dropna`)
- Use all numeric columns except `Customer ID` as features

---

## Model: K-Means clustering

### What K-Means outputs
After fitting, K-Means produces:
- `labels_`: an array of length `N` where `labels_[i]` is the cluster ID for sample `i`
- `cluster_centers_`: a `(k, d)` array containing the learned centroid coordinates

Important: cluster IDs (0,1,2,...) are just names; they have no inherent order.

---

## Math (plain format)

Let:
- data points be x_1, x_2, ..., x_N, each in R^d
- k = number of clusters
- c_i in {1,...,k} be the cluster assignment for point i
- mu_j be the centroid of cluster j

K-Means solves this optimization problem:

min over assignments (c_1,...,c_N) and centroids (mu_1,...,mu_k):
sum_{i=1..N} || x_i - mu_{c_i} ||^2

Where ||a - b||^2 is squared Euclidean distance.

### The algorithm (iterative)
It alternates between two steps until convergence:

1) Assignment step:
For each point x_i, choose the nearest centroid:
c_i = argmin_{j in {1..k}} || x_i - mu_j ||^2

2) Update step:
For each cluster j, recompute the centroid as the mean of assigned points:
mu_j = (1 / |C_j|) * sum_{i: c_i = j} x_i

This is why `cluster_centers_` usually ends up close to the “true” centers in the blob demo,
but not exactly equal (randomness + estimation).

---

## Why scaling matters

Customer features can have very different magnitudes (example: Age vs Income vs Debt).
Without scaling, large-range features dominate distances and clustering.

This project uses standardization:

For each feature column:
z = (x - mean) / std

In scikit-learn:
`StandardScaler().fit_transform(X)`

---

## Choosing k (number of clusters)

K-Means does NOT automatically discover k.
You must set it (example: `n_clusters=3`).

Common ways to pick k:
- Elbow method: plot inertia (within-cluster SSE) vs k and look for the “elbow”
- Silhouette score: measures how well-separated clusters are

In this notebook, k is selected manually for demonstration.

---

## Visualizations included

- 2D scatter plots of clustered points with centroid markers
- Bubble plot (size based on a feature) for Age vs Income
- Interactive Plotly 3D scatter:
  - axes: Education, Age, Income
  - color: cluster label
- Seaborn pairplot for (Age, Edu, Income) colored by cluster

---

## How to run

1) Install dependencies (example):
- numpy, pandas, matplotlib, scikit-learn, seaborn, plotly, nbformat, ipython

2) Open and run the notebook:
- `k-means-clustering-customer-segmentation.ipynb`

If Plotly errors with:
"Mime type rendering requires nbformat>=4.2.0 but it is not installed"
then install:
- nbformat, ipython
and restart the kernel.
(Alternatively, set Plotly renderer to browser.)

---

## Output / Interpretation

After fitting:
- Each customer gets a cluster label (segment).
- You can inspect:
  `cust_df.groupby('Clus_km').mean()`
to understand typical profiles per segment (average age, income, debt, etc.).

Use these segments for:
- targeted marketing
- product personalization
- risk profiling
- customer lifecycle analysis

---

## Files

- `k-means-clustering-customer-segmentation.ipynb` : main notebook with full workflow
