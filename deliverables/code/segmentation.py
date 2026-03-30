"""
Part 2: Customer Segmentation Model

Goal: Group the population into distinct marketing segments based on shared
demographic and employment characteristics, using unsupervised learning.

Approach (unsupervised - no labels used for clustering):
  1. Load data and handle missing values
  2. Remove label and weight columns (not used for clustering)
  3. One-hot encode categorical text columns + scale all features
  4. Apply PCA to reduce dimensionality (curse of dimensionality)
  5. Determine optimal number of clusters (Elbow + Silhouette)
  6. Fit K-Means on full dataset
  7. Profile each segment and visualize differences
  8. Provide marketing recommendations per segment

Why K-Means:
  - Scales to 200K+ records efficiently
  - Each segment defined by a centroid ("average customer profile")
  - Clean, non-overlapping groups - every customer in exactly one segment
  - Client controls the number of segments via k parameter

Why PCA before clustering:
  - One-hot encoding expands ~40 features to hundreds of binary columns
  - K-Means uses Euclidean distance; in high dimensions, all distances
    become nearly equal ("curse of dimensionality"), making clusters meaningless
  - PCA compresses to components capturing 90% of variance, giving K-Means
    cleaner distances to work with
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib
# matplotlib.use('Agg')  # Uncomment this line if running without a display (e.g., server)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# =============================================================================
# Step 1: Load and Preprocess Data
# =============================================================================
print("="*60)
print("CUSTOMER SEGMENTATION MODEL")
print("="*60)

# Load data - same format as classification (no header, separate columns file)
df = pd.read_csv("census-bureau.data", header=None)

with open("census-bureau.columns", "r") as f:
    columns = f.read().splitlines()

df.columns = columns

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# --- Handle missing values ---
# Same approach as classification: fill 'hispanic origin' nulls with 'Unknown'
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
df['hispanic origin'] = df['hispanic origin'].fillna('Unknown')

# --- Separate label and weight (not used as clustering features) ---
# The income label is stored for later profiling ("what % of each segment earns >$50K?")
# but is NOT fed into the clustering algorithm - segmentation is unsupervised.
label_col = "label"
y_labels = df[label_col].copy()

# The instance weight column represents how many people in the population each
# record stands for (due to stratified sampling). Excluded from clustering features
# but useful for understanding population-level segment sizes.
weight_candidates = [c for c in df.columns if 'weight' in c.lower() or 'instance' in c.lower()]
weight_col = weight_candidates[0] if weight_candidates else None
if weight_col:
    weights = df[weight_col].copy()
    print(f"Instance weight column detected: '{weight_col}'")
else:
    weights = None
    print("No instance weight column detected.")

# Drop label and weight - only demographic/employment features remain
drop_cols = [label_col]
if weight_col:
    drop_cols.append(weight_col)
X = df.drop(columns=drop_cols)

print(f"\nFeatures for segmentation: {X.shape[1]}")

# =============================================================================
# Step 2: Feature Engineering for Segmentation
# =============================================================================

# Identify column types:
#   - Numeric columns (age, capital gains, weeks worked, etc.) - keep as-is
#   - Object/text columns (education, occupation, industry, etc.) - one-hot encode
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# Strip leading/trailing whitespace from categorical values
# (CSV parsing can leave spaces: " Construction" instead of "Construction")
for col in categorical_cols:
    X[col] = X[col].astype(str).str.strip()

# One-hot encode all text columns into binary 0/1 columns
# drop_first=True removes one category per variable to avoid redundancy
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"Total encoded features: {X_encoded.shape[1]}")

# Scale ALL features to mean=0, std=1
# Critical for K-Means: it uses Euclidean distance, so features with larger
# ranges (e.g., capital gains 0-99999) would dominate features with smaller
# ranges (e.g., binary dummies 0-1) without scaling.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# =============================================================================
# Step 3: Dimensionality Reduction with PCA
# =============================================================================
# After one-hot encoding, we have hundreds of features. PCA compresses them
# into a smaller set of "principal components" - new variables that capture
# the most important patterns (variance) in the data.

# First pass: fit PCA with ALL components to see how variance distributes
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

# Calculate cumulative explained variance
# e.g., [0.15, 0.28, 0.39, ...] means PC1 explains 15%, PC1+PC2 explain 28%, etc.
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find how many components capture 90% of total variance
# We use 90% (not 95% or 99%) as a balance between information retention
# and dimensionality reduction. 90% keeps the signal while discarding noise.
n_components_95 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"\nPCA: {n_components_95} components capture 90% of variance")

# Plot: shows the "diminishing returns" of adding more components
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-o', markersize=2)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_explained_variance.png", dpi=150)
plt.show()
print("Saved: pca_explained_variance.png")

# Second pass: apply PCA with the selected number of components
# This transforms our data from hundreds of features to n_components_95 features
pca = PCA(n_components=n_components_95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# =============================================================================
# Step 4: Determine Optimal Number of Clusters
# =============================================================================
# We test k=2 through k=10 and evaluate each using two methods:
#   1. Elbow Method: plots inertia (within-cluster sum of squared distances)
#      - look for the "elbow" where adding more clusters stops helping much
#   2. Silhouette Score: measures how well-separated clusters are
#      - ranges from -1 (bad) to 1 (perfect), higher = better-defined clusters
print("\nFinding optimal number of clusters...")

# For speed: if dataset is large, sample 30K points for cluster evaluation
# (K-Means on 200K points x 9 values of k is slow; results are similar on a sample)
MAX_SAMPLES = 30000
if X_pca.shape[0] > MAX_SAMPLES:
    np.random.seed(42)
    sample_idx = np.random.choice(X_pca.shape[0], MAX_SAMPLES, replace=False)
    X_sample = X_pca[sample_idx]
    print(f"Using {MAX_SAMPLES} samples for cluster selection (full data: {X_pca.shape[0]})")
else:
    sample_idx = np.arange(X_pca.shape[0])
    X_sample = X_pca

K_range = range(2, 11)  # Test k=2, 3, 4, ..., 10
inertias = []
silhouette_scores = []

for k in K_range:
    # n_init=10: Run K-Means 10 times with different random starting centroids,
    #   keep the best result. Avoids getting stuck in a bad local minimum.
    # max_iter=300: Maximum iterations per run for centroid convergence.
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_sample)
    inertias.append(km.inertia_)  # Sum of squared distances to nearest centroid
    # sample_size=5000: compute silhouette on a subsample for speed
    sil = silhouette_score(X_sample, km.labels_, sample_size=min(5000, len(X_sample)))
    silhouette_scores.append(sil)
    print(f"  k={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.4f}")

# Plot both metrics side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'b-o')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
axes[0].set_title('Elbow Method')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'r-o')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cluster_selection.png", dpi=150)
plt.show()
print("Saved: cluster_selection.png")

# Select optimal k: start with the best silhouette score
optimal_k = K_range[np.argmax(silhouette_scores)]

# Business override: if the best k is only 2, that's too coarse for marketing.
# We check if k=3, 4, or 5 has a silhouette within 70% of the best -
# if so, prefer it for more actionable segmentation.
# (Too few segments = everyone gets the same campaign; too many = impossible to manage)
if optimal_k <= 2 and len(silhouette_scores) > 2:
    best_sil = max(silhouette_scores)
    for candidate_k in [4, 5, 3]:
        idx = candidate_k - 2  # offset because K_range starts at 2
        if idx < len(silhouette_scores) and silhouette_scores[idx] >= 0.7 * best_sil:
            optimal_k = candidate_k
            break

print(f"\nSelected k={optimal_k} clusters")

# =============================================================================
# Step 5: Fit Final K-Means Model on Full Dataset
# =============================================================================
# Now that we've chosen k, fit K-Means on ALL data (not just the sample)
print(f"\nFitting K-Means with k={optimal_k} on full dataset...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(X_pca)  # Returns cluster assignment (0, 1, 2, ...) per row

# Attach segment labels to the original dataframe for profiling
df_segmented = df.copy()
df_segmented['Segment'] = cluster_labels

print(f"\nSegment distribution:")
print(df_segmented['Segment'].value_counts().sort_index())

# =============================================================================
# Step 6: Visualize Clusters in 2D (PCA Projection)
# =============================================================================
# Project the full scaled data to just 2 dimensions for a scatter plot.
# This is a SEPARATE PCA from Step 3 - here we use only 2 components for
# visualization, while clustering used more components for accuracy.
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

segment_names = {i: f"Segment {i}" for i in range(optimal_k)}
colors = plt.cm.Set2(np.linspace(0, 1, optimal_k))

plt.figure(figsize=(12, 8))
for seg in range(optimal_k):
    mask = cluster_labels == seg
    # Sample up to 3000 points per segment for readability
    n_plot = min(mask.sum(), 3000)
    idx = np.where(mask)[0]
    np.random.seed(42)
    plot_idx = np.random.choice(idx, n_plot, replace=False)
    plt.scatter(X_2d[plot_idx, 0], X_2d[plot_idx, 1],
                c=[colors[seg]], label=segment_names[seg],
                alpha=0.3, s=10)

# Axis labels show how much variance each PC captures
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('Customer Segments (PCA 2D Projection)')
plt.legend(markerscale=3, fontsize=10)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("segments_2d.png", dpi=150)
plt.show()
print("Saved: segments_2d.png")

# =============================================================================
# Step 7: Profile Each Segment
# =============================================================================
# This is the most important output for the client: what does each segment
# "look like" in terms of demographics, employment, and income?
print("\n" + "="*60)
print("SEGMENT PROFILES")
print("="*60)

# Map income label to numeric (0/1) for calculating % high-income per segment
# Note: here >=$50K = 1 for easier interpretation ("% high income")
df_segmented['income_binary'] = df_segmented[label_col].map({
    '- 50000.': 0, '50000+.': 1
})

# --- Numeric feature averages per segment ---
# Shows how segments differ on continuous variables (age, weeks worked, etc.)
profile_numeric = [c for c in numeric_cols if c in df_segmented.columns]
segment_profile_numeric = df_segmented.groupby('Segment')[profile_numeric].mean().round(2)
print("\nNumeric Feature Averages by Segment:")
print(segment_profile_numeric.T.to_string())

# --- Income distribution per segment ---
# The most actionable metric: which segments have more high-income customers?
income_by_segment = df_segmented.groupby('Segment')['income_binary'].agg(['mean', 'count'])
income_by_segment.columns = ['pct_high_income', 'count']
income_by_segment['pct_high_income'] = (income_by_segment['pct_high_income'] * 100).round(1)
print("\nIncome Distribution by Segment:")
print(income_by_segment)

# --- Most common category per segment (key categorical features) ---
# For each segment, find the mode (most frequent value) of important categoricals
# This tells us: "Segment 0 is mostly private-sector, high-school educated, etc."
key_categoricals = [c for c in categorical_cols if c in df_segmented.columns]
interesting_cats = []
for col in key_categoricals:
    if any(keyword in col.lower() for keyword in [
        'education', 'occupation', 'sex', 'race', 'marital', 'class',
        'industry', 'citizen', 'employ'
    ]):
        interesting_cats.append(col)

if interesting_cats:
    print("\nMost Common Category per Segment (key features):")
    for col in interesting_cats[:8]:  # Limit to top 8 most relevant
        modes = df_segmented.groupby('Segment')[col].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
        )
        print(f"\n  {col}:")
        for seg, mode_val in modes.items():
            print(f"    Segment {seg}: {mode_val}")

# =============================================================================
# Step 8: Segment Comparison Visualizations
# =============================================================================
# Four-panel dashboard showing how segments differ on key variables

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Income distribution (stacked bar) - which segments are richer?
income_crosstab = pd.crosstab(df_segmented['Segment'], df_segmented[label_col], normalize='index')
income_crosstab.plot(kind='bar', stacked=True, ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Income Distribution by Segment')
axes[0, 0].set_ylabel('Proportion')
axes[0, 0].set_xlabel('Segment')
axes[0, 0].legend(['<$50K', '>=$50K'], loc='best')
axes[0, 0].tick_params(axis='x', rotation=0)

# Panel 2: Age distribution (box plot) - which segments are younger/older?
df_segmented.boxplot(column='age', by='Segment', ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Segment')
axes[0, 1].set_xlabel('Segment')
axes[0, 1].set_ylabel('Age')
plt.sca(axes[0, 1])
plt.title('Age Distribution by Segment')

# Panel 3: Weeks worked (box plot) - employment intensity by segment
if 'weeks worked in year' in df_segmented.columns:
    df_segmented.boxplot(column='weeks worked in year', by='Segment', ax=axes[1, 0])
    axes[1, 0].set_title('Weeks Worked by Segment')
    axes[1, 0].set_xlabel('Segment')
    axes[1, 0].set_ylabel('Weeks')
    plt.sca(axes[1, 0])
    plt.title('Weeks Worked by Segment')

# Panel 4: Capital gains (bar chart) - investment activity by segment
if 'capital gains' in df_segmented.columns:
    cap_gains_by_seg = df_segmented.groupby('Segment')['capital gains'].mean()
    cap_gains_by_seg.plot(kind='bar', ax=axes[1, 1], color='teal')
    axes[1, 1].set_title('Average Capital Gains by Segment')
    axes[1, 1].set_xlabel('Segment')
    axes[1, 1].set_ylabel('Capital Gains ($)')
    axes[1, 1].tick_params(axis='x', rotation=0)

plt.suptitle('Segment Comparison Dashboard', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("segment_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: segment_comparison.png")

# Pie chart: relative size of each segment
plt.figure(figsize=(8, 8))
seg_counts = df_segmented['Segment'].value_counts().sort_index()
plt.pie(seg_counts.values,
        labels=[f'Segment {i}\n(n={c:,})' for i, c in zip(seg_counts.index, seg_counts.values)],
        autopct='%1.1f%%',
        colors=colors[:optimal_k],
        startangle=90)
plt.title('Segment Size Distribution')
plt.tight_layout()
plt.savefig("segment_sizes.png", dpi=150)
plt.show()
print("Saved: segment_sizes.png")

# =============================================================================
# Step 9: Marketing Recommendations
# =============================================================================
print("\n" + "="*60)
print("MARKETING RECOMMENDATIONS")
print("="*60)

print("""
Based on the segmentation analysis, the retail client can use these segments
for targeted marketing strategies:

1. SEGMENT-SPECIFIC TARGETING:
   - Each segment represents a distinct demographic group with different
     income levels, employment patterns, and demographic characteristics.
   - Marketing messages and product offerings should be tailored to the
     dominant characteristics of each segment.

2. INCOME-AWARE CAMPAIGNS:
   - Segments with higher % of high-income individuals (>$50K) should receive
     premium product marketing and loyalty programs.
   - Segments with predominantly lower income (<$50K) should receive
     value-oriented promotions and essential product campaigns.

3. LIFECYCLE MARKETING:
   - Age and employment patterns differ across segments, enabling the client
     to design age-appropriate and employment-status-aware campaigns.

4. CHANNEL STRATEGY:
   - Demographic differences across segments can inform which marketing
     channels (digital, direct mail, in-store) will be most effective
     for each group.

See the segment profiles above for specific characteristics of each group.
""")

# =============================================================================
# Step 10: Save Segmented Data
# =============================================================================
# Export the full dataset with segment assignments so the client can use it
# for campaign targeting, further analysis, or CRM integration.
output_file = "segmented_data.csv"
df_segmented.to_csv(output_file, index=False)
print(f"Saved segmented dataset to: {output_file}")

print("\nSegmentation complete.")