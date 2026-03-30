# JPMC-Project

## Overview

This project addresses two objectives using U.S. Census Bureau data (1994-1995 Current
Population Surveys, 199,523 records, 40 demographic/employment variables):

1. **Income Classification** — Binary classifier predicting whether an individual earns
   less than or more than $50,000. Best model (XGBoost) achieves 96% accuracy, AUC=0.950.
2. **Customer Segmentation** — Unsupervised clustering model that groups the population
   into distinct marketing segments. Identifies 3 population groups (retirees, dependents,
   workforce) and sub-segments the workforce into actionable marketing targets.

## Prerequisites

- Python 3.8+
- Required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## Data Files

Place these files in the project root directory:

- `census-bureau.data` — Census data 
- `census-bureau.columns` — Column header names 

## Running the Code

### Part 1: Income Classification

```bash
python deliverables/code/classification.py
```

Trains and evaluates three models in sequence:
- **Random Forest** — identifies which features matter (85% accuracy, AUC=0.937)
- **Logistic Regression** — reveals how each feature affects the outcome (85% accuracy, AUC=0.941)
- **XGBoost** — maximizes predictive accuracy (96% accuracy, AUC=0.950)

### Part 2: Customer Segmentation

```bash
python deliverables/code/segmentation.py
```

Builds a K-Means clustering model with PCA dimensionality reduction:
1. Preprocesses and one-hot encodes all features (40 to 380)
2. Applies PCA for dimensionality reduction (213 components for 90% variance)
3. Determines optimal cluster count via Elbow and Silhouette methods
4. Fits K-Means (k=3) and profiles each segment

### Part 3: Supplementary Analysis

```bash
python deliverables/code/supplementary_analysis.py
```

Additional analyses:
- ROC curves and AUC scores for all three classifiers
- Logistic Regression classification report
- Education level vs income breakdown
- Workforce-only sub-segmentation (filtering out children and retirees)

## Project Structure

```
deliverables/
├── README.md                          # This file
├── project_report.md                  # Project report with embedded figures
├── code/
│   ├── classification.py              # Part 1: Income classification (RF, LR, XGBoost)
│   ├── segmentation.py                # Part 2: Customer segmentation (K-Means + PCA)
│   └── supplementary_analysis.py      # ROC/AUC, education analysis, workforce sub-segments
└── plots/
    ├── rf_feature_importance.png      # Top 10 feature importances (Random Forest)
    ├── xgb_confusion_matrix.png       # Confusion matrix heatmap (XGBoost)
    ├── roc_curves.png                 # ROC curves for all 3 classifiers
    ├── education_vs_income.png        # Income % by education level
    ├── pca_explained_variance.png     # Cumulative PCA variance
    ├── cluster_selection.png          # Elbow and Silhouette analysis
    ├── segment_sizes.png              # Segment size pie chart
    ├── segment_comparison.png         # Segment comparison dashboard
    ├── segments_2d.png                # 2D PCA scatter of segments
    ├── workforce_segments.png         # Workforce sub-segmentation dashboard
    └── workforce_segments_2d.png      # Workforce segments 2D scatter
```
