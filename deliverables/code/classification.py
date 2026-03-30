"""
Part 1: Income Classification Model

Goal: Predict whether a person earns <$50K or >=$50K based on 40 demographic
and employment variables from the U.S. Census Bureau (1994-1995 CPS data).

Approach:
  - We train three models in sequence, each answering a different business question:
    1. Random Forest   -> "Which features matter?" (feature importance ranking)
    2. Logistic Regression -> "How does each feature affect the outcome?" (signed coefficients)
    3. XGBoost         -> "What is the best prediction we can make?" (highest accuracy)

Flow:
  Load data -> Handle missing values -> Encode label (risk framing) ->
  One-hot encode categoricals -> Train/test split -> Train 3 models -> Evaluate
"""

import pandas as pd
import numpy as np
import re
import matplotlib
# matplotlib.use('Agg')  # Uncomment this line if running without a display (e.g., server)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# XGBoost is optional -- gracefully skip if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Skipping XGBoost model. Install with: pip install xgboost")

# =============================================================================
# Step 1: Load Data
# =============================================================================
# The data file has no header row; column names are in a separate file
df = pd.read_csv("census-bureau.data", header=None)

with open("census-bureau.columns", "r") as f:
    columns = f.read().splitlines()

df.columns = columns  # Assign the 42 column names to the dataframe

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn types:")
print(df.dtypes)
print("\nUnique label values:", df["label"].unique())

# =============================================================================
# Step 2: Preprocess
# =============================================================================

# --- 2a. Handle missing values ---
# 'hispanic origin' has ~874 null values (~0.4% of data).
# We fill with 'Unknown' rather than dropping rows or imputing with mode.
# Rationale: In banking risk, missing data often carries signal -- a person who
# didn't disclose this information may have a different risk profile. Creating
# an explicit 'Unknown' category lets the model learn if non-response is predictive.
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
df['hispanic origin'] = df['hispanic origin'].fillna('Unknown')

# --- 2b. Label encoding (risk-oriented framing) ---
# We map <$50K = 1 (positive class) and >=$50K = 0.
# This is a deliberate business decision: the model learns to detect RISK INDICATORS
# ("red flags") rather than success indicators ("green flags").
# - Positive coefficients in LR -> feature pushes toward lower income (risk)
# - Negative coefficients in LR -> feature pushes toward higher income (safe)
# This mirrors how banking risk models work: predict the adverse outcome (default,
# fraud, low income) so every signal points toward risk.
df["label"] = df["label"].map({'- 50000.': 1, '50000+.': 0})
print(f"\nNaN values in label after mapping: {df['label'].isna().sum()}")
print("\nLabel distribution:")
print(df['label'].value_counts())

# --- 2c. Separate features and target ---
# Drop both the label and the instance weight column.
# The weight represents sampling distribution, not a demographic feature.
X = df.drop(columns=['label', 'weight'])
y = df['label']

# --- 2d. One-hot encode categorical features ---
# pd.get_dummies converts all object-type (text) columns into binary 0/1 columns.
# drop_first=True removes one category per variable to avoid multicollinearity
# (e.g., if sex has Male/Female, we only need one column -- the other is implied).
# This expands the feature space from ~40 raw columns to several hundred.
# Note: Some variables like 'education' are ordinal and could use integer encoding,
# but one-hot is the safest approach -- it makes no ordering assumptions.
X_encoded = pd.get_dummies(X, drop_first=True)
print(f"\nOriginal features: {X.shape[1]}")
print(f"Encoded features: {X_encoded.shape[1]}")

# --- 2e. Train-test split ---
# test_size=0.2: Hold out 20% of data for evaluation (80% for training)
# random_state=42: Fixed seed for reproducibility -- same split every run
# stratify=y: Preserves the class ratio in both train and test sets.
#   Without this, the test set could have a different <$50K/>=$50K ratio,
#   making evaluation unreliable.
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# Model 1: Random Forest -- "Which features matter?"
# =============================================================================
# Purpose: Establish a strong baseline and identify the most important predictors.
# Random Forest builds many independent decision trees and aggregates their votes.
# It handles mixed data types well and provides feature importance rankings.
#
# Key parameters:
#   n_estimators=100: Build 100 trees. More trees = more stable predictions,
#     but diminishing returns beyond ~100. Each tree sees a random bootstrap
#     sample of the data and a random subset of features at each split.
#   max_depth=15: Maximum depth of each tree. Limits how "specific" each tree
#     can get. Without this cap, trees would memorize the training data (overfit).
#     15 is a moderate depth -- deep enough to capture patterns, shallow enough
#     to generalize.
#   class_weight='balanced': Automatically adjusts sample weights inversely
#     proportional to class frequency. Since <$50K is ~93% of data, the model
#     would otherwise ignore the >=$50K minority. 'balanced' forces equal
#     attention to both classes by penalizing minority-class errors more heavily.
#   random_state=42: Fixed seed for reproducibility.
print("\n" + "="*60)
print("MODEL 1: Random Forest")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=15,
    class_weight='balanced', random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# classification_report shows precision, recall, and F1 for EACH class separately.
# This is critical for imbalanced data -- overall accuracy can be misleading.
# - Precision: Of all predicted <$50K, how many actually are? (false positive rate)
# - Recall: Of all actual <$50K people, how many did we find? (false negative rate)
# - F1: Harmonic mean of precision and recall (balanced single metric)
print(classification_report(y_test, y_pred_rf))

# Feature importance: measures how much each feature reduces impurity (Gini)
# across all trees. Higher = more predictive. We plot the top 10 to see which
# variables drive income prediction (e.g., education, capital gains, age).
feat_importances = pd.Series(rf_model.feature_importances_, index=X_encoded.columns)
plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title("Top 10 Factors for Income <$50k (Random Forest)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150)
plt.show()
print("Saved: rf_feature_importance.png")

# =============================================================================
# Model 2: Logistic Regression -- "How does each feature affect the outcome?"
# =============================================================================
# Purpose: Understand the DIRECTION and MAGNITUDE of each feature's effect.
# RF tells us education matters, but LR tells us HOW -- e.g., "a bachelor's
# degree decreases the log-odds of being <$50K by 0.8" (a protective factor).
#
# Why we need scaling: LR treats all features as having equal scale. Without
# scaling, a feature ranging 0-99999 (capital gains) would dominate one
# ranging 0-1 (binary dummy). StandardScaler normalizes all features to
# mean=0, std=1 so coefficients are comparable.
print("\n" + "="*60)
print("MODEL 2: Logistic Regression")
print("="*60)

# fit_transform on training data: computes mean/std AND transforms
# transform on test data: uses the TRAINING mean/std (no data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Key parameters:
#   max_iter=1000: Maximum optimization iterations. Default (100) may not converge
#     with hundreds of features. 1000 ensures the optimizer fully converges.
#   class_weight='balanced': Same as RF -- forces equal attention to both classes.
# Training uses L-BFGS optimizer to minimize cross-entropy loss.
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)

# Extract coefficients: each feature gets a signed weight.
# Since our target is <$50K = 1:
#   Positive coefficient -> feature INCREASES probability of <$50K (risk factor / "red flag")
#   Negative coefficient -> feature DECREASES probability of <$50K (protective / "green flag")
importance_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coefficient': log_reg.coef_[0]
})

# Top risk factors: features most associated with earning <$50K
print("\nTop 5 Risk Factors (<$50k):")
print(importance_df.sort_values(by='Coefficient', ascending=False).head(5))

# Top safety factors: features most associated with earning >=$50K
print("\nTop 5 Safety Factors (>$50k):")
print(importance_df.sort_values(by='Coefficient', ascending=True).head(5))

# =============================================================================
# Model 3: XGBoost -- "What is the best prediction we can make?"
# =============================================================================
# Purpose: Maximize predictive accuracy using gradient boosting.
# Unlike RF (independent trees), XGBoost builds trees SEQUENTIALLY --
# each new tree specifically corrects the errors of the previous ensemble.
# This focused error correction typically yields the highest accuracy on
# tabular data.
if HAS_XGBOOST:
    print("\n" + "="*60)
    print("MODEL 3: XGBoost")
    print("="*60)

    # XGBoost doesn't accept special characters in column names ([, ], <)
    # which can appear in one-hot encoded column names. Replace with underscore.
    def clean_column_names(dataframe):
        new_cols = [re.sub(r'[\[\]<]', '_', col) for col in dataframe.columns]
        dataframe.columns = new_cols
        return dataframe

    # Use copies to avoid modifying the original train/test data
    X_train_xgb = clean_column_names(X_train.copy())
    X_test_xgb = clean_column_names(X_test.copy())

    # Key parameters:
    #   n_estimators=100: Number of boosting rounds (sequential trees).
    #   learning_rate=0.1: Shrinkage factor -- scales the contribution of each tree.
    #     Lower values (0.01-0.1) = more trees needed but better generalization.
    #     Higher values (0.3+) = faster training but risk overfitting.
    #     0.1 is a standard starting point.
    #   max_depth=5: Shallower than RF's 15 because boosting already focuses on
    #     hard cases -- deep trees in boosting tend to overfit. Depth 5 captures
    #     interactions up to 5 features deep.
    #   eval_metric='logloss': Logarithmic loss -- standard for binary classification.
    #     Measures how well predicted probabilities match actual labels.
    #   use_label_encoder=False: Suppresses deprecation warning in newer XGBoost.
    #   random_state=42: Reproducibility.
    xgb_model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    xgb_model.fit(X_train_xgb, y_train)
    y_pred_xgb = xgb_model.predict(X_test_xgb)

    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb))

    # Confusion matrix: 2x2 grid showing:
    #   Top-left (TN): Correctly predicted >=$50K
    #   Top-right (FP): Predicted <$50K but actually >=$50K (false alarm)
    #   Bottom-left (FN): Predicted >=$50K but actually <$50K (missed risk)
    #   Bottom-right (TP): Correctly predicted <$50K
    # For the client: FN (bottom-left) is the most costly -- missing a low-income
    # person means targeting them with wrong products.
    cm = confusion_matrix(y_test, y_pred_xgb)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Safe (>=50k)', 'Risk (<50k)'],
                yticklabels=['Safe (>=50k)', 'Risk (<$50k)'])
    plt.title('Income Classification (XGBoost)', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig("xgb_confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: xgb_confusion_matrix.png")

print("\nClassification complete.")