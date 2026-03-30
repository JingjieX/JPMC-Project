"""
Supplementary Analysis: Additional insights for the project report.

1. ROC curves and AUC scores for all classifiers
2. Logistic Regression classification report
3. Education level vs income breakdown
4. Workforce-only sub-segmentation (filter out children and retirees)
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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_curve, auc,
                             roc_auc_score, silhouette_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# =============================================================================
# Load and preprocess (same as classification.py)
# =============================================================================
df = pd.read_csv("census-bureau.data", header=None)
with open("census-bureau.columns", "r") as f:
    columns = f.read().splitlines()
df.columns = columns

df['hispanic origin'] = df['hispanic origin'].fillna('Unknown')
df["label"] = df["label"].map({'- 50000.': 1, '50000+.': 0})

X = df.drop(columns=['label', 'weight'])
y = df['label']
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 1. Train all models and get ROC curves
# =============================================================================
print("="*60)
print("TRAINING MODELS FOR ROC/AUC ANALYSIS")
print("="*60)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                           class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]  # Probability of class 1 (<$50K)

# Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = lr.predict(X_test_scaled)

# XGBoost
if HAS_XGBOOST:
    def clean_cols(dataframe):
        dataframe.columns = [re.sub(r'[\[\]<]', '_', c) for c in dataframe.columns]
        return dataframe
    X_train_xgb = clean_cols(X_train.copy())
    X_test_xgb = clean_cols(X_test.copy())
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                        random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_xgb, y_train)
    xgb_proba = xgb.predict_proba(X_test_xgb)[:, 1]

# --- 1a. LR Classification Report ---
print("\n" + "="*60)
print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred_lr))
print(f"LR AUC: {roc_auc_score(y_test, lr_proba):.4f}")

# --- 1b. ROC Curves ---
plt.figure(figsize=(8, 6))

# RF ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
rf_auc = auc(rf_fpr, rf_tpr)
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})', linewidth=2)

# LR ROC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
lr_auc = auc(lr_fpr, lr_tpr)
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.3f})', linewidth=2)

# XGBoost ROC
if HAS_XGBOOST:
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
    xgb_auc = auc(xgb_fpr, xgb_tpr)
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves: Model Comparison', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()
print(f"\nAUC Scores: RF={rf_auc:.3f}, LR={lr_auc:.3f}", end="")
if HAS_XGBOOST:
    print(f", XGBoost={xgb_auc:.3f}")
else:
    print()
print("Saved: roc_curves.png")

# =============================================================================
# 2. Education vs Income Breakdown
# =============================================================================
print("\n" + "="*60)
print("EDUCATION VS INCOME BREAKDOWN")
print("="*60)

# Reload original data for this analysis
df_orig = pd.read_csv("census-bureau.data", header=None)
df_orig.columns = columns
df_orig['hispanic origin'] = df_orig['hispanic origin'].fillna('Unknown')

# Strip whitespace from education and label
df_orig['education'] = df_orig['education'].str.strip()
df_orig['label'] = df_orig['label'].str.strip()

# Calculate % earning >=$50K by education level
edu_income = df_orig.groupby('education')['label'].apply(
    lambda x: (x == '50000+.').sum() / len(x) * 100
).sort_values(ascending=True)

print("\n% Earning >=$50K by Education Level:")
for edu, pct in edu_income.items():
    print(f"  {edu}: {pct:.1f}%")

# Plot
plt.figure(figsize=(10, 7))
edu_income.plot(kind='barh', color='teal')
plt.xlabel('% Earning >=$50K', fontsize=12)
plt.ylabel('Education Level', fontsize=12)
plt.title('Income Distribution by Education Level', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("education_vs_income.png", dpi=150)
plt.show()
print("Saved: education_vs_income.png")

# =============================================================================
# 3. Workforce-Only Sub-Segmentation
# =============================================================================
print("\n" + "="*60)
print("WORKFORCE-ONLY SUB-SEGMENTATION")
print("="*60)

# Filter to working-age adults only (age >= 18, weeks worked > 0)
df_work = df_orig[
    (df_orig['age'] >= 18) &
    (df_orig['weeks worked in year'] > 0)
].copy()
print(f"Workforce subset: {len(df_work)} records (from {len(df_orig)} total)")
print(f"% earning >=$50K in workforce: {(df_work['label'] == '50000+.').mean()*100:.1f}%")

# Prepare features
label_col = 'label'
weight_candidates = [c for c in df_work.columns if 'weight' in c.lower()]
weight_col = weight_candidates[0] if weight_candidates else None
drop_cols = [label_col]
if weight_col:
    drop_cols.append(weight_col)

X_work = df_work.drop(columns=drop_cols)
numeric_cols = X_work.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_work.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    X_work[col] = X_work[col].astype(str).str.strip()

X_work_encoded = pd.get_dummies(X_work, columns=categorical_cols, drop_first=True)
scaler_work = StandardScaler()
X_work_scaled = scaler_work.fit_transform(X_work_encoded)

# PCA
pca_work = PCA(random_state=42)
pca_work.fit(X_work_scaled)
cumvar = np.cumsum(pca_work.explained_variance_ratio_)
n_comp = np.argmax(cumvar >= 0.90) + 1
print(f"PCA: {n_comp} components for 90% variance")

pca_work_final = PCA(n_components=n_comp, random_state=42)
X_work_pca = pca_work_final.fit_transform(X_work_scaled)

# Find optimal k for workforce
MAX_SAMPLES = 30000
if X_work_pca.shape[0] > MAX_SAMPLES:
    np.random.seed(42)
    sample_idx = np.random.choice(X_work_pca.shape[0], MAX_SAMPLES, replace=False)
    X_sample = X_work_pca[sample_idx]
else:
    X_sample = X_work_pca

print("\nCluster evaluation (workforce only):")
K_range = range(2, 8)
sil_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_sample)
    sil = silhouette_score(X_sample, km.labels_, sample_size=min(5000, len(X_sample)))
    sil_scores.append(sil)
    print(f"  k={k}: Silhouette={sil:.4f}")

# Use k=4 for workforce (more actionable sub-segments)
work_k = 4
print(f"\nSelected k={work_k} for workforce sub-segmentation")

kmeans_work = KMeans(n_clusters=work_k, random_state=42, n_init=10, max_iter=300)
work_labels = kmeans_work.fit_predict(X_work_pca)

df_work_seg = df_work.copy()
df_work_seg['Segment'] = work_labels

# Profile workforce segments
print("\nWorkforce Segment Profiles:")
df_work_seg['income_binary'] = (df_work_seg['label'] == '50000+.').astype(int)

profile_cols = ['age', 'weeks worked in year', 'capital gains', 'capital losses',
                'dividends from stocks', 'wage per hour', 'num persons worked for employer']
available_cols = [c for c in profile_cols if c in df_work_seg.columns]

seg_profile = df_work_seg.groupby('Segment')[available_cols].mean().round(1)
print(seg_profile.T.to_string())

income_dist = df_work_seg.groupby('Segment')['income_binary'].agg(['mean', 'count'])
income_dist.columns = ['pct_high_income', 'count']
income_dist['pct_high_income'] = (income_dist['pct_high_income'] * 100).round(1)
print("\nIncome Distribution:")
print(income_dist)

# Key categoricals
for col in ['education', 'major occupation code', 'major industry code', 'sex', 'marital stat']:
    if col in df_work_seg.columns:
        modes = df_work_seg.groupby('Segment')[col].agg(
            lambda x: x.str.strip().value_counts().index[0] if len(x) > 0 else 'N/A'
        )
        print(f"\n  {col}:")
        for seg, val in modes.items():
            print(f"    Segment {seg}: {val}")

# Visualize workforce segments
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Income by segment
income_ct = pd.crosstab(df_work_seg['Segment'], df_work_seg['label'].str.strip(), normalize='index')
income_ct.plot(kind='bar', stacked=True, ax=axes[0, 0], color=['#e74c3c', '#2ecc71'])
axes[0, 0].set_title('Income Distribution by Workforce Segment')
axes[0, 0].set_ylabel('Proportion')
axes[0, 0].set_xlabel('Segment')
axes[0, 0].tick_params(axis='x', rotation=0)

# Age by segment
df_work_seg.boxplot(column='age', by='Segment', ax=axes[0, 1])
axes[0, 1].set_title('Age by Workforce Segment')
axes[0, 1].set_xlabel('Segment')
axes[0, 1].set_ylabel('Age')
plt.sca(axes[0, 1])
plt.title('Age by Workforce Segment')

# Weeks worked by segment
df_work_seg.boxplot(column='weeks worked in year', by='Segment', ax=axes[1, 0])
axes[1, 0].set_title('Weeks Worked by Workforce Segment')
axes[1, 0].set_xlabel('Segment')
axes[1, 0].set_ylabel('Weeks')
plt.sca(axes[1, 0])
plt.title('Weeks Worked by Workforce Segment')

# Capital gains by segment
cap_gains = df_work_seg.groupby('Segment')['capital gains'].mean()
cap_gains.plot(kind='bar', ax=axes[1, 1], color='teal')
axes[1, 1].set_title('Avg Capital Gains by Workforce Segment')
axes[1, 1].set_xlabel('Segment')
axes[1, 1].set_ylabel('Capital Gains ($)')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.suptitle('Workforce Sub-Segmentation Dashboard', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("workforce_segments.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: workforce_segments.png")

# 2D visualization
pca_2d = PCA(n_components=2, random_state=42)
X_work_2d = pca_2d.fit_transform(X_work_scaled)

colors = plt.cm.Set2(np.linspace(0, 1, work_k))
plt.figure(figsize=(10, 7))
for seg in range(work_k):
    mask = work_labels == seg
    n_plot = min(mask.sum(), 3000)
    idx = np.where(mask)[0]
    np.random.seed(42)
    plot_idx = np.random.choice(idx, n_plot, replace=False)
    plt.scatter(X_work_2d[plot_idx, 0], X_work_2d[plot_idx, 1],
                c=[colors[seg]], label=f'Segment {seg}', alpha=0.3, s=10)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)')
plt.title('Workforce Segments (PCA 2D)')
plt.legend(markerscale=3, fontsize=10)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("workforce_segments_2d.png", dpi=150)
plt.show()
print("Saved: workforce_segments_2d.png")

print("\nSupplementary analysis complete.")
