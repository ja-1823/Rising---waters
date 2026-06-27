# ============================================================
# FLOOD PREDICTION — EARLY WARNING SYSTEM
# Epic 3 : Data Pre-processing
# Step 5  : Feature Scaling
# Team    : Pedalanka Jahnavi (Lead), Nohith, Orsu, Bahujanku, Maddu
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv('flood_data.csv')

print("=" * 55)
print("  STEP 5 — FEATURE SCALING")
print("=" * 55)
print(f"\n📋 Dataset Shape : {df.shape}")

# ── 1. Define features and target ────────────────────────────
feature_cols = ['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'YEAR']

X = df[feature_cols]
y = df['FLOODS']

# ── 2. Train/Test split (must scale AFTER split to avoid leakage) ─
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Train size : {X_train.shape}  |  Test size : {X_test.shape}")
print("\n📌 Rule: Fit scaler ONLY on X_train → transform X_train & X_test")
print("         (prevents data leakage from test set into training)\n")

# ── 3. Before scaling — statistics ───────────────────────────
print("📊 Feature Statistics BEFORE Scaling:")
print("-" * 50)
print(X_train.describe().loc[['mean', 'std', 'min', 'max']].round(2).to_string())

# ── 4. Standard Scaler (Z-score normalisation) ───────────────
#   Best for: algorithms sensitive to feature scale (KNN, SVM, Logistic Regression)
#   Formula : z = (x - mean) / std  →  mean=0, std=1
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std  = scaler_std.transform(X_test)

X_train_std_df = pd.DataFrame(X_train_std, columns=feature_cols)

print("\n📊 Feature Statistics AFTER Standard Scaling (Z-score):")
print("-" * 50)
print(X_train_std_df.describe().loc[['mean', 'std', 'min', 'max']].round(4).to_string())

# ── 5. MinMax Scaler (normalisation to [0, 1]) ───────────────
#   Best for: Neural networks, image data
#   Formula : x_norm = (x - x_min) / (x_max - x_min)
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm  = scaler_mm.transform(X_test)

X_train_mm_df = pd.DataFrame(X_train_mm, columns=feature_cols)

print("\n📊 Feature Statistics AFTER MinMax Scaling ([0, 1]):")
print("-" * 50)
print(X_train_mm_df.describe().loc[['mean', 'std', 'min', 'max']].round(4).to_string())

# ── 6. Robust Scaler (IQR-based) ─────────────────────────────
#   Best for: datasets with outliers
#   Formula : x_robust = (x - median) / IQR
scaler_rb = RobustScaler()
X_train_rb = scaler_rb.fit_transform(X_train)
X_test_rb  = scaler_rb.transform(X_test)

X_train_rb_df = pd.DataFrame(X_train_rb, columns=feature_cols)

print("\n📊 Feature Statistics AFTER Robust Scaling (IQR-based):")
print("-" * 50)
print(X_train_rb_df.describe().loc[['mean', 'std', 'min', 'max']].round(4).to_string())

# ── 7. Visualise — Before vs After Standard Scaling ──────────
fig, axes = plt.subplots(2, len(feature_cols), figsize=(18, 7))

for i, col in enumerate(feature_cols):
    # Before
    axes[0][i].hist(X_train[col], bins=35, color='#5BA8E0',
                    edgecolor='white', alpha=0.85)
    axes[0][i].set_title(col, fontweight='bold', fontsize=9)
    axes[0][i].set_ylabel('Count' if i == 0 else '')
    if i == 0:
        axes[0][i].set_xlabel('Original Scale')

    # After Standard Scaling
    axes[1][i].hist(X_train_std_df[col], bins=35, color='#FFA500',
                    edgecolor='white', alpha=0.85)
    axes[1][i].set_title(col, fontweight='bold', fontsize=9)
    axes[1][i].set_ylabel('Count' if i == 0 else '')
    if i == 0:
        axes[1][i].set_xlabel('After Standard Scaling')

axes[0][0].set_ylabel('Before Scaling\nCount')
axes[1][0].set_ylabel('After Standard Scaling\nCount')

plt.suptitle('Feature Distributions — Before vs After Standard Scaling',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_scaling_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
print("\n📊 Chart saved : feature_scaling_comparison.png")

# ── 8. Scaling Strategy Summary ──────────────────────────────
print("\n📌 Scaling Strategy for this Project:")
print("-" * 50)
print("  StandardScaler  ─ used for KNN (distance-based)")
print("  No scaling      ─ Decision Tree, Random Forest, XGBoost")
print("                    (tree models are scale-invariant)")
print("  ➜ StandardScaler selected as default for this project")

# ── 9. Save the StandardScaler (used in Flask deployment) ────
joblib.dump(scaler_std, 'scaler.pkl')
print("\n💾 Saved : scaler.pkl  (StandardScaler — for Flask deployment)")

# Save scaled arrays
pd.DataFrame(X_train_std, columns=feature_cols).to_csv('X_train_scaled.csv', index=False)
pd.DataFrame(X_test_std,  columns=feature_cols).to_csv('X_test_scaled.csv',  index=False)
print("💾 Saved : X_train_scaled.csv  |  X_test_scaled.csv")
print(f"\n✅ Feature scaling complete. Ready for model training.")
print("=" * 55)
