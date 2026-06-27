# ============================================================
# FLOOD PREDICTION — EARLY WARNING SYSTEM
# Epic 3 : Data Pre-processing
# Step 4  : Splitting Data into Training and Test Sets
# Team    : Pedalanka Jahnavi (Lead), Nohith, Orsu, Bahujanku, Maddu
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv('flood_data.csv')

print("=" * 55)
print("  STEP 4 — SPLITTING DATA INTO TRAIN & TEST SETS")
print("=" * 55)
print(f"\n📋 Dataset Shape : {df.shape}")

# ── 1. Define Feature matrix (X) and Target (y) ──────────────
feature_cols = ['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'YEAR']

X = df[feature_cols]
y = df['FLOODS']

print(f"\n✅ Feature matrix X : {X.shape}  →  {list(X.columns)}")
print(f"✅ Target vector  y : {y.shape}  →  values: {sorted(y.unique())}")

# ── 2. Check class distribution before split ─────────────────
print("\n📊 Class distribution (full dataset):")
vc_full = y.value_counts()
print(f"  No Flood (0) : {vc_full[0]}  ({vc_full[0]/len(y)*100:.1f}%)")
print(f"  Flood    (1) : {vc_full[1]}  ({vc_full[1]/len(y)*100:.1f}%)")

# ── 3. Standard 80/20 Split (stratified) ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # preserves class ratio in both splits
)

print("\n" + "─" * 45)
print("  80/20 Stratified Split")
print("─" * 45)
print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
print(f"  X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")

# Verify class ratio preserved
vc_train = y_train.value_counts()
vc_test  = y_test.value_counts()
print(f"\n  Train — No Flood: {vc_train[0]}  ({vc_train[0]/len(y_train)*100:.1f}%)  |  Flood: {vc_train[1]}  ({vc_train[1]/len(y_train)*100:.1f}%)")
print(f"  Test  — No Flood: {vc_test[0]}   ({vc_test[0]/len(y_test)*100:.1f}%)  |  Flood: {vc_test[1]}   ({vc_test[1]/len(y_test)*100:.1f}%)")
print("  ✅ Stratified split preserves class ratio correctly")

# ── 4. Alternative: 70/30 Split ──────────────────────────────
X_train70, X_test30, y_train70, y_test30 = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n  70/30 Split — Train: {X_train70.shape}  |  Test: {X_test30.shape}")

# ── 5. Visualise split sizes ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Split sizes
splits = ['Training Set\n(80%)', 'Test Set\n(20%)']
sizes  = [len(X_train), len(X_test)]
colors = ['#1E90FF', '#FF6347']
axes[0].bar(splits, sizes, color=colors, edgecolor='white', linewidth=0.8, width=0.4)
axes[0].set_title('Train / Test Split (80/20)', fontweight='bold')
axes[0].set_ylabel('Number of Samples')
for i, v in enumerate(sizes):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=11)

# Class distribution side by side
x = np.arange(2)
width = 0.3
labels = ['No Flood (0)', 'Flood (1)']
train_counts = [vc_train[0], vc_train[1]]
test_counts  = [vc_test[0],  vc_test[1]]

axes[1].bar(x - width/2, train_counts, width, label='Train', color='#1E90FF', edgecolor='white')
axes[1].bar(x + width/2, test_counts,  width, label='Test',  color='#FF6347', edgecolor='white')
axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
axes[1].set_title('Class Distribution — Train vs Test', fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.suptitle('Data Split Visualisation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('split_visualisation.png', dpi=100, bbox_inches='tight')
plt.show()
print("\n📊 Chart saved : split_visualisation.png")

# ── 6. Save split datasets ────────────────────────────────────
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv',   index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv',   index=False)

print("\n💾 Saved split files:")
print("   X_train.csv  X_test.csv  y_train.csv  y_test.csv")
print(f"\n✅ Final split ready for model training.")
print("=" * 55)
