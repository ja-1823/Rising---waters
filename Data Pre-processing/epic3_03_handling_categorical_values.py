# ============================================================
# FLOOD PREDICTION — EARLY WARNING SYSTEM
# Epic 3 : Data Pre-processing
# Step 3  : Handling Categorical Values
# Team    : Pedalanka Jahnavi (Lead), Nohith, Orsu, Bahujanku, Maddu
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv('flood_data.csv')

print("=" * 55)
print("  STEP 3 — HANDLING CATEGORICAL VALUES")
print("=" * 55)
print(f"\n📋 Dataset Shape : {df.shape}")

# ── 1. Identify column types ─────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\n🔢 Numerical columns  : {len(num_cols)}")
print(f"🔤 Categorical columns: {len(cat_cols)}")

if cat_cols:
    print("\nCategorical columns found:")
    for col in cat_cols:
        print(f"  {col} — unique values: {df[col].nunique()}")
        print(f"  {df[col].value_counts().head(5).to_string()}\n")
else:
    print("\n  ✅ No object/string categorical columns in this dataset.")

# ── 2. Encode Target variable (FLOODS) ───────────────────────
#   FLOODS is already binary (0/1). We verify and document it.
print("\n🎯 Target Variable — FLOODS:")
print("-" * 35)
print(f"  dtype   : {df['FLOODS'].dtype}")
print(f"  values  : {sorted(df['FLOODS'].unique())}")
print(f"  0 → No Flood  |  1 → Flood  ✅ Already encoded")

# ── 3. Simulate a categorical column for demonstration ────────
#   Many real flood datasets include a 'Season' or 'Region' column
df_demo = df.copy()

# Add a 'Season' feature based on Jun-Sep values
df_demo['Season'] = pd.cut(
    df_demo['Jun-Sep'],
    bins=[0, 400, 700, 1000, np.inf],
    labels=['Dry', 'Normal', 'Wet', 'Very Wet']
)

print("\n📌 Simulated categorical column — 'Season':")
print(df_demo['Season'].value_counts().to_string())

# ── 4. Label Encoding ─────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

df_label = df_demo.copy()
le = LabelEncoder()
df_label['Season_LabelEncoded'] = le.fit_transform(df_label['Season'].astype(str))

print("\n🔧 Label Encoding:")
for original, encoded in zip(le.classes_, le.transform(le.classes_)):
    print(f"  '{original}' → {encoded}")

# ── 5. One-Hot Encoding ──────────────────────────────────────
df_ohe = df_demo.copy()
df_ohe = pd.get_dummies(df_ohe, columns=['Season'], prefix='Season', drop_first=False)

ohe_cols = [c for c in df_ohe.columns if c.startswith('Season_')]
print(f"\n🔧 One-Hot Encoding — new columns created: {ohe_cols}")
print(df_ohe[ohe_cols].head(5).to_string())

# ── 6. Which encoding to use? ────────────────────────────────
print("\n📌 Encoding Strategy Summary:")
print("-" * 45)
print("  Label Encoding : suitable for ordinal data (Dry < Normal < Wet)")
print("  One-Hot Encoding: suitable for nominal data (no natural order)")
print("  ➜ For 'Season' (ordinal) — Label Encoding preferred here.")
print("  ➜ For 'Region' (nominal) — One-Hot Encoding preferred.")

# ── 7. Final dataset — use Label Encoded Season ──────────────
df_final = df_label[num_cols + ['Season_LabelEncoded']].copy()
df_final.to_csv('flood_data_step3_categorical_handled.csv', index=False)

print(f"\n💾 Saved : flood_data_step3_categorical_handled.csv  (shape: {df_final.shape})")
print("=" * 55)
