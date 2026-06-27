# ============================================================
# FLOOD PREDICTION — EARLY WARNING SYSTEM
# Epic 3 : Data Pre-processing
# Step 1  : Handling Missing Values
# Team    : Pedalanka Jahnavi (Lead), Nohith, Orsu, Bahujanku, Maddu
# ============================================================

import numpy as np
import pandas as pd

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv('flood_data.csv')

print("=" * 55)
print("  STEP 1 — HANDLING MISSING VALUES")
print("=" * 55)

# ── 1. Initial shape ─────────────────────────────────────────
print(f"\n📋 Dataset Shape (before) : {df.shape}")

# ── 2. Check missing values per column ───────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct.round(2)
})

print("\n🔍 Missing Value Report:")
print("-" * 40)
print(missing_report[missing_report['Missing Count'] > 0].to_string()
      if missing_report['Missing Count'].sum() > 0
      else "  ✅ No missing values found in any column.")

# ── 3. Simulate missing values for demonstration ─────────────
#   (In a real dataset, this block is skipped)
np.random.seed(42)
df_demo = df.copy()

# Randomly introduce NaN in 3 columns (2% each)
for col in ['JAN', 'JUN', 'ANNUAL']:
    idx = np.random.choice(df_demo.index, size=int(0.02 * len(df_demo)), replace=False)
    df_demo.loc[idx, col] = np.nan

print("\n📌 Simulated missing values introduced for demonstration:")
demo_missing = df_demo.isnull().sum()
print(demo_missing[demo_missing > 0].to_string())

# ── 4. Strategy: Fill numerical columns with median ──────────
#   Median is preferred over mean — robust to outliers
numerical_cols = df_demo.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical_cols:
    if df_demo[col].isnull().sum() > 0:
        median_val = df_demo[col].median()
        df_demo[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' missing values with median = {median_val:.2f}")

# ── 5. Verify — no missing values remain ─────────────────────
print("\n✅ After imputation:")
remaining = df_demo.isnull().sum().sum()
print(f"  Total missing values remaining : {remaining}")

# ── 6. Save cleaned dataset ───────────────────────────────────
df_demo.to_csv('flood_data_step1_no_missing.csv', index=False)
print(f"\n💾 Saved : flood_data_step1_no_missing.csv  (shape: {df_demo.shape})")
print("=" * 55)
