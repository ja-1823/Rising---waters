# ============================================================
# FLOOD PREDICTION — EARLY WARNING SYSTEM
# Epic 3 : Data Pre-processing
# Step 2  : Handling Outliers
# Team    : Pedalanka Jahnavi (Lead), Nohith, Orsu, Bahujanku, Maddu
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid')

# ── Load Dataset ─────────────────────────────────────────────
df = pd.read_csv('flood_data.csv')

print("=" * 55)
print("  STEP 2 — HANDLING OUTLIERS")
print("=" * 55)
print(f"\n📋 Dataset Shape (before) : {df.shape}")

# ── Target columns to check for outliers ─────────────────────
target_cols = ['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec',
               'JAN', 'JUL', 'AUG']

# ── 1. Detect outliers using IQR method ──────────────────────
print("\n🔍 Outlier Detection — IQR Method:")
print("-" * 50)

outlier_report = []
for col in target_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report.append({
        'Column'   : col,
        'Q1'       : round(Q1, 2),
        'Q3'       : round(Q3, 2),
        'IQR'      : round(IQR, 2),
        'Lower Bound': round(lower, 2),
        'Upper Bound': round(upper, 2),
        'Outliers' : n_outliers,
        'Outlier %': round(n_outliers / len(df) * 100, 2)
    })

report_df = pd.DataFrame(outlier_report)
print(report_df.to_string(index=False))

# ── 2. Visualise outliers — Box plots ────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for ax, col in zip(axes, target_cols):
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor='#5BA8E0', color='#1E4070'),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', color='#FF6347', markersize=4, alpha=0.5))
    ax.set_title(col, fontweight='bold', fontsize=10)
    ax.set_ylabel('mm')

plt.suptitle('Box Plots — Outlier Detection per Feature', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outlier_boxplots.png', dpi=100, bbox_inches='tight')
plt.show()
print("\n📊 Box plot saved : outlier_boxplots.png")

# ── 3. Handle outliers — Capping (Winsorisation) ─────────────
#   Cap values at 1.5×IQR bounds instead of dropping rows
#   (preserves dataset size; appropriate for rainfall data)
df_clean = df.copy()

print("\n🔧 Applying Winsorisation (IQR Capping):")
print("-" * 45)
for col in target_cols:
    Q1  = df_clean[col].quantile(0.25)
    Q3  = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    print(f"  {col:<12} : {before:>3} outliers capped  "
          f"[{lower:.1f}, {upper:.1f}]")

# ── 4. Verify — Z-score check post-capping ───────────────────
print("\n✅ Z-score check after capping (values > 3σ):")
for col in target_cols:
    z = ((df_clean[col] - df_clean[col].mean()) / df_clean[col].std()).abs()
    extreme = (z > 3).sum()
    print(f"  {col:<12} : {extreme} extreme values remaining")

# ── 5. Save ──────────────────────────────────────────────────
df_clean.to_csv('flood_data_step2_no_outliers.csv', index=False)
print(f"\n💾 Saved : flood_data_step2_no_outliers.csv  (shape: {df_clean.shape})")
print("=" * 55)
