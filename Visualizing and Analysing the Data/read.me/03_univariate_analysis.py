# ============================================================
# Step 3: Univariate Analysis
# ============================================================

# ── 3.1  Monthly rainfall bar chart ──────────────
months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
means  = df[months].mean()

colors = ['#1E90FF' if m in ('JUN','JUL','AUG','SEP') else '#5BA8E0' for m in months]

fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(months, means, color=colors, edgecolor='white', linewidth=0.6)
ax.set_title('Average Monthly Rainfall (mm) — Monsoon months highlighted', fontweight='bold')
ax.set_ylabel('Rainfall (mm)')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f mm'))

for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8.5, color='#444')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#1E90FF', label='Monsoon (Jun–Sep)'),
                   Patch(facecolor='#5BA8E0', label='Other months')]
ax.legend(handles=legend_elements, fontsize=9)
plt.tight_layout()
plt.show()
print(f"  Peak month : {means.idxmax()} ({means.max():.1f} mm)")
print(f"  Low  month : {means.idxmin()} ({means.min():.1f} mm)")


# ── 3.2  Distribution of Annual Rainfall ─────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(df['ANNUAL'], bins=50, color='#1E90FF', edgecolor='white', alpha=0.85)
axes[0].axvline(df['ANNUAL'].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {df["ANNUAL"].mean():.0f}')
axes[0].axvline(df['ANNUAL'].median(), color='orange', linestyle='--', linewidth=1.5, label=f'Median: {df["ANNUAL"].median():.0f}')
axes[0].set_title('Distribution of Annual Rainfall', fontweight='bold')
axes[0].set_xlabel('Annual Rainfall (mm)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Box plot by FLOODS
flood_labels = {0: 'No Flood', 1: 'Flood'}
df['Flood_Label'] = df['FLOODS'].map(flood_labels)
sns.boxplot(data=df, x='Flood_Label', y='ANNUAL', ax=axes[1],
            palette={'No Flood': '#5BA8E0', 'Flood': '#FF6347'})
axes[1].set_title('Annual Rainfall by Flood Class', fontweight='bold')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Annual Rainfall (mm)')

plt.tight_layout()
plt.show()

skew = df['ANNUAL'].skew()
print(f"  Skewness of ANNUAL : {skew:.4f}  ({'right-skewed ➜ few very wet years' if skew > 0.5 else 'approx. symmetric'})")


# ── 3.3  Target class balance ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

counts = df['FLOODS'].value_counts()
labels = ['No Flood (0)', 'Flood (1)']
colors_pie = ['#5BA8E0', '#FF6347']

axes[0].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors_pie,
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[0].set_title('Target Class Distribution', fontweight='bold')

axes[1].bar(labels, counts, color=colors_pie, edgecolor='white', linewidth=0.8)
axes[1].set_title('Flood vs No-Flood Count', fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(counts):
    axes[1].text(i, v + 10, str(v), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
print(f"  Class imbalance ratio : 1 : {counts[1]/counts[0]:.1f}")


# ── 3.4  Distribution of all seasonal features ───
seasonal = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'ANNUAL']
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for ax, col in zip(axes, seasonal):
    ax.hist(df[col], bins=40, color='#1E90FF', edgecolor='white', alpha=0.8)
    ax.set_title(col, fontweight='bold', fontsize=10)
    ax.set_xlabel('mm')
    ax.set_ylabel('Count')
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.2)

plt.suptitle('Seasonal Rainfall Distributions (red dashed = mean)', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

