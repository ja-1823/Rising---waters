# ============================================================
# Step 4: Multivariate Analysis
# ============================================================

# ── 4.1  Correlation Heatmap ─────────────────────
features_for_corr = ['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec','ANNUAL','FLOODS']
corr_matrix = df[features_for_corr].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.zeros_like(corr_matrix, dtype=bool)
# no mask — show full matrix

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues',
            linewidths=0.5, linecolor='white',
            ax=ax, annot_kws={'size': 11, 'weight': 'bold'},
            vmin=-0.2, vmax=1.0, cbar_kws={'shrink': 0.8})

ax.set_title('Correlation Matrix — Seasonal Features vs Flood', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

print("\n📌 Top correlations with FLOODS:")
corr_flood = corr_matrix['FLOODS'].drop('FLOODS').sort_values(ascending=False)
for feat, val in corr_flood.items():
    bar = '█' * int(abs(val) * 30)
    print(f"  {feat:<12} r = {val:+.3f}  {bar}")


# ── 4.2  Scatter Plots — key features vs ANNUAL ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pairs = [('Jun-Sep', 'ANNUAL'), ('Mar-May', 'ANNUAL'), ('Oct-Dec', 'ANNUAL')]

for ax, (x_feat, y_feat) in zip(axes, pairs):
    scatter = ax.scatter(df[x_feat], df[y_feat],
                         c=df['FLOODS'], cmap='coolwarm', alpha=0.4, s=10)
    ax.set_xlabel(x_feat); ax.set_ylabel(y_feat)
    ax.set_title(f'{x_feat} vs {y_feat}', fontweight='bold')

from matplotlib.lines import Line2D
legend_el = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#3274A1', markersize=8, label='No Flood'),
             Line2D([0],[0], marker='o', color='w', markerfacecolor='#E1513D', markersize=8, label='Flood')]
axes[2].legend(handles=legend_el, loc='upper left', fontsize=9)

plt.suptitle('Scatter Plots Coloured by Flood Label', fontweight='bold', fontsize=12, y=1.01)
plt.tight_layout()
plt.show()


# ── 4.3  Seasonal rainfall by flood label ─────────
seasonal_feats = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, feat in zip(axes, seasonal_feats):
    sns.violinplot(data=df, x='Flood_Label', y=feat, ax=ax,
                   palette={'No Flood': '#5BA8E0', 'Flood': '#FF6347'},
                   inner='box', linewidth=0.8)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('')

plt.suptitle('Seasonal Rainfall Distribution by Flood Class', fontweight='bold', fontsize=12, y=1.01)
plt.tight_layout()
plt.show()


# ── 4.4  Feature Importance (Random Forest preview) ─
_X = df[['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec','ANNUAL','YEAR']]
_y = df['FLOODS']
_rf = RandomForestClassifier(n_estimators=100, random_state=42)
_rf.fit(_X, _y)

importances = pd.Series(_rf.feature_importances_, index=_X.columns).sort_values()
colors_imp = ['#1E90FF' if i == importances.index[-1] else '#5BA8E0' for i in importances.index]

fig, ax = plt.subplots(figsize=(8, 4))
importances.plot.barh(ax=ax, color=colors_imp, edgecolor='white')
ax.set_title('Feature Importance — Random Forest (preview)', fontweight='bold')
ax.set_xlabel('Importance Score')
for patch, val in zip(ax.patches, importances):
    ax.text(val + 0.002, patch.get_y() + patch.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.show()

