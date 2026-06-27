# ============================================================
# Step 5: Descriptive Analysis
# ============================================================

# ── 5.1  Full statistical summary ────────────────
desc = df.drop(columns=['FLOODS','Flood_Label']).describe().T
desc['skewness'] = df.drop(columns=['FLOODS','Flood_Label']).skew()
desc['kurtosis'] = df.drop(columns=['FLOODS','Flood_Label']).kurt()
desc = desc.round(2)

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 120)
print("📊 Descriptive Statistics:")
print("=" * 95)
print(desc.to_string())


# ── 5.2  Group statistics by flood label ─────────
group_stats = df.groupby('Flood_Label')[['ANNUAL','Jun-Sep','Mar-May','Jan-Feb','Oct-Dec']].agg(['mean','std']).round(1)
print("\n📊 Group Statistics (Flood vs No Flood):")
print("=" * 70)
print(group_stats.to_string())


# ── 5.3  Box plots — descriptive spread ──────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

feats_box = ['JAN','APR','JUL','Oct-Dec','Mar-May','ANNUAL']
for ax, feat in zip(axes, feats_box):
    bp = ax.boxplot([df[df['FLOODS']==0][feat], df[df['FLOODS']==1][feat]],
                    labels=['No Flood','Flood'], patch_artist=True,
                    medianprops={'color':'red','linewidth':2})
    bp['boxes'][0].set_facecolor('#5BA8E0')
    bp['boxes'][1].set_facecolor('#FF6347')
    ax.set_title(f'{feat}', fontweight='bold')
    ax.set_ylabel('mm')

plt.suptitle('Box Plots: Rainfall Spread by Flood Class', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ── 5.4  Year-wise flood frequency ───────────────
flood_by_year = df.groupby('YEAR')['FLOODS'].sum().reset_index()
flood_by_year.columns = ['YEAR', 'FloodCount']

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(flood_by_year['YEAR'], flood_by_year['FloodCount'],
                alpha=0.35, color='#1E90FF')
ax.plot(flood_by_year['YEAR'], flood_by_year['FloodCount'],
        color='#1E90FF', linewidth=1.5)
ax.set_title('Year-wise Flood Frequency (Smoothed Trend)', fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('Flood Occurrences')
ax.axhline(flood_by_year['FloodCount'].mean(), color='red', linestyle='--',
           linewidth=1.2, label=f"Mean: {flood_by_year['FloodCount'].mean():.1f}")
ax.legend()
plt.tight_layout()
plt.show()

