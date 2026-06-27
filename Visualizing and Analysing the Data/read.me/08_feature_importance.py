# ============================================================
# Step 8: XGBoost Feature Importance
# ============================================================

xgb_model = models['XGBoost']
importance_scores = pd.Series(
    xgb_model.feature_importances_, index=feature_cols
).sort_values(ascending=True)

colors_feat = ['#FFA500' if i == importance_scores.index[-1] else '#1E90FF'
               for i in importance_scores.index]

fig, ax = plt.subplots(figsize=(8, 4))
bars = importance_scores.plot.barh(ax=ax, color=colors_feat, edgecolor='white')
ax.set_title('XGBoost Feature Importances', fontweight='bold', fontsize=13)
ax.set_xlabel('Importance Score')
for patch, val in zip(ax.patches, importance_scores):
    ax.text(val + 0.005, patch.get_y() + patch.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9.5)
plt.tight_layout()
plt.show()

print("\n🏆 Most important feature:", importance_scores.idxmax(),
      f"({importance_scores.max():.3f})")

