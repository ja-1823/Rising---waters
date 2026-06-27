# ============================================================
# Step 7: Model Training & Evaluation
# ============================================================

# ── Define models ─────────────────────────────────
models = {
    'Decision Tree'  : DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest'  : RandomForestClassifier(n_estimators=150, random_state=42),
    'KNN'            : KNeighborsClassifier(n_neighbors=7),
    'XGBoost'        : XGBClassifier(n_estimators=200, learning_rate=0.1,
                                      max_depth=6, random_state=42,
                                      eval_metric='logloss', verbosity=0),
}

results = {}
for name, model in models.items():
    # KNN & DT use scaled features; tree ensembles don't need scaling
    X_tr = X_train_sc if name in ('KNN',) else X_train
    X_te = X_test_sc  if name in ('KNN',) else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:,1]

    results[name] = {
        'accuracy'  : accuracy_score(y_test, y_pred),
        'roc_auc'   : roc_auc_score(y_test, y_prob),
        'y_pred'    : y_pred,
        'y_prob'    : y_prob,
        'report'    : classification_report(y_test, y_pred, target_names=['No Flood','Flood']),
        'cm'        : confusion_matrix(y_test, y_pred),
    }
    print(f"  ✅ {name:<18} Accuracy: {results[name]['accuracy']*100:.2f}%   AUC: {results[name]['roc_auc']:.4f}")

print("\n  Models trained successfully!")


# ── Accuracy comparison bar chart ─────────────────
names  = list(results.keys())
accs   = [results[n]['accuracy']*100 for n in names]
aucs   = [results[n]['roc_auc'] for n in names]
colors_bar = ['#5BA8E0','#5BA8E0','#5BA8E0','#FFA500']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bars = axes[0].bar(names, accs, color=colors_bar, edgecolor='white', linewidth=0.8)
axes[0].set_ylim(80, 101)
axes[0].set_title('Test Accuracy (%) by Model', fontweight='bold')
axes[0].set_ylabel('Accuracy (%)')
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')

bars2 = axes[1].bar(names, aucs, color=colors_bar, edgecolor='white', linewidth=0.8)
axes[1].set_ylim(0.8, 1.01)
axes[1].set_title('ROC-AUC Score by Model', fontweight='bold')
axes[1].set_ylabel('AUC Score')
for bar, val in zip(bars2, aucs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Model Performance Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ── Confusion Matrices ────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, name in zip(axes, names):
    cm = results[name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=0.5, linecolor='white',
                xticklabels=['No Flood','Flood'],
                yticklabels=['No Flood','Flood'],
                cbar=False, annot_kws={'size':13,'weight':'bold'})
    ax.set_title(f'{name}\nAcc: {results[name]["accuracy"]*100:.2f}%', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ── ROC Curves ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ['#5BA8E0','#32CD32','#9B59B6','#FFA500']

for (name, color) in zip(names, colors_roc):
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
    auc = results[name]['roc_auc']
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name}  (AUC = {auc:.4f})')

ax.plot([0,1],[0,1],'k--', lw=1, label='Random Classifier')
ax.set_title('ROC Curves — All Models', fontweight='bold', fontsize=13)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=10, loc='lower right')
ax.fill_between([0,1],[0,1], alpha=0.05, color='gray')
plt.tight_layout()
plt.show()


# ── Classification Report — XGBoost ───────────────
print("\n" + "=" * 55)
print("  📊 XGBoost Classification Report (Best Model)")
print("=" * 55)
print(results['XGBoost']['report'])

