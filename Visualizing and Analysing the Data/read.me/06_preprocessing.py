# ============================================================
# Step 6: Data Preprocessing
# ============================================================

# Feature matrix and target
feature_cols = ['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'YEAR']
X = df[feature_cols]
y = df['FLOODS']

# Train/test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("=" * 45)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")
print(f"  Features         : {X_train.shape[1]}")
print("=" * 45)
print(f"  Train flood rate : {y_train.mean()*100:.1f}%")
print(f"  Test  flood rate : {y_test.mean()*100:.1f}%")
print("  ✅ Stratified split preserves class ratio")

