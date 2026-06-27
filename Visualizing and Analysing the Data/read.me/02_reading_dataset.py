# ============================================================
# Step 2: Reading the Dataset
# ============================================================

# Load dataset
df = pd.read_csv('flood_data.csv')

print("=" * 55)
print(f"  Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print("=" * 55)
print("\n📋 First 5 rows:")
df.head()


# Column data types and non-null counts
print("\n📋 Dataset Info:")
print("-" * 45)
df.info()


# Missing value check
missing = df.isnull().sum()
print("\n🔍 Missing Values per Column:")
print("-" * 35)
if missing.sum() == 0:
    print("  ✅ No missing values found!")
else:
    print(missing[missing > 0])

# Target distribution
print("\n🎯 Target Variable — FLOODS:")
print("-" * 35)
vc = df['FLOODS'].value_counts()
print(f"  No Flood (0) : {vc[0]:>5}  ({vc[0]/len(df)*100:.1f}%)")
print(f"  Flood    (1) : {vc[1]:>5}  ({vc[1]/len(df)*100:.1f}%)")

