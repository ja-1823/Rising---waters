# ============================================================
# FLOOD PREDICTION — EARLY WARNING SYSTEM
# Step 1: Importing Libraries
# ============================================================

# ── Numerical & Data ──────────────────────────
import numpy as np
import pandas as pd

# ── Visualisation ─────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Scikit-learn: Preprocessing ───────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Scikit-learn: Models ──────────────────────
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ── XGBoost ───────────────────────────────────
from xgboost import XGBClassifier

# ── Evaluation ────────────────────────────────
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)

# ── Model Persistence ─────────────────────────
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Styling ───────────────────────────────────
sns.set_theme(style='darkgrid', palette='muted')
plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'font.family': 'DejaVu Sans'})

print("✅ All libraries imported successfully!")
print(f"   NumPy      {np.__version__}")
print(f"   Pandas     {pd.__version__}")
print(f"   Seaborn    {sns.__version__}")

