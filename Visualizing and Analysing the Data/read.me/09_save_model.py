# ============================================================
# Step 9: Saving the Best Model (Joblib & Pickle)
# ============================================================

# Save XGBoost model and scaler using Joblib
joblib.dump(models['XGBoost'], 'flood_xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Also save with Pickle
with open('flood_model_pickle.pkl', 'wb') as f:
    pickle.dump(models['XGBoost'], f)

print("✅ Model saved:")
print("   flood_xgb_model.pkl  (Joblib)")
print("   scaler.pkl           (Joblib)")
print("   flood_model_pickle.pkl (Pickle)")

# Verify by reloading
loaded_model  = joblib.load('flood_xgb_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
verify_pred   = loaded_model.predict(X_test[:5])
print("\n✅ Reload verification — first 5 predictions:", verify_pred)
print("   Actual labels                             :", y_test.values[:5])

