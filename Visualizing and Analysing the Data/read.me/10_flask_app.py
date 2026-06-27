# ============================================================
# Step 10: Flask Web Application
# ============================================================

flask_app_code = '''
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model  = joblib.load("flood_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([[
        data["ANNUAL"],
        data["Jan_Feb"],
        data["Mar_May"],
        data["Jun_Sep"],
        data["Oct_Dec"],
        data["YEAR"]
    ]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = {
        "prediction" : int(prediction),
        "label"      : "🌊 FLOOD RISK DETECTED" if prediction == 1 else "✅ NO FLOOD RISK",
        "confidence" : f"{max(model.predict_proba(features)[0]) * 100:.2f}%"
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
'''

print("📄 Flask app code (app.py):")
print("=" * 60)
print(flask_app_code)
print("=" * 60)
print("\n▶  To run locally : python app.py")
print("▶  To deploy IBM Cloud: ibmcloud cf push flood-predictor")

