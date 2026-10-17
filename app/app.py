"""
app.py — Flask REST API for Disease Risk Prediction
Run: python app.py
Endpoints:
  GET  /health
  POST /predict/heart      { age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal }
  POST /predict/diabetes   { Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age }
  POST /predict/liver      { Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                             Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio }
"""
import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_disease

app = Flask(__name__)
CORS(app)  # Allow all origins for local development

# ── Load models at startup ───────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load(name):
    model  = joblib.load(os.path.join(MODELS_DIR, f"{name}_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, f"{name}_scaler.pkl"))
    return model, scaler

heart_model,   heart_scaler   = load("heart")
diabetes_model, diabetes_scaler = load("diabetes")
liver_model,   liver_scaler   = load("liver")

print("[OK] All models loaded successfully.")


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": ["heart", "diabetes", "liver"]})


# ── Heart Disease Prediction ──────────────────────────────────────────────────
@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    data = request.get_json(force=True)
    try:
        features = [
            float(data["age"]),
            float(data["sex"]),
            float(data["cp"]),
            float(data["trestbps"]),
            float(data["chol"]),
            float(data["fbs"]),
            float(data["restecg"]),
            float(data["thalach"]),
            float(data["exang"]),
            float(data["oldpeak"]),
            float(data["slope"]),
            float(data["ca"]),
            float(data["thal"]),
        ]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    result = predict_disease(heart_model, heart_scaler, features)
    result["disease"] = "Heart Disease"
    return jsonify(result)


# ── Diabetes Prediction ───────────────────────────────────────────────────────
@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    data = request.get_json(force=True)
    try:
        features = [
            float(data["Pregnancies"]),
            float(data["Glucose"]),
            float(data["BloodPressure"]),
            float(data["SkinThickness"]),
            float(data["Insulin"]),
            float(data["BMI"]),
            float(data["DiabetesPedigreeFunction"]),
            float(data["Age"]),
        ]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    result = predict_disease(diabetes_model, diabetes_scaler, features)
    result["disease"] = "Diabetes"
    return jsonify(result)


# ── Liver Disease Prediction ──────────────────────────────────────────────────
@app.route("/predict/liver", methods=["POST"])
def predict_liver():
    data = request.get_json(force=True)
    try:
        gender_val = 1 if str(data["Gender"]).strip().lower() == "male" else 0
        features = [
            float(data["Age"]),
            float(gender_val),
            float(data["Total_Bilirubin"]),
            float(data["Direct_Bilirubin"]),
            float(data["Alkaline_Phosphotase"]),
            float(data["Alamine_Aminotransferase"]),
            float(data["Aspartate_Aminotransferase"]),
            float(data["Total_Protiens"]),
            float(data["Albumin"]),
            float(data["Albumin_and_Globulin_Ratio"]),
        ]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Missing or invalid field: {e}"}), 400

    result = predict_disease(liver_model, liver_scaler, features)
    result["disease"] = "Liver Disease"
    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=5000, host="0.0.0.0", use_reloader=False)
