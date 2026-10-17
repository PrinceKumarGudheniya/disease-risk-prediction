"""
predict.py — Helper functions for disease prediction
"""
import numpy as np


def predict_disease(model, scaler, features: list) -> dict:
    """
    Given a trained model, its scaler, and a list of raw feature values,
    returns a prediction dict with risk label and confidence score.
    """
    X = np.array(features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    confidence = float(proba[prediction])
    risk = "High" if prediction == 1 else "Low"
    return {
        "prediction": prediction,
        "risk": risk,
        "confidence": round(confidence * 100, 1)
    }
