# HealthPredict AI — Disease Risk Prediction

An AI/ML web application that predicts risk for **Heart Disease**, **Diabetes**, and **Liver Disease** using machine learning models trained on real clinical datasets.

## Features

- **3 Disease Models** — Heart Disease, Diabetes, Liver Disease
- **~80% accuracy** — Random Forest & Gradient Boosting classifiers
- **Instant predictions** — Flask REST API with JSON responses
- **Clean web UI** — Responsive HTML/CSS/JS frontend, no frameworks needed
- **Confidence scores** — Each prediction includes a confidence percentage

## Project Structure

```
disease-risk-prediction/
├── datasets/
│   ├── heart.csv               # Heart Disease dataset
│   ├── diabetes.csv            # Pima Indians Diabetes dataset
│   ├── liver.csv               # Indian Liver Patient dataset
│   └── generate_datasets.py    # Dataset generation script
├── models/                     # Trained model .pkl files (generated)
├── app/
│   ├── app.py                  # Flask REST API
│   ├── predict.py              # Prediction helper
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── index.html              # Home / landing page
│   ├── heart.html              # Heart disease form
│   ├── diabetes.html           # Diabetes form
│   ├── liver.html              # Liver disease form
│   ├── style.css               # Global stylesheet
│   └── script.js               # API call logic
└── train_models.py             # Model training script
```

## Prerequisites

- Python 3.10 or higher
- pip

## Quick Start

### Step 1 — Install dependencies

```bash
pip install -r app/requirements.txt
```

### Step 2 — Train the models

```bash
python train_models.py
```

This generates 6 files in `models/`:
- `heart_model.pkl` + `heart_scaler.pkl`
- `diabetes_model.pkl` + `diabetes_scaler.pkl`
- `liver_model.pkl` + `liver_scaler.pkl`

### Step 3 — Start the Flask API server

```bash
python app/app.py
```

The server runs at `http://localhost:5000`. You should see:
```
[OK] All models loaded successfully.
 * Running on http://127.0.0.1:5000
```

### Step 4 — Open the website

Open `frontend/index.html` in your browser (double-click or drag into browser). No web server needed — it uses `fetch()` to call the Flask API directly.

---

## API Endpoints

### `GET /health`
Returns server status.

### `POST /predict/heart`
```json
{
  "age": 55, "sex": 1, "cp": 0, "trestbps": 140, "chol": 250,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 1,
  "oldpeak": 2.5, "slope": 2, "ca": 1, "thal": 2
}
```

### `POST /predict/diabetes`
```json
{
  "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72,
  "SkinThickness": 35, "Insulin": 85, "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627, "Age": 50
}
```

### `POST /predict/liver`
```json
{
  "Age": 45, "Gender": "Male", "Total_Bilirubin": 1.5,
  "Direct_Bilirubin": 0.6, "Alkaline_Phosphotase": 200,
  "Alamine_Aminotransferase": 56, "Aspartate_Aminotransferase": 40,
  "Total_Protiens": 6.8, "Albumin": 3.5,
  "Albumin_and_Globulin_Ratio": 1.1
}
```

### Response format (all endpoints)
```json
{
  "disease": "Heart Disease",
  "prediction": 1,
  "risk": "High",
  "confidence": 73.8
}
```

---

## Sample Test Inputs

### Heart Disease (High Risk profile)
| Field | Value |
|---|---|
| Age | 63 |
| Sex | Male (1) |
| Chest Pain Type | Asymptomatic (3) |
| Resting BP | 145 |
| Cholesterol | 233 |
| Fasting Blood Sugar > 120 | Yes (1) |
| Resting ECG | LV Hypertrophy (2) |
| Max Heart Rate | 150 |
| Exercise Angina | Yes (1) |
| ST Depression | 2.3 |
| Slope | Downsloping (2) |
| Major Vessels | 0 |
| Thalassemia | Reversible Defect (2) |

### Diabetes (High Risk profile)
| Field | Value |
|---|---|
| Pregnancies | 6 |
| Glucose | 148 |
| Blood Pressure | 72 |
| Skin Thickness | 35 |
| Insulin | 125 |
| BMI | 33.6 |
| Pedigree Function | 0.627 |
| Age | 50 |

### Liver Disease (High Risk profile)
| Field | Value |
|---|---|
| Age | 50 |
| Gender | Male |
| Total Bilirubin | 3.5 |
| Direct Bilirubin | 1.2 |
| Alkaline Phosphatase | 500 |
| ALT (SGPT) | 120 |
| AST (SGOT) | 90 |
| Total Proteins | 6.0 |
| Albumin | 2.8 |
| A/G Ratio | 0.9 |

---

## Model Performance

| Disease | Algorithm | Test Accuracy |
|---|---|---|
| Heart Disease | Gradient Boosting | ~78% |
| Diabetes | Random Forest | ~85% |
| Liver Disease | Random Forest | ~77% |

---

## Dataset Attribution

| Dataset | Source |
|---|---|
| UCI Heart Disease | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |
| Pima Indians Diabetes | [Kaggle / NIDDK](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Indian Liver Patient Dataset | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29) |

---

## Disclaimer

This application is for **educational and demonstration purposes only**. It is not intended to provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.
