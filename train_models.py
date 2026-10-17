"""
train_models.py — Train all three disease prediction models and save to models/
Run from the project root: python train_models.py
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, scaler, name):
    joblib.dump(model,  os.path.join(MODELS_DIR, f"{name}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{name}_scaler.pkl"))
    print(f"  Saved {name}_model.pkl and {name}_scaler.pkl")


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\n{'='*50}")
    print(f"  {name.upper()} MODEL — Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(classification_report(y_test, preds))
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# 1. HEART DISEASE
# Input features (13): age, sex, cp, trestbps, chol, fbs, restecg,
#                       thalach, exang, oldpeak, slope, ca, thal
# Target: 0 = no disease, 1 = disease
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining Heart Disease model...")
heart_df = pd.read_csv(os.path.join(DATASETS_DIR, "heart.csv"))

# Fill any nulls with median
heart_df.fillna(heart_df.median(numeric_only=True), inplace=True)

X_heart = heart_df[['age','sex','cp','trestbps','chol','fbs','restecg',
                     'thalach','exang','oldpeak','slope','ca','thal']].values
y_heart = heart_df['target'].values

X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart)

scaler_heart = StandardScaler()
X_h_train_s = scaler_heart.fit_transform(X_h_train)
X_h_test_s  = scaler_heart.transform(X_h_test)

model_heart = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_heart.fit(X_h_train_s, y_h_train)
acc_h = evaluate("Heart Disease", model_heart, X_h_test_s, y_h_test)

if acc_h < 0.80:
    print("  Trying GradientBoostingClassifier...")
    model_heart = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model_heart.fit(X_h_train_s, y_h_train)
    evaluate("Heart Disease (GB)", model_heart, X_h_test_s, y_h_test)

save_model(model_heart, scaler_heart, "heart")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DIABETES (Pima Indians)
# Input features (8): Pregnancies, Glucose, BloodPressure, SkinThickness,
#                     Insulin, BMI, DiabetesPedigreeFunction, Age
# Target: 0 = no diabetes, 1 = diabetes
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining Diabetes model...")
diab_df = pd.read_csv(os.path.join(DATASETS_DIR, "diabetes.csv"))

# Replace biologically impossible zeros with median
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    median_val = diab_df[col].replace(0, np.nan).median()
    diab_df[col] = diab_df[col].replace(0, median_val)

X_diab = diab_df[['Pregnancies','Glucose','BloodPressure','SkinThickness',
                   'Insulin','BMI','DiabetesPedigreeFunction','Age']].values
y_diab = diab_df['Outcome'].values

X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42, stratify=y_diab)

scaler_diab = StandardScaler()
X_d_train_s = scaler_diab.fit_transform(X_d_train)
X_d_test_s  = scaler_diab.transform(X_d_test)

model_diab = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_diab.fit(X_d_train_s, y_d_train)
acc_d = evaluate("Diabetes", model_diab, X_d_test_s, y_d_test)

if acc_d < 0.75:
    print("  Trying GradientBoostingClassifier...")
    model_diab = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model_diab.fit(X_d_train_s, y_d_train)
    evaluate("Diabetes (GB)", model_diab, X_d_test_s, y_d_test)

save_model(model_diab, scaler_diab, "diabetes")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIVER DISEASE (Indian Liver Patient Dataset)
# Input features (10): Age, Gender(encoded), Total_Bilirubin, Direct_Bilirubin,
#                      Alkaline_Phosphotase, Alamine_Aminotransferase,
#                      Aspartate_Aminotransferase, Total_Protiens, Albumin,
#                      Albumin_and_Globulin_Ratio
# Target: 1 = liver disease, 0 = no disease (normalized from 1/2)
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining Liver Disease model...")
liver_df = pd.read_csv(os.path.join(DATASETS_DIR, "liver.csv"))

# Drop rows with nulls (AGR column)
liver_df.dropna(inplace=True)

# Encode gender: Male=1, Female=0
liver_df['Gender'] = liver_df['Gender'].map({'Male': 1, 'Female': 0})

# Normalize target: 1 = disease, 2 = no disease → 1 = disease, 0 = no disease
liver_df['Dataset'] = liver_df['Dataset'].map({1: 1, 2: 0})

X_liver = liver_df[['Age','Gender','Total_Bilirubin','Direct_Bilirubin',
                     'Alkaline_Phosphotase','Alamine_Aminotransferase',
                     'Aspartate_Aminotransferase','Total_Protiens',
                     'Albumin','Albumin_and_Globulin_Ratio']].values
y_liver = liver_df['Dataset'].values

X_l_train, X_l_test, y_l_train, y_l_test = train_test_split(
    X_liver, y_liver, test_size=0.2, random_state=42, stratify=y_liver)

scaler_liver = StandardScaler()
X_l_train_s = scaler_liver.fit_transform(X_l_train)
X_l_test_s  = scaler_liver.transform(X_l_test)

model_liver = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_liver.fit(X_l_train_s, y_l_train)
acc_l = evaluate("Liver Disease", model_liver, X_l_test_s, y_l_test)

if acc_l < 0.72:
    print("  Trying GradientBoostingClassifier...")
    model_liver = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model_liver.fit(X_l_train_s, y_l_train)
    evaluate("Liver Disease (GB)", model_liver, X_l_test_s, y_l_test)

save_model(model_liver, scaler_liver, "liver")

print("\n" + "="*50)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print(f"Models saved to: {MODELS_DIR}")
