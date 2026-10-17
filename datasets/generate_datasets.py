"""
Generate realistic synthetic datasets that mirror the schema of the public UCI/Kaggle datasets.
These datasets are statistically representative and suitable for ML training/demo purposes.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

# ─────────────────────────────────────────
# 1. Heart Disease Dataset (UCI Cleveland)
# Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
# Target: 0 = no disease, 1 = disease
# ─────────────────────────────────────────
age = np.random.randint(29, 78, N)
sex = np.random.randint(0, 2, N)
cp = np.random.randint(0, 4, N)
trestbps = np.random.randint(94, 200, N)
chol = np.random.randint(126, 564, N)
fbs = (np.random.rand(N) > 0.85).astype(int)
restecg = np.random.randint(0, 3, N)
thalach = np.random.randint(71, 202, N)
exang = (np.random.rand(N) > 0.68).astype(int)
oldpeak = np.round(np.random.uniform(0, 6.2, N), 1)
slope = np.random.randint(0, 3, N)
ca = np.random.randint(0, 5, N)
thal = np.random.choice([0, 1, 2, 3], N)

# Disease risk correlated with features
risk_score = (
    0.03 * age +
    0.3 * sex +
    -0.25 * cp +
    0.01 * (trestbps - 130) +
    0.002 * (chol - 200) +
    0.3 * fbs +
    0.15 * exang +
    0.2 * oldpeak +
    0.2 * ca +
    np.random.normal(0, 0.5, N)
)
target_heart = (risk_score > np.percentile(risk_score, 46)).astype(int)

heart_df = pd.DataFrame({
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
    'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
    'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
    'thal': thal, 'target': target_heart
})
heart_df.to_csv('heart.csv', index=False)
print(f"heart.csv: {len(heart_df)} rows, target distribution: {heart_df['target'].value_counts().to_dict()}")

# ─────────────────────────────────────────
# 2. Pima Indians Diabetes Dataset
# Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
# Target: 0 = no diabetes, 1 = diabetes
# ─────────────────────────────────────────
pregnancies = np.random.randint(0, 18, N)
glucose = np.random.randint(56, 199, N)
blood_pressure = np.random.randint(24, 122, N)
skin_thickness = np.random.randint(7, 99, N)
insulin = np.random.randint(14, 846, N)
bmi = np.round(np.random.uniform(18.0, 67.1, N), 1)
dpf = np.round(np.random.uniform(0.078, 2.42, N), 3)
d_age = np.random.randint(21, 81, N)

risk_score_d = (
    0.05 * pregnancies +
    0.04 * (glucose - 100) +
    0.008 * (bmi - 30) +
    0.3 * dpf +
    0.02 * d_age +
    np.random.normal(0, 0.8, N)
)
target_diabetes = (risk_score_d > np.percentile(risk_score_d, 65)).astype(int)

diabetes_df = pd.DataFrame({
    'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
    'DiabetesPedigreeFunction': dpf, 'Age': d_age, 'Outcome': target_diabetes
})
diabetes_df.to_csv('diabetes.csv', index=False)
print(f"diabetes.csv: {len(diabetes_df)} rows, target distribution: {diabetes_df['Outcome'].value_counts().to_dict()}")

# ─────────────────────────────────────────
# 3. Indian Liver Patient Dataset (ILPD)
# Features: Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
#           Alamine_Aminotransferase, Aspartate_Aminotransferase,
#           Total_Protiens, Albumin, Albumin_and_Globulin_Ratio
# Target: 1 = liver disease, 2 = no disease (we normalize to 1/0 during training)
# ─────────────────────────────────────────
l_age = np.random.randint(4, 90, N)
gender = np.random.choice(['Male', 'Female'], N, p=[0.75, 0.25])
total_bilirubin = np.round(np.random.exponential(1.5, N) + 0.4, 2).clip(0.4, 75)
direct_bilirubin = np.round(total_bilirubin * np.random.uniform(0.2, 0.6, N), 2)
alk_phos = np.random.randint(63, 2110, N)
alt = np.random.randint(10, 2000, N)
ast = np.random.randint(10, 4929, N)
total_proteins = np.round(np.random.uniform(2.7, 9.6, N), 1)
albumin = np.round(np.random.uniform(0.9, 5.5, N), 1)
agr = np.round(albumin / (total_proteins - albumin + 0.001), 2).clip(0.3, 2.8)

# Introduce some NaN in agr (as in original dataset)
nan_idx = np.random.choice(N, 4, replace=False)
agr[nan_idx] = np.nan

risk_score_l = (
    0.02 * l_age +
    0.3 * (gender == 'Male').astype(int) +
    0.15 * total_bilirubin +
    0.0001 * alt +
    0.0001 * ast +
    np.random.normal(0, 0.5, N)
)
target_liver = np.where(risk_score_l > np.percentile(risk_score_l, 28), 1, 2)

liver_df = pd.DataFrame({
    'Age': l_age, 'Gender': gender,
    'Total_Bilirubin': total_bilirubin, 'Direct_Bilirubin': direct_bilirubin,
    'Alkaline_Phosphotase': alk_phos, 'Alamine_Aminotransferase': alt,
    'Aspartate_Aminotransferase': ast, 'Total_Protiens': total_proteins,
    'Albumin': albumin, 'Albumin_and_Globulin_Ratio': agr,
    'Dataset': target_liver
})
liver_df.to_csv('liver.csv', index=False)
print(f"liver.csv: {len(liver_df)} rows, target distribution: {liver_df['Dataset'].value_counts().to_dict()}")

print("\nAll datasets generated successfully!")
