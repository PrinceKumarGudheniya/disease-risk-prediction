import sys, os, json

app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
sys.path.insert(0, app_dir)
os.chdir(app_dir)

import app as flask_app
client = flask_app.app.test_client()

print("=== INTEGRATION TEST ===")

# Health check
r = client.get('/health')
assert r.status_code == 200
assert r.get_json()['status'] == 'ok'
print("[PASS] GET /health")

# Heart Disease
payload = {'age':63,'sex':1,'cp':3,'trestbps':145,'chol':233,'fbs':1,'restecg':2,
           'thalach':150,'exang':1,'oldpeak':2.3,'slope':2,'ca':0,'thal':2}
r = client.post('/predict/heart', data=json.dumps(payload), content_type='application/json')
assert r.status_code == 200
d = r.get_json()
assert d['disease'] == 'Heart Disease'
assert d['risk'] in ['High', 'Low']
assert 0 <= d['confidence'] <= 100
print("[PASS] POST /predict/heart  -> Risk:", d['risk'], "  Confidence:", d['confidence'], "%")

# Diabetes
payload = {'Pregnancies':6,'Glucose':148,'BloodPressure':72,'SkinThickness':35,
           'Insulin':125,'BMI':33.6,'DiabetesPedigreeFunction':0.627,'Age':50}
r = client.post('/predict/diabetes', data=json.dumps(payload), content_type='application/json')
assert r.status_code == 200
d = r.get_json()
assert d['disease'] == 'Diabetes'
assert d['risk'] in ['High', 'Low']
print("[PASS] POST /predict/diabetes -> Risk:", d['risk'], " Confidence:", d['confidence'], "%")

# Liver Disease
payload = {'Age':50,'Gender':'Male','Total_Bilirubin':3.5,'Direct_Bilirubin':1.2,
           'Alkaline_Phosphotase':500,'Alamine_Aminotransferase':120,
           'Aspartate_Aminotransferase':90,'Total_Protiens':6.0,
           'Albumin':2.8,'Albumin_and_Globulin_Ratio':0.9}
r = client.post('/predict/liver', data=json.dumps(payload), content_type='application/json')
assert r.status_code == 200
d = r.get_json()
assert d['disease'] == 'Liver Disease'
assert d['risk'] in ['High', 'Low']
print("[PASS] POST /predict/liver   -> Risk:", d['risk'], "  Confidence:", d['confidence'], "%")

# Missing field should return 400
r = client.post('/predict/heart', data=json.dumps({'age':55}), content_type='application/json')
assert r.status_code == 400
print("[PASS] POST /predict/heart (missing fields) -> 400 error handled correctly")

# Healthy profile (low risk)
payload = {'age':35,'sex':0,'cp':2,'trestbps':110,'chol':180,'fbs':0,'restecg':0,
           'thalach':170,'exang':0,'oldpeak':0.0,'slope':0,'ca':0,'thal':1}
r = client.post('/predict/heart', data=json.dumps(payload), content_type='application/json')
d = r.get_json()
print("[INFO] Healthy heart profile -> Risk:", d['risk'], " Confidence:", d['confidence'], "%")

print()
print("=== ALL TESTS PASSED ===")
