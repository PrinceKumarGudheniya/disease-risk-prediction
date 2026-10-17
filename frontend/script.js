/**
 * script.js — HealthPredict AI frontend logic
 * Handles form submission, API calls, result rendering, and loading state.
 */

const API_BASE = "http://localhost:5000";

/**
 * Generic form submit handler.
 * @param {string} endpoint  - e.g. "heart", "diabetes", "liver"
 * @param {Object} payload   - key/value pairs matching the API's expected fields
 */
async function submitPrediction(endpoint, payload) {
  const resultBox = document.getElementById("result-box");
  const spinner   = document.getElementById("spinner");
  const submitBtn = document.getElementById("submit-btn");

  // Show loading state
  resultBox.className = "";
  resultBox.style.display = "none";
  spinner.style.display = "block";
  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing...";

  try {
    const response = await fetch(`${API_BASE}/predict/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(resultBox, data.error || "Server error. Please check your inputs.");
      return;
    }

    showResult(resultBox, data);

  } catch (err) {
    showError(resultBox, "Could not connect to the prediction server. Make sure the Flask server is running on port 5000.");
  } finally {
    spinner.style.display = "none";
    submitBtn.disabled = false;
    submitBtn.textContent = "Get Prediction";
  }
}

function showResult(box, data) {
  const isHigh = data.risk === "High";
  box.className = isHigh ? "high-risk show" : "low-risk show";
  box.style.display = "block";
  box.innerHTML = `
    <div class="result-header">
      <span class="result-icon">${isHigh ? "&#9888;" : "&#10003;"}</span>
      <div>
        <div class="result-title">${isHigh ? "High Risk" : "Low Risk"} — ${data.disease}</div>
        <div class="result-details">
          Model confidence: <strong>${data.confidence}%</strong>
        </div>
      </div>
    </div>
    <div class="confidence-bar-wrap">
      <div class="confidence-label">Confidence Score</div>
      <div class="confidence-bar">
        <div class="confidence-fill" style="width: ${data.confidence}%"></div>
      </div>
    </div>
    <p style="margin-top:1rem; font-size:0.85rem; color:#555;">
      ${isHigh
        ? "The model indicates an elevated risk. Please consult a qualified healthcare provider for a proper diagnosis."
        : "The model indicates a lower risk. Regular checkups and a healthy lifestyle are always recommended."}
    </p>
  `;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(box, message) {
  box.className = "high-risk show";
  box.style.display = "block";
  box.innerHTML = `
    <div class="result-header">
      <span class="result-icon">&#9888;</span>
      <div class="result-title">Error</div>
    </div>
    <p style="margin-top:0.5rem; font-size:0.9rem; color:#555;">${message}</p>
  `;
}

// ─── HEART DISEASE FORM ───────────────────────────────────────────────────────
function submitHeart(event) {
  event.preventDefault();
  const f = event.target;
  const payload = {
    age:      parseFloat(f.age.value),
    sex:      parseFloat(f.sex.value),
    cp:       parseFloat(f.cp.value),
    trestbps: parseFloat(f.trestbps.value),
    chol:     parseFloat(f.chol.value),
    fbs:      parseFloat(f.fbs.value),
    restecg:  parseFloat(f.restecg.value),
    thalach:  parseFloat(f.thalach.value),
    exang:    parseFloat(f.exang.value),
    oldpeak:  parseFloat(f.oldpeak.value),
    slope:    parseFloat(f.slope.value),
    ca:       parseFloat(f.ca.value),
    thal:     parseFloat(f.thal.value)
  };
  submitPrediction("heart", payload);
}

// ─── DIABETES FORM ────────────────────────────────────────────────────────────
function submitDiabetes(event) {
  event.preventDefault();
  const f = event.target;
  const payload = {
    Pregnancies:              parseFloat(f.Pregnancies.value),
    Glucose:                  parseFloat(f.Glucose.value),
    BloodPressure:            parseFloat(f.BloodPressure.value),
    SkinThickness:            parseFloat(f.SkinThickness.value),
    Insulin:                  parseFloat(f.Insulin.value),
    BMI:                      parseFloat(f.BMI.value),
    DiabetesPedigreeFunction: parseFloat(f.DiabetesPedigreeFunction.value),
    Age:                      parseFloat(f.Age.value)
  };
  submitPrediction("diabetes", payload);
}

// ─── LIVER DISEASE FORM ───────────────────────────────────────────────────────
function submitLiver(event) {
  event.preventDefault();
  const f = event.target;
  const payload = {
    Age:                          parseFloat(f.Age.value),
    Gender:                       f.Gender.value,
    Total_Bilirubin:              parseFloat(f.Total_Bilirubin.value),
    Direct_Bilirubin:             parseFloat(f.Direct_Bilirubin.value),
    Alkaline_Phosphotase:         parseFloat(f.Alkaline_Phosphotase.value),
    Alamine_Aminotransferase:     parseFloat(f.Alamine_Aminotransferase.value),
    Aspartate_Aminotransferase:   parseFloat(f.Aspartate_Aminotransferase.value),
    Total_Protiens:               parseFloat(f.Total_Protiens.value),
    Albumin:                      parseFloat(f.Albumin.value),
    Albumin_and_Globulin_Ratio:   parseFloat(f.Albumin_and_Globulin_Ratio.value)
  };
  submitPrediction("liver", payload);
}
