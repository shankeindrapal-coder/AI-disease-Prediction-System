/**
 * AI Disease Prediction System - Frontend Script
 * Handles form submission and API communication
 */

const form = document.getElementById("predictionForm");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const loadingBox = document.getElementById("loading");
const modelInfoBox = document.getElementById("modelInfo");

// Load model statistics on page load
document.addEventListener("DOMContentLoaded", async () => {
  await loadModelInfo();
});

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-stats");
    const stats = await response.json();
    
    modelInfoBox.innerHTML = `
      <strong>📊 Model Information:</strong><br>
      Accuracy: ${(stats.accuracy * 100).toFixed(2)}% | 
      ROC-AUC: ${stats.roc_auc.toFixed(4)} | 
      Dataset Size: ${stats.dataset_size} samples
    `;
  } catch (error) {
    console.error("Failed to load model info:", error);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  
  // Clear previous results
  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  loadingBox.classList.remove("hidden");

  // Collect form data
  const formData = new FormData(form);
  const data = {
    Glucose: parseFloat(formData.get("Glucose")),
    BloodPressure: parseFloat(formData.get("BloodPressure")),
    SkinThickness: parseFloat(formData.get("SkinThickness")),
    Insulin: parseFloat(formData.get("Insulin")),
    BMI: parseFloat(formData.get("BMI")),
    DiabetesPedigreeFunction: parseFloat(formData.get("DiabetesPedigreeFunction")),
    Age: parseFloat(formData.get("Age"))
  };

  // Validate all values are numbers
  const allValid = Object.values(data).every(val => !isNaN(val));
  if (!allValid) {
    showError("❌ Error: Please enter valid numeric values for all fields.");
    loadingBox.classList.add("hidden");
    return;
  }

  try {
    // Make API request
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    loadingBox.classList.add("hidden");

    if (!response.ok) {
      const errorData = await response.json();
      showError(
        `❌ Prediction Error (${response.status}): ${errorData.detail || "Unknown error occurred"}`
      );
      return;
    }

    const result = await response.json();
    displayResult(result, data);

  } catch (error) {
    loadingBox.classList.add("hidden");
    showError(`❌ Network Error: ${error.message}`);
    console.error("Error details:", error);
  }
});

function displayResult(prediction, inputData) {
  resultBox.classList.remove("hidden");
  errorBox.classList.add("hidden");

  const confidencePercent = parseFloat(prediction.confidence);
  const riskColor = confidencePercent > 70 ? "🔴" : 
                    confidencePercent > 40 ? "🟡" : "🟢";

  resultBox.innerHTML = `
    <h3>${riskColor} Prediction Result</h3>
    <div class="result-item">
      <strong>Disease Status:</strong> ${prediction.prediction}
    </div>
    <div class="result-item">
      <strong>Conditions:</strong> ${prediction.disease_results
        .map(item => `${item.name}: ${item.status}`)
        .join(' | ')}
    </div>
    <div class="result-item">
      <strong>Confidence:</strong> ${prediction.confidence}%
    </div>
    <div class="result-item">
      <strong>Risk Level:</strong> ${prediction.risk_level}
    </div>
    <div class="result-item" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(0,0,0,0.1);">
      <small><strong>Input Summary:</strong></small><br>
      <small>
        Glucose: ${inputData.Glucose} | 
        BP: ${inputData.BloodPressure} | 
        BMI: ${inputData.BMI} | 
        Age: ${inputData.Age}
      </small>
    </div>
  `;

  if (prediction.disease_results && prediction.disease_results.length > 0) {
    resultBox.innerHTML += `
      <div class="result-item" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(0,0,0,0.1);">
        <small><strong>Condition Details:</strong></small><br>
        <small>${prediction.disease_results
          .map(item => `${item.name}: ${item.status} (${item.risk_level})`)
          .join(' | ')}</small>
      </div>
    `;
  }

  resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(message) {
  errorBox.classList.remove("hidden");
  resultBox.classList.add("hidden");
  errorBox.innerHTML = `<h3>⚠️ ${message}</h3>`;
  errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Add keyboard shortcut to submit form (Ctrl+Enter)
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    form.dispatchEvent(new Event("submit"));
  }
});
