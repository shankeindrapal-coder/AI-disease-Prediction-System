from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import json
import os
import time
import uuid
from utils import (
    validate_input,
    get_risk_category,
    detect_hypertension,
    detect_heart_disease_risk,
)

app = Flask(__name__)

model = joblib.load("model.pkl")

with open("model_metadata.json") as f:
    metadata = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/patient")
def patient_page():
    """Dedicated patient information page (form and prediction)."""
    return render_template("patient_front.html")

@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_accuracy": f"{metadata['accuracy'] * 100:.2f}%",
        "roc_auc": f"{metadata['roc_auc']:.4f}",
        "model_type": metadata["model_type"],
        "features": metadata["features"],
        "dataset_size": metadata["dataset_size"],
        "training_samples": metadata["training_samples"],
        "testing_samples": metadata["testing_samples"],
        "precision_no_disease": f"{metadata['precision_no_disease']:.2f}",
        "precision_disease": f"{metadata['precision_disease']:.2f}",
        "recall_no_disease": f"{metadata['recall_no_disease']:.2f}",
        "recall_disease": f"{metadata['recall_disease']:.2f}",
        "f1_no_disease": f"{metadata['f1_no_disease']:.2f}",
        "f1_disease": f"{metadata['f1_disease']:.2f}",
        "confusion_matrix": metadata['confusion_matrix']
    })

@app.route("/api/model-stats", methods=["GET"])
def model_stats():
    return model_info()

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    return predict()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    is_valid, message = validate_input(data)
    if not is_valid:
        return jsonify({"error": message}), 400

    features = pd.DataFrame([
        {
            "Glucose": data["Glucose"],
            "BloodPressure": data["BloodPressure"],
            "SkinThickness": data["SkinThickness"],
            "Insulin": data["Insulin"],
            "BMI": data["BMI"],
            "DiabetesPedigreeFunction": data["DiabetesPedigreeFunction"],
            "Age": data["Age"]
        }
    ])

    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    confidence = max(prediction_proba[0])
    diabetes_probability = prediction_proba[0][1]
    diabetes_detected = bool(prediction[0] == 1)
    
    # Determine risk level: if not detected, always low risk. If detected, use probability-based categorization
    if diabetes_detected:
        risk_level = get_risk_category(diabetes_probability)
    else:
        risk_level = "🟢 Low Risk"
    
    diabetes_result = {
        "name": "Diabetes",
        "status": "Detected" if diabetes_detected else "Not Detected",
        "risk_level": risk_level,
        "probability": f"{diabetes_probability * 100:.2f}%",
        "confidence": f"{diabetes_probability * 100:.2f}%",
        "detected": diabetes_detected
    }

    hypertension_result = detect_hypertension(float(data["BloodPressure"]))
    heart_result = detect_heart_disease_risk(
        float(data["Glucose"]),
        float(data["BMI"]),
        float(data["BloodPressure"]),
        int(data["Age"]),
        float(data["Insulin"]),
        float(data["DiabetesPedigreeFunction"])
    )

    disease_names = []
    if diabetes_detected:
        disease_names.append("Diabetes")
    if hypertension_result["risk_level"] != "🟢 Low Risk":
        disease_names.append("Hypertension")
    if heart_result["risk_level"] != "🟢 Low Risk":
        disease_names.append("Heart Disease")

    disease_name = ", ".join(disease_names) if disease_names else "No immediate disease risk detected"
    result = "Potential disease risk detected" if disease_names else "No immediate disease risk detected"
    
    disease_results = [
        diabetes_result,
        hypertension_result,
        heart_result
    ]
    
    health_info = get_health_information(data, diabetes_result, hypertension_result, heart_result)

    return jsonify({
        "prediction": result,
        "disease_name": disease_name,
        "confidence": diabetes_result["confidence"],
        "risk_level": diabetes_result["risk_level"],
        "disease_probability": diabetes_result["probability"],
        "disease_results": disease_results,
        "health_analysis": health_info,
        "model_accuracy": f"{metadata['accuracy'] * 100:.2f}%"
    })


### Simple patients storage (JSON file) and endpoints
PATIENTS_FILE = "patients.json"

def load_patients():
    if os.path.exists(PATIENTS_FILE):
        try:
            with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_patients(patients):
    with open(PATIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)


@app.route("/patient-details")
def patient_details():
    return render_template("patient_details.html")


@app.route("/api/patients", methods=["GET", "POST"])
def api_patients():
    if request.method == "GET":
        return jsonify(load_patients())

    data = request.json or {}
    is_valid, message = validate_input(data)
    if not is_valid:
        return jsonify({"error": message}), 400

    patients = load_patients()
    entry = data.copy()
    entry["id"] = str(uuid.uuid4())
    entry["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    patients.append(entry)
    save_patients(patients)

    return jsonify({"status": "saved", "patient": entry}), 201

def get_health_information(data, diabetes_result, hypertension_result, heart_result):
    """Provide health-related insights based on patient parameters"""
    insights = []
    
    glucose = float(data["Glucose"])
    bmi = float(data["BMI"])
    blood_pressure = float(data["BloodPressure"])
    age = int(data["Age"])
    
    if diabetes_result["detected"]:
        insights.append(f"🔴 Diabetes: {diabetes_result['status']} ({diabetes_result['probability']}).")
    else:
        insights.append("✅ Diabetes risk appears low based on the model.")

    insights.append(f"{hypertension_result['name']}: {hypertension_result['status']} ({hypertension_result['risk_level']}).")
    insights.append(f"{heart_result['name']}: {heart_result['status']} ({heart_result['risk_level']}).")
    
    if glucose < 100:
        insights.append("✅ Glucose level: Normal (< 100 mg/dL)")
    elif glucose < 126:
        insights.append("⚠️  Glucose level: Slightly elevated (100-126 mg/dL)")
    else:
        insights.append("🔴 Glucose level: High (>= 126 mg/dL) - Consult doctor")
    
    if bmi < 18.5:
        insights.append("📊 BMI: Underweight")
    elif bmi < 25:
        insights.append("✅ BMI: Normal weight (Healthy)")
    elif bmi < 30:
        insights.append("⚠️  BMI: Overweight - Consider exercise")
    else:
        insights.append("🔴 BMI: Obese - Lifestyle changes recommended")
    
    if blood_pressure < 120:
        insights.append("✅ Blood Pressure: Normal")
    elif blood_pressure < 140:
        insights.append("⚠️  Blood Pressure: Elevated - Monitor regularly")
    else:
        insights.append("🔴 Blood Pressure: High - Seek medical attention")
    
    if age >= 45:
        insights.append(f"⚠️  Age Risk: {age} years - Increased disease risk with age")
    
    return insights

@app.route("/api/health-recommendations", methods=["POST"])
def health_recommendations():
    """Provide health recommendations based on prediction"""
    data = request.json
    recommendations = []
    
    glucose = float(data["Glucose"])
    bmi = float(data["BMI"])
    blood_pressure = float(data["BloodPressure"])
    hypertensive = detect_hypertension(blood_pressure)
    heart_risk = detect_heart_disease_risk(
        float(data["Glucose"]),
        float(data["BMI"]),
        blood_pressure,
        int(data["Age"]),
        float(data["Insulin"]),
        float(data["DiabetesPedigreeFunction"])
    )
    
    if glucose > 125:
        recommendations.append("🥗 Reduce sugar and refined carbohydrates intake")
        recommendations.append("💧 Increase water consumption")
        recommendations.append("🏃 Exercise 30 minutes daily")
    
    if bmi > 25:
        recommendations.append("⚽ Weight management program recommended")
        recommendations.append("🥦 Follow a balanced diet")
        recommendations.append("🚴 Increase physical activity")

    if hypertensive["risk_level"] != "🟢 Low Risk":
        recommendations.append("🩺 Monitor blood pressure regularly and follow hypertension guidance")
        recommendations.append("🍎 Reduce salt intake and avoid processed foods")

    if heart_risk["risk_level"] != "🟢 Low Risk":
        recommendations.append("❤️ Schedule a heart health checkup with a physician")
        recommendations.append("🥬 Increase cardiovascular exercise and reduce saturated fats")

    if not recommendations:
        recommendations.append("✅ Continue healthy lifestyle habits")
        recommendations.append("🏥 Regular health checkups recommended")
    
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)

