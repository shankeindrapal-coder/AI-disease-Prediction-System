"""
AI Disease Prediction System - Model Training Script
Trains a Random Forest classifier on diabetes dataset
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

print("\n" + "="*60)
print("🏥 AI DISEASE PREDICTION SYSTEM - MODEL TRAINING")
print("="*60 + "\n")

# Load dataset
print("📊 Loading dataset...")
df = pd.read_csv("dataset.csv")
print(f"✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")

# Check for missing values
print("\n🔍 Checking data quality...")
missing = df.isnull().sum().sum()
if missing > 0:
    print(f"⚠️  Found {missing} missing values - handling...")
    df = df.dropna()
else:
    print("✅ No missing values found")

# Separate features and target
print("\n🎯 Preparing features...")
feature_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = df[feature_columns]
y = df['Outcome']

print(f"✅ Features: {feature_columns}")
print(f"✅ Target variable: Outcome (0=No Disease, 1=Disease)")
print(f"   - Class 0: {(y == 0).sum()} records")
print(f"   - Class 1: {(y == 1).sum()} records")

# Split dataset
print("\n📈 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training set: {X_train.shape[0]} records")
print(f"✅ Testing set: {X_test.shape[0]} records")

# Train Random Forest model
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("✅ Model training completed")

# Make predictions
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print(f"\n{'='*60}")
print("📈 MODEL PERFORMANCE METRICS")
print(f"{'='*60}")
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

print(f"🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Negatives:  {cm[0, 0]}")
print(f"   False Positives: {cm[0, 1]}")
print(f"   False Negatives: {cm[1, 0]}")
print(f"   True Positives:  {cm[1, 1]}")

# Get feature importance
print(f"\n🔍 Feature Importance:")
feature_importance = dict(zip(feature_columns, model.feature_importances_))
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"   {feature}: {importance:.4f}")

# Save model
print("\n💾 Saving model...")
model_path = Path("model.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

# Save metadata
# Calculate classification report metrics
report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'], output_dict=True)

metadata = {
    "accuracy": round(accuracy, 4),
    "roc_auc": round(roc_auc, 4),
    "precision_no_disease": round(report['No Disease']['precision'], 4),
    "precision_disease": round(report['Disease']['precision'], 4),
    "recall_no_disease": round(report['No Disease']['recall'], 4),
    "recall_disease": round(report['Disease']['recall'], 4),
    "f1_no_disease": round(report['No Disease']['f1-score'], 4),
    "f1_disease": round(report['Disease']['f1-score'], 4),
    "confusion_matrix": {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    },
    "features": feature_columns,
    "classes": ["No Disease", "Disease"],
    "model_type": "RandomForestClassifier",
    "dataset_size": len(df),
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

metadata_path = Path("model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved to {metadata_path}")

print(f"\n{'='*60}")
print("✅ TRAINING COMPLETE - Model is ready for deployment!")
print(f"{'='*60}\n")