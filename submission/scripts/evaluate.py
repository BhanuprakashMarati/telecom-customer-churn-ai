"""
Independent Model Evaluation Script.
Loads the saved pipeline and evaluates on the dataset or arbitrary split,
printing detailed metrics and classification report.
"""
import json
import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scripts.train import load_and_clean_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")


def evaluate_saved_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run scripts/train.py first.")
        sys.exit(1)

    print(f"Loading model pipeline from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)

    df = load_and_clean_data(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n================ FULL DATASET EVALUATION REPORT ================")
    print(f"Total Evaluated Records: {len(X)}")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Retained (No)", "Churned (Yes)"]))

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            saved_metrics = json.load(f)
        print("\n--- Saved Test-Set Benchmark Metrics ---")
        best_name = saved_metrics.get("best_model")
        print(f"Best Model Selected: {best_name}")
        best_test_metrics = saved_metrics["metrics"][best_name]
        for k, v in best_test_metrics.items():
            print(f"  {k}: {v}")

    print("================================================================")


if __name__ == "__main__":
    evaluate_saved_model()
