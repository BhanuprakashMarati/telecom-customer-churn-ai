"""
Reproducible Training Pipeline for Telecom Customer Churn Prediction.
- Loads genuine IBM Telco dataset
- Cleans missing values (e.g. whitespace in TotalCharges)
- Implements ColumnTransformer with OneHotEncoder and StandardScaler
- Evaluates Logistic Regression and Random Forest
- Computes Precision, Recall, F1, ROC-AUC, and Confusion Matrix
- Saves best pipeline, preprocessor, metrics, and visualization artifacts
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Set Matplotlib cache dir before importing matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplcache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature definitions
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET = "Churn"


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load and clean dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run scripts/download_data.py first.")

    df = pd.read_csv(filepath)
    print(f"Loaded raw data with shape: {df.shape}")

    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Clean TotalCharges: string with spaces -> numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    # For new customers (tenure = 0), fill TotalCharges with 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target explicitly as binary integer (1 for Yes, 0 for No)
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)

    # Ensure SeniorCitizen is treated as categorical/string for consistent one-hot encoding
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    print(f"Cleaned dataset shape: {df.shape}, Churn balance:\n{df[TARGET].value_counts(normalize=True).to_dict()}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessor for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate pipeline on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    print(f"\n--- {name} Results ---")
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1-Score:  {metrics['f1_score']}")
    print(f"ROC-AUC:   {metrics['roc_auc']}")
    print(f"Confusion Matrix [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
    return metrics, y_pred, y_proba


def train_and_evaluate():
    """Main training routine."""
    df = load_and_clean_data(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    preprocessor = build_preprocessor()

    # Define Candidate Models
    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                        C=0.8,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=8,
                        min_samples_split=6,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    all_metrics = {}
    best_model_name = None
    best_score = -1.0
    best_pipeline = None
    best_y_proba = None
    best_y_pred = None

    for name, pipe in models.items():
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(name, pipe, X_test, y_test)
        all_metrics[name] = metrics

        # Optimization criterion: Balance of ROC-AUC and F1
        score = metrics["roc_auc"] * 0.6 + metrics["f1_score"] * 0.4
        if score > best_score:
            best_score = score
            best_model_name = name
            best_pipeline = pipe
            best_y_proba = y_proba
            best_y_pred = y_pred

    print(f"\n[BEST MODEL SELECTED] {best_model_name} (Combined Score: {best_score:.4f})")

    # Extract transformed feature names for explainability
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERIC_FEATURES + cat_feature_names

    # Compute Feature Importances / Coefficients
    clf = best_pipeline.named_steps["classifier"]
    importance_dict = {}
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        for f, imp in zip(feature_names, importances):
            importance_dict[f] = float(imp)
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        for f, c in zip(feature_names, coefs):
            importance_dict[f] = float(c)

    # Sort top features
    sorted_importance = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    # Save artifacts
    pipeline_path = os.path.join(MODEL_DIR, "model.joblib")
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    classifier_path = os.path.join(MODEL_DIR, "classifier.joblib")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    features_path = os.path.join(MODEL_DIR, "feature_names.json")
    importance_path = os.path.join(MODEL_DIR, "feature_importance.json")

    joblib.dump(best_pipeline, pipeline_path)
    joblib.dump(fitted_preprocessor, preprocessor_path)
    joblib.dump(clf, classifier_path)

    summary = {
        "best_model": best_model_name,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "all_encoded": feature_names,
        },
        "metrics": all_metrics,
        "top_features": sorted_importance[:15],
    }

    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(features_path, "w") as f:
        json.dump({"features": feature_names, "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}, f, indent=2)

    with open(importance_path, "w") as f:
        json.dump(dict(sorted_importance), f, indent=2)

    # Generate Plots
    cm_plot_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, best_y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
    )
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    roc_plot_path = os.path.join(MODEL_DIR, "roc_curve.png")
    plt.figure(figsize=(7, 5))
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()

    # Feature Importance Plot
    fi_plot_path = os.path.join(MODEL_DIR, "feature_importance.png")
    top_10 = sorted_importance[:10]
    plt.figure(figsize=(8, 5))
    feat_names = [x[0] for x in top_10][::-1]
    feat_vals = [abs(x[1]) for x in top_10][::-1]
    plt.barh(feat_names, feat_vals, color="#1f77b4")
    plt.title(f"Top 10 Feature Importances ({best_model_name})")
    plt.xlabel("Absolute Importance / Magnitude")
    plt.tight_layout()
    plt.savefig(fi_plot_path, dpi=200)
    plt.close()

    print("\n[SUCCESS] Saved Artifacts:")
    print(f" - Pipeline: {pipeline_path}")
    print(f" - Preprocessor: {preprocessor_path}")
    print(f" - Metrics: {metrics_path}")
    print(f" - Confusion Matrix Plot: {cm_plot_path}")
    print(f" - ROC Curve Plot: {roc_plot_path}")
    print(f" - Feature Importance Plot: {fi_plot_path}")

    return summary


if __name__ == "__main__":
    train_and_evaluate()
