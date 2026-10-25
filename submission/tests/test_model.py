"""
Tests for model loading, prediction validity, risk categorization, and sanity checks.
"""
import os
import pytest
from app.inference import ModelService
from app.schemas import CustomerInput

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")


def test_model_artifacts_exist():
    assert os.path.exists(MODEL_PATH), "model.joblib missing"
    assert os.path.exists(os.path.join(BASE_DIR, "model", "preprocessor.joblib"))
    assert os.path.exists(os.path.join(BASE_DIR, "model", "metrics.json"))
    assert os.path.exists(os.path.join(BASE_DIR, "model", "feature_names.json"))


def test_model_inference_output():
    service = ModelService.get_instance()
    assert service.is_ready, "ModelService failed to load pipeline"

    # Test high risk profile
    high_risk_cust = CustomerInput(
        gender="Male",
        SeniorCitizen=1,
        Partner="No",
        Dependents="No",
        tenure=1,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=95.0,
        TotalCharges=95.0,
    )

    pred = service.predict(high_risk_cust)
    assert 0.0 <= pred.churn_probability <= 1.0
    assert pred.risk_category in ["Low Risk", "Medium Risk", "High Risk"]
    assert len(pred.key_risk_factors) > 0
    assert len(pred.recommended_retention_actions) > 0
    assert "certainties" in pred.disclaimer.lower()


def test_model_relative_risk_sanity():
    service = ModelService.get_instance()

    high_risk_cust = CustomerInput(
        gender="Male",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=1,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=85.0,
    )

    loyal_cust = CustomerInput(
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="Yes",
        tenure=70,
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        Contract="Two year",
        PaperlessBilling="No",
        PaymentMethod="Credit card (automatic)",
        MonthlyCharges=50.0,
    )

    res_high = service.predict(high_risk_cust)
    res_low = service.predict(loyal_cust)

    # Sanity check: Short tenure + month-to-month must have strictly higher churn probability than 70-month + 2-year contract
    assert res_high.churn_probability > res_low.churn_probability
    assert res_low.risk_category == "Low Risk"


def test_predict_batch_vectorized():
    import pandas as pd
    service = ModelService.get_instance()

    # Test full dataset batch inference
    data_path = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(data_path)
    res = service.predict_batch(df)

    assert len(res) == len(df)
    assert set(res.columns) == {"Customer_Row", "Churn_Probability", "Risk_Tier", "Predicted_Label"}
    assert res["Churn_Probability"].between(0.0, 1.0).all()
    assert set(res["Risk_Tier"].unique()).issubset({"Low Risk", "Medium Risk", "High Risk"})
    assert set(res["Predicted_Label"].unique()).issubset({"Churn", "Retain"})
    assert res.isnull().sum().sum() == 0
