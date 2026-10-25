"""
Automated unit and integration tests for FastAPI endpoint functions and Pydantic schemas.
Tests root, health, metadata, prediction logic, and input validation synchronously and reliably.
"""
import pytest
from pydantic import ValidationError
from app.api import root, health_check, predict_churn, get_metadata
from app.schemas import CustomerInput, HealthResponse, PredictionResponse


def test_root_endpoint():
    res = root()
    assert isinstance(res, dict)
    assert res["status"] == "operational"
    assert "predict" in res["endpoints"]
    assert "health" in res["endpoints"]


def test_health_endpoint():
    res = health_check()
    assert isinstance(res, HealthResponse)
    assert res.status == "healthy"
    assert res.model_loaded is True
    assert res.model_name in ["RandomForest", "LogisticRegression"]
    assert res.features_count > 0


def test_metadata_endpoint():
    res = get_metadata()
    assert isinstance(res, dict)
    assert "best_model" in res
    assert "metrics" in res
    assert "top_features" in res
    assert len(res["top_features"]) > 0


def test_predict_endpoint_valid():
    cust = CustomerInput(
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=24,
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="No",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="One year",
        PaperlessBilling="No",
        PaymentMethod="Bank transfer (automatic)",
        MonthlyCharges=60.50,
        TotalCharges=1450.00,
    )
    res = predict_churn(cust)
    assert isinstance(res, PredictionResponse)
    assert res.churn_prediction in [0, 1]
    assert 0.0 <= res.churn_probability <= 1.0
    assert res.risk_category in ["Low Risk", "Medium Risk", "High Risk"]
    assert "never present predictions as certainties" in res.disclaimer.lower()
    assert len(res.key_risk_factors) > 0
    assert len(res.recommended_retention_actions) > 0


def test_predict_endpoint_auto_total_charges():
    # TotalCharges is omitted; should be auto-derived
    cust = CustomerInput(
        gender="Male",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=10,
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
        MonthlyCharges=90.0,
    )
    res = predict_churn(cust)
    assert isinstance(res, PredictionResponse)
    assert res.risk_category in ["Medium Risk", "High Risk"]
    assert res.churn_prediction == 1 or res.churn_probability >= 0.5


def test_predict_validation_errors():
    # Negative monthly charges
    with pytest.raises(ValidationError):
        CustomerInput(MonthlyCharges=-10.0)

    # Invalid contract
    with pytest.raises(ValidationError):
        CustomerInput(Contract="Lifetime")

    # Invalid tenure
    with pytest.raises(ValidationError):
        CustomerInput(tenure=-5)

    # Invalid SeniorCitizen
    with pytest.raises(ValidationError):
        CustomerInput(SeniorCitizen=9)
