"""
Data schemas and input validation for Telecom Customer Churn Prediction.
"""
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


class CustomerInput(BaseModel):
    """Schema for individual customer prediction input with validation."""
    gender: Literal["Male", "Female"] = Field(
        default="Female", description="Customer biological gender"
    )
    SeniorCitizen: Union[int, str] = Field(
        default=0, description="Whether customer is a senior citizen (0 or 1)"
    )
    Partner: Literal["Yes", "No"] = Field(
        default="No", description="Whether customer has a partner"
    )
    Dependents: Literal["Yes", "No"] = Field(
        default="No", description="Whether customer has dependents"
    )
    tenure: int = Field(
        default=1, ge=0, le=120, description="Number of months customer has subscribed"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        default="Yes", description="Whether customer has telephone service"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        default="No", description="Whether customer has multiple phone lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        default="Fiber optic", description="Customer's internet service provider"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Whether customer has online security add-on"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Whether customer has online backup service"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Whether customer has device protection coverage"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Whether customer has premium technical support"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Whether customer subscribes to streaming TV"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Whether customer subscribes to streaming movies"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        default="Month-to-month", description="Contract term duration"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        default="Yes", description="Whether customer receives paperless invoices"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        default="Electronic check", description="Payment method used for bills"
    )
    MonthlyCharges: float = Field(
        default=70.35, ge=0.0, le=250.0, description="Monthly subscription charge in USD"
    )
    TotalCharges: Optional[float] = Field(
        default=None, ge=0.0, description="Cumulative charges incurred over tenure in USD"
    )

    @field_validator("SeniorCitizen")
    @classmethod
    def validate_senior(cls, v):
        if str(v) not in ["0", "1"]:
            raise ValueError("SeniorCitizen must be 0 or 1")
        return str(v)

    def to_model_dict(self) -> dict:
        """Convert to dict with derived TotalCharges if omitted."""
        data = self.model_dump()
        data["SeniorCitizen"] = str(data["SeniorCitizen"])
        if data["TotalCharges"] is None or data["TotalCharges"] <= 0.0:
            data["TotalCharges"] = round(float(data["tenure"] * data["MonthlyCharges"]), 2)
        return data


class RiskFactorDetail(BaseModel):
    """Explains a single risk driver contributing to churn probability."""
    feature: str
    impact: Literal["Increases Churn Risk", "Decreases Churn Risk", "Neutral"]
    description: str


class PredictionResponse(BaseModel):
    """Structured response model for churn prediction."""
    churn_prediction: int = Field(
        description="Binary prediction: 1 (Likely Churn), 0 (Likely Retained)"
    )
    churn_probability: float = Field(
        description="Estimated probability of churn between 0.0 and 1.0"
    )
    churn_percentage: str = Field(
        description="Formatted percentage string of churn probability"
    )
    risk_category: Literal["Low Risk", "Medium Risk", "High Risk"] = Field(
        description="Qualitative risk tier"
    )
    disclaimer: str = Field(
        default="Never present predictions as certainties: Predictions represent probabilistic estimates based on historical statistical patterns, not guaranteed individual outcomes."
    )
    key_risk_factors: List[RiskFactorDetail] = Field(
        default_factory=list, description="Top observed risk and protective factors"
    )
    recommended_retention_actions: List[str] = Field(
        default_factory=list, description="Targeted retention recommendations"
    )


class HealthResponse(BaseModel):
    """Schema for API health status."""
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    features_count: int
