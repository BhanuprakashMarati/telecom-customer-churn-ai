"""
Inference engine: loads model pipeline, computes predictions, categorizes risk,
and produces defensible factor explanations.
"""
import json
import os
from typing import Dict, List, Tuple
import joblib
import pandas as pd
from app.schemas import CustomerInput, PredictionResponse, RiskFactorDetail

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "model", "feature_importance.json")


class ModelService:
    """Singleton service for loading model artifacts and executing predictions."""
    _instance = None

    def __init__(self):
        self.pipeline = None
        self.metrics = {}
        self.feature_importance = {}
        self.model_name = "Unknown"
        self.load_artifacts()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def load_artifacts(self):
        """Load joblib pipeline and metadata."""
        if os.path.exists(MODEL_PATH):
            self.pipeline = joblib.load(MODEL_PATH)
        else:
            self.pipeline = None

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                self.metrics = json.load(f)
                self.model_name = self.metrics.get("best_model", "RandomForest")

        if os.path.exists(IMPORTANCE_PATH):
            with open(IMPORTANCE_PATH, "r") as f:
                self.feature_importance = json.load(f)

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict(self, customer: CustomerInput) -> PredictionResponse:
        """Execute end-to-end inference and explanation for a customer."""
        if not self.is_ready:
            raise RuntimeError("Model pipeline artifact not loaded. Train model first.")

        data_dict = customer.to_model_dict()
        df = pd.DataFrame([data_dict])

        # Predict probability
        proba = float(self.pipeline.predict_proba(df)[0, 1])
        prediction = int(proba >= 0.5)

        # Categorize risk tier
        if proba < 0.35:
            risk_category = "Low Risk"
        elif proba < 0.65:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"

        # Generate defensible risk factors
        factors = self._explain_risk_factors(customer, proba)

        # Recommend targeted actions
        actions = self._generate_recommendations(customer, proba, risk_category)

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(proba, 4),
            churn_percentage=f"{proba * 100:.1f}%",
            risk_category=risk_category,
            disclaimer="Never present predictions as certainties: Predictions represent probabilistic estimates based on historical statistical patterns, not guaranteed individual outcomes.",
            key_risk_factors=factors,
            recommended_retention_actions=actions,
        )

    def _explain_risk_factors(self, cust: CustomerInput, proba: float) -> List[RiskFactorDetail]:
        """Produce defensible customer-specific risk drivers."""
        factors: List[RiskFactorDetail] = []

        # 1. Contract Analysis
        if cust.Contract == "Month-to-month":
            factors.append(
                RiskFactorDetail(
                    feature="Contract: Month-to-month",
                    impact="Increases Churn Risk",
                    description="Flexible rolling contract provides zero barrier to switching carriers; historically the strongest statistical driver of churn.",
                )
            )
        else:
            factors.append(
                RiskFactorDetail(
                    feature=f"Contract: {cust.Contract}",
                    impact="Decreases Churn Risk",
                    description=f"{cust.Contract} commitment provides institutional retention stability and lowers churn probability.",
                )
            )

        # 2. Tenure Evaluation
        if cust.tenure <= 6:
            factors.append(
                RiskFactorDetail(
                    feature=f"Tenure: {cust.tenure} months (Early Lifecycle)",
                    impact="Increases Churn Risk",
                    description="Customers within their first 6 months have historically the highest attrition rates before loyalty stabilizes.",
                )
            )
        elif cust.tenure >= 36:
            factors.append(
                RiskFactorDetail(
                    feature=f"Tenure: {cust.tenure} months (Established)",
                    impact="Decreases Churn Risk",
                    description="Long tenure reflects strong habituation and significantly lower historical churn rates.",
                )
            )

        # 3. Internet Service & Tech Support
        if cust.InternetService == "Fiber optic":
            if cust.TechSupport == "No":
                factors.append(
                    RiskFactorDetail(
                        feature="Internet: Fiber Optic without TechSupport",
                        impact="Increases Churn Risk",
                        description="High-bandwidth Fiber optic accounts without dedicated tech support exhibit heightened frustration-driven churn.",
                    )
                )
            if cust.OnlineSecurity == "No":
                factors.append(
                    RiskFactorDetail(
                        feature="Security: No OnlineSecurity Addon",
                        impact="Increases Churn Risk",
                        description="Lack of bundled security protection correlates with higher vulnerability to competitive offers.",
                    )
                )
        elif cust.InternetService == "No":
            factors.append(
                RiskFactorDetail(
                    feature="Internet: No Internet Service",
                    impact="Decreases Churn Risk",
                    description="Phone-only accounts have low price volatility and consistently low churn rates.",
                )
            )

        # 4. Payment Method
        if cust.PaymentMethod == "Electronic check":
            factors.append(
                RiskFactorDetail(
                    feature="Payment Method: Electronic Check",
                    impact="Increases Churn Risk",
                    description="Manual electronic checking correlates with episodic bill scrutiny and higher voluntary cancellation.",
                )
            )
        elif "automatic" in cust.PaymentMethod:
            factors.append(
                RiskFactorDetail(
                    feature=f"Payment Method: {cust.PaymentMethod}",
                    impact="Decreases Churn Risk",
                    description="Automated billing reduces payment friction and prevents inadvertent lapse.",
                )
            )

        # 5. Pricing & Charges
        if cust.MonthlyCharges > 75.0:
            factors.append(
                RiskFactorDetail(
                    feature=f"Monthly Charges: ${cust.MonthlyCharges:.2f}",
                    impact="Increases Churn Risk",
                    description="Premium monthly charge tier creates price sensitivity if value perception declines.",
                )
            )
        elif cust.MonthlyCharges < 35.0:
            factors.append(
                RiskFactorDetail(
                    feature=f"Monthly Charges: ${cust.MonthlyCharges:.2f}",
                    impact="Decreases Churn Risk",
                    description="Low recurring cost increases customer retention threshold.",
                )
            )

        return factors[:5]

    def _generate_recommendations(
        self, cust: CustomerInput, proba: float, risk_tier: str
    ) -> List[str]:
        """Generate targeted retention interventions based on customer profile."""
        actions: List[str] = []

        if cust.Contract == "Month-to-month":
            actions.append(
                "Propose 1-year contract renewal with a 15% promotional discount or bundle credit."
            )

        if cust.PaymentMethod == "Electronic check":
            actions.append(
                "Incentivize enrollment in automatic bank/credit card payments with a $5 recurring bill credit."
            )

        if cust.InternetService == "Fiber optic" and cust.TechSupport == "No":
            actions.append(
                "Offer 3 months of complimentary VIP Tech Support and Security Suite upgrade."
            )

        if cust.tenure < 12:
            actions.append(
                "Schedule proactive customer success check-in to verify satisfaction and resolve service pain points."
            )

        if risk_tier == "High Risk" and not actions:
            actions.append(
                "Escalate to Priority Retention Team for personalized contract restructuring."
            )
        elif risk_tier == "Low Risk":
            actions.append(
                "Candidate for loyalty reward program or cross-sell/upsell premium streaming add-ons."
            )

        return actions
