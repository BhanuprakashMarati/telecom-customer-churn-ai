"""
FastAPI Backend for Telecom Customer Churn Prediction.
Provides GET /health and POST /predict with full validation and error handling.
"""
import logging
import os
import sys
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure repository root and app directory are on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from app.inference import ModelService
    from app.schemas import CustomerInput, HealthResponse, PredictionResponse
except ModuleNotFoundError:
    from inference import ModelService
    from schemas import CustomerInput, HealthResponse, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="Production-grade AI microservice for churn probability estimation, risk categorization, and defensible factor analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService.get_instance()


@app.get("/", tags=["System"])
def root():
    """Welcome and quick link documentation endpoint."""
    return {
        "service": "Telecom Customer Churn Prediction Service",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "metadata": "/metadata",
        },
        "disclaimer": "Predictions represent probabilistic risk assessments based on historical data; never present as certainties.",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check API health and model readiness."""
    is_ready = model_service.is_ready
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_name=model_service.model_name,
        model_version="1.0.0",
        features_count=len(model_service.metrics.get("features", {}).get("all_encoded", [])) or 19,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict_churn(customer: CustomerInput):
    """
    Predict telecom customer churn risk.
    Accepts full customer demographic, subscription, and billing details.
    Returns:
    - churn_prediction (0 or 1)
    - churn_probability (0.0 - 1.0)
    - risk_category (Low Risk, Medium Risk, High Risk)
    - key_risk_factors (defensible explanations)
    - recommended_retention_actions
    """
    if not model_service.is_ready:
        logger.error("Predict endpoint called but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is currently unavailable. Ensure training artifacts exist.",
        )

    try:
        logger.info(f"Received prediction request for tenure={customer.tenure}, contract={customer.Contract}")
        response = model_service.predict(customer)
        return response
    except Exception as e:
        logger.exception(f"Error during churn prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error encountered: {str(e)}",
        )


@app.get("/metadata", tags=["Model"])
def get_metadata():
    """Return model evaluation metrics and feature importances."""
    return {
        "best_model": model_service.model_name,
        "metrics": model_service.metrics.get("metrics", {}),
        "top_features": model_service.metrics.get("top_features", []),
    }
