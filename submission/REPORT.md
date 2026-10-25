# End-to-End Telecom Customer Churn Prediction & Deployment
## Technical Project Report & Implementation Analysis

---

## Executive Summary

Customer attrition represents one of the largest revenue leaks in the telecommunications industry, where customer acquisition costs ($CAC$) typically exceed retention costs ($CRC$) by a factor of 5 to 7. This project establishes an end-to-end, production-ready machine learning system that ingests telecommunications customer demographics, service contracts, and billing behavior to:
1. Predict customer churn probabilities using calibrated classifiers.
2. Stratify accounts into defensible risk tiers (**Low Risk**, **Medium Risk**, **High Risk**).
3. Automatically diagnose customer-specific churn drivers with explainable rationales.
4. Provide targeted, actionable retention tactics for customer success teams.
5. Expose these capabilities via a high-performance **FastAPI backend** and an intuitive, interactive **Streamlit frontend**.

The selected champion model (**Tuned Random Forest with Balanced Class Weights**) achieves an empirical **ROC-AUC of 0.8423** and a **Recall of 0.7941 (79.4%)** on the held-out test set ($N=1,409$), demonstrating strong discriminant capacity while ensuring that attrition-prone customers are proactively captured.

---

## 1. Problem Formulation & Business Economics

### 1.1 Business Context
In subscription telecommunications services (mobile telephony, fiber optic broadband, streaming entertainment), customers routinely face low switching costs, aggressive competitor promotions, and service friction. Losing an established customer results in:
- Loss of immediate Monthly Recurring Revenue (MRR).
- Forfeiture of Customer Lifetime Value (LTV).
- High sales and onboarding expenses to acquire a replacement subscriber.

### 1.2 Objective Function
The machine learning objective is to model the conditional probability $P(\text{Churn} = 1 \mid X)$, where $X$ represents the vector of customer demographic, service, and contract attributes. 

In attrition analytics, **Recall (Sensitivity)** is prioritized over raw Accuracy. A False Negative (failing to identify an account that subsequently cancels) results in unmitigated churn ($LTV \approx \$1,500 - \$3,000$). Conversely, a False Positive (offering a loyalty discount or outreach call to a subscriber who intended to remain) incurs modest operational cost ($\approx \$15 - \$30$). Therefore, the loss matrix heavily penalizes Type II errors.

---

## 2. Dataset Exploration & Statistical Analysis

### 2.1 Dataset Provenance
The model utilizes the benchmark **IBM Telco Customer Churn** dataset:
- **Total Records:** 7,043 customer accounts.
- **Total Features:** 20 predictor attributes + 1 target variable (`Churn`).
- **Class Balance:** 
  - Retained (`No`): 5,174 records (73.46%)
  - Churned (`Yes`): 1,869 records (26.54%)
  - Native class imbalance ratio: $\approx 2.77 : 1$.

### 2.2 Feature Taxonomy
1. **Demographic Identifiers:** `gender`, `SeniorCitizen` (0/1), `Partner` (Yes/No), `Dependents` (Yes/No).
2. **Account Tenure:** `tenure` (continuous, 0 to 72 months).
3. **Subscribed Services:**
   - Telephony: `PhoneService`, `MultipleLines`.
   - Connectivity: `InternetService` (DSL, Fiber optic, No).
   - Value-Added Add-ons: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
4. **Contractual & Financials:**
   - `Contract` (Month-to-month, One year, Two year).
   - `PaperlessBilling` (Yes/No).
   - `PaymentMethod` (Electronic check, Mailed check, Bank transfer auto, Credit card auto).
   - `MonthlyCharges` (continuous float).
   - `TotalCharges` (continuous float).

### 2.3 Data Hygiene & Preprocessing Transformations
- **Missing Value Rectification:** The raw `TotalCharges` feature contained 11 records with whitespace strings (`" "`) corresponding to new subscribers whose tenure was 0 months. These were coerced to float and imputed with $0.00$.
- **Identifier Removal:** `customerID` was stripped from the training matrix to prevent spurious memorization.
- **Target Encoding:** Binary mapping: $\text{"Yes"} \mapsto 1$, $\text{"No"} \mapsto 0$.
- **Categorical Formatting:** Ensured `SeniorCitizen` was handled as a discrete category for uniform one-hot expansion.

---

## 3. Methodology & Machine Learning Modeling Pipeline

### 3.1 Data Partitioning Strategy
To prevent data leakage, a **Stratified 80/20 Train/Test Split** was executed on the 7,043 records:
- **Training Set:** 5,634 samples (4,139 Retained, 1,495 Churned).
- **Test Set:** 1,409 samples (1,035 Retained, 374 Churned).
- Random state seed fixed at `42` for strict reproducibility.

### 3.2 Preprocessing Pipeline (`ColumnTransformer`)
All feature transformations are encapsulated into a single unified Scikit-Learn `ColumnTransformer` fitted exclusively on the training partition:
1. **Numerical Pipeline:**
   - Features: `tenure`, `MonthlyCharges`, `TotalCharges`.
   - Transformer: `StandardScaler` (zero mean, unit variance).
2. **Categorical Pipeline:**
   - Features: 16 discrete categorical attributes.
   - Transformer: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - Results in 43 encoded indicator columns, producing a 46-dimensional feature space.

### 3.3 Candidate Models & Hyperparameter Design
Two distinct modeling paradigms were benchmarked:
1. **Linear Baseline — Regularized Logistic Regression:**
   - Class Weight: `balanced` (inversely proportional to class frequencies).
   - Regularization: L2 penalty, $C=0.8$, solver `lbfgs`, max iterations 1,000.
   - Serves as an interpretable generalized linear model baseline.
2. **Non-Linear Ensemble — Random Forest Classifier:**
   - Estimators: 250 decision trees.
   - Maximum Depth: 8 levels (prevents deep-tree overfitting).
   - Minimum Samples Split: 6.
   - Class Weight: `balanced`.
   - Multithreading: Deterministic execution (`n_jobs=1` for deployment safety).

---

## 4. Empirical Evaluation Results

The models were evaluated against the held-out test partition ($N = 1,409$). Results are summarized below:

### 4.1 Comparative Model Performance Benchmarks

| Metric | Logistic Regression | Random Forest (Champion) | Variance / Delta |
| :--- | :---: | :---: | :---: |
| **ROC-AUC Score** | 0.8415 | **0.8423** | +0.0008 |
| **Recall (Sensitivity)** | 0.7834 | **0.7941** | **+0.0107 (+1.1%)** |
| **F1-Score** | 0.6136 | **0.6266** | **+0.0130 (+1.3%)** |
| **Precision** | 0.5043 | **0.5174** | +0.0131 |
| **Accuracy** | 0.7381 | **0.7488** | +0.0107 |

### 4.2 Test Partition Confusion Matrix (Random Forest)

$$\begin{pmatrix} \text{True Negatives} & \text{False Positives} \\ \text{False Negatives} & \text{True Positives} \end{pmatrix} = \begin{pmatrix} 758 & 277 \\ 77 & 297 \end{pmatrix}$$

- **True Negatives ($TN = 758$):** Non-churning customers accurately verified.
- **False Positives ($FP = 277$):** Retained customers classified as elevated risk. In business terms, this represents accounts that receive proactive loyalty checks.
- **False Negatives ($FN = 77$):** Churning customers missed by the model. Minimized from 374 total churners to only 77.
- **True Positives ($TP = 297$):** Churning accounts accurately intercepted, enabling immediate retention outreach.

### 4.3 Key Feature Importance Drivers
Empirical analysis of tree impurity reduction revealed the top 5 global drivers of customer churn:
1. **Contract: Month-to-month:** Strongly elevated churn probability. Rolling commitments provide zero barrier to switching carriers.
2. **Tenure:** Inversely correlated with churn. The initial 0–6 months present the highest vulnerability period.
3. **Internet Service: Fiber optic:** Associated with elevated churn when unaccompanied by premium tech support.
4. **Payment Method: Electronic check:** Manual non-automated billing increases churn propensity due to recurring invoice friction.
5. **Total & Monthly Charges:** Price sensitivity spikes when bills exceed \$75/month without perceived service differentiation.

---

## 5. System Architecture & Engineering

The deployment follows a decoupled microservices architecture designed for reliability, scalability, and testability.

### 5.1 Architecture Diagram
```
[ User Browser ]
       │
       ▼
[ Streamlit Frontend (Port 8501) ]
       │  (HTTP / JSON via Requests)
       ▼
[ FastAPI Microservice (Port 8000) ]
       │  (Pydantic Validation)
       ▼
[ ModelService Inference Engine ]
       │  (Joblib Pipeline)
       ▼
[ Preprocessor + Random Forest ]
       │
       ▼
[ Probability + Risk Tier + Factor Rationales ]
```

### 5.2 Key Engineering Modules
- **`app/schemas.py`:** Enforces strict Pydantic schemas on all 19 input parameters, including range bounds and discrete literal validations. Invalid inputs return HTTP 422 with informative validation errors.
- **`app/inference.py`:** Singleton pattern service managing artifact lifecycles. Computes calibrated probabilities, assigns risk tiers, and dynamically derives human-readable risk drivers.
- **`app/api.py`:** FastAPI microservice exposing:
  - `GET /health`: Model readiness, status, feature dimension check.
  - `POST /predict`: Input validation and structured inference response.
  - `GET /metadata`: Performance metrics and top features.
- **`app/ui.py`:** Streamlit dashboard featuring:
  - Interactive single-customer risk evaluation form.
  - Quick-load customer profile presets (High Risk, Low Risk, Medium Risk).
  - Prominent statistical governance disclaimer.
  - Batch CSV customer scoring engine with downloadable results.
  - Embedded model diagnostic charts (Confusion Matrix, ROC Curve, Feature Importances).
- **`Dockerfile` & `docker-compose.yml`:** Container configuration packaging Python 3.11-slim with multi-port orchestration.

---

## 6. Ethical AI, Governance & Probabilistic Risk Stratification

### 6.1 Avoiding Deterministic Absolutism
A core requirement of ethical enterprise AI is never presenting statistical predictions as certainties. Machine learning models produce probabilities conditioned on historical data distributions; they do not dictate individual human behavior.

### 6.2 Implementation Safeguards
1. **Explicit Risk Tiers:** Accounts are grouped into three operational bands:
   - **Low Risk ($P < 0.35$):** Stable accounts; standard engagement.
   - **Medium Risk ($0.35 \le P < 0.65$):** Borderline accounts; eligible for service satisfaction surveys.
   - **High Risk ($P \ge 0.65$):** Vulnerable accounts; prioritized for retention incentives.
2. **Mandatory Disclaimer:** Every API payload and UI assessment screen enforces:
   > *"Predictions represent probabilistic estimates based on historical statistical patterns, not guaranteed individual outcomes."*
3. **Defensible Explanations:** Every prediction is paired with the top underlying operational factors (contract type, tenure band, support coverage) to prevent black-box decisions.

---

## 7. System Limitations & Continuous Improvement

1. **Static Historical Data:** The IBM Telco dataset reflects a specific historical period. Future production iterations should implement streaming data ingestion (e.g., via Kafka) to monitor for covariate shift and concept drift.
2. **Absence of Customer Support Sentiment:** The current dataset relies primarily on demographic and billing fields. Incorporating unstructured customer service interaction logs via NLP would improve early-warning churn detection.
3. **Price Elasticity Modeling:** Current recommendations propose fixed discount percentages (e.g., 15%). Incorporating causal inference / uplift modeling would optimize individual discount allocation to maximize net revenue retention.

---

## 8. Screenshot Capture Guide for Documentation

When compiling visual submission assets, capture the following 6 key screenshots:

1. **Screenshot 1 — High-Risk Customer Assessment (`screenshot_ui_high_risk.png`):**
   - In Streamlit, select the *"High Risk Customer"* preset.
   - Capture the result displaying **~86% Churn Probability**, the **Red CRITICAL Badge**, the **Governance Disclaimer**, and the **Defensible Key Risk Drivers**.
2. **Screenshot 2 — Low-Risk Loyal Customer Assessment (`screenshot_ui_low_risk.png`):**
   - Select the *"Low Risk Loyal Customer"* preset.
   - Capture the result displaying **Low Risk Tier**, green progress meter, and long-term contract protective factors.
3. **Screenshot 3 — Model Performance & Evaluation Tab (`screenshot_ui_benchmarks.png`):**
   - Navigate to Tab 2 (*Model Performance & Architecture*).
   - Capture the 5 KPI metric cards (ROC-AUC 0.8423, Recall 0.7941, etc.) along with the Confusion Matrix and ROC Curve plots.
4. **Screenshot 4 — Batch CSV Customer Scoring (`screenshot_ui_batch.png`):**
   - Navigate to Tab 3 (*Batch CSV Scoring*).
   - Capture the processed preview table with scored churn probabilities and the CSV download button.
5. **Screenshot 5 — Interactive FastAPI Swagger Documentation (`screenshot_api_docs.png`):**
   - Open browser at `http://127.0.0.1:8000/docs`.
   - Expand the `POST /predict` and `GET /health` endpoints showing schema definitions and execution response.
6. **Screenshot 6 — Automated Pytest Test Suite Execution (`screenshot_terminal_tests.png`):**
   - Capture terminal executing `python -m pytest -v` showing all 11 unit/integration tests passing green.

---

## 9. Instructions to Reproduce the Work

### 9.1 Clone & Environment Initialization
```bash
git clone https://github.com/your-username/telecom-customer-churn-ai.git
cd telecom-customer-churn-ai

# Initialize virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 9.2 Data Download & Model Training
```bash
# Verify or download dataset
python scripts/download_data.py

# Train models, evaluate test set, generate plots and artifacts
python scripts/train.py

# Run standalone evaluation
python -m scripts.evaluate
```

### 9.3 Run Automated Test Suite
```bash
python -m pytest -v
```

### 9.4 Launch Local Development Servers
```bash
# Terminal 1: Backend API
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend Dashboard
streamlit run app/ui.py
```

### 9.5 Containerized Execution via Docker
```bash
# Build and run with Docker
docker build -t telecom-churn:latest .
docker run -d -p 8000:8000 -p 8501:8501 --name telecom_app telecom-churn:latest

# Or using Docker Compose
docker-compose up -d --build
```
