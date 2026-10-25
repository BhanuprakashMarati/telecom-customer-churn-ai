# 🎥 Telecom Customer Churn Intelligence System
## Demonstration Video Recording Protocol & Verification Checklist

This protocol provides a step-by-step recording guide to create a professional 3–5 minute demonstration video for project submission and evaluation.

---

## 📋 Pre-Recording Setup Checklist

- [ ] **System Services Running:**
  - FastAPI Backend active on `http://127.0.0.1:8000`
  - Streamlit Frontend active on `http://localhost:8501`
- [ ] **Browser Tabs Prepared:**
  - Tab 1: Streamlit Dashboard (`http://localhost:8501`)
  - Tab 2: FastAPI Interactive Docs (`http://127.0.0.1:8000/docs`)
  - Tab 3: GitHub Repository / README page
- [ ] **Terminal Prepared:**
  - Clean terminal window inside project root with `.venv` activated
- [ ] **Audio & Display Settings:**
  - Resolution set to 1080p (1920x1080)
  - Microphone tested for clear audio with minimal background noise
  - Browser zoom set to 100% or 110% for crisp readability

---

## ⏱️ Video Script & Segment Breakdown (Target: 3–5 Minutes)

### 1. Introduction & Executive Overview (0:00 – 0:40)
- [ ] State your name and project title: *"Telecom Customer Churn Intelligence System"*.
- [ ] Explain the business challenge: High customer acquisition costs ($CAC$) vs customer retention economics ($CRC$) in subscription telecoms.
- [ ] Summarize the technical stack: Python, Scikit-Learn, FastAPI, Streamlit, Pytest, and Docker.

### 2. Backend API Microservice & OpenAPI Docs (0:40 – 1:30)
- [ ] Navigate to `http://127.0.0.1:8000/docs`.
- [ ] Demonstrate `GET /health`:
  - Execute live and show `status: "healthy"`, `model_loaded: true`, and model metadata.
- [ ] Demonstrate `POST /predict`:
  - Highlight the Pydantic schema validation covering all 19 customer features.
  - Execute a test request showing structured response with `churn_probability`, `risk_category`, `key_risk_factors`, and `disclaimer`.

### 3. Streamlit Dashboard — Single Customer Assessment (1:30 – 2:45)
- [ ] Switch to `http://localhost:8501`.
- [ ] Show the green *"Backend API Connected (Healthy)"* indicator in the sidebar.
- [ ] **Profile 1: High-Risk Customer Demo:**
  - Click the *"High Risk Customer"* preset from the sidebar.
  - Review customer details (Month-to-month, tenure=2, Fiber optic, no tech support, electronic check, $89.20/mo).
  - Click *"Evaluate Customer Churn Risk"*.
  - Point out the **~86% churn probability**, the **Red CRITICAL Badge**, and the **Governance Disclaimer**.
  - Review the **Defensible Key Risk Drivers** (early lifecycle, month-to-month, lack of tech support).
  - Review the **Targeted Retention Actions** (1-year renewal with 15% discount, complimentary tech support).
- [ ] **Profile 2: Low-Risk Loyal Customer Demo:**
  - Click *"Low Risk Loyal Customer"* preset.
  - Click *"Evaluate Customer Churn Risk"*.
  - Point out the **Low Risk Tier** (~3% churn probability) and long-term contract stability.

### 4. Ethical AI Governance & Disclaimer Highlight (2:45 – 3:15)
- [ ] Emphasize the prominent statistical disclaimer:
  > *"Never present predictions as certainties: Predictions represent probabilistic estimates based on historical statistical patterns, not guaranteed individual outcomes."*
- [ ] Explain how this prevents deterministic bias in customer relationship management.

### 5. Model Benchmarks & Batch Processing (3:15 – 4:00)
- [ ] Navigate to Tab 2 (*Model Performance & Architecture*):
  - Show test set benchmark cards: **ROC-AUC: 0.8423**, **Recall: 0.7941**, **F1: 0.6266**.
  - Show the Confusion Matrix and ROC Curve comparison plots.
  - Explain why **Recall** and **ROC-AUC** are the primary business optimization metrics.
- [ ] Navigate to Tab 3 (*Batch CSV Scoring*):
  - Briefly demonstrate batch evaluation of customer records.

### 6. Terminal Test Suite & Containerization (4:00 – 4:30)
- [ ] Switch to terminal window.
- [ ] Run `python -m pytest -v`.
- [ ] Show all 11 automated unit and integration tests passing green.
- [ ] Mention Docker containerization support (`docker-compose up` or `Dockerfile`).
- [ ] Conclude with summary of deliverables.

---

## 📦 Post-Recording Verification Checklist

- [ ] Video playback verified: audio is synchronized and clear.
- [ ] Video file uploaded to hosting platform (YouTube unlisted, Google Drive with public access, or Loom).
- [ ] Shareable link copied and pasted into `submission/LINKS.txt`.
- [ ] Video length confirmed between 3 and 5 minutes.
