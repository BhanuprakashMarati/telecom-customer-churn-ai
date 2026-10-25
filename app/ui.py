"""
Streamlit Web Application: Telecom Customer Churn Intelligence System.
Communicates with FastAPI backend with fallback to direct model inference.
"""
import io
import json
import os
import sys

# Ensure repository root and app directory are on sys.path for robust imports across environments (e.g. Streamlit Cloud)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import requests
import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px;
        border-radius: 4px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 0.9rem;
        color: #1E40AF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/telecom.png", width=64)
st.sidebar.title("System Control")
api_url = st.sidebar.text_input("FastAPI Backend URL", value="http://127.0.0.1:8000")

# Backend Health Check
backend_healthy = False
try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
        backend_healthy = True
        st.sidebar.success("Backend API Connected (Healthy)")
    else:
        st.sidebar.warning("Backend API Reachable but Degraded")
except Exception:
    st.sidebar.info("API disconnected; fallback local mode available")

# Demo Presets
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Customer Presets")

preset_choice = st.sidebar.radio(
    "Load Profile Preset",
    ["Custom / Manual Input", "High Risk Customer", "Low Risk Loyal Customer", "Moderate Risk Borderline"],
)

# Preset Values
defaults = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 171.00,
}

if preset_choice == "High Risk Customer":
    defaults.update({
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.20,
        "TotalCharges": 178.40,
    })
elif preset_choice == "Low Risk Loyal Customer":
    defaults.update({
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 55.40,
        "TotalCharges": 3601.00,
    })
elif preset_choice == "Moderate Risk Borderline":
    defaults.update({
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 16,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 74.00,
        "TotalCharges": 1184.00,
    })

# Main Header
st.markdown('<div class="main-header">Telecom Customer Churn Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production-grade predictive risk stratification, defensible driver analytics, and actionable retention recommendations.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Single Customer Assessment", "Model Performance & Architecture", "Batch CSV Scoring"])

with tab1:
    st.subheader("Customer Profile & Contract Details")
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographic Information")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if defaults["gender"] == "Female" else 1)
            senior = st.selectbox("Senior Citizen", [0, 1], index=int(defaults["SeniorCitizen"]))
            partner = st.selectbox("Partner", ["No", "Yes"], index=0 if defaults["Partner"] == "No" else 1)
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if defaults["Dependents"] == "No" else 1)
            tenure = st.slider("Tenure (Months Subscribed)", min_value=0, max_value=72, value=int(defaults["tenure"]))

        with col2:
            st.markdown("##### 📡 Subscriptions & Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if defaults["PhoneService"] == "Yes" else 1)
            lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(defaults["MultipleLines"]))
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(defaults["InternetService"]))
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineSecurity"]))
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineBackup"]))
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["DeviceProtection"]))
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["TechSupport"]))
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingTV"]))
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingMovies"]))

        with col3:
            st.markdown("##### 💳 Contract & Invoicing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if defaults["PaperlessBilling"] == "Yes" else 1)
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ].index(defaults["PaymentMethod"]),
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=15.0, max_value=200.0, value=float(defaults["MonthlyCharges"]), step=0.5
            )
            calc_total = round(float(tenure * monthly_charges), 2)
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(defaults["TotalCharges"] or calc_total), step=10.0
            )

        submit_btn = st.form_submit_button("🔍 Evaluate Customer Churn Risk", use_container_width=True)

    if submit_btn:
        payload = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        result = None
        # Try API First
        if backend_healthy:
            try:
                res = requests.post(f"{api_url}/predict", json=payload, timeout=5)
                if res.status_code == 200:
                    result = res.json()
            except Exception as e:
                st.warning(f"API request failed ({e}), falling back to local model service...")

        # Fallback to local inference if API was not reachable
        if result is None:
            try:
                try:
                    from app.inference import ModelService
                    from app.schemas import CustomerInput
                except ModuleNotFoundError:
                    from inference import ModelService
                    from schemas import CustomerInput

                service = ModelService.get_instance()
                cust_obj = CustomerInput(**payload)
                resp_obj = service.predict(cust_obj)
                result = resp_obj.model_dump()
            except Exception as ex:
                st.error(f"Inference execution failed: {ex}")

        if result:
            st.markdown("---")
            st.subheader("Assessment Findings")

            # KPI Summary
            r_col1, r_col2, r_col3 = st.columns(3)

            prob = result["churn_probability"]
            cat = result["risk_category"]

            with r_col1:
                st.metric("Estimated Churn Probability", result["churn_percentage"])
                st.progress(min(max(prob, 0.0), 1.0))

            with r_col2:
                if cat == "High Risk":
                    badge_html = '<div class="risk-high">CRITICAL: HIGH CHURN RISK</div>'
                elif cat == "Medium Risk":
                    badge_html = '<div class="risk-medium">WARNING: MEDIUM CHURN RISK</div>'
                else:
                    badge_html = '<div class="risk-low">STABLE: LOW CHURN RISK</div>'
                st.markdown(f"**Assigned Risk Tier:**<br>{badge_html}", unsafe_allow_html=True)

            with r_col3:
                prediction_label = "Likely to Churn" if result["churn_prediction"] == 1 else "Likely to Remain"
                st.metric("Statistical Binary Classification", prediction_label)

            # Prominent Disclaimer
            st.markdown(
                f"""
                <div class="disclaimer-box">
                    <strong>⚠️ Important Statistical Governance Notice:</strong><br>
                    {result.get('disclaimer', 'Predictions represent probabilistic estimates based on historical statistical patterns, never certainties.')}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Defensible Explanations & Action Recommendations
            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                st.markdown("##### 📌 Defensible Key Risk Drivers")
                factors = result.get("key_risk_factors", [])
                if factors:
                    for f in factors:
                        icon = "🔺" if "Increases" in f["impact"] else "🔻"
                        st.markdown(f"**{icon} {f['feature']}**  \n*{f['description']}*")
                else:
                    st.info("No dominant risk drivers flagged.")

            with exp_col2:
                st.markdown("##### 🎯 Targeted Retention Interventions")
                actions = result.get("recommended_retention_actions", [])
                if actions:
                    for a in actions:
                        st.markdown(f"- ✅ **{a}**")
                else:
                    st.info("Standard retention workflow applies.")

with tab2:
    st.subheader("Model Validation & Performance Benchmarks")
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            meta = json.load(f)

        best_m = meta.get("best_model", "RandomForest")
        m_data = meta["metrics"][best_m]

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("ROC-AUC Score", f"{m_data['roc_auc']:.4f}")
        m_col2.metric("Recall (Sensitivity)", f"{m_data['recall']:.4f}")
        m_col3.metric("F1-Score", f"{m_data['f1_score']:.4f}")
        m_col4.metric("Precision", f"{m_data['precision']:.4f}")
        m_col5.metric("Accuracy", f"{m_data['accuracy']:.4f}")

        st.markdown(
            """
            > **Why Prioritize Recall & ROC-AUC?** In telecommunications churn management, a False Negative (missing a churning customer) is significantly more expensive than a False Positive (offering a loyalty discount to someone who would have stayed). Our model optimizes for sensitivity while preserving strong discriminant power (ROC-AUC 0.8423).
            """
        )

        img_col1, img_col2 = st.columns(2)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cm_path = os.path.join(base_dir, "model", "confusion_matrix.png")
        roc_path = os.path.join(base_dir, "model", "roc_curve.png")
        fi_path = os.path.join(base_dir, "model", "feature_importance.png")

        if os.path.exists(cm_path):
            img_col1.image(cm_path, caption="Confusion Matrix on Held-Out Test Set (20%)", use_container_width=True)
        if os.path.exists(roc_path):
            img_col2.image(roc_path, caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Global Feature Importance Drivers", use_container_width=True)

with tab3:
    st.subheader("Batch Customer Scoring")
    st.write("Upload a CSV file matching the IBM Telco schema to batch evaluate churn probabilities.")

    uploaded_file = st.file_uploader("Upload Telecom Customer CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows. Sample preview:")
            st.dataframe(batch_df.head(5))

            if st.button("🚀 Process Batch Predictions"):
                try:
                    from app.inference import ModelService
                    from app.schemas import CustomerInput
                except ModuleNotFoundError:
                    from inference import ModelService
                    from schemas import CustomerInput

                service = ModelService.get_instance()

                results = []
                for idx, row in batch_df.iterrows():
                    try:
                        c_dict = row.to_dict()
                        inp = CustomerInput(**c_dict)
                        pred = service.predict(inp)
                        results.append({
                            "Customer_Row": idx + 1,
                            "Churn_Probability": pred.churn_probability,
                            "Risk_Tier": pred.risk_category,
                            "Predicted_Label": "Churn" if pred.churn_prediction == 1 else "Retain",
                        })
                    except Exception as err:
                        results.append({
                            "Customer_Row": idx + 1,
                            "Churn_Probability": None,
                            "Risk_Tier": "Error",
                            "Predicted_Label": str(err),
                        })

                res_df = pd.DataFrame(results)
                combined_df = pd.concat([batch_df.reset_index(drop=True), res_df], axis=1)
                st.success(f"Successfully processed {len(res_df)} customer records!")
                st.dataframe(combined_df.head(10))

                csv_buffer = io.StringIO()
                combined_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Scored Customer Predictions CSV",
                    data=csv_buffer.getvalue(),
                    file_name="telco_churn_batch_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Failed to process CSV file: {e}")
