"""
Download the official IBM Telco Customer Churn dataset from verified public repositories.
Includes fallback mirrors and integrity verification.
"""
import os
import sys
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-node-red/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 500_000:
        print(f"Dataset already exists at: {DATA_FILE} ({os.path.getsize(DATA_FILE)} bytes)")
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded existing dataset with shape: {df.shape}")
        return DATA_FILE

    print(f"Attempting to download IBM Telco Customer Churn dataset to {DATA_FILE}...")
    success = False

    for url in URLS:
        try:
            print(f"Trying URL: {url} ...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=15) as response, open(DATA_FILE, "wb") as out_file:
                out_file.write(response.read())

            if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 500_000:
                print(f"Successfully downloaded from {url}!")
                success = True
                break
        except Exception as e:
            print(f"Failed from {url}: {e}")

    if not success:
        print("\n[ERROR] Could not download dataset automatically from mirrors.")
        print(f"Please manually place the official 'WA_Fn-UseC_-Telco-Customer-Churn.csv' file into:\n  {DATA_FILE}")
        sys.exit(1)

    # Verification
    df = pd.read_csv(DATA_FILE)
    print(f"Verification successful: shape={df.shape}")
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[WARNING] Missing expected columns: {missing_cols}")
    else:
        print(f"All {len(EXPECTED_COLUMNS)} expected columns verified.")

    return DATA_FILE

if __name__ == "__main__":
    download_dataset()
