#!/bin/sh
set -e

# Support selecting individual service via CLI argument
if [ "$1" = "api" ]; then
    echo "Starting FastAPI backend service on port 8000..."
    exec uvicorn app.api:app --host 0.0.0.0 --port 8000
elif [ "$1" = "ui" ]; then
    echo "Starting Streamlit frontend service on port 8501..."
    exec streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0
elif [ "$1" = "train" ]; then
    echo "Running model training pipeline..."
    exec python scripts/train.py
elif [ "$1" = "test" ]; then
    echo "Running automated test suite..."
    exec pytest -v
else
    echo "Starting combined services (FastAPI on 8000 + Streamlit on 8501)..."
    uvicorn app.api:app --host 0.0.0.0 --port 8000 &
    API_PID=$!

    # Wait for API to initialize
    sleep 3

    streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0 &
    UI_PID=$!

    # Wait on processes
    trap "kill $API_PID $UI_PID; exit 0" SIGINT SIGTERM
    wait -n $API_PID $UI_PID
fi
