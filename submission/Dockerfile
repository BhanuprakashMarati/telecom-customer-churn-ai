# Production Dockerfile for Telecom Customer Churn Prediction System
# Supports running FastAPI backend (Port 8000) and Streamlit UI (Port 8501)

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/app/.mplcache

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and artifacts
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY model/ ./model/
COPY data/ ./data/
COPY entrypoint.sh .

# Ensure cache directories and execute permissions
RUN mkdir -p /app/.mplcache && chmod +x entrypoint.sh

# Expose FastAPI backend and Streamlit frontend ports
EXPOSE 8000 8501

# Healthcheck for container stability
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint starts both services concurrently
ENTRYPOINT ["./entrypoint.sh"]
