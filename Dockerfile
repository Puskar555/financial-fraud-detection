FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# System packages required by common scientific Python wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install only packages required by the production API
COPY requirements.txt .

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    xgboost \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    joblib

# Copy backend application
COPY backend ./backend

# Copy production model artifacts
COPY artifacts/production ./artifacts/production

# Create runtime directory for SQLite
RUN mkdir -p /app/backend/app/db

EXPOSE 8001

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8001"]