FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

# App code
COPY . .

# Python deps
RUN pip install --no-cache-dir \
    "ccxt>=4.0" "pydantic>=2.0" "pyyaml>=6.0" "python-dotenv>=1.0" \
    "pandas>=2.0" "numpy>=1.24" "sqlalchemy>=2.0" "aiosqlite>=0.19" \
    "orjson>=3.9" "scikit-learn>=1.3" "joblib>=1.3" \
    "fastapi>=0.110" "uvicorn[standard]>=0.27" "jinja2>=3.1" "python-multipart>=0.0.6"

# Create data dir for SQLite
RUN mkdir -p /app/data

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "scripts/run_web.py"]
