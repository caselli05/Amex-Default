# Image that serves the LightGBM default model with the Flask app (app.py) behind gunicorn.
# Inference only: training (src/train.py) needs duckdb, mlflow and the parquet data, none of which are in the image.
#
# Build:  docker build -t amex-default .
# Run:    docker run --rm -p 5000:5000 amex-default   ->  http://localhost:5000
#
# The model pkl (src/models/lightgbm.pkl) is git-ignored, so it must exist locally before building.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM's wheel links against OpenMP, which the slim image does not ship
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes don't reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt gunicorn==23.0.0

COPY app.py .
COPY templates/ templates/
COPY src/ src/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json, urllib.request; r = json.load(urllib.request.urlopen('http://localhost:5000/health')); exit(0 if r['model_loaded'] else 1)"

# Each worker loads its own copy of the model (~1 MB); --timeout covers large CSV uploads on /predict_csv
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", \
     "--access-logfile", "-", "app:app"]
