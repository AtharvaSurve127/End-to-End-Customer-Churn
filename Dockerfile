FROM python:3.12-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# The API needs the feature code, the UI, and the trained artifacts
# (run Scripts/run_pipeline.py before building so artifacts/ exists).
COPY src/features src/features
COPY src/api src/api
COPY artifacts artifacts

ENV PREDICTION_LOG=/app/logs/predictions.jsonl
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
