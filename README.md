# Telco Customer Churn Prediction System

A production oriented machine learning system that predicts customer churn and provides structured explanations for each prediction. The system includes a full pipeline from raw data processing through model training, inference, and API serving.

---

## Overview

This project delivers an end to end workflow for churn prediction using the Telco Customer Churn dataset. It ensures consistency between training and inference, validates incoming data, and exposes a FastAPI service for real time predictions.

The system is designed with a focus on reliability, reproducibility, and observability.

---

## Architecture

The repository is organized around a modular pipeline:

- Data layer handles ingestion and preprocessing of raw datasets
- Feature layer builds model ready features
- Model layer supports training, tuning, and evaluation
- Serving layer provides inference and explanation endpoints
- Observability layer integrates tracing for request level visibility

The API layer sits on top of this pipeline and exposes endpoints for prediction and explanation.

---

## Key Capabilities

- Deterministic feature engineering between training and inference
- Model training with XGBoost and experiment tracking via MLflow
- Real time inference through FastAPI
- Natural language explanations using an LLM
- Request validation using Pydantic
- Dataset validation using Pandera
- Distributed tracing with OpenTelemetry and Google Cloud Trace
- Containerized deployment with Docker

---

## Project Structure

```

.
├── data/
│   ├── raw/
│   ├── processed/
├── artifacts/
│   └── feature_columns.json
├── src/
│   ├── app/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── serving/
│   ├── observability/
│   └── utils/
├── scripts/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md

````

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/chrissaia/Customer_Churn_Prediction_EndtoEnd
cd Customer_Churn_Prediction_EndtoEnd

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-dev.txt
# pip install -e .

````

---

## Running the Pipeline

Prepare processed data:

```bash
python scripts/prepare_processed_data.py
```

Run the full training pipeline:

```bash
python scripts/run_pipeline.py
```

Artifacts including the trained model and feature schema are saved locally and tracked in MLflow.

---

## Running the API Locally

Start the FastAPI server:

```bash
uvicorn src.app.main:app --reload
```

Access endpoints:

* Health check: [http://localhost:8000/health](http://localhost:8000/health)
* Prediction: [http://localhost:8000/predict](http://localhost:8000/predict)
* Explanation: [http://localhost:8000/explain](http://localhost:8000/explain)

---

## Docker

Build the container:

```bash
docker build -t churn-app .
```

Run the container:

```bash
docker run -p 8080:8080 churn-app
```

---

## API Usage

### Prediction

```json
POST /predict
```

Request body:

```json
{
  "gender": "Male",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure": 12,
  "MonthlyCharges": 85.5,
  "TotalCharges": 1026.0
}
```

Response:

```json
{
  "prediction": "Likely to churn",
  "llm_context": {
    "input_data": {},
    "proba": [],
    "result": "",
    "top_features": []
  }
}
```

---

### Explanation

```json
POST /explain
```

Generates a structured explanation based on prediction output.

---

## Testing

Run tests:

```bash
pytest -q
```

Tests cover:

* API endpoints
* request validation
* inference behavior
* error handling

---

## Validation Strategy

Two levels of validation are used:

### Request validation

Handled by Pydantic in the API layer. Ensures correct types, required fields, and allowed values.

### Dataset validation

Handled by Pandera in the pipeline. Ensures schema integrity, numeric bounds, and business logic constraints before training.

---

## Observability

Tracing is implemented using OpenTelemetry and exported to Google Cloud Trace.

Each request can be tracked end to end, including model inference and LLM calls.

---

## Environment Variables

Key variables:

```
MODEL_DIR=/app/model
LANGFUSE_PUBLIC_KEY=<key>
LANGFUSE_SECRET_KEY=<key>
LANGFUSE_OTEL_HOST=<url>
```

---

## Deployment

The system is designed for deployment on container based platforms such as Google Cloud Run.

Steps:

1. Build container
2. Push to container registry
3. Deploy service
4. Configure environment variables
5. Enable tracing and logging

---

## Future Improvements

* Model version selection via environment configuration
* Batch prediction endpoint
* Structured logging for production debugging
* CI pipeline with automated testing
* Performance monitoring and alerting

---

## License

....

```

## Other Tips

### Creating similar folder structure
```
mkdir -p \
	data/raw data/processed data/external \
	notebooks \
	src/{data,features,models,utils} \
	app \
	configs \
	scripts \
	tests \
	.github/workflows \
	docker \
	data-validation \
	mlruns \
	artifacts
```

### Basic ML imports 
```commandline
cat > requirements.txt << 'EOF'
pandas
numpy
scikit-learn
mlflow
fastapi
uvicorn
pydantic
python-dotenv
joblib
pytest
pytest
EOF
```
