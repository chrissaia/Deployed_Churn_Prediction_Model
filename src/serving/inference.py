"""
INFERENCE PIPELINE - Production ML Model Serving ensuring Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import pandas as pd
import mlflow
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from litellm import completion
import json
# -----------------------------------------------------
# This beginning part will run when the script is imported
# It defines the model, loads the list of feature columns and defines the BINARY_MAP and NUMERIC_COLS constants.

# This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
# -----------------------------------------------------

MODEL_DIR = "/app/model"

# Initializes an OpenTelemetry tracer instance, allowing Python code to create manual spans for distributed tracing.
tracer = trace.get_tracer(__name__)


try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.sklearn.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development (OPTIONAL)
    try:
        # Try loading from local MLflow tracking
        import glob

        local_model_paths = (
                glob.glob("./mlruns/*/*/artifacts/model")
                + glob.glob("./src/serving/model/*/artifacts/model")
        )
        if local_model_paths:
            local = True
            for local_model_path in local_model_paths:
                latest_model = max(local_model_paths, key=os.path.getmtime)
                try:
                    model = mlflow.sklearn.load_model(latest_model)
                except Exception as e:
                    local_model_paths.remove(latest_model)
                    continue

                MODEL_DIR = os.path.join(latest_model, "..")
                print(f"Fallback: Loaded model from {latest_model}")

        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# -----------------------------------------------------
# Load the exact feature column order used during training
# This ensures the model receives features in the expected order
# -----------------------------------------------------

try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.json")
    with open(feature_file) as f:
        FEATURE_COLS = json.load(f)
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load model from {MODEL_DIR}: {e}")

# -----------------------------------------------------
# These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance
# -----------------------------------------------------

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},  # Demographics
    "Partner": {"No": 0, "Yes": 1},  # Has partner
    "Dependents": {"No": 0, "Yes": 1},  # Has dependents
    "PhoneService": {"No": 0, "Yes": 1},  # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},  # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.

    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.

    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features
    4. Convert boolean columns to integers
    5. Align features with training schema and order

    Args:
        df: Single-row DataFrame with raw customer data

    Returns:
        DataFrame with features transformed and ordered for model input

    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()

    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()

    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)

    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)  # Convert to string
                .str.strip()  # Remove whitespace
                .map(mapping)  # Apply binary mapping
                .astype("Int64")  # Handle NaN values
                .fillna(0)  # Fill unknown values with 0
                .astype(int)  # Final integer conversion
            )

    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

def _get_top_features(df_enc: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Build a compact top-features payload for the LLM.

    Strategy:
    - If model feature importances are available, rank active row features
      by their global importance.
    - Otherwise, fall back to active non-zero features from the row.

    Returns:
        [
          {"feature": "MonthlyCharges", "value": 85.5, "importance": 0.12},
          ...
        ]
    """

    # -----------------------------------------------------
    # STEP 1: Extract the single row used for prediction
    # This assumes df_enc contains exactly one row
    # -----------------------------------------------------
    row = df_enc.iloc[0]

    # -----------------------------------------------------
    # STEP 2: Attempt to locate the underlying trained model
    # -----------------------------------------------------
    try:
        importances = model.feature_importances_
    except Exception:
        importances = None

    # -----------------------------------------------------
    # STEP 3: Identify active features (non-zero values)
    # These are the only features relevant for THIS prediction
    # -----------------------------------------------------
    active_features = []
    for feature_name, value in row.items():
        if value != 0:
            active_features.append((feature_name, value))

    # -----------------------------------------------------
    # STEP 4: If feature importances exist, rank features
    # by importance to provide stronger signal to the LLM
    # -----------------------------------------------------
    if importances is not None and len(importances) == len(df_enc.columns):
        importance_map = dict(zip(df_enc.columns, importances))
        ranked = []

        for feature_name, value in active_features:
            ranked.append(
                {
                    "feature": feature_name,
                    "value": value.item() if hasattr(value, "item") else value,
                    "importance": float(importance_map.get(feature_name, 0.0)),
                }
            )

        # Sort descending by importance (highest impact first)
        ranked.sort(key=lambda x: x["importance"], reverse=True)
        return ranked[:top_n]

    # -----------------------------------------------------
    # STEP 5: Fallback behavior (no feature importances)
    # Still return meaningful context using active features
    # -----------------------------------------------------
    return [
        {
            "feature": feature_name,
            "value": value.item() if hasattr(value, "item") else value,
        }
        for feature_name, value in active_features[:top_n]
    ]

def llm_prediction_explanation(input_dict, proba, label, top_features):
    """
    Generate a natural-language explanation of the model prediction using an LLM.

    PURPOSE:
    This function translates raw model outputs into a human-readable explanation
    for business users. It does NOT change the prediction — it only explains it.

    DESIGN PRINCIPLES:
    - Ground explanation ONLY in provided input data
    - Never contradict the model prediction (result is source of truth)
    - Keep output concise and business-friendly
    - Avoid hallucinating features not present in input_dict

    INPUTS:
    - input_dict: dict
        Raw customer input data BEFORE feature engineering
        (human-readable fields like Contract, MonthlyCharges, etc.)

    - proba: list or float
        Model predicted probabilities (typically [[p_no_churn, p_churn]])
        Used to communicate confidence level

    - preds: list or int
        Raw model prediction output (usually [0] or [1])

    - result: int
        Final normalized prediction (0 or 1)
        Used to determine business label

    OUTPUT:
    - explanation: str
        Concise explanation (3–5 bullets + short summary)
    """

    # === Normalize probability for readability ===
    # Extract churn probability from model output
    if isinstance(proba, list):
        if isinstance(proba[0], (list, tuple)):
            churn_prob = proba[0][1]
        else:
            churn_prob = proba[0]
    else:
        churn_prob = float(proba)


    # -----------------------------------------------------
    # Prompt construction
    # IMPORTANT:
    # - We explicitly anchor the model output
    # - We prevent hallucination
    # - We guide structure tightly
    # -----------------------------------------------------
    prompt = f"""
        You are an expert churn analyst explaining a machine learning prediction.
        
        Customer data (raw input features):
        {input_dict}
        
        Top features:
        {top_features}
        
        Model output:
        - Churn probability: {round(churn_prob, 4)}
        - Prediction: {label}
        
        STRICT RULES:
        - The prediction is CORRECT. Do not contradict it.
        - Only reference fields explicitly present in the customer data.
        - Do NOT invent features or assumptions.
        - Keep explanations grounded in realistic business reasoning.
        
        TASK:
        1. Provide 3-5 concise bullet points explaining the main drivers of this prediction.
        2. Focus on how the customer's attributes relate to churn risk.
        3. End with a 1-2 sentence business-friendly summary.
        
        OUTPUT FORMAT:
        - Bullet points
        - Then a short summary paragraph
        """

    # === Call LiteLLM SDK directly with OpenAI  ===
    # Uses OPENAI_API_KEY for authentication
    response = completion(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract response safely
    explanation = response["choices"][0]["message"]["content"]

    return explanation


def predict(input_dict: dict) -> tuple[str, tuple[dict, list, list[dict]]]:
    """
    Main prediction function for customer churn inference.

    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.

    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string

    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)

    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)

    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    # Start a span to track the whole root function
    with tracer.start_as_current_span("churn_prediction") as root_span:
        try:
            with tracer.start_as_current_span("business_data") as data_span:
                # === Attach Business Attributes ===

                # Demographics
                data_span.set_attribute("customer.demographics.gender", input_dict.get("gender"))
                data_span.set_attribute("customer.demographics.partner", input_dict.get("Partner"))
                data_span.set_attribute("customer.demographics.dependents", input_dict.get("Dependents"))

                # Phone services
                data_span.set_attribute("customer.phone_services.phone_service", input_dict.get("PhoneService"))
                data_span.set_attribute("customer.phone_services.multiple_lines", input_dict.get("MultipleLines"))

                # Internet services
                data_span.set_attribute("customer.internet_services.internet_service", input_dict.get("InternetService"))
                data_span.set_attribute("customer.internet_services.online_security", input_dict.get("OnlineSecurity"))
                data_span.set_attribute("customer.internet_services.online_backup", input_dict.get("OnlineBackup"))
                data_span.set_attribute("customer.internet_services.device_protection", input_dict.get("DeviceProtection"))
                data_span.set_attribute("customer.internet_services.tech_support", input_dict.get("TechSupport"))
                data_span.set_attribute("customer.internet_services.streamingTV", input_dict.get("StreamingTV"))
                data_span.set_attribute("customer.internet_services.streaming_movies", input_dict.get("StreamingMovies"))

                # Account information
                data_span.set_attribute("customer.account_info.contract", input_dict.get("Contract"))
                data_span.set_attribute("customer.account_info.paperless_billing", input_dict.get("PaperlessBilling"))
                data_span.set_attribute("customer.account_info.payment_method", input_dict.get("PaymentMethod"))

                # Numeric information
                data_span.set_attribute("customer.numeric.tenure", int(input_dict.get("tenure", 0)))
                data_span.set_attribute("customer.numeric.monthly_charges", float(input_dict.get("MonthlyCharges", 0)))
                data_span.set_attribute("customer.numeric.total_charges", float(input_dict.get("TotalCharges", 0)))


            # === STEP 1: Convert Input to DataFrame ===
            # Create single-row DataFrame for pandas transformations
            # Start a span to track this specific function
            df = pd.DataFrame([input_dict])

            # === STEP 2: Apply Feature Transformations ===
            df_enc = _serve_transform(df)


            # === STEP 3: Generate Model Prediction ===
            # Call the loaded MLflow model for inference
            # The model returns predictions in various formats depending on the ML library
            # Start a span to track this specific function
            with tracer.start_as_current_span("model_inference") as model_span:

                preds = model.predict(df_enc)
                proba = model.predict_proba(df_enc)

                # Normalize prediction output to consistent format
                if hasattr(preds, "tolist"):
                    preds = preds.tolist()  # Convert numpy array to list

                if hasattr(proba, "tolist"):
                    proba = proba.tolist()  # Convert numpy array to list


                # Extract single prediction value (for single-row input)
                if isinstance(preds, (list, tuple)) and len(preds) == 1:
                    result = preds[0]
                else:
                    result = preds

                # calculate which risk bucket the customer is in
                risk = float(proba[0][1])
                if risk > 0.7:
                    bucket = "high"
                elif risk > 0.4:
                    bucket = "medium"
                else:
                    bucket = "low"

                # span the raw prediction
                model_span.set_attribute("prediction.risk_bucket", bucket)
                model_span.set_attribute("prediction.probability_churn", risk)


            # === STEP 4: Convert to Business-Friendly Output ===
            # Convert binary prediction (0/1) to actionable business language
            if result == 1:
                label = "Likely to churn"  # High risk - needs intervention
            else:
                label = "Not likely to churn"  # Low risk - maintain normal service

            # raw churn prediction and label
            root_span.set_attribute("label", label)

            # === STEP 5: Build grounded context for the LLM explanation. ===
            # These are the most relevant transformed features for this row.
            # If model importances are unavailable, this falls back to active features.
            top_features = _get_top_features(df_enc, top_n=5)


            root_span.add_event(
                "prediction_completed",
                {
                    "label": label,
                    "probability": float(proba[0][1]),
                    "top_feature_1": top_features[0]["feature"] if top_features else "none"
                }
            )


            # === STEP 6: Mark the overall prediction workflow as successful ===
            # and return both the business label and the explanation.
            # send back the remaining variables for llm call
            root_span.set_attribute("model.version", type(model).__name__)
            root_span.set_status(Status(StatusCode.OK))
            return label, (input_dict, proba, top_features)

        except Exception as e:
            root_span.record_exception(e)
            root_span.set_status(Status(StatusCode.ERROR))
            raise
