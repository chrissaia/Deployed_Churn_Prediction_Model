import os
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.observability.tracing import setup_tracing
import gradio as gr
import litellm


from src.serving.inference import predict, llm_prediction_explanation
import json


def configure_langfuse():
    """
    Configure LiteLLM -> Langfuse tracing via environment variables.
    Returns a small status dict for debugging.
    """
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_otel_host = os.getenv("LANGFUSE_BASE_URL")

    configured = all([langfuse_public_key, langfuse_secret_key, langfuse_otel_host])


    if configured:
        # LiteLLM reads these from env
        litellm.callbacks = ["langfuse_otel"]
        return {
            "langfuse_enabled": True,
            "langfuse_host": langfuse_otel_host,
        }

    litellm.callbacks = []
    return {
        "langfuse_enabled": False,
        "langfuse_host": None,
    }




def load_feature_columns():
    with open("artifacts/feature_columns.json") as f:
        return json.load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.feature_columns = load_feature_columns()
    app.state.observability = configure_langfuse()
    yield
    app.state.clear()

app = FastAPI(
    title="Telco Churn Model",
    description="Building a model that predicts whether a person is churn or not",
    version="0.0.1",
    lifespan=lifespan,
)





# check root health
@app.get("/")
def root():
    return {"message": "hello"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "n_features": len(app.state.feature_columns),
        "first_3": app.state.feature_columns[:3],
        "langfuse_enabled": app.state.observability["langfuse_enabled"],
        "langfuse_host": app.state.observability["langfuse_host"],
    }


# === REQUEST DATA SCHEMA ===
class ChurnData(BaseModel):
    """
    Customer data schema for churn prediction.

    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """

    # Demographics
    gender: str  # "Male" or "Female"
    Partner: str  # "Yes" or "No"
    Dependents: str  # "Yes" or "No"

    # Phone services
    PhoneService: str  # "Yes" or "No"
    MultipleLines: str  # "Yes", "No", or "No phone service"

    # Internet services
    InternetService: str  # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str  # "Yes", "No", or "No internet service"
    OnlineBackup: str  # "Yes", "No", or "No internet service"
    DeviceProtection: str  # "Yes", "No", or "No internet service"
    TechSupport: str  # "Yes", "No", or "No internet service"
    StreamingTV: str  # "Yes", "No", or "No internet service"
    StreamingMovies: str  # "Yes", "No", or "No internet service"

    # Account information
    Contract: str  # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str  # "Yes" or "No"
    PaymentMethod: str  # "Electronic check", "Mailed check", etc.

    # Numeric features
    tenure: int  # Number of months with company
    MonthlyCharges: float  # Monthly charges in dollars
    TotalCharges: float  # Total charges to date


class LLMVars(BaseModel):
    """
    llm call variables needed to execute call.
    """
    input_data: dict
    proba: list
    result: str
    top_features: list[dict]



@app.post("/predict")
def get_prediction(data: ChurnData):
    """
    Main prediction endpoint for customer churn prediction.

    Receives validated customer data from Basemodel
    Calls the inference pipeline to transform features and predict
    Returns churn prediction in JSON format

    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result, vars = predict(data.model_dump())

        return {
            "prediction": result,
            "llm_context": {
                "input_data": vars[0],
                "proba": vars[1],
                "result": result,
                "top_features": vars[2],
            }
        }
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}



@app.post("/explain")
def get_explanation(data: LLMVars):
    """
    Main llm call endpoint for customer explanation.

    Receives prediction data from /predict endpoint
    Calls the inference pipeline to get explanation
    Returns explanation as a string

    """
    try:
        # call inference explanation function
        explanation = llm_prediction_explanation(data.input_data, data.proba, data.result, data.top_features)
        return {
            "llm_call_explanation": explanation,
            "llm_call_succeeded": True
        }
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {
            "llm_call_errpr": str(e),
            "llm_call_succeeded": False
        }



# ---------------------------------------------------
# Gradio Web interface
with gr.Blocks() as demo:

    gr.Markdown("""
    # Telco Customer Churn Predictor

    Fill in the customer details below to get a churn prediction.
    The explanation will generate after the prediction.
    """)

    # -------------------------
    # INPUTS
    # -------------------------
    with gr.Row():
        with gr.Column():

            gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
            Partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
            Dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
            PhoneService = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")

            MultipleLines = gr.Dropdown(
                ["Yes", "No", "No phone service"],
                label="Multiple Lines",
                value="No"
            )

            InternetService = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                label="Internet Service",
                value="Fiber optic"
            )

            OnlineSecurity = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Security",
                value="No"
            )

            OnlineBackup = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Backup",
                value="No"
            )

            DeviceProtection = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Device Protection",
                value="No"
            )

            TechSupport = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Tech Support",
                value="No"
            )

            StreamingTV = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming TV",
                value="Yes"
            )

            StreamingMovies = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming Movies",
                value="Yes"
            )

            Contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                label="Contract",
                value="Month-to-month"
            )

            PaperlessBilling = gr.Dropdown(
                ["Yes", "No"],
                label="Paperless Billing",
                value="Yes"
            )

            PaymentMethod = gr.Dropdown(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                label="Payment Method",
                value="Electronic check"
            )

            tenure = gr.Number(label="Tenure (months)", value=1)
            MonthlyCharges = gr.Number(label="Monthly Charges ($)", value=85.0)
            TotalCharges = gr.Number(label="Total Charges ($)", value=85.0)

            generate_explanation = gr.Checkbox(
                label="Generate LLM Explanation",
                value=False
            )


    # -------------------------
    # OUTPUTS
    # -------------------------
    prediction_output = gr.Textbox(label="Churn Prediction")
    explanation_output = gr.Textbox(label="LLM Explanation", lines=20)

    # hidden state to pass vars between steps
    state = gr.State(value={})


    # -------------------------
    # FUNCTIONS
    # -------------------------
    def gradio_predict(
        gender, Partner, Dependents, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies, Contract,
        PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges,
        generate_explanation
    ):
        data = {
            "gender": gender,
            "Partner": Partner,
            "Dependents": Dependents,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "tenure": int(tenure),
            "MonthlyCharges": float(MonthlyCharges),
            "TotalCharges": float(TotalCharges),
        }

        result, vars = predict(data)

        llm_context = {
            "input_data": vars[0],
            "proba": vars[1],
            "result": result,
            "top_features": vars[2],
            "generate_explanation": generate_explanation,
        }

        initial_explanation = "Generating explanation..." if generate_explanation else "LLM explanation skipped."

        return str(result), initial_explanation, llm_context

    def gradio_explain(llm_context):
        if not llm_context:
            return "No prediction context found."

        if not llm_context.get("generate_explanation", False):
            return "LLM explanation skipped."

        explanation = llm_prediction_explanation(
            llm_context["input_data"],
            llm_context["proba"],
            llm_context["result"],
            llm_context["top_features"]
        )
        return explanation

    # -------------------------
    # BUTTONS
    # -------------------------
    predict_btn = gr.Button("Predict")

    # -------------------------
    # EVENTS
    # -------------------------
    predict_event = predict_btn.click(
        fn=gradio_predict,
        inputs=[
            gender, Partner, Dependents, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies, Contract,
            PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges,
            generate_explanation
        ],
        outputs=[prediction_output, explanation_output, state]
    )

    predict_event.then(
        fn=gradio_explain,
        inputs=state,
        outputs=explanation_output
    )

    # -------------------------
    # EXAMPLES
    # -------------------------
    gr.Examples(
        examples=[
            [
                "Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No",
                "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",
                1, 85.0, 85.0, False
            ],
            [
                "Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
                "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
                60, 45.0, 2700.0, True
            ]
        ],
        inputs=[
            gender, Partner, Dependents, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies, Contract,
            PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges, generate_explanation
        ],
        fn=gradio_predict,
        outputs=[prediction_output, explanation_output, state],
        cache_examples=False,
        label="Try example customers"
    )


# Mounting gradio onto fastapi
# creates the /ui endpoint that serves the Gradio interface
app = gr.mount_gradio_app(
    app,  # FastAPI application instance
    demo,  # Gradio interface
    path="/ui"  # URL path where Gradio will be accessible
)