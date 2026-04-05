import pandas as pd
from src.serving.inference import _serve_transform, FEATURE_COLS


def make_df(**overrides):
    base = {
        "gender": "Female",
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
        "tenure": 1,
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_serve_transform_returns_expected_columns_in_order():
    df = make_df()
    out = _serve_transform(df)

    assert list(out.columns) == FEATURE_COLS


def test_serve_transform_binary_mapping_is_applied():
    df = make_df(
        gender="Male",
        Partner="Yes",
        Dependents="Yes",
        PhoneService="No",
        PaperlessBilling="No"
    )
    out = _serve_transform(df)

    assert out.shape[0] == 1
    assert list(out.columns) == FEATURE_COLS


def test_serve_transform_handles_unknown_category_without_crashing():
    df = make_df(InternetService="Satellite")
    out = _serve_transform(df)

    assert out.shape[0] == 1
    assert list(out.columns) == FEATURE_COLS


def test_serve_transform_coerces_bad_numeric_values_to_zero():
    df = make_df(tenure="bad", MonthlyCharges="bad", TotalCharges="bad")
    out = _serve_transform(df)

    assert out.shape[0] == 1
    assert list(out.columns) == FEATURE_COLS