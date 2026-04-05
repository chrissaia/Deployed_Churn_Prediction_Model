import pandas as pd
from src.utils.validate_data import validate_telco_data


def valid_df():
    return pd.DataFrame([{
        "customerID": "1234",
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "tenure": 5,
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0,
    }])


def test_validate_telco_data_passes():
    ok, errors = validate_telco_data(valid_df())
    assert ok is True
    assert errors == []


def test_validate_telco_data_fails_on_missing_column():
    df = valid_df().drop(columns=["customerID"])
    ok, errors = validate_telco_data(df)
    assert ok is False
    assert len(errors) > 0


def test_validate_telco_data_fails_on_invalid_category():
    df = valid_df()
    df.loc[0, "gender"] = "Other"
    ok, errors = validate_telco_data(df)
    assert ok is False
    assert len(errors) > 0


def test_validate_telco_data_fails_on_negative_tenure():
    df = valid_df()
    df.loc[0, "tenure"] = -1
    ok, errors = validate_telco_data(df)
    assert ok is False