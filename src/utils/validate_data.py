import pandas as pd
import pandera.pandas as pa
from pandera import Check
from pandera.errors import SchemaErrors


TELCO_SCHEMA = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, nullable=False),
        "gender": pa.Column(str, checks=Check.isin(["Male", "Female"])),
        "Partner": pa.Column(str, checks=Check.isin(["Yes", "No"])),
        "Dependents": pa.Column(str, checks=Check.isin(["Yes", "No"])),
        "PhoneService": pa.Column(str, checks=Check.isin(["Yes", "No"])),
        "InternetService": pa.Column(str, checks=Check.isin(["DSL", "Fiber optic", "No"])),
        "Contract": pa.Column(str, checks=Check.isin(["Month-to-month", "One year", "Two year"])),
        "tenure": pa.Column(int, checks=[Check.ge(0), Check.le(120)], nullable=False),
        "MonthlyCharges": pa.Column(float, checks=[Check.ge(0), Check.le(200)], nullable=False),
        "TotalCharges": pa.Column(float, checks=Check.ge(0), nullable=True),
    },
    strict=False,
    coerce=True,
    checks=[
        Check(
            lambda df: (df["TotalCharges"] >= df["MonthlyCharges"]) | (df["tenure"] <= 1),
            error="TotalCharges should usually be >= MonthlyCharges unless customer is very new",
        )
    ],
)

def validate_telco_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Data validation for Telco Customer Churn dataset using Pandera.

    This function is extremely important in enterprise ready workflows!

    It implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    """
    try:
        TELCO_SCHEMA.validate(df, lazy=True)
        return True, []
    except SchemaErrors as e:
        failures = e.failure_cases.copy()
        messages = [
            f"{row.get('column', 'dataframe')}: {row.get('failure_case')}"
            for _, row in failures.iterrows()
        ]
        return False, messages