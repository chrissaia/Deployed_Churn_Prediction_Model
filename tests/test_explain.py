from unittest.mock import patch
from tests.conftest import client, valid_payload


def test_explain_returns_success_when_llm_returns_text(client):
    explain_payload = {
        "input_data": valid_payload(),
        "proba": [[0.12, 0.88]],
        "result": "Likely to churn",
        "top_features": [
            {"feature": "Contract_Month-to-month", "value": 1, "importance": 0.4}
        ],
    }

    with patch("src.app.main.llm_prediction_explanation", return_value="Customer is high risk."):
        response = client.post("/explain", json=explain_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["llm_call_succeeded"] is True
    assert body["llm_call_explanation"] == "Customer is high risk."


def test_explain_returns_failure_when_llm_raises(client):
    explain_payload = {
        "input_data": valid_payload(),
        "proba": [[0.12, 0.88]],
        "result": "Likely to churn",
        "top_features": [
            {"feature": "Contract_Month-to-month", "value": 1, "importance": 0.4}
        ],
    }

    with patch("src.app.main.llm_prediction_explanation", side_effect=Exception("llm blew up")):
        response = client.post("/explain", json=explain_payload)

    assert response.status_code == 500


def test_explain_rejects_bad_payload(client):
    response = client.post("/explain", json={"foo": "bar"})
    assert response.status_code == 422