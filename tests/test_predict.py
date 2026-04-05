from tests.conftest import client, valid_payload


def test_predict_returns_prediction_and_llm_context(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert body["prediction"] in ["Likely to churn", "Not likely to churn"]

    assert "llm_context" in body
    assert "input_data" in body["llm_context"]
    assert "proba" in body["llm_context"]
    assert "result" in body["llm_context"]
    assert "top_features" in body["llm_context"]


def test_predict_rejects_missing_required_field(client):
    payload = valid_payload()
    payload.pop("Contract")

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_wrong_type_for_tenure(client):
    payload = valid_payload()
    payload["tenure"] = "abc"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422