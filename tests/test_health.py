from tests.conftest import client


def test_root_returns_hello(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "hello"}


def test_health_returns_expected_shape(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "n_features" in body
    assert "first_3" in body
    assert isinstance(body["first_3"], list)