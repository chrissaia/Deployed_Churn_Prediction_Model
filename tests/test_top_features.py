import pandas as pd


class DummyModel:
    feature_importances_ = [0.9, 0.1, 0.0]


def test_get_top_features_returns_ranked_active_features(monkeypatch):
    from src.serving import inference

    monkeypatch.setattr(inference, "model", DummyModel())

    df = pd.DataFrame([{
        "feature_a": 1,
        "feature_b": 2,
        "feature_c": 0,
    }])

    result = inference._get_top_features(df, top_n=2)

    assert len(result) == 2
    assert result[0]["feature"] == "feature_a"
    assert "importance" in result[0]


def test_get_top_features_falls_back_when_no_importances(monkeypatch):
    from src.serving import inference

    class NoImportanceModel:
        pass

    monkeypatch.setattr(inference, "model", NoImportanceModel())

    df = pd.DataFrame([{
        "feature_a": 1,
        "feature_b": 0,
        "feature_c": 3,
    }])

    result = inference._get_top_features(df, top_n=5)

    assert len(result) == 2
    assert result[0]["feature"] == "feature_a"