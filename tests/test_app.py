from __future__ import annotations

from app import app as flask_app
import app as app_module


class DummyModel:
    def predict(self, df):
        return [1]

    def predict_proba(self, df):
        return [[0.1, 0.9]]


def test_predict_missing_field_returns_400():
    client = flask_app.test_client()

    response = client.post(
        "/predict",
        json={
            "study_hours": 5,
            "attendance": 80,
        },
    )

    assert response.status_code == 400


def test_predict_invalid_range_returns_400():
    client = flask_app.test_client()

    response = client.post(
        "/predict",
        json={
            "study_hours": 5,
            "attendance": 140,
            "assignments_completed": 8,
        },
    )

    assert response.status_code == 400


def test_predict_success_returns_prediction(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL", DummyModel())
    monkeypatch.setattr(app_module, "MODEL_VERSION", "v-test")

    client = flask_app.test_client()
    response = client.post(
        "/predict",
        json={
            "study_hours": 4.5,
            "attendance": 85,
            "assignments_completed": 7,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] in [0, 1]
    assert payload["prediction_label"] in ["Pass", "Fail"]
    assert "model_version" in payload
