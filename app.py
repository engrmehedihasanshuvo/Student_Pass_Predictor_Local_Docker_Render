from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/student_pass_model.pkl")
MODEL_INFO_PATH = os.getenv("MODEL_INFO_PATH", "models/model_info.json")
FEATURES: List[str] = ["study_hours", "attendance", "assignments_completed"]
VALID_RANGES = {
    "study_hours": (0.0, 24.0),
    "attendance": (0.0, 100.0),
    "assignments_completed": (0.0, 20.0),
}


def load_model_bundle() -> Tuple[Any | None, str]:
    model_version = "unknown"

    if os.path.exists(MODEL_INFO_PATH):
        try:
            with open(MODEL_INFO_PATH, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            model_version = str(metadata.get("active_version", model_version))
        except (OSError, json.JSONDecodeError):
            model_version = "unknown"

    if not os.path.exists(MODEL_PATH):
        return None, model_version

    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file), model_version


MODEL, MODEL_VERSION = load_model_bundle()


def parse_features(payload: Dict[str, float]) -> pd.DataFrame:
    values = {feature: float(payload[feature]) for feature in FEATURES}
    return pd.DataFrame([values])


def validate_payload(payload: Any) -> Tuple[bool, Dict[str, float] | None, str | None]:
    if not isinstance(payload, dict):
        return False, None, "Request body must be a JSON object."

    missing = [field for field in FEATURES if field not in payload]
    if missing:
        return False, None, f"Missing required fields: {', '.join(missing)}"

    cleaned: Dict[str, float] = {}
    for field in FEATURES:
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            return False, None, f"Field '{field}' must be numeric."

        min_value, max_value = VALID_RANGES[field]
        if not (min_value <= value <= max_value):
            return (
                False,
                None,
                f"Field '{field}' must be between {min_value} and {max_value}.",
            )

        cleaned[field] = value

    return True, cleaned, None


@app.get("/")
def home() -> str:
    return render_template("index.html", model_version=MODEL_VERSION)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    is_valid, cleaned_payload, error_message = validate_payload(payload)
    if not is_valid:
        return jsonify({"error": error_message}), 400

    if MODEL is None:
        return (
            jsonify(
                {
                    "error": "Model file not found. Run train_and_export.py first.",
                    "expected_path": MODEL_PATH,
                }
            ),
            503,
        )

    df = parse_features(cleaned_payload)

    prediction = int(MODEL.predict(df)[0])
    probability = float(MODEL.predict_proba(df)[0][1])

    return jsonify(
        {
            "prediction": prediction,
            "prediction_label": "Pass" if prediction == 1 else "Fail",
            "pass_probability": probability,
            "model_version": MODEL_VERSION,
            "features": cleaned_payload,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
