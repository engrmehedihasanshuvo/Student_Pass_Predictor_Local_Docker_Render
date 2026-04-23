from __future__ import annotations

import os
import pickle
from typing import Dict, List

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/student_pass_model.pkl")
FEATURES: List[str] = ["study_hours", "attendance", "assignments_completed"]


with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


def parse_features(payload: Dict[str, float]) -> pd.DataFrame:
    values = {feature: float(payload[feature]) for feature in FEATURES}
    return pd.DataFrame([values])


@app.get("/")
def home() -> str:
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    df = parse_features(payload)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return jsonify(
        {
            "prediction": prediction,
            "prediction_label": "Pass" if prediction == 1 else "Fail",
            "pass_probability": probability,
            "features": payload,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
