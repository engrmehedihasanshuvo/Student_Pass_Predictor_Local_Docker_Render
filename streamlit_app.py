from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/student_pass_model.pkl")
MODEL_INFO_PATH = Path("models/model_info.json")
FEATURES = ["study_hours", "attendance", "assignments_completed"]


def load_model():
    if not MODEL_PATH.exists():
        st.error("Model not found. Run train_and_export.py first.")
        st.stop()

    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


def load_model_version() -> str:
    if not MODEL_INFO_PATH.exists():
        return "unknown"

    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    return str(metadata.get("active_version", "unknown"))


st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓", layout="centered")
st.title("Student Pass Predictor - Streamlit")

model = load_model()
model_version = load_model_version()
st.caption(f"Model version: {model_version}")

study_hours = st.slider("Study hours per day", min_value=0.0, max_value=24.0, value=4.5, step=0.1)
attendance = st.slider("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
assignments_completed = st.slider(
    "Assignments completed", min_value=0, max_value=20, value=7, step=1
)

if st.button("Predict"):
    sample = pd.DataFrame(
        [
            {
                "study_hours": float(study_hours),
                "attendance": float(attendance),
                "assignments_completed": float(assignments_completed),
            }
        ],
        columns=FEATURES,
    )

    prediction = int(model.predict(sample)[0])
    pass_probability = float(model.predict_proba(sample)[0][1])

    st.success(f"Prediction: {'Pass' if prediction == 1 else 'Fail'}")
    st.write(f"Pass probability: {pass_probability:.3f}")
