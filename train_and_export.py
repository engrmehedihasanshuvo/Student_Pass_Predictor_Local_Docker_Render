from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/dummy_students.csv")
MODEL_PATH = Path("models/student_pass_model.pkl")
MODEL_INFO_PATH = Path("models/model_info.json")
FEATURES = ["study_hours", "attendance", "assignments_completed"]


def generate_dummy_dataset(path: Path, n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    study_hours = rng.uniform(0.5, 8.0, size=n_rows)
    attendance = rng.integers(45, 100, size=n_rows)
    assignments_completed = rng.integers(0, 11, size=n_rows)

    score = (
        0.5 * study_hours
        + 0.03 * attendance
        + 0.25 * assignments_completed
        + rng.normal(0, 0.7, size=n_rows)
    )
    pass_exam = (score >= 5.5).astype(int)

    df = pd.DataFrame(
        {
            "study_hours": study_hours.round(2),
            "attendance": attendance,
            "assignments_completed": assignments_completed,
            "pass_exam": pass_exam,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dummy_dataset(DATA_PATH)

    target = "pass_exam"

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[target], test_size=0.2, random_state=42, stratify=df[target]
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_version = f"v{timestamp}"
    versioned_model_path = MODEL_PATH.parent / f"student_pass_model_{model_version}.pkl"

    with open(versioned_model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    model_info = {
        "active_version": model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accuracy": round(float(accuracy), 3),
        "features": FEATURES,
        "model_file": str(MODEL_PATH),
        "versioned_model_file": str(versioned_model_path),
    }
    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as metadata_file:
        json.dump(model_info, metadata_file, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved versioned model to {versioned_model_path}")
    print(f"Saved model metadata to {MODEL_INFO_PATH}")


if __name__ == "__main__":
    main()
