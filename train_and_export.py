from __future__ import annotations

import pickle
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

    features = ["study_hours", "attendance", "assignments_completed"]
    target = "pass_exam"

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42, stratify=df[target]
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
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
