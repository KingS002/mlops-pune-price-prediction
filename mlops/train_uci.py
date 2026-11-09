
"""Train and evaluate a Random Forest model for UCI Real Estate data."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_CSV = (
    PROJECT_ROOT / "data" / "processed" / "real_estate_features.csv"
)
TARGET_CSV = (
    PROJECT_ROOT / "data" / "processed" / "real_estate_target.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "uci"
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def train_model() -> dict:
    """Train, evaluate, and save the UCI regression model."""

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Features not found: {FEATURES_CSV}. "
            "Run python -m mlops.build_uci_features first."
        )

    if not TARGET_CSV.exists():
        raise FileNotFoundError(
            f"Target not found: {TARGET_CSV}. "
            "Run python -m mlops.build_uci_features first."
        )

    X = pd.read_csv(FEATURES_CSV)
    y = pd.read_csv(TARGET_CSV)["price_per_unit_area"]

    if len(X) != len(y):
        raise ValueError("Feature and target row counts do not match.")

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Training data contains missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training Random Forest Regressor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "model": "RandomForestRegressor",
        "dataset": "UCI Real Estate Valuation",
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel training completed.")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")

    print("\nEvaluation results:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    return metrics


if __name__ == "__main__":
    train_model()
