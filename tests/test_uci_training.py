
"""Tests for the UCI Real Estate model training pipeline."""

import json

import joblib
import pandas as pd
import pytest

from mlops import train_uci


def test_training_creates_model_and_metrics(monkeypatch, tmp_path):
    """Verify training produces a model and metrics artifacts."""

    features = pd.DataFrame({
        "transaction_date": [2012.5 + i / 100 for i in range(100)],
        "house_age": [float(i % 30) for i in range(100)],
        "distance_to_mrt": [float(100 + i * 10) for i in range(100)],
        "convenience_stores": [float(i % 8) for i in range(100)],
        "latitude": [24.9 + i / 10000 for i in range(100)],
        "longitude": [121.5 + i / 10000 for i in range(100)],
        "transaction_year": [2012 + (i % 2) for i in range(100)],
        "transaction_month": [(i % 12) + 1 for i in range(100)],
        "house_age_squared": [float(i % 30) ** 2 for i in range(100)],
        "distance_to_mrt_log": [
            __import__("math").log1p(100 + i * 10)
            for i in range(100)
        ],
    })

    target = pd.DataFrame({
        "price_per_unit_area": [
            20.0 + (i * 0.5) + (i % 7) for i in range(100)
        ]
    })

    features_path = tmp_path / "features.csv"
    target_path = tmp_path / "target.csv"
    model_dir = tmp_path / "models"

    features.to_csv(features_path, index=False)
    target.to_csv(target_path, index=False)

    monkeypatch.setattr(train_uci, "FEATURES_CSV", features_path)
    monkeypatch.setattr(train_uci, "TARGET_CSV", target_path)
    monkeypatch.setattr(train_uci, "MODEL_DIR", model_dir)
    monkeypatch.setattr(
        train_uci, "MODEL_PATH", model_dir / "model.joblib"
    )
    monkeypatch.setattr(
        train_uci, "METRICS_PATH", model_dir / "metrics.json"
    )

    metrics = train_uci.train_model()

    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "metrics.json").exists()

    assert metrics["training_samples"] == 80
    assert metrics["testing_samples"] == 20
    assert 0 <= metrics["test_size"] < 1
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1

    with open(model_dir / "metrics.json", encoding="utf-8") as file:
        saved_metrics = json.load(file)

    assert saved_metrics["model"] == "RandomForestRegressor"
    assert saved_metrics["dataset"] == "UCI Real Estate Valuation"

    model = joblib.load(model_dir / "model.joblib")
    predictions = model.predict(features.head(5))

    assert len(predictions) == 5
    assert all(pd.notna(predictions))


def test_missing_features_file_raises_error(monkeypatch, tmp_path):
    """Verify missing feature data is reported clearly."""

    monkeypatch.setattr(
        train_uci, "FEATURES_CSV", tmp_path / "missing_features.csv"
    )

    with pytest.raises(FileNotFoundError, match="Features not found"):
        train_uci.train_model()


def test_missing_target_file_raises_error(monkeypatch, tmp_path):
    """Verify missing target data is reported clearly."""

    features_path = tmp_path / "features.csv"
    pd.DataFrame({"feature": [1, 2, 3]}).to_csv(
        features_path, index=False
    )

    monkeypatch.setattr(train_uci, "FEATURES_CSV", features_path)
    monkeypatch.setattr(
        train_uci, "TARGET_CSV", tmp_path / "missing_target.csv"
    )

    with pytest.raises(FileNotFoundError, match="Target not found"):
        train_uci.train_model()
