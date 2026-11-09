
"""Feature engineering for the UCI Real Estate Valuation dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "real_estate_clean.csv"
)

FEATURES_CSV = (
    PROJECT_ROOT / "data" / "processed" / "real_estate_features.csv"
)

TARGET_CSV = (
    PROJECT_ROOT / "data" / "processed" / "real_estate_target.csv"
)

TARGET_COLUMN = "price_per_unit_area"

BASE_FEATURES = [
    "transaction_date",
    "house_age",
    "distance_to_mrt",
    "convenience_stores",
    "latitude",
    "longitude",
]


def build_features() -> tuple[pd.DataFrame, pd.Series]:
    """Build deterministic features from the cleaned UCI dataset."""

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_CSV}. "
            "Run python -m mlops.clean_uci_data first."
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = BASE_FEATURES + [TARGET_COLUMN]
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    X = df[BASE_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    # Extract year and approximate month from decimal transaction date.
    X["transaction_year"] = np.floor(
        X["transaction_date"]
    ).astype(int)

    fractional_year = X["transaction_date"] - X["transaction_year"]

    X["transaction_month"] = (
        np.floor(fractional_year * 12).astype(int) + 1
    ).clip(1, 12)

    # Nonlinear and skewness-reducing features.
    X["house_age_squared"] = X["house_age"] ** 2

    X["distance_to_mrt_log"] = np.log1p(
        X["distance_to_mrt"]
    )

    # Keep original transaction_date for traceability.
    if X.isnull().any().any():
        raise ValueError("Feature matrix contains missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains non-finite values.")

    return X, y


def main() -> None:
    """Generate and save UCI feature and target datasets."""

    X, y = build_features()

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)

    X.to_csv(FEATURES_CSV, index=False)
    y.to_frame(name=TARGET_COLUMN).to_csv(TARGET_CSV, index=False)

    print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")
    print(f"Target vector: {y.shape[0]} rows")
    print(f"Features saved: {FEATURES_CSV}")
    print(f"Target saved: {TARGET_CSV}")


if __name__ == "__main__":
    main()
