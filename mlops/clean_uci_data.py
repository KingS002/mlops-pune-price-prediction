"""Clean the UCI Real Estate Valuation dataset for the new MLOps pipeline."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_XLSX = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uci_real_estate"
    / "Real estate valuation data set.xlsx"
)
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "real_estate_clean.csv"

COLUMN_MAP = {
    "X1 transaction date": "transaction_date",
    "X2 house age": "house_age",
    "X3 distance to the nearest MRT station": "distance_to_mrt",
    "X4 number of convenience stores": "convenience_stores",
    "X5 latitude": "latitude",
    "X6 longitude": "longitude",
    "Y house price of unit area": "price_per_unit_area",
}

FEATURE_COLUMNS = [
    "transaction_date",
    "house_age",
    "distance_to_mrt",
    "convenience_stores",
    "latitude",
    "longitude",
]
TARGET_COLUMN = "price_per_unit_area"


def clean_data() -> pd.DataFrame:
    """Load, validate, and clean the UCI dataset."""
    if not RAW_XLSX.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_XLSX}")

    df = pd.read_excel(RAW_XLSX)

    missing_columns = set(COLUMN_MAP) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    # Select predictive features and target; exclude the row identifier.
    df = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)

    # Ensure every feature and the target are numeric.
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Reject invalid rows instead of silently imputing values.
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    if df.empty:
        raise ValueError("No valid rows remain after data cleaning.")

    if (df[TARGET_COLUMN] <= 0).any():
        raise ValueError("Target prices must be positive.")

    if (df["distance_to_mrt"] < 0).any():
        raise ValueError("MRT distance cannot be negative.")

    if (df["house_age"] < 0).any():
        raise ValueError("House age cannot be negative.")

    return df.reset_index(drop=True)


def main() -> Path:
    df = clean_data()
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)

    print(f"Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Output: {CLEAN_CSV}")

    return CLEAN_CSV


if __name__ == "__main__":
    main()
