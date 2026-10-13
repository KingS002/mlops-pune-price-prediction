import pandas as pd
import pytest

from mlops.clean_uci_data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_data,
)


def test_clean_data_has_expected_shape_and_columns():
    df = clean_data()

    assert df.shape == (414, 7)
    assert df.columns.tolist() == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_clean_data_has_no_missing_values():
    df = clean_data()

    assert not df.isnull().any().any()


def test_clean_data_excludes_row_identifier():
    df = clean_data()

    assert "No" not in df.columns


def test_clean_data_has_positive_target():
    df = clean_data()

    assert (df[TARGET_COLUMN] > 0).all()


def test_clean_data_has_valid_feature_ranges():
    df = clean_data()

    assert (df["house_age"] >= 0).all()
    assert (df["distance_to_mrt"] >= 0).all()
    assert (df["convenience_stores"] >= 0).all()


def test_missing_dataset_raises_error(monkeypatch, tmp_path):
    from mlops import clean_uci_data

    monkeypatch.setattr(
        clean_uci_data, "RAW_XLSX", tmp_path / "missing.xlsx"
    )

    with pytest.raises(FileNotFoundError):
        clean_data()
