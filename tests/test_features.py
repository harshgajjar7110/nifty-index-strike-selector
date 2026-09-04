import pandas as pd
import numpy as np
import pytest
from pathlib import Path


def test_no_dead_features():
    feat_path = Path("data/feature_matrix.parquet")
    if not feat_path.exists():
        pytest.skip("feature_matrix.parquet not found")
    df = pd.read_parquet(feat_path)
    assert "days_to_expiry" not in df.columns, "dead feature still present"


def test_new_features_present():
    feat_path = Path("data/feature_matrix.parquet")
    if not feat_path.exists():
        pytest.skip("feature_matrix.parquet not found")
    df = pd.read_parquet(feat_path)
    for col in ("return_4w", "vol_skew", "vix_zscore"):
        assert col in df.columns, f"missing feature: {col}"


def test_new_features_no_all_nan():
    feat_path = Path("data/feature_matrix.parquet")
    if not feat_path.exists():
        pytest.skip("feature_matrix.parquet not found")
    df = pd.read_parquet(feat_path)
    for col in ("return_4w", "vol_skew", "vix_zscore"):
        if col in df.columns:
            assert df[col].notna().sum() > 0, f"{col} is all NaN"
