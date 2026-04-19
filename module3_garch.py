"""
Module 3: GARCH(1,1) Conditional Volatility
Fits GARCH(1,1) on Nifty 50 daily log returns, extracts conditional volatility,
aggregates to weekly features, and merges into the feature matrix.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from arch import arch_model
from loguru import logger

BASE_DIR = Path(__file__).parent


def run_garch_pipeline() -> pd.DataFrame:
    # 1. Load daily data
    daily_path = BASE_DIR / "data" / "nifty_daily.parquet"
    if not daily_path.exists():
        raise FileNotFoundError("Run module1_data_pipeline.py first")

    logger.info(f"Loading daily data from {daily_path}")
    daily_df = pd.read_parquet(daily_path)

    # Ensure DatetimeIndex with no timezone
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df.index = pd.to_datetime(daily_df.index)
    if daily_df.index.tz is not None:
        daily_df.index = daily_df.index.tz_localize(None)

    close = daily_df["close"]

    # 2. Compute log returns scaled by 100
    returns = np.log(close / close.shift(1)).dropna() * 100
    logger.info(f"Computed {len(returns)} log returns")

    # 3. Fit GARCH(1,1)
    logger.info("Fitting GARCH(1,1) model...")
    model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    result = model.fit(disp="off")
    logger.info("GARCH model fitted successfully")

    # 4. Extract conditional volatility and convert back from percentage
    cond_vol = result.conditional_volatility / 100

    # 5. Aggregate to weekly (W-FRI)
    weekly_mean = cond_vol.resample("W-FRI").mean().rename("garch_sigma_mean")
    weekly_max = cond_vol.resample("W-FRI").max().rename("garch_sigma_max")
    garch_weekly = pd.concat([weekly_mean, weekly_max], axis=1)
    garch_weekly.index.name = "week_end"
    logger.info(f"Aggregated GARCH vol to {len(garch_weekly)} weekly observations")

    # 6. Merge into feature matrix
    feat_path = BASE_DIR / "data" / "feature_matrix.parquet"
    if not feat_path.exists():
        raise FileNotFoundError("Run module2_features.py first")

    logger.info(f"Loading feature matrix from {feat_path}")
    feat_df = pd.read_parquet(feat_path)

    if feat_df.index.name != "week_end":
        feat_df.index.name = "week_end"

    merged = feat_df.join(garch_weekly, how="left")

    # Drop rows where GARCH features are NaN (warm-up period)
    before = len(merged)
    merged = merged.dropna(subset=["garch_sigma_mean", "garch_sigma_max"])
    dropped = before - len(merged)
    logger.info(f"Dropped {dropped} rows during GARCH warm-up period")

    out_path = BASE_DIR / "data" / "feature_matrix_with_garch.parquet"
    merged.to_parquet(out_path)
    logger.info(f"Saved feature matrix with GARCH features to {out_path}")

    # 7. Save GARCH model
    model_path = BASE_DIR / "models" / "garch_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, model_path)
    logger.info(f"Saved GARCH model to {model_path}")

    # 8. Print summary
    params = result.params
    omega = params.get("omega", params.iloc[0])
    alpha = params.get("alpha[1]", params.iloc[1])
    beta = params.get("beta[1]", params.iloc[2])

    print(f"\n--- GARCH(1,1) Summary ---")
    print(f"Rows in final feature matrix: {len(merged)}")
    print(f"omega  = {omega:.6f}")
    print(f"alpha  = {alpha:.6f}")
    print(f"beta   = {beta:.6f}")
    print(f"alpha + beta = {alpha + beta:.6f}")

    return merged


if __name__ == "__main__":
    df = run_garch_pipeline()
    print(df.shape)
