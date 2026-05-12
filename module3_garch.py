"""
Module 3: GJR-GARCH(1,1,1) Conditional Volatility
Fits GJR-GARCH(1,1,1) with skewed-t distribution on Nifty 50 daily log returns, extracts conditional volatility,
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

    # 3. Fit GJR-GARCH(1,1,1) with skewed-t distribution
    logger.info("Fitting GJR-GARCH(1,1,1) with skewed-t distribution...")
    model = arch_model(returns, vol="Garch", p=1, o=1, q=1, dist="skewt")
    result = model.fit(disp="off")
    if result.convergence_flag != 0:
        logger.warning(f"GARCH model did not converge (flag={result.convergence_flag}). Results may be unstable.")
    logger.info("GJR-GARCH model fitted successfully")

    # 4. Extract conditional volatility and convert back from percentage
    cond_vol = result.conditional_volatility / 100

    # 5. Aggregate to weekly (W-TUE = Tuesday expiry) and LAG by 1 week
    # This ensures that for the row indexed Friday T, we use volatility from week T-1
    # to predict log_range of week T.
    weekly_mean = cond_vol.resample("W-TUE").mean().rename("garch_sigma_mean").shift(1)
    weekly_max = cond_vol.resample("W-TUE").max().rename("garch_sigma_max").shift(1)
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
    omega = float(params.get("omega", params.iloc[0]))
    alpha = float(params.get("alpha[1]", 0.0))
    gamma = float(params.get("gamma[1]", 0.0))
    beta  = float(params.get("beta[1]", 0.0))
    nu    = params.get("nu", None)
    lam   = params.get("lambda", None)

    persistence = alpha + 0.5 * gamma + beta

    print(f"\n--- GJR-GARCH(1,1,1) Summary ---")
    print(f"Rows in final feature matrix: {len(merged)}")
    print(f"omega       = {omega:.6f}")
    print(f"alpha[1]    = {alpha:.6f}")
    print(f"gamma[1]    = {gamma:.6f}  (leverage: neg returns increase vol more)")
    print(f"beta[1]     = {beta:.6f}")
    print(f"persistence = alpha + 0.5*gamma + beta = {persistence:.6f}")
    if nu is not None:
        print(f"nu (df)     = {float(nu):.4f}")
    if lam is not None:
        print(f"lambda (skew) = {float(lam):.4f}")

    return merged


if __name__ == "__main__":
    df = run_garch_pipeline()
    print(df.shape)
