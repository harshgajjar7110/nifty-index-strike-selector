"""
Module 2: Feature Engineering
Build the full weekly feature matrix used by the ML model for Nifty 50
Weekly Range Predictor (iron condor options trading).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

NIFTY_DAILY_PATH = DATA_DIR / "nifty_daily.parquet"
NIFTY_1H_PATH   = DATA_DIR / "nifty_1h.parquet"
NIFTY_5MIN_PATH = DATA_DIR / "nifty_5min.parquet"  # legacy fallback
INDIA_VIX_PATH = DATA_DIR / "india_vix_daily.parquet"
NIFTY_WEEKLY_PATH = DATA_DIR / "nifty_weekly.parquet"
FEATURE_MATRIX_PATH = DATA_DIR / "feature_matrix.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _days_to_next_expiry(friday: pd.Timestamp) -> float:
    """Return calendar days from `friday` (week_end) to the next weekly expiry Thursday.

    Nifty options expire every Thursday (weekly). Since week_end is a Friday,
    the next expiry Thursday is always 6 calendar days away.
    """
    return 6.0


def _compute_atr(daily: pd.DataFrame, n: int) -> pd.Series:
    """Compute N-day ATR from daily OHLCV DataFrame."""
    high = daily["high"]
    low = daily["low"]
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    atr.name = f"atr_{n}"
    return atr


def _resample_weekly_last(series: pd.Series) -> pd.Series:
    """Resample a daily series to weekly (Friday), taking the last value."""
    return series.resample("W-FRI").last()


def _is_event_week(friday: pd.Timestamp) -> int:
    """Budget/RBI/earnings week indicator."""
    month = friday.month
    week_of_month = (friday.day - 1) // 7 + 1
    rbi_months = {4, 6, 8, 10, 12}
    earnings_months = {1, 4, 7, 10}
    is_budget = (month == 2 and week_of_month == 1)
    is_rbi = (month in rbi_months and week_of_month == 1)
    is_earnings = (month in earnings_months and week_of_month == 2)
    return int(is_budget or is_rbi or is_earnings)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features() -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    for path in [NIFTY_DAILY_PATH, NIFTY_5MIN_PATH, INDIA_VIX_PATH, NIFTY_WEEKLY_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run module1_data_pipeline.py first"
            )

    logger.info("Loading daily OHLCV data...")
    daily = pd.read_parquet(NIFTY_DAILY_PATH)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Load intraday data — prefer 1h (2 years), fall back to legacy 5min file
    if NIFTY_1H_PATH.exists():
        logger.info("Loading 1h intraday OHLCV data (2 years)...")
        min5 = pd.read_parquet(NIFTY_1H_PATH)
    elif NIFTY_5MIN_PATH.exists():
        logger.info("Loading 5-min OHLCV data (60-day fallback)...")
        min5 = pd.read_parquet(NIFTY_5MIN_PATH)
    else:
        raise FileNotFoundError("Run module1_data_pipeline.py first")
    min5.index = pd.to_datetime(min5.index)
    min5 = min5.sort_index()

    logger.info("Loading India VIX data...")
    vix = pd.read_parquet(INDIA_VIX_PATH)
    vix.index = pd.to_datetime(vix.index)
    vix = vix.sort_index()
    if "close" not in vix.columns:
        raise ValueError(
            f"India VIX parquet has no 'close' column. Found: {vix.columns.tolist()}"
        )

    logger.info("Loading weekly OHLCV data...")
    weekly = pd.read_parquet(NIFTY_WEEKLY_PATH)
    weekly.index = pd.to_datetime(weekly.index)
    weekly = weekly.sort_index()
    # Ensure index is Friday-anchored (resample if needed)
    if not (weekly.index.dayofweek == 4).all():
        logger.warning("Weekly data index not all Fridays — resampling daily to W-FRI")
        weekly = daily.resample("W-FRI").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )

    # ------------------------------------------------------------------
    # 2. ATR features (daily, resampled to weekly last value)
    # ------------------------------------------------------------------
    logger.info("Computing ATR features...")
    atr5 = _resample_weekly_last(_compute_atr(daily, 5))
    atr14 = _resample_weekly_last(_compute_atr(daily, 14))
    atr21 = _resample_weekly_last(_compute_atr(daily, 21))

    # ------------------------------------------------------------------
    # 3. Volatility estimators (computed per week from daily bars)
    # ------------------------------------------------------------------
    logger.info("Computing Parkinson and Garman-Klass volatility...")

    ln2 = np.log(2)
    daily["log_hl"] = np.log(daily["high"] / daily["low"])
    daily["log_co"] = np.log(daily["close"] / daily["open"])

    # Parkinson: sqrt( 1/(4*ln2) * mean(ln(H/L)^2) ) per week
    park_weekly = (
        (daily["log_hl"] ** 2)
        .resample("W-FRI")
        .mean()
        .pipe(lambda s: np.sqrt(s / (4 * ln2)))
    )
    park_weekly.name = "parkinson_vol"

    # Garman-Klass: sqrt( mean( 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2 ) ) per week
    gk_term = 0.5 * (daily["log_hl"] ** 2) - (2 * ln2 - 1) * (daily["log_co"] ** 2)
    gk_weekly = (
        gk_term
        .resample("W-FRI")
        .mean()
        .pipe(lambda s: np.sqrt(s.clip(lower=0)))
    )
    gk_weekly.name = "garman_klass_vol"

    # ------------------------------------------------------------------
    # 4. Realized volatility from 5-min data
    # ------------------------------------------------------------------
    logger.info("Computing 5-min realized volatility...")
    # Compute log returns within each trading date to avoid overnight gap contamination.
    # Group by date, compute intra-session log returns (first bar of each session gets NaN
    # and is dropped), then reassemble.
    def _intraday_log_rets(grp: pd.DataFrame) -> pd.Series:
        return np.log(grp["close"] / grp["close"].shift(1))

    min5["log_ret"] = (
        min5.groupby(min5.index.normalize(), group_keys=False)
        .apply(_intraday_log_rets)
    )
    # Drop NaN entries (first bar of each session)
    rv_5min = (
        (min5["log_ret"].dropna() ** 2)
        .resample("W-FRI")
        .sum()
        .pipe(np.sqrt)
    )
    rv_5min.name = "realized_vol_5min"

    # ------------------------------------------------------------------
    # 5. VIX features
    # ------------------------------------------------------------------
    logger.info("Computing VIX features...")
    vix_weekly = vix["close"].resample("W-FRI").last()
    vix_weekly.name = "vix_level"
    vix_change = vix_weekly.diff(1)
    vix_change.name = "vix_change_1w"

    # ------------------------------------------------------------------
    # 6. Volatility risk premium
    # yfinance provides up to 2 years of 1h intraday data (730 days).
    # For weeks older than 2 years, fall back to Parkinson vol
    # (already weekly) as the realized vol proxy.
    # ------------------------------------------------------------------
    logger.info("Computing vol risk premium...")
    # Build a realized vol series: use 5-min RV where available, else Parkinson
    rv_combined = rv_5min.reindex(park_weekly.index)
    rv_combined = rv_combined.fillna(park_weekly)  # park_weekly covers full history
    rv_combined.name = "realized_vol_5min"         # keep column name consistent
    rv_5min = rv_combined                          # replace sparse series with filled one

    rv_ann = rv_5min * np.sqrt(52)
    vix_aligned, rv_aligned = vix_weekly.align(rv_ann, join="left")
    vrp = vix_aligned / 100 - rv_aligned
    vrp.name = "vol_risk_premium"

    # ------------------------------------------------------------------
    # 7. Weekly range features (lagged to avoid lookahead)
    # ------------------------------------------------------------------
    logger.info("Computing weekly range features...")
    weekly_range = (weekly["high"] - weekly["low"]) / weekly["close"]
    range_1w = weekly_range.shift(1)
    range_1w.name = "range_1w"
    range_4w_avg = weekly_range.shift(1).rolling(4).mean()
    range_4w_avg.name = "range_4w_avg"

    # Previous week gap: (open - prev_close) / prev_close, lagged by 1
    logger.info("Computing previous week gap...")
    prev_week_gap = (weekly["open"] - weekly["close"].shift(1)) / weekly["close"].shift(1)
    prev_week_gap = prev_week_gap.shift(1)
    prev_week_gap.name = "prev_week_gap"

    # ------------------------------------------------------------------
    # 8. Bollinger Band width (daily close, 20-day, 2std → weekly last)
    # ------------------------------------------------------------------
    logger.info("Computing Bollinger Band width...")
    close_d = daily["close"]
    bb_mid = close_d.rolling(20).mean()
    bb_std = close_d.rolling(20).std()
    bb_width_daily = (bb_mid + 2 * bb_std - (bb_mid - 2 * bb_std)) / bb_mid
    bb_width = _resample_weekly_last(bb_width_daily)
    bb_width.name = "bb_width"

    # ------------------------------------------------------------------
    # 9. Days to expiry
    # ------------------------------------------------------------------
    logger.info("Computing days-to-expiry...")
    # Nifty options expire every Thursday (weekly). week_end is always a Friday,
    # so the next expiry Thursday is always 6 calendar days away.
    all_fridays = weekly.index
    dte = pd.Series(
        [_days_to_next_expiry(f) for f in all_fridays],
        index=all_fridays,
        name="days_to_expiry",
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 10. Event week indicator
    # ------------------------------------------------------------------
    logger.info("Computing event-week flag...")
    is_event = pd.Series(
        [_is_event_week(f) for f in all_fridays],
        index=all_fridays,
        name="is_event_week",
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 11. Target variable: log range
    # ------------------------------------------------------------------
    logger.info("Computing target variable log_range...")
    log_range = np.log(weekly["high"] / weekly["low"])
    log_range.name = "log_range"

    # ------------------------------------------------------------------
    # 12. Merge all features
    # ------------------------------------------------------------------
    logger.info("Merging all features...")
    frames = [
        atr5, atr14, atr21,
        park_weekly, gk_weekly,
        rv_5min,
        vix_weekly, vix_change, vrp,
        range_1w, range_4w_avg,
        prev_week_gap,
        bb_width,
        dte,
        is_event,
        log_range,
    ]

    df = pd.concat(frames, axis=1)
    df.index.name = "week_end"

    # Keep only rows where weekly data exists
    df = df[df.index.isin(weekly.index)]

    before = len(df)
    df = df.dropna()
    after = len(df)
    logger.info(f"Dropped {before - after} rows with NaN (warm-up period). Remaining: {after}")

    # ------------------------------------------------------------------
    # 13. Save
    # ------------------------------------------------------------------
    logger.info(f"Saving feature matrix to {FEATURE_MATRIX_PATH}")
    df.to_parquet(FEATURE_MATRIX_PATH)
    logger.success(f"Feature matrix saved: shape={df.shape}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_features()
    print(df.shape, df.columns.tolist())
