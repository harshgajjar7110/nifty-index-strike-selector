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
    return series.resample("W-TUE").last()


def _load_event_calendar() -> list[dict]:
    """Load event calendar from JSON; fallback to hardcoded defaults."""
    import json
    cal_path = BASE_DIR / "data" / "event_calendar.json"
    if cal_path.exists():
        with open(cal_path) as f:
            return json.load(f)
    # Fallback hardcoded (same as before, but now centralized)
    return [
        {"name": "Union Budget", "month": 2, "week": 1, "start_year": 2020},
        {"name": "RBI MPC", "month": 4, "week": 1, "start_year": 2020},
        {"name": "RBI MPC", "month": 6, "week": 1, "start_year": 2020},
        {"name": "RBI MPC", "month": 8, "week": 1, "start_year": 2020},
        {"name": "RBI MPC", "month": 10, "week": 1, "start_year": 2020},
        {"name": "RBI MPC", "month": 12, "week": 1, "start_year": 2020},
        {"name": "Earnings Season", "month": 1, "week": 2, "start_year": 2020},
        {"name": "Earnings Season", "month": 4, "week": 2, "start_year": 2020},
        {"name": "Earnings Season", "month": 7, "week": 2, "start_year": 2020},
        {"name": "Earnings Season", "month": 10, "week": 2, "start_year": 2020},
    ]


# Module-level cache (events rarely change during a single run)
_EVENT_CALENDAR = None


def _is_event_week(week_end: pd.Timestamp) -> int:
    """Budget/RBI/earnings week indicator (year-aware)."""
    global _EVENT_CALENDAR
    if _EVENT_CALENDAR is None:
        _EVENT_CALENDAR = _load_event_calendar()

    month = week_end.month
    week_of_month = (week_end.day - 1) // 7 + 1
    year = week_end.year

    for event in _EVENT_CALENDAR:
        if (
            event["month"] == month
            and event["week"] == week_of_month
            and year >= event.get("start_year", 2000)
        ):
            return 1
    return 0


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features() -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    for path in [NIFTY_DAILY_PATH, INDIA_VIX_PATH, NIFTY_WEEKLY_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run module1_data_pipeline.py first"
            )
    if not NIFTY_1H_PATH.exists() and not NIFTY_5MIN_PATH.exists():
        raise FileNotFoundError(
            f"Neither {NIFTY_1H_PATH} nor {NIFTY_5MIN_PATH} found. Run module1_data_pipeline.py first"
        )

    logger.info("Loading daily OHLCV data...")
    daily = pd.read_parquet(NIFTY_DAILY_PATH)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    daily = daily.copy()

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
    min5 = min5.copy()

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
    weekly = weekly.copy()
    # Ensure index is Friday-anchored (resample if needed)
    if not (weekly.index.dayofweek == 4).all():
        logger.warning("Weekly data index not all Tuesdays — resampling daily to W-TUE")
        weekly = daily.resample("W-TUE").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna(subset=["open", "close"])

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
        .resample("W-TUE")
        .mean()
        .pipe(lambda s: np.sqrt(s / (4 * ln2)))
    )
    park_weekly.name = "parkinson_vol"

    # Garman-Klass: sqrt( mean( 0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2 ) ) per week
    gk_term = 0.5 * (daily["log_hl"] ** 2) - (2 * ln2 - 1) * (daily["log_co"] ** 2)
    gk_weekly = (
        gk_term
        .resample("W-TUE")
        .mean()
        .pipe(lambda s: np.sqrt(s.clip(lower=0)))
    )
    gk_weekly.name = "garman_klass_vol"

    # ------------------------------------------------------------------
    # 4. Realized volatility from 5-min data
    # ------------------------------------------------------------------
    logger.info("Computing 5-min realized volatility...")
    # Compute log returns per date, masking overnight gaps
    min5["log_ret"] = np.log(min5["close"] / min5["close"].shift(1))
    # Identify first bar of each trading day by checking if date changed
    dates = min5.index.normalize().values
    prev_dates = np.concatenate([np.array([np.datetime64("NaT")]), dates[:-1]])
    is_first_bar = (dates != prev_dates)
    min5.loc[is_first_bar, "log_ret"] = np.nan
    # Compute realized vol per week from non-NaN returns
    rv_5min = (
        (min5["log_ret"].dropna() ** 2)
        .resample("W-TUE")
        .sum()
        .pipe(np.sqrt)
    )
    rv_5min.name = "realized_vol_5min"

    # ------------------------------------------------------------------
    # 5. VIX features
    # ------------------------------------------------------------------
    logger.info("Computing VIX features...")
    vix_weekly = vix["close"].resample("W-TUE").last()
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
    # Build a hybrid realized vol series: use 5-min RV where available,
    # else fall back to Parkinson volatility for weeks >2 years ago.
    rv_hybrid = rv_5min.reindex(park_weekly.index)
    rv_hybrid = rv_hybrid.fillna(park_weekly)  # park_weekly covers full history
    rv_hybrid.name = "realized_vol_5min"         # keep column name consistent for backward compat

    rv_ann = rv_hybrid * np.sqrt(252)
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
    # 9. 4-week momentum (lagged to avoid lookahead)
    # ------------------------------------------------------------------
    logger.info("Computing 4-week momentum...")
    return_4w = np.log(weekly["close"] / weekly["close"].shift(4)).shift(1)
    return_4w = return_4w.replace([np.inf, -np.inf], np.nan)
    return_4w.name = "return_4w"

    # ------------------------------------------------------------------
    # 10. Realized vol skewness (rolling 20 daily log-returns, resampled weekly)
    # ------------------------------------------------------------------
    logger.info("Computing realized vol skewness...")
    daily_log_ret = np.log(daily["close"] / daily["close"].shift(1))
    vol_skew_daily = daily_log_ret.rolling(20, min_periods=10).skew()
    vol_skew = _resample_weekly_last(vol_skew_daily).shift(1)
    vol_skew.name = "vol_skew"

    # ------------------------------------------------------------------
    # 11. VIX z-score (deviation from 52-week rolling mean)
    # ------------------------------------------------------------------
    logger.info("Computing VIX z-score...")
    vix_52w_mean = vix_weekly.rolling(52, min_periods=20).mean()
    vix_52w_std  = vix_weekly.rolling(52, min_periods=20).std()
    vix_zscore   = (vix_weekly - vix_52w_mean) / vix_52w_std.clip(lower=0.01)
    vix_zscore.name = "vix_zscore"

    # ------------------------------------------------------------------
    # 12. Close position within weekly range (where did week close?)
    # ------------------------------------------------------------------
    logger.info("Computing close position within range...")
    close_position = (weekly["close"] - weekly["low"]) / (weekly["high"] - weekly["low"])
    close_position = close_position.replace([np.inf, -np.inf], np.nan).shift(1)
    close_position.name = "close_position_in_range"

    # ------------------------------------------------------------------
    # 13. Trend strength proxy (|close-open| / true range, averaged weekly)
    # ------------------------------------------------------------------
    logger.info("Computing trend strength proxy...")
    daily["tr_tmp"] = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - daily["close"].shift(1)).abs(),
        (daily["low"] - daily["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    daily["co_range"] = (daily["close"] - daily["open"]).abs()
    daily["trend_str"] = daily["co_range"] / daily["tr_tmp"]
    trend_str_weekly = daily["trend_str"].resample("W-TUE").mean().shift(1)
    trend_str_weekly.name = "trend_strength_proxy"

    # ------------------------------------------------------------------
    # 14. Event week indicator
    # ------------------------------------------------------------------
    all_fridays = weekly.index
    logger.info("Computing event-week flag...")
    is_event = pd.Series(
        [_is_event_week(f) for f in all_fridays],
        index=all_fridays,
        name="is_event_week",
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 15. Target variable: log range
    # ------------------------------------------------------------------
    logger.info("Computing target variable log_range...")
    log_range = np.log(weekly["high"] / weekly["low"])
    log_range.name = "log_range"

    # ------------------------------------------------------------------
    # 16. Merge all features
    # ------------------------------------------------------------------
    logger.info("Merging all features...")
    # LAGGING: Most features must be shifted by 1 to represent state at START of week
    frames = [
        atr5.shift(1), atr14.shift(1), atr21.shift(1),
        park_weekly.shift(1), gk_weekly.shift(1),
        rv_hybrid.shift(1),
        vix_weekly.shift(1), vix_change.shift(1), vrp.shift(1),
        range_1w, range_4w_avg,
        prev_week_gap,
        bb_width.shift(1),
        return_4w,
        vol_skew,
        vix_zscore.shift(1),
        close_position,
        trend_str_weekly,
        is_event.shift(1),
        log_range,
    ]

    df = pd.concat(frames, axis=1)
    df.index.name = "week_end"

    # ------------------------------------------------------------------
    # 17. Merge macro features (if available)
    # ------------------------------------------------------------------
    macro_path = BASE_DIR / "data" / "macro_daily.parquet"
    if macro_path.exists():
        logger.info("Merging macro features...")
        try:
            from module1b_macro import build_macro_features
            macro_feat = build_macro_features()
            df = df.join(macro_feat, how="left")
            # Forward-fill macro gaps (e.g., US holidays vs Indian trading)
            macro_cols = [c for c in macro_feat.columns if c in df.columns]
            df[macro_cols] = df[macro_cols].ffill(limit=2)
            logger.info(f"Macro features merged: {macro_cols}")
        except Exception as e:
            logger.warning(f"Macro feature merge failed: {e}")
    else:
        logger.info("Macro data not found; skipping. Run module1b_macro.py to fetch.")

    # Keep only rows where weekly data exists
    df = df[df.index.isin(weekly.index)]

    before = len(df)
    df = df.dropna()
    after = len(df)
    logger.info(f"Dropped {before - after} rows with NaN (warm-up period). Remaining: {after}")

    # ------------------------------------------------------------------
    # 18. Save
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
