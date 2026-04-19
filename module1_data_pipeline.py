"""
Module 1: Data Pipeline (yfinance — no API key required)
Fetch, clean, and store historical OHLCV data using Yahoo Finance.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NIFTY_SYMBOL    = "^NSEI"       # Nifty 50
VIX_SYMBOL      = "^INDIAVIX"   # India VIX

NIFTY_DAILY_PATH  = DATA_DIR / "nifty_daily.parquet"
NIFTY_1H_PATH     = DATA_DIR / "nifty_1h.parquet"
NIFTY_5MIN_PATH   = DATA_DIR / "nifty_5min.parquet"   # legacy alias
INDIA_VIX_PATH    = DATA_DIR / "india_vix_daily.parquet"
NIFTY_WEEKLY_PATH = DATA_DIR / "nifty_weekly.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance output to lowercase columns, tz-naive DatetimeIndex."""
    if df.empty:
        return df
    # yfinance may return MultiIndex columns — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index.name = "date"
    return df.sort_index()


def _last_date(path: Path) -> datetime | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return None if df.empty else df.index.max().to_pydatetime()


def _save(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} rows → {path.name}")


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------

def fetch_nifty_daily() -> pd.DataFrame:
    """Fetch / incrementally update Nifty 50 daily OHLCV (5 years)."""
    logger.info("=== Fetching Nifty 50 daily OHLCV ===")
    last = _last_date(NIFTY_DAILY_PATH)
    start = (last + timedelta(days=1)) if last else (datetime.today() - timedelta(days=5 * 365))
    today = datetime.today()

    if last and start.date() >= today.date():
        logger.info("nifty_daily.parquet is already up to date.")
        return pd.read_parquet(NIFTY_DAILY_PATH)

    raw = yf.download(NIFTY_SYMBOL, start=start.strftime("%Y-%m-%d"),
                      end=today.strftime("%Y-%m-%d"), interval="1d",
                      progress=False, auto_adjust=True)
    new_df = _clean_df(raw)

    if new_df.empty:
        logger.warning("No new daily data returned.")
        return pd.read_parquet(NIFTY_DAILY_PATH) if NIFTY_DAILY_PATH.exists() else new_df

    if NIFTY_DAILY_PATH.exists():
        existing = pd.read_parquet(NIFTY_DAILY_PATH)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df

    _save(combined, NIFTY_DAILY_PATH)
    return combined


def fetch_nifty_intraday() -> pd.DataFrame:
    """
    Fetch Nifty 50 intraday OHLCV for realized volatility computation.

    Strategy (best free data available):
      - Primary  : 1-hour bars via yfinance — covers last 730 days (2 years)
      - Fallback : 5-minute bars — covers last 60 days (used if 1h fails)

    Yahoo Finance does NOT provide 5+ year intraday data on any free tier.
    For weeks beyond 2 years, module2 falls back to Parkinson vol (daily H/L).
    Saved to: data/nifty_1h.parquet
    """
    logger.info("=== Fetching Nifty 50 intraday OHLCV (1h, last 2 years) ===")

    last = _last_date(NIFTY_1H_PATH)
    # 1h data: yfinance allows up to 730 days; always re-pull last 730 days
    # because incremental 1h fetching is unreliable on yfinance
    start = datetime.today() - timedelta(days=729)
    if last and (datetime.today() - last).days < 3:
        logger.info("nifty_1h.parquet is already up to date (updated within 3 days).")
        return pd.read_parquet(NIFTY_1H_PATH)

    raw = yf.download(
        NIFTY_SYMBOL,
        start=start.strftime("%Y-%m-%d"),
        end=datetime.today().strftime("%Y-%m-%d"),
        interval="1h",
        progress=False,
        auto_adjust=True,
    )
    new_df = _clean_df(raw)

    if new_df.empty:
        logger.warning("1h data empty — falling back to 5-minute (60 days).")
        raw5 = yf.download(NIFTY_SYMBOL, period="60d", interval="5m",
                           progress=False, auto_adjust=True)
        new_df = _clean_df(raw5)
        if new_df.empty:
            logger.error("No intraday data returned from yfinance.")
            return pd.read_parquet(NIFTY_1H_PATH) if NIFTY_1H_PATH.exists() else new_df

    # Remove first bar of each session (overnight gap)
    bar_duration = pd.Timedelta(hours=1)
    dates = new_df.index.normalize()
    is_first_bar = dates != (new_df.index - bar_duration).normalize()
    new_df = new_df[~is_first_bar]

    _save(new_df, NIFTY_1H_PATH)
    return new_df


def fetch_nifty_5min() -> pd.DataFrame:
    """Legacy alias — redirects to fetch_nifty_intraday() for backward compat."""
    return fetch_nifty_intraday()


def fetch_india_vix() -> pd.DataFrame:
    """Fetch / incrementally update India VIX daily close (5 years)."""
    logger.info("=== Fetching India VIX daily ===")
    last = _last_date(INDIA_VIX_PATH)
    start = (last + timedelta(days=1)) if last else (datetime.today() - timedelta(days=5 * 365))
    today = datetime.today()

    if last and start.date() >= today.date():
        logger.info("india_vix_daily.parquet is already up to date.")
        return pd.read_parquet(INDIA_VIX_PATH)

    raw = yf.download(VIX_SYMBOL, start=start.strftime("%Y-%m-%d"),
                      end=today.strftime("%Y-%m-%d"), interval="1d",
                      progress=False, auto_adjust=True)
    new_df = _clean_df(raw)

    if new_df.empty:
        logger.warning("No new VIX data returned.")
        return pd.read_parquet(INDIA_VIX_PATH) if INDIA_VIX_PATH.exists() else new_df

    if INDIA_VIX_PATH.exists():
        existing = pd.read_parquet(INDIA_VIX_PATH)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df

    _save(combined, INDIA_VIX_PATH)
    return combined


def build_nifty_weekly(nifty_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily Nifty data to weekly (week ending Friday)."""
    logger.info("=== Building Nifty weekly aggregation ===")
    if nifty_daily_df.empty:
        logger.warning("Daily Nifty data is empty; skipping weekly aggregation.")
        return pd.DataFrame()

    weekly = nifty_daily_df.resample("W-FRI").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    weekly.index.name = "week_end"
    weekly = weekly.dropna(subset=["open", "close"])
    _save(weekly, NIFTY_WEEKLY_PATH)
    return weekly


# ---------------------------------------------------------------------------
# Live spot (no credentials)
# ---------------------------------------------------------------------------

def fetch_live_spot_yf() -> float:
    """Fetch current Nifty 50 spot price via yfinance (no API key needed)."""
    ticker = yf.Ticker(NIFTY_SYMBOL)
    spot = ticker.fast_info["last_price"]
    logger.info(f"Live Nifty spot (yfinance): {spot:.2f}")
    return float(spot)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline() -> dict:
    nifty_daily  = fetch_nifty_daily()
    nifty_5min   = fetch_nifty_5min()
    india_vix    = fetch_india_vix()
    nifty_weekly = build_nifty_weekly(nifty_daily)

    summary = {
        "nifty_daily_rows":  len(nifty_daily),
        "nifty_5min_rows":   len(nifty_5min),
        "india_vix_rows":    len(india_vix),
        "nifty_weekly_rows": len(nifty_weekly),
    }

    print("\n=== Pipeline Summary ===")
    print(f"  Nifty 50 daily  : {summary['nifty_daily_rows']:>6} rows")
    print(f"  Nifty 50 5-min  : {summary['nifty_5min_rows']:>6} rows  (last 60 days)")
    print(f"  India VIX daily : {summary['india_vix_rows']:>6} rows")
    print(f"  Nifty weekly    : {summary['nifty_weekly_rows']:>6} rows")
    print("========================\n")
    return summary


if __name__ == "__main__":
    run_pipeline()
