"""
Module 1b: Macro Data Fetcher
Fetches global macro indicators that drive Nifty 50 sentiment via yfinance.
Assets: US VIX, S&P 500, Crude Oil (Brent), USD/INR, US 10Y Treasury yield.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MACRO_PARQUET = DATA_DIR / "macro_daily.parquet"

TICKERS = {
    "us_vix": "^VIX",
    "spx": "^GSPC",
    "crude": "BZ=F",        # Brent crude (USD/bbl)
    "usd_inr": "USDINR=X",
    "us_10y": "^TNX",       # US 10Y yield (%)
}


def _last_date(path: Path) -> datetime | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return None if df.empty else df.index.max().to_pydatetime()


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index.name = "date"
    return df.sort_index()


def fetch_macro_daily() -> pd.DataFrame:
    """Fetch / incrementally update macro daily close prices."""
    logger.info("=== Fetching macro data ===")
    last = _last_date(MACRO_PARQUET)
    start = (last + timedelta(days=1)) if last else (datetime.today() - timedelta(days=5 * 365))
    today = datetime.today()

    if last and start.date() >= today.date():
        logger.info("macro_daily.parquet is already up to date.")
        return pd.read_parquet(MACRO_PARQUET)

    frames = []
    for name, symbol in TICKERS.items():
        logger.info(f"Fetching {name} ({symbol})...")
        try:
            raw = yf.download(
                symbol,
                start=start.strftime("%Y-%m-%d"),
                end=today.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            df = _clean_df(raw)
            if df.empty:
                logger.warning(f"{name}: no data returned")
                continue
            # Use close price; fall back to adj close or last column
            col = "close" if "close" in df.columns else (df.columns[-1])
            series = df[col].rename(name)
            frames.append(series)
        except Exception as e:
            logger.warning(f"{name}: fetch failed: {e}")

    if not frames:
        logger.error("No macro data fetched.")
        return pd.read_parquet(MACRO_PARQUET) if MACRO_PARQUET.exists() else pd.DataFrame()

    new_df = pd.concat(frames, axis=1)
    new_df = new_df.sort_index()

    if MACRO_PARQUET.exists():
        existing = pd.read_parquet(MACRO_PARQUET)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df

    combined.to_parquet(MACRO_PARQUET)
    logger.info(f"Saved macro_daily.parquet: {combined.shape}")
    return combined


def build_macro_features() -> pd.DataFrame:
    """
    Build weekly macro features aligned to Friday week-ends.
    Returns DataFrame with columns:
      us_vix_level, spx_return_1w, spx_return_4w, spx_volatility_20d,
      crude_return_1w, usd_inr_level, usd_inr_change_1w,
      us_10y_level, us_10y_change_1w, term_premium_proxy
    """
    if not MACRO_PARQUET.exists():
        raise FileNotFoundError("Run fetch_macro_daily() first")

    df = pd.read_parquet(MACRO_PARQUET)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Resample to weekly (Friday) — last observation
    weekly = df.resample("W-FRI").last()

    # 1. US VIX level (lagged)
    us_vix_level = weekly["us_vix"].rename("us_vix_level")

    # 2. SPX returns
    spx = weekly["spx"]
    spx_ret_1w = spx.pct_change().rename("spx_return_1w")
    spx_ret_4w = spx.pct_change(4).rename("spx_return_4w")

    # 3. SPX 20-day realized volatility (from daily)
    spx_daily = df["spx"].dropna()
    spx_daily_ret = spx_daily.pct_change()
    spx_vol_20d = (
        (spx_daily_ret.rolling(20, min_periods=10).std() * (252 ** 0.5))
        .resample("W-FRI")
        .last()
        .rename("spx_volatility_20d")
    )

    # 4. Crude oil
    crude = weekly["crude"]
    crude_ret_1w = crude.pct_change().rename("crude_return_1w")

    # 5. USD/INR
    usd_inr_level = weekly["usd_inr"].rename("usd_inr_level")
    usd_inr_chg_1w = weekly["usd_inr"].pct_change().rename("usd_inr_change_1w")

    # 6. US 10Y yield
    us_10y_level = weekly["us_10y"].rename("us_10y_level")
    us_10y_chg_1w = weekly["us_10y"].diff().rename("us_10y_change_1w")

    # 7. Term premium proxy (US 10Y yield - India VIX as rough risk-free spread)
    # This will be merged later with India VIX; placeholder here
    term_premium = us_10y_level.rename("us_10y_level_raw")

    frames = [
        us_vix_level.shift(1),
        spx_ret_1w.shift(1),
        spx_ret_4w.shift(1),
        spx_vol_20d.shift(1),
        crude_ret_1w.shift(1),
        usd_inr_level.shift(1),
        usd_inr_chg_1w.shift(1),
        us_10y_level.shift(1),
        us_10y_chg_1w.shift(1),
    ]

    feat = pd.concat(frames, axis=1)
    feat.index.name = "week_end"
    feat = feat.dropna()
    logger.info(f"Macro feature matrix: {feat.shape}")
    return feat


if __name__ == "__main__":
    fetch_macro_daily()
    feat = build_macro_features()
    print(feat.tail())
    print(feat.columns.tolist())
