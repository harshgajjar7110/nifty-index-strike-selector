"""Shared per-quarter GARCH refit helper (no lookahead).

Single source of truth for the quarterly GARCH refit used by the static
backtest (module7) and the expanding-window walk-forward (module7b).

No-lookahead contract: for each calendar quarter, the GARCH model is fit on
daily log-returns dated <= (first test week of the quarter - 1 day). The
fitted terminal conditional volatility is then applied forward to every test
week in that quarter, so no week `t` ever uses returns dated after `t`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config import REPO_ROOT


def refit_garch_per_quarter(
    test_df: pd.DataFrame,
    cache: dict,
    daily_path: Path | None = None,
) -> dict:
    """Refit GARCH per calendar quarter with no lookahead.

    Returns a dict keyed by ``week_end`` Timestamp -> refit ``garch_sigma_mean``
    for that week. The fit is cached per quarter so only ~N_quarters models are
    fit. Silently returns an empty dict if the daily data file is missing or
    the ``arch`` package is unavailable.
    """
    daily_path = daily_path or (REPO_ROOT / "data" / "nifty_daily.parquet")
    if not daily_path.exists():
        logger.warning("nifty_daily.parquet missing -- skipping per-quarter GARCH refit")
        return {}
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch package unavailable -- skipping per-quarter GARCH refit")
        return {}

    try:
        daily = pd.read_parquet(daily_path)
        daily = daily.sort_index()
        close_col = None
        for col in ("close", "Close", "CLOSE", "adj_close", "Adj Close"):
            if col in daily.columns:
                close_col = col
                break
        if close_col is None and len(daily.columns) > 0:
            close_col = daily.columns[0]
        if close_col is None:
            return {}

        closes = daily[close_col]
        if hasattr(closes.index, "normalize"):
            closes.index = closes.index.normalize()
    except Exception as e:
        logger.warning(f"Per-quarter GARCH: failed to load daily data ({e})")
        return {}

    overrides: dict = {}
    quarter_starts = pd.to_datetime(test_df.index).to_period("Q").unique()
    for q in quarter_starts:
        if q in cache:
            quarter_sigma = cache[q]
        else:
            # Fit only through the last day before the first test week of this
            # quarter. Never use returns dated after the week being predicted.
            q_weeks = pd.to_datetime(test_df.index[test_df.index.to_period("Q") == q]).sort_values()
            if len(q_weeks) == 0:
                cache[q] = None
                continue
            first_test_week = q_weeks[0]
            fit_end = first_test_week - pd.Timedelta(days=1)
            hist = closes.loc[closes.index <= fit_end]
            if len(hist) < 60:
                cache[q] = None
                continue
            returns = np.log(hist / hist.shift(1)).dropna() * 100
            if len(returns) < 60:
                cache[q] = None
                continue
            try:
                model = arch_model(returns, vol="Garch", p=1, o=1, q=1, dist="skewt")
                res = model.fit(disp="off")
                cond_vol = res.conditional_volatility / 100
                # Apply the fitted terminal volatility forward to all test weeks
                # in this quarter. This is a forecast from the fit endpoint, so
                # no week in the quarter uses returns dated after itself.
                terminal_sigma = float(cond_vol.iloc[-1])
                quarter_sigma = {week_end: terminal_sigma for week_end in q_weeks}
            except Exception as e:
                logger.warning(f"Per-quarter GARCH fit failed for {q}: {e}")
                quarter_sigma = None
            cache[q] = quarter_sigma
        if not quarter_sigma:
            continue
        for week_end, sigma in quarter_sigma.items():
            try:
                week_end_ts = pd.Timestamp(week_end).normalize()
            except Exception:
                continue
            if week_end_ts in test_df.index and sigma is not None and not (isinstance(sigma, float) and np.isnan(sigma)):
                overrides[week_end_ts] = float(sigma)

    logger.info(f"Per-quarter GARCH: built {len(overrides)} weekly overrides "
                f"({len(cache)} quarters fit)")
    return overrides
