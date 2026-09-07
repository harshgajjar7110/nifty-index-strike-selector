"""
Module 7b: Expanding-Window Walk-Forward Backtest
True out-of-sample simulation with periodic model retraining.

Logic:
  1. Start with initial_train_weeks of history.
  2. Every RETRAIN_EVERY_WEEKS:
     a. Refit GARCH on expanding daily data.
     b. Retrain per-regime LightGBM quantile models.
     c. Calibrate per-regime MAPIE on recent validation window.
  3. For each test week:
     a. Predict range using current models.
     b. Generate IC strikes.
     c. Simulate P&L (with costs, slippage, stop-loss).
  4. Report walk-forward metrics (no lookahead bias).
"""

import json
import sys
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from config import cfg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "feature_matrix_with_garch.parquet"
WEEKLY_PATH = BASE_DIR / "data" / "nifty_weekly.parquet"
DAILY_PATH = BASE_DIR / "data" / "nifty_daily.parquet"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

from module6_strikes import generate_strikes
from spreads.module10_nse_costs import calculate_nse_charges, apply_slippage, estimate_ic_premium
from utils.utils_constants import REGIMES, load_regime_thresholds, assign_regime_series
from utils.models_utils import RegimeLGBQuantileWrapper, extract_pis, predict_mu_sigma_p90

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INITIAL_TRAIN_WEEKS = cfg.wf_initial_train_weeks   # ~2.3 years
RETRAIN_EVERY_WEEKS = cfg.wf_retrain_every_weeks
CALIBRATION_WEEKS = cfg.wf_calibration_weeks
LOT_SIZE = cfg.nifty_lot_size
SL_MULTIPLIER = cfg.wf_sl_multiplier
SLIPPAGE_ENTRY = cfg.wf_slippage_entry
SLIPPAGE_EXIT = cfg.wf_slippage_exit
WF_MIN_PREMIUM_PTS = cfg.wf_min_premium_pts
WF_MAX_VIX_TRADE = cfg.wf_max_vix_trade

# VIX regime thresholds and series assignment
WF_LOW_THRESH, WF_HIGH_THRESH = load_regime_thresholds()

# ---------------------------------------------------------------------------
# GARCH refit
# ---------------------------------------------------------------------------

def _refit_garch_per_quarter(test_df, cache):
    """Refit GARCH per calendar quarter on data <= current quarter end, no lookahead.

    Returns a dict keyed by ``week_end`` Timestamp -> refit ``garch_sigma_mean``
    for that week. The fit is cached per quarter so we only fit ~N_quarters models.
    Silently returns an empty dict if daily data or arch package is missing.
    """
    daily_path = BASE_DIR / "data" / "nifty_daily.parquet"
    if not daily_path.exists():
        logger.warning("nifty_daily.parquet missing -- skipping per-quarter GARCH refit")
        return {}
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch package unavailable -- skipping per-quarter GARCH refit")
        return {}

    try:
        daily = pd.read_parquet(daily_path).sort_index()
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

    overrides = {}
    quarter_starts = pd.to_datetime(test_df.index).to_period("Q").unique()
    for q in quarter_starts:
        if q in cache:
            quarter_sigma = cache[q]
        else:
            q_start = max(q.start_time, closes.index.min())
            q_end = q.end_time
            hist = closes.loc[closes.index <= q_end]
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
                weekly_mean = cond_vol.resample("W-TUE").mean()
                quarter_sigma = weekly_mean.to_dict()
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
            if (
                week_end_ts in test_df.index
                and sigma is not None
                and not (isinstance(sigma, float) and np.isnan(sigma))
            ):
                overrides[week_end_ts] = float(sigma)

    logger.info(f"Per-quarter GARCH: built {len(overrides)} weekly overrides ({len(cache)} quarters fit)")
    return overrides


# ---------------------------------------------------------------------------
# Regime model training (self-contained, no side effects)
# ---------------------------------------------------------------------------




def _train_regime_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    low_thresh: float,
    high_thresh: float,
) -> Tuple[dict, list]:
    """Train per-regime LightGBM quantile models. Reads alphas from .env."""
    feature_columns = list(X_train.columns)
    train_regimes = assign_regime_series(X_train["vix_level"], low_thresh, high_thresh)
    lgb_models = {}

    regime_alphas = {
        "low": (cfg.alpha_low_p10, cfg.alpha_low_p90),
        "mid": (cfg.alpha_mid_p10, cfg.alpha_mid_p90),
        "high": (cfg.alpha_high_p10, cfg.alpha_high_p90),
    }

    for regime in REGIMES:
        mask = (train_regimes == regime).values
        X_r = X_train[mask].values
        y_r = y_train[mask]
        if len(X_r) < 10:
            logger.warning(f"Walk-forward train: regime {regime} only {len(X_r)} samples — skipping")
            continue
        alpha_lower, alpha_upper = regime_alphas[regime]
        train_data = lgb.Dataset(X_r, label=y_r)
        n_samples = len(X_r)
        num_leaves = min(15, max(4, n_samples // 3))
        base_params = {
            "objective": "quantile",
            "metric": "quantile",
            "verbose": -1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 5,
            "max_depth": 4,
            "random_state": 42,
        }
        p10_params = {**base_params, "alpha": alpha_lower, "num_leaves": num_leaves, "learning_rate": 0.1}
        lgb_p10 = lgb.train(p10_params, train_data, num_boost_round=100)
        p90_params = {**base_params, "alpha": alpha_upper, "num_leaves": num_leaves, "learning_rate": 0.1}
        lgb_p90 = lgb.train(p90_params, train_data, num_boost_round=100)
        lgb_models[regime] = {"p10": lgb_p10, "p90": lgb_p90}
        logger.info(f"Walk-forward trained {regime}: alpha={alpha_lower}/{alpha_upper}, n={n_samples}, leaves={num_leaves}")

    return lgb_models, feature_columns


def _calibrate_mapie(
    lgb_models: dict,
    feature_columns: list,
    X_conf: pd.DataFrame,
    y_conf: np.ndarray,
    low_thresh: float,
    high_thresh: float,
    target_coverage: float,
) -> dict:
    """Build a single global MAPIE conformal model (more stable than per-regime with tiny samples)."""
    from mapie.regression import SplitConformalRegressor

    Xc = X_conf.values
    yc = y_conf
    if len(Xc) < 7:
        logger.warning(f"Calibration set too small ({len(Xc)}), returning raw models only")
        return {}

    wrapper = RegimeLGBQuantileWrapper(lgb_models, feature_columns, low_thresh, high_thresh)
    mapie_global = SplitConformalRegressor(estimator=wrapper, confidence_level=target_coverage, prefit=True)
    mapie_global.conformalize(Xc, yc)
    logger.info(f"Global MAPIE calibrated on {len(Xc)} samples")
    return {"global": mapie_global}


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_range_mapie(
    feature_row: pd.Series,
    lgb_models: dict,
    feature_columns: list,
    mapie_per_regime: dict,
    low_thresh: float,
    high_thresh: float,
) -> dict:
    X = feature_row[feature_columns].values.reshape(1, -1)
    vix = float(feature_row["vix_level"])
    regime = "low" if vix < low_thresh else ("mid" if vix < high_thresh else "high")

    mapie_model = mapie_per_regime.get("global")
    if mapie_model is not None:
        _, y_pis = mapie_model.predict_interval(X)
        y_low, y_high = extract_pis(y_pis)
        log_range_p10 = float(y_low[0])
        log_range_p90 = float(y_high[0])
    elif regime in lgb_models:
        models = lgb_models[regime]
        log_range_p10 = float(models["p10"].predict(X)[0])
        log_range_p90 = float(models["p90"].predict(X)[0])
    else:
        log_range_p10, log_range_p90 = cfg.fallback_log_range_p10, cfg.fallback_log_range_p90

    log_range_mu, log_range_sigma, _ = predict_mu_sigma_p90(log_range_p10, log_range_p90, regime)
    return {
        "log_range_p10": log_range_p10,
        "log_range_p90": log_range_p90,
        "log_range_mu": log_range_mu,
        "log_range_sigma": log_range_sigma,
        "regime": regime,
    }


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def run_walkforward_backtest() -> dict:
    logger.info("=" * 60)
    logger.info("  WALK-FORWARD BACKTEST  —  Expanding Window + Periodic Retrain")
    logger.info("=" * 60)

    # Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run module3_garch.py first")
    df = pd.read_parquet(DATA_PATH).sort_index()
    logger.info(f"Feature matrix: {df.shape}")

    # Weekly close for actuals
    weekly_close = None
    if WEEKLY_PATH.exists():
        weekly = pd.read_parquet(WEEKLY_PATH).sort_index()
        if hasattr(weekly.index, "normalize"):
            weekly.index = weekly.index.normalize()
        for col in ("close", "Close", "CLOSE", "adj_close", "Adj Close"):
            if col in weekly.columns:
                weekly_close = weekly[col]
                break
        if weekly_close is None and len(weekly.columns) > 0:
            weekly_close = weekly.iloc[:, -1]

    if hasattr(df.index, "normalize"):
        df.index = df.index.normalize()

    if "log_range" not in df.columns or "vix_level" not in df.columns:
        raise ValueError("Required columns missing")

    # Daily for GARCH refits
    daily_df = pd.read_parquet(DAILY_PATH) if DAILY_PATH.exists() else None
    if daily_df is not None:
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df = daily_df.sort_index()

    n_total = len(df)
    if n_total <= INITIAL_TRAIN_WEEKS + 5:
        raise ValueError(f"Not enough data ({n_total} rows) for initial_train={INITIAL_TRAIN_WEEKS}")

    target_coverage = cfg.target_coverage

    # Determine thresholds from full-data optimization (or use defaults for stability)
    # In walk-forward we can compute on initial train only to avoid lookahead
    vix_train_init = df["vix_level"].iloc[:INITIAL_TRAIN_WEEKS]
    low_thresh = float(vix_train_init.quantile(0.33))
    high_thresh = float(vix_train_init.quantile(0.67))
    logger.info(f"Regime thresholds (initial train): low={low_thresh:.1f}, high={high_thresh:.1f}")

    records = []
    garch_cache: dict = {}
    test_df = df.iloc[INITIAL_TRAIN_WEEKS:]
    garch_override = _refit_garch_per_quarter(test_df, garch_cache)
    current_models = None
    current_mapie = None
    current_features = None
    last_retrain_idx = -999

    # Main loop
    for i in range(INITIAL_TRAIN_WEEKS, n_total):
        week_end = df.index[i]
        feature_row = df.iloc[i].copy()
        
        # 1. Apply per-quarter GARCH override (no lookahead)
        if week_end in garch_override:
            feature_row["garch_sigma_mean"] = garch_override[week_end]

        actual_log_range = float(feature_row["log_range"]) if "log_range" in feature_row.index else np.nan

        # Retrain check
        weeks_since_retrain = i - last_retrain_idx
        if weeks_since_retrain >= RETRAIN_EVERY_WEEKS or current_models is None:
            logger.info(f"[Retrain @ {week_end.date()}] Using {i} weeks of history")
            train_df = df.iloc[:i].copy()

            # Apply per-quarter GARCH override to training features for consistency
            if garch_override:
                train_df["garch_sigma_mean"] = (
                    train_df.index.to_series()
                    .map(lambda d: garch_override.get(d, np.nan))
                    .ffill()
                    .bfill()
                )
                logger.info(f"  Applied {len(garch_override)} GARCH overrides to training features")

            y_train = train_df["log_range"].values
            X_train = train_df.drop(columns=["log_range"])

            # Subset to columns available in both train and predict
            avail_cols = [c for c in X_train.columns if c in feature_row.index]
            X_train = X_train[avail_cols]

            # Train models
            current_models, current_features = _train_regime_models(
                X_train, y_train, low_thresh, high_thresh
            )

            # Calibration window: most recent CALIBRATION_WEEKS
            cal_start = max(0, i - CALIBRATION_WEEKS)
            cal_df = df.iloc[cal_start:i]
            X_conf = cal_df[current_features]
            y_conf = cal_df["log_range"].values
            current_mapie = _calibrate_mapie(
                current_models, current_features, X_conf, y_conf,
                low_thresh, high_thresh, target_coverage,
            )
            last_retrain_idx = i

        # Predict
        range_pred = _predict_range_mapie(
            feature_row, current_models, current_features, current_mapie,
            low_thresh, high_thresh,
        )

        # Actual close
        current_close = None
        if weekly_close is not None and week_end in weekly_close.index:
            current_close = float(weekly_close.loc[week_end])
        if current_close is None:
            for col in ("close", "Close", "weekly_close", "spot"):
                if col in feature_row.index and not np.isnan(feature_row[col]):
                    current_close = float(feature_row[col])
                    break
        if current_close is None or np.isnan(current_close):
            logger.warning(f"Skipping {week_end}: no close price")
            continue

        vix_level = float(feature_row["vix_level"]) if "vix_level" in feature_row.index else 16.0
        garch_vol = float(feature_row["garch_sigma_mean"]) if "garch_sigma_mean" in feature_row.index else None

        strikes = generate_strikes(
            current_close,
            range_pred["log_range_p10"],
            range_pred["log_range_p90"],
            vix_level=vix_level,
            garch_vol_weekly=garch_vol,
            log_range_mu=range_pred.get("log_range_mu"),
            log_range_sigma=range_pred.get("log_range_sigma"),
        )

        # Premium estimate
        vix_used = vix_level if not np.isnan(vix_level) else 16.0
        bs_premium = estimate_ic_premium(
            spot=current_close,
            short_put=strikes["short_put"], long_put=strikes["long_put"],
            short_call=strikes["short_call"], long_call=strikes["long_call"],
            dte_days=5, vix_level=vix_used,
        )
        premium_pts = max(bs_premium, 5.0)
        wing_width_used = strikes.get("wing_width_pts", 200)
        max_loss_pts = max(wing_width_used - premium_pts, 1.0)

        records.append({
            "week_end": week_end,
            "current_close": current_close,
            "short_put": strikes["short_put"],
            "short_call": strikes["short_call"],
            "long_put": strikes["long_put"],
            "long_call": strikes["long_call"],
            "log_range_p10": range_pred["log_range_p10"],
            "log_range_p90": range_pred["log_range_p90"],
            "regime": range_pred["regime"],
            "actual_log_range": actual_log_range,
            "vix_level": vix_level,
            "premium_pts": premium_pts,
            "wing_width_pts": wing_width_used,
            "max_loss_pts": max_loss_pts,
        })

    if not records:
        raise RuntimeError("No records processed")

    results = pd.DataFrame(records).set_index("week_end")
    logger.info(f"Processed {len(results)} walk-forward weeks")

    # P&L simulation with position sizing + circuit breakers
    pnl_gross_list = []
    pnl_net_list = []
    won_list = []
    breach_up_list = []
    breach_down_list = []
    costs_list = []
    size_mult_list = []
    skipped_list = []

    peak_equity = 0.0
    current_equity = 0.0

    regime_series = assign_regime_series(results["vix_level"], WF_LOW_THRESH, WF_HIGH_THRESH)
    for idx, regime in regime_series.items():
        row = results.loc[idx]
        close = row["current_close"]
        actual_log_range = row["actual_log_range"]
        vix_level = row["vix_level"]
        premium_pts = row["premium_pts"]

        # --- Filters ---
        # 1. Premium floor: skip low-premium weeks (bad R/R)
        if premium_pts < WF_MIN_PREMIUM_PTS:
            pnl_gross_list.append(np.nan)
            pnl_net_list.append(np.nan)
            won_list.append(np.nan)
            breach_up_list.append(np.nan)
            breach_down_list.append(np.nan)
            costs_list.append(0.0)
            size_mult_list.append(0.0)
            skipped_list.append(True)
            continue

        # 2. VIX ceiling: skip extreme vol (breach rate 25%+)
        if vix_level > WF_MAX_VIX_TRADE:
            pnl_gross_list.append(np.nan)
            pnl_net_list.append(np.nan)
            won_list.append(np.nan)
            breach_up_list.append(np.nan)
            breach_down_list.append(np.nan)
            costs_list.append(0.0)
            size_mult_list.append(0.0)
            skipped_list.append(True)
            continue

        size_mult = 1.0

        if (
            vix_level >= cfg.wf_soft_vix_lower
            and vix_level <= WF_MAX_VIX_TRADE
        ):
            size_mult = cfg.wf_soft_size_mult
            logger.debug(f"Soft-VIX band: vix={vix_level:.1f} → size_mult={size_mult}")

        if np.isnan(actual_log_range):
            pnl_gross_list.append(np.nan)
            pnl_net_list.append(np.nan)
            won_list.append(np.nan)
            breach_up_list.append(np.nan)
            breach_down_list.append(np.nan)
            costs_list.append(0.0)
            size_mult_list.append(size_mult)
            skipped_list.append(False)
            continue

        actual_half_range = close * (np.exp(actual_log_range) - 1) / 2
        actual_high = close + actual_half_range
        actual_low = close - actual_half_range
        max_loss_pts = row["max_loss_pts"]
        entry_premium_net = apply_slippage(premium_pts, num_legs=4, slippage_per_leg=SLIPPAGE_ENTRY)

        breach_up = actual_high > row["short_call"]
        breach_down = actual_low < row["short_put"]
        won = not (breach_up or breach_down)

        if won:
            gross_pnl = entry_premium_net * size_mult
        else:
            gross_pnl = (entry_premium_net - (SL_MULTIPLIER * premium_pts)) * size_mult
            # Cap loss at maximum possible (using max for negative cap bounds)
            gross_pnl = max(gross_pnl, -(max_loss_pts + (4 * SLIPPAGE_EXIT)) * size_mult)

        entry_charges = calculate_nse_charges(premium_pts, num_legs=4, is_sell=True)
        exit_charges = calculate_nse_charges(max_loss_pts if not won else premium_pts, num_legs=4, is_sell=False)
        total_charges_pts = (entry_charges["cost_per_lot_pts"] + exit_charges["cost_per_lot_pts"]) * size_mult
        # Exit slippage only applies on losing (early exited) trades. Options that expire worthless have no exit slippage.
        exit_slippage_pts = 0 if won else (4 * SLIPPAGE_EXIT * size_mult)
        net_pnl = gross_pnl - total_charges_pts - exit_slippage_pts

        pnl_gross_list.append(gross_pnl)
        pnl_net_list.append(net_pnl)
        won_list.append(won)
        breach_up_list.append(breach_up)
        breach_down_list.append(breach_down)
        costs_list.append(total_charges_pts)
        size_mult_list.append(size_mult)
        skipped_list.append(False)

        current_equity += net_pnl
        peak_equity = max(peak_equity, current_equity)

    results["pnl_points"] = pnl_net_list
    results["pnl_gross"] = pnl_gross_list
    results["won"] = won_list
    results["breach_up"] = breach_up_list
    results["breach_down"] = breach_down_list
    results["pnl_inr"] = results["pnl_points"] * LOT_SIZE
    results["txn_costs_pts"] = costs_list
    results["size_mult"] = size_mult_list
    results["skipped"] = skipped_list

    valid = results.dropna(subset=["pnl_points"]).reset_index(drop=True)
    total_trades = len(valid)
    win_count = valid["won"].sum()
    win_rate = win_count / total_trades * 100 if total_trades > 0 else 0.0
    total_pnl_points = valid["pnl_points"].sum()
    total_pnl_inr = valid["pnl_inr"].sum()
    cum_pnl = valid["pnl_points"].cumsum().values
    max_dd = float(np.max(np.maximum.accumulate(cum_pnl) - cum_pnl)) if len(cum_pnl) > 0 else 0.0
    pnl_arr = valid["pnl_points"].values
    sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(52)) if np.std(pnl_arr) > 0 else 0.0
    expectancy = float(np.mean(pnl_arr))

    # Breach rate by VIX regime
    low_thresh, high_thresh = load_regime_thresholds()
    regime_stats = {}
    for regime in ("low", "mid", "high"):
        mask = valid["vix_level"].apply(lambda v: ("low" if v < low_thresh else ("mid" if v < high_thresh else "high")) == regime if not np.isnan(v) else False)
        subset = valid[mask]
        regime_stats[f"breach_rate_{regime}_vix_pct"] = round(float((subset["won"] == False).sum() / len(subset) * 100), 2) if len(subset) > 0 else None

    n_skipped = int(results["skipped"].sum())
    avg_size_mult = float(results["size_mult"].mean())

    summary = {
        "total_trades": int(total_trades),
        "skipped_trades": n_skipped,
        "avg_size_mult": round(avg_size_mult, 3),
        "win_rate_pct": round(float(win_rate), 2),
        "total_pnl_points": round(float(total_pnl_points), 2),
        "total_pnl_inr": round(float(total_pnl_inr), 2),
        "max_drawdown_points": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 4),
        "expectancy_per_trade_points": round(expectancy, 4),
        **regime_stats,
        "lot_size": LOT_SIZE,
        "retrain_every_weeks": RETRAIN_EVERY_WEEKS,
        "calibration_weeks": CALIBRATION_WEEKS,
        "initial_train_weeks": INITIAL_TRAIN_WEEKS,
    }

    logger.info(f"Walk-forward summary: {summary}")

    # Equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(valid.index, cum_pnl, linewidth=1.5, color="steelblue", label="Cumulative P&L")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(valid.index, cum_pnl, 0, where=(cum_pnl >= 0), alpha=0.2, color="green")
    ax.fill_between(valid.index, cum_pnl, 0, where=(cum_pnl < 0), alpha=0.2, color="red")
    ax.set_title("Walk-Forward Backtest — Expanding Window + Periodic Retrain", fontsize=14)
    ax.set_xlabel("Week End")
    ax.set_ylabel("Cumulative P&L (Points)")
    ax.legend()
    plt.tight_layout()
    eq_path = OUTPUTS_DIR / "walkforward_equity_curve.png"
    fig.savefig(eq_path, dpi=150)
    plt.close(fig)
    logger.info(f"Equity curve saved to {eq_path}")

    results.to_csv(OUTPUTS_DIR / "walkforward_results.csv")
    with open(OUTPUTS_DIR / "walkforward_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved.")
    return summary


if __name__ == "__main__":
    summary = run_walkforward_backtest()
    print(summary)
