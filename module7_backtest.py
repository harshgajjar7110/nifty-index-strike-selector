"""
Module 7: Walk-Forward Backtest
Simulates iron condor P&L over the historical test set using model predictions.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "feature_matrix_with_garch.parquet"
WEEKLY_PATH = BASE_DIR / "data" / "nifty_weekly.parquet"
OUTPUTS_DIR = BASE_DIR / "outputs"

# WING_WIDTH_POINTS is now dynamic (see generate_strikes() regime selection)
LOT_SIZE = 25              # Nifty lot size (₹ per point)
SKEW_PTS_PER_PERCENT_IMBALANCE = 25  # empirical: 1% breach diff → 25 pts skew

# ---------------------------------------------------------------------------
# Module 6 import via sys.path
# ---------------------------------------------------------------------------
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from module6_strikes import predict_range, generate_strikes, _load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Load configuration from module6 and environment
# ---------------------------------------------------------------------------
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# Load VIX baseline from module6 config (ensures consistency with live trading)
_, _, VIX_BASELINE, _, _, _ = _load_config()
logger.info(f"VIX_BASELINE loaded from module6 config: {VIX_BASELINE:.2f}")

# Load PREMIUM_POINTS_BASE from environment (with fallback)
PREMIUM_POINTS_BASE = int(os.getenv("PREMIUM_POINTS_BASE", 80))
logger.info(f"PREMIUM_POINTS_BASE loaded from environment: {PREMIUM_POINTS_BASE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """Peak-to-trough drawdown in points."""
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdowns = peak - cumulative_pnl
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def _vix_regime(vix: float) -> str:
    if vix < 15:
        return "low"
    elif vix < 20:
        return "mid"
    return "high"


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def run_backtest() -> dict:
    """
    Walk-forward backtest of the iron condor strategy.

    Returns
    -------
    dict — summary metrics
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run module3_garch.py first")

    logger.info(f"Loading feature matrix from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_index()
    logger.info(f"Feature matrix shape: {df.shape}")

    # Load weekly OHLC for actual closes
    weekly_close: pd.Series | None = None
    if WEEKLY_PATH.exists():
        weekly = pd.read_parquet(WEEKLY_PATH)
        weekly = weekly.sort_index()
        # Normalise index to date only (drop time component if present)
        if hasattr(weekly.index, "normalize"):
            weekly.index = weekly.index.normalize()
        # Try common column names for close price
        for col in ("close", "Close", "CLOSE", "adj_close", "Adj Close"):
            if col in weekly.columns:
                weekly_close = weekly[col]
                break
        if weekly_close is None and len(weekly.columns) > 0:
            weekly_close = weekly.iloc[:, -1]
        logger.info(f"Loaded weekly close from {WEEKLY_PATH}")
    else:
        logger.warning(f"{WEEKLY_PATH} not found — will derive close from feature matrix")

    # Normalise df index to date only
    if hasattr(df.index, "normalize"):
        df.index = df.index.normalize()

    # ------------------------------------------------------------------
    # 2. Test split (last 20%)
    # ------------------------------------------------------------------
    n = len(df)
    split = int(n * 0.80)
    test_df = df.iloc[split:]
    logger.info(f"Test set: {len(test_df)} rows  ({test_df.index[0].date()} → {test_df.index[-1].date()})")

    # ------------------------------------------------------------------
    # 3. Walk-forward prediction loop
    # ------------------------------------------------------------------
    records = []

    for week_end, feature_row in test_df.iterrows():
        # Actual log_range for this week
        actual_log_range = float(feature_row["log_range"]) if "log_range" in feature_row.index else np.nan

        # Derive current_close from weekly parquet or fall back to feature
        current_close: float | None = None
        if weekly_close is not None:
            if week_end in weekly_close.index:
                current_close = float(weekly_close.loc[week_end])

        if current_close is None:
            # Try feature-derived close columns
            for col in ("close", "Close", "weekly_close", "spot"):
                if col in feature_row.index and not np.isnan(feature_row[col]):
                    current_close = float(feature_row[col])
                    break

        if current_close is None or np.isnan(current_close):
            logger.warning(f"Skipping {week_end}: could not determine current_close")
            continue

        # Predict range
        range_pred = predict_range(feature_row)
        log_range_p10 = range_pred["log_range_p10"]
        log_range_p90 = range_pred["log_range_p90"]

        # Generate strikes (with VIX level if available)
        vix_level: float = np.nan
        for col in ("vix", "india_vix", "VIX", "INDIA_VIX", "vix_level"):
            if col in feature_row.index:
                try:
                    val = float(feature_row[col])
                    if val > 0:
                        vix_level = val
                        break
                except (ValueError, TypeError):
                    pass

        # Extract GARCH volatility
        garch_vol = float(feature_row["garch_sigma_mean"]) if "garch_sigma_mean" in feature_row.index else None

        strikes = generate_strikes(
            current_close,
            log_range_p10,
            log_range_p90,
            vix_level=vix_level if not np.isnan(vix_level) else None,
            garch_vol_weekly=garch_vol,
        )

        # Compute per-row premium scaled by VIX
        vix_used = vix_level if not np.isnan(vix_level) else VIX_BASELINE
        premium_pts = int(PREMIUM_POINTS_BASE * (vix_used / VIX_BASELINE))
        premium_pts = max(60, min(120, premium_pts))  # clamp [60, 120]
        wing_width_used = strikes.get("wing_width_pts", 200)  # from result dict
        max_loss_pts = wing_width_used - premium_pts

        records.append({
            "week_end": week_end,
            "current_close": current_close,
            "short_put": strikes["short_put"],
            "short_call": strikes["short_call"],
            "long_put": strikes["long_put"],
            "long_call": strikes["long_call"],
            "log_range_p10": log_range_p10,
            "log_range_p90": log_range_p90,
            "actual_log_range": actual_log_range,
            "vix_level": vix_level,
            "premium_pts": premium_pts,
            "wing_width_pts": wing_width_used,
            "max_loss_pts": max_loss_pts,
            "breach_prob_call": strikes.get("breach_prob_call"),
            "breach_prob_put": strikes.get("breach_prob_put"),
            "prob_of_profit": strikes.get("prob_of_profit"),
        })

    if not records:
        raise RuntimeError("No test records processed — check data and model files.")

    results = pd.DataFrame(records).set_index("week_end")
    logger.info(f"Processed {len(results)} test weeks")

    # ------------------------------------------------------------------
    # 4. P&L simulation (per-row premium)
    # ------------------------------------------------------------------
    pnl_list = []
    won_list = []
    breach_up_list = []
    breach_down_list = []

    for _, row in results.iterrows():
        close = row["current_close"]
        actual_log_range = row["actual_log_range"]

        if np.isnan(actual_log_range):
            pnl_list.append(np.nan)
            won_list.append(np.nan)
            breach_up_list.append(np.nan)
            breach_down_list.append(np.nan)
            continue

        actual_half_range = close * (np.exp(actual_log_range) - 1) / 2
        actual_high = close + actual_half_range
        actual_low = close - actual_half_range

        # Use per-row premium and max_loss
        premium_pts = row["premium_pts"]
        max_loss_pts = row["max_loss_pts"]

        # Track individual breaches
        breach_up = actual_high > row["short_call"]
        breach_down = actual_low < row["short_put"]

        won = (actual_low >= row["short_put"]) and (actual_high <= row["short_call"])
        pnl = premium_pts if won else -max_loss_pts

        pnl_list.append(pnl)
        won_list.append(won)
        breach_up_list.append(breach_up)
        breach_down_list.append(breach_down)

    results["pnl_points"] = pnl_list
    results["won"] = won_list
    results["breach_up"] = breach_up_list
    results["breach_down"] = breach_down_list
    results["pnl_inr"] = results["pnl_points"] * LOT_SIZE

    # Drop weeks where actual_log_range was unavailable
    valid = results.dropna(subset=["pnl_points"])

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    total_trades = len(valid)
    win_count = valid["won"].sum()
    win_rate = win_count / total_trades * 100 if total_trades > 0 else 0.0

    total_pnl_points = valid["pnl_points"].sum()
    total_pnl_inr = valid["pnl_inr"].sum()

    cum_pnl = valid["pnl_points"].cumsum().values
    max_dd = _max_drawdown(cum_pnl)

    pnl_arr = valid["pnl_points"].values
    sharpe = (
        float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(52))
        if np.std(pnl_arr) > 0 else 0.0
    )
    expectancy = float(np.mean(pnl_arr))

    # Breach rate by VIX regime
    regime_stats: dict = {}
    for regime in ("low", "mid", "high"):
        mask = valid["vix_level"].apply(
            lambda v: _vix_regime(v) == regime if not np.isnan(v) else False
        )
        subset = valid[mask]
        if len(subset) > 0:
            breach_rate = float((subset["won"] == False).sum() / len(subset) * 100)  # noqa: E712
        else:
            breach_rate = np.nan
        regime_stats[f"breach_rate_{regime}_vix_pct"] = round(breach_rate, 2) if not np.isnan(breach_rate) else None

    # Compute premium statistics
    premium_stats = {
        "avg_premium_pts": round(float(valid["premium_pts"].mean()), 1),
        "min_premium_pts": int(valid["premium_pts"].min()),
        "max_premium_pts": int(valid["premium_pts"].max()),
    }

    # Compute probability of profit (POP) statistics
    valid_pop = results.dropna(subset=["prob_of_profit"])
    pop_stats = {
        "avg_pop_pct": round(float(valid_pop["prob_of_profit"].mean() * 100), 2) if len(valid_pop) > 0 else None,
        "avg_breach_prob_call_pct": round(float(valid_pop["breach_prob_call"].mean() * 100), 2) if len(valid_pop) > 0 else None,
        "avg_breach_prob_put_pct": round(float(valid_pop["breach_prob_put"].mean() * 100), 2) if len(valid_pop) > 0 else None,
    }

    # Breach symmetry analysis for skew calibration
    breach_up = (valid["breach_up"] == True).sum()  # noqa: E712
    breach_down = (valid["breach_down"] == True).sum()  # noqa: E712
    breach_rate_up = float(breach_up / len(valid) * 100) if len(valid) > 0 else 0.0
    breach_rate_down = float(breach_down / len(valid) * 100) if len(valid) > 0 else 0.0

    # If down breaches more than up, recommend wider put placement
    skew_imbalance = float(breach_rate_down - breach_rate_up)  # positive = more down breaches
    # Rough heuristic: 1% breach difference ≈ 25 pts of skew needed
    recommended_put_skew = int(max(0, skew_imbalance * SKEW_PTS_PER_PERCENT_IMBALANCE))

    summary = {
        "total_trades": int(total_trades),
        "win_rate_pct": round(float(win_rate), 2),
        "total_pnl_points": round(float(total_pnl_points), 2),
        "total_pnl_inr": round(float(total_pnl_inr), 2),
        "max_drawdown_points": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 4),
        "expectancy_per_trade_points": round(expectancy, 4),
        **regime_stats,
        "premium_points_base": PREMIUM_POINTS_BASE,
        "wing_width_points": "dynamic (see backtest_results.csv column 'wing_width_pts')",
        **premium_stats,
        **pop_stats,
        "lot_size": LOT_SIZE,
        "breach_rate_up_pct": round(float(breach_rate_up), 2),
        "breach_rate_down_pct": round(float(breach_rate_down), 2),
        "skew_imbalance_pct": round(float(skew_imbalance), 2),
        "recommended_put_skew_pts": int(recommended_put_skew),
    }

    logger.info(f"Backtest summary: {summary}")

    # ------------------------------------------------------------------
    # 6. Equity curve plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(valid.index, cum_pnl, linewidth=1.5, color="steelblue", label="Cumulative P&L")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(valid.index, cum_pnl, 0, where=(cum_pnl >= 0), alpha=0.2, color="green")
    ax.fill_between(valid.index, cum_pnl, 0, where=(cum_pnl < 0), alpha=0.2, color="red")
    ax.set_title("Iron Condor Backtest — Cumulative P&L (Points)", fontsize=14)
    ax.set_xlabel("Week End")
    ax.set_ylabel("Cumulative P&L (Points)")

    # Add secondary y-axis for POP
    valid_with_pop = valid.dropna(subset=["prob_of_profit"])
    if len(valid_with_pop) > 0:
        ax2 = ax.twinx()
        ax2.plot(valid_with_pop.index, valid_with_pop["prob_of_profit"] * 100,
                 linewidth=1, color="orange", alpha=0.5, label="POP %")
        ax2.set_ylabel("Probability of Profit (%)", color="orange")
        ax2.tick_params(axis='y', labelcolor="orange")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        ax.legend()

    plt.tight_layout()
    equity_path = OUTPUTS_DIR / "backtest_equity_curve.png"
    fig.savefig(equity_path, dpi=150)
    plt.close(fig)
    logger.info(f"Equity curve saved to {equity_path}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    results_path = OUTPUTS_DIR / "backtest_results.csv"
    results.to_csv(results_path)
    logger.info(f"Results saved to {results_path}")

    summary_path = OUTPUTS_DIR / "backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    summary = run_backtest()
    print(summary)
