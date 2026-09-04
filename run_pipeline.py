"""
Nifty 50 Iron Condor — Master Pipeline
========================================
Single entry point for all modes:

  python run_pipeline.py --mode setup       # First-time: fetch data + train + calibrate
  python run_pipeline.py --mode backtest    # Static 80/20 backtest
  python run_pipeline.py --mode walkforward # True expanding-window backtest (periodic retrain)
  python run_pipeline.py --mode live        # Sunday night: get this week's strikes
  python run_pipeline.py --mode macro       # Fetch US/global macro data
  python run_pipeline.py --mode monitor     # Check model drift & coverage decay
  python run_pipeline.py --mode retrain     # Retrain models on latest data

Run once in setup mode, then use live mode every Sunday.
"""

import argparse
import io
import sys
import traceback
from pathlib import Path

# Ensure UTF-8 output on Windows (handles box-drawing characters)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _step(msg: str):
    logger.info(f"{'─'*60}")
    logger.info(f"  {msg}")
    logger.info(f"{'─'*60}")


def _data_exists() -> bool:
    return (BASE_DIR / "data" / "nifty_daily.parquet").exists()


def _models_exist() -> bool:
    required_files = [
        "lgb_low.pkl",
        "lgb_mid.pkl",
        "lgb_high.pkl",
        "feature_columns.pkl",
        "regime_thresholds.json",
        "regime_model_meta.json",
        "regime_lgb_wrapper.pkl",
        "mapie_calibrated.pkl",
        "garch_model.pkl"
    ]
    return all((BASE_DIR / "models" / f).exists() for f in required_files)


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def mode_setup():
    """
    Full first-time setup:
    M1 → M2 → M3 → M4 → M5
    Fetches 5yr data, engineers features, fits GARCH, trains models, calibrates.
    """
    print("\n" + "═"*60)
    print("  SETUP MODE  —  First-time full pipeline")
    print("═"*60 + "\n")

    # M1: Data pipeline
    _step("M1 — Fetching historical data (Yahoo Finance — no credentials needed)")
    from module1_data_pipeline import fetch_nifty_daily, fetch_nifty_intraday, fetch_india_vix, build_nifty_weekly
    daily = fetch_nifty_daily()
    fetch_nifty_intraday()   # 1h bars, 2 years — best available free intraday
    fetch_india_vix()
    build_nifty_weekly(daily)
    logger.success(f"M1 done — {len(daily)} daily bars saved.")

    # M2: Feature engineering
    _step("M2 — Building feature matrix")
    from module2_features import build_features
    feat_df = build_features()
    logger.success(f"M2 done — {feat_df.shape[0]} weekly rows, {feat_df.shape[1]} features.")

    # M3: GARCH
    _step("M3 — Fitting GARCH(1,1) conditional volatility")
    from module3_garch import run_garch_pipeline
    garch_df = run_garch_pipeline()
    logger.success(f"M3 done — {len(garch_df)} rows with GARCH features.")

    # M4: Train LightGBM regime models
    _step("M4 — Training LightGBM per-regime models (low/mid/high VIX)")
    from module4_model import train_models
    eval_results = train_models()
    coverage = eval_results.get("coverage_rate", 0)
    regime_cov = eval_results.get("regime_coverage", {})
    logger.success(f"M4 done — Coverage: {coverage:.1%} | Per-regime: {regime_cov}")

    # M5: Conformal calibration
    _step("M5 — Conformal calibration (MAPIE)")
    from module5_calibration import run_calibration
    cal_report = run_calibration()
    target_cov = cal_report.get('target_coverage', 0.85)
    logger.success(f"M5 done — Coverage @{target_cov:.0%}: {cal_report.get('actual_oos_coverage', '?')}")

    print("\n" + "═"*60)
    print("  SETUP COMPLETE")
    print(f"  Target Coverage : {cal_report.get('target_coverage', '?')}")
    print(f"  Actual OOS Cov  : {cal_report.get('actual_oos_coverage', '?')}")
    print("  → Run backtest:  python run_pipeline.py --mode backtest")
    print("  → Get strikes:   python run_pipeline.py --mode live")
    print("═"*60 + "\n")


def mode_backtest():
    """
    Walk-forward backtest of IC strategy over test set.
    Requires setup mode to have been run first.
    """
    print("\n" + "═"*60)
    print("  BACKTEST MODE  —  Walk-forward P&L simulation")
    print("═"*60 + "\n")

    if not _models_exist():
        print("[ERROR] Models not found. Run --mode setup first.\n")
        sys.exit(1)

    _step("M7 — Running walk-forward backtest")
    from module7_backtest import run_backtest
    summary = run_backtest()

    print("\n" + "═"*60)
    print("  BACKTEST RESULTS")
    print(f"  Win rate         : {summary.get('win_rate_pct', 0):.1f}%")
    print(f"  Total P&L        : ₹{summary.get('total_pnl_inr', 0):,.0f}")
    print(f"  Sharpe ratio     : {summary.get('sharpe_ratio', 0):.2f}")
    print(f"  Max drawdown     : {summary.get('max_drawdown_points', 0):.0f} pts")
    print(f"  Expectancy/trade : {summary.get('expectancy_per_trade_points', 0):.1f} pts")
    low = summary.get('breach_rate_low_vix_pct', 0)
    mid = summary.get('breach_rate_mid_vix_pct', 0)
    high = summary.get('breach_rate_high_vix_pct', 0)
    print(f"  Breach rate (VIX low/mid/high): {low:.0f}% / {mid:.0f}% / {high:.0f}%")
    print(f"\n  Equity curve  → outputs/backtest_equity_curve.png")
    print(f"  Full results  → outputs/backtest_results.csv")
    print("═"*60 + "\n")


def mode_live():
    """
    Sunday-night live prediction: fetches latest data and returns this week's strikes.
    """
    print("\n" + "═"*60)
    print("  LIVE MODE  —  This week's Iron Condor strikes")
    print("═"*60 + "\n")

    if not _models_exist():
        print("[ERROR] Models not found. Run --mode setup first.\n")
        sys.exit(1)

    _step("Updating data + generating strikes")
    from module8_live import run_live_pipeline
    strikes = run_live_pipeline()

    if strikes:
        print(f"\n  Strikes saved → outputs/strikes_live.json\n")


def mode_walkforward():
    """
    Expanding-window walk-forward backtest with periodic retraining.
    Most realistic validation of live performance.
    """
    print("\n" + "═"*60)
    print("  WALK-FORWARD MODE  —  Expanding window + periodic retrain")
    print("═"*60 + "\n")

    if not _data_exists():
        print("[ERROR] Data not found. Run --mode setup first.\n")
        sys.exit(1)

    _step("M7b — Running expanding-window walk-forward backtest")
    from module7b_walkforward import run_walkforward_backtest
    summary = run_walkforward_backtest()

    print("\n" + "═"*60)
    print("  WALK-FORWARD RESULTS")
    print(f"  Win rate         : {summary.get('win_rate_pct', 0):.1f}%")
    print(f"  Total P&L        : ₹{summary.get('total_pnl_inr', 0):,.0f}")
    print(f"  Sharpe ratio     : {summary.get('sharpe_ratio', 0):.2f}")
    print(f"  Max drawdown     : {summary.get('max_drawdown_points', 0):.0f} pts")
    print(f"  Expectancy/trade : {summary.get('expectancy_per_trade_points', 0):.1f} pts")
    print(f"\n  Equity curve  → outputs/walkforward_equity_curve.png")
    print(f"  Full results  → outputs/walkforward_results.csv")
    print("═"*60 + "\n")


def mode_macro():
    """Fetch US/global macro data (US VIX, SPX, crude, USD/INR, US 10Y)."""
    print("\n" + "═"*60)
    print("  MACRO MODE  —  Fetching global macro data")
    print("═"*60 + "\n")

    from module1b_macro import fetch_macro_daily
    fetch_macro_daily()
    print("\nMacro data saved to data/macro_daily.parquet")
    print("Re-run --mode setup or --mode retrain to merge macro features.\n")


def mode_monitor():
    """Check model drift, coverage decay, and feature shift."""
    print("\n" + "═"*60)
    print("  MONITOR MODE  —  Model health check")
    print("═"*60 + "\n")

    from module13_monitor import run_monitor
    run_monitor()


def mode_retrain():
    """
    Retrain models on existing features without modifying backtest set boundaries.
    Skips data fetch (M1) and feature engineering (M2) if they already exist.
    Re-fits GARCH, retrains LightGBM, and recalibrates MAPIE.
    """
    print("\n" + "═"*60)
    print("  RETRAIN MODE  —  Model retrain on existing features")
    print("═"*60 + "\n")

    # Only fetch data if it doesn't exist (incremental)
    if not _data_exists():
        _step("M1 — Incremental data fetch")
        from module1_data_pipeline import fetch_nifty_daily, fetch_nifty_intraday, fetch_india_vix, build_nifty_weekly
        daily = fetch_nifty_daily()
        fetch_nifty_intraday()
        fetch_india_vix()
        build_nifty_weekly(daily)
        logger.success("M1 done — incremental data fetch.")
    else:
        logger.info("M1 skipped — data already exists.")

    # Only rebuild features if they don't exist
    feat_path = BASE_DIR / "data" / "feature_matrix_with_garch.parquet"
    if not feat_path.exists():
        _step("M2 — Building feature matrix")
        from module2_features import build_features
        build_features()
        logger.success("M2 done — feature matrix built.")
    else:
        logger.info("M2 skipped — feature matrix already exists.")

    # M3: Re-fit GARCH
    _step("M3 — Re-fitting GARCH(1,1)")
    from module3_garch import run_garch_pipeline
    garch_df = run_garch_pipeline()
    logger.success(f"M3 done — {len(garch_df)} rows with GARCH features.")

    # M4: Retrain models
    _step("M4 — Retraining LightGBM per-regime models")
    from module4_model import train_models
    eval_results = train_models()
    coverage = eval_results.get("coverage_rate", 0)
    regime_cov = eval_results.get("regime_coverage", {})
    logger.success(f"M4 done — Coverage: {coverage:.1%} | Per-regime: {regime_cov}")

    # M5: Recalibrate
    _step("M5 — Recalibrating conformal intervals")
    from module5_calibration import run_calibration
    cal_report = run_calibration()
    target_cov = cal_report.get('target_coverage', 0.85)
    logger.success(f"M5 done — Coverage @{target_cov:.0%}: {cal_report.get('actual_oos_coverage', '?')}")

    # Invalidate model cache so live mode picks up fresh models
    _step("Clearing model cache")
    from module6_strikes import _clear_model_cache
    _clear_model_cache()
    logger.success("Model cache cleared — live predictions will use retrained models.")

    print("\n" + "═"*60)
    print("  RETRAIN COMPLETE")
    print("═"*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nifty 50 Iron Condor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --mode setup       # First run: fetch data + train
  python run_pipeline.py --mode backtest    # Static 80/20 backtest
  python run_pipeline.py --mode walkforward # True walk-forward (periodic retrain)
  python run_pipeline.py --mode live        # Sunday night: get this week's strikes
  python run_pipeline.py --mode macro       # Fetch US/global macro data
  python run_pipeline.py --mode monitor     # Check model drift & coverage decay
  python run_pipeline.py --mode retrain     # Retrain on latest data (monthly)
        """
    )
    parser.add_argument(
        "--mode",
        choices=["setup", "backtest", "walkforward", "live", "macro", "monitor", "retrain"],
        default="live",
        help="Pipeline mode to run (default: live)"
    )
    args = parser.parse_args()

    try:
        if args.mode == "setup":
            mode_setup()
        elif args.mode == "backtest":
            mode_backtest()
        elif args.mode == "walkforward":
            mode_walkforward()
        elif args.mode == "live":
            mode_live()
        elif args.mode == "macro":
            mode_macro()
        elif args.mode == "monitor":
            mode_monitor()
        elif args.mode == "retrain":
            mode_retrain()
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Pipeline failed in --mode {args.mode}: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
