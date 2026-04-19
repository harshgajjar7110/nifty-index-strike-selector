"""
Nifty 50 Iron Condor — Master Pipeline
========================================
Single entry point for all modes:

  python run_pipeline.py --mode setup       # First-time: fetch data + train + calibrate
  python run_pipeline.py --mode backtest    # Validate historical performance
  python run_pipeline.py --mode live        # Sunday night: get this week's strikes
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

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

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
    return (
        (BASE_DIR / "models" / "lgbm_p10.pkl").exists()
        and (BASE_DIR / "models" / "lgbm_p90.pkl").exists()
    )


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

    # M4: Train LightGBM quantile models
    _step("M4 — Training LightGBM P10/P90 quantile models")
    from module4_model import train_models
    eval_results = train_models()
    coverage = eval_results.get("coverage_rate", 0)
    logger.success(f"M4 done — Coverage: {coverage:.1%} | Pinball P10: {eval_results.get('pinball_p10', '?'):.4f}")

    # M5: Conformal calibration
    _step("M5 — Conformal calibration (MAPIE)")
    from module5_calibration import run_calibration
    cal_report = run_calibration()
    logger.success(f"M5 done — Coverage @85%: {cal_report.get('coverage_at_85', '?')}")

    print("\n" + "═"*60)
    print("  SETUP COMPLETE")
    print(f"  Coverage @80%: {cal_report.get('coverage_at_80', '?')}")
    print(f"  Coverage @85%: {cal_report.get('coverage_at_85', '?')}")
    print(f"  Coverage @90%: {cal_report.get('coverage_at_90', '?')}")
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
    print(f"  Win rate         : {summary.get('win_rate_pct', '?'):.1f}%")
    print(f"  Total P&L        : ₹{summary.get('total_pnl_inr', 0):,.0f}")
    print(f"  Sharpe ratio     : {summary.get('sharpe', '?'):.2f}")
    print(f"  Max drawdown     : {summary.get('max_drawdown_pts', '?')} pts")
    print(f"  Expectancy/trade : {summary.get('expectancy_pts', '?'):.1f} pts")
    vix = summary.get("breach_rate_by_vix", {})
    print(f"  Breach rate (VIX low/mid/high): "
          f"{vix.get('low', '?'):.0%} / {vix.get('mid', '?'):.0%} / {vix.get('high', '?'):.0%}")
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


def mode_retrain():
    """
    Incremental retrain: fetch new data + rebuild features + GARCH + retrain models.
    Use periodically (monthly) to keep models fresh.
    """
    print("\n" + "═"*60)
    print("  RETRAIN MODE  —  Incremental update + retrain")
    print("═"*60 + "\n")

    # Same as setup but data fetch is incremental (M1 handles this automatically)
    mode_setup()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nifty 50 Iron Condor Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --mode setup      # First run: fetch data + train
  python run_pipeline.py --mode backtest   # Check historical performance
  python run_pipeline.py --mode live       # Sunday night: get this week's strikes
  python run_pipeline.py --mode retrain    # Retrain on latest data (monthly)
        """
    )
    parser.add_argument(
        "--mode",
        choices=["setup", "backtest", "live", "retrain"],
        default="live",
        help="Pipeline mode to run (default: live)"
    )
    args = parser.parse_args()

    try:
        if args.mode == "setup":
            mode_setup()
        elif args.mode == "backtest":
            mode_backtest()
        elif args.mode == "live":
            mode_live()
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
