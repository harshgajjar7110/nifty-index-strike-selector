"""
Nifty 50 Iron Condor — Master Pipeline
========================================
Single entry point for all modes:

  # SETUP & TRAINING (one-time)
  python run_pipeline.py --mode setup              # Fetch data + train + calibrate
  python run_pipeline.py --mode regime-train       # Train HMM regime detector (Phase 1)
  
  # ANALYSIS & FORECASTING
  python run_pipeline.py --mode forecast-vol       # Generate vol forecasts (Phase 1)
  python run_pipeline.py --mode analyze-regime     # Current regime analysis (Phase 1)
  
  # STRIKE SELECTION (NEW - Phase 2.5)
  python run_pipeline.py --mode select-strikes     # Optimal strikes + EV analysis
  
  # VALIDATION & EXECUTION
  python run_pipeline.py --mode backtest           # Walk-forward validation
  python run_pipeline.py --mode live               # This week's strikes (original)
  python run_pipeline.py --mode retrain            # Incremental retrain (monthly)

Typical workflow:
  1. One-time: python run_pipeline.py --mode setup
  2. One-time: python run_pipeline.py --mode regime-train
  3. Weekly:   python run_pipeline.py --mode select-strikes
  4. Sunday:   python run_pipeline.py --mode live
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


def mode_regime_train():
    """
    Phase 1: Train/retrain HMM regime detection model on historical features.
    """
    print("\n" + "═"*60)
    print("  REGIME TRAINING MODE  —  HMM Regime Detection")
    print("═"*60 + "\n")

    _step("M1-M2 — Data & Feature pipeline")
    from module1_data_pipeline import run_pipeline as data_pipeline
    from module2_features import build_features
    
    data_pipeline()
    features_df = build_features()
    logger.success(f"Features ready: {features_df.shape}")

    _step("Phase 1.3 — Training HMM Regime Detector")
    from engine_regime_detection import train_hmm, save_hmm_model
    
    model_dict = train_hmm(
        features_df,
        test_size=0.2,
        n_iter=100,
        random_state=42,
    )
    save_hmm_model(model_dict)
    
    print(f"\n  HMM Model trained successfully")
    print(f"  States: Quiet Bull / Range Bound / High Expansion / Panic")
    print(f"  Test likelihood: {model_dict['test_likelihood']:.4f}")
    print("═"*60 + "\n")


def mode_forecast_vol():
    """
    Phase 1: Generate volatility forecasts (GARCH/EWMA/historical ensemble).
    """
    print("\n" + "═"*60)
    print("  VOLATILITY FORECAST MODE")
    print("═"*60 + "\n")

    _step("M1 — Fetching latest data")
    from module1_data_pipeline import fetch_nifty_daily
    
    daily = fetch_nifty_daily()
    logger.success(f"Daily data: {len(daily)} rows")

    _step("Phase 1.4 — Generating Volatility Forecasts")
    from engine_volatility_forecast import generate_forecast_report, save_forecast_report
    
    report = generate_forecast_report(daily, lookback_days=252)
    save_forecast_report(report)
    
    print(f"\n  Current Volatility: {report['current_vol']:.4f} ({report['current_regime']})")
    print(f"  Weekly Forecast:   {report['forecast_weekly']['ensemble'][0]:.4f} ({report['forecast_weekly']['regime']})")
    print(f"  Swing Forecast:    {report['forecast_swing']['ensemble'][0]:.4f} ({report['forecast_swing']['regime']})")
    print(f"  Monthly Forecast:  {report['forecast_monthly']['ensemble'][0]:.4f} ({report['forecast_monthly']['regime']})")
    print("═"*60 + "\n")


def mode_analyze_regime():
    """
    Phase 1: Analyze current market regime using trained HMM.
    """
    print("\n" + "═"*60)
    print("  REGIME ANALYSIS MODE")
    print("═"*60 + "\n")

    _step("Loading latest features")
    from module2_features import build_features
    
    try:
        features_df = build_features()
    except Exception as e:
        logger.error(f"Failed to build features: {e}")
        sys.exit(1)

    _step("Loading trained HMM model")
    from engine_regime_detection import load_hmm_model, predict_regime_probabilities
    
    try:
        hmm_model, metadata = load_hmm_model()
        feature_cols = metadata.get("feature_cols", [])
    except FileNotFoundError as e:
        logger.error(f"{e}. Train first using --mode regime-train")
        sys.exit(1)

    _step("Predicting current regime")
    probs = predict_regime_probabilities(features_df, hmm_model, feature_cols, lookback=1)
    
    print(f"\n  Most Likely Regime: {probs['most_likely_regime']}")
    print(f"  Allocation Score: {probs['allocation_score']}/100")
    print(f"  Allocation Factor: {probs['allocation_factor']:.1%}")
    print(f"  Confidence: {probs['confidence']:.1%}")
    print(f"\n  Regime Probabilities:")
    for regime, prob in probs["probabilities"].items():
        print(f"    {regime:20} : {prob:.1%}")
    print("═"*60 + "\n")


def mode_select_strikes():
    """
    Phase 2.5: Select optimal strikes combining regime, vol, probability, and EV analysis.
    Generates actionable strike recommendations for paper trading.
    """
    print("\n" + "═"*60)
    print("  STRIKE SELECTION MODE  —  Phase 2.5")
    print("═"*60 + "\n")

    # Fetch latest market data
    _step("Fetching latest market data")
    from module1_data_pipeline import fetch_nifty_daily, fetch_india_vix, fetch_live_spot_yf
    
    daily = fetch_nifty_daily()
    spot = fetch_live_spot_yf()
    logger.success(f"Spot price: {spot:.2f}")

    # Get current regime
    _step("Loading current market regime")
    from module2_features import build_features
    from engine_regime_detection import load_hmm_model, predict_regime_probabilities
    
    try:
        features_df = build_features()
        hmm_model, metadata = load_hmm_model()
        feature_cols = metadata.get("feature_cols", [])
        regime_probs = predict_regime_probabilities(features_df, hmm_model, feature_cols, lookback=1)
        regime_factor = regime_probs['allocation_factor']
        logger.info(f"Regime: {regime_probs['most_likely_regime']} | Allocation: {regime_factor:.0%}")
    except Exception as e:
        logger.warning(f"Regime analysis failed: {e}. Using default allocation 100%")
        regime_factor = 1.0

    # Generate volatility forecast
    _step("Generating volatility forecast")
    from engine_volatility_forecast import generate_forecast_report
    
    vol_report = generate_forecast_report(daily, lookback_days=252)
    vol_weekly = vol_report['forecast_weekly']['ensemble'][0]
    iv_rank = vol_report.get('iv_rank', 50.0)
    logger.info(f"Weekly vol forecast: {vol_weekly:.4f} | IV rank: {iv_rank:.0f}")

    # Select strikes for upcoming expiry (weekly)
    _step("Selecting optimal strikes")
    from engine_strike_selection import generate_strike_recommendations, save_strike_recommendations
    
    dte = 7  # Weekly expiry
    recommendations = generate_strike_recommendations(
        spot=spot,
        volatility=vol_weekly,
        dte=dte,
        regime_allocation_factor=regime_factor,
        iv_rank=iv_rank,
    )
    
    save_strike_recommendations(recommendations)

    # Display results
    print(f"\n" + "═"*60)
    print(f"  STRIKE RECOMMENDATIONS")
    
    if recommendations['status'] == 'SUCCESS':
        print(f"  Status: ✓ Tradeable")
        print(f"  Horizon: {recommendations['horizon'].upper()}")
        print(f"  DTE: {recommendations['market']['dte']:.0f} | IV Rank: {recommendations['market']['iv_rank']:.0f}")
        print(f"  Regime Allocation: {recommendations['market']['regime_allocation']:.0%}")
        
        if 'put_recommendations' in recommendations and 'best_strike' in recommendations['put_recommendations']:
            pe = recommendations['put_recommendations']['best_strike']
            print(f"\n  📉 SHORT PUT")
            print(f"     Strike: {pe.get('strike', 'N/A'):.2f}")
            print(f"     Delta: {pe.get('delta', 'N/A'):.4f} | Score: {pe.get('opportunity_score', 'N/A'):.1f}")
            print(f"     EV/contract: ₹{pe.get('expected_value', 0) * 100:.0f}")
        
        if 'call_recommendations' in recommendations and 'best_strike' in recommendations['call_recommendations']:
            ce = recommendations['call_recommendations']['best_strike']
            print(f"\n  📈 SHORT CALL")
            print(f"     Strike: {ce.get('strike', 'N/A'):.2f}")
            print(f"     Delta: {ce.get('delta', 'N/A'):.4f} | Score: {ce.get('opportunity_score', 'N/A'):.1f}")
            print(f"     EV/contract: ₹{ce.get('expected_value', 0) * 100:.0f}")
        
        if 'multi_leg_strategies' in recommendations and 'strangle' in recommendations['multi_leg_strategies']:
            strangle = recommendations['multi_leg_strategies']['strangle']
            print(f"\n  🎯 RECOMMENDED STRATEGY: SHORT STRANGLE")
            print(f"     Total EV: ₹{strangle.get('total_expected_value', 0) * 100:.0f}")
            print(f"     Opportunity Score: {strangle.get('opportunity_score', 0):.1f}")
            print(f"     Portfolio Greeks (Delta/Gamma/Vega/Theta):")
            greeks = strangle.get('portfolio_greeks', {})
            print(f"       Δ {greeks.get('delta', 0):.4f} | Γ {greeks.get('gamma', 0):.6f} | V {greeks.get('vega', 0):.3f} | Θ {greeks.get('theta', 0):.3f}")
    else:
        print(f"  Status: ✗ {recommendations['status']}")
        print(f"  Reason: {recommendations.get('reason', 'N/A')}")
    
    print(f"\n  Recommendations saved → data/strikes_recommendation_latest.json")
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
  python run_pipeline.py --mode setup           # First run: fetch data + train
  python run_pipeline.py --mode regime-train    # Train HMM regime detector
  python run_pipeline.py --mode forecast-vol    # Generate vol forecasts
  python run_pipeline.py --mode analyze-regime  # Current regime analysis
  python run_pipeline.py --mode backtest        # Check historical performance
  python run_pipeline.py --mode live            # Sunday night: get this week's strikes
  python run_pipeline.py --mode retrain         # Retrain on latest data (monthly)
        """
    )
    parser.add_argument(
        "--mode",
        choices=["setup", "backtest", "live", "retrain", "regime-train", "forecast-vol", "analyze-regime", "select-strikes"],
        default="live",
        help="Pipeline mode to run (default: live)"
    )
    args = parser.parse_args()

    try:
        if args.mode == "setup":
            mode_setup()
        elif args.mode == "regime-train":
            mode_regime_train()
        elif args.mode == "forecast-vol":
            mode_forecast_vol()
        elif args.mode == "analyze-regime":
            mode_analyze_regime()
        elif args.mode == "select-strikes":
            mode_select_strikes()
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
