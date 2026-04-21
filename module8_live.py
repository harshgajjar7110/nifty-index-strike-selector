"""
Module 8: Live Sunday-Night Orchestration
Run before each weekly expiry to fetch latest data, rebuild features/GARCH,
and generate iron condor strike recommendations.

Usage:
    # Run every Sunday before market open:
    # venv/Scripts/python module8_live.py
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

STRIKES_JSON = OUTPUTS_DIR / "strikes_live.json"
FEATURE_PARQUET = BASE_DIR / "data" / "feature_matrix_with_garch.parquet"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Format a float as a comma-separated integer string."""
    return f"{int(round(value)):,}"


def _print_table(strikes: dict, week_date: str) -> None:
    """Print a formatted iron condor summary table to stdout."""
    spot = strikes["current_close"]
    short_put = strikes["short_put"]
    short_call = strikes["short_call"]
    long_put = strikes["long_put"]
    long_call = strikes["long_call"]
    p10 = strikes["predicted_range_p10"]
    p90 = strikes["predicted_range_p90"]
    buffer_pts = strikes["buffer_pts"]
    effective_buffer = strikes.get("effective_buffer_pts", buffer_pts)
    wing_width = strikes["wing_width_pts"]
    vix_level = strikes.get("vix_level", "?")
    vix_baseline = strikes.get("vix_baseline", "?")
    put_skew_pts = strikes.get("put_skew_pts", 0)

    sep = "=" * 43
    thin = "-" * 43

    print(sep)
    print(f"     NIFTY IRON CONDOR — WEEK OF {week_date}")
    print(sep)
    print(f"Nifty Spot        :  {_fmt(spot)}")
    print(f"Short PUT strike  :  {_fmt(short_put)}")
    print(f"Short CALL strike :  {_fmt(short_call)}")
    print(f"Long  PUT  strike :  {_fmt(long_put)}  (wing)")
    print(f"Long  CALL strike :  {_fmt(long_call)}  (wing)")
    print(thin)
    print(f"Predicted range   :  {int(round(p10))} - {int(round(p90))} pts (P10-P90)")
    vix_str = vix_level if isinstance(vix_level, str) else f"{vix_level:.1f}"
    vix_base_str = vix_baseline if isinstance(vix_baseline, str) else f"{vix_baseline:.1f}"
    print(f"Current VIX       :  {vix_str}  (baseline: {vix_base_str})")
    print(f"Buffer applied    :  {int(buffer_pts)} pts -> {int(effective_buffer)} pts (VIX-scaled)")
    if put_skew_pts > 0:
        print(f"Put skew applied  :  {int(put_skew_pts)} pts")
    call_skew_pts = strikes.get("call_skew_pts", 0)
    if call_skew_pts > 0:
        print(f"Call skew applied :  {int(call_skew_pts)} pts")
    print(f"Wing width        :  {int(wing_width)} pts")

    # Display probability of profit if available
    pop_pct = strikes.get("prob_of_profit")
    bc_pct = strikes.get("breach_prob_call")
    bp_pct = strikes.get("breach_prob_put")

    if pop_pct is not None:
        print(f"Prob of Profit    :  {pop_pct*100:.1f}%  (call {bc_pct*100:.1f}% / put {bp_pct*100:.1f}% breach)")
    print(sep)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_live_pipeline() -> dict:
    """
    Run the full Sunday-night live pipeline.

    Returns
    -------
    dict — strike recommendations (also written to outputs/strikes_live.json)
    """
    import pandas as pd

    strikes: dict = {}
    step = "initialisation"

    try:
        # ------------------------------------------------------------------
        # Step 1 — Incremental data update
        # ------------------------------------------------------------------
        step = "data fetch (module1)"
        logger.info(f"[Step 1] {step}")
        from module1_data_pipeline import (
            fetch_nifty_daily,
            fetch_india_vix,
            build_nifty_weekly,
            fetch_nifty_intraday,
        )

        daily_df = fetch_nifty_daily()
        fetch_india_vix()
        fetch_nifty_intraday()
        build_nifty_weekly(daily_df)
        logger.success("Step 1 complete — data updated.")

        # ------------------------------------------------------------------
        # Step 2 — Rebuild feature matrix
        # ------------------------------------------------------------------
        step = "feature rebuild (module2)"
        logger.info(f"[Step 2] {step}")
        from module2_features import build_features

        build_features()
        logger.success("Step 2 complete — features rebuilt.")

        # ------------------------------------------------------------------
        # Step 3 — Update GARCH
        # ------------------------------------------------------------------
        step = "GARCH pipeline (module3)"
        logger.info(f"[Step 3] {step}")
        from module3_garch import run_garch_pipeline

        run_garch_pipeline()
        logger.success("Step 3 complete — GARCH updated.")

        # ------------------------------------------------------------------
        # Step 4 — Load latest feature row
        # ------------------------------------------------------------------
        step = "load feature matrix"
        logger.info(f"[Step 4] {step}")
        feat_df = pd.read_parquet(FEATURE_PARQUET)
        feature_row = feat_df.iloc[-1]
        logger.info(f"Using feature row for week ending: {feature_row.name}")

        # Extract VIX level and GARCH vol from feature row
        vix_level = None
        for col in ("vix_level", "vix", "india_vix", "VIX", "INDIA_VIX"):
            if col in feature_row.index:
                try:
                    vix_level = float(feature_row[col])
                    if vix_level > 0:
                        break
                except (ValueError, TypeError):
                    pass

        garch_vol = None
        if "garch_sigma_mean" in feature_row.index:
            try:
                garch_vol = float(feature_row["garch_sigma_mean"])
            except (ValueError, TypeError):
                pass

        # ------------------------------------------------------------------
        # Step 5 — Live spot price
        # ------------------------------------------------------------------
        step = "fetch live spot (module6)"
        logger.info(f"[Step 5] {step}")
        from module6_strikes import fetch_live_spot

        spot = fetch_live_spot()
        logger.info(f"Live Nifty spot: {spot:,.2f}")

        # ------------------------------------------------------------------
        # Step 6 — Generate strikes
        # ------------------------------------------------------------------
        step = "predict range & generate strikes (module6)"
        logger.info(f"[Step 6] {step}")
        from module6_strikes import predict_range, generate_strikes, _load_config

        # Load put_skew_pts and call_skew_pts from config (single source of truth)
        _, _, _, put_skew_pts, _, call_skew_pts = _load_config()

        range_pred = predict_range(feature_row)
        strikes = generate_strikes(
            spot,
            range_pred["log_range_p10"],
            range_pred["log_range_p90"],
            vix_level=vix_level,
            put_skew_pts=put_skew_pts,
            call_skew_pts=call_skew_pts,
            garch_vol_weekly=garch_vol,
        )

        # ------------------------------------------------------------------
        # Step 7 — Output
        # ------------------------------------------------------------------
        step = "output"
        week_label = date.today().isoformat()
        strikes["week_of"] = week_label

        # Console table
        _print_table(strikes, week_label)

        # JSON file
        with open(STRIKES_JSON, "w") as fh:
            json.dump(strikes, fh, indent=2)
        logger.success(f"Strikes saved to {STRIKES_JSON}")

        # ------------------------------------------------------------------
        # Step 8 — Credit Spreads (module9) — non-blocking additive
        # ------------------------------------------------------------------
        step = "credit spread generation (module9)"
        logger.info(f"{'─'*60}")
        logger.info(f"  {step}")
        logger.info(f"{'─'*60}")

        try:
            from module9_spreads import generate_all_spreads
            
            spreads_result = generate_all_spreads(
                feature_row=feature_row,
                spot=spot,
                vix_level=vix_level if vix_level else 16.0,
                garch_vol=garch_vol,
            )
            
            # Write output
            SPREADS_JSON = OUTPUTS_DIR / "spreads_live.json"
            with open(SPREADS_JSON, "w") as fh:
                json.dump(spreads_result, fh, indent=2, default=str)
            logger.success(f"Credit spreads → {SPREADS_JSON}")
            
            # Console summary
            spreads = spreads_result.get("spreads", [])
            top_bull = next((s for s in spreads if s["spread_type"] == "bull_put"), None)
            top_bear = next((s for s in spreads if s["spread_type"] == "bear_call"), None)

            if top_bull or top_bear:
                print(f"\n  ★ Top Credit Spreads by Regime")
                
                for label, s in [("Bull Put ", top_bull), ("Bear Call", top_bear)]:
                    if s:
                        meets = "✓" if s.get("meets_min_rr") else "✗"
                        print(f"    {label}:  {s['short_strike']}/{s['long_strike']} | Expiry: {s['expiry_date']} ({s['expiry_type']})")
                        print(f"                Premium: {s['premium_pts']:.1f} pts | R:R: {s['rr_ratio']:.2%} | POP: {s.get('pop_pct', 0)*100:.1f}% | Min RR: {meets}")
            
        except ImportError:
            logger.warning("module9_spreads not found — skipping credit spread generation")
        except Exception as e9:
            logger.warning(f"Credit spread generation failed (non-fatal): {e9}")
            import traceback
            traceback.print_exc()

    except SystemExit:
        # Propagate intentional exits (e.g. missing .env)
        raise

    except Exception as exc:  # noqa: BLE001
        logger.error(f"Pipeline failed at step '{step}': {exc}")

        # Attempt to show last known strikes
        if STRIKES_JSON.exists():
            logger.warning("Showing last known strikes from cache.")
            try:
                with open(STRIKES_JSON) as fh:
                    strikes = json.load(fh)
                week_label = strikes.get("week_of", "unknown")
                print(f"\n[WARNING] Using cached strikes from {week_label}\n")
                _print_table(strikes, week_label)
            except Exception as load_exc:
                logger.error(f"Could not load cached strikes: {load_exc}")
        else:
            logger.error("No cached strikes available.")

    return strikes


if __name__ == "__main__":
    run_live_pipeline()
