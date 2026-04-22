"""
Module 8: Live Sunday-Night Orchestration
Run before each weekly expiry to fetch latest data, rebuild features/GARCH,
and generate credit spread recommendations (bull put / bear call).

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

SPREADS_JSON = OUTPUTS_DIR / "spreads_live.json"
FEATURE_PARQUET = BASE_DIR / "data" / "feature_matrix_with_garch.parquet"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _print_spreads(spreads_result: dict, week_date: str) -> None:
    """Print credit spread recommendations to stdout."""
    spreads = spreads_result.get("spreads", [])
    spot    = spreads_result.get("spot", 0)
    vix     = spreads_result.get("vix_level", 0)
    atm_iv  = spreads_result.get("atm_iv_pct")
    pcr     = spreads_result.get("pcr")
    dir_sig = spreads_result.get("direction_signal", {})

    sep  = "=" * 60
    thin = "-" * 60

    print(sep)
    print(f"  NIFTY CREDIT SPREADS — {week_date}")
    print(sep)
    print(f"  Spot: {spot:,.0f}  |  VIX: {vix:.1f}  |  ATM IV: {f'{atm_iv:.2f}%' if atm_iv else 'N/A'}  |  PCR: {f'{pcr:.2f}' if pcr else 'N/A'}")
    print(f"  Direction: {dir_sig.get('direction','?').upper()}  (confidence: {dir_sig.get('confidence',0):.2f})")
    print(thin)

    if not spreads:
        print("  No feasible spreads found (all below min R:R or OI threshold).")
        print(sep)
        return

    for s in spreads:
        stype = "Bull Put " if s["spread_type"] == "bull_put" else "Bear Call"
        pop   = s.get("pop_pct") or 0
        print(f"  {stype}  {int(s['short_strike'])}/{int(s['long_strike'])}"
              f"  |  Expiry: {s['expiry_date']} ({s['expiry_type']})  DTE={s['dte_days']}d")
        print(f"    Max Profit: {s['premium_pts']:.1f} pts  (₹{s['max_profit_inr']:,.0f})"
              f"  |  Max Loss: {s['max_loss_pts']:.1f} pts  (₹{s['max_loss_inr']:,.0f})")
        print(f"    R:R: {s['rr_ratio']:.2%}  |  POP: {pop*100:.1f}%"
              f"  |  Breakeven: {s['breakeven']:.0f}  |  EV: {s.get('ev_proxy',0):.4f}")
        print()
    print(sep)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_live_pipeline() -> dict:
    """
    Run the full Sunday-night live pipeline.

    Returns
    -------
    dict — spread recommendations (also written to outputs/spreads_live.json)
    """
    import pandas as pd

    spreads_result: dict = {}
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
        # Step 5.5 — NSE Option Chain OI (non-blocking)
        # ------------------------------------------------------------------
        step = "option chain fetch (module11)"
        logger.info(f"[Step 5.5] {step}")
        oi_data = None
        try:
            from module11_option_chain import fetch_option_chain
            oi_data = fetch_option_chain()
            if oi_data:
                if oi_data.get("atm_iv"):
                    logger.info(
                        f"OI loaded: ATM IV={oi_data['atm_iv']:.1%}  "
                        f"PCR={oi_data['pcr']:.2f}  MaxPain={oi_data['max_pain']}"
                    )
                else:
                    logger.info(f"OI loaded: PCR={oi_data['pcr']:.2f}  MaxPain={oi_data['max_pain']}")
        except Exception as e11:
            logger.warning(f"Option chain fetch failed (non-fatal): {e11} — GARCH-only mode")

        # ------------------------------------------------------------------
        # Step 6 — Credit Spreads (module9)
        # ------------------------------------------------------------------
        step = "credit spread generation (module9)"
        logger.info(f"[Step 6] {step}")
        from module9_spreads import generate_all_spreads

        week_label = date.today().isoformat()

        spreads_result = generate_all_spreads(
            feature_row=feature_row,
            spot=spot,
            vix_level=vix_level if vix_level else 16.0,
            garch_vol=garch_vol,
            oi_data=oi_data,
        )

        # Console table
        _print_spreads(spreads_result, week_label)

        # JSON file
        with open(SPREADS_JSON, "w") as fh:
            json.dump(spreads_result, fh, indent=2, default=str)
        logger.success(f"Spreads saved to {SPREADS_JSON}")

    except SystemExit:
        # Propagate intentional exits (e.g. missing .env)
        raise

    except Exception as exc:  # noqa: BLE001
        logger.error(f"Pipeline failed at step '{step}': {exc}")

        # Attempt to show last known spreads
        if SPREADS_JSON.exists():
            logger.warning("Showing last known spreads from cache.")
            try:
                with open(SPREADS_JSON) as fh:
                    spreads_result = json.load(fh)
                print(f"\n[WARNING] Using cached spreads from {spreads_result.get('generated_at', 'unknown')}\n")
                _print_spreads(spreads_result, spreads_result.get("generated_at", "unknown"))
            except Exception as load_exc:
                logger.error(f"Could not load cached spreads: {load_exc}")
        else:
            logger.error("No cached spreads available.")

    return spreads_result


if __name__ == "__main__":
    run_live_pipeline()
