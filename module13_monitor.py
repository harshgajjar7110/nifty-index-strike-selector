"""
Module 13: Model Drift Monitoring Pipeline
Runs after each weekly expiry to detect:
  1. Coverage decay (actual vs target coverage)
  2. Feature distribution drift (recent vs training)
  3. Strike quality degradation (breach rates, POP accuracy)

Outputs: outputs/monitor_report_YYYY-MM-DD.json
"""

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from scipy import stats

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = DATA_DIR / "feature_matrix_with_garch.parquet"
BACKTEST_PATH = OUTPUTS_DIR / "walkforward_results.csv"
CALIBRATION_PATH = OUTPUTS_DIR / "calibration_report.json"

load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_COVERAGE = float(os.getenv("TARGET_COVERAGE", "0.85"))
COVERAGE_DECAY_THRESHOLD = float(os.getenv("MONITOR_COVERAGE_DECAY", "0.05"))  # 5pp drop
DRIFT_WINDOW_WEEKS = int(os.getenv("MONITOR_DRIFT_WINDOW_WEEKS", "12"))
FEATURE_DRIFT_PVAL = float(os.getenv("MONITOR_FEATURE_DRIFT_PVAL", "0.01"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_backtest_or_results() -> pd.DataFrame | None:
    """Load most recent P&L results (walk-forward preferred, fallback to static backtest)."""
    for path in [BACKTEST_PATH, OUTPUTS_DIR / "backtest_results.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            if "week_end" in df.columns:
                df["week_end"] = pd.to_datetime(df["week_end"])
                df = df.set_index("week_end").sort_index()
            return df
    return None


def _load_feature_matrix() -> pd.DataFrame | None:
    if not FEATURE_PATH.exists():
        return None
    df = pd.read_parquet(FEATURE_PATH).sort_index()
    if hasattr(df.index, "normalize"):
        df.index = df.index.normalize()
    return df


def _ks_drift(ref: pd.Series, recent: pd.Series) -> dict:
    """Two-sample KS test for feature drift."""
    ref = ref.dropna()
    recent = recent.dropna()
    if len(ref) < 20 or len(recent) < 10:
        return {"drift_detected": False, "pvalue": None, "statistic": None}
    ks_stat, pvalue = stats.ks_2samp(ref, recent)
    return {
        "drift_detected": bool(pvalue < FEATURE_DRIFT_PVAL),
        "pvalue": round(float(pvalue), 6),
        "statistic": round(float(ks_stat), 6),
    }


def _compute_rolling_coverage(results_df: pd.DataFrame, window: int = 8) -> pd.Series:
    """Rolling coverage: did actual log_range fall inside [p10, p90]?"""
    df = results_df.copy()
    covered = (df["log_range_p10"] <= df["actual_log_range"]) & (df["actual_log_range"] <= df["log_range_p90"])
    return covered.rolling(window, min_periods=3).mean()


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------

def run_monitor() -> dict:
    logger.info("=" * 60)
    logger.info("  MODEL DRIFT MONITOR")
    logger.info("=" * 60)

    report = {
        "run_date": date.today().isoformat(),
        "target_coverage": TARGET_COVERAGE,
        "alerts": [],
        "coverage_analysis": {},
        "feature_drift": {},
        "strike_quality": {},
        "recommendation": "none",
    }

    # ------------------------------------------------------------------
    # 1. Coverage decay analysis
    # ------------------------------------------------------------------
    results = _load_backtest_or_results()
    if results is not None and len(results) > 0:
        # Overall coverage
        if "actual_log_range" in results.columns and "log_range_p10" in results.columns:
            covered = (results["log_range_p10"] <= results["actual_log_range"]) & (results["actual_log_range"] <= results["log_range_p90"])
            overall_cov = float(covered.mean())
            report["coverage_analysis"]["overall_coverage"] = round(overall_cov, 4)
            report["coverage_analysis"]["target_coverage"] = TARGET_COVERAGE

            if overall_cov < TARGET_COVERAGE - COVERAGE_DECAY_THRESHOLD:
                report["alerts"].append(
                    f"COVERAGE_DECAY: overall={overall_cov:.2%} vs target={TARGET_COVERAGE:.0%}"
                )

            # Rolling coverage
            rolling_cov = _compute_rolling_coverage(results, window=8)
            if len(rolling_cov) > 0 and not rolling_cov.iloc[-1] is np.nan:
                latest_rolling = float(rolling_cov.iloc[-1])
                report["coverage_analysis"]["rolling_8w_coverage"] = round(latest_rolling, 4)
                if latest_rolling < TARGET_COVERAGE - COVERAGE_DECAY_THRESHOLD:
                    report["alerts"].append(
                        f"ROLLING_COVERAGE_LOW: last 8w={latest_rolling:.2%}"
                    )

        # Win rate trend
        if "won" in results.columns:
            won = results["won"].dropna().astype(bool)
            if len(won) >= 8:
                recent_wr = float(won.tail(8).mean())
                report["coverage_analysis"]["recent_8w_win_rate"] = round(recent_wr, 4)
                if recent_wr < 0.50:
                    report["alerts"].append(f"WIN_RATE_LOW: last 8w={recent_wr:.1%}")

        # P&L trajectory
        if "pnl_points" in results.columns:
            pnl = results["pnl_points"].dropna()
            if len(pnl) >= 4:
                recent_pnl = float(pnl.tail(4).sum())
                report["strike_quality"]["last_4w_pnl_points"] = round(recent_pnl, 2)
                if recent_pnl < -2 * pnl.std():
                    report["alerts"].append(f"P&L_ANOMALY: last 4w={recent_pnl:.1f} pts")
    else:
        report["alerts"].append("NO_RESULTS: backtest/walkforward results not found")

    # ------------------------------------------------------------------
    # 2. Feature drift
    # ------------------------------------------------------------------
    feat_df = _load_feature_matrix()
    if feat_df is not None and len(feat_df) > DRIFT_WINDOW_WEEKS * 2:
        # Training reference = everything before last DRIFT_WINDOW_WEEKS
        ref_df = feat_df.iloc[:-DRIFT_WINDOW_WEEKS]
        recent_df = feat_df.iloc[-DRIFT_WINDOW_WEEKS:]

        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target and known-noisy columns from drift checks
        exclude = {"log_range", "is_event_week"}
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        drift_detected_any = False
        for col in numeric_cols:
            drift_info = _ks_drift(ref_df[col], recent_df[col])
            if drift_info["drift_detected"]:
                drift_detected_any = True
                report["feature_drift"][col] = drift_info
                report["alerts"].append(
                    f"FEATURE_DRIFT: {col} p={drift_info['pvalue']:.4f}"
                )

        report["feature_drift"]["features_checked"] = len(numeric_cols)
        report["feature_drift"]["drifted_features_count"] = sum(
            1 for v in report["feature_drift"].values() if isinstance(v, dict) and v.get("drift_detected")
        )
        if not drift_detected_any:
            report["feature_drift"]["status"] = "no_significant_drift"
    else:
        report["alerts"].append("NO_FEATURES: feature matrix not found or too short")

    # ------------------------------------------------------------------
    # 3. Strike quality (POP accuracy if available)
    # ------------------------------------------------------------------
    if results is not None and "prob_of_profit" in results.columns and "won" in results.columns:
        pop = results["prob_of_profit"].dropna()
        won = results["won"].dropna()
        if len(pop) == len(won) and len(pop) > 10:
            pop_error = float(pop.mean() - won.mean())
            report["strike_quality"]["pop_vs_actual_error"] = round(pop_error, 4)
            if abs(pop_error) > 0.10:
                report["alerts"].append(f"POP_DRIFT: predicted-actual={pop_error:+.1%}")

    # ------------------------------------------------------------------
    # 4. Recommendation
    # ------------------------------------------------------------------
    if any("COVERAGE_DECAY" in a or "WIN_RATE_LOW" in a for a in report["alerts"]):
        report["recommendation"] = "retrain"
    elif any("FEATURE_DRIFT" in a for a in report["alerts"]):
        report["recommendation"] = "review_features"
    elif any("POP_DRIFT" in a for a in report["alerts"]):
        report["recommendation"] = "recalibrate"
    else:
        report["recommendation"] = "ok"

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out_path = OUTPUTS_DIR / f"monitor_report_{date.today().isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Monitor report saved to {out_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("  MONITOR SUMMARY")
    print("=" * 60)
    print(f"  Recommendation : {report['recommendation'].upper()}")
    print(f"  Alerts ({len(report['alerts'])}):")
    for a in report["alerts"]:
        print(f"    • {a}")
    cov = report.get("coverage_analysis", {})
    print(f"  Overall coverage: {cov.get('overall_coverage', 'N/A')}")
    print(f"  Recent 8w WR    : {cov.get('recent_8w_win_rate', 'N/A')}")
    print("=" * 60 + "\n")

    return report


if __name__ == "__main__":
    run_monitor()
