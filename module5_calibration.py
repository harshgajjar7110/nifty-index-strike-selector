"""
Module 5: Conformal Calibration
Wraps per-regime LightGBM models with MAPIE conformal prediction
to provide guaranteed coverage bounds on predicted weekly range.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, RegressorMixin

from utils_constants import REGIMES

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"


def _load_regime_thresholds() -> tuple:
    """Load optimized VIX regime thresholds from models/; fall back to defaults (15, 20)."""
    thresh_path = Path(__file__).parent / "models" / "regime_thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            t = json.load(f)
        return t["low_thresh"], t["high_thresh"]
    return 15.0, 20.0


class RegimeLGBQuantileWrapper(BaseEstimator, RegressorMixin):
    """Route predictions by VIX regime to per-regime LightGBM quantile models."""

    def __init__(self, lgb_models: dict, feature_columns: list, low_thresh: float = 15.0, high_thresh: float = 20.0):
        self.lgb_models = lgb_models
        self.feature_columns = feature_columns
        self.vix_col_idx = feature_columns.index("vix_level")
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.fitted_ = True

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        preds = np.zeros(len(X))
        for i, row in enumerate(X):
            vix = row[self.vix_col_idx]
            regime = (
                "low" if vix < self.low_thresh
                else "mid" if vix < self.high_thresh
                else "high"
            )
            if regime not in self.lgb_models:
                logger.warning(f"No model for regime '{regime}' — using training median")
                # Fallback to training median (roughly (P10+P90)/2)
                preds[i] = np.median([
                    (float(m["p10"].predict(row.reshape(1, -1))[0]) + 
                     float(m["p90"].predict(row.reshape(1, -1))[0])) / 2.0
                    for m in self.lgb_models.values()
                ])
            else:
                model = self.lgb_models[regime]
                # Use P50 (median) as point prediction
                p10 = float(model["p10"].predict(row.reshape(1, -1))[0])
                p90 = float(model["p90"].predict(row.reshape(1, -1))[0])
                preds[i] = (p10 + p90) / 2.0

        return preds


def _coverage_at_target(mid: np.ndarray, half_width: np.ndarray, y_test: np.ndarray, target: float) -> float:
    """Binary search for interval scale that achieves target empirical coverage."""
    lo, hi = 0.5, 5.0
    for _ in range(50):
        scale = (lo + hi) / 2
        lower = mid - scale * half_width
        upper = mid + scale * half_width
        cov = float(np.mean((lower <= y_test) & (y_test <= upper)))
        if cov < target:
            lo = scale
        else:
            hi = scale
    return round(cov, 4)


def run_calibration() -> dict:
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    target_coverage = float(os.getenv("TARGET_COVERAGE", "0.85"))
    logger.info(f"Loaded TARGET_COVERAGE={target_coverage} from .env")

    # Load regime models
    meta_path = MODELS_DIR / "regime_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run module4_model.py first")

    with open(meta_path) as f:
        regime_meta = json.load(f)

    lgb_models = {}
    for regime in REGIMES:
        if regime in regime_meta:
            model_file = MODELS_DIR / regime_meta[regime]["model_file"]
            if model_file.exists():
                lgb_models[regime] = joblib.load(model_file)

    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    logger.info("Loaded per-regime LightGBM models and feature columns")

    data_path = DATA_DIR / "feature_matrix_with_garch.parquet"
    if not data_path.exists():
        raise FileNotFoundError("Run module3_garch.py first")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded feature matrix: {df.shape}")

    # Prepare data — 80/20 split, then conf/eval
    df = df.sort_index()
    target_col = "log_range"
    available_features = [c for c in feature_columns if c in df.columns]
    if len(available_features) < len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        logger.warning(f"Missing feature columns in data: {missing}")

    df_clean = df[available_features + [target_col]].dropna()
    X = df_clean[available_features].values
    y = df_clean[target_col].values

    split_idx = int(len(X) * 0.80)
    X_calib_all = df_clean[available_features].iloc[split_idx:]
    y_calib_all = df_clean[target_col].iloc[split_idx:]
    n = len(X_calib_all)
    mid = int(n * 0.80)  # Use 80% for conformalization, 20% for evaluation
    X_conf, y_conf = X_calib_all.iloc[:mid], y_calib_all.iloc[:mid]
    X_eval, y_eval = X_calib_all.iloc[mid:], y_calib_all.iloc[mid:]

    X_conf_arr = X_conf.values
    y_conf_arr = y_conf.values
    X_eval_arr = X_eval.values
    y_eval_arr = y_eval.values
    logger.info(f"Calibration set size: {n} samples (conf={mid}, eval={n - mid})")

    low_thresh, high_thresh = _load_regime_thresholds()
    logger.info(f"Loaded regime thresholds: low={low_thresh}, high={high_thresh}")

    # Build per-regime MAPIE models
    from mapie.regression import SplitConformalRegressor

    regime_names = REGIMES
    thresholds = {"low": low_thresh, "high": high_thresh}
    mapie_per_regime = {}
    coverage_per_regime = {}

    # Extract VIX column for masking
    vix_col_idx = feature_columns.index("vix_level")

    for regime in regime_names:
        # Subset conformal and eval sets to this regime
        if regime == "low":
            mask_conf = X_conf_arr[:, vix_col_idx] < thresholds["low"]
            mask_eval = X_eval_arr[:, vix_col_idx] < thresholds["low"]
        elif regime == "mid":
            mask_conf = (X_conf_arr[:, vix_col_idx] >= thresholds["low"]) & \
                        (X_conf_arr[:, vix_col_idx] < thresholds["high"])
            mask_eval = (X_eval_arr[:, vix_col_idx] >= thresholds["low"]) & \
                        (X_eval_arr[:, vix_col_idx] < thresholds["high"])
        else:  # high
            mask_conf = X_conf_arr[:, vix_col_idx] >= thresholds["high"]
            mask_eval = X_eval_arr[:, vix_col_idx] >= thresholds["high"]

        X_r_conf, y_r_conf = X_conf_arr[mask_conf], y_conf_arr[mask_conf]
        X_r_eval, y_r_eval = X_eval_arr[mask_eval], y_eval_arr[mask_eval]
        logger.info(f"Regime {regime}: conf={len(X_r_conf)}, eval={len(X_r_eval)}")

        if len(X_r_conf) < 7:
            logger.warning(f"Regime {regime}: only {len(X_r_conf)} conf samples — skipping, using global fallback")
            mapie_per_regime[regime] = None
            coverage_per_regime[regime] = None
            continue

        wrapper_r = RegimeLGBQuantileWrapper(
            lgb_models=lgb_models,
            feature_columns=feature_columns,
            low_thresh=low_thresh,
            high_thresh=high_thresh,
        )
        mapie_r = SplitConformalRegressor(
            estimator=wrapper_r,
            confidence_level=target_coverage,
            prefit=True,
        )
        mapie_r.conformalize(X_r_conf, y_r_conf)

        if len(X_r_eval) >= 3:
            y_pred_r, y_pis_r = mapie_r.predict_interval(X_r_eval)
            # Handle different MAPIE versions
            if len(y_pis_r.shape) == 3:
                y_low_r = y_pis_r[:, 0, 0]
                y_high_r = y_pis_r[:, 1, 0]
            else:
                y_low_r = y_pis_r[:, 0]
                y_high_r = y_pis_r[:, 1]
            covered_r = (y_low_r <= y_r_eval) & (y_r_eval <= y_high_r)
            coverage_per_regime[regime] = float(np.mean(covered_r))
            logger.info(f"Regime {regime} OOS coverage: {coverage_per_regime[regime]:.4f}")
        else:
            coverage_per_regime[regime] = None

        mapie_per_regime[regime] = mapie_r

    # Clear stale per-regime MAPIE files before saving new ones
    for regime in REGIMES:
        stale_path = MODELS_DIR / f"mapie_{regime}.pkl"
        if stale_path.exists():
            stale_path.unlink()
            logger.info(f"Removed stale {stale_path}")

    # Save per-regime MAPIE models
    for regime, mapie_r in mapie_per_regime.items():
        if mapie_r is not None:
            out_path = MODELS_DIR / f"mapie_{regime}.pkl"
            joblib.dump(mapie_r, out_path)
            logger.info(f"Saved {regime} MAPIE to {out_path}")

    # Save global fallback (original single MAPIE)
    wrapper_global = RegimeLGBQuantileWrapper(
        lgb_models=lgb_models,
        feature_columns=feature_columns,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )
    mapie_global = SplitConformalRegressor(
        estimator=wrapper_global,
        confidence_level=target_coverage,
        prefit=True,
    )
    mapie_global.conformalize(X_conf_arr, y_conf_arr)
    joblib.dump(mapie_global, MODELS_DIR / "mapie_calibrated.pkl")
    logger.info("Saved global fallback MAPIE to mapie_calibrated.pkl")

    # Compute coverage: Use the evaluation set (X_eval_arr) which was NOT used for conformalization
    def compute_coverage_mapie() -> dict:
        """Compute coverage using fitted global MAPIE model on out-of-sample eval set."""
        # Get intervals from MAPIE
        y_pred, y_pis = mapie_global.predict_interval(X_eval_arr)
        
        # y_pis shape is (n_samples, 2, 1) in some MAPIE versions
        if len(y_pis.shape) == 3:
            y_low = y_pis[:, 0, 0]
            y_high = y_pis[:, 1, 0]
        else:
            y_low = y_pis[:, 0]
            y_high = y_pis[:, 1]
        
        covered = (y_low <= y_eval_arr) & (y_eval_arr <= y_high)
        actual_coverage = np.mean(covered)
        
        half_width = (y_high - y_low) / 2
        mid = (y_low + y_high) / 2

        return {
            "actual_coverage": float(actual_coverage),
            "_mid": mid,
            "_half_width": half_width,
            "_y_test": y_eval_arr,
        }

    coverage_results = compute_coverage_mapie()
    actual_coverage = coverage_results["actual_coverage"]
    _mid = coverage_results.pop("_mid")
    _half_width = coverage_results.pop("_half_width")
    _y_test = coverage_results.pop("_y_test")

    logger.info(f"Empirical OOS coverage @ {target_coverage:.0%}: {actual_coverage:.4f}")

    if actual_coverage < target_coverage:
        logger.warning(f"Coverage {actual_coverage:.4f} is below target {target_coverage:.2f}")

    # Calibration curve: compute real empirical coverage at each nominal level
    nominal_levels = np.arange(0.70, 0.96, 0.05)
    empirical_levels = []
    for lvl in nominal_levels:
        m = SplitConformalRegressor(estimator=wrapper_global, confidence_level=lvl, prefit=True)
        m.conformalize(X_conf_arr, y_conf_arr)
        _, pis = m.predict_interval(X_eval_arr)
        if len(pis.shape) == 3:
            low, high = pis[:, 0, 0], pis[:, 1, 0]
        else:
            low, high = pis[:, 0], pis[:, 1]
        empirical_levels.append(np.mean((low <= y_eval_arr) & (y_eval_arr <= high)))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(nominal_levels, empirical_levels, "o-", color="steelblue",
            label="Empirical coverage", linewidth=2, markersize=6)
    ax.plot([0.70, 0.95], [0.70, 0.95], "--", color="gray",
            label="Perfect calibration", linewidth=1.5)
    ax.set_xlabel("Nominal Coverage", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_title("MAPIE Conformal Calibration Curve\n(Nifty 50 Weekly Log-Range)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0.68, 0.97)
    ax.set_ylim(0.68, 0.97)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = OUTPUTS_DIR / "calibration_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved calibration curve to {plot_path}")

    # Save wrapper
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    wrapper_path = MODELS_DIR / "regime_lgb_wrapper.pkl"
    joblib.dump(wrapper_global, wrapper_path)
    logger.info(f"Saved regime wrapper to {wrapper_path}")

    report = {
        "target_coverage": target_coverage,
        "actual_oos_coverage": round(actual_coverage, 4),
        "per_regime_coverage": {k: round(v, 4) if v is not None else None for k, v in coverage_per_regime.items()},
        "calibration_data": {
            "nominal": [round(float(l), 2) for l in nominal_levels],
            "empirical": [round(float(l), 4) for l in empirical_levels]
        }
    }

    report_path = OUTPUTS_DIR / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved calibration report to {report_path}")

    return report


if __name__ == "__main__":
    report = run_calibration()
    print(report)
